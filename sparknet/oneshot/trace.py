"""Per-op timing trace of the one-shot all-reduce: file layout, reader and analysis.

Set ``SPARKNET_ROCE_TRACE=1`` before a runtime is created to trace it (CuTe kernel family
only). The runtime then writes ``/dev/shm/sparknet-trace-r<rank>-p<pid>-<id>.bin``: a
4096-byte header, then ``RECORDS`` records of ``RECORD_WORDS`` little-endian u64 words,
each op's record at its sequence number modulo ``RECORDS``. Tracing selects a separately
compiled kernel; untraced runtimes run the same code as before and the proxy only tests a
null pointer per op.

The kernel fills its words with the GPU's ``%globaltimer``: block 0's start, the doorbell
(written by the last block to finish staging), the arrival of every peer lane's flag as
block 0 saw it, the end of the wait, and the end of the op (the last block to finish
reducing). The proxy thread fills its words with ``CLOCK_MONOTONIC_RAW``: doorbell seen,
direct posts issued, ring relay done, completions drained. The header carries the
GPU-minus-CPU clock offset measured on the node when the trace opened, so the proxy's
times land on the GPU clock.

Clocks differ between nodes. Every all-reduce is a two-way exchange between directly
linked ranks, so :func:`summarize` estimates each pair's clock offset the NTP way, from
the sends and arrivals in both directions, and reports one-way delays on a common clock.

usage: python -m sparknet.oneshot.trace [--min-seq N] [--max-seq N] [--nbytes N] FILE [FILE ...]
       (one file per rank; prints JSON; --last-seq prints each file's newest sequence)
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
from array import array
from dataclasses import dataclass

ENV_TRACE = "SPARKNET_ROCE_TRACE"
MAGIC = b"SPKTRACE"
VERSION = 1
HEADER_BYTES = 4096
RECORDS = 65536
RECORD_WORDS = 32
RECORD_BYTES = RECORD_WORDS * 8

# Kernel words (GPU %globaltimer, ns).
W_SEQ = 0
W_START = 1
W_DOORBELL = 2
W_WAIT_DONE = 3
W_END = 4
W_META = 5  # nbytes | grid blocks << 32
W_FLAGS = 6  # MAX_FLAG_WORDS words: peer * lanes + lane
MAX_FLAG_WORDS = 16
# Proxy words (CLOCK_MONOTONIC_RAW, ns), in the C proxy's TRACE_* order.
W_PROXY = W_FLAGS + MAX_FLAG_WORDS
PROXY_WORDS = ("seen", "posted", "relayed", "drained")
# The sequence the proxy marked. All-gathers share the doorbell but run an untraced kernel,
# so proxy words are used only when this matches the kernel's sequence (0: older traces).
W_PROXY_SEQ = W_PROXY + len(PROXY_WORDS)

TOPOLOGY_CODES = {"direct": 0, "ring4": 1, "mesh4": 2}
_HEADER = struct.Struct("<8sIIIIIIIIqqq64s")


def enabled() -> bool:
    """True when ``SPARKNET_ROCE_TRACE`` asks for tracing."""
    value = os.environ.get(ENV_TRACE, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def file_bytes() -> int:
    """Size of a trace file."""
    return HEADER_BYTES + RECORDS * RECORD_BYTES


def pack_header(*, world: int, rank: int, lanes: int, topology: str, clock_offset_ns: int,
                clock_halfwidth_ns: int, created_ns: int) -> bytes:
    """The header bytes (``HEADER_BYTES`` long)."""
    raw = _HEADER.pack(MAGIC, VERSION, world, rank, lanes, TOPOLOGY_CODES[topology], RECORDS, RECORD_WORDS,
                       os.getpid(), clock_offset_ns, clock_halfwidth_ns, created_ns,
                       socket.gethostname().encode()[:64])
    return raw + bytes(HEADER_BYTES - len(raw))


@dataclass
class Trace:
    """One rank's trace: header fields and the valid records (rows of u64 words)."""

    world: int
    rank: int
    lanes: int
    topology: str
    clock_offset_ns: int
    clock_halfwidth_ns: int
    host: str
    records: list  # rows of RECORD_WORDS ints

    def by_seq(self) -> dict[int, list]:
        return {int(row[W_SEQ]): row for row in self.records}


def read(path: str) -> Trace:
    """Read a trace file; keep records whose kernel finished (start, doorbell and end set)."""
    with open(path, "rb") as handle:
        header = handle.read(HEADER_BYTES)
        body = handle.read()
    magic, version, world, rank, lanes, topology, records, words, _pid, offset, halfwidth, _created, host = \
        _HEADER.unpack(header[: _HEADER.size])
    if magic != MAGIC or version != VERSION or words != RECORD_WORDS:
        raise ValueError(f"{path}: not a version {VERSION} sparknet trace")
    data = array("Q")
    data.frombytes(body[: records * words * 8])
    if sys.byteorder != "little":
        data.byteswap()
    valid = []
    for index in range(records):
        row = data[index * words:(index + 1) * words]
        if row[W_SEQ] and row[W_START] and row[W_DOORBELL] and row[W_END]:
            valid.append(list(row))
    names = {code: name for name, code in TOPOLOGY_CODES.items()}
    return Trace(world, rank, lanes, names.get(topology, str(topology)), offset, halfwidth,
                 host.rstrip(b"\0").decode(errors="replace"), valid)


def _percentile(ordered: list[float], q: float) -> float:
    """Linear-interpolated percentile of sorted values (numpy's default method)."""
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _median(values: list[float]) -> float:
    return _percentile(sorted(values), 50)


def _stats(values) -> dict[str, float] | None:
    ordered = sorted(float(v) for v in values if v is not None)
    if not ordered:
        return None
    p5, p50, p95 = (_percentile(ordered, q) / 1000.0 for q in (5, 50, 95))
    return {"p5_us": round(p5, 2), "p50_us": round(p50, 2), "p95_us": round(p95, 2), "n": len(ordered)}


def _delta(later, earlier):
    """Signed nanoseconds, or None when either time is missing."""
    if later == 0 or earlier == 0:
        return None
    return int(later) - int(earlier)


def _proxy(row, trace: Trace, which: str):
    """A proxy time of this record's op on the GPU clock, or 0 (unset, or another op's)."""
    marked = int(row[W_PROXY_SEQ])
    if marked and marked != int(row[W_SEQ]):
        return 0
    value = int(row[W_PROXY + PROXY_WORDS.index(which)])
    return value + trace.clock_offset_ns if value else 0


def peer_arrival(row, trace: Trace, peer: int, last: bool = True) -> int:
    """GPU time block 0 saw ``peer``'s first or last lane flag (0 when untraced)."""
    times = [int(row[W_FLAGS + peer * trace.lanes + lane]) for lane in range(trace.lanes)]
    times = [t for t in times if t]
    if not times:
        return 0
    return max(times) if last else min(times)


def adjacent(world: int, rank: int, peer: int, topology: str) -> bool:
    """Peers the proxy posts to directly (the ring relays the opposite rank)."""
    if peer == rank:
        return False
    if topology != "ring4":
        return True
    return peer in ((rank + 1) % world, (rank - 1) % world)


def phases(trace: Trace) -> dict[str, dict | None]:
    """Distributions of each phase of this rank's ops, on its GPU clock."""
    out: dict[str, list] = {name: [] for name in ("total", "staging", "wait", "reduce", "proxy_wake",
                                                  "proxy_post", "proxy_relay")}
    arrivals: dict[str, list] = {}
    for row in trace.records:
        start, doorbell, wait_done, end = (int(row[w]) for w in (W_START, W_DOORBELL, W_WAIT_DONE, W_END))
        seen, posted, relayed = (_proxy(row, trace, w) for w in ("seen", "posted", "relayed"))
        out["total"].append(_delta(end, start))
        out["staging"].append(_delta(doorbell, start))
        out["wait"].append(_delta(wait_done, doorbell))
        out["reduce"].append(_delta(end, wait_done))
        out["proxy_wake"].append(_delta(seen, doorbell))
        out["proxy_post"].append(_delta(posted, seen))
        out["proxy_relay"].append(_delta(relayed, posted))
        for peer in range(trace.world):
            if peer == trace.rank:
                continue
            route = "direct" if adjacent(trace.world, trace.rank, peer, trace.topology) else "relayed"
            arrivals.setdefault(f"peer{peer}_{route}", []).append(_delta(peer_arrival(row, trace, peer), doorbell))
    result = {name: _stats(values) for name, values in out.items()}
    result.update({f"arrival_{name}": _stats(values) for name, values in sorted(arrivals.items())})
    return result


def pair_clock(a: Trace, b: Trace) -> dict | None:
    """Offset of b's GPU clock from a's and the one-way delays, for directly linked ranks.

    For every op both traced: d_ab = (b saw a's flag) - (a posted), d_ba likewise. With
    symmetric paths the offset is (d_ab - d_ba) / 2 and the delay (d_ab + d_ba) / 2; the
    offset is taken as the median over the tenth of ops with the smallest round trips,
    where queueing noise is least.
    """
    rows_b = b.by_seq()
    samples = []
    for row_a in a.records:
        row_b = rows_b.get(int(row_a[W_SEQ]))
        if row_b is None:
            continue
        post_a, post_b = _proxy(row_a, a, "posted"), _proxy(row_b, b, "posted")
        seen_a = peer_arrival(row_a, a, b.rank, last=False)
        seen_b = peer_arrival(row_b, b, a.rank, last=False)
        if not (post_a and post_b and seen_a and seen_b):
            continue
        samples.append((seen_b - post_a, seen_a - post_b, int(row_a[W_DOORBELL]), int(row_b[W_DOORBELL])))
    # Records written before the proxy stamped its sequence can pair one op's kernel times
    # with another op's proxy times; those lie far from the bulk, so drop anything more than
    # a millisecond from the median in either direction.
    mid_ab = _median([s[0] for s in samples]) if samples else 0
    mid_ba = _median([s[1] for s in samples]) if samples else 0
    samples = [s for s in samples if abs(s[0] - mid_ab) < 1_000_000 and abs(s[1] - mid_ba) < 1_000_000]
    if len(samples) < 10:
        return None
    cutoff = _percentile(sorted(d_ab + d_ba for d_ab, d_ba, _, _ in samples), 10)
    offset = _median([(d_ab - d_ba) / 2 for d_ab, d_ba, _, _ in samples if d_ab + d_ba <= cutoff])  # b - a
    return {
        "ranks": [a.rank, b.rank],
        "offset_us": round(offset / 1000, 2),
        "one_way_a_to_b": _stats(d_ab - offset for d_ab, _, _, _ in samples),
        "one_way_b_to_a": _stats(d_ba + offset for _, d_ba, _, _ in samples),
        "doorbell_b_minus_a": _stats(door_b - door_a - offset for _, _, door_a, door_b in samples),
        "samples": len(samples),
    }


def select(t: Trace, *, min_seq: int = 0, max_seq: int | None = None, nbytes: int | None = None) -> Trace:
    """The records of a sequence range and, optionally, one payload size."""
    rows = [r for r in t.records if r[W_SEQ] >= min_seq and (max_seq is None or r[W_SEQ] <= max_seq)
            and (nbytes is None or (r[W_META] & 0xFFFFFFFF) == nbytes)]
    return Trace(t.world, t.rank, t.lanes, t.topology, t.clock_offset_ns, t.clock_halfwidth_ns, t.host, rows)


def sizes(t: Trace, top: int = 8) -> dict[int, int]:
    """Op counts of the most frequent payload sizes (bytes)."""
    counts: dict[int, int] = {}
    for r in t.records:
        size = r[W_META] & 0xFFFFFFFF
        counts[size] = counts.get(size, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1])[:top])


def summarize(traces: list[Trace]) -> dict:
    """Per-rank phases and, for every directly linked pair, clock offset and one-way delays."""
    traces = sorted(traces, key=lambda t: t.rank)
    out = {"ranks": {t.rank: {"host": t.host, "ops": len(t.records), "sizes": sizes(t),
                              "clock_halfwidth_us": round(t.clock_halfwidth_ns / 1000, 2),
                              "phases": phases(t)} for t in traces}, "pairs": []}
    for i, a in enumerate(traces):
        for b in traces[i + 1:]:
            if adjacent(a.world, a.rank, b.rank, a.topology):
                pair = pair_clock(a, b)
                if pair is not None:
                    out["pairs"].append(pair)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m sparknet.oneshot.trace", description="one-shot trace summary")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--min-seq", type=int, default=0)
    parser.add_argument("--max-seq", type=int)
    parser.add_argument("--nbytes", type=int)
    parser.add_argument("--last-seq", action="store_true", help="print each file's newest sequence and exit")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    traces = [read(path) for path in args.files]
    if args.last_seq:
        print(json.dumps({path: max((r[W_SEQ] for r in t.records), default=0)
                          for path, t in zip(args.files, traces, strict=True)}))
        return 0
    traces = [select(t, min_seq=args.min_seq, max_seq=args.max_seq, nbytes=args.nbytes) for t in traces]
    print(json.dumps(summarize(traces), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
