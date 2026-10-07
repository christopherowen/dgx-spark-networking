"""Host-independent tests of the one-shot trace format and analysis (sparknet.oneshot.trace)."""

from __future__ import annotations

import importlib.util
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "sparknet" / "oneshot"


def _load_trace():
    """The trace module alone: the package imports torch, which these host tests do without."""
    spec = importlib.util.spec_from_file_location("sparknet_oneshot_trace", ROOT / "trace.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


trace = _load_trace()


def write_trace(path: str, *, world: int, rank: int, lanes: int, topology: str, offset_ns: int, rows) -> None:
    records = bytearray(trace.RECORDS * trace.RECORD_BYTES)
    for row in rows:
        at = (int(row[trace.W_SEQ]) % trace.RECORDS) * trace.RECORD_BYTES
        records[at:at + trace.RECORD_BYTES] = struct.pack(f"<{trace.RECORD_WORDS}Q", *row)
    with open(path, "wb") as handle:
        handle.write(trace.pack_header(world=world, rank=rank, lanes=lanes, topology=topology,
                                       clock_offset_ns=offset_ns, clock_halfwidth_ns=500, created_ns=1))
        handle.write(bytes(records))


def row(seq: int, *, start: int, doorbell: int, wait_done: int, end: int, flags: dict[int, int],
        cpu: dict[str, int]) -> list[int]:
    words = [0] * trace.RECORD_WORDS
    words[trace.W_SEQ], words[trace.W_START], words[trace.W_DOORBELL] = seq, start, doorbell
    words[trace.W_WAIT_DONE], words[trace.W_END] = wait_done, end
    for index, value in flags.items():
        words[trace.W_FLAGS + index] = value
    for name, value in cpu.items():
        words[trace.W_PROXY + trace.PROXY_WORDS.index(name)] = value
    return words


class TraceFormatTest(unittest.TestCase):
    def test_layout_fits_and_header_round_trips(self) -> None:
        self.assertLessEqual(trace.W_PROXY + len(trace.PROXY_WORDS), trace.RECORD_WORDS)
        self.assertEqual(trace.RECORDS & (trace.RECORDS - 1), 0)  # the kernel masks with RECORDS - 1
        self.assertEqual(len(trace.pack_header(world=4, rank=2, lanes=4, topology="ring4", clock_offset_ns=-7,
                                               clock_halfwidth_ns=3, created_ns=9)), trace.HEADER_BYTES)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "t.bin")
            write_trace(path, world=4, rank=2, lanes=4, topology="ring4", offset_ns=-7, rows=[
                row(5, start=10, doorbell=20, wait_done=30, end=40, flags={}, cpu={}),
                row(6, start=10, doorbell=0, wait_done=30, end=40, flags={}, cpu={}),  # no doorbell: dropped
            ])
            loaded = trace.read(path)
        self.assertEqual((loaded.world, loaded.rank, loaded.lanes, loaded.topology, loaded.clock_offset_ns),
                         (4, 2, 4, "ring4", -7))
        self.assertEqual([r[trace.W_SEQ] for r in loaded.records], [5])

    def test_the_proxy_and_kernel_agree_on_the_record(self) -> None:
        proxy = (ROOT / "_roce_proxy.c").read_text()
        for index, name in enumerate(trace.PROXY_WORDS):
            self.assertIn(f"TRACE_{name.upper()} = {index}", proxy)
        self.assertIn(f"TRACE_SEQ = {trace.W_PROXY_SEQ - trace.W_PROXY}", proxy)
        self.assertLess(trace.W_PROXY_SEQ, trace.RECORD_WORDS)
        self.assertIn("#define ROCE_ABI_VERSION 10", proxy)  # wire ABI unchanged: tracing is local
        self.assertIn("lib.roce_trace_open", (ROOT / "_proxy.py").read_text())
        runtime = (ROOT / "runtime.py").read_text()
        self.assertIn("_trace.W_PROXY", runtime)
        self.assertIn("NotImplementedError", (ROOT / "_oneshot_tilelang.py").read_text())

    def test_untraced_kernels_compile_none_of_the_trace(self) -> None:
        lines = (ROOT / "_oneshot_cute.py").read_text().splitlines()
        guarded: set[int] = set()
        guards = 0
        for i, line in enumerate(lines):
            if line.strip() == "if cutlass.const_expr(self._trace):":
                guards += 1
                indent = len(line) - len(line.lstrip())
                j = i + 1
                while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent):
                    guarded.add(j)
                    j += 1
        stores = [i for i, line in enumerate(lines) if "st_relaxed_sys_u64(" in line and "import" not in line]
        self.assertEqual(guards, 5)  # start, doorbell, peer flags, wait done, end
        self.assertTrue(stores)
        self.assertTrue(all(i in guarded for i in stores), "a trace store outside const_expr(self._trace)")


class TraceAnalysisTest(unittest.TestCase):
    def test_phases_put_the_proxy_on_the_gpu_clock(self) -> None:
        offset = 1_000_000  # GPU = CPU + 1 ms
        records = [row(s, start=100_000 * s, doorbell=100_000 * s + 3_000, wait_done=100_000 * s + 23_000,
                                end=100_000 * s + 30_000, flags={1: 100_000 * s + 20_000},
                                cpu={"seen": 100_000 * s + 4_000 - offset, "posted": 100_000 * s + 6_000 - offset})
                   for s in range(1, 21)]
        t = trace.Trace(2, 0, 1, "direct", offset, 0, "h", records)
        phases = trace.phases(t)
        self.assertEqual(phases["staging"]["p50_us"], 3.0)
        self.assertEqual(phases["proxy_wake"]["p50_us"], 1.0)
        self.assertEqual(phases["proxy_post"]["p50_us"], 2.0)
        self.assertEqual(phases["wait"]["p50_us"], 20.0)
        self.assertEqual(phases["reduce"]["p50_us"], 7.0)
        self.assertEqual(phases["arrival_peer1_direct"]["p50_us"], 17.0)
        self.assertIsNone(phases["proxy_relay"])

    def test_pair_clock_recovers_offset_and_one_way_delay(self) -> None:
        rng = random.Random(1)
        b_ahead, delay = 5_000, 3_000  # b's GPU clock reads 5 us more; 3 us one way
        rows_a, rows_b = [], []
        for seq in range(1, 401):
            t0 = 1_000_000 * seq
            post_a, post_b = t0 + 2_000, t0 + 2_500  # true times
            jitter = (rng.randrange(4_000), rng.randrange(4_000))
            seen_b = post_a + delay + int(jitter[0])  # b sees a's flag (true time)
            seen_a = post_b + delay + int(jitter[1])
            rows_a.append(row(seq, start=t0, doorbell=t0 + 1_000, wait_done=seen_a + 500, end=seen_a + 5_000,
                              flags={1: seen_a}, cpu={"posted": post_a}))
            rows_b.append(row(seq, start=t0 + b_ahead, doorbell=t0 + 1_500 + b_ahead,
                              wait_done=seen_b + 500 + b_ahead, end=seen_b + 5_000 + b_ahead,
                              flags={0: seen_b + b_ahead}, cpu={"posted": post_b + b_ahead}))
        a = trace.Trace(2, 0, 1, "direct", 0, 0, "a", rows_a)
        b = trace.Trace(2, 1, 1, "direct", 0, 0, "b", rows_b)
        pair = trace.pair_clock(a, b)
        self.assertAlmostEqual(pair["offset_us"], 5.0, delta=0.5)
        self.assertAlmostEqual(pair["one_way_a_to_b"]["p5_us"], 3.0, delta=0.6)
        self.assertAlmostEqual(pair["doorbell_b_minus_a"]["p50_us"], 0.5, delta=0.5)
        summary = trace.summarize([b, a])
        self.assertEqual([p["ranks"] for p in summary["pairs"]], [[0, 1]])

    def test_proxy_words_of_another_op_are_ignored(self) -> None:
        r = row(7, start=1, doorbell=10, wait_done=20, end=30, flags={}, cpu={"seen": 15})
        t = trace.Trace(2, 0, 1, "direct", 0, 0, "h", [r])
        r[trace.W_PROXY_SEQ] = 7
        self.assertEqual(trace.phases(t)["proxy_wake"]["p50_us"], 0.01)
        r[trace.W_PROXY_SEQ] = 7 + trace.RECORDS  # an all-gather reused the slot
        self.assertIsNone(trace.phases(t)["proxy_wake"])

    def test_select_keeps_a_sequence_range_and_size(self) -> None:
        rows = [row(s, start=1, doorbell=2, wait_done=3, end=4, flags={}, cpu={}) for s in range(1, 11)]
        for r in rows:
            r[trace.W_META] = (61440 if r[trace.W_SEQ] % 2 else 4096) + (4 << 32)
        t = trace.Trace(2, 0, 1, "direct", 0, 0, "h", rows)
        picked = trace.select(t, min_seq=3, max_seq=8, nbytes=61440)
        self.assertEqual([r[trace.W_SEQ] for r in picked.records], [3, 5, 7])
        self.assertEqual(trace.sizes(t), {61440: 5, 4096: 5})

    def test_ring_relays_only_the_opposite_rank(self) -> None:
        self.assertTrue(trace.adjacent(4, 0, 1, "ring4"))
        self.assertTrue(trace.adjacent(4, 0, 3, "ring4"))
        self.assertFalse(trace.adjacent(4, 0, 2, "ring4"))
        self.assertTrue(trace.adjacent(3, 0, 2, "direct"))


if __name__ == "__main__":
    unittest.main()
