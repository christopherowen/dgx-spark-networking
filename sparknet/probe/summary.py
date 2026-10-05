"""Latency tables from probe receipts, in the form the documentation uses.

A receipt is one rank's JSON result from ``sparknet probe collectives
--benchmark``. Each timing row holds five samples of microseconds per call
(16 graph replays of 16 calls each). The fabric number for a case is the
median over samples of the slowest rank, so a rank that lags shows up
instead of being averaged away.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ITEM_SIZES = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8, "int8": 1, "uint8": 1, "int32": 4, "int64": 8}
OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter")


def load_results(directory: str | Path) -> dict[str, dict]:
    """``<node>.json`` receipts in a directory, keyed by node name."""
    results = {}
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "timings" in data:
            results[path.stem] = data
    if not results:
        raise FileNotFoundError(f"no probe receipts (<node>.json with timings) under {directory}")
    return results


def dtype_name(value: str) -> str:
    return str(value).replace("torch.", "")


def payload_bytes(row: dict) -> int:
    return int(row["elements_per_rank"]) * ITEM_SIZES[dtype_name(row["dtype"])]


def size_label(nbytes: int) -> str:
    for unit, scale in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if nbytes % scale == 0:
            return f"{nbytes // scale} {unit}"
    return f"{nbytes} B"


def cases(results: dict[str, dict]) -> dict[tuple[str, str, int], dict]:
    """Per (operation, dtype, bytes): the slowest-rank median and each rank's median."""
    samples: dict[tuple[str, str, int], dict[str, list[float]]] = {}
    for node, result in results.items():
        for row in result.get("timings", []):
            key = (row["operation"], dtype_name(row["dtype"]), payload_bytes(row))
            samples.setdefault(key, {})[node] = [float(v) for v in row["microseconds_per_call"]]
    table = {}
    for key, by_node in samples.items():
        count = min(len(v) for v in by_node.values())
        slowest = [max(v[i] for v in by_node.values()) for i in range(count)]
        table[key] = {
            "slowest_rank_median_us": statistics.median(slowest),
            "rank_median_us": {node: statistics.median(v) for node, v in by_node.items()},
            "ranks": len(by_node),
            "samples": count,
            "backend": next((results[n]["timings"][i].get("expected_backend") for n in by_node for i, r in enumerate(results[n]["timings"])
                             if (r["operation"], dtype_name(r["dtype"]), payload_bytes(r)) == key and r.get("expected_backend")), None),
        }
    return table


def markdown(labelled: dict[str, dict[tuple[str, str, int], dict]], *, dtype: str = "bfloat16") -> str:
    """One table per operation: sizes down, one column per label, deltas against the first label."""
    labels = list(labelled)
    lines = []
    for operation in OPERATIONS:
        sizes = sorted({key[2] for table in labelled.values() for key in table if key[0] == operation and key[1] == dtype})
        if not sizes:
            continue
        header = [f"{operation.replace('_', '-')} ({dtype}), per rank"] + labels[:1] + [f"{label} (vs {labels[0]})" for label in labels[1:]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in labels) + " |")
        for nbytes in sizes:
            cells = [size_label(nbytes)]
            base = labelled[labels[0]].get((operation, dtype, nbytes))
            for index, label in enumerate(labels):
                entry = labelled[label].get((operation, dtype, nbytes))
                if entry is None:
                    cells.append("")
                    continue
                value = entry["slowest_rank_median_us"]
                text = f"{value:.1f} us"
                if entry.get("backend"):
                    text += f" ({entry['backend']})"
                if index and base:
                    delta = (value - base["slowest_rank_median_us"]) / base["slowest_rank_median_us"] * 100
                    text += f", {delta:+.1f}%"
                cells.append(text)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["ITEM_SIZES", "OPERATIONS", "cases", "load_results", "markdown", "payload_bytes", "size_label"]
