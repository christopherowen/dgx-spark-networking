"""Triangle tables from the probe receipts: per run, the slowest rank's median of its samples;
per arm, the median over runs. python3 tables.py DIR (probe-adapter, diag or diag2)."""
import collections
import glob
import json
import os
import re
import statistics
import sys

directory = sys.argv[1]
table = collections.defaultdict(lambda: collections.defaultdict(list))
for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
    match = re.match(r"(?:sp2?-)?(.+)-(\d)-(dgx\d)\.json", os.path.basename(path))
    if not match:
        continue
    arm, run, _ = match.groups()
    record = json.load(open(path))
    assert record["passed"], path
    for timing in record["timings"]:
        key = (timing["dtype"].split(".")[-1], timing["operation"], timing["elements_per_rank"])
        table[key][(arm, run)].append(statistics.median(timing["microseconds_per_call"]))
arms = sorted({arm for cases in table.values() for arm, _ in cases})
print("dtype operation KiB " + " ".join(arms))
for (dtype, operation, elements), runs in sorted(table.items()):
    kib = elements * (2 if dtype == "bfloat16" else 4) // 1024
    per_arm = {arm: statistics.median(max(v) for (a, _), v in runs.items() if a == arm) for arm in arms}
    print(f"{dtype} {operation} {kib} " + " ".join(f"{per_arm[arm]:.1f}" for arm in arms))
