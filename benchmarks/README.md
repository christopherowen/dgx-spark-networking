# Benchmarks

`benchmark_oneshot.py` times the one-shot all-reduce and all-gather
against torch.distributed (NCCL) on every rank, eager and graph-replayed,
with correctness gates before and after timing and one JSON receipt (schema
`sparknet.oneshot.benchmark`) holding the command, source revision, GPU
identities, raw samples, per-rank medians and the cross-rank
median-of-slowest summary. Launch with torchrun, one process per node, in an
owned cluster window with serving stopped.
