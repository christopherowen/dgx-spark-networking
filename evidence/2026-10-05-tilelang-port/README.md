# 2026-10-05: first fleet qualification through this package; TileLang kernel family

Window on the fleet (serving stopped, r6 image `vllm-ds41f-kkref:04c30fa98e79-r6`,
this checkout mounted over the image's `sparknet` package), the four-node ring
cabled, `dgx1`-`dgx2` carved from it as the two-Spark pair
(`sparknet topology subset`). Every directory holds the per-rank JSON receipts of
one `sparknet probe fleet` run (`--benchmark`, five samples of 256 graph-replayed
calls per case; the fabric number is the median over samples of the slowest rank).

| Directory | Fabric | Transport, profile | Kernels | Notes |
| --- | --- | --- | --- | --- |
| `pair-cute*`, `pair-tilelang*` | pair | `oneshot-direct`, `tp2-direct` | both | three unpinned runs each; `*big*` pins the proxy to the highest-capacity core |
| `pair-nccl-*` | pair | `nccl-direct`, `direct-nccl-only` | none | NCCL carries everything (control) |
| `ring-cute*`, `ring-tilelang*` | ring | `oneshot-ring4`, `tp4-ring` | both | three runs each |
| `small-*` | ring | `oneshot-ring4`, `tp4-ring` | both | 10 KiB and 60 KiB only, interleaved, with and without proxy pinning |
| `ring-nccl-1` | ring | `nccl-ring`, `tp4-ring-nccl-only` | none | control |
| `gpu-tests/` | pair, ring | | both | `tests/gpu/test_oneshot_gpu.py` summaries per node |

Tables: `sparknet probe summarize cute=pair-cute tilelang=pair-tilelang nccl=pair-nccl-1`.

## Second window the same evening (serving benchmark and the crossover)

| Directory | What |
| --- | --- |
| `serving-bench/` | spark-ds41f lab `measure` run `sparknet1` (lean profile, bracket): arms `cute`, `tilelang`, `cute-big`, `tilelang-big`, `cute-end` on the production TP4 recipe with this checkout mounted over the image's package; `sparknet1-table.txt` is `tables_arms.py`'s comparison, `bench-<arm>.json` the bench receipts |
| `latency-bench/` | `benchmarks/benchmark_oneshot.py` on the pair, both families: eager and graph-replayed one-shot against NCCL |
| `cut-ring-oneshot-*`, `cut-ring-nccl-*` | ring, 480 KiB to 2 MiB, one-shot forced to the 2 MiB capacity (`tp4-ring`) against `tp4-ring-nccl-only` |
| `cut-ring-oneshot4m-*`, `cut-ring-nccl4m` | ring, 2 to 4 MiB with capacity and gather limit raised to 4 MiB, against `tp4-ring-nccl-only` |
| `cut-pair-oneshot-*`, `cut-pair-nccl-*` | pair, 480 KiB to 4 MiB with capacity raised to 4 MiB (`tp2-direct`), against `direct-nccl-only` |

Site management addresses in the receipts are replaced by documentation
addresses (`192.0.2.N` for node N).

The `*-nccl-only` controls are the upstream (one-direction, 1 MiB buffer) NCCL
profiles, which is what a pair or triangle uses above the cut. The ring's policy
hands large collectives to the balanced four-channel NCCL of `tp4-ring`, which is
faster than that control; compare the ring's cut against the balanced numbers in
`docs/nccl.md`.
