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
