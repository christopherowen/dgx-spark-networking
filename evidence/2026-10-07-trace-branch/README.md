# 2026-10-07: the trace branch on the four-node ring

Two windows on the fleet (serving stopped, r6 image `vllm-ds41f-kkref:04c30fa98e79-r6`,
the checkout mounted over the image's `sparknet` package, the four-node ring).

| Directory | What |
| --- | --- |
| `gpu-tests/failstop-cute`, `gpu-tests/failstop-tilelang` | `tests/gpu/test_oneshot_gpu.py -k fail_stop` per node, with each kernel family as the runtime's family: 2 passed on every rank |
| `ring-tilelang-1`, `ring-cute-1`, `ring-tilelang-2` | `sparknet probe fleet --transport oneshot-ring4 --profile tp4-ring -- --benchmark`, run in that order in one window |

The fail-stop tests ran on the ring for the first time: the 2026-10-05 ring
qualification deselected them, because on the ring a stopped peer's neighbours
report their relay's proxy failure (`RoCE proxy failed`) before their own wait
times out (`poisoned`). The tests now accept either; both are the fail-stop
outcome.

The earlier window that day ran the rest of the GPU suite on a superset of this
code (the branch before it was split into separate pieces): 136 passed with the
CuTe family as the runtime's (TileLang as the second family), 105 passed with the
TileLang family traced (`SPARKNET_ROCE_TRACE=1`), on every rank.

The probe needs `management_interface` in its node map on this fleet: every
hostname resolves to 127.0.0.1, so without `GLOO_SOCKET_IFNAME` the probe's Gloo
bootstrap binds to loopback and times out (the CLI now warns).

Tables: `sparknet probe summarize tl-1005=../2026-10-05-tilelang-port/ring-tilelang tl-today=ring-tilelang-1 tl-today2=ring-tilelang-2`.

| All-reduce, ring | TileLang 10-05 (two runs) | TileLang today (two runs) | CuTe 10-05 | CuTe today |
| --- | ---: | ---: | ---: | ---: |
| 10 KiB | 18.1, 18.1 us | 20.5, 18.6 us | 16.2 us | 17.9 us |
| 60 KiB | 28.5, 31.2 us | 30.0, 29.5 us | 28.3 us | 27.9 us |
| 480 KiB | 92.9, 92.2 us | 96.0, 95.3 us | 96.6 us | 94.5 us |

The untraced TileLang kernel of this branch is byte-identical to main's: the
generated CUDA of both, compiled with nvcc 13.0 for `sm_121a`, gives identical
PTX and the same cubin (40 registers each). The tracing code is compiled only
into the traced kernel. The day-to-day movement above (TileLang 3% slower and
CuTe 2% faster at 480 KiB than on 10-05) is therefore the fabric's spread
between days, not this branch; a code comparison belongs in one window with the
two packages as alternating arms.
