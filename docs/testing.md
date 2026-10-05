# Testing

| Layer | What runs | Where |
| --- | --- | --- |
| Unit | topology validation and rendering, discovery from synthetic LLDP data, profiles, policy, probe planning, GPUDirect report from a fake root, CLI | any host, `make test` |
| Proxy simulator | the production C proxy under a fake verbs layer with ASan/UBSan: direct3, ring4, mesh4, 1/2/4 paths, 26,000 collectives | any host with a C compiler, part of `make test` |
| Launcher geometry | the kernels' flag-selection expressions and cache keys executed from the source AST | any host |
| GPU | `tests/gpu/test_oneshot_gpu.py` under torchrun, 2 or more Sparks, NCCL parity, fault injection, and bit equality between the two kernel families; `scripts/gpu-test-fleet.sh` runs it on every node of a map inside the serving image | the fleet, in an owned window |
| Fabric | `sparknet probe collectives`: policy correctness with graph replay, dispatch boundaries, latency screen with RDMA and port counters | the fleet, one bounded container per rank |
| NCCL build | the series applies to the pinned release and reproduces the recorded tree | CI (`nccl-patches` job) and `scripts/build-nccl.sh` |
| Lint and versions | ruff (`make lint`); the suite on Python 3.10, 3.12 and 3.14 | CI (`library` matrix) |
| Packaging | the wheel installs with no dependencies and its CLI validates, renders, exports the NCCL series and plans a probe | CI (`library` job) |
| Dockerfile | `docker buildx build --check` on `docker/Dockerfile`; the image is built by hand on a Spark, not in CI | CI (`dockerfile` job) |

First fleet run of the GPU suite and the probe through this package:
2026-10-05, dgx1-dgx2 pair (`tp2-direct`) and the four-node ring
(`tp4-ring`), r6 image, both kernel families (`evidence/2026-10-05-tilelang-port`).
Two findings about the suite itself: the per-HCA striping check is a
direct-clique property and skips on a ring, and the fault-injection tests
(`test_fail_stop_*`) are not yet stable on the four-node ring (the fresh
runtime's teardown synchronizes the device while a peer is still spinning,
and NCCL's timeout aborts the process); they pass on the pair and are
deselected on the ring until the harness handles a ring-wide timeout.

The probe is a prerequisite for serving, not serving acceptance: model
quality, sustained thermal behaviour and long-context admission are the
recipe's own gates.
