# Testing

| Layer | What runs | Where |
| --- | --- | --- |
| Unit | topology validation and rendering, discovery from synthetic LLDP data, profiles, policy, probe planning, GPUDirect report from a fake root, CLI | any host, `make test` |
| Proxy simulator | the production C proxy under a fake verbs layer with ASan/UBSan: direct3, ring4, mesh4, 1/2/4 paths, 26,000 collectives | any host with a C compiler, part of `make test` |
| Launcher geometry | the kernels' flag-selection expressions and cache keys executed from the source AST | any host |
| GPU | `tests/gpu/test_oneshot_gpu.py` under torchrun, 2 or more Sparks, NCCL parity and fault injection | the fleet, in an owned window |
| Fabric | `sparknet probe collectives`: policy correctness with graph replay, dispatch boundaries, latency screen with RDMA and port counters | the fleet, one bounded container per rank |
| NCCL build | the series applies to the pinned release and reproduces the recorded tree | CI (`nccl-patches` job) and `scripts/build-nccl.sh` |

The probe is a prerequisite for serving, not serving acceptance: model
quality, sustained thermal behaviour and long-context admission are the
recipe's own gates.
