# one-shot proxy simulator

`simulate.c` includes the production `sparknet/oneshot/_roce_proxy.c` and
replaces only libibverbs (`infiniband/verbs.h`) and the GPU endpoint. It runs
26,000 collectives across direct3, ring4 and mesh4 with one, two and four
paths: delayed DMA that reads source bytes at delivery (exposing premature
slot reuse), independent per-direction progress, sequence wrap, a proxy that
misses two doorbells, mixed-ABI rejection, oversized payloads, a relay
stopped by a timed-out collective or by its own timeout, stop paths, bidirectional byte balance and unique pack delivery,
and the source-GID scan over fake GID tables (a healthy port, a GID moved to
the next slot after a flap, another subnet's address, none, two) with each
queue pair's address vector using its own HCA's slot.

`tests/test_oneshot_cpu.py` builds and runs it with AddressSanitizer and
UndefinedBehaviorSanitizer. It validates host protocol logic; it does not
emulate NIC coherence, GPU memory ordering, registration, the kernel's GID table or
compiled code generation. The GPU test in `tests/gpu` and the collective
probe supply those.
