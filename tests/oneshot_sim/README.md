# one-shot proxy simulator

`simulate.c` includes the production `sparknet/oneshot/_roce_proxy.c` and
replaces only libibverbs (`infiniband/verbs.h`) and the GPU endpoint. It runs
26,000 collectives across direct3, ring4 and mesh4 with one, two and four
paths: delayed DMA that reads source bytes at delivery (exposing premature
slot reuse), independent per-direction progress, sequence wrap, a proxy that
misses two doorbells, mixed-ABI rejection, oversized payloads, GPU poison and
stop paths, bidirectional byte balance and unique pack delivery.

`tests/test_oneshot_cpu.py` builds and runs it with AddressSanitizer and
UndefinedBehaviorSanitizer. It validates host protocol logic; it does not
emulate NIC coherence, GPU memory ordering, registration, GID addressing or
compiled code generation. The GPU test in `tests/gpu` and the collective
probe supply those.
