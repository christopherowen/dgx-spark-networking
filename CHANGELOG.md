# Changelog

## Unreleased

- **A wider data phase with one poller.** The launch grid gives every thread
  one 16-byte pack, up to 64 blocks of 512 threads (0.4.0: about two packs a
  thread on at most 8 blocks), and only block 0 polls the peers' flags in the
  pinned region, once the doorbell has rung; the other blocks wait in device
  memory for block 0's arrival word. The data phase reads the peers' slots
  from pinned host memory, and beside the model's L2 prefetch each round trip
  there waits out the DRAM queue (about 6 us on a GB10); a thread's packs are
  serial round trips. Every waiting thread's poll is a system-scope load of the
  same region, and with every block polling the wide grid was up to 58 percent
  slower on the triangle: blocks that had staged polled while the others were
  still storing, so the doorbell came late. Triangle, BF16, against 0.4.0:
  all-reduce 19.6 to 17.6 us at 60 KiB, 32.5 to 28.7 at 160 KiB, 62.2 to 59.0
  at 480 KiB; all-gather 19.6 to 17.5, 35.3 to 29.5, 69.2 to 65.0; level at
  10 KiB. In TP3 serving the engine grid's decode step time is about
  1.6 percent lower with identical outputs (`evidence/2026-10-20-wide-read`).
  The runtime refuses a grid the
  device cannot hold at once: blocks waiting for peers keep their SMs while
  the doorbell waits for every block to stage. `blocks` is in the setup
  handshake, so ranks of different versions fail at construction.
- **The all-gather's data phase is one round trip.** It issued a load and a
  store per source, and a store orders the loads after it, so every peer was
  a round trip of its own. Every peer's load of a pack now issues before the
  first store (groups of four peers bound the registers), and staging writes
  the rank's own shard to its block of the output, so the data phase reads
  only the peers.
- Every launcher still compiles to 40 registers (2 to 16 ranks, every
  dtype, ring4, traced) and the all-gather uses no local memory; the
  eight-rank all-reduce keeps the one spill it had in 0.4.0.
- `benchmarks/benchmark_oneshot.py` uses the runtime's grid limit unless
  `--runtime-blocks` is given.

## 0.4.0 (2026-10-16)

- **One kernel family: TileLang.** The vendored CuTe DSL kernels
  (`_oneshot_cute.py`, `_allgather_cute.py`, `_cute_intrinsics.py`,
  `_compile.py`) are retired; the TileLang kernels they were bit-identical to
  (2026-10-05 qualification) are the only ones. Every protocol change had to be
  written and qualified twice, and a family production does not run is an
  untested path. `SPARKNET_ROCE_KERNELS`, the runtime's `kernels` keyword and
  the probe's `--kernels` are refused or removed; the `runtime` extra needs
  `tilelang` instead of `nvidia-cutlass-dsl`. The trace's clock calibration
  kernel is TileLang too. The CPU test compiles the generated header's
  flag selection with the host compiler instead of executing the CuTe
  kernel's Python expressions.
- **Fail-stop that stops** (API version 2). A collective wait that sees nothing
  from a peer for the timeout now writes its record (the peer, its HCA lane,
  the time waited, the sequence) and stops the GPU with `trap`. Before, the
  failing collective and every later one became no-ops and the step kept
  computing on unwritten buffers until the host's health check; a check
  inside the step (a model's NaN assert) or a probe's verification could fire
  first and hide the cause. The poison word, the skipped data phase and the
  kept epoch are gone from both kernels.
- **The timeout is a duration and can change after capture.**
  `SPARKNET_ROCE_TIMEOUT_MS` (default 10,000) replaces
  `SPARKNET_ROCE_SPIN_LIMIT`, a poll count whose duration was never measured
  on GB10; the old variable is refused with the new name. Kernels read the
  timeout from the control record when a wait passes its first clock check
  (after 64 polls, so short waits cost nothing extra), and
  `set_timeout_ms()` changes it for captured graphs too.
- **The ring relay uses the same timeout.** It had a fixed 5 s of its own, so
  a legitimately stalled neighbour failed the relay whatever the kernels'
  limit. Its stop message now says whether the runtime was closing or this
  rank's collective had timed out (it said "relay stopped or GPU failed").
- `failure()` returns the report without raising (readable after the GPU
  stopped: the record is in pinned host memory) and puts the kernel's record
  before a relay's failure, which follows from it. `check_health` raises it.
  The vLLM adapter exposes `failure()`.
- The probe reports a stopped collective's record instead of CUDA's launch
  failure or a numerical mismatch.
- GPU fail-stop tests run each fault in a child process per rank and check
  the detection time against the timeout, including a timeout lowered after
  graph capture.

- Per-op timing trace (`SPARKNET_ROCE_TRACE=1`, `sparknet.oneshot.trace`):
  the kernels stamp `%globaltimer` at start, doorbell, every peer lane's
  flag, end of wait and end; the proxy stamps doorbell seen, posts, relay
  and drain on `CLOCK_MONOTONIC_RAW`, put on the GPU clock by a per-node
  calibration; the analysis estimates pair clock offsets from the two-way
  exchanges. Both kernel families; untraced kernels compile none of it.
- The TileLang kernels declare full residency (`T.annotate_min_blocks_per_sm`
  with the SM's thread limit over the launch's threads: 3 of 512 on the
  GB10), which bounds them to CuTe's 40 registers. Left to nvcc they used 54
  to 56, and decode profiles showed every graph-replayed all-reduce about
  15 us slower beside the model's L2 prefetch; with the bound the TileLang
  family is level with CuTe in serving. Receipts:
  `evidence/2026-10-05-tilelang-port/decode-profiles`, `serving-bench-regs40`.
- First fleet qualification of the package's own GPU suite and probe
  (2026-10-05, dgx1-dgx2 pair and the four-node ring, r6 image, both kernel
  families): bit equality between the CuTe and TileLang kernels on every
  fabric; the two-Spark pair measured for the first time; receipts under
  `evidence/2026-10-05-tilelang-port`. `scripts/gpu-test-fleet.sh` runs the
  GPU suite on every node of a map.
- Serving benchmark of both kernel families and of proxy pinning on the TP4
  recipe, a one-shot latency benchmark on the pair (TileLang's eager launch
  is 15 us cheaper, graph latency equal), and a crossover sweep that places
  the ring's all-reduce cut between 1.25 and 1.5 MiB and finds the pair's
  cut capacity-bound; receipts under `evidence/2026-10-05-tilelang-port`.
  Neither family change nor pinning is promoted. The probe's default sizes
  gain the 960 KiB point (TP4's largest decode capture).
- Both kernel families launch from one `Launch` description; the TileLang
  kernels take the input, output, pinned region and counters as buffer
  arguments with their real shapes (which binds the launch to the current
  device and stream), and the device header takes pointers.
- The integration guide names the kernel caches (`CUTE_DSL_CACHE_DIR`,
  `TILELANG_CACHE_DIR`) as the toolchains' own, placed by the recipe with its
  other JIT caches; the reference Dockerfile sets neither.
- A TileLang kernel family for the one-shot all-reduce and all-gather
  (`SPARKNET_ROCE_KERNELS=tilelang`, or `kernels=` on the runtime): the same
  protocol phases generated as CUDA source with the device side in
  `_device.py`, meant to be bit-identical to the CuTe DSL kernels; the GPU
  test compares the two families. The default stays `cute` until the fleet
  has qualified the port. The kernel-resolution freeze is shared (`_freeze`).
- `SPARKNET_ROCE_PROXY_CPU`: proxy thread placement (`none`, a CPU number, or
  `big`, which confines the thread to the big-core cluster, the cores above the
  midpoint between the smallest and largest capacity: the GB10's ten X925 cores). The thread is named `sparknet-proxy`, and `stats()`
  reports `proxy_cpus` and the CPU it first ran on. Measured on the probe
  (pinning held the 10 KiB case within 0.6 us where unpinned runs spread
  over 2.6 us); a candidate for the recipe environment pending a serving
  benchmark.
- `sparknet topology subset`: carve a pair or a triangle out of a cabled map,
  so a ring owner can measure a pair without recabling.
- `sparknet probe fleet`: run every rank's probe container at once over ssh,
  keep the receipts, print the latency table; `--package-source` mounts a
  checkout over the image's package, `--env` adds a candidate setting.
  `sparknet probe summarize` tabulates receipts with deltas. The container
  plan runs the image's own probe unless `--probe-source` is given.
- The probe takes `--runtime-threads` and `--runtime-blocks` for launch
  geometry sweeps and records its `SPARKNET_ROCE_*` and `NCCL_*` environment
  in the receipt.

## 0.3.0 (2026-10-05)

- The NCCL patch series ships inside the package (`sparknet/nccl/patches`,
  formerly `patches/nccl`): `sparknet nccl patches` works after
  `pip install`, and `sparknet nccl patches --export DIR` writes the series
  and its patches for an image build. `sparknet.nccl.patchset` is the API.
- `docker/Dockerfile`: a reference image for the fabric parts of a serving
  image (patched NCCL built from the pinned release, the wheel, the proxy
  built at image time, an import check of the NCCL hash and proxy ABI).
- CLI: `probe render-command ... -- --benchmark` works on Python 3.10 (the
  `--` separator is taken out before argparse sees it); `policy show`
  without a profile reports a missing environment instead of a traceback.
- The vLLM adapter's export list no longer names the `enabled` function
  removed in 0.2.0.
- CI runs ruff and the suite on Python 3.10, 3.12 and 3.14, installs the
  wheel and drives the CLI from it, and lints the Dockerfile.
- README: production status (spark-ds41f r6 serves both promoted recipes on
  this package), the measured numbers up front, install, questions,
  contributing; CONTRIBUTING.md and issue templates.

## 0.2.0 (2026-10-04)

- Only `SPARKNET_ROCE_*` names are read; the `B12X_ROCE_*` and `VLLM_*`
  aliases and the renderer's engine switches are gone. The NCCL fallbacks
  (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_IB_TC`) stay.
- One-shot is always on where it is constructed: no enable switch, no
  `enabled()`.

## 0.1.0 (2026-10-04)

- First release: the one-shot collectives vendored from the hardware-qualified
  spark-ds41f tree (direct, ring4, mesh4), the NCCL 2.30.7 series and the
  measured profiles, the policy, topology validation, rendering and LLDP
  discovery, the fabric doctor, the collective probe and the GPUDirect
  readiness report, the transport boundary with GPUNetIO staged, the vLLM
  adapter, the integration guide and a minimal engine example.
- Two-node, switched and NCCL-only fabrics as configuration paths.
