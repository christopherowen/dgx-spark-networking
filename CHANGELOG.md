# Changelog

## Unreleased

- First fleet qualification of the package's own GPU suite and probe
  (2026-10-05, dgx1-dgx2 pair and the four-node ring, r6 image, both kernel
  families): bit equality between the CuTe and TileLang kernels on every
  fabric; the two-Spark pair measured for the first time; receipts under
  `evidence/2026-10-05-tilelang-port`. `scripts/gpu-test-fleet.sh` runs the
  GPU suite on every node of a map.
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
