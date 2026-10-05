# Changelog

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
