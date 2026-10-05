# Contributing

Issues and pull requests are welcome. The most useful contribution is a
measurement from a fabric this fleet does not have: a two-Spark pair, a
switch, other firmware or driver versions. The
[fabric report](.github/ISSUE_TEMPLATE/fabric_report.yml) template lists
what to include.

## Before a pull request

```sh
make lint        # ruff, the configuration in pyproject.toml
make test        # unit tests and the C proxy simulator (needs a C compiler)
git diff --check
```

CI repeats both on Python 3.10, 3.12 and 3.14, installs the wheel and drives
the CLI from it, re-applies the NCCL series to the pinned release and checks
the resulting tree hash, and lints `docker/Dockerfile`.

## What a change needs

- **Protocol and kernel changes** (`sparknet/oneshot`, `sparknet/nccl/patches`)
  are hardware changes. They need the C simulator, the GPU test under torchrun
  (`tests/gpu`) and the collective probe on a real fabric before any profile
  uses them. Bump the proxy ABI whenever the wire layout or the geometry
  handshake changes, so mixed ranks fail at connect instead of hanging.
- **Profiles** (`sparknet/nccl/profiles.py`) hold measured settings. A number
  there names the experiment that measured it; a retune comes with a new
  measurement and its receipt (the probe's JSON result, the NCCL init logs).
- **Vendored files** (`upstreams.lock.json`) change only with a note in
  `docs/provenance.md` and an updated `local_changes` entry.
- **Eligibility stays rank-invariant** (dtype, shape, contiguity, byte size)
  and failures fail-stop. No fallback may let one rank take a different
  backend than its peers.
- **Hosts are not this library's to change.** The doctor reports; netplan,
  drivers, NIC parameters and the kernel belong to a site's own tooling.

Plain commit messages that say what changed and why; no trailers.
