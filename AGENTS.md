# Agent guide

Read `README.md`, `docs/design.md`, `docs/provenance.md` and
`upstreams.lock.json` before changing runtime or build state.

## Sources of truth

- `upstreams.lock.json` pins every external source, the patch heads and the
  resulting tree hashes. A vendored file is changed only by recording why in
  `docs/provenance.md` and updating `local_changes`.
- `sparknet/nccl/profiles.py` holds the measured profiles. A number there
  names the experiment that measured it; do not retune a profile without a
  new measurement and its receipt.
- `sparknet/topology/examples/` are documentation maps with documentation
  addresses. Site maps (`nodes.json`, `*.local.json`) are git-ignored.
- Results of inspiration sources and comparisons with other stacks stay out
  of this repository. An idea enters as a change measured on this stack.

## Change discipline

- Protocol and kernel changes (`sparknet/oneshot`, `patches/nccl`) are
  hardware changes. They need the C simulator, the GPU test under torchrun
  and the collective probe on the actual fabric before any profile uses them.
  Bump the proxy ABI whenever the wire layout or the geometry handshake
  changes, so mixed ranks fail at connect instead of hanging.
- Keep eligibility rank-invariant (dtype, shape, contiguity, byte size) and
  failures fail-stop. Never add a fallback that lets one rank take a
  different backend than its peers.
- No agent or tool attribution in branches, commits or pull requests. Plain
  commit messages, no trailers.
- Run `make test` and `git diff --check` before every push, and watch the
  Actions run for the pushed branch.
- Host changes (netplan, drivers, NIC parameters, kernel) belong to the
  owner's fleet tooling, not to this library; the doctor reports, it never
  mutates.

## Shared cluster

The Sparks serve production. Probes and GPU tests run only inside an owned
cluster window (`~/spark3-hold.json` on the head node) with serving stopped,
one rank per node, bounded containers. Never start, stop or restart serving
from this repository.
