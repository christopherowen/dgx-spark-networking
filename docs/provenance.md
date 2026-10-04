# Provenance

Everything with a measured number in this repository names where it was
measured. The deployment repository is spark3-vllm-ds41f
(`https://github.com/christopherowen/spark3-vllm-ds41f`); its experiments
are referenced by directory name.

## RoCEnante (`sparknet/rocenante`)

Vendored from local-inference-lab/b12x `b12x/comm/roce/` at
`f8069b2c0be1311df3b112591c6b8876a843f8be` (`integration/karmic-kraken-beta`)
with the eleven spark3 B12X patches applied in order (patch head
`abe4b2af`, tree `cd615bd6`), the tree labelled in the
`roce-balanced-dispatch-v1` serving image that measured the balanced policy:

| Patch | What it adds |
| --- | --- |
| 0001 switchless-rocenante | per-peer HCA routes for non-clique fabrics; flag layout by route lanes |
| 0002 cutlass-dsl-4.7.1 | (b12x pin; not relevant to the vendored files) |
| 0003, 0004, 0005 | (kernel fixes outside `comm/roce`) |
| 0006 rocenante-ring4 | neighbour-only QPs, host relay of the opposite rank, ABI 5, the fake-verbs simulator |
| 0007 rocenante-mesh4 | NIC-forwarded opposite QPs with flow-label marking, ABI 6 |
| 0008 four-path-mesh | four opposite paths, ABI 8 |
| 0009 bidirectional-relay | disjoint halves relayed in both directions, independent per-direction progress, ABI 9 |
| 0010 dispatch-capacity | `ALLREDUCE_DISPATCH_MAX_BYTES` distinct from registered capacity, ABI 10 |
| 0011 loader-abi | Python loader pinned to ABI 10 |

Local changes on top (not hardware changes): imports moved from
`b12x._lib.*` to `sparknet.rocenante._compile` (CuTe DSL `cute.compile` with
the DSL's own cache, `functools.cache` in place of b12x's program cache, a
freeze flag); the b12x preparation plan (`plan=` argument,
`b12x.preparation`) replaced by in-process `prepare`; `SPARKNET_ROCE_*`
names with `B12X_ROCE_*` aliases; a `transport` factory and `topology`
keyword; `make_ptr` from `cutlass.cute.runtime`. The C proxy differs from
the vendored file only in two comment lines and the `MESH_ROTATE`
environment name lookup. Verify with `diff` against the tree named in
`upstreams.lock.json`.

Not carried: the relay streaming and progress-window patches (0012 to 0014,
measured slower), the mesh4 host marker and TC tooling (not recommended),
the b12x preparation session.

## NCCL (`patches/nccl`)

NCCL `v2.30.7-1`; the series is spark3's `nccl-adaptive` series (patch head
`eeacf1c6`, tree `6af10aa7`). 0001 by Stanislav Bardyuk (NVIDIA/nccl#2393);
0002 to 0004 by Christopher Owen from `2026-10-03-nccl-bidirectional` and
`2026-10-03-balanced-policy`.

## Profiles (`sparknet/nccl/profiles.py`)

`tp3-triangle` from spark3 `config/cluster.json` at baseline
`2026-10-02-karmic-kraken-r5o-64k`; `tp4-ring` from
`experiments/2026-10-03-balanced-policy/selected.json`; `tp4-ring-nccl-only`
from `experiments/2026-10-03-collective-serving` (four-channel arm).

## Topology tooling

Validation ported from spark3 `scripts/topology.py` (the vLLM-specific
argument checks stayed there); discovery from the fleet's `cx7-config.py`.
Example maps use documentation addresses (RFC 5737) and the fleet's cable
subnet rule.

## GPUNetIO (`native/gpunetio`)

NVIDIA-DOCA/gpunetio at `586453728bca`, patches from
`2026-10-03-relay-progress` (heads in `sources.json`).

## Licenses

b12x and SparkRing: Apache-2.0. NCCL and DOCA GPUNetIO: BSD-3-Clause. See
`NOTICE`.
