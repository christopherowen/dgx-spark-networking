# Recipes

Rendered environments for the named profiles against the example node maps,
regenerated from the library (`sparknet topology render` per node with
`--profile`). Each file lists the common environment, the per-node
additions (peer routes, exact HCA names, GID index, socket interfaces), the
NCCL patches the profile needs and where its settings were measured.

A real deployment renders against its own `nodes.json` and keeps the output
beside its container configuration; the example addresses here are
documentation addresses.

| Recipe | Nodes | Transport | Status |
| --- | --- | --- | --- |
| `tp2-direct.json` | 2 | One-shot direct over one cable + NCCL over both | configuration path, unmeasured here |
| `direct-nccl-only.json` | 2 or 3 | NCCL only, every pair cabled | control |
| `tp3-triangle.json` | 3 | One-shot direct + NCCL | promoted spark3 baseline |
| `tp4-ring.json` | 4 | One-shot ring4 relay + balanced NCCL | measured candidate |
| `tp4-ring-nccl-only.json` | 4 | NCCL neighbour ring | control |
| `switched.json` | 2 to 16 | One-shot clique over two rails + NCCL over every rail | configuration path, unmeasured here |
| `switched-nccl-only.json` | 2 to 16 | NCCL only behind a switch | control |

For the vLLM fork, the environment goes into the container's `environment`
block together with `--distributed-executor-backend mp`, `--nnodes N`,
`--tensor-parallel-size N` and, for one-shot transports, custom all-reduce
left enabled; see `sparknet/integration/vllm/README.md`.
