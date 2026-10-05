"""dgx-spark-networking: switchless RoCE collectives, NCCL profiles and fabric tooling for DGX Spark inference recipes.

Import name: ``sparknet``. Subpackages:

- ``sparknet.topology``: node maps, cabling validation, per-node environment rendering, LLDP discovery.
- ``sparknet.nccl``: measured NCCL environment profiles and the patch series the ring profiles need.
- ``sparknet.policy``: the explicit collective policy (which backend carries which collective).
- ``sparknet.oneshot``: the one-shot RoCE all-reduce and all-gather runtime (needs torch and the CuTe DSL).
- ``sparknet.transport``: the RDMA transport boundary, today the host proxy, with the GPU-initiated path staged.
- ``sparknet.probe``: fabric doctor, collective correctness/latency probe, GPUDirect capability report.
- ``sparknet.integration.vllm``: the vLLM device-communicator adapter.

The CPU-only subpackages import nothing from torch, so the CLI and recipe
tooling run on any host; ``sparknet.oneshot`` is imported only where a GPU
runtime is constructed.
"""

__version__ = "0.3.0"

# Bumped when the runtime surface used by integrations changes incompatibly.
API_VERSION = 1

__all__ = ["API_VERSION", "__version__"]
