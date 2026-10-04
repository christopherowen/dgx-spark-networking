"""Read-only fabric checks and the model-free collective probe.

- ``doctor``: configuration and live-link checks on one node, before any queue pair opens.
- ``counters``: RDMA error and physical-port counters around a measurement.
- ``container``: the bounded Docker command that runs the probe on one rank.
- ``gpudirect``: what the host offers for the GPU-initiated transport.
- ``collectives``: the torch.distributed probe (needs torch; run on the nodes).
"""

from .container import docker_probe_command
from .counters import port_counters, rdma_error_counters
from .doctor import local_problems
from .gpudirect import gpudirect_report

__all__ = ["docker_probe_command", "gpudirect_report", "local_problems", "port_counters", "rdma_error_counters"]
