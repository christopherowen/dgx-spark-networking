"""Local fabric doctor: the node map against this host, before any queue pair opens."""

from __future__ import annotations

import shutil
from pathlib import Path

from sparknet.topology.discover import local_inventory
from sparknet.topology.gids import port_gid_problems
from sparknet.topology.nodes import nccl_hcas, node_by_name, node_hcas, problems as map_problems, uses_oneshot


def local_problems(
    nodes: dict,
    node_name: str,
    *,
    transport: str,
    inventory: dict | None = None,
    cmdline: str | None = None,
    memlock_unlimited: bool | None = None,
    compiler_present: bool | None = None,
    verbs_header: bool | None = None,
) -> list[str]:
    """Configuration mistakes plus live-link mismatches on ``node_name``.

    Keyword overrides let tests supply what sysfs would say. Everything is
    read-only; this never changes an interface, a route or a driver setting.
    """
    errors = list(map_problems(nodes, transport))
    try:
        node = node_by_name(nodes, node_name)
    except KeyError as exc:
        return errors + [str(exc)]
    inventory = local_inventory() if inventory is None else inventory
    routed = sorted(set(node_hcas(node, transport)) | set(nccl_hcas(node, transport)))
    for hca in routed:
        entry = inventory.get(hca)
        if entry is None:
            errors.append(f"{node_name}: RDMA device {hca} is not present")
            continue
        if "ACTIVE" not in (entry.get("state") or ""):
            errors.append(f"{node_name}: {hca} port is {entry.get('state')!r}, not ACTIVE")
        # One IPv4 RoCE v2 GID in the cable's subnet (any IPv4 address when the map
        # declares none), at whatever index: what the proxy and NCCL will select.
        for problem in port_gid_problems(hca, entry.get("gids", []), node.get("roce_subnets", {}).get(hca)):
            errors.append(f"{node_name}: {problem}")
        mtu = entry.get("mtu")
        if mtu is not None and mtu != 9000:
            errors.append(f"{node_name}: {hca} netdev MTU is {mtu}, the fabric uses 9000")
    cmdline = Path("/proc/cmdline").read_text() if cmdline is None and Path("/proc/cmdline").exists() else (cmdline or "")
    if cmdline and "kho=off" not in cmdline.split() and "nvidia" in cmdline:
        errors.append("kernel command line lacks kho=off: ibv_reg_mr can fail with ENOMEM under memory pressure")
    if memlock_unlimited is None:
        try:
            import resource

            memlock_unlimited = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] == resource.RLIM_INFINITY
        except (ImportError, ValueError, OSError):
            memlock_unlimited = None
    if memlock_unlimited is False:
        errors.append("locked-memory limit is not unlimited; RDMA registration of the pinned region can fail")
    if compiler_present is None:
        compiler_present = any(shutil.which(c) for c in ("gcc", "cc", "clang"))
    if verbs_header is None:
        verbs_header = Path("/usr/include/infiniband/verbs.h").exists()
    if uses_oneshot(transport):
        if not compiler_present:
            errors.append("no C compiler for the one-shot proxy (gcc, cc or clang)")
        if not verbs_header:
            errors.append("libibverbs development headers missing (/usr/include/infiniband/verbs.h)")
    return errors


__all__ = ["local_problems"]
