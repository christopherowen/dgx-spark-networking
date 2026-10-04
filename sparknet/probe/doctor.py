"""Local fabric doctor: the node map against this host, before any queue pair opens."""

from __future__ import annotations

import shutil
from pathlib import Path

from sparknet.topology.discover import gid_ipv4, local_inventory
from sparknet.topology.nodes import node_by_name, problems as map_problems


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
    gid_index = node.get("roce_gid_index", 3)
    inventory = local_inventory(gid_index=gid_index) if inventory is None else inventory
    routed = sorted({h for route in node["roce_peer_hcas"].values() for h in route})
    for hca in routed:
        entry = inventory.get(hca)
        if entry is None:
            errors.append(f"{node_name}: RDMA device {hca} is not present")
            continue
        if "ACTIVE" not in (entry.get("state") or ""):
            errors.append(f"{node_name}: {hca} port is {entry.get('state')!r}, not ACTIVE")
        address = gid_ipv4(entry.get("gid"))
        subnet = node.get("roce_subnets", {}).get(hca)
        if subnet:
            import ipaddress

            if address is None or ipaddress.IPv4Address(address) not in ipaddress.IPv4Network(subnet):
                errors.append(f"{node_name}: {hca} GID {gid_index} is {entry.get('gid')}, expected an IPv4 address in {subnet}")
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
    if transport.startswith("rocenante"):
        if not compiler_present:
            errors.append("no C compiler for the RoCEnante proxy (gcc, cc or clang)")
        if not verbs_header:
            errors.append("libibverbs development headers missing (/usr/include/infiniband/verbs.h)")
    return errors


__all__ = ["local_problems"]
