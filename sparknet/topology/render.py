"""Per-node environment for a transport: the one-shot routes or rails, and the NCCL fabric policy."""

from __future__ import annotations

import json

from .nodes import (
    RING_TRANSPORTS,
    SWITCHED_TRANSPORTS,
    logical_peer_hcas,
    nccl_hcas,
    node_hcas,
    roce_topology,
    uses_oneshot,
)

# NCCL policy for a neighbour ring (four nodes in cable order). These are the
# settings the four-node qualification depends on; channel counts and buffers
# come from the tuning profile (``sparknet.nccl.profiles``). Direct and
# switched fabrics keep NCCL's own topology and algorithm selection.
RING_ENV = {
    "NCCL_ALGO": "Ring",
    # NCCL 2.30.7 init.cc gates runtimeConn on cuMemSupport. Without this,
    # communicator initialization eagerly connects non-neighbour trees/PAT.
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_CUMEM_HOST_ENABLE": "0",
    "NCCL_RUNTIME_CONNECT": "1",
    "NCCL_PAT_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_COLLNET_ENABLE": "0",
    "NCCL_MNNVL_ENABLE": "0",
    "NCCL_RMA_DISABLE": "1",
    "NCCL_GIN_ENABLE": "0",
    "NCCL_NET": "IB",
    "NCCL_NET_PLUGIN": "none",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_MERGE_NICS": "1",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_IB_SUBNET_PREFIX_LEN": "24",
    "NCCL_IB_ADDR_FAMILY": "AF_INET",
    "NCCL_IB_ROCE_VERSION_NUM": "2",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
}

# Keys that must never be set on a neighbour ring: they would let NCCL build
# routes through ranks that have no cable.
FORBIDDEN_RING_KEYS = ("NCCL_ALGO_PLUGIN", "NCCL_TUNER_PLUGIN", "NCCL_GRAPH_FILE", "NCCL_TOPO_FILE")

_ROCE_KEYS = ("PEER_HCAS", "TOPOLOGY", "HCA", "GID_INDEX", "TRAFFIC_CLASS")


def node_environment(
    nodes: dict,
    node: dict,
    *,
    transport: str,
    base: dict[str, str] | None = None,
    mesh_paths: int = 2,
) -> dict[str, str]:
    """The environment one rank needs for ``transport`` on top of ``base``.

    Emits the ``SPARKNET_ROCE_*`` settings. Stale routing keys in ``base`` are
    replaced, never merged.
    """
    env = {k: str(v) for k, v in (base or {}).items()}
    for key in list(env):
        if any(key == f"SPARKNET_ROCE_{name}" for name in _ROCE_KEYS):
            env.pop(key)

    def put(name: str, value: str) -> None:
        env[f"SPARKNET_ROCE_{name}"] = value

    if uses_oneshot(transport):
        put("TOPOLOGY", roce_topology(transport))
        if transport in SWITCHED_TRANSPORTS:
            # A clique over the switch: the same rails reach every peer.
            put("HCA", ",".join(node_hcas(node, transport)))
        else:
            put("PEER_HCAS", json.dumps(logical_peer_hcas(node, transport, mesh_paths), separators=(",", ":")))
    if transport in RING_TRANSPORTS:
        env.update(RING_ENV)
    # '=' makes NCCL match device names exactly rather than by prefix.
    env["NCCL_IB_HCA"] = "=" + ",".join(nccl_hcas(node, transport))
    if "roce_gid_index" in node:
        env["NCCL_IB_GID_INDEX"] = str(node["roce_gid_index"])
        put("GID_INDEX", str(node["roce_gid_index"]))
    if "roce_traffic_class" in node:
        env["NCCL_IB_TC"] = str(node["roce_traffic_class"])
        put("TRAFFIC_CLASS", str(node["roce_traffic_class"]))
    iface = node.get("management_interface")
    if iface:
        for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"):
            env[key] = iface
    return env


def environment_problems(env: dict[str, str], transport: str, node_count: int) -> list[str]:
    """Settings in ``env`` that contradict the transport's fabric policy."""
    errors = []
    if transport in RING_TRANSPORTS:
        for key, value in RING_ENV.items():
            if str(env.get(key)) != value:
                errors.append(f"{transport} requires {key}={value}")
        channels = [str(env.get(k)) for k in ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")]
        allowed = ("1", "2", "4", "8") if node_count == 4 else ("1",)
        if channels[0] not in allowed or channels[0] != channels[1]:
            errors.append(f"{transport} requires matching NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS in {', '.join(allowed)}")
        for key in FORBIDDEN_RING_KEYS:
            if env.get(key):
                errors.append(f"{transport} cannot override topology/algorithm through {key}")
    topology = roce_topology(transport)
    if "SPARKNET_ROCE_TOPOLOGY" in env and env["SPARKNET_ROCE_TOPOLOGY"] != topology:
        errors.append(f"SPARKNET_ROCE_TOPOLOGY must be {topology} for {transport}")
    if not uses_oneshot(transport):
        for key in ("SPARKNET_ROCE_PEER_HCAS", "SPARKNET_ROCE_HCA"):
            if key in env:
                errors.append(f"{transport} carries every collective on NCCL; {key} must not be set")
    return errors


__all__ = ["FORBIDDEN_RING_KEYS", "RING_ENV", "environment_problems", "node_environment"]
