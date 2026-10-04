"""Named transport profiles: the NCCL and RoCEnante settings that were measured together.

Native environment names are the tuning API; a value of ``None`` means "omit
the variable, use the pinned implementation's default", never zero. Each
profile records where its numbers were measured (spark3-vllm-ds41f
experiments, see ``docs/nccl.md``) and which ``patches/nccl`` entries its
settings need: a control that the unpatched library ignores must never be
passed silently.
"""

from __future__ import annotations

import re
from typing import Any

from sparknet.topology.render import RING_ENV

# RoCE v2 over the ConnectX-7 functions, one GPU per node, no PCIe peers, no
# shared memory: the part of the fabric policy every profile shares.
COMMON_IB_ENV = {
    "NCCL_NET": "IB",
    "NCCL_NET_PLUGIN": "none",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_MERGE_NICS": "1",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_IB_ADDR_FAMILY": "AF_INET",
    "NCCL_IB_ROCE_VERSION_NUM": "2",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_CROSS_NIC": "1",
    "NCCL_IGNORE_CPU_AFFINITY": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_DEBUG": "WARN",
    "NCCL_DEBUG_SUBSYS": "INIT",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}

# Controls that only exist in the patched NCCL (patches/nccl). Passing them to
# an unpatched library would be ignored silently, so profiles declare them.
PATCH_CONTROLS = {
    "NCCL_SWITCHLESS_BIDIRECTIONAL": ("0002-bidirectional-switchless-rings.patch", "0003-balanced-channel-allocation.patch"),
    "NCCL_MIN_TRAFFIC_PER_CHANNEL": ("0003-balanced-channel-allocation.patch",),
}
# The adaptive thread patch has no control of its own; mode 2 activates it.
ADAPTIVE_THREADS_PATCH = "0004-adaptive-small-ring-threads.patch"
FENCE_PATCH = "0001-ib-cts-nreqs-acquire-fence.patch"

PROFILES: dict[str, dict[str, Any]] = {
    "tp3-triangle": {
        "transport": "rocenante-direct",
        "node_count": 3,
        "status": "promoted: spark3-vllm-ds41f baseline 2026-10-02-karmic-kraken-r5o-64k (TP3, every pair cabled)",
        "nccl": {
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_MIN_NCHANNELS": None,
            "NCCL_MAX_NCHANNELS": "8",
            "NCCL_BUFFSIZE": "1048576",
            "NCCL_LL128_BUFFSIZE": "262144",
            "NCCL_PROTO": "^LL128",
            "NCCL_SWITCHLESS_BIDIRECTIONAL": None,
            "NCCL_MIN_TRAFFIC_PER_CHANNEL": None,
            "NCCL_THREAD_THRESHOLDS": None,
        },
        "rocenante": {
            "ALLREDUCE_CAPACITY_BYTES": "2097152",
            "ALLREDUCE_DISPATCH_MAX_BYTES": None,
            "ALLGATHER_MAX_BYTES": "4194304",
            "SPIN_LIMIT": "5000000",
        },
        "evidence": "spark3 manifests/baselines/2026-10-02-karmic-kraken-r5o-64k.json; experiments/2026-09-27-nccl-fence",
    },
    "tp4-ring": {
        "transport": "rocenante-ring4",
        "node_count": 4,
        "status": "measured balanced candidate: spark3 experiments/2026-10-03-balanced-policy selected.json (TP4, cable loop, bidirectional relay, four balanced NCCL channels)",
        "nccl": {
            **RING_ENV,
            "NCCL_MIN_NCHANNELS": "4",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "4194304",
            "NCCL_LL128_BUFFSIZE": "262144",
            "NCCL_PROTO": "^LL128",
            "NCCL_SWITCHLESS_BIDIRECTIONAL": "2",
            "NCCL_MIN_TRAFFIC_PER_CHANNEL": "512",
            "NCCL_THREAD_THRESHOLDS": "-2 -2 -2 1 1 1",
        },
        "rocenante": {
            "ALLREDUCE_CAPACITY_BYTES": "2097152",
            "ALLREDUCE_DISPATCH_MAX_BYTES": "1048576",
            "ALLGATHER_MAX_BYTES": "2097152",
            "SPIN_LIMIT": "5000000",
        },
        "evidence": "spark3 experiments/2026-10-03-balanced-policy (decision.md), 2026-10-03-collective-serving, 2026-10-03-tp3-tp4-comparison",
    },
    "tp4-ring-nccl-only": {
        "transport": "nccl-ring",
        "node_count": 4,
        "status": "control: four-node neighbour ring with every collective on NCCL (spark3 experiments/2026-10-03-collective-serving one-channel and four-channel arms)",
        "nccl": {
            **RING_ENV,
            "NCCL_MIN_NCHANNELS": "4",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "1048576",
            "NCCL_LL128_BUFFSIZE": "262144",
            "NCCL_PROTO": "^LL128",
            "NCCL_SWITCHLESS_BIDIRECTIONAL": None,
            "NCCL_MIN_TRAFFIC_PER_CHANNEL": None,
            "NCCL_THREAD_THRESHOLDS": None,
        },
        "rocenante": {
            "ALLREDUCE_CAPACITY_BYTES": None,
            "ALLREDUCE_DISPATCH_MAX_BYTES": None,
            "ALLGATHER_MAX_BYTES": None,
            "SPIN_LIMIT": None,
        },
        "evidence": "spark3 experiments/2026-10-03-collective-serving/decision.md",
    },
}


def size_bytes(value: str) -> int:
    """``2MB`` -> bytes; KB/MB/GB are binary units, as the pinned vLLM parses them."""
    match = re.fullmatch(r"([1-9][0-9]*)(B|KB|MB|GB)?", str(value))
    if not match:
        raise ValueError(f"invalid positive byte size: {value!r}")
    return int(match[1]) * {None: 1, "B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}[match[2]]


def profile(name: str) -> dict[str, Any]:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown transport profile {name!r}; choose {', '.join(PROFILES)}") from None


def environment(name: str, *, compat: bool = True) -> dict[str, str]:
    """The rank-invariant environment of a profile (per-node routes come from ``sparknet.topology``)."""
    entry = profile(name)
    env = dict(COMMON_IB_ENV)
    for key, value in entry["nccl"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    roce = entry["rocenante"]
    if roce["ALLREDUCE_CAPACITY_BYTES"] is not None:
        names = {
            "ALLREDUCE_CAPACITY_BYTES": "ALLREDUCE_CAPACITY_BYTES",
            "ALLREDUCE_DISPATCH_MAX_BYTES": "ALLREDUCE_DISPATCH_MAX_BYTES",
            "ALLGATHER_MAX_BYTES": "ALLGATHER_MAX_BYTES",
            "SPIN_LIMIT": "SPIN_LIMIT",
        }
        for key, value in roce.items():
            if value is None:
                continue
            env[f"SPARKNET_ROCE_{names[key]}"] = value
            if compat and key in ("ALLREDUCE_DISPATCH_MAX_BYTES", "SPIN_LIMIT"):
                env[f"B12X_ROCE_{key}"] = value
        if compat:
            # The vLLM adapter reads its limits in its own size syntax.
            env["VLLM_ROCE_ALLREDUCE_MAX_SIZE"] = _vllm_size(roce["ALLREDUCE_CAPACITY_BYTES"])
            env["VLLM_ROCE_ALLGATHER_MAX_SIZE"] = _vllm_size(roce["ALLGATHER_MAX_BYTES"])
            env["VLLM_ENABLE_ROCE_ALLREDUCE"] = "1"
            env["VLLM_ENABLE_PCIE_ALLREDUCE"] = "0"
    elif compat:
        env["VLLM_ENABLE_ROCE_ALLREDUCE"] = "0"
        env["VLLM_ENABLE_PCIE_ALLREDUCE"] = "0"
        env["VLLM_DISABLE_PYNCCL"] = "0"
    return env


def _vllm_size(value: str) -> str:
    n = int(value)
    return f"{n // (1024 ** 2)}MB" if n % (1024 ** 2) == 0 else f"{n}B"


def required_patches(env: dict[str, str]) -> list[str]:
    """The ``patches/nccl`` entries an environment needs beyond the fence fix."""
    needed = [FENCE_PATCH]
    for key, patches in PATCH_CONTROLS.items():
        if env.get(key):
            needed.extend(p for p in patches if p not in needed)
    if env.get("NCCL_SWITCHLESS_BIDIRECTIONAL") == "2" and ADAPTIVE_THREADS_PATCH not in needed:
        needed.append(ADAPTIVE_THREADS_PATCH)
    return needed


def problems(env: dict[str, str], *, node_count: int, patched_nccl: bool = True) -> list[str]:
    """Settings that contradict each other, the node count or the library build."""
    errors: list[str] = []
    for key in ("NCCL_BUFFSIZE", "NCCL_LL128_BUFFSIZE", "NCCL_MIN_TRAFFIC_PER_CHANNEL",
                "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
                "SPARKNET_ROCE_SPIN_LIMIT", "B12X_ROCE_SPIN_LIMIT"):
        if key in env and (not str(env[key]).isdigit() or int(env[key]) <= 0):
            errors.append(f"{key} must be a positive integer")
    if "NCCL_MAX_NCHANNELS" in env and env["NCCL_MAX_NCHANNELS"].isdigit():
        if int(env.get("NCCL_MIN_NCHANNELS", "1") or 1) > int(env["NCCL_MAX_NCHANNELS"]):
            errors.append("NCCL minimum channels exceeds maximum")
    if "NCCL_THREAD_THRESHOLDS" in env and not re.fullmatch(r"-?\d+( -?\d+){5}", env["NCCL_THREAD_THRESHOLDS"]):
        errors.append("NCCL_THREAD_THRESHOLDS requires six space-separated integers")
    if "NCCL_PROTO" in env and env["NCCL_PROTO"] != "^LL128":
        errors.append("these profiles retain NCCL_PROTO=^LL128 (LL128 is not qualified on the Spark fabric)")
    bidirectional = env.get("NCCL_SWITCHLESS_BIDIRECTIONAL")
    if bidirectional is not None:
        if bidirectional not in ("1", "2"):
            errors.append("NCCL_SWITCHLESS_BIDIRECTIONAL must be 1 or 2")
        channels = env.get("NCCL_MAX_NCHANNELS", "")
        if channels.isdigit() and (int(channels) < 2 or int(channels) % 2):
            errors.append("NCCL_SWITCHLESS_BIDIRECTIONAL needs an even channel count of at least 2")
        if bidirectional == "2" and channels.isdigit() and int(channels) != 2 and int(channels) % 4:
            errors.append("NCCL_SWITCHLESS_BIDIRECTIONAL=2 needs two or a multiple of four channels")
        if node_count not in (3, 4):
            errors.append("NCCL_SWITCHLESS_BIDIRECTIONAL needs three or four nodes")
    traffic = env.get("NCCL_MIN_TRAFFIC_PER_CHANNEL")
    if traffic is not None and traffic.isdigit() and (int(traffic) < 16 or int(traffic) > 32768 or int(traffic) % 16):
        errors.append("NCCL_MIN_TRAFFIC_PER_CHANNEL must be a multiple of 16 in [16, 32768]")
    if not patched_nccl:
        for key in PATCH_CONTROLS:
            if env.get(key):
                errors.append(f"{key} needs the patched NCCL ({', '.join(PATCH_CONTROLS[key])})")
    capacity = env.get("SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES") or (
        str(size_bytes(env["VLLM_ROCE_ALLREDUCE_MAX_SIZE"])) if env.get("VLLM_ROCE_ALLREDUCE_MAX_SIZE") else None)
    dispatch = env.get("SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES") or env.get("B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES")
    gather = env.get("SPARKNET_ROCE_ALLGATHER_MAX_BYTES") or (
        str(size_bytes(env["VLLM_ROCE_ALLGATHER_MAX_SIZE"])) if env.get("VLLM_ROCE_ALLGATHER_MAX_SIZE") else None)
    for label, value in (("all-reduce capacity", capacity), ("all-gather limit", gather), ("all-reduce dispatch", dispatch)):
        if value is not None and (not str(value).isdigit() or int(value) <= 0 or int(value) % 16):
            errors.append(f"RoCEnante {label} must be a positive multiple of 16 bytes")
    if capacity and dispatch and str(capacity).isdigit() and str(dispatch).isdigit() and int(dispatch) > int(capacity):
        errors.append("RoCEnante dispatch limit must not exceed registered capacity")
    return errors


__all__ = [
    "ADAPTIVE_THREADS_PATCH", "COMMON_IB_ENV", "FENCE_PATCH", "PATCH_CONTROLS", "PROFILES",
    "environment", "problems", "profile", "required_patches", "size_bytes",
]
