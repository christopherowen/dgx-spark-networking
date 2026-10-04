"""RDMA error and physical-port counters; read before and after a timed case to attribute loss."""

from __future__ import annotations

import subprocess
from pathlib import Path

RDMA_ERROR_COUNTERS = ("roce_adp_retrans", "packet_seq_err", "out_of_sequence", "np_cnp_sent")
PORT_COUNTERS = ("tx_bytes_phy", "rx_bytes_phy", "rx_out_of_buffer",
                 "tx_vport_rdma_unicast_bytes", "rx_vport_rdma_unicast_bytes")


def rdma_error_counters(sysfs: str | Path = "/sys/class/infiniband") -> dict[str, dict[str, int]]:
    counters = {}
    for device in sorted(Path(sysfs).iterdir()) if Path(sysfs).exists() else []:
        values = {}
        for name in RDMA_ERROR_COUNTERS:
            path = device / "ports/1/hw_counters" / name
            if path.exists():
                values[name] = int(path.read_text())
        counters[device.name] = values
    if not counters:
        raise RuntimeError("RDMA counter sampling requested but no devices visible")
    return counters


def port_counters(sysfs: str | Path = "/sys/class/infiniband") -> dict[str, dict[str, int]]:
    """``ethtool -S`` physical and vport counters per RDMA netdev.

    The two PCIe functions of one QSFP port report the same physical-port
    counters; sum ``*_phy`` across functions as duplicates, never as wire bytes.
    """
    counters = {}
    for hca in sorted(Path(sysfs).iterdir()) if Path(sysfs).exists() else []:
        for netdev in (hca / "device/net").iterdir() if (hca / "device/net").exists() else []:
            output = subprocess.check_output(["ethtool", "-S", netdev.name], text=True, timeout=10)
            values = {key.strip(): int(value.strip()) for line in output.splitlines()
                      if ":" in line for key, value in [line.split(":", 1)] if value.strip().isdigit()}
            counters[netdev.name] = {name: values[name] for name in PORT_COUNTERS if name in values}
    if not counters:
        raise RuntimeError("port counter sampling requested but no RDMA netdevs visible")
    return counters


def deltas(before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {dev: {k: v - before.get(dev, {}).get(k, 0) for k, v in values.items()} for dev, values in after.items()}


__all__ = ["PORT_COUNTERS", "RDMA_ERROR_COUNTERS", "deltas", "port_counters", "rdma_error_counters"]
