"""RoCE GIDs by address: a port's source GID is found by what it is, not by its slot.

A ConnectX port's GID table holds the link-local default and, for each IPv4
address of its netdev, a RoCE v1 and a RoCE v2 entry. The kernel files a
re-added address in the lowest free slot, so a link flap while RDMA resources
still hold the old entry moves the IPv4 RoCE v2 GID (3 to 4 on the Spark ring).
The one-shot proxy therefore takes each HCA's source GID from the single IPv4
RoCE v2 entry in that HCA's subnet (``select_gid`` in ``_roce_proxy.c``,
through ``ibv_query_gid_ex``), and NCCL with ``NCCL_IB_GID_INDEX`` unset scans
the same table by ``NCCL_IB_ADDR_FAMILY`` and ``NCCL_IB_ROCE_VERSION_NUM``.
This module is the same rule over sysfs and ``show_gids``-style lines, for the
doctors, discovery and device auto-discovery. Nothing here changes a host.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

ANY_IPV4 = "0.0.0.0/0"


class GidSelectionError(ValueError):
    """A port without exactly one IPv4 RoCE v2 GID in its subnet."""


def gid_ipv4(gid: str | None) -> str | None:
    """The IPv4 address an IPv4-mapped GID encodes (sysfs or ``::ffff:a.b.c.d`` form), else None."""
    if not gid:
        return None
    try:
        mapped = ipaddress.IPv6Address(gid).ipv4_mapped
    except ipaddress.AddressValueError:
        return None
    return str(mapped) if mapped is not None else None


def is_roce_v2(kind: str | None) -> bool:
    """sysfs writes ``RoCE v2``; one-line queries drop the space; ``show_gids`` prints ``v2``."""
    return (kind or "").replace(" ", "").lower() in ("rocev2", "v2")


def read_gid_table(port: str | Path) -> list[dict]:
    """Every populated slot of one RDMA port (``/sys/class/infiniband/<dev>/ports/1``), by index.

    An empty slot reads as the all-zero GID and its type file cannot be read.
    """
    port = Path(port)
    gids = port / "gids"
    slots = sorted(int(p.name) for p in gids.glob("*") if p.name.isdigit()) if gids.is_dir() else []
    table = []
    for index in slots:
        gid = _read(port / "gids" / str(index))
        kind = _read(port / "gid_attrs" / "types" / str(index))
        if not gid or not kind or not gid.replace(":", "").strip("0"):
            continue
        table.append({"index": index, "type": kind, "gid": gid, "ndev": _read(port / "gid_attrs" / "ndevs" / str(index))})
    return table


def parse_gid_lines(text: str) -> dict[str, list[dict]]:
    """``<hca> <index> <type> <gid> [netdev]`` lines (type without spaces) as tables per HCA."""
    tables: dict[str, list[dict]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[1].isdigit():
            continue
        tables.setdefault(fields[0], []).append(
            {"index": int(fields[1]), "type": fields[2], "gid": fields[3], "ndev": fields[4] if len(fields) > 4 else None})
    return tables


def ipv4_roce_v2(table: list[dict]) -> list[tuple[int, str]]:
    """(index, address) of every IPv4 RoCE v2 entry, by index."""
    found = []
    for entry in sorted(table, key=lambda e: e["index"]):
        address = gid_ipv4(entry.get("gid")) if is_roce_v2(entry.get("type")) else None
        if address is not None:
            found.append((entry["index"], address))
    return found


def select_gid(table: list[dict], subnet: str | None, device: str) -> tuple[int, str]:
    """The one IPv4 RoCE v2 entry of ``device``'s table in ``subnet`` (any IPv4 address when None).

    Returns (index, address); raises ``GidSelectionError`` naming the device
    when there is none or more than one, as the proxy does.
    """
    network = ipaddress.IPv4Network(subnet or ANY_IPV4)
    candidates = ipv4_roce_v2(table)
    found = [(index, address) for index, address in candidates if ipaddress.IPv4Address(address) in network]
    if len(found) == 1:
        return found[0]
    if found:
        indices = ", ".join(str(index) for index, _ in found)
        raise GidSelectionError(f"{device} has {len(found)} IPv4 RoCE v2 GIDs in {network} (indices {indices}); "
                                "its port needs one address there")
    elsewhere = ", ".join(f"{address} at index {index}" for index, address in candidates)
    raise GidSelectionError(f"{device} has no IPv4 RoCE v2 GID in {network}"
                            + (f" (it has {elsewhere}); check the link's address and cable" if elsewhere else ""))


def port_gid_problems(device: str, table: list[dict], subnet: str | None) -> list[str]:
    """The doctors' rule: the port has exactly one IPv4 RoCE v2 GID, and it is in ``subnet``.

    The proxy needs one in the subnet. NCCL scans without a per-port range and
    takes the lowest-index IPv4 RoCE v2 entry, so any other one on the port
    could become NCCL's source address.
    """
    try:
        index, _ = select_gid(table, subnet, device)
    except GidSelectionError as error:
        return [str(error)]
    others = [f"{address} at index {i}" for i, address in ipv4_roce_v2(table) if i != index]
    if others:
        return [f"{device} also has an IPv4 RoCE v2 GID outside {subnet or ANY_IPV4} ({', '.join(others)}); "
                "NCCL takes the lowest-index one, so give the port only its cable's address"]
    return []


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


__all__ = ["ANY_IPV4", "GidSelectionError", "gid_ipv4", "ipv4_roce_v2", "is_roce_v2", "parse_gid_lines",
           "port_gid_problems", "read_gid_table", "select_gid"]
