"""Read-only fabric discovery: local RDMA inventory and LLDP-based cabling.

``local_inventory`` reads sysfs on the node it runs on, every port's GID table
included (``sparknet.topology.gids``). ``collect_lldp`` runs
one read-only script over SSH per host (interface MACs, operstate and LLDP
neighbour MACs) and ``resolve_links`` turns the answers into cables.
``generate_nodes`` writes a node map from the cables under the fleet's
addressing rule, ``10.<a><b>.<path>.<N>``: ``a < b`` are the cabled node
numbers, path 1 is the ``enp1s0*`` PCIe function and path 2 the ``enP2p1s0*``
one, ``N`` is the node number. Nothing here changes a host.
"""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
from pathlib import Path

from .gids import GidSelectionError, gid_ipv4, parse_gid_lines, read_gid_table, select_gid

# DGX Spark ConnectX-7 netdev -> (QSFP port, PCIe path). Port 0 cables to the
# next node in the ring, port 1 to the previous one.
INTERFACES = {
    "enp1s0f0np0": (0, 1),
    "enP2p1s0f0np0": (0, 2),
    "enp1s0f1np1": (1, 1),
    "enP2p1s0f1np1": (1, 2),
}

INTERFACES_BY_HCA = {
    "rocep1s0f0": (0, 1), "roceP2p1s0f0": (0, 2), "rocep1s0f1": (1, 1), "roceP2p1s0f1": (1, 2),
}

COLLECT = r"""
for i in enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1; do
  printf '%s %s %s ' "$i" "$(cat /sys/class/net/$i/address)" "$(cat /sys/class/net/$i/operstate)"
  sudo -n lldpctl -f keyvalue "$i" 2>/dev/null | sed -n 's/^lldp\.[^.]*\.port\.mac=//p' | tr '\n' ' '
  echo
done
"""


def hca_name(netdev: str) -> str:
    """``enp1s0f0np0`` -> ``rocep1s0f0``, ``enP2p1s0f1np1`` -> ``roceP2p1s0f1``."""
    if not netdev.startswith("en") or "np" not in netdev:
        raise ValueError(f"not a ConnectX-7 netdev name: {netdev}")
    return "roce" + netdev[2:].rsplit("np", 1)[0]


def node_number(host: str) -> int:
    match = re.fullmatch(r"[a-z]+(\d+)", host)
    if not match:
        raise ValueError(f"host name must end in its node number: {host}")
    return int(match.group(1))


def local_inventory(sysfs: str | Path = "/sys/class/infiniband") -> dict:
    """RDMA devices of this host: state, rate, firmware, netdev, MTU and the port's GID table."""
    root = Path(sysfs)
    devices = {}
    for dev in sorted(root.glob("*")) if root.exists() else []:
        port = dev / "ports" / "1"
        netdevs = sorted(p.name for p in (dev / "device" / "net").glob("*")) if (dev / "device" / "net").exists() else []
        entry = {
            "state": _read(port / "state"),
            "rate": _read(port / "rate"),
            "firmware": _read(dev / "fw_ver"),
            "netdev": netdevs[0] if netdevs else None,
            "gids": read_gid_table(port),
        }
        if entry["netdev"]:
            mtu = Path("/sys/class/net") / entry["netdev"] / "mtu"
            entry["mtu"] = int(_read(mtu) or 0) if mtu.exists() else None
        devices[dev.name] = entry
    return devices


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def collect_lldp(host: str, *, ssh_user: str | None = None) -> list[dict]:
    """Interface MAC, state and LLDP neighbour MACs of one host, over SSH."""
    target = f"{ssh_user}@{host}" if ssh_user else host
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", target, "bash", "-s"], input=COLLECT,
                         capture_output=True, text=True, check=True, timeout=60).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in INTERFACES:
            rows.append({"iface": parts[0], "mac": parts[1], "state": parts[2], "peer_macs": parts[3:]})
    return rows


def resolve_links(data: dict[str, list[dict]]) -> tuple[dict, list[str]]:
    """Map (host, iface) -> (peer host, peer iface) from per-host LLDP rows; report problems."""
    owner = {row["mac"]: (h, row["iface"]) for h, rows in data.items() for row in rows}
    links: dict[tuple[str, str], tuple[str, str]] = {}
    problems: list[str] = []
    for host, rows in data.items():
        for row in rows:
            peers = {owner[m] for m in row["peer_macs"] if m in owner and owner[m][0] != host}
            if row["state"] != "up":
                problems.append(f"{host} {row['iface']}: link {row['state']}")
                continue
            # Each port sees both PCIe functions of the far port; keep the one on the same path.
            same_path = [p for p in peers if INTERFACES[p[1]][1] == INTERFACES[row["iface"]][1]]
            if len(same_path) != 1:
                problems.append(f"{host} {row['iface']}: unresolved peer {sorted(peers)}")
                continue
            links[(host, row["iface"])] = same_path[0]
    for (host, iface), (peer, peer_iface) in links.items():
        if links.get((peer, peer_iface)) != (host, iface):
            problems.append(f"asymmetric: {host} {iface} -> {peer} {peer_iface}")
    return links, problems


def cable_subnet(a: int, b: int, path: int) -> str:
    """``10.<a><b>.<path>.0/24``; a second cable between the same pair uses paths 3 and 4."""
    lo, hi = sorted((a, b))
    return f"10.{lo}{hi}.{path}.0/24"


# One ``rail <dev> <state> <netdev> <mtu>`` line per device, then one
# ``<dev> <index> <type> <gid> <ndev>`` line per populated GID slot.
RAILS = r"""
for d in /sys/class/infiniband/*; do
  n=$(basename "$d"); p="$d/ports/1"; i=$(ls "$d/device/net" | head -1)
  printf 'rail %s %s %s %s\n' "$n" "$(tr -d ' ' < "$p/state")" "$i" "$(cat /sys/class/net/$i/mtu)"
  for g in "$p"/gids/*; do s=${g##*/}
    t=$(cat "$p/gid_attrs/types/$s" 2>/dev/null | tr -d ' '); [ -n "$t" ] || continue
    printf '%s %s %s %s %s\n' "$n" "$s" "$t" "$(cat "$g")" "$(cat "$p/gid_attrs/ndevs/$s" 2>/dev/null)"
  done
done
"""


def parse_rails(out: str) -> dict[str, dict]:
    """``RAILS`` output as {device: {state, netdev, mtu, gids}}."""
    rails = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "rail":
            rails[parts[1]] = {"state": parts[2], "netdev": parts[3], "mtu": int(parts[4]) if parts[4].isdigit() else None,
                               "gids": []}
    for device, table in parse_gid_lines(out).items():
        if device in rails:
            rails[device]["gids"] = table
    return rails


def collect_rails(host: str, *, ssh_user: str | None = None) -> dict[str, dict]:
    """A host's RDMA devices with state, netdev, MTU and GID table, over SSH (read-only)."""
    target = f"{ssh_user}@{host}" if ssh_user else host
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", target, "bash", "-s"], input=RAILS,
                         capture_output=True, text=True, check=True, timeout=60).stdout
    return parse_rails(out)


def generate_switched_nodes(
    rails: dict[str, dict[str, dict]], hosts: list[str], *, management_ips: dict[str, str], ssh_user: str,
    management_interface: str | None = None, traffic_class: int | None = None,
) -> tuple[dict, list[str]]:
    """A switched node map from each host's active rails; rail subnets come from the live addresses.

    A rail's address is its port's one IPv4 RoCE v2 GID, wherever it sits in the table.

    One-shot takes the first port's two functions (one QSFP port, two PCIe
    paths); NCCL takes every active rail. Rank order is host order; rank 0 is the head.
    """
    problems: list[str] = []
    nodes = []
    for rank, host in enumerate(hosts):
        active = {hca: entry for hca, entry in sorted(rails.get(host, {}).items()) if "ACTIVE" in (entry.get("state") or "")}
        subnets = {}
        for hca, entry in active.items():
            try:
                _, address = select_gid(entry.get("gids", []), None, f"{host} {hca}")
            except GidSelectionError as error:
                problems.append(str(error))
                continue
            subnets[hca] = str(ipaddress.ip_network(f"{address}/24", strict=False))
        ordered = [h for h in INTERFACES_BY_HCA if h in subnets] + [h for h in subnets if h not in INTERFACES_BY_HCA]
        roce = [h for h in ordered if INTERFACES_BY_HCA.get(h, (0, 0))[0] == 0][:2] or ordered[:2]
        if not roce:
            problems.append(f"{host}: no active rail with an IPv4 RoCE v2 GID")
            continue
        node = {"name": host, "rank": rank, "management_ip": management_ips[host], "head": rank == 0,
                "roce_hcas": roce, "nccl_hcas": ordered, "roce_subnets": subnets}
        if traffic_class is not None:
            node["roce_traffic_class"] = traffic_class
        if management_interface:
            node["management_interface"] = management_interface
        nodes.append(node)
    return {"schema_version": 1, "ssh_user": ssh_user, "nodes": nodes}, problems


def generate_nodes(
    links: dict, hosts: list[str], *, management_ips: dict[str, str], ssh_user: str,
    management_interface: str | None = None,
) -> dict:
    """A node map in cable order from resolved links; rank 0 is ``hosts[0]`` (the head)."""
    numbers = {h: node_number(h) for h in hosts}
    adjacency: dict[str, dict[str, list[tuple[str, int, str]]]] = {h: {} for h in hosts}
    for (host, iface), (peer, peer_iface) in links.items():
        adjacency[host].setdefault(peer, []).append((iface, INTERFACES[iface][1], peer_iface))
    order = _cable_order(adjacency, hosts)
    rank_of = {h: i for i, h in enumerate(order)}
    nodes = []
    for host in order:
        routes, subnets, extra = {}, {}, []
        for peer, ifaces in adjacency[host].items():
            # Path 1 then path 2. With two cables to the same peer, both ends keep the cable on
            # the lower-numbered node's port 0 for the one-shot stripes; the other cable is NCCL's.
            def primary_cable(item):
                iface, _path, peer_iface = item
                port = INTERFACES[iface][0] if numbers[host] < numbers[peer] else INTERFACES[peer_iface][0]
                return port != 0
            lanes = sorted(ifaces, key=lambda item: (primary_cable(item), item[1]))
            primary, second = lanes[:2], lanes[2:]
            routes[str(rank_of[peer])] = [hca_name(i) for i, _, _ in primary]
            for iface, path, _ in primary:
                subnets[hca_name(iface)] = cable_subnet(numbers[host], numbers[peer], path)
            for iface, path, _ in second:
                # A second cable between the same pair: the one-shot stripes keep one cable, NCCL may use both.
                subnets[hca_name(iface)] = cable_subnet(numbers[host], numbers[peer], path + 2)
                extra.append(hca_name(iface))
        node = {
            "name": host, "rank": rank_of[host], "management_ip": management_ips[host],
            "head": rank_of[host] == 0,
            "roce_peer_hcas": dict(sorted(routes.items(), key=lambda kv: int(kv[0]))),
            "roce_subnets": subnets,
        }
        if extra:
            node["nccl_hcas"] = [h for route in node["roce_peer_hcas"].values() for h in route] + extra
        if management_interface:
            node["management_interface"] = management_interface
        nodes.append(node)
    return {"schema_version": 1, "ssh_user": ssh_user, "nodes": nodes}


def _cable_order(adjacency: dict, hosts: list[str]) -> list[str]:
    """Walk port-0 cables from the head so ranks follow the loop; a full mesh keeps host order."""
    if all(len(adjacency[h]) == len(hosts) - 1 for h in hosts):
        return list(hosts)
    order = [hosts[0]]
    while len(order) < len(hosts):
        current = order[-1]
        nxt = None
        for peer, ifaces in adjacency[current].items():
            if peer not in order and any(INTERFACES[i][0] == 0 for i, _, _ in ifaces):
                nxt = peer
        if nxt is None:
            candidates = [p for p in adjacency[current] if p not in order]
            if not candidates:
                raise ValueError(f"cannot continue the cable loop after {current}")
            nxt = candidates[0]
        order.append(nxt)
    return order


def netplan_yaml(host: str, links: dict, numbers: dict[str, int], mtu: int = 9000) -> str:
    """The fleet's hand-written ``40-cx7.yaml`` shape for one host (no YAML dependency)."""
    lines = ["network:", "  version: 2", "  renderer: NetworkManager", "  ethernets:"]
    for iface in INTERFACES:
        if (host, iface) not in links:
            continue
        peer, _ = links[(host, iface)]
        path = INTERFACES[iface][1]
        a, b = sorted((numbers[host], numbers[peer]))
        net = f"10.{a}{b}.{path}"
        lines += [
            f"    {iface}:",
            "      addresses:",
            f"        - {net}.{numbers[host]}/24",
            "      dhcp4: false", "      dhcp6: false", "      accept-ra: false",
            "      link-local: []", f"      mtu: {mtu}", "      optional: true",
            "      networkmanager:",
            f"        name: \"cx7 dgx{a}-dgx{b} path{path}\"",
        ]
    return "\n".join(lines) + "\n"


def write_json(path: str | Path, document: dict) -> None:
    Path(path).write_text(json.dumps(document, indent=2) + "\n")


__all__ = [
    "COLLECT", "INTERFACES", "INTERFACES_BY_HCA", "RAILS", "cable_subnet", "collect_lldp", "collect_rails",
    "generate_nodes", "generate_switched_nodes", "gid_ipv4", "hca_name", "local_inventory", "netplan_yaml",
    "node_number", "parse_rails", "resolve_links", "write_json",
]
