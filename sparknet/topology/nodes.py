"""Node map schema and validation for three- and four-node switchless fabrics.

A node map (``nodes.json``) is the site's description of cables, not of a
model: name, rank in cable order, management address, which local RDMA
devices reach which peer rank (``roce_peer_hcas``) and which IPv4 ``/24``
each cable/stripe uses (``roce_subnets``). Every check here runs before a
node is contacted or a queue pair is opened; it reports configuration
mistakes, not whether a cable actually delivers packets (the collective probe
does that).
"""

from __future__ import annotations

import copy
import ipaddress
import json
from pathlib import Path

SCHEMA_VERSION = 1
TRANSPORTS = ("rocenante-direct", "nccl-ring", "rocenante-ring4", "rocenante-mesh4")
RING_TRANSPORTS = ("nccl-ring", "rocenante-ring4", "rocenante-mesh4")
ROCENANTE_TRANSPORTS = ("rocenante-direct", "rocenante-ring4", "rocenante-mesh4")
_NODE_FIELDS = {"name", "rank", "management_ip", "head", "roce_peer_hcas", "roce_subnets",
                "roce_gid_index", "management_interface", "mesh_ports", "mesh_hairpin_queue_size"}


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def node_by_name(nodes: dict, name: str) -> dict:
    for node in nodes.get("nodes", []):
        if node.get("name") == name:
            return node
    raise KeyError(f"node {name!r} is not in the map")


def node_by_rank(nodes: dict, rank: int) -> dict:
    for node in nodes.get("nodes", []):
        if node.get("rank") == rank:
            return node
    raise KeyError(f"rank {rank} is not in the map")


def head_node(nodes: dict) -> dict:
    heads = [n for n in nodes.get("nodes", []) if n.get("head")]
    if len(heads) != 1:
        raise ValueError("exactly one head node is required")
    return heads[0]


def roce_topology(transport: str) -> str:
    """The runtime routing mode (``SPARKNET_ROCE_TOPOLOGY``) for a transport."""
    return {"rocenante-direct": "direct", "rocenante-ring4": "ring4",
            "rocenante-mesh4": "mesh4", "nccl-ring": "direct"}[transport]


def expected_peers(count: int, rank: int, transport: str) -> set[int]:
    """Ranks a node must have a physical route to under ``transport``."""
    if transport in RING_TRANSPORTS:
        return {(rank - 1) % count, (rank + 1) % count}
    return set(range(count)) - {rank}


def mesh_path_specs(rank: int, paths: int = 2) -> list[tuple[int, int]]:
    """(interface stripe, intermediate rank) pairs in reciprocal QP path order, four-node mesh."""
    if paths not in (2, 4):
        raise ValueError("mesh_paths must be 2 or 4")
    opposite = (rank + 2) % 4
    via = ((rank + 1) % 4, (rank - 1) % 4) if rank < opposite else ((rank - 1) % 4, (rank + 1) % 4)
    specs = [(0, via[0]), (1, via[1])]
    if paths == 4:
        specs += [(0, via[1]), (1, via[0])]
    return specs


def logical_peer_hcas(node: dict, transport: str, mesh_paths: int = 2) -> dict[str, list[str]]:
    """Physical cable routes stay authoritative; mesh4 adds the derived opposite-rank path."""
    routes = copy.deepcopy(node["roce_peer_hcas"])
    if transport == "rocenante-mesh4":
        rank = node["rank"]
        specs = mesh_path_specs(rank, mesh_paths)
        routes[str((rank + 2) % 4)] = [routes[str(peer)][lane] for lane, peer in specs]
    return routes


def problems(nodes: dict, transport: str, *, mesh_paths: int = 2) -> list[str]:
    """Every configuration mistake the map and transport selection contain."""
    errors: list[str] = []
    if transport not in TRANSPORTS:
        return [f"unknown fabric transport {transport!r}; choose one of {', '.join(TRANSPORTS)}"]
    if nodes.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    entries = nodes.get("nodes", [])
    count = len(entries)
    ranks = [n.get("rank") for n in entries]
    if count not in (2, 3, 4):
        errors.append(f"switchless deployment requires 2, 3 or 4 nodes, got {count}")
    if any(type(r) is not int for r in ranks) or sorted(ranks) != list(range(count)):
        return errors + [f"node ranks must be contiguous 0..{count - 1}, got {ranks}"]
    by_rank = {n["rank"]: n for n in entries}
    for field in ("name", "management_ip"):
        values = [n.get(field) for n in entries]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != count:
            errors.append(f"nodes must have distinct nonempty {field} values")
    for node in entries:
        unknown = set(node) - _NODE_FIELDS
        if unknown:
            errors.append(f"{node.get('name')}: unknown fields {sorted(unknown)}")
    heads = [n for n in entries if n.get("head")]
    if len(heads) != 1 or heads[0]["rank"] != 0:
        errors.append("exactly one head node is required, at rank 0")
    if mesh_paths not in (2, 4) or (mesh_paths == 4 and transport != "rocenante-mesh4"):
        errors.append("mesh_paths must be 2, or 4 for rocenante-mesh4")
    if count == 4 and transport not in RING_TRANSPORTS:
        errors.append("a four-node switchless fabric requires nccl-ring, rocenante-ring4 or rocenante-mesh4")
    if transport in ("rocenante-ring4", "rocenante-mesh4") and count != 4:
        errors.append(f"{transport} requires exactly four nodes")
    networks: dict[str, list] = {}
    for rank, node in by_rank.items():
        expected = expected_peers(count, rank, transport)
        routes = node.get("roce_peer_hcas")
        if not isinstance(routes, dict) or set(routes) != {str(p) for p in expected}:
            errors.append(f"{node.get('name')}: roce_peer_hcas must name peers {sorted(expected)} in cable/rank order")
            continue
        if transport == "rocenante-mesh4" and any(len(v) != 2 for v in routes.values() if isinstance(v, list)):
            errors.append(f"{node['name']}: rocenante-mesh4 requires two stripes per cable")
        all_hcas: list[str] = []
        for peer, hcas in routes.items():
            if not isinstance(hcas, list) or len(hcas) not in (1, 2) or any(not isinstance(h, str) or not h for h in hcas):
                errors.append(f"{node['name']}: peer {peer} needs one or two HCA names")
                continue
            all_hcas.extend(hcas)
            reverse = by_rank[int(peer)].get("roce_peer_hcas", {})
            back = reverse.get(str(rank)) if isinstance(reverse, dict) else None
            if not isinstance(back, list) or len(back) != len(hcas):
                errors.append(f"{node['name']}: link to rank {peer} must have reciprocal stripe counts")
            if transport in RING_TRANSPORTS or "roce_subnets" in node:
                for lane, hca in enumerate(hcas):
                    raw = node.get("roce_subnets", {}).get(hca)
                    try:
                        net = ipaddress.IPv4Network(raw, strict=True)
                        if net.prefixlen != 24:
                            raise ValueError("expected /24")
                    except (ValueError, TypeError, ipaddress.AddressValueError):
                        errors.append(f"{node['name']}: roce_subnets[{hca}] must name its cable's IPv4 /24 network")
                        continue
                    networks.setdefault(str(net), []).append((rank, int(peer), hca, lane))
        if len(set(all_hcas)) != len(all_hcas):
            errors.append(f"{node['name']}: switchless links must use distinct local HCAs")
        if transport in ROCENANTE_TRANSPORTS and len({len(v) for v in routes.values() if isinstance(v, list)}) != 1:
            errors.append(f"{node['name']}: RoCEnante requires equal stripe counts for every peer")
        gid = node.get("roce_gid_index", 3)
        if type(gid) is not int or gid < 0:
            errors.append(f"{node['name']}: roce_gid_index must be a nonnegative integer")
        iface = node.get("management_interface")
        if iface is not None and (not isinstance(iface, str) or not iface):
            errors.append(f"{node['name']}: management_interface must be a nonempty string")
    if transport in ROCENANTE_TRANSPORTS:
        widths = {len(v) for n in entries if isinstance(n.get("roce_peer_hcas"), dict)
                  for v in n["roce_peer_hcas"].values() if isinstance(v, list)}
        if len(widths) > 1:
            errors.append("RoCEnante requires one common stripe count across all ranks")
    for net, endpoints in networks.items():
        ranks_on = {r for r, _, _, _ in endpoints}
        if len(endpoints) != 2 or len(ranks_on) != 2:
            errors.append(f"subnet {net} must appear at exactly the two endpoints of one cable stripe, found {sorted(endpoints)}")
            continue
        (r1, p1, _, l1), (r2, p2, _, l2) = endpoints
        if p1 != r2 or p2 != r1:
            errors.append(f"subnet {net} joins ranks {r1} and {r2}, but their routes name other peers")
        if transport in ROCENANTE_TRANSPORTS and l1 != l2:
            errors.append(f"subnet {net}: both endpoints must use the same stripe position (lane {l1} vs {l2})")
    return errors


def gid_subnet_problems(node: dict, gid_index: int, show_gids_output: str) -> list[str]:
    """Compare live ``show_gids``-style lines with the declared cable subnets.

    Each line is ``<hca> <index> <RoCE version> <gid or ::ffff:a.b.c.d> <netdev>``.
    The selected index must hold an IPv4 RoCE v2 GID inside the cable's
    declared subnet for every routed HCA.
    """
    problems: list[str] = []
    live: dict[tuple[str, int], str] = {}
    for line in show_gids_output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            live[(parts[0], int(parts[1]))] = parts[3]
        except ValueError:
            continue
    for hca, subnet in node.get("roce_subnets", {}).items():
        gid = live.get((hca, gid_index))
        if gid is None:
            problems.append(f"{node.get('name')}: {hca} has no GID at index {gid_index}")
            continue
        try:
            address = ipaddress.IPv6Address(gid).ipv4_mapped
        except ipaddress.AddressValueError:
            address = None
        if address is None or address not in ipaddress.IPv4Network(subnet):
            problems.append(
                f"{node.get('name')}: {hca} GID {gid_index} is {gid}, expected the cable subnet {subnet}; "
                "check the link's address and GID index"
            )
    return problems


__all__ = [
    "RING_TRANSPORTS", "ROCENANTE_TRANSPORTS", "SCHEMA_VERSION", "TRANSPORTS",
    "expected_peers", "gid_subnet_problems", "head_node", "load", "logical_peer_hcas",
    "mesh_path_specs", "node_by_name", "node_by_rank", "problems", "roce_topology",
]
