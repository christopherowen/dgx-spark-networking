"""Node map schema and validation for the four Spark fabrics.

A node map (``nodes.json``) is the site's description of the fabric, not of a
model. For a cabled fabric (two nodes on one or two cables, three nodes in a
triangle, four nodes in a loop) each node lists which local RDMA devices
reach which peer rank (``roce_peer_hcas``) and the IPv4 ``/24`` of each cable
stripe (``roce_subnets``). For a switched fabric each node lists its rails
(``roce_hcas`` for one-shot, ``nccl_hcas`` for NCCL) and the rail subnets
shared by every node. Every check here runs before a node is contacted or a
queue pair is opened; it reports configuration mistakes, not whether a cable
or a switch actually delivers packets (the collective probe does that).
"""

from __future__ import annotations

import copy
import ipaddress
import json
from pathlib import Path

SCHEMA_VERSION = 1
TRANSPORTS = (
    "oneshot-direct",   # 2 or 3 nodes, every pair cabled; one-shot direct + NCCL
    "nccl-direct",        # 2 or 3 nodes, every pair cabled; NCCL only
    "oneshot-ring4",    # 4 nodes in a loop; one-shot host relay + NCCL ring
    "nccl-ring",          # 4 nodes in a loop; NCCL ring only
    "oneshot-mesh4",    # 4 nodes in a loop; one-shot over NIC forwarding + NCCL ring
    "oneshot-switched", # 2..16 nodes through a switch; one-shot direct + NCCL
    "nccl-switched",      # 2..16 nodes through a switch; NCCL only
)
RING_TRANSPORTS = ("nccl-ring", "oneshot-ring4", "oneshot-mesh4")
DIRECT_TRANSPORTS = ("oneshot-direct", "nccl-direct")
SWITCHED_TRANSPORTS = ("oneshot-switched", "nccl-switched")
CABLED_TRANSPORTS = DIRECT_TRANSPORTS + RING_TRANSPORTS
ONESHOT_TRANSPORTS = ("oneshot-direct", "oneshot-ring4", "oneshot-mesh4", "oneshot-switched")
MAX_SWITCHED_NODES = 16  # the proxy's ROCE_MAX_PEERS
_NODE_FIELDS = {"name", "rank", "management_ip", "head", "roce_peer_hcas", "roce_subnets", "roce_hcas", "nccl_hcas",
                "roce_gid_index", "roce_traffic_class", "management_interface", "mesh_ports", "mesh_hairpin_queue_size"}


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


def fabric_kind(transport: str, node_count: int) -> str:
    """``direct2``, ``triangle3``, ``ring4``, ``mesh4`` or ``switched``."""
    if transport in SWITCHED_TRANSPORTS:
        return "switched"
    if transport == "oneshot-mesh4":
        return "mesh4"
    if transport in RING_TRANSPORTS:
        return "ring4"
    return "direct2" if node_count == 2 else "triangle3"


def uses_oneshot(transport: str) -> bool:
    return transport in ONESHOT_TRANSPORTS


def roce_topology(transport: str) -> str:
    """The runtime routing mode (``SPARKNET_ROCE_TOPOLOGY``) for a transport."""
    return {"oneshot-ring4": "ring4", "oneshot-mesh4": "mesh4"}.get(transport, "direct")


def expected_peers(count: int, rank: int, transport: str) -> set[int]:
    """Ranks a node must have a physical route to under a cabled ``transport``."""
    if transport in RING_TRANSPORTS:
        return {(rank - 1) % count, (rank + 1) % count}
    return set(range(count)) - {rank}


def node_hcas(node: dict, transport: str) -> list[str]:
    """The local RDMA devices one-shot stripes over (switched: the rails; cabled: every routed device)."""
    if transport in SWITCHED_TRANSPORTS:
        return list(node.get("roce_hcas", []))
    return sorted({h for route in node.get("roce_peer_hcas", {}).values() if isinstance(route, list) for h in route})


def nccl_hcas(node: dict, transport: str) -> list[str]:
    """The devices NCCL may use: ``nccl_hcas`` when given (a second rail or cable), else the one-shot set."""
    return list(node.get("nccl_hcas") or node_hcas(node, transport))


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
    routes = copy.deepcopy(node.get("roce_peer_hcas", {}))
    if transport == "oneshot-mesh4":
        rank = node["rank"]
        specs = mesh_path_specs(rank, mesh_paths)
        routes[str((rank + 2) % 4)] = [routes[str(peer)][lane] for lane, peer in specs]
    return routes


def _hca_list(value, *, minimum: int, maximum: int) -> str | None:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        return f"needs {minimum} to {maximum} HCA names"
    if any(not isinstance(h, str) or not h for h in value):
        return "contains an empty HCA name"
    if len(set(value)) != len(value):
        return "repeats an HCA name"
    return None


def _subnet(raw) -> ipaddress.IPv4Network | None:
    try:
        net = ipaddress.IPv4Network(raw, strict=True)
        return net if net.prefixlen == 24 else None
    except (ValueError, TypeError, ipaddress.AddressValueError):
        return None


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
    if transport in SWITCHED_TRANSPORTS:
        if not 2 <= count <= MAX_SWITCHED_NODES:
            errors.append(f"a switched fabric takes 2 to {MAX_SWITCHED_NODES} nodes, got {count}")
    elif transport in RING_TRANSPORTS:
        if count != 4:
            errors.append(f"{transport} requires exactly four nodes in a cable loop, got {count}")
    elif count not in (2, 3):
        errors.append(f"{transport} requires two nodes on a cable or three in a triangle, got {count}"
                      + ("; four nodes without a switch use nccl-ring, oneshot-ring4 or oneshot-mesh4" if count == 4 else ""))
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
        gid = node.get("roce_gid_index", 3)
        if type(gid) is not int or gid < 0:
            errors.append(f"{node.get('name')}: roce_gid_index must be a nonnegative integer")
        tc = node.get("roce_traffic_class")
        if tc is not None and (type(tc) is not int or not 0 <= tc <= 255):
            errors.append(f"{node.get('name')}: roce_traffic_class must be an integer from 0 to 255")
        iface = node.get("management_interface")
        if iface is not None and (not isinstance(iface, str) or not iface):
            errors.append(f"{node.get('name')}: management_interface must be a nonempty string")
    heads = [n for n in entries if n.get("head")]
    if len(heads) != 1 or heads[0]["rank"] != 0:
        errors.append("exactly one head node is required, at rank 0")
    if mesh_paths not in (2, 4) or (mesh_paths == 4 and transport != "oneshot-mesh4"):
        errors.append("mesh_paths must be 2, or 4 for oneshot-mesh4")
    if transport in SWITCHED_TRANSPORTS:
        return errors + _switched_problems(entries, transport)
    return errors + _cabled_problems(by_rank, count, transport)


def _switched_problems(entries: list[dict], transport: str) -> list[str]:
    errors: list[str] = []
    rails: dict[int, set[str]] = {}
    widths: set[tuple[int, int]] = set()
    for node in entries:
        name = node.get("name")
        if "roce_peer_hcas" in node:
            errors.append(f"{name}: a switched fabric has no per-peer cables; use roce_hcas and nccl_hcas")
        roce = node.get("roce_hcas")
        problem = _hca_list(roce, minimum=1, maximum=2)
        if problem:
            errors.append(f"{name}: roce_hcas {problem} (one-shot stripes over one or two rails)")
            continue
        nccl = node.get("nccl_hcas", roce)
        problem = _hca_list(nccl, minimum=1, maximum=4)
        if problem:
            errors.append(f"{name}: nccl_hcas {problem}")
            continue
        if not set(roce) <= set(nccl):
            errors.append(f"{name}: nccl_hcas must include every roce_hcas rail")
        subnets = node.get("roce_subnets", {})
        for index, hca in enumerate(nccl):
            net = _subnet(subnets.get(hca))
            if net is None:
                errors.append(f"{name}: roce_subnets[{hca}] must name the rail's IPv4 /24 network")
                continue
            rails.setdefault(index, set()).add(str(net))
        widths.add((len(roce), len(nccl)))
    if len(widths) > 1:
        errors.append(f"every node must have the same rail counts, found {sorted(widths)}")
    for index, nets in sorted(rails.items()):
        if len(nets) != 1:
            errors.append(f"rail {index} must use one subnet on every node, found {sorted(nets)}")
    return errors


def _cabled_problems(by_rank: dict[int, dict], count: int, transport: str) -> list[str]:
    errors: list[str] = []
    networks: dict[str, list] = {}
    for rank, node in by_rank.items():
        name = node.get("name")
        if "roce_hcas" in node:
            errors.append(f"{name}: roce_hcas describes a switched rail; cabled fabrics use roce_peer_hcas")
        expected = expected_peers(count, rank, transport)
        routes = node.get("roce_peer_hcas")
        if not isinstance(routes, dict) or set(routes) != {str(p) for p in expected}:
            errors.append(f"{name}: roce_peer_hcas must name peers {sorted(expected)} in cable/rank order")
            continue
        if transport == "oneshot-mesh4" and any(len(v) != 2 for v in routes.values() if isinstance(v, list)):
            errors.append(f"{name}: oneshot-mesh4 requires two stripes per cable")
        all_hcas: list[str] = []
        for peer, hcas in routes.items():
            problem = _hca_list(hcas, minimum=1, maximum=2)
            if problem:
                errors.append(f"{name}: route to rank {peer} {problem} (one cable, one or two stripes)")
                continue
            all_hcas.extend(hcas)
            reverse = by_rank[int(peer)].get("roce_peer_hcas", {})
            back = reverse.get(str(rank)) if isinstance(reverse, dict) else None
            if not isinstance(back, list) or len(back) != len(hcas):
                errors.append(f"{name}: link to rank {peer} must have reciprocal stripe counts")
            if transport in RING_TRANSPORTS or "roce_subnets" in node:
                for lane, hca in enumerate(hcas):
                    net = _subnet(node.get("roce_subnets", {}).get(hca))
                    if net is None:
                        errors.append(f"{name}: roce_subnets[{hca}] must name its cable's IPv4 /24 network")
                        continue
                    networks.setdefault(str(net), []).append((rank, int(peer), hca, lane))
        if len(set(all_hcas)) != len(all_hcas):
            errors.append(f"{name}: switchless links must use distinct local HCAs")
        extra = node.get("nccl_hcas")
        if extra is not None:
            problem = _hca_list(extra, minimum=1, maximum=4)
            if problem:
                errors.append(f"{name}: nccl_hcas {problem}")
            elif not set(all_hcas) <= set(extra):
                errors.append(f"{name}: nccl_hcas must include every routed HCA")
            elif transport in RING_TRANSPORTS and set(extra) != set(all_hcas):
                errors.append(f"{name}: a neighbour ring has no spare rail; nccl_hcas must equal the routed HCAs")
            else:
                for hca in set(extra) - set(all_hcas):
                    if _subnet(node.get("roce_subnets", {}).get(hca)) is None:
                        errors.append(f"{name}: roce_subnets[{hca}] must name the second cable's IPv4 /24 network")
        if transport in ONESHOT_TRANSPORTS and len({len(v) for v in routes.values() if isinstance(v, list)}) != 1:
            errors.append(f"{name}: the one-shot runtime requires equal stripe counts for every peer")
    if transport in ONESHOT_TRANSPORTS:
        widths = {len(v) for n in by_rank.values() if isinstance(n.get("roce_peer_hcas"), dict)
                  for v in n["roce_peer_hcas"].values() if isinstance(v, list)}
        if len(widths) > 1:
            errors.append("the one-shot runtime requires one common stripe count across all ranks")
    for net, endpoints in networks.items():
        ranks_on = {r for r, _, _, _ in endpoints}
        if len(endpoints) != 2 or len(ranks_on) != 2:
            errors.append(f"subnet {net} must appear at exactly the two endpoints of one cable stripe, found {sorted(endpoints)}")
            continue
        (r1, p1, _, l1), (r2, p2, _, l2) = endpoints
        if p1 != r2 or p2 != r1:
            errors.append(f"subnet {net} joins ranks {r1} and {r2}, but their routes name other peers")
        if transport in ONESHOT_TRANSPORTS and l1 != l2:
            errors.append(f"subnet {net}: both endpoints must use the same stripe position (lane {l1} vs {l2})")
    return errors


def gid_subnet_problems(node: dict, gid_index: int, show_gids_output: str) -> list[str]:
    """Compare live ``show_gids``-style lines with the declared subnets.

    Each line is ``<hca> <index> <RoCE version> <gid or ::ffff:a.b.c.d> <netdev>``.
    The selected index must hold an IPv4 RoCE v2 GID inside the declared
    subnet for every HCA the map names.
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
                f"{node.get('name')}: {hca} GID {gid_index} is {gid}, expected the subnet {subnet}; "
                "check the link's address and GID index"
            )
    return problems


__all__ = [
    "CABLED_TRANSPORTS", "DIRECT_TRANSPORTS", "MAX_SWITCHED_NODES", "RING_TRANSPORTS", "ONESHOT_TRANSPORTS",
    "SCHEMA_VERSION", "SWITCHED_TRANSPORTS", "TRANSPORTS", "expected_peers", "fabric_kind", "gid_subnet_problems",
    "head_node", "load", "logical_peer_hcas", "mesh_path_specs", "nccl_hcas", "node_by_name", "node_by_rank",
    "node_hcas", "problems", "roce_topology", "uses_oneshot",
]
