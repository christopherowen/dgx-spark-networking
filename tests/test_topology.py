"""Topology mistakes must fail before contacting a node or opening a queue pair."""

import copy
import json
import unittest
from pathlib import Path

from sparknet.topology import discover, nodes as topology, render

EXAMPLES = Path(__file__).resolve().parents[1] / "sparknet" / "topology" / "examples"


def example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


class NodeMapTest(unittest.TestCase):
    def setUp(self):
        self.three = example("tp3-triangle")
        self.four = example("tp4-ring")

    def test_examples_validate_for_their_transports(self):
        self.assertEqual(topology.problems(self.three, "rocenante-direct"), [])
        for transport in ("rocenante-ring4", "nccl-ring", "rocenante-mesh4"):
            self.assertEqual(topology.problems(self.four, transport), [], transport)
        self.assertEqual(topology.problems(self.four, "rocenante-mesh4", mesh_paths=4), [])

    def test_four_nodes_cannot_use_direct_peer_transport(self):
        self.assertTrue(any("requires nccl-ring" in p for p in topology.problems(self.four, "rocenante-direct")))
        self.assertTrue(any("exactly four" in p for p in topology.problems(self.three, "rocenante-ring4")))
        self.assertTrue(any("unknown fabric transport" in p for p in topology.problems(self.four, "switched")))

    def test_ring_requires_neighbours_in_rank_order(self):
        node = self.four["nodes"][0]
        node["roce_peer_hcas"]["2"] = node["roce_peer_hcas"].pop("3")
        self.assertTrue(any("cable/rank order" in p for p in topology.problems(self.four, "nccl-ring")))

    def test_ring_rejects_reused_interface_and_asymmetric_stripes(self):
        self.four["nodes"][0]["roce_peer_hcas"]["1"] = ["rocep1s0f1"]
        errors = topology.problems(self.four, "rocenante-ring4")
        self.assertTrue(any("distinct local HCAs" in p for p in errors))
        self.assertTrue(any("reciprocal" in p for p in errors))

    def test_ring_rejects_missing_or_shared_cable_subnets(self):
        self.four["nodes"][0]["roce_subnets"].pop("rocep1s0f0")
        errors = topology.problems(self.four, "nccl-ring")
        self.assertTrue(any("roce_subnets" in p for p in errors))
        self.assertTrue(any("exactly the two endpoints" in p for p in errors))

    def test_relay_stripe_order_must_match_the_remote_cable(self):
        self.four["nodes"][0]["roce_peer_hcas"]["1"].reverse()
        self.assertTrue(any("same stripe position" in p for p in topology.problems(self.four, "rocenante-ring4")))
        # NCCL selects by subnet rather than route-list position.
        self.assertEqual(topology.problems(self.four, "nccl-ring"), [])

    def test_duplicate_ranks_missing_head_and_unknown_fields_rejected(self):
        self.four["nodes"][3]["rank"] = 2
        self.assertTrue(any("contiguous" in p for p in topology.problems(self.four, "nccl-ring")))
        self.three["nodes"][0]["head"] = False
        self.assertTrue(any("head node" in p for p in topology.problems(self.three, "rocenante-direct")))
        self.three["nodes"][1]["cable_colour"] = "blue"
        self.assertTrue(any("unknown fields" in p for p in topology.problems(self.three, "rocenante-direct")))

    def test_mixed_stripe_widths_rejected_for_rocenante(self):
        node = self.three["nodes"][0]
        node["roce_peer_hcas"]["1"] = ["rocep1s0f0"]
        self.three["nodes"][1]["roce_peer_hcas"]["0"] = ["rocep1s0f1"]
        errors = topology.problems(self.three, "rocenante-direct")
        self.assertTrue(any("equal stripe counts" in p or "common stripe count" in p for p in errors))

    def test_live_gid_matches_declared_cable_and_per_node_index(self):
        node = self.four["nodes"][3]
        node["roce_gid_index"] = 4
        lines = [f"{hca} 4 RoCEv2 ::ffff:{subnet.replace('.0/24', '.4')} en0" for hca, subnet in node["roce_subnets"].items()]
        output = "\n".join(lines)
        self.assertEqual(topology.gid_subnet_problems(node, 4, output), [])
        self.assertTrue(topology.gid_subnet_problems(node, 4, output.replace("10.14.1.4", "10.14.2.4")))
        self.assertTrue(topology.gid_subnet_problems(node, 3, output))
        sysfs = "rocep1s0f0 4 RoCEv2 0000:0000:0000:0000:0000:ffff:0a0e:0104 enp1s0f0np0"
        self.assertEqual(len(topology.gid_subnet_problems(node, 4, sysfs)), 3)

    def test_mesh_logical_routes_derive_the_opposite_path(self):
        for node in self.four["nodes"]:
            routes = topology.logical_peer_hcas(node, "rocenante-mesh4")
            opposite = str((node["rank"] + 2) % 4)
            self.assertEqual(len(routes[opposite]), 2)
            self.assertEqual({k: v for k, v in routes.items() if k != opposite}, node["roce_peer_hcas"])
            four = topology.logical_peer_hcas(node, "rocenante-mesh4", mesh_paths=4)
            self.assertEqual(len(four[opposite]), 4)
        self.assertEqual(topology.logical_peer_hcas(self.four["nodes"][0], "rocenante-ring4"), self.four["nodes"][0]["roce_peer_hcas"])


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.three = example("tp3-triangle")
        self.four = example("tp4-ring")

    def test_ring_environment_names_neighbours_only_and_exact_hcas(self):
        node = self.four["nodes"][3]
        env = render.node_environment(self.four, node, transport="rocenante-ring4")
        self.assertEqual(set(json.loads(env["SPARKNET_ROCE_PEER_HCAS"])), {"2", "0"})
        self.assertEqual(env["SPARKNET_ROCE_PEER_HCAS"], env["B12X_ROCE_PEER_HCAS"])
        self.assertEqual(env["SPARKNET_ROCE_TOPOLOGY"], "ring4")
        self.assertTrue(env["NCCL_IB_HCA"].startswith("="))
        self.assertEqual(env["NCCL_ALGO"], "Ring")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "enP7s7")
        self.assertEqual(env["SPARKNET_ROCE_GID_INDEX"], "3")
        self.assertEqual(render.environment_problems({**env, "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}, "rocenante-ring4", 4), [])

    def test_nccl_only_ring_drops_the_rocenante_routes(self):
        env = render.node_environment(self.four, self.four["nodes"][0], transport="nccl-ring",
                                      base={"B12X_ROCE_PEER_HCAS": "stale", "B12X_ROCE_TOPOLOGY": "ring4"})
        self.assertNotIn("SPARKNET_ROCE_PEER_HCAS", env)
        self.assertNotIn("B12X_ROCE_PEER_HCAS", env)
        self.assertNotIn("B12X_ROCE_TOPOLOGY", env)

    def test_triangle_environment_keeps_every_peer_and_upstream_nccl_selection(self):
        env = render.node_environment(self.three, self.three["nodes"][1], transport="rocenante-direct", compat=False)
        self.assertEqual(set(json.loads(env["SPARKNET_ROCE_PEER_HCAS"])), {"0", "2"})
        self.assertNotIn("B12X_ROCE_PEER_HCAS", env)
        self.assertNotIn("NCCL_ALGO", env)
        self.assertEqual(env["SPARKNET_ROCE_TOPOLOGY"], "direct")

    def test_ring_policy_rejects_each_unsafe_override(self):
        env = render.node_environment(self.four, self.four["nodes"][0], transport="nccl-ring",
                                      base={"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"})
        self.assertEqual(render.environment_problems(env, "nccl-ring", 4), [])
        for key in render.RING_ENV:
            with self.subTest(key=key):
                broken = dict(env, **{key: "invalid"})
                self.assertTrue(any(key in p for p in render.environment_problems(broken, "nccl-ring", 4)))
        for key in render.FORBIDDEN_RING_KEYS:
            self.assertTrue(any(key in p for p in render.environment_problems(dict(env, **{key: "x"}), "nccl-ring", 4)))
        for lower, upper in (("1", "4"), ("3", "3"), ("16", "16")):
            broken = dict(env, NCCL_MIN_NCHANNELS=lower, NCCL_MAX_NCHANNELS=upper)
            self.assertTrue(any("NCCL_MIN_NCHANNELS" in p for p in render.environment_problems(broken, "nccl-ring", 4)))


def _synthetic_ring(hosts):
    """LLDP rows for a cable loop: port 0 of each host to port 1 of the next, both PCIe paths."""
    mac = lambda host, iface: f"02:00:00:{hosts.index(host):02x}:{list(discover.INTERFACES).index(iface):02x}:00"
    data = {}
    for i, host in enumerate(hosts):
        nxt, prv = hosts[(i + 1) % len(hosts)], hosts[(i - 1) % len(hosts)]
        rows = []
        for iface, (port, _path) in discover.INTERFACES.items():
            peer = nxt if port == 0 else prv
            far = [f for f, (p, _) in discover.INTERFACES.items() if p == 1 - port]
            rows.append({"iface": iface, "mac": mac(host, iface), "state": "up", "peer_macs": [mac(peer, f) for f in far]})
        data[host] = rows
    return data


class DiscoverTest(unittest.TestCase):
    def test_hca_names_follow_the_spark_netdev_convention(self):
        self.assertEqual(discover.hca_name("enp1s0f0np0"), "rocep1s0f0")
        self.assertEqual(discover.hca_name("enP2p1s0f1np1"), "roceP2p1s0f1")
        self.assertEqual(discover.gid_ipv4("0000:0000:0000:0000:0000:ffff:0a0e:0104"), "10.14.1.4")
        self.assertIsNone(discover.gid_ipv4("fe80:0000:0000:0000:4ebb:47ff:feeb:1e0d"))

    def test_ring_discovery_generates_a_valid_map_matching_the_example(self):
        hosts = ["dgx1", "dgx2", "dgx3", "dgx4"]
        links, problems = discover.resolve_links(_synthetic_ring(hosts))
        self.assertEqual(problems, [])
        self.assertEqual(links[("dgx1", "enp1s0f0np0")], ("dgx2", "enp1s0f1np1"))
        self.assertEqual(links[("dgx4", "enP2p1s0f0np0")], ("dgx1", "enP2p1s0f1np1"))
        document = discover.generate_nodes(links, hosts, management_ips={h: f"192.0.2.{i + 1}" for i, h in enumerate(hosts)},
                                           ssh_user="spark", management_interface="enP7s7")
        self.assertEqual(topology.problems(document, "rocenante-ring4"), [])
        expected = example("tp4-ring")
        for got, want in zip(document["nodes"], expected["nodes"]):
            self.assertEqual(got["roce_peer_hcas"], want["roce_peer_hcas"], got["name"])
            self.assertEqual(got["roce_subnets"], want["roce_subnets"], got["name"])
        yaml = discover.netplan_yaml("dgx4", links, {h: discover.node_number(h) for h in hosts})
        self.assertIn("- 10.14.1.4/24", yaml)
        self.assertIn('name: "cx7 dgx3-dgx4 path2"', yaml)

    def test_asymmetric_or_down_links_are_reported(self):
        data = _synthetic_ring(["dgx1", "dgx2", "dgx3", "dgx4"])
        data["dgx2"][0]["state"] = "down"
        data["dgx3"][2]["peer_macs"] = []
        _, problems = discover.resolve_links(data)
        self.assertTrue(any("link down" in p for p in problems))
        self.assertTrue(any("unresolved peer" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
