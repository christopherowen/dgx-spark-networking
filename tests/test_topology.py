"""Topology mistakes must fail before contacting a node or opening a queue pair."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from sparknet.topology import discover, gids, nodes as topology, render

EXAMPLES = Path(__file__).resolve().parents[1] / "sparknet" / "topology" / "examples"


def example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


class NodeMapTest(unittest.TestCase):
    def setUp(self):
        self.two = example("tp2-direct")
        self.three = example("tp3-triangle")
        self.four = example("tp4-ring")
        self.switched = example("switched")

    def test_examples_validate_for_their_transports(self):
        for transport in ("oneshot-direct", "nccl-direct"):
            self.assertEqual(topology.problems(self.two, transport), [], transport)
            self.assertEqual(topology.problems(self.three, transport), [], transport)
        for transport in ("oneshot-ring4", "nccl-ring", "oneshot-mesh4"):
            self.assertEqual(topology.problems(self.four, transport), [], transport)
        self.assertEqual(topology.problems(self.four, "oneshot-mesh4", mesh_paths=4), [])
        for transport in ("oneshot-switched", "nccl-switched"):
            self.assertEqual(topology.problems(self.switched, transport), [], transport)

    def test_fabric_kinds_and_wrong_shapes(self):
        self.assertEqual(topology.fabric_kind("oneshot-direct", 2), "direct2")
        self.assertEqual(topology.fabric_kind("nccl-direct", 3), "triangle3")
        self.assertEqual(topology.fabric_kind("nccl-ring", 4), "ring4")
        self.assertEqual(topology.fabric_kind("oneshot-mesh4", 4), "mesh4")
        self.assertEqual(topology.fabric_kind("nccl-switched", 8), "switched")
        self.assertTrue(any("nccl-ring, oneshot-ring4 or oneshot-mesh4" in p for p in topology.problems(self.four, "oneshot-direct")))
        self.assertTrue(any("exactly four" in p for p in topology.problems(self.three, "oneshot-ring4")))
        self.assertTrue(any("unknown fabric transport" in p for p in topology.problems(self.four, "switched")))
        self.assertTrue(any("switched fabric has no per-peer cables" in p for p in topology.problems(self.four, "oneshot-switched")))
        self.assertTrue(any("describes a switched rail" in p for p in topology.problems(self.switched, "nccl-ring")))
        big = copy.deepcopy(self.switched)
        for extra in range(4, 17):
            node = copy.deepcopy(big["nodes"][0])
            node.update(name=f"dgx{extra + 1}", rank=extra, management_ip=f"192.0.2.{extra + 1}", head=False)
            big["nodes"].append(node)
        self.assertTrue(any("2 to 16 nodes" in p for p in topology.problems(big, "oneshot-switched")))
        del big["nodes"][-1]
        self.assertEqual(topology.problems(big, "oneshot-switched"), [])

    def test_two_nodes_second_cable_goes_to_nccl(self):
        node = self.two["nodes"][0]
        self.assertEqual(topology.node_hcas(node, "oneshot-direct"), ["roceP2p1s0f0", "rocep1s0f0"])
        self.assertEqual(len(topology.nccl_hcas(node, "oneshot-direct")), 4)
        node["nccl_hcas"] = ["rocep1s0f1"]
        self.assertTrue(any("must include every routed HCA" in p for p in topology.problems(self.two, "oneshot-direct")))
        node["nccl_hcas"] = ["rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"]
        node["roce_subnets"].pop("rocep1s0f1")
        self.assertTrue(any("second cable" in p for p in topology.problems(self.two, "oneshot-direct")))
        ring = example("tp4-ring")
        ring["nodes"][0]["nccl_hcas"] = ring["nodes"][0].get("nccl_hcas", ["rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"])
        self.assertEqual(topology.problems(ring, "nccl-ring"), [])
        ring["nodes"][0]["nccl_hcas"] = ["rocep1s0f0", "roceP2p1s0f0"]
        self.assertTrue(any("must include every routed HCA" in p for p in topology.problems(ring, "nccl-ring")))

    def test_subset_carves_a_pair_out_of_a_ring(self):
        ring = example("tp4-ring")
        pair = topology.subset(ring, ["dgx2", "dgx3"])
        self.assertEqual(topology.problems(pair, "oneshot-direct"), [])
        self.assertEqual(topology.problems(pair, "nccl-direct"), [])
        self.assertEqual([n["name"] for n in pair["nodes"]], ["dgx2", "dgx3"])
        self.assertEqual([n["rank"] for n in pair["nodes"]], [0, 1])
        self.assertTrue(pair["nodes"][0]["head"] and not pair["nodes"][1]["head"])
        self.assertEqual(pair["nodes"][0]["roce_peer_hcas"], {"1": ring["nodes"][1]["roce_peer_hcas"]["2"]})
        self.assertEqual(pair["nodes"][1]["roce_peer_hcas"], {"0": ring["nodes"][2]["roce_peer_hcas"]["1"]})
        self.assertEqual(set(pair["nodes"][0]["roce_subnets"].values()), {"10.23.1.0/24", "10.23.2.0/24"})
        self.assertNotIn("nccl_hcas", pair["nodes"][0])
        self.assertEqual(pair["ssh_user"], ring["ssh_user"])
        self.assertEqual(pair["nodes"][0]["management_ip"], ring["nodes"][1]["management_ip"])
        env = render.node_environment(pair, pair["nodes"][0], transport="oneshot-direct")
        self.assertEqual(env["NCCL_IB_HCA"], "=roceP2p1s0f0,rocep1s0f0")  # exact names, sorted
        with self.assertRaisesRegex(ValueError, "share no cable"):
            topology.subset(ring, ["dgx1", "dgx3"])
        with self.assertRaisesRegex(ValueError, "two or three"):
            topology.subset(ring, ["dgx1"])
        with self.assertRaises(KeyError):
            topology.subset(ring, ["dgx1", "dgx9"])

    def test_subset_keeps_a_triangle_and_a_second_cable(self):
        triangle = example("tp3-triangle")
        three = topology.subset(triangle, ["dgx3", "dgx1", "dgx2"])
        self.assertEqual(topology.problems(three, "oneshot-direct"), [])
        self.assertEqual([n["name"] for n in three["nodes"]], ["dgx3", "dgx1", "dgx2"])
        self.assertEqual(set(three["nodes"][1]["roce_peer_hcas"]), {"0", "2"})
        two = topology.subset(example("tp2-direct"), ["dgx1", "dgx2"])
        self.assertEqual(topology.problems(two, "oneshot-direct"), [])
        self.assertEqual(two["nodes"][0]["nccl_hcas"], example("tp2-direct")["nodes"][0]["nccl_hcas"])
        self.assertEqual(len(two["nodes"][0]["roce_subnets"]), 4)
        with self.assertRaisesRegex(ValueError, "cabled map"):
            topology.subset(example("switched"), ["dgx1", "dgx2"])

    def test_switched_rails_must_agree_across_nodes(self):
        self.switched["nodes"][2]["roce_subnets"]["rocep1s0f0"] = "10.200.1.0/24"
        self.assertTrue(any("rail 0 must use one subnet" in p for p in topology.problems(self.switched, "oneshot-switched")))
        fresh = example("switched")
        fresh["nodes"][1]["nccl_hcas"] = ["rocep1s0f0", "roceP2p1s0f0"]
        self.assertTrue(any("same rail counts" in p for p in topology.problems(fresh, "nccl-switched")))
        fresh = example("switched")
        fresh["nodes"][0]["roce_hcas"] = ["rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1"]
        self.assertTrue(any("one or two rails" in p for p in topology.problems(fresh, "oneshot-switched")))
        fresh = example("switched")
        fresh["nodes"][0]["nccl_hcas"] = ["rocep1s0f1"]
        self.assertTrue(any("must include every roce_hcas rail" in p for p in topology.problems(fresh, "oneshot-switched")))
        fresh = example("switched")
        fresh["nodes"][3]["roce_traffic_class"] = 300
        self.assertTrue(any("roce_traffic_class" in p for p in topology.problems(fresh, "oneshot-switched")))

    def test_ring_requires_neighbours_in_rank_order(self):
        node = self.four["nodes"][0]
        node["roce_peer_hcas"]["2"] = node["roce_peer_hcas"].pop("3")
        self.assertTrue(any("cable/rank order" in p for p in topology.problems(self.four, "nccl-ring")))

    def test_ring_rejects_reused_interface_and_asymmetric_stripes(self):
        self.four["nodes"][0]["roce_peer_hcas"]["1"] = ["rocep1s0f1"]
        errors = topology.problems(self.four, "oneshot-ring4")
        self.assertTrue(any("distinct local HCAs" in p for p in errors))
        self.assertTrue(any("reciprocal" in p for p in errors))

    def test_ring_rejects_missing_or_shared_cable_subnets(self):
        self.four["nodes"][0]["roce_subnets"].pop("rocep1s0f0")
        errors = topology.problems(self.four, "nccl-ring")
        self.assertTrue(any("roce_subnets" in p for p in errors))
        self.assertTrue(any("exactly the two endpoints" in p for p in errors))

    def test_relay_stripe_order_must_match_the_remote_cable(self):
        self.four["nodes"][0]["roce_peer_hcas"]["1"].reverse()
        self.assertTrue(any("same stripe position" in p for p in topology.problems(self.four, "oneshot-ring4")))
        # NCCL selects by subnet rather than route-list position.
        self.assertEqual(topology.problems(self.four, "nccl-ring"), [])

    def test_duplicate_ranks_missing_head_and_unknown_fields_rejected(self):
        self.four["nodes"][3]["rank"] = 2
        self.assertTrue(any("contiguous" in p for p in topology.problems(self.four, "nccl-ring")))
        self.three["nodes"][0]["head"] = False
        self.assertTrue(any("head node" in p for p in topology.problems(self.three, "oneshot-direct")))
        self.three["nodes"][1]["cable_colour"] = "blue"
        self.assertTrue(any("unknown fields" in p for p in topology.problems(self.three, "oneshot-direct")))

    def test_mixed_stripe_widths_rejected_for_oneshot(self):
        node = self.three["nodes"][0]
        node["roce_peer_hcas"]["1"] = ["rocep1s0f0"]
        self.three["nodes"][1]["roce_peer_hcas"]["0"] = ["rocep1s0f1"]
        errors = topology.problems(self.three, "oneshot-direct")
        self.assertTrue(any("equal stripe counts" in p or "common stripe count" in p for p in errors))

    def test_live_gid_matches_declared_cable_at_any_index(self):
        node = self.four["nodes"][3]
        lines = [f"{hca} 4 RoCEv2 ::ffff:{subnet.replace('.0/24', '.4')} en0" for hca, subnet in node["roce_subnets"].items()]
        output = "\n".join(lines)
        self.assertEqual(topology.gid_subnet_problems(node, output), [])
        [problem] = topology.gid_subnet_problems(node, output.replace("10.14.1.4", "10.14.2.4"))
        self.assertIn("dgx4: rocep1s0f0 has no IPv4 RoCE v2 GID in 10.14.1.0/24 (it has 10.14.2.4 at index 4)", problem)
        sysfs = "rocep1s0f0 4 RoCEv2 0000:0000:0000:0000:0000:ffff:0a0e:0104 enp1s0f0np0"
        self.assertEqual(len(topology.gid_subnet_problems(node, sysfs)), 3)
        [problem] = topology.gid_subnet_problems(node, output + "\nrocep1s0f0 6 RoCEv2 ::ffff:10.14.1.9 en0")
        self.assertIn("rocep1s0f0 has 2 IPv4 RoCE v2 GIDs in 10.14.1.0/24 (indices 4, 6)", problem)

    def test_retired_gid_index_is_refused_with_its_reason(self):
        self.four["nodes"][0]["roce_gid_index"] = 5
        errors = topology.problems(self.four, "oneshot-ring4")
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("dgx1: roce_gid_index is no longer read", errors[0])
        self.assertIn("delete the field", errors[0])

    def test_mesh_logical_routes_derive_the_opposite_path(self):
        for node in self.four["nodes"]:
            routes = topology.logical_peer_hcas(node, "oneshot-mesh4")
            opposite = str((node["rank"] + 2) % 4)
            self.assertEqual(len(routes[opposite]), 2)
            self.assertEqual({k: v for k, v in routes.items() if k != opposite}, node["roce_peer_hcas"])
            four = topology.logical_peer_hcas(node, "oneshot-mesh4", mesh_paths=4)
            self.assertEqual(len(four[opposite]), 4)
        self.assertEqual(topology.logical_peer_hcas(self.four["nodes"][0], "oneshot-ring4"), self.four["nodes"][0]["roce_peer_hcas"])


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.three = example("tp3-triangle")
        self.four = example("tp4-ring")

    def test_ring_environment_names_neighbours_only_and_exact_hcas(self):
        node = self.four["nodes"][3]
        env = render.node_environment(self.four, node, transport="oneshot-ring4")
        self.assertEqual(set(json.loads(env["SPARKNET_ROCE_PEER_HCAS"])), {"2", "0"})
        self.assertFalse([k for k in env if k.startswith(("B12X_", "VLLM_"))])
        self.assertEqual(env["SPARKNET_ROCE_TOPOLOGY"], "ring4")
        self.assertTrue(env["NCCL_IB_HCA"].startswith("="))
        self.assertEqual(env["NCCL_ALGO"], "Ring")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "enP7s7")
        # Each one-shot HCA's cable subnet, and no GID slot for either library.
        self.assertEqual(json.loads(env["SPARKNET_ROCE_SUBNETS"]), node["roce_subnets"])
        self.assertFalse([k for k in env if "GID_INDEX" in k])
        self.assertEqual(render.environment_problems({**env, "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4"}, "oneshot-ring4", 4), [])

    def test_a_fixed_gid_slot_is_refused_for_both_libraries(self):
        node = self.four["nodes"][0]
        env = render.node_environment(self.four, node, transport="oneshot-ring4",
                                      base={"NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4", "SPARKNET_ROCE_GID_INDEX": "3"})
        self.assertNotIn("SPARKNET_ROCE_GID_INDEX", env)  # a stale base value is dropped
        [problem] = render.environment_problems(dict(env, NCCL_IB_GID_INDEX="3"), "oneshot-ring4", 4)
        self.assertIn("NCCL_IB_GID_INDEX pins NCCL to one GID slot", problem)
        [problem] = render.environment_problems(dict(env, SPARKNET_ROCE_GID_INDEX="3"), "oneshot-ring4", 4)
        self.assertIn("SPARKNET_ROCE_GID_INDEX is no longer read", problem)

    def test_nccl_only_ring_drops_the_oneshot_routes(self):
        env = render.node_environment(self.four, self.four["nodes"][0], transport="nccl-ring",
                                      base={"SPARKNET_ROCE_PEER_HCAS": "stale", "SPARKNET_ROCE_TOPOLOGY": "ring4"})
        self.assertNotIn("SPARKNET_ROCE_PEER_HCAS", env)
        self.assertNotIn("SPARKNET_ROCE_TOPOLOGY", env)

    def test_triangle_environment_keeps_every_peer_and_upstream_nccl_selection(self):
        env = render.node_environment(self.three, self.three["nodes"][1], transport="oneshot-direct")
        self.assertEqual(set(json.loads(env["SPARKNET_ROCE_PEER_HCAS"])), {"0", "2"})
        self.assertNotIn("NCCL_ALGO", env)
        self.assertEqual(env["SPARKNET_ROCE_TOPOLOGY"], "direct")
        self.assertEqual(env["NCCL_IB_HCA"], "=roceP2p1s0f0,roceP2p1s0f1,rocep1s0f0,rocep1s0f1")

    def test_two_node_and_switched_environments(self):
        two = example("tp2-direct")
        env = render.node_environment(two, two["nodes"][1], transport="oneshot-direct")
        self.assertEqual(json.loads(env["SPARKNET_ROCE_PEER_HCAS"]), {"0": ["rocep1s0f1", "roceP2p1s0f1"]})
        self.assertEqual(env["NCCL_IB_HCA"], "=rocep1s0f1,roceP2p1s0f1,rocep1s0f0,roceP2p1s0f0")
        env = render.node_environment(two, two["nodes"][1], transport="nccl-direct", base={"SPARKNET_ROCE_PEER_HCAS": "stale"})
        self.assertNotIn("SPARKNET_ROCE_PEER_HCAS", env)
        self.assertNotIn("SPARKNET_ROCE_TOPOLOGY", env)
        self.assertEqual(render.environment_problems(env, "nccl-direct", 2), [])
        self.assertTrue(render.environment_problems(dict(env, SPARKNET_ROCE_PEER_HCAS="{}"), "nccl-direct", 2))
        switched = example("switched")
        env = render.node_environment(switched, switched["nodes"][3], transport="oneshot-switched")
        self.assertEqual(env["SPARKNET_ROCE_HCA"], "rocep1s0f0,roceP2p1s0f0")
        # Only the one-shot rails' subnets: the proxy opens no other device.
        self.assertEqual(set(json.loads(env["SPARKNET_ROCE_SUBNETS"])), {"rocep1s0f0", "roceP2p1s0f0"})
        self.assertNotIn("SPARKNET_ROCE_PEER_HCAS", env)
        self.assertEqual(env["SPARKNET_ROCE_TOPOLOGY"], "direct")
        self.assertEqual(env["NCCL_IB_HCA"], "=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1")
        self.assertEqual((env["NCCL_IB_TC"], env["SPARKNET_ROCE_TRAFFIC_CLASS"]), ("106", "106"))
        self.assertNotIn("NCCL_ALGO", env)
        self.assertEqual(render.environment_problems(env, "oneshot-switched", 4), [])
        env = render.node_environment(switched, switched["nodes"][0], transport="nccl-switched")
        self.assertNotIn("SPARKNET_ROCE_HCA", env)
        self.assertNotIn("SPARKNET_ROCE_SUBNETS", env)
        self.assertTrue(render.environment_problems(dict(env, SPARKNET_ROCE_HCA="x"), "nccl-switched", 4))

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

    def mac(host, iface):
        return f"02:00:00:{hosts.index(host):02x}:{list(discover.INTERFACES).index(iface):02x}:00"

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
        self.assertEqual(topology.problems(document, "oneshot-ring4"), [])
        expected = example("tp4-ring")
        for got, want in zip(document["nodes"], expected["nodes"], strict=True):
            self.assertEqual(got["roce_peer_hcas"], want["roce_peer_hcas"], got["name"])
            self.assertEqual(got["roce_subnets"], want["roce_subnets"], got["name"])
        yaml = discover.netplan_yaml("dgx4", links, {h: discover.node_number(h) for h in hosts})
        self.assertIn("- 10.14.1.4/24", yaml)
        self.assertIn('name: "cx7 dgx3-dgx4 path2"', yaml)

    def test_two_nodes_with_one_or_two_cables(self):
        one = _synthetic_ring(["dgx1", "dgx2"])
        # One cable: port 0 of dgx1 to port 1 of dgx2; drop the port-1 rows of dgx1 and port-0 rows of dgx2.
        one["dgx1"] = [r for r in one["dgx1"] if discover.INTERFACES[r["iface"]][0] == 0]
        one["dgx2"] = [r for r in one["dgx2"] if discover.INTERFACES[r["iface"]][0] == 1]
        links, problems = discover.resolve_links(one)
        self.assertEqual(problems, [])
        document = discover.generate_nodes(links, ["dgx1", "dgx2"], management_ips={"dgx1": "192.0.2.1", "dgx2": "192.0.2.2"}, ssh_user="spark")
        self.assertEqual(topology.problems(document, "oneshot-direct"), [])
        self.assertEqual(document["nodes"][0]["roce_peer_hcas"], {"1": ["rocep1s0f0", "roceP2p1s0f0"]})
        self.assertNotIn("nccl_hcas", document["nodes"][0])
        # Two cables (both ports): the one-shot stripes keep the lower node's port-0 cable, NCCL gets all four functions.
        links, problems = discover.resolve_links(_synthetic_ring(["dgx1", "dgx2"]))
        self.assertEqual(problems, [])
        document = discover.generate_nodes(links, ["dgx1", "dgx2"], management_ips={"dgx1": "192.0.2.1", "dgx2": "192.0.2.2"}, ssh_user="spark")
        self.assertEqual(topology.problems(document, "oneshot-direct"), [])
        expected = example("tp2-direct")
        for got, want in zip(document["nodes"], expected["nodes"], strict=True):
            self.assertEqual(got["roce_peer_hcas"], want["roce_peer_hcas"], got["name"])
            self.assertEqual(got["roce_subnets"], want["roce_subnets"], got["name"])
            self.assertEqual(got["nccl_hcas"], want["nccl_hcas"], got["name"])

    def test_switched_map_from_rails(self):
        def table(address, slot):
            raw = "".join(f"{int(o):02x}" for o in address.split("."))
            return [{"index": 1, "type": "RoCE v2", "gid": "fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b"},
                    {"index": slot, "type": "RoCE v2", "gid": "0000:0000:0000:0000:0000:ffff:" + raw[:4] + ":" + raw[4:]}]
        rails = {}
        for n, host in enumerate(["dgx1", "dgx2", "dgx3"], start=1):
            # dgx2's GIDs sit one slot higher, as after a flap: the subnets still come from the addresses.
            rails[host] = {hca: {"state": "4: ACTIVE", "gids": table(f"10.{100 + port}.{path}.{n}", 3 + (n == 2)),
                                 "netdev": "x", "mtu": 9000}
                           for hca, (port, path) in discover.INTERFACES_BY_HCA.items()}
        rails["dgx3"]["roceP2p1s0f1"]["state"] = "1: DOWN"
        document, problems = discover.generate_switched_nodes(rails, ["dgx1", "dgx2", "dgx3"], management_ips={h: f"192.0.2.{i}" for i, h in enumerate(["dgx1", "dgx2", "dgx3"], start=1)},
                                                              ssh_user="spark", traffic_class=106)
        self.assertEqual(problems, [])
        self.assertEqual(document["nodes"][0]["roce_hcas"], ["rocep1s0f0", "roceP2p1s0f0"])
        self.assertEqual(document["nodes"][0]["roce_subnets"]["rocep1s0f1"], "10.101.1.0/24")
        self.assertEqual(document["nodes"][0]["roce_traffic_class"], 106)
        self.assertEqual(document["nodes"][1]["roce_subnets"]["rocep1s0f1"], "10.101.1.0/24")
        self.assertNotIn("roce_gid_index", document["nodes"][0])
        # dgx3's fourth rail is down, so the rail counts differ and validation says so.
        self.assertTrue(any("same rail counts" in p for p in topology.problems(document, "oneshot-switched")))
        rails["dgx3"]["roceP2p1s0f1"]["state"] = "4: ACTIVE"
        document, _ = discover.generate_switched_nodes(rails, ["dgx1", "dgx2", "dgx3"], management_ips={h: f"192.0.2.{i}" for i, h in enumerate(["dgx1", "dgx2", "dgx3"], start=1)}, ssh_user="spark")
        self.assertEqual(topology.problems(document, "oneshot-switched"), [])
        self.assertEqual(topology.problems(document, "nccl-switched"), [])

    def test_rails_script_output_parses_into_gid_tables(self):
        out = ("rail rocep1s0f0 4:ACTIVE enp1s0f0np0 9000\n"
               "rocep1s0f0 0 IB/RoCEv1 fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b enp1s0f0np0\n"
               "rocep1s0f0 4 RoCEv2 0000:0000:0000:0000:0000:ffff:0a65:0101 enp1s0f0np0\n"
               "rail roceP2p1s0f0 1:DOWN enP2p1s0f0np0 9000\n")
        rails = discover.parse_rails(out)
        self.assertEqual(rails["rocep1s0f0"]["mtu"], 9000)
        self.assertEqual(gids.select_gid(rails["rocep1s0f0"]["gids"], None, "rocep1s0f0"), (4, "10.101.1.1"))
        self.assertEqual(rails["roceP2p1s0f0"]["gids"], [])
        document, problems = discover.generate_switched_nodes({"dgx1": rails}, ["dgx1"], management_ips={"dgx1": "192.0.2.1"},
                                                              ssh_user="spark")
        self.assertEqual(problems, [])
        self.assertEqual(document["nodes"][0]["roce_subnets"], {"rocep1s0f0": "10.101.1.0/24"})

    def test_asymmetric_or_down_links_are_reported(self):
        data = _synthetic_ring(["dgx1", "dgx2", "dgx3", "dgx4"])
        data["dgx2"][0]["state"] = "down"
        data["dgx3"][2]["peer_macs"] = []
        _, problems = discover.resolve_links(data)
        self.assertTrue(any("link down" in p for p in problems))
        self.assertTrue(any("unresolved peer" in p for p in problems))


def _port(root: Path, slots: dict[int, tuple[str, str]]) -> Path:
    """A sysfs port directory: every slot of a 16-entry table, populated ones with a type and netdev."""
    port = root / "rocep1s0f0" / "ports" / "1"
    for name in ("gids", "gid_attrs/types", "gid_attrs/ndevs"):
        (port / name).mkdir(parents=True)
    for index in range(16):
        kind, gid = slots.get(index, (None, "0000:0000:0000:0000:0000:0000:0000:0000"))
        (port / "gids" / str(index)).write_text(gid + "\n")
        if kind:  # the kernel's type file of an empty slot cannot be read
            (port / "gid_attrs" / "types" / str(index)).write_text(kind + "\n")
            (port / "gid_attrs" / "ndevs" / str(index)).write_text("enp1s0f0np0\n")
    (port / "state").write_text("4: ACTIVE\n")
    return port


LOCAL = "fe80:0000:0000:0000:4ebb:47ff:fee9:7f2b"
HERE = "0000:0000:0000:0000:0000:ffff:0a0e:0104"  # 10.14.1.4


class GidTableTest(unittest.TestCase):
    """The rule the proxy applies through ibv_query_gid_ex, over sysfs."""

    def test_the_gid_is_found_where_the_kernel_filed_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            healthy = {0: ("IB/RoCE v1", LOCAL), 1: ("RoCE v2", LOCAL), 2: ("IB/RoCE v1", HERE), 3: ("RoCE v2", HERE)}
            table = gids.read_gid_table(_port(Path(tmp) / "a", healthy))
            self.assertEqual([e["index"] for e in table], [0, 1, 2, 3])
            self.assertEqual(table[3]["ndev"], "enp1s0f0np0")
            self.assertEqual(gids.select_gid(table, "10.14.1.0/24", "rocep1s0f0"), (3, "10.14.1.4"))
            # After a flap under live RDMA: slot 3 empty, the address re-added at 4.
            moved = {k: v for k, v in healthy.items() if k != 3} | {4: ("RoCE v2", HERE)}
            table = gids.read_gid_table(_port(Path(tmp) / "b", moved))
            self.assertEqual(gids.select_gid(table, "10.14.1.0/24", "rocep1s0f0"), (4, "10.14.1.4"))
            self.assertEqual(gids.select_gid(table, None, "rocep1s0f0"), (4, "10.14.1.4"))
            self.assertEqual(gids.port_gid_problems("rocep1s0f0", table, "10.14.1.0/24"), [])
            info = discover.local_inventory(Path(tmp) / "b")
            self.assertEqual(info["rocep1s0f0"]["gids"], table)

    def test_none_or_two_fail_naming_the_device(self):
        v2 = [{"index": 1, "type": "RoCE v2", "gid": LOCAL}, {"index": 2, "type": "IB/RoCE v1", "gid": HERE}]
        with self.assertRaisesRegex(gids.GidSelectionError, r"^roceP2p1s0f1 has no IPv4 RoCE v2 GID in 10.14.1.0/24$"):
            gids.select_gid(v2, "10.14.1.0/24", "roceP2p1s0f1")
        other = v2 + [{"index": 3, "type": "RoCE v2", "gid": "::ffff:10.12.1.4"}]
        with self.assertRaisesRegex(gids.GidSelectionError, r"\(it has 10.12.1.4 at index 3\); check the link's address"):
            gids.select_gid(other, "10.14.1.0/24", "roceP2p1s0f1")
        two = other + [{"index": 5, "type": "RoCEv2", "gid": "::ffff:10.14.1.4"}, {"index": 6, "type": "v2", "gid": "::ffff:10.14.1.5"}]
        with self.assertRaisesRegex(gids.GidSelectionError, r"has 2 IPv4 RoCE v2 GIDs in 10.14.1.0/24 \(indices 5, 6\)"):
            gids.select_gid(two, "10.14.1.0/24", "roceP2p1s0f1")
        # The subnet filters; the doctors' rule also refuses an address NCCL could take instead.
        one = other + [{"index": 5, "type": "RoCE v2", "gid": "::ffff:10.14.1.4"}]
        self.assertEqual(gids.select_gid(one, "10.14.1.0/24", "roceP2p1s0f1"), (5, "10.14.1.4"))
        [problem] = gids.port_gid_problems("roceP2p1s0f1", one, "10.14.1.0/24")
        self.assertIn("also has an IPv4 RoCE v2 GID outside 10.14.1.0/24 (10.12.1.4 at index 3)", problem)
        with self.assertRaisesRegex(gids.GidSelectionError, r"has 2 IPv4 RoCE v2 GIDs in 0.0.0.0/0"):
            gids.select_gid(one, None, "roceP2p1s0f1")

    def test_ipv4_mapped_forms(self):
        self.assertEqual(gids.gid_ipv4(HERE), "10.14.1.4")
        self.assertEqual(gids.gid_ipv4("::ffff:10.14.1.4"), "10.14.1.4")
        self.assertIsNone(gids.gid_ipv4(LOCAL))
        self.assertIsNone(gids.gid_ipv4("not a gid"))
        self.assertIs(discover.gid_ipv4, gids.gid_ipv4)


if __name__ == "__main__":
    unittest.main()
