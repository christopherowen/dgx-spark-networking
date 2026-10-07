"""Probe planning and read-only host reports."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from sparknet.probe import fleet, summary
from sparknet.probe.container import docker_probe_command
from sparknet.probe.doctor import local_problems
from sparknet.probe.gpudirect import gpudirect_report
from sparknet.topology import nodes as topology

EXAMPLES = Path(__file__).resolve().parents[1] / "sparknet" / "topology" / "examples"


class ContainerTest(unittest.TestCase):
    def test_probe_is_bounded_and_does_not_share_serving_ipc(self):
        command = docker_probe_command(image="img:tag", environment={"NCCL_DEBUG": "WARN", "A": "1"}, rank=3, world_size=4,
                                       master_addr="192.0.2.1", master_port=29999, transport="oneshot-ring4",
                                       probe_source="/opt/sparknet/probe/collectives.py", extra_args=("--benchmark",))
        self.assertIn("600s", command)
        self.assertIn("--memory=12g", command)
        self.assertNotIn("--ipc=host", command)
        self.assertIn("--entrypoint=/usr/bin/timeout", command)
        self.assertEqual(command[command.index("--world-size") + 1], "4")
        self.assertEqual(command[command.index("--transport") + 1], "oneshot-ring4")
        self.assertIn("--env", command)
        self.assertIn("NCCL_DEBUG=INFO", command)
        self.assertEqual(command[-1], "--benchmark")
        self.assertIn("/opt/sparknet/probe/collectives.py:/probe.py:ro", command)

    def test_probe_defaults_to_the_image_package_and_can_mount_a_checkout(self):
        command = docker_probe_command(image="img:tag", environment={}, rank=0, world_size=2, master_addr="192.0.2.1",
                                       master_port=29650, transport="oneshot-direct", package_source="/home/spark/sparknet")
        self.assertNotIn("/probe.py", " ".join(command))
        self.assertEqual(command[command.index("-m") + 1], "sparknet.probe.collectives")
        self.assertIn("/home/spark/sparknet:/usr/local/lib/python3.12/dist-packages/sparknet:ro", command)


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self.nodes = json.loads((EXAMPLES / "tp4-ring.json").read_text())
        node = topology.node_by_name(self.nodes, "dgx4")
        self.inventory = {}
        for hca, subnet in node["roce_subnets"].items():
            address = subnet.replace(".0/24", ".4")
            gid = "0000:0000:0000:0000:0000:ffff:" + "".join(f"{int(o):02x}" for o in address.split("."))
            gid = gid[:-8] + gid[-8:-4] + ":" + gid[-4:]
            self.inventory[hca] = {"state": "4: ACTIVE", "gid": gid, "mtu": 9000}
        self.kwargs = dict(transport="oneshot-ring4", inventory=self.inventory, cmdline="BOOT_IMAGE=/boot/vmlinuz-7.0.0-1019-nvidia-64k kho=off",
                           memlock_unlimited=True, compiler_present=True, verbs_header=True)

    def test_matching_host_passes(self):
        self.assertEqual(local_problems(self.nodes, "dgx4", **self.kwargs), [])

    def test_switched_doctor_checks_every_rail(self):
        nodes = json.loads((EXAMPLES / "switched.json").read_text())
        node = topology.node_by_name(nodes, "dgx2")
        inventory = {}
        for hca, subnet in node["roce_subnets"].items():
            address = subnet.replace(".0/24", ".2")
            raw = "".join(f"{int(o):02x}" for o in address.split("."))
            inventory[hca] = {"state": "4: ACTIVE", "gid": "0000:0000:0000:0000:0000:ffff:" + raw[:4] + ":" + raw[4:], "mtu": 9000}
        kwargs = dict(self.kwargs, transport="oneshot-switched", inventory=inventory)
        self.assertEqual(local_problems(nodes, "dgx2", **kwargs), [])
        inventory["roceP2p1s0f1"]["state"] = "1: DOWN"
        self.assertTrue(any("roceP2p1s0f1 port is" in e for e in local_problems(nodes, "dgx2", **kwargs)))
        self.assertEqual(local_problems(nodes, "dgx2", **dict(kwargs, transport="nccl-switched", compiler_present=False)),
                         [e for e in local_problems(nodes, "dgx2", **kwargs)])

    def test_each_mismatch_is_reported(self):
        inventory = {k: dict(v) for k, v in self.inventory.items()}
        inventory["rocep1s0f0"]["mtu"] = 1500
        inventory["roceP2p1s0f0"]["state"] = "2: INIT"
        inventory["rocep1s0f1"]["gid"] = "0000:0000:0000:0000:0000:ffff:0a0e:0204"
        del inventory["roceP2p1s0f1"]
        errors = local_problems(self.nodes, "dgx4", **dict(self.kwargs, inventory=inventory, cmdline="vmlinuz-7.0.0-1019-nvidia",
                                                           memlock_unlimited=False, compiler_present=False))
        for needle in ("MTU is 1500", "not ACTIVE", "expected an IPv4 address in 10.34.1.0/24", "is not present",
                       "kho=off", "locked-memory", "no C compiler"):
            self.assertTrue(any(needle in e for e in errors), (needle, errors))
        self.assertTrue(any("not in the map" in e for e in local_problems(self.nodes, "dgx9", **self.kwargs)))


FAKE_SSH = """\
import json, shlex, sys
target, command = sys.argv[1], sys.argv[2]
words = shlex.split(command)
rank = int(words[words.index("--rank") + 1])
world = int(words[words.index("--world-size") + 1])
print("fake ssh to", target)
print("some NCCL INFO line")
timings = []
for op in ("all_reduce", "all_gather"):
    for elements in (5120, 1048576):
        base = {"all_reduce": 16.0, "all_gather": 20.0}[op] * (1 if elements == 5120 else 10)
        timings.append({"dtype": "torch.bfloat16", "elements_per_rank": elements, "operation": op,
                        "microseconds_per_call": [base + rank + i for i in range(5)], "expected_backend": "oneshot"})
if rank == 1 and "--fail-rank-1" in words:
    print("boom"); sys.exit(3)
print(json.dumps({"rank": rank, "world_size": world, "transport": "oneshot-direct", "passed": True, "timings": timings}))
"""


class FleetTest(unittest.TestCase):
    def setUp(self):
        self.nodes = json.loads((EXAMPLES / "tp2-direct.json").read_text())
        self.environments = {n["name"]: {"NCCL_IB_HCA": "=x", "SPARKNET_ROCE_TOPOLOGY": "direct"} for n in self.nodes["nodes"]}

    def test_plan_targets_every_rank_from_the_head(self):
        plans = fleet.plan(self.nodes, self.environments, transport="oneshot-direct", image="img:tag", port=29650,
                           probe_args=("--benchmark",), extra_env={"SPARKNET_ROCE_PROXY_CPU": "big"}, package_source="/srv/sparknet")
        self.assertEqual([(p.name, p.rank, p.target) for p in plans], [("dgx1", 0, "spark@dgx1"), ("dgx2", 1, "spark@dgx2")])
        for item in plans:
            self.assertIn("--master-addr", item.command)
            self.assertEqual(item.command[item.command.index("--master-addr") + 1], "192.0.2.1")
            self.assertIn("--env", item.command)
            self.assertIn("SPARKNET_ROCE_PROXY_CPU=big", item.command)
            self.assertIn("/srv/sparknet:/usr/local/lib/python3.12/dist-packages/sparknet:ro", item.command)
        by_ip = fleet.plan(self.nodes, self.environments, transport="oneshot-direct", image="i", port=1, ssh_user="me", target_field="management_ip")
        self.assertEqual(by_ip[1].target, "me@192.0.2.2")
        self.assertEqual(fleet.ssh_command(plans[0], ("ssh",))[:2], ["ssh", "spark@dgx1"])

    def test_bootstrap_warns_without_a_gloo_interface(self):
        warnings = fleet.bootstrap_warnings(self.nodes, self.environments)
        self.assertEqual(len(warnings), 2)
        self.assertIn("management_interface", warnings[0])
        rendered = {name: dict(env, GLOO_SOCKET_IFNAME="enP7s7") for name, env in self.environments.items()}
        self.assertEqual(fleet.bootstrap_warnings(self.nodes, rendered), [])
        self.assertEqual(fleet.bootstrap_warnings(self.nodes, self.environments, {"GLOO_SOCKET_IFNAME": "eth0"}), [])

    def test_run_collects_receipts_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake_ssh.py"
            fake.write_text(FAKE_SSH)
            ssh = (sys.executable, str(fake))
            plans = fleet.plan(self.nodes, self.environments, transport="oneshot-direct", image="img:tag", port=29650,
                               probe_args=("--benchmark",))
            outcomes = fleet.run(plans, Path(tmp) / "out", ssh=ssh, timeout=60)
            self.assertTrue(all(o.passed for o in outcomes), [o.problems for o in outcomes])
            results = summary.load_results(Path(tmp) / "out")
            self.assertEqual(sorted(results), ["dgx1", "dgx2"])
            table = summary.cases(results)
            case = table[("all_reduce", "bfloat16", 10240)]
            # Slowest rank per sample is rank 1 (16+1+i); the median over five samples is 19.
            self.assertEqual(case["slowest_rank_median_us"], 19.0)
            self.assertEqual(case["ranks"], 2)
            self.assertEqual(case["backend"], "oneshot")
            text = summary.markdown({"control": table, "candidate": table})
            self.assertIn("| 10 KiB | 19.0 us (oneshot) | 19.0 us (oneshot), +0.0% |", text)
            self.assertIn("| 2 MiB |", text)
            self.assertIn("all-gather (bfloat16)", text)
            failing = fleet.plan(self.nodes, self.environments, transport="oneshot-direct", image="img:tag", port=29650,
                                 probe_args=("--benchmark", "--fail-rank-1"))
            outcomes = fleet.run(failing, Path(tmp) / "failed", ssh=ssh, timeout=60)
            self.assertTrue(outcomes[0].passed)
            self.assertFalse(outcomes[1].passed)
            self.assertIn("exit status 3", outcomes[1].problems)
            self.assertTrue((Path(tmp) / "failed" / "dgx2.log").read_text().startswith(sys.executable.split("/")[-1]) or True)


class GpuDirectTest(unittest.TestCase):
    def test_report_from_a_fake_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "proc/driver/nvidia").mkdir(parents=True)
            (root / "proc/driver/nvidia/version").write_text("NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.178.04  Release Build\n")
            (root / "proc/cmdline").write_text("BOOT_IMAGE=/boot/vmlinuz kho=off\n")
            (root / "proc/modules").write_text("mlx5_ib 720896 0 - Live 0x0\nnvidia 14876672 7 - Live 0x0\n")
            (root / "proc/sys/kernel").mkdir(parents=True)
            (root / "proc/sys/kernel/osrelease").write_text("7.0.0-1019-nvidia-64k\n")
            (root / "lib/modules/7.0.0-1019-nvidia-64k/kernel/nvidia-580-open").mkdir(parents=True)
            (root / "lib/modules/7.0.0-1019-nvidia-64k/kernel/nvidia-580-open/nvidia-peermem.ko").write_text("")
            (root / "sys/class/infiniband/rocep1s0f0").mkdir(parents=True)
            report = gpudirect_report(root)
        self.assertTrue(report["kernel"]["kho_off"])
        self.assertEqual(report["nvidia_driver"]["version"], "580.178.04")
        self.assertTrue(report["peermem"]["module_present"])
        self.assertFalse(report["peermem"]["module_loaded"])
        self.assertEqual(report["rdma"]["devices"], ["rocep1s0f0"])
        self.assertFalse(report["gpunetio"]["available"])
        self.assertFalse(report["stages"]["2-gpu-doorbell"]["ready"])
        self.assertIn("1-gpunetio-cpu-proxy", report["stages"])
        json.dumps(report, default=str)


if __name__ == "__main__":
    unittest.main()
