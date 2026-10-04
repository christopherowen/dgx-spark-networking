"""Probe planning and read-only host reports."""

import json
import tempfile
import unittest
from pathlib import Path

from sparknet.probe.container import docker_probe_command
from sparknet.probe.doctor import local_problems
from sparknet.probe.gpudirect import gpudirect_report
from sparknet.topology import nodes as topology

EXAMPLES = Path(__file__).resolve().parents[1] / "sparknet" / "topology" / "examples"


class ContainerTest(unittest.TestCase):
    def test_probe_is_bounded_and_does_not_share_serving_ipc(self):
        command = docker_probe_command(image="img:tag", environment={"NCCL_DEBUG": "WARN", "A": "1"}, rank=3, world_size=4,
                                       master_addr="192.0.2.1", master_port=29999, transport="rocenante-ring4",
                                       probe_source="/opt/sparknet/probe/collectives.py", extra_args=("--benchmark",))
        self.assertIn("600s", command)
        self.assertIn("--memory=12g", command)
        self.assertNotIn("--ipc=host", command)
        self.assertIn("--entrypoint=/usr/bin/timeout", command)
        self.assertEqual(command[command.index("--world-size") + 1], "4")
        self.assertEqual(command[command.index("--transport") + 1], "rocenante-ring4")
        self.assertIn("--env", command)
        self.assertIn("NCCL_DEBUG=INFO", command)
        self.assertEqual(command[-1], "--benchmark")


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
        self.kwargs = dict(transport="rocenante-ring4", inventory=self.inventory, cmdline="BOOT_IMAGE=/boot/vmlinuz-7.0.0-1019-nvidia-64k kho=off",
                           memlock_unlimited=True, compiler_present=True, verbs_header=True)

    def test_matching_host_passes(self):
        self.assertEqual(local_problems(self.nodes, "dgx4", **self.kwargs), [])

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
