"""The CLI runs without torch and composes the library's checks."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sparknet import cli

EXAMPLES = Path(__file__).resolve().parents[1] / "sparknet" / "topology" / "examples"
EXAMPLE = str(EXAMPLES / "tp4-ring.json")
TRIANGLE = str(EXAMPLES / "tp3-triangle.json")
TWO = str(EXAMPLES / "tp2-direct.json")
SWITCHED = str(EXAMPLES / "switched.json")


def run(*argv):
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli.main(list(argv))
    return code, out.getvalue()


class CliTest(unittest.TestCase):
    def test_validate_and_render(self):
        self.assertEqual(run("topology", "validate", EXAMPLE, "--transport", "oneshot-ring4")[0], 0)
        self.assertEqual(run("topology", "validate", EXAMPLE, "--transport", "oneshot-direct")[0], 1)
        code, text = run("topology", "render", EXAMPLE, "dgx2", "--transport", "oneshot-ring4", "--profile", "tp4-ring", "--json")
        self.assertEqual(code, 0)
        env = json.loads(text)
        self.assertEqual(env["NCCL_SWITCHLESS_BIDIRECTIONAL"], "2")
        self.assertEqual(set(json.loads(env["SPARKNET_ROCE_PEER_HCAS"])), {"0", "2"})
        self.assertEqual(env["SPARKNET_ROCE_TOPOLOGY"], "ring4")
        self.assertFalse([k for k in env if k.startswith(("B12X_", "VLLM_"))])
        code, text = run("topology", "render", TRIANGLE, "dgx1", "--transport", "oneshot-direct", "--profile", "tp3-triangle")
        self.assertEqual(code, 0)
        self.assertIn("NCCL_MAX_NCHANNELS=8", text)
        # A triangle profile on a ring map is refused: the profile's transport and node count must match.
        self.assertEqual(run("topology", "render", EXAMPLE, "dgx1", "--transport", "nccl-ring", "--profile", "tp3-triangle")[0], 1)

    def test_every_fabric_renders(self):
        cases = ((TWO, "dgx2", "oneshot-direct", "tp2-direct", "SPARKNET_ROCE_PEER_HCAS"),
                 (TWO, "dgx1", "nccl-direct", "direct-nccl-only", "NCCL_IB_HCA"),
                 (TRIANGLE, "dgx3", "nccl-direct", "direct-nccl-only", "NCCL_IB_HCA"),
                 (SWITCHED, "dgx4", "oneshot-switched", "switched", "SPARKNET_ROCE_HCA"),
                 (SWITCHED, "dgx1", "nccl-switched", "switched-nccl-only", "NCCL_IB_TC"))
        for nodes, node, transport, profile, key in cases:
            with self.subTest(transport=transport):
                code, text = run("topology", "render", nodes, node, "--transport", transport, "--profile", profile, "--json")
                self.assertEqual(code, 0, text)
                self.assertIn(key, json.loads(text))
                code, text = run("probe", "render-command", nodes, node, "--transport", transport, "--profile", profile, "--image", "x:y")
                self.assertEqual(code, 0)
                self.assertIn(f"--transport {transport}", text)
        self.assertEqual(run("nccl", "validate", "--profile", "switched", "--nodes", "9")[0], 0)
        self.assertEqual(run("nccl", "validate", "--profile", "switched", "--nodes", "17")[0], 1)

    def test_examples_nccl_and_policy(self):
        code, text = run("topology", "example", "tp4-ring")
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(text)["nodes"]), 4)
        self.assertEqual(run("topology", "example", "tp9")[0], 1)
        code, text = run("nccl", "env", "--profile", "tp4-ring", "--json")
        self.assertEqual(json.loads(text)["NCCL_MIN_TRAFFIC_PER_CHANNEL"], "512")
        self.assertEqual(run("nccl", "validate", "--profile", "tp4-ring")[0], 0)
        self.assertEqual(run("nccl", "validate", "--profile", "tp4-ring", "--unpatched")[0], 1)
        self.assertIn("0004-adaptive", run("nccl", "patches")[1])
        self.assertIn("tp4-ring:", run("nccl", "profiles")[1])
        code, text = run("policy", "show", "--profile", "tp4-ring")
        self.assertEqual(json.loads(text)["all_reduce_dispatch_bytes"], 1048576)
        self.assertIn("NCCL carries every collective", run("policy", "show", "--profile", "tp4-ring-nccl-only")[1])

    def test_policy_show_without_a_profile_reports_cleanly(self):
        import os
        from contextlib import redirect_stderr
        from unittest import mock

        err = io.StringIO()
        with mock.patch.dict(os.environ, {}, clear=True), redirect_stderr(err):
            code, text = run("policy", "show")
        self.assertEqual(code, 1)
        self.assertIn("pass --profile", err.getvalue())
        with mock.patch.dict(os.environ, {"SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES": "2097152",
                                          "SPARKNET_ROCE_ALLGATHER_MAX_BYTES": "4194304"}, clear=True):
            code, text = run("policy", "show")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(text)["all_reduce_dispatch_bytes"], 2097152)

    def test_pass_through_arguments_are_split_before_parsing(self):
        self.assertEqual(cli.split_pass_through(["probe", "x", "--", "--benchmark", "--"]), (["probe", "x"], ["--benchmark", "--"]))
        self.assertEqual(cli.split_pass_through(["nccl", "profiles"]), (["nccl", "profiles"], None))
        with self.assertRaises(SystemExit):
            run("nccl", "profiles", "--", "extra")

    def test_patch_series_is_packaged_and_exports(self):
        import tempfile
        from sparknet.nccl import patchset, profiles

        names = patchset.series()
        self.assertEqual(names[0], profiles.FENCE_PATCH)
        self.assertIn(profiles.ADAPTIVE_THREADS_PATCH, names)
        for control in profiles.PATCH_CONTROLS.values():
            for patch in control:
                self.assertIn(patch, names)
        self.assertEqual(patchset.missing(), [])
        code, text = run("nccl", "patches")
        self.assertEqual(code, 0)
        self.assertEqual([line for line in text.splitlines() if line and not line.startswith("#")], names)
        with tempfile.TemporaryDirectory() as tmp:
            code, text = run("nccl", "patches", "--export", tmp)
            self.assertEqual(code, 0)
            written = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(written, sorted([patchset.SERIES_FILE, *names]))
            self.assertEqual((Path(tmp) / names[0]).read_bytes(), (patchset.patch_directory() / names[0]).read_bytes())

    def test_subset_fleet_and_summarize(self):
        import tempfile

        code, text = run("topology", "subset", EXAMPLE, "dgx2", "dgx3")
        self.assertEqual(code, 0)
        pair = json.loads(text)
        self.assertEqual([n["name"] for n in pair["nodes"]], ["dgx2", "dgx3"])
        self.assertEqual(run("topology", "subset", EXAMPLE, "dgx1", "dgx3")[0], 1)
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "pair.json")
            self.assertEqual(run("topology", "subset", EXAMPLE, "dgx1", "dgx2", "--out", out)[0], 0)
            code, text = run("probe", "fleet", out, "--transport", "oneshot-direct", "--profile", "tp2-direct", "--image", "img:tag",
                             "--env", "SPARKNET_ROCE_PROXY_CPU=big", "--dry-run", "--", "--benchmark")
            self.assertEqual(code, 0, text)
            lines = [line for line in text.splitlines() if not line.startswith("#")]
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith("ssh -o BatchMode=yes -o ConnectTimeout=15 spark@dgx1 "))
            self.assertIn("SPARKNET_ROCE_PROXY_CPU=big", lines[0])
            self.assertIn("sparknet.probe.collectives", lines[1])
            self.assertTrue(lines[1].rstrip("'").endswith("--benchmark"), lines[1])
            # A profile that does not fit the carved map is refused before any ssh.
            self.assertEqual(run("probe", "fleet", out, "--transport", "oneshot-direct", "--profile", "tp4-ring", "--image", "i", "--dry-run")[0], 1)
            receipts = Path(tmp) / "receipts"
            receipts.mkdir()
            for rank, name in enumerate(("dgx1", "dgx2")):
                (receipts / f"{name}.json").write_text(json.dumps({"rank": rank, "passed": True, "timings": [
                    {"dtype": "torch.bfloat16", "elements_per_rank": 5120, "operation": "all_reduce",
                     "microseconds_per_call": [17.0 + rank] * 5, "expected_backend": "oneshot"}]}))
            code, text = run("probe", "summarize", f"ring={receipts}")
            self.assertEqual(code, 0)
            self.assertIn("| 10 KiB | 18.0 us (oneshot) |", text)
            self.assertIn("| all-reduce (bfloat16), per rank | ring |", text)

    def test_probe_command_plan(self):
        code, text = run("probe", "render-command", EXAMPLE, "dgx4", "--transport", "oneshot-ring4", "--profile", "tp4-ring",
                         "--image", "example:tag", "--", "--benchmark")
        self.assertEqual(code, 0)
        self.assertIn("--world-size 4", text)
        self.assertIn("--master-addr 192.0.2.1", text)
        self.assertIn("NCCL_SWITCHLESS_BIDIRECTIONAL=2", text)
        self.assertTrue(text.endswith("--benchmark\n"))


if __name__ == "__main__":
    unittest.main()
