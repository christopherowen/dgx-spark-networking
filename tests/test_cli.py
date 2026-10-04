"""The CLI runs without torch and composes the library's checks."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sparknet import cli

EXAMPLE = str(Path(__file__).resolve().parents[1] / "sparknet" / "topology" / "examples" / "tp4-ring.json")
TRIANGLE = str(Path(__file__).resolve().parents[1] / "sparknet" / "topology" / "examples" / "tp3-triangle.json")


def run(*argv):
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli.main(list(argv))
    return code, out.getvalue()


class CliTest(unittest.TestCase):
    def test_validate_and_render(self):
        self.assertEqual(run("topology", "validate", EXAMPLE, "--transport", "rocenante-ring4")[0], 0)
        self.assertEqual(run("topology", "validate", EXAMPLE, "--transport", "rocenante-direct")[0], 1)
        code, text = run("topology", "render", EXAMPLE, "dgx2", "--transport", "rocenante-ring4", "--profile", "tp4-ring", "--json")
        self.assertEqual(code, 0)
        env = json.loads(text)
        self.assertEqual(env["NCCL_SWITCHLESS_BIDIRECTIONAL"], "2")
        self.assertEqual(set(json.loads(env["SPARKNET_ROCE_PEER_HCAS"])), {"0", "2"})
        self.assertEqual(env["B12X_ROCE_TOPOLOGY"], "ring4")
        code, text = run("topology", "render", TRIANGLE, "dgx1", "--transport", "rocenante-direct", "--profile", "tp3-triangle")
        self.assertEqual(code, 0)
        self.assertIn("NCCL_MAX_NCHANNELS=8", text)
        # A triangle profile on a ring map is refused: the profile's node count must match.
        self.assertEqual(run("topology", "render", EXAMPLE, "dgx1", "--transport", "nccl-ring", "--profile", "tp3-triangle")[0], 1)

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

    def test_probe_command_plan(self):
        code, text = run("probe", "render-command", EXAMPLE, "dgx4", "--transport", "rocenante-ring4", "--profile", "tp4-ring",
                         "--image", "example:tag", "--", "--benchmark")
        self.assertEqual(code, 0)
        self.assertIn("--world-size 4", text)
        self.assertIn("--master-addr 192.0.2.1", text)
        self.assertIn("NCCL_SWITCHLESS_BIDIRECTIONAL=2", text)
        self.assertTrue(text.endswith("--benchmark\n"))


if __name__ == "__main__":
    unittest.main()
