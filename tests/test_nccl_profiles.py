"""Named profiles must reproduce the measured settings and reject silent drift."""

import unittest

from sparknet.nccl import profiles


class ProfileTest(unittest.TestCase):
    def test_tp4_ring_reproduces_the_balanced_policy(self):
        env = profiles.environment("tp4-ring")
        expected = {
            "NCCL_ALGO": "Ring", "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "4194304", "NCCL_LL128_BUFFSIZE": "262144", "NCCL_PROTO": "^LL128",
            "NCCL_SWITCHLESS_BIDIRECTIONAL": "2", "NCCL_MIN_TRAFFIC_PER_CHANNEL": "512",
            "NCCL_THREAD_THRESHOLDS": "-2 -2 -2 1 1 1", "NCCL_CUMEM_ENABLE": "1", "NCCL_RUNTIME_CONNECT": "1",
            "NCCL_IB_SUBNET_PREFIX_LEN": "24", "NCCL_IB_GID_INDEX": "3", "NCCL_P2P_DISABLE": "1",
            "SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES": "2097152",
            "SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": "1048576",
            "SPARKNET_ROCE_ALLGATHER_MAX_BYTES": "2097152", "SPARKNET_ROCE_SPIN_LIMIT": "5000000",
            "B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": "1048576", "B12X_ROCE_SPIN_LIMIT": "5000000",
            "VLLM_ROCE_ALLREDUCE_MAX_SIZE": "2MB", "VLLM_ROCE_ALLGATHER_MAX_SIZE": "2MB",
            "VLLM_ENABLE_ROCE_ALLREDUCE": "1", "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
        }
        for key, value in expected.items():
            self.assertEqual(env.get(key), value, key)
        self.assertEqual(profiles.problems(env, node_count=4), [])
        self.assertEqual(profiles.required_patches(env), [
            "0001-ib-cts-nreqs-acquire-fence.patch", "0002-bidirectional-switchless-rings.patch",
            "0003-balanced-channel-allocation.patch", "0004-adaptive-small-ring-threads.patch"])

    def test_tp3_triangle_reproduces_the_promoted_baseline(self):
        env = profiles.environment("tp3-triangle")
        self.assertEqual(env["NCCL_MAX_NCHANNELS"], "8")
        self.assertEqual(env["NCCL_BUFFSIZE"], "1048576")
        self.assertEqual(env["NCCL_CUMEM_ENABLE"], "0")
        for absent in ("NCCL_MIN_NCHANNELS", "NCCL_SWITCHLESS_BIDIRECTIONAL", "NCCL_MIN_TRAFFIC_PER_CHANNEL",
                       "NCCL_THREAD_THRESHOLDS", "NCCL_ALGO", "SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES"):
            self.assertNotIn(absent, env)
        self.assertEqual(env["VLLM_ROCE_ALLGATHER_MAX_SIZE"], "4MB")
        self.assertEqual(profiles.problems(env, node_count=3), [])
        self.assertEqual(profiles.required_patches(env), ["0001-ib-cts-nreqs-acquire-fence.patch"])
        bare = profiles.environment("tp3-triangle", compat=False)
        self.assertNotIn("VLLM_ROCE_ALLREDUCE_MAX_SIZE", bare)
        self.assertNotIn("B12X_ROCE_SPIN_LIMIT", bare)

    def test_nccl_only_profile_disables_the_custom_collectives(self):
        env = profiles.environment("tp4-ring-nccl-only")
        self.assertEqual(env["VLLM_ENABLE_ROCE_ALLREDUCE"], "0")
        self.assertFalse(any(k.startswith("SPARKNET_ROCE") for k in env))
        self.assertEqual(profiles.problems(env, node_count=4), [])

    def test_validation_rejects_contradictions(self):
        env = profiles.environment("tp4-ring")
        cases = {
            "NCCL_MIN_NCHANNELS": ("8", "exceeds maximum"),
            "NCCL_THREAD_THRESHOLDS": ("1 2 3", "six space-separated"),
            "NCCL_PROTO": ("Simple", "LL128"),
            "NCCL_SWITCHLESS_BIDIRECTIONAL": ("3", "must be 1 or 2"),
            "NCCL_MIN_TRAFFIC_PER_CHANNEL": ("100", "multiple of 16"),
            "NCCL_BUFFSIZE": ("-1", "positive integer"),
            "SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": ("4194304", "must not exceed"),
            "SPARKNET_ROCE_ALLGATHER_MAX_BYTES": ("1000", "multiple of 16"),
        }
        for key, (value, message) in cases.items():
            with self.subTest(key=key):
                errors = profiles.problems(dict(env, **{key: value}), node_count=4)
                self.assertTrue(any(message in e for e in errors), errors)
        odd = dict(env, NCCL_MIN_NCHANNELS="6", NCCL_MAX_NCHANNELS="6")
        self.assertTrue(any("multiple of four" in e for e in profiles.problems(odd, node_count=4)))
        self.assertTrue(any("three or four nodes" in e for e in profiles.problems(env, node_count=2)))

    def test_unpatched_library_refuses_patch_controls(self):
        env = profiles.environment("tp4-ring")
        errors = profiles.problems(env, node_count=4, patched_nccl=False)
        self.assertTrue(any("NCCL_SWITCHLESS_BIDIRECTIONAL needs the patched NCCL" in e for e in errors))
        self.assertEqual(profiles.problems(profiles.environment("tp3-triangle"), node_count=3, patched_nccl=False), [])

    def test_two_node_and_switched_profiles(self):
        for name, transport, counts in (("tp2-direct", "oneshot-direct", (2,)), ("direct-nccl-only", "nccl-direct", (2, 3)),
                                        ("switched", "oneshot-switched", tuple(range(2, 17))), ("switched-nccl-only", "nccl-switched", tuple(range(2, 17)))):
            with self.subTest(name=name):
                entry = profiles.profile(name)
                self.assertEqual((entry["transport"], entry["node_counts"]), (transport, counts))
                env = profiles.environment(name)
                for count in counts[:3]:
                    self.assertEqual(profiles.problems(env, node_count=count), [])
                self.assertEqual(profiles.required_patches(env), ["0001-ib-cts-nreqs-acquire-fence.patch"])
                self.assertNotIn("NCCL_ALGO", env)
                if transport.startswith("oneshot"):
                    self.assertEqual(env["VLLM_ROCE_ALLREDUCE_MAX_SIZE"], "2MB")
                    self.assertEqual(env["SPARKNET_ROCE_ALLGATHER_MAX_BYTES"], "4194304")
                else:
                    self.assertEqual(env["VLLM_ENABLE_ROCE_ALLREDUCE"], "0")
        self.assertEqual(profiles.profiles_for("oneshot-switched", 12), ["switched"])
        self.assertEqual(profiles.profiles_for("nccl-direct", 3), ["direct-nccl-only"])
        self.assertEqual(profiles.profiles_for("oneshot-direct", 4), [])
        self.assertEqual(profiles.profile_problems("switched", "oneshot-switched", 16), [])
        self.assertTrue(profiles.profile_problems("tp2-direct", "oneshot-direct", 3))

    def test_size_syntax(self):
        self.assertEqual(profiles.size_bytes("2MB"), 2 * 1024 ** 2)
        self.assertEqual(profiles.size_bytes("512"), 512)
        with self.assertRaises(ValueError):
            profiles.size_bytes("2 MB")
        with self.assertRaises(ValueError):
            profiles.profile("tp5")


if __name__ == "__main__":
    unittest.main()
