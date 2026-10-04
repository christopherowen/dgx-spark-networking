"""The collective policy must be rank-invariant and keep its three limits distinct."""

import unittest

from sparknet.policy import TP3_POLICY, TP4_POLICY, CollectivePolicy, policy_for_profile


class PolicyTest(unittest.TestCase):
    def test_dispatch_boundaries(self):
        p = TP4_POLICY
        self.assertEqual(p.all_reduce_backend(1048576, "bfloat16"), "oneshot")
        self.assertEqual(p.all_reduce_backend(1048576 + 16, "bfloat16"), "nccl")
        self.assertEqual(p.all_reduce_backend(1048576, "bfloat16", contiguous=False), "nccl")
        self.assertEqual(p.all_reduce_backend(24, "bfloat16"), "nccl")  # not a multiple of 16
        self.assertEqual(p.all_reduce_backend(16, "int32"), "nccl")
        self.assertEqual(p.all_reduce_backend(0, "float32"), "nccl")
        self.assertEqual(p.all_gather_backend(2097152, "int64", dim=-1, ndim=2), "oneshot")
        self.assertEqual(p.all_gather_backend(2097152 + 1, "int64", dim=-1, ndim=2), "nccl")
        self.assertEqual(p.all_gather_backend(1024, "bfloat16", dim=1, ndim=3), "nccl")
        self.assertEqual(p.all_gather_backend(1024, "bool", dim=0, ndim=1), "nccl")
        self.assertEqual(p.reduce_scatter_backend(), "nccl")

    def test_capacity_is_independent_of_dispatch(self):
        self.assertEqual(TP4_POLICY.all_reduce_capacity_bytes, 2 * 1024 ** 2)
        self.assertEqual(TP4_POLICY.all_reduce_dispatch_bytes, 1024 ** 2)
        self.assertEqual(TP3_POLICY.all_reduce_dispatch_bytes, TP3_POLICY.all_reduce_capacity_bytes)
        self.assertTrue(CollectivePolicy(4 * 1024 ** 2, 2 * 1024 ** 2, 1024).problems())
        self.assertTrue(CollectivePolicy(24, 1024, 1024).problems())

    def test_environment_round_trip(self):
        env = TP4_POLICY.environment()
        self.assertEqual(env["B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES"], "1048576")
        self.assertEqual(CollectivePolicy.from_environment(env), TP4_POLICY)
        vllm = {"VLLM_ROCE_ALLREDUCE_MAX_SIZE": "2MB", "VLLM_ROCE_ALLGATHER_MAX_SIZE": "4MB"}
        self.assertEqual(CollectivePolicy.from_environment(vllm), TP3_POLICY)
        self.assertEqual(CollectivePolicy.from_environment({**vllm, "B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": "1048576"}).all_reduce_dispatch_bytes, 1048576)
        with self.assertRaises(ValueError):
            CollectivePolicy.from_environment({})
        with self.assertRaises(ValueError):
            CollectivePolicy.from_environment({**vllm, "SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES": "4194304"})

    def test_profiles_map_to_policies(self):
        self.assertEqual(policy_for_profile("tp4-ring"), TP4_POLICY)
        self.assertEqual(policy_for_profile("tp3-triangle"), TP3_POLICY)
        self.assertIsNone(policy_for_profile("tp4-ring-nccl-only"))


if __name__ == "__main__":
    unittest.main()
