"""The transport boundary: geometry checks and the staged GPU-initiated path."""

import os
import tempfile
import unittest
from pathlib import Path

from sparknet.transport import Geometry, GpuNetIOTransport, TransportCapability, gpunetio_capability
from sparknet.transport.host_proxy import host_proxy_capability


class GeometryTest(unittest.TestCase):
    def test_validation(self):
        good = Geometry(world_size=4, rank=1, topology="ring4", hca_names=("a", "b", "c", "d"),
                        peer_hca_indices=((0, 1), (), (2, 3), ()), stripe_count=2, gid_index=3,
                        slot_bytes=4096, region_ptr=0, region_bytes=1 << 20)
        self.assertEqual(good.validate(), [])
        bad = Geometry(world_size=3, rank=5, topology="ring4", hca_names=(), peer_hca_indices=(), stripe_count=1,
                       gid_index=3, slot_bytes=100, region_ptr=0, region_bytes=0)
        problems = bad.validate()
        for needle in ("rank", "four ranks", "1..4", "every rank", "4096"):
            self.assertTrue(any(needle in p for p in problems), (needle, problems))


class GpuNetIOTest(unittest.TestCase):
    def test_staged_transport_fails_loudly(self):
        os.environ.pop("SPARKNET_GPUNETIO_DIR", None)
        capability = gpunetio_capability()
        self.assertIsInstance(capability, TransportCapability)
        self.assertFalse(capability.available)
        self.assertTrue(any("SPARKNET_GPUNETIO_DIR" in r for r in capability.reasons))
        with self.assertRaises(NotImplementedError):
            GpuNetIOTransport(world_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lib").mkdir()
            (root / "include").mkdir()
            (root / "lib/libdoca_gpunetio.so").write_text("")
            (root / "include/doca_gpunetio_device.h").write_text("")
            self.assertTrue(gpunetio_capability(root).available)

    def test_host_proxy_capability_reports_reasons(self):
        capability = host_proxy_capability()
        self.assertEqual(capability.name, "host-proxy")
        self.assertIsInstance(capability.details["rdma_devices"], list)


if __name__ == "__main__":
    unittest.main()
