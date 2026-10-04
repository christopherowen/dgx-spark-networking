"""What this host offers for the GPU-initiated transport. Read-only; loads no driver, opens no device.

The report distinguishes the three pivot stages (see ``docs/gpudirect-roadmap.md``):
the CPU-proxy GPUNetIO path (needs the built library), the GPU doorbell path
(needs a UAR mapping the GPU may write, and a resolved host failure) and
device-memory registration (dma-buf export from the driver plus
``ibv_reg_dmabuf_mr`` in rdma-core, or ``nvidia-peermem``).
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any

from sparknet.transport.gpunetio import ENV_GPUNETIO_DIR, gpunetio_capability
from sparknet.transport.host_proxy import host_proxy_capability


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _symbol(library: str, name: str) -> bool | None:
    try:
        return hasattr(ctypes.CDLL(library), name)
    except OSError:
        return None


def gpudirect_report(root: str | Path = "/", *, gpunetio_root: str | None = None) -> dict[str, Any]:
    """A JSON-serializable report; ``root`` lets tests point at a fake filesystem."""
    root = Path(root)
    proc_cmdline = _read(root / "proc/cmdline") or ""
    nvidia_version = _read(root / "proc/driver/nvidia/version") or ""
    driver = re.search(r"\b(\d{3}\.\d+(?:\.\d+)?)\b", nvidia_version)
    modules = _read(root / "proc/modules") or ""
    loaded = {line.split()[0] for line in modules.splitlines() if line.strip()}
    release = platform.release() if root == Path("/") else (_read(root / "proc/sys/kernel/osrelease") or "")
    module_dir = root / "lib/modules" / release / "kernel"
    peermem_files = list(module_dir.rglob("nvidia-peermem.ko*")) if module_dir.exists() else []
    rdma_devices = sorted(p.name for p in (root / "sys/class/infiniband").glob("*")) if (root / "sys/class/infiniband").exists() else []
    report: dict[str, Any] = {
        "kernel": {
            "release": release,
            "kho_off": "kho=off" in proc_cmdline.split(),
            "note": "without kho=off, ibv_reg_mr fails with ENOMEM under memory pressure on the 7.0 kernels",
        },
        "nvidia_driver": {"version": driver.group(1) if driver else None, "open_kernel_module": "Open Kernel Module" in nvidia_version},
        "rdma": {
            "devices": rdma_devices,
            "libibverbs_reg_dmabuf_mr": _symbol("libibverbs.so.1", "ibv_reg_dmabuf_mr") if root == Path("/") else None,
            "mlx5dv_headers": (root / "usr/include/infiniband/mlx5dv.h").exists(),
        },
        "peermem": {
            "module_present": bool(peermem_files),
            "module_loaded": "nvidia_peermem" in loaded,
            "note": "present but not loaded on the fleet; loading it is a host decision, not a library one",
        },
        "gdrcopy": {"device": (root / "dev/gdrdrv").exists(), "module_loaded": "gdrdrv" in loaded},
        "host_proxy": host_proxy_capability().__dict__ if root == Path("/") else {"available": None},
        "gpunetio": gpunetio_capability(gpunetio_root).__dict__,
        "cuda_dmabuf": _cuda_dmabuf_attribute() if root == Path("/") else {"checked": False},
        "memlock_unlimited": _memlock_unlimited(),
        "stages": {},
    }
    stages = report["stages"]
    stages["1-gpunetio-cpu-proxy"] = {
        "ready": bool(report["gpunetio"]["available"]) and bool(rdma_devices),
        "needs": [] if report["gpunetio"]["available"] else list(report["gpunetio"]["reasons"]),
    }
    stages["2-gpu-doorbell"] = {
        "ready": False,
        "needs": ["a host-level investigation of the dgx1 reboot seen with NIC_HANDLER_GPU_SM_DB (spark3 2026-10-03-relay-progress)",
                  "doca_gpu_verbs_can_gpu_register_uar true on this driver"],
    }
    dmabuf = report["cuda_dmabuf"].get("supported")
    stages["3-device-memory-registration"] = {
        "ready": bool(dmabuf) and bool(report["rdma"]["libibverbs_reg_dmabuf_mr"]),
        "needs": ([] if dmabuf else ["CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED on the GB10 driver"])
                 + ([] if report["rdma"]["libibverbs_reg_dmabuf_mr"] else ["ibv_reg_dmabuf_mr in libibverbs"]),
        "alternative": "nvidia-peermem (present, not loaded)" if peermem_files else "nvidia-peermem module",
    }
    return report


def _cuda_dmabuf_attribute() -> dict[str, Any]:
    """``CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`` through cuda-python, if a GPU is visible."""
    if os.environ.get("SPARKNET_PROBE_SKIP_CUDA") == "1":
        return {"checked": False, "reason": "SPARKNET_PROBE_SKIP_CUDA=1"}
    try:
        from cuda.bindings import driver as cuda
    except ImportError:
        return {"checked": False, "reason": "cuda-python not installed"}
    try:
        if cuda.cuInit(0)[0] != cuda.CUresult.CUDA_SUCCESS:
            return {"checked": False, "reason": "cuInit failed"}
        err, count = cuda.cuDeviceGetCount()
        if err != cuda.CUresult.CUDA_SUCCESS or count < 1:
            return {"checked": False, "reason": "no CUDA device"}
        err, device = cuda.cuDeviceGet(0)
        err, value = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device)
        return {"checked": err == cuda.CUresult.CUDA_SUCCESS, "supported": bool(value)}
    except Exception as exc:  # noqa: BLE001 - a report, not a gate
        return {"checked": False, "reason": str(exc)}


def _memlock_unlimited() -> bool | None:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        return soft == resource.RLIM_INFINITY
    except (ImportError, ValueError, OSError):
        return None


__all__ = ["gpudirect_report"]
