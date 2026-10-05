"""Measured NCCL environment profiles for switchless DGX Spark fabrics, and the patch series behind them.

The series ships in ``sparknet/nccl/patches``; ``sparknet.nccl.patchset`` reads and exports it.
"""

from .patchset import export as export_patches, patch_directory, series
from .profiles import (
    COMMON_IB_ENV,
    PROFILES,
    PATCH_CONTROLS,
    environment,
    problems,
    profile,
    profile_problems,
    profiles_for,
    required_patches,
    size_bytes,
)

__all__ = [
    "COMMON_IB_ENV",
    "PATCH_CONTROLS",
    "PROFILES",
    "environment",
    "export_patches",
    "patch_directory",
    "problems",
    "profile",
    "profile_problems",
    "profiles_for",
    "required_patches",
    "series",
    "size_bytes",
]
