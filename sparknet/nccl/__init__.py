"""Measured NCCL environment profiles for switchless DGX Spark fabrics, and the patch series behind them."""

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
    "problems",
    "profile",
    "profile_problems",
    "profiles_for",
    "required_patches",
    "size_bytes",
]
