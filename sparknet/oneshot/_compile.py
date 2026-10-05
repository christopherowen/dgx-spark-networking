"""Kernel compilation for the one-shot CuTe DSL launchers.

A thin layer over ``cutlass.cute.compile``: the DSL keeps its own on-disk JIT
cache (``CUTE_DSL_CACHE_DIR``), so serving images warm it at build time and
production boots compile nothing. The process-wide freeze lives in
``_freeze`` and is shared with the TileLang family.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch

from ._freeze import (
    KernelResolutionFrozenError,
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    raise_if_kernel_resolution_frozen,
    thaw_kernel_resolution,
)

logger = logging.getLogger(__name__)


def current_cuda_stream() -> cuda.CUstream:
    """The current torch CUDA stream as a CUDA driver stream handle."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def compile_kernel(
    launch: Any, *args: Any, name: str, cache_key: tuple[object, ...]
) -> Callable[..., Any]:
    """Compile ``launch`` for the traced argument types and return the executor."""
    started = time.monotonic()
    compiled = cute.compile(launch, *args)
    logger.info(
        "compiled %s %s in %.1f s", name, cache_key, time.monotonic() - started
    )
    return compiled


__all__ = [
    "KernelResolutionFrozenError",
    "compile_kernel",
    "current_cuda_stream",
    "freeze_kernel_resolution",
    "kernel_resolution_frozen",
    "raise_if_kernel_resolution_frozen",
    "thaw_kernel_resolution",
]
