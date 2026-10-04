"""Kernel compilation for the RoCEnante CuTe DSL launchers.

A thin layer over ``cutlass.cute.compile``: the DSL keeps its own on-disk JIT
cache (``CUTE_DSL_CACHE_DIR``), so serving images warm it at build time and
production boots compile nothing. A process-wide freeze mirrors b12x's
``freeze_kernel_resolution``: once a serving engine has warmed every shape it
needs, any further compilation is a bug and raises instead of stalling a step.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_FROZEN = False
_FREEZE_REASON: str | None = None


class KernelResolutionFrozenError(RuntimeError):
    """Raised when a kernel would be compiled after resolution was frozen."""


def freeze_kernel_resolution(reason: str | None = None) -> None:
    """Refuse every later compilation; call after the serving warm-up."""
    global _FROZEN, _FREEZE_REASON
    with _STATE_LOCK:
        _FROZEN = True
        _FREEZE_REASON = reason


def thaw_kernel_resolution() -> None:
    """Allow compilation again (tests and tooling only)."""
    global _FROZEN, _FREEZE_REASON
    with _STATE_LOCK:
        _FROZEN = False
        _FREEZE_REASON = None


def kernel_resolution_frozen() -> bool:
    with _STATE_LOCK:
        return _FROZEN


def raise_if_kernel_resolution_frozen(
    kind: str, *, target: object | None = None, cache_key: object | None = None
) -> None:
    with _STATE_LOCK:
        frozen, reason = _FROZEN, _FREEZE_REASON
    if not frozen:
        return
    details = [f"sparknet kernel resolution is frozen; refusing {kind}"]
    if target is not None:
        details.append(f"target={type(target).__name__}")
    if cache_key is not None:
        details.append(f"key={cache_key!r}")
    if reason is not None:
        details.append(f"reason={reason}")
    details.append("prepare this launcher before freezing kernel resolution")
    raise KernelResolutionFrozenError("; ".join(details))


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
