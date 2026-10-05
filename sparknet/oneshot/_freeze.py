"""Process-wide kernel-resolution freeze, shared by every kernel family.

Once a serving engine has warmed every launcher it needs, any further
compilation is a bug and raises instead of stalling a step. This module has
no heavy imports so the CLI and the probe planner can import the package
without a GPU stack.
"""

from __future__ import annotations

import threading

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


__all__ = [
    "KernelResolutionFrozenError",
    "freeze_kernel_resolution",
    "kernel_resolution_frozen",
    "raise_if_kernel_resolution_frozen",
    "thaw_kernel_resolution",
]
