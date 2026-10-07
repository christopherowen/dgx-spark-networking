"""TileLang reference kernels for the fused halves of the one-shot all-reduce.

The send kernel stages its input into the send slot and
rings the doorbell; the receive kernel waits for the peers, writes the reduced
output and advances the epoch. Together they store the bits of the standalone
all-reduce. They call only the header's public ``roce_fused_*`` functions, so
they double as the reference for a producer or consumer kernel written
elsewhere: the GPU test runs every pairing of these kernels with the standalone
paths. Both take the runtime's fused header (``device_header()``), which bakes
its region layout and counter indices in, so one launcher serves one runtime.

Importing this module needs nothing; the ``get_*`` functions import TileLang.
"""

# No postponed annotations: TileLang evaluates the prim_func's annotations
# when the function is defined, and the symbolic shapes must be visible then.

import functools
import hashlib
import logging
import time
from typing import Callable

from . import trace as _trace
from ._device import PACK_BYTES
from ._freeze import raise_if_kernel_resolution_frozen
from ._oneshot_tilelang import _DTYPE_PACK_ELEMS, _pass_configs, _resident_blocks

logger = logging.getLogger(__name__)

_FUSED_REDUCE_FUNCTION = {"float32": "roce_fused_reduce_pack_f32", "float16": "roce_fused_reduce_pack_f16",
                          "bfloat16": "roce_fused_reduce_pack_bf16"}


def _header_id(header: str) -> str:
    return hashlib.sha256(header.encode()).hexdigest()[:12]


def _build_send(dtype_name: str, header: str, threads: int, traced: bool):
    """The send half: stage into the send slot, the last block rings the doorbell."""
    import tilelang
    import tilelang.language as T

    resident = _resident_blocks(threads)
    pack_words = PACK_BYTES // 4
    w_doorbell = _trace.W_DOORBELL

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_in, n_counters = T.dynamic("n_in, n_counters")

        @T.prim_func
        def oneshot_fused_send(
            inp: T.Tensor((n_in,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            size_packs: T.int32,
            nbytes: T.int32,
            grid_x: T.int32,
            trace_base: T.int64,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(header)
                tx = T.get_thread_binding()
                seq = T.alloc_var(T.uint32)
                seq = T.call_extern("roce_fused_seq", T.address_of(counters[0]), dtype=T.uint32)
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                poisoned = T.alloc_var(T.uint32)
                poisoned = T.call_extern("roce_fused_poisoned", T.address_of(counters[0]), dtype=T.uint32)
                if poisoned == T.uint32(0):
                    if traced:
                        if bx == 0:
                            if tx == 0:
                                T.call_extern("roce_trace_begin", trace_base, seq, T.cast(nbytes, T.uint32),
                                              T.cast(grid_x, T.uint32), dtype="handle")
                    count = T.alloc_var(T.int32)
                    count = T.max(0, (size_packs - index + stride - 1) // stride)
                    for i in T.serial(count):
                        pack = index + i * stride
                        T.call_extern("roce_copy_pack", T.address_of(inp[pack * pack_words]),
                                      T.call_extern("roce_fused_send_slot_at", region_base, seq,
                                                    T.cast(pack, T.int64) * PACK_BYTES, dtype="handle"), dtype="handle")
                    T.sync_threads()
                    if tx == 0:
                        rang = T.alloc_var(T.uint32)
                        rang = T.call_extern("roce_fused_send_commit", T.address_of(counters[0]), T.cast(grid_x, T.uint32),
                                             region_base, T.cast(nbytes, T.uint32), seq, dtype=T.uint32)
                        if traced:
                            if rang != T.uint32(0):
                                T.call_extern("roce_trace_stamp", trace_base, seq, w_doorbell, dtype="handle")

        return oneshot_fused_send

    return program


def _build_receive(dtype_name: str, header: str, threads: int, traced: bool):
    """The receive half: wait, reduce into the output, the last block advances the epoch."""
    import tilelang
    import tilelang.language as T

    reduce_pack = _FUSED_REDUCE_FUNCTION[dtype_name]
    resident = _resident_blocks(threads)
    w_wait_done, w_end = _trace.W_WAIT_DONE, _trace.W_END

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_out, n_counters = T.dynamic("n_out, n_counters")

        @T.prim_func
        def oneshot_fused_receive(
            out: T.Tensor((n_out,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            size_packs: T.int32,
            grid_x: T.int32,
            trace_base: T.int64,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(header)
                tx = T.get_thread_binding()
                seq = T.alloc_var(T.uint32)
                seq = T.call_extern("roce_fused_seq", T.address_of(counters[0]), dtype=T.uint32)
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                poisoned = T.alloc_var(T.uint32)
                poisoned = T.call_extern("roce_fused_poisoned", T.address_of(counters[0]), dtype=T.uint32)
                if poisoned == T.uint32(0):
                    if traced:
                        T.call_extern("roce_fused_wait", tx, region_base, seq, T.address_of(counters[0]),
                                      T.call_extern("roce_trace_record_if", trace_base, seq, T.cast(bx == 0, T.int32), dtype="handle"),
                                      dtype="handle")
                    else:
                        T.call_extern("roce_fused_wait", tx, region_base, seq, T.address_of(counters[0]), dtype="handle")
                    T.sync_threads()
                    if traced:
                        if bx == 0:
                            if tx == 0:
                                T.call_extern("roce_trace_stamp", trace_base, seq, w_wait_done, dtype="handle")
                    failed = T.alloc_var(T.uint32)
                    failed = T.call_extern("roce_fused_poisoned", T.address_of(counters[0]), dtype=T.uint32)
                    if failed == T.uint32(0):
                        count = T.alloc_var(T.int32)
                        count = T.max(0, (size_packs - index + stride - 1) // stride)
                        for i in T.serial(count):
                            pack = index + i * stride
                            T.call_extern(reduce_pack, region_base, seq, T.address_of(out[0]),
                                          T.cast(pack, T.int64) * PACK_BYTES, dtype="handle")
                    T.call_extern("roce_fence_sc_gpu", dtype="handle")
                    T.sync_threads()
                    if tx == 0:
                        last = T.alloc_var(T.uint32)
                        last = T.call_extern("roce_fused_receive_tail", T.address_of(counters[0]), T.cast(grid_x, T.uint32),
                                             region_base, seq, dtype=T.uint32)
                        if traced:
                            if last != T.uint32(0):
                                T.call_extern("roce_trace_stamp", trace_base, seq, w_end, dtype="handle")

        return oneshot_fused_receive

    return program


@functools.cache
def get_send_launcher(dtype_name: str, header: str, threads: int, trace: bool = False) -> Callable[..., None]:
    """Compile the send launcher for one runtime's fused header once and return it."""
    if dtype_name not in _DTYPE_PACK_ELEMS:
        raise ValueError(f"unsupported RoCE one-shot dtype {dtype_name!r}")
    cache_key = ("send", dtype_name, _header_id(header), int(threads), bool(trace))
    raise_if_kernel_resolution_frozen("tilelang.jit", target="oneshot_fused_send", cache_key=cache_key)
    started = time.monotonic()
    kernel = _build_send(dtype_name, header, int(threads), bool(trace))()
    logger.info("compiled tilelang oneshot.fused_send %s in %.1f s", cache_key, time.monotonic() - started)

    def run(inp, counters, region_base: int, size_packs: int, nbytes: int, grid_x: int, trace_base: int) -> None:
        """Launch on the current stream (bound through the tensor arguments)."""
        kernel(inp, counters, int(region_base), int(size_packs), int(nbytes), int(grid_x), int(trace_base))

    return run


@functools.cache
def get_receive_launcher(dtype_name: str, header: str, threads: int, trace: bool = False) -> Callable[..., None]:
    """Compile the receive launcher for one runtime's fused header once and return it."""
    if dtype_name not in _FUSED_REDUCE_FUNCTION:
        raise ValueError(f"unsupported RoCE one-shot dtype {dtype_name!r}")
    cache_key = ("receive", dtype_name, _header_id(header), int(threads), bool(trace))
    raise_if_kernel_resolution_frozen("tilelang.jit", target="oneshot_fused_receive", cache_key=cache_key)
    started = time.monotonic()
    kernel = _build_receive(dtype_name, header, int(threads), bool(trace))()
    logger.info("compiled tilelang oneshot.fused_receive %s in %.1f s", cache_key, time.monotonic() - started)

    def run(out, counters, region_base: int, size_packs: int, grid_x: int, trace_base: int) -> None:
        """Launch on the current stream (bound through the tensor arguments)."""
        kernel(out, counters, int(region_base), int(size_packs), int(grid_x), int(trace_base))

    return run


__all__ = ["get_receive_launcher", "get_send_launcher"]
