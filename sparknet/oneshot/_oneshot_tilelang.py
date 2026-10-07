"""TileLang kernel for the RoCE one-shot all-reduce.

The same five phases as ``_oneshot_cute.py`` (stage, doorbell, wait, reduce
in fixed rank order, advance the epoch), generated as CUDA source by
TileLang with the protocol steps in ``_device.py``. The kernel's arguments
are the device buffers (the input and output as 4-byte words, the
counters tensor), which bind the launch to the current device and stream,
plus the pinned region's base address, offsets, indices and sizes as scalars.
Message size and launch grid are runtime scalars; each power-of-two grid has
its own staging and tail counters, as in the CuTe kernel. One launcher is
compiled per dtype, geometry and trace choice; ``run`` takes the
family-neutral ``Launch``. The traced kernel writes the CuTe kernels' trace
words (``sparknet.oneshot.trace``); the untraced one compiles none of it.

Importing this module needs nothing; ``get_launcher`` imports TileLang.
"""

# No postponed annotations: TileLang evaluates the prim_func's annotations
# when the function is defined, and the symbolic shapes must be visible then.

import functools
import logging
import time
from typing import Callable

from . import trace as _trace
from ._device import PACK_BYTES, render_source
from ._freeze import raise_if_kernel_resolution_frozen
from ._kernels import Launch

logger = logging.getLogger(__name__)

_DTYPE_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}
_REDUCE_FUNCTION = {"float32": "roce_reduce_pack_f32", "float16": "roce_reduce_pack_f16", "bfloat16": "roce_reduce_pack_bf16"}
_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


def _lanes(hca_count: int, ring4: bool) -> tuple[int, int]:
    """(flag lanes per peer, neighbour lanes): ring4 doubles the lanes of the opposite rank."""
    neighbor_lanes = int(hca_count) if ring4 else (2 if hca_count == 4 else int(hca_count))
    return int(hca_count) * (2 if ring4 else 1), neighbor_lanes


def _resident_blocks(threads: int) -> int:
    """Blocks of ``threads`` an SM can hold by its thread limit (3 of 512 on the GB10).

    The kernels declare this as their minimum blocks per SM, which bounds the
    compiler's register budget to the full-occupancy figure (40 registers at 512
    threads, as the CuTe kernels compile). Left to itself nvcc spent 54 to 56
    registers, which halves the blocks per SM, and in a decode step, beside the
    model's L2 prefetch and kernels, every launch then took about 15 us longer
    (2026-10-05 decode profiles, docs/oneshot.md); in isolation the two were equal.
    """
    import torch

    per_sm = torch.cuda.get_device_properties(torch.cuda.current_device()).max_threads_per_multi_processor
    return max(1, int(per_sm) // int(threads))


def _pass_configs():
    import tilelang

    key = tilelang.PassConfigKey
    return {key.TL_DISABLE_THREAD_STORAGE_SYNC: True, key.TL_DISABLE_SAFE_MEMORY_ACCESS: True}


def _build(dtype_name: str, world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
           hca_count: int, ring4: bool, traced: bool = False):
    """The TileLang program for one specialization."""
    import tilelang
    import tilelang.language as T

    lanes, neighbor_lanes = _lanes(hca_count, ring4)
    source = render_source(world_size=world_size, rank=rank, slots=slots, flag_stride=flag_stride,
                           hca_count=lanes, neighbor_lanes=neighbor_lanes,
                           trace=_trace.device_defines() if traced else None)
    reduce_pack = _REDUCE_FUNCTION[dtype_name]
    resident = _resident_blocks(threads)
    pack_words = PACK_BYTES // 4
    w_doorbell, w_wait_done, w_end = _trace.W_DOORBELL, _trace.W_WAIT_DONE, _trace.W_END

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_in, n_out, n_counters = T.dynamic("n_in, n_out, n_counters")

        @T.prim_func
        def oneshot_allreduce(
            inp: T.Tensor((n_in,), T.int32),
            out: T.Tensor((n_out,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            size_packs: T.int32,
            nbytes: T.int32,
            recv_off: T.int32,
            flag_off: T.int32,
            send_off: T.int32,
            ctrl_off: T.int32,
            slot_bytes: T.int32,
            stage_index: T.int32,
            tail_index: T.int32,
            poison_index: T.int32,
            spin_limit: T.int32,
            grid_x: T.int32,
            trace_base: T.int64,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(source)
                tx = T.get_thread_binding()
                # Every block reads the epoch before any block can advance it:
                # the advance happens only after all blocks arrived at the tail.
                epoch = T.alloc_var(T.uint32)
                epoch = T.call_extern("roce_ld_relaxed_gpu_u32", T.address_of(counters[0]), dtype=T.uint32)
                seq = T.alloc_var(T.uint32)
                seq = epoch + T.uint32(1)
                slot = T.alloc_var(T.int32)
                slot = T.cast(seq & T.uint32(1), T.int32)
                send_slot = T.alloc_var(T.int32)
                send_slot = send_off + slot * slot_bytes
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                # A recorded timeout poisons the runtime: later launches do
                # nothing so the host sees the failure without another spin
                # limit per op.
                poisoned = T.alloc_var(T.uint32)
                poisoned = T.call_extern("roce_ld_relaxed_gpu_u32", T.address_of(counters[poison_index]), dtype=T.uint32)
                if poisoned == T.uint32(0):
                    if traced:
                        if bx == 0:
                            if tx == 0:
                                T.call_extern("roce_trace_begin", trace_base, seq, T.cast(nbytes, T.uint32),
                                              T.cast(grid_x, T.uint32), dtype="handle")
                    # 1. stage the input into the pinned send slot
                    count = T.alloc_var(T.int32)
                    count = T.max(0, (size_packs - index + stride - 1) // stride)
                    for i in T.serial(count):
                        pack = index + i * stride
                        T.call_extern("roce_copy_pack", T.address_of(inp[pack * pack_words]),
                                      T.call_extern("roce_ptr", region_base + T.cast(send_slot + pack * PACK_BYTES, T.int64), dtype="handle"), dtype="handle")
                    T.sync_threads()
                    # 2. the last block to finish staging rings the proxy doorbell
                    if tx == 0:
                        rang = T.alloc_var(T.uint32)
                        rang = T.call_extern("roce_doorbell", T.address_of(counters[stage_index]), T.cast(grid_x, T.uint32),
                                             T.call_extern("roce_ptr", region_base + T.cast(ctrl_off, T.int64), dtype="handle"), T.cast(nbytes, T.uint32), seq, dtype=T.uint32)
                        if traced:
                            if rang != T.uint32(0):
                                T.call_extern("roce_trace_stamp", trace_base, seq, w_doorbell, dtype="handle")
                    # 3. wait for every peer's payload-stripe flags
                    if traced:
                        T.call_extern("roce_wait_flags", tx, T.call_extern("roce_ptr", region_base + T.cast(flag_off, T.int64), dtype="handle"), seq, T.cast(spin_limit, T.uint32),
                                      T.call_extern("roce_ptr", region_base + T.cast(ctrl_off, T.int64), dtype="handle"), T.address_of(counters[poison_index]),
                                      T.call_extern("roce_trace_record_if", trace_base, seq, T.cast(bx == 0, T.int32), dtype="handle"), dtype="handle")
                    else:
                        T.call_extern("roce_wait_flags", tx, T.call_extern("roce_ptr", region_base + T.cast(flag_off, T.int64), dtype="handle"), seq, T.cast(spin_limit, T.uint32),
                                      T.call_extern("roce_ptr", region_base + T.cast(ctrl_off, T.int64), dtype="handle"), T.address_of(counters[poison_index]), dtype="handle")
                    T.sync_threads()
                    if traced:
                        if bx == 0:
                            if tx == 0:
                                T.call_extern("roce_trace_stamp", trace_base, seq, w_wait_done, dtype="handle")
                    # A wait that timed out in this block leaves the peer slot
                    # unreliable: skip the data phase.
                    failed = T.alloc_var(T.uint32)
                    failed = T.call_extern("roce_ld_relaxed_gpu_u32", T.address_of(counters[poison_index]), dtype=T.uint32)
                    if failed == T.uint32(0):
                        # 4. reduce in fixed rank order so every rank stores identical bits
                        for i in T.serial(count):
                            pack = index + i * stride
                            T.call_extern(reduce_pack, T.address_of(inp[0]), T.call_extern("roce_ptr", region_base + T.cast(recv_off, T.int64), dtype="handle"),
                                          T.cast(slot_bytes, T.int64), seq, T.address_of(out[0]),
                                          T.cast(pack, T.int64) * PACK_BYTES, dtype="handle")
                    # 5. the last block to finish reduction publishes the next epoch
                    T.call_extern("roce_fence_sc_gpu", dtype="handle")
                    T.sync_threads()
                    if tx == 0:
                        last = T.alloc_var(T.uint32)
                        last = T.call_extern("roce_tail", T.address_of(counters[tail_index]), T.cast(grid_x, T.uint32),
                                             T.call_extern("roce_ptr", region_base + T.cast(ctrl_off, T.int64), dtype="handle"), T.address_of(counters[0]), seq, dtype=T.uint32)
                        if traced:
                            if last != T.uint32(0):
                                T.call_extern("roce_trace_stamp", trace_base, seq, w_end, dtype="handle")

        return oneshot_allreduce

    return program


def _process_key(dtype_name, world_size, rank, threads, slots, flag_stride, hca_count, device_index, ring4=False,
                 trace=False):
    return (str(dtype_name), int(world_size), int(rank), int(threads), int(slots), int(flag_stride), int(hca_count),
            int(device_index), bool(ring4), bool(trace))


def is_launcher_prepared(*key) -> bool:
    """Return whether this exact process-local launcher is already compiled."""
    return _process_key(*key) in _PREPARED_LAUNCHERS


@functools.cache
def get_launcher(
    dtype_name: str,
    world_size: int,
    rank: int,
    threads: int,
    slots: int,
    flag_stride: int,
    hca_count: int,
    device_index: int,
    ring4: bool = False,
    trace: bool = False,
) -> Callable[[Launch], None]:
    """Compile the launcher for the key once and return it."""
    if dtype_name not in _DTYPE_PACK_ELEMS:
        raise ValueError(f"unsupported RoCE one-shot dtype {dtype_name!r}")
    lanes, _ = _lanes(hca_count, ring4)
    if int(threads) < int(world_size) * lanes:
        raise ValueError(
            "RoCE kernels need threads >= world_size * hca_count (one thread per stripe flag), "
            f"got threads={threads} world_size={world_size} hca_count={lanes}"
        )
    if trace and int(world_size) * lanes > _trace.MAX_FLAG_WORDS:
        raise ValueError(f"the trace record holds {_trace.MAX_FLAG_WORDS} peer lanes, got {int(world_size) * lanes}")
    process_key = _process_key(dtype_name, world_size, rank, threads, slots, flag_stride, hca_count, device_index, ring4,
                               trace)
    cache_key = process_key[:7] + process_key[8:]
    raise_if_kernel_resolution_frozen("tilelang.jit", target="oneshot_allreduce", cache_key=cache_key)
    started = time.monotonic()
    kernel = _build(dtype_name, world_size, rank, threads, slots, flag_stride, hca_count, ring4, bool(trace))()
    logger.info("compiled tilelang oneshot.allreduce %s in %.1f s", cache_key, time.monotonic() - started)

    def run(launch: Launch) -> None:
        """Launch the compiled kernel on the current stream (bound through the tensor arguments)."""
        kernel(
            launch.input, launch.output, launch.counters,
            int(launch.region.data_ptr()), int(launch.size_packs), int(launch.nbytes), int(launch.recv_off),
            int(launch.flag_off), int(launch.send_off), int(launch.ctrl_off), int(launch.slot_bytes),
            int(launch.stage_index), int(launch.tail_index), int(launch.poison_index), int(launch.spin_limit),
            int(launch.grid_x), int(launch.trace_base),
        )

    _PREPARED_LAUNCHERS.add(process_key)
    return run


__all__ = ["PACK_BYTES", "get_launcher", "is_launcher_prepared"]
