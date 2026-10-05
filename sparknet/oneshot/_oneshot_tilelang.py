"""TileLang kernel for the RoCE one-shot all-reduce.

The same five phases as ``_oneshot_cute.py`` (stage, doorbell, wait, reduce
in fixed rank order, advance the epoch), generated as CUDA source by
TileLang with the protocol steps in ``_device.py``. Message size and launch
grid are runtime scalars; each power-of-two grid has its own staging and
tail counters, as in the CuTe kernel. One launcher is compiled per dtype and
geometry; ``run`` takes the CuTe launcher's positional arguments.

Importing this module needs nothing; ``get_launcher`` imports TileLang.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from ._device import PACK_BYTES, render_source
from ._freeze import raise_if_kernel_resolution_frozen

logger = logging.getLogger(__name__)

_DTYPE_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}
_REDUCE_FUNCTION = {"float32": "roce_reduce_pack_f32", "float16": "roce_reduce_pack_f16", "bfloat16": "roce_reduce_pack_bf16"}
_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


def _lanes(hca_count: int, ring4: bool) -> tuple[int, int]:
    """(flag lanes per peer, neighbour lanes): ring4 doubles the lanes of the opposite rank."""
    neighbor_lanes = int(hca_count) if ring4 else (2 if hca_count == 4 else int(hca_count))
    return int(hca_count) * (2 if ring4 else 1), neighbor_lanes


def _pass_configs():
    import tilelang

    key = tilelang.PassConfigKey
    return {key.TL_DISABLE_THREAD_STORAGE_SYNC: True, key.TL_DISABLE_SAFE_MEMORY_ACCESS: True}


def _build(dtype_name: str, world_size: int, rank: int, threads: int, slots: int, flag_stride: int,
           hca_count: int, ring4: bool):
    """The TileLang program for one specialization."""
    import tilelang
    import tilelang.language as T

    lanes, neighbor_lanes = _lanes(hca_count, ring4)
    source = render_source(world_size=world_size, rank=rank, slots=slots, flag_stride=flag_stride,
                           hca_count=lanes, neighbor_lanes=neighbor_lanes)
    reduce_pack = _REDUCE_FUNCTION[dtype_name]
    pack_bytes = PACK_BYTES

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        @T.prim_func
        def oneshot_allreduce(
            input_base: T.int64,
            output_base: T.int64,
            size_packs: T.int32,
            nbytes: T.int32,
            recv_base: T.int64,
            flag_base: T.int64,
            send_base: T.int64,
            ctrl_base: T.int64,
            slot_bytes: T.int64,
            epoch_ptr: T.int64,
            stage_counter_ptr: T.int64,
            tail_counter_ptr: T.int64,
            poison_ptr: T.int64,
            spin_limit: T.int32,
            grid_x: T.int32,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.import_source(source)
                tx = T.get_thread_binding()
                # Every block reads the epoch before any block can advance it:
                # the advance happens only after all blocks arrived at the tail.
                epoch = T.alloc_var(T.uint32)
                epoch = T.call_extern("roce_ld_relaxed_gpu_u32", epoch_ptr, dtype=T.uint32)
                seq = T.alloc_var(T.uint32)
                seq = epoch + T.uint32(1)
                slot = T.alloc_var(T.int64)
                slot = T.cast(seq & T.uint32(1), T.int64)
                send_slot = T.alloc_var(T.int64)
                send_slot = send_base + slot * slot_bytes
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                # A recorded timeout poisons the runtime: later launches do
                # nothing so the host sees the failure without another spin
                # limit per op.
                poisoned = T.alloc_var(T.uint32)
                poisoned = T.call_extern("roce_ld_relaxed_gpu_u32", poison_ptr, dtype=T.uint32)
                if poisoned == T.uint32(0):
                    # 1. stage the input into the pinned send slot
                    count = T.alloc_var(T.int32)
                    count = T.max(0, (size_packs - index + stride - 1) // stride)
                    for i in T.serial(count):
                        pack = T.cast(index + i * stride, T.int64)
                        T.call_extern("roce_copy_pack", input_base + pack * pack_bytes, send_slot + pack * pack_bytes,
                                      dtype="handle")
                    T.sync_threads()
                    # 2. the last block to finish staging rings the proxy doorbell
                    if tx == 0:
                        T.call_extern("roce_doorbell", stage_counter_ptr, T.cast(grid_x, T.uint32), ctrl_base,
                                      T.cast(nbytes, T.uint32), seq, dtype="handle")
                    # 3. wait for every peer's payload-stripe flags
                    T.call_extern("roce_wait_flags", tx, flag_base, seq, T.cast(spin_limit, T.uint32), ctrl_base,
                                  poison_ptr, dtype="handle")
                    T.sync_threads()
                    # A wait that timed out in this block leaves the peer slot
                    # unreliable: skip the data phase.
                    failed = T.alloc_var(T.uint32)
                    failed = T.call_extern("roce_ld_relaxed_gpu_u32", poison_ptr, dtype=T.uint32)
                    if failed == T.uint32(0):
                        # 4. reduce in fixed rank order so every rank stores identical bits
                        for i in T.serial(count):
                            pack = T.cast(index + i * stride, T.int64)
                            T.call_extern(reduce_pack, input_base, recv_base, slot_bytes, seq, output_base,
                                          pack * pack_bytes, dtype="handle")
                    # 5. the last block to finish reduction publishes the next epoch
                    T.call_extern("roce_fence_sc_gpu", dtype="handle")
                    T.sync_threads()
                    if tx == 0:
                        T.call_extern("roce_tail", tail_counter_ptr, T.cast(grid_x, T.uint32), ctrl_base, epoch_ptr,
                                      seq, dtype="handle")

        return oneshot_allreduce

    return program


def _process_key(dtype_name, world_size, rank, threads, slots, flag_stride, hca_count, device_index, ring4=False):
    return (str(dtype_name), int(world_size), int(rank), int(threads), int(slots), int(flag_stride), int(hca_count),
            int(device_index), bool(ring4))


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
) -> Callable[..., None]:
    """Compile the launcher for the key once and return it."""
    if dtype_name not in _DTYPE_PACK_ELEMS:
        raise ValueError(f"unsupported RoCE one-shot dtype {dtype_name!r}")
    lanes, _ = _lanes(hca_count, ring4)
    if int(threads) < int(world_size) * lanes:
        raise ValueError(
            "RoCE kernels need threads >= world_size * hca_count (one thread per stripe flag), "
            f"got threads={threads} world_size={world_size} hca_count={lanes}"
        )
    process_key = _process_key(dtype_name, world_size, rank, threads, slots, flag_stride, hca_count, device_index, ring4)
    cache_key = process_key[:-2] + (bool(ring4),)
    raise_if_kernel_resolution_frozen("tilelang.jit", target="oneshot_allreduce", cache_key=cache_key)
    started = time.monotonic()
    kernel = _build(dtype_name, world_size, rank, threads, slots, flag_stride, hca_count, ring4)()
    logger.info("compiled tilelang oneshot.allreduce %s in %.1f s", cache_key, time.monotonic() - started)

    def run(
        input_address: int,
        output_address: int,
        size_packs: int,
        nbytes: int,
        recv_base: int,
        flag_base: int,
        send_base: int,
        ctrl_base: int,
        slot_bytes: int,
        epoch_address: int,
        stage_counter_address: int,
        tail_counter_address: int,
        poison_address: int,
        spin_limit: int,
        grid_x: int,
    ) -> None:
        """Launch the compiled kernel with runtime scalar arguments on the current stream."""
        kernel(
            int(input_address), int(output_address), int(size_packs), int(nbytes), int(recv_base), int(flag_base),
            int(send_base), int(ctrl_base), int(slot_bytes), int(epoch_address), int(stage_counter_address),
            int(tail_counter_address), int(poison_address), int(spin_limit), int(grid_x),
        )

    _PREPARED_LAUNCHERS.add(process_key)
    return run


__all__ = ["PACK_BYTES", "get_launcher", "is_launcher_prepared"]
