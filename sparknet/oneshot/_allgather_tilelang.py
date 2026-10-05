"""TileLang kernel for the RoCE one-shot all-gather.

Same transport and protocol as the all-reduce with the reduction replaced by
a strided copy that writes the concatenated output directly (dim-0 and
last-dim concatenation; see ``_allgather_cute.py``). Generated as CUDA source
by TileLang with the protocol steps in ``_device.py``; ``run`` takes the CuTe
launcher's positional arguments.
"""

# No postponed annotations: TileLang evaluates the prim_func's annotations
# when the function is defined, and the symbolic shape must be visible then.

import functools
import logging
import time
from typing import Callable

from ._device import PACK_BYTES, render_source
from ._freeze import raise_if_kernel_resolution_frozen
from ._oneshot_tilelang import _lanes, _pass_configs

logger = logging.getLogger(__name__)

_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


def _build(world_size: int, rank: int, threads: int, slots: int, flag_stride: int, hca_count: int, ring4: bool):
    import tilelang
    import tilelang.language as T

    lanes, neighbor_lanes = _lanes(hca_count, ring4)
    source = render_source(world_size=world_size, rank=rank, slots=slots, flag_stride=flag_stride,
                           hca_count=lanes, neighbor_lanes=neighbor_lanes)
    pack_bytes = PACK_BYTES

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        # The counters tensor anchors the launch: TileLang's tvm_ffi backend takes the
        # device and the current stream from the tensor arguments (through the DLPack
        # exchange), and a scalar-only kernel would otherwise launch on the default
        # stream, outside a CUDA graph capture.
        anchor_words = T.dynamic("anchor_words")

        @T.prim_func
        def oneshot_allgather(
            anchor: T.Tensor((anchor_words,), T.int32),
            input_base: T.int64,
            output_base: T.int64,
            shard_packs: T.int32,
            nbytes: T.int32,
            row_packs: T.int32,
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
                poisoned = T.alloc_var(T.uint32)
                poisoned = T.call_extern("roce_ld_relaxed_gpu_u32", poison_ptr, dtype=T.uint32)
                if poisoned == T.uint32(0):
                    # 1. stage the local shard into the pinned send slot
                    count = T.alloc_var(T.int32)
                    count = T.max(0, (shard_packs - index + stride - 1) // stride)
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
                    failed = T.alloc_var(T.uint32)
                    failed = T.call_extern("roce_ld_relaxed_gpu_u32", poison_ptr, dtype=T.uint32)
                    if failed == T.uint32(0):
                        # 4. concatenate: shard s occupies column block s of every output row
                        for i in T.serial(count):
                            T.call_extern("roce_gather_pack", input_base, recv_base, slot_bytes, seq, index + i * stride,
                                          row_packs, output_base, dtype="handle")
                    # 5. the last block to finish publishes the next epoch
                    T.call_extern("roce_fence_sc_gpu", dtype="handle")
                    T.sync_threads()
                    if tx == 0:
                        T.call_extern("roce_tail", tail_counter_ptr, T.cast(grid_x, T.uint32), ctrl_base, epoch_ptr,
                                      seq, dtype="handle")

        return oneshot_allgather

    return program


def _process_key(world_size, rank, threads, slots, flag_stride, hca_count, device_index, ring4=False):
    return (int(world_size), int(rank), int(threads), int(slots), int(flag_stride), int(hca_count), int(device_index),
            bool(ring4))


def is_launcher_prepared(*key) -> bool:
    """True when the launcher for ``key`` is already compiled."""
    return _process_key(*key) in _PREPARED_LAUNCHERS


@functools.cache
def get_launcher(
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
    lanes, _ = _lanes(hca_count, ring4)
    if int(threads) < int(world_size) * lanes:
        raise ValueError(
            "RoCE kernels need threads >= world_size * hca_count (one thread per stripe flag), "
            f"got threads={threads} world_size={world_size} hca_count={lanes}"
        )
    process_key = _process_key(world_size, rank, threads, slots, flag_stride, hca_count, device_index, ring4)
    cache_key = process_key[:-2] + (bool(ring4),)
    raise_if_kernel_resolution_frozen("tilelang.jit", target="oneshot_allgather", cache_key=cache_key)
    started = time.monotonic()
    kernel = _build(world_size, rank, threads, slots, flag_stride, hca_count, ring4)()
    logger.info("compiled tilelang oneshot.allgather %s in %.1f s", cache_key, time.monotonic() - started)

    def run(
        input_address: int,
        output_address: int,
        shard_packs: int,
        nbytes: int,
        row_packs: int,
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
        *,
        anchor,
    ) -> None:
        """Launch the compiled kernel on the current stream; ``anchor`` is the runtime's counters tensor."""
        kernel(
            anchor,
            int(input_address), int(output_address), int(shard_packs), int(nbytes), int(row_packs), int(recv_base),
            int(flag_base), int(send_base), int(ctrl_base), int(slot_bytes), int(epoch_address),
            int(stage_counter_address), int(tail_counter_address), int(poison_address), int(spin_limit), int(grid_x),
        )

    _PREPARED_LAUNCHERS.add(process_key)
    return run


__all__ = ["PACK_BYTES", "get_launcher", "is_launcher_prepared"]
