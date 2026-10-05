"""TileLang kernel for the RoCE one-shot all-gather.

Same transport and protocol as the all-reduce with the reduction replaced by
a strided copy that writes the concatenated output directly (dim-0 and
last-dim concatenation; see ``_allgather_cute.py``). Generated as CUDA source
by TileLang with the protocol steps in ``_device.py``; the buffers are the
kernel's arguments and ``run`` takes the family-neutral ``Launch``.
"""

# No postponed annotations: TileLang evaluates the prim_func's annotations
# when the function is defined, and the symbolic shapes must be visible then.

import functools
import logging
import time
from typing import Callable

from ._device import PACK_BYTES, render_source
from ._freeze import raise_if_kernel_resolution_frozen
from ._kernels import Launch
from ._oneshot_tilelang import _lanes, _pass_configs, _resident_blocks

logger = logging.getLogger(__name__)

_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


def _build(world_size: int, rank: int, threads: int, slots: int, flag_stride: int, hca_count: int, ring4: bool):
    import tilelang
    import tilelang.language as T

    lanes, neighbor_lanes = _lanes(hca_count, ring4)
    source = render_source(world_size=world_size, rank=rank, slots=slots, flag_stride=flag_stride,
                           hca_count=lanes, neighbor_lanes=neighbor_lanes)
    resident = _resident_blocks(threads)

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_in, n_out, n_counters = T.dynamic("n_in, n_out, n_counters")

        @T.prim_func
        def oneshot_allgather(
            inp: T.Tensor((n_in,), T.int32),
            out: T.Tensor((n_out,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            shard_packs: T.int32,
            nbytes: T.int32,
            row_packs: T.int32,
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
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(source)
                tx = T.get_thread_binding()
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
                poisoned = T.alloc_var(T.uint32)
                poisoned = T.call_extern("roce_ld_relaxed_gpu_u32", T.address_of(counters[poison_index]), dtype=T.uint32)
                if poisoned == T.uint32(0):
                    # 1. stage the local shard into the pinned send slot
                    T.call_extern("roce_stage", T.address_of(inp[0]),
                                  T.call_extern("roce_ptr", region_base + T.cast(send_slot, T.int64), dtype="handle"),
                                  shard_packs, index, stride, dtype="handle")
                    T.sync_threads()
                    # 2. the last block to finish staging rings the proxy doorbell
                    if tx == 0:
                        T.call_extern("roce_doorbell", T.address_of(counters[stage_index]), T.cast(grid_x, T.uint32),
                                      T.call_extern("roce_ptr", region_base + T.cast(ctrl_off, T.int64), dtype="handle"), T.cast(nbytes, T.uint32), seq, dtype="handle")
                    # 3. wait for every peer's payload-stripe flags
                    T.call_extern("roce_wait_flags", tx, T.call_extern("roce_ptr", region_base + T.cast(flag_off, T.int64), dtype="handle"), seq, T.cast(spin_limit, T.uint32),
                                  T.call_extern("roce_ptr", region_base + T.cast(ctrl_off, T.int64), dtype="handle"), T.address_of(counters[poison_index]), dtype="handle")
                    T.sync_threads()
                    failed = T.alloc_var(T.uint32)
                    failed = T.call_extern("roce_ld_relaxed_gpu_u32", T.address_of(counters[poison_index]), dtype=T.uint32)
                    if failed == T.uint32(0):
                        # 4. concatenate: shard s occupies column block s of every output row
                        T.call_extern("roce_gather", T.address_of(inp[0]),
                                      T.call_extern("roce_ptr", region_base + T.cast(recv_off, T.int64), dtype="handle"),
                                      T.cast(slot_bytes, T.int64), seq, shard_packs, row_packs, T.address_of(out[0]),
                                      index, stride, dtype="handle")
                    # 5. the last block to finish publishes the next epoch
                    T.call_extern("roce_fence_sc_gpu", dtype="handle")
                    T.sync_threads()
                    if tx == 0:
                        T.call_extern("roce_tail", T.address_of(counters[tail_index]), T.cast(grid_x, T.uint32),
                                      T.call_extern("roce_ptr", region_base + T.cast(ctrl_off, T.int64), dtype="handle"), T.address_of(counters[0]), seq, dtype="handle")

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
) -> Callable[[Launch], None]:
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

    def run(launch: Launch) -> None:
        """Launch the compiled kernel on the current stream (bound through the tensor arguments)."""
        kernel(
            launch.input, launch.output, launch.counters, int(launch.region.data_ptr()),
            int(launch.size_packs), int(launch.nbytes), int(launch.row_packs), int(launch.recv_off), int(launch.flag_off),
            int(launch.send_off), int(launch.ctrl_off), int(launch.slot_bytes), int(launch.stage_index),
            int(launch.tail_index), int(launch.poison_index), int(launch.spin_limit), int(launch.grid_x),
        )

    _PREPARED_LAUNCHERS.add(process_key)
    return run


__all__ = ["PACK_BYTES", "get_launcher", "is_launcher_prepared"]
