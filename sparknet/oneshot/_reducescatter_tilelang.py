"""TileLang kernel for the RoCE one-shot reduce-scatter.

Stages each peer p's chunk (p != rank) into send[slot] + p * chunk_bytes, rings
the proxy doorbell in exchange mode (ROCE_MODE_EXCHANGE), waits for every peer's
stripe flags, and reduces the local chunk and every peer's received chunk in
fixed rank order with float32 accumulation into out. Multi-stage launches
execute all stages inside a single persistent kernel across the two slots.
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
from ._oneshot_tilelang import (
    _DTYPE_PACK_ELEMS,
    _REDUCE_FUNCTION,
    _lanes,
    _pass_configs,
    _resident_blocks,
)

logger = logging.getLogger(__name__)

_PREPARED_LAUNCHERS: set[tuple[object, ...]] = set()


def _build(
    dtype_name: str,
    world_size: int,
    rank: int,
    threads: int,
    slots: int,
    flag_stride: int,
    hca_count: int,
    ring4: bool,
):
    import tilelang
    import tilelang.language as T

    lanes, neighbor_lanes = _lanes(hca_count, ring4)
    source = render_source(
        world_size=world_size,
        rank=rank,
        slots=slots,
        flag_stride=flag_stride,
        hca_count=lanes,
        neighbor_lanes=neighbor_lanes,
    )
    reduce_pack = _REDUCE_FUNCTION[dtype_name]
    resident = _resident_blocks(threads)
    pack_words = PACK_BYTES // 4

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_in, n_out, n_counters = T.dynamic("n_in, n_out, n_counters")

        @T.prim_func
        def oneshot_reducescatter(
            inp: T.Tensor((n_in,), T.int32),
            out: T.Tensor((n_out,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            chunk_packs: T.int32,
            chunk_bytes: T.int32,
            src_peer_packs: T.int32,
            recv_off: T.int32,
            flag_off: T.int32,
            send_off: T.int32,
            ctrl_off: T.int32,
            slot_bytes: T.int32,
            stage_index: T.int32,
            tail_index: T.int32,
            rung_index: T.int32,
            arrival_index: T.int32,
            grid_x: T.int32,
            num_stages: T.int32,
            stage_packs: T.int32,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(source)
                tx = T.get_thread_binding()
                epoch = T.alloc_var(T.uint32)
                epoch = T.call_extern(
                    "roce_ld_relaxed_gpu_u32", T.address_of(counters[0]), dtype=T.uint32
                )
                seq = T.alloc_var(T.uint32)
                slot = T.alloc_var(T.int32)
                send_slot = T.alloc_var(T.int32)
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                stage_offset = T.alloc_var(T.int32)
                cur_packs = T.alloc_var(T.int32)
                count = T.alloc_var(T.int32)
                rang = T.alloc_var(T.uint32)
                pack = T.alloc_var(T.int32)

                for s in T.serial(num_stages):
                    seq = epoch + T.uint32(1) + T.cast(s, T.uint32)
                    slot = T.cast(seq & T.uint32(slots - 1), T.int32)
                    send_slot = send_off + slot * slot_bytes
                    stage_offset = s * stage_packs
                    cur_packs = T.min(stage_packs, chunk_packs - stage_offset)
                    count = T.max(0, (cur_packs - index + stride - 1) // stride)

                    for i in T.serial(count):
                        T.call_extern(
                            "roce_stage_scatter_pack",
                            T.address_of(inp[stage_offset * pack_words]),
                            T.call_extern(
                                "roce_ptr",
                                region_base + T.cast(send_slot, T.int64),
                                dtype="handle",
                            ),
                            index + i * stride,
                            cur_packs,
                            src_peer_packs,
                            dtype="handle",
                        )
                    T.sync_threads()
                    if tx == 0:
                        rang = T.call_extern(
                            "roce_doorbell_exchange",
                            T.address_of(counters[stage_index]),
                            T.cast(grid_x, T.uint32),
                            T.call_extern(
                                "roce_ptr",
                                region_base + T.cast(ctrl_off, T.int64),
                                dtype="handle",
                            ),
                            T.cast(cur_packs * PACK_BYTES, T.uint32),
                            seq,
                            dtype=T.uint32,
                        )
                        if rang != T.uint32(0):
                            T.call_extern(
                                "roce_st_release_gpu_u32",
                                T.address_of(counters[rung_index]),
                                seq,
                                dtype="handle",
                            )

                    T.call_extern(
                        "roce_wait_peers",
                        bx,
                        tx,
                        T.call_extern(
                            "roce_ptr",
                            region_base + T.cast(flag_off, T.int64),
                            dtype="handle",
                        ),
                        seq,
                        T.call_extern(
                            "roce_ptr",
                            region_base + T.cast(ctrl_off, T.int64),
                            dtype="handle",
                        ),
                        T.address_of(counters[rung_index]),
                        T.address_of(counters[arrival_index]),
                        dtype="handle",
                    )
                    T.sync_threads()
                    if bx == 0:
                        if tx == 0:
                            T.call_extern(
                                "roce_st_release_gpu_u32",
                                T.address_of(counters[arrival_index]),
                                seq,
                                dtype="handle",
                            )

                    for i in T.serial(count):
                        pack = index + i * stride
                        T.call_extern(
                            reduce_pack,
                            T.address_of(
                                inp[(rank * src_peer_packs + stage_offset) * pack_words]
                            ),
                            T.call_extern(
                                "roce_ptr",
                                region_base + T.cast(recv_off, T.int64),
                                dtype="handle",
                            ),
                            T.cast(slot_bytes, T.int64),
                            seq,
                            T.address_of(out[stage_offset * pack_words]),
                            T.cast(pack, T.int64) * PACK_BYTES,
                            dtype="handle",
                        )
                    T.sync_threads()

                T.call_extern("roce_fence_sc_gpu", dtype="handle")
                T.sync_threads()
                if tx == 0:
                    T.call_extern(
                        "roce_tail",
                        T.address_of(counters[tail_index]),
                        T.cast(grid_x, T.uint32),
                        T.address_of(counters[0]),
                        epoch + T.cast(num_stages, T.uint32),
                        dtype="handle",
                    )

        return oneshot_reducescatter

    return program


def _process_key(
    dtype_name,
    world_size,
    rank,
    threads,
    slots,
    flag_stride,
    hca_count,
    device_index,
    ring4=False,
):
    return (
        str(dtype_name),
        int(world_size),
        int(rank),
        int(threads),
        int(slots),
        int(flag_stride),
        int(hca_count),
        int(device_index),
        bool(ring4),
    )


def is_launcher_prepared(*key) -> bool:
    """True when the launcher for ``key`` is already compiled."""
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
) -> Callable[[Launch], None]:
    """Compile the launcher for the key once and return it."""
    if dtype_name not in _DTYPE_PACK_ELEMS:
        raise ValueError(f"unsupported RoCE reduce-scatter dtype {dtype_name!r}")
    lanes, _ = _lanes(hca_count, ring4)
    if int(threads) < int(world_size) * lanes:
        raise ValueError(
            "RoCE kernels need threads >= world_size * hca_count (one thread per stripe flag), "
            f"got threads={threads} world_size={world_size} hca_count={lanes}"
        )
    process_key = _process_key(
        dtype_name,
        world_size,
        rank,
        threads,
        slots,
        flag_stride,
        hca_count,
        device_index,
        ring4,
    )
    cache_key = process_key[:-2] + (bool(ring4),)
    raise_if_kernel_resolution_frozen(
        "tilelang.jit", target="oneshot_reducescatter", cache_key=cache_key
    )
    started = time.monotonic()
    kernel = _build(
        dtype_name, world_size, rank, threads, slots, flag_stride, hca_count, ring4
    )()
    logger.info(
        "compiled tilelang oneshot.reducescatter %s in %.1f s",
        cache_key,
        time.monotonic() - started,
    )

    def run(launch: Launch) -> None:
        """Launch the compiled kernel on the current stream (bound through the tensor arguments)."""
        chunk_packs = int(launch.size_packs)
        src_stride = (
            int(launch.src_peer_packs) if launch.src_peer_packs else chunk_packs
        )
        num_stages = int(launch.num_stages) if launch.num_stages > 0 else 1
        stage_packs = int(launch.stage_packs) if launch.stage_packs > 0 else chunk_packs
        kernel(
            launch.input,
            launch.output,
            launch.counters,
            int(launch.region.data_ptr()),
            chunk_packs,
            int(launch.nbytes),
            src_stride,
            int(launch.recv_off),
            int(launch.flag_off),
            int(launch.send_off),
            int(launch.ctrl_off),
            int(launch.slot_bytes),
            int(launch.stage_index),
            int(launch.tail_index),
            int(launch.rung_index),
            int(launch.arrival_index),
            int(launch.grid_x),
            num_stages,
            stage_packs,
        )

    _PREPARED_LAUNCHERS.add(process_key)
    return run


launcher = get_launcher

__all__ = ["PACK_BYTES", "get_launcher", "is_launcher_prepared"]
