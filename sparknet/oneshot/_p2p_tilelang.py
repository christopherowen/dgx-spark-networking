"""TileLang kernels for the RoCE one-shot point-to-point send and recv.

Each kernel streams an arbitrary 16-byte-aligned tensor in ``num_chunks``
pipelined chunks of up to ``slot_bytes`` over a 4-slot credit-flow-controlled
RDMA ring. Single-chunk decode transfers (``num_chunks == 1``) stage and ring
the proxy doorbell and return without waiting for the receiver as long as at
least one credit slot is free; multi-chunk prefill transfers pipeline up to
four slots in flight without yielding the GPU between chunks.

Importing this module needs nothing; ``get_send_launcher`` and
``get_recv_launcher`` import TileLang.
"""

# No postponed annotations: TileLang evaluates the prim_func's annotations
# when the function is defined, and the symbolic shapes must be visible then.

import functools
import logging
import time
from typing import Callable

from ._device import PACK_BYTES, render_source
from ._freeze import raise_if_kernel_resolution_frozen
from ._kernels import P2PLaunch
from ._oneshot_tilelang import _pass_configs, _resident_blocks

logger = logging.getLogger(__name__)

_PREPARED_SEND_LAUNCHERS: set[tuple[object, ...]] = set()
_PREPARED_RECV_LAUNCHERS: set[tuple[object, ...]] = set()


def _build_send(world_size: int, rank: int, threads: int, slots: int, flag_stride: int, hca_count: int):
    import tilelang
    import tilelang.language as T

    source = render_source(
        world_size=world_size,
        rank=rank,
        slots=slots,
        flag_stride=flag_stride,
        hca_count=hca_count,
        neighbor_lanes=hca_count,
    )
    resident = _resident_blocks(threads)

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_buf, n_counters = T.dynamic("n_buf, n_counters")

        @T.prim_func
        def oneshot_p2p_send(
            inp: T.Tensor((n_buf,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            total_packs: T.int64,
            chunk_packs: T.int32,
            num_chunks: T.int32,
            peer_slot: T.int32,
            dst_rank: T.int32,
            send_off: T.int64,
            tx_ack_off: T.int64,
            tx_ctrl_off: T.int64,
            ctrl_off: T.int64,
            slot_bytes: T.int64,
            epoch_index: T.int32,
            stage_index: T.int32,
            credit_ready_index: T.int32,
            stage_done_index: T.int32,
            grid_x: T.int32,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(source)
                tx = T.get_thread_binding()
                epoch = T.alloc_var(T.uint32)
                epoch = T.call_extern("roce_ld_relaxed_gpu_u32", T.address_of(counters[epoch_index]), dtype=T.uint32)
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                peer_ctrl_off = T.alloc_var(T.int64)
                peer_ctrl_off = T.cast(peer_slot, T.int64) * T.int64(flag_stride)
                tx_ack_addr = T.alloc_var(T.int64)
                tx_ack_addr = region_base + tx_ack_off + peer_ctrl_off
                tx_ctrl_addr = T.alloc_var(T.int64)
                tx_ctrl_addr = region_base + tx_ctrl_off + peer_ctrl_off
                ctrl_addr = T.alloc_var(T.int64)
                ctrl_addr = region_base + ctrl_off

                for s in T.serial(num_chunks):
                    seq = T.alloc_var(T.uint32)
                    seq = epoch + T.cast(s, T.uint32) + T.uint32(1)
                    if s > 0:
                        if bx == 0:
                            if tx == 0:
                                T.call_extern(
                                    "roce_wait_device_eq",
                                    T.address_of(counters[stage_done_index]),
                                    epoch + T.cast(s, T.uint32),
                                    T.call_extern("roce_ptr", ctrl_addr, dtype="handle"),
                                    dtype="handle",
                                )
                        T.sync_threads()
                    # 1. wait until receiver has ACKed seq - slots so send[slot] and recv[slot] are free
                    T.call_extern(
                        "roce_p2p_wait_credit",
                        bx,
                        tx,
                        T.call_extern("roce_ptr", tx_ack_addr, dtype="handle"),
                        seq,
                        dst_rank,
                        T.call_extern("roce_ptr", ctrl_addr, dtype="handle"),
                        T.address_of(counters[credit_ready_index]),
                        dtype="handle",
                    )
                    T.sync_threads()
                    # 2. stage this chunk into send[peer_slot][seq & (slots - 1)]
                    slot = T.alloc_var(T.int64)
                    slot = T.cast(seq & T.uint32(slots - 1), T.int64)
                    send_slot_addr = T.alloc_var(T.int64)
                    send_slot_addr = region_base + send_off + (T.cast(peer_slot, T.int64) * T.int64(slots) + slot) * slot_bytes
                    pack_base = T.alloc_var(T.int64)
                    pack_base = T.cast(s, T.int64) * T.cast(chunk_packs, T.int64)
                    rem_packs = T.alloc_var(T.int64)
                    rem_packs = total_packs - pack_base
                    this_packs = T.alloc_var(T.int32)
                    this_packs = T.if_then_else(rem_packs < T.cast(chunk_packs, T.int64), T.cast(rem_packs, T.int32), chunk_packs)
                    count = T.alloc_var(T.int32)
                    count = T.max(0, (this_packs - index + stride - 1) // stride)
                    for i in T.serial(count):
                        local_pack = T.cast(index + i * stride, T.int64)
                        T.call_extern(
                            "roce_copy_pack_at",
                            T.address_of(inp[0]),
                            pack_base + local_pack,
                            T.call_extern("roce_ptr", send_slot_addr, dtype="handle"),
                            local_pack,
                            dtype="handle",
                        )
                    T.sync_threads()
                    # 3. last block publishes slot byte count, rings tx_seq, and advances epoch on final chunk
                    if tx == 0:
                        T.call_extern(
                            "roce_p2p_tx_doorbell",
                            T.address_of(counters[stage_index]),
                            T.cast(grid_x, T.uint32),
                            T.call_extern("roce_ptr", tx_ctrl_addr, dtype="handle"),
                            T.cast(this_packs, T.uint32) * T.uint32(PACK_BYTES),
                            seq,
                            T.address_of(counters[stage_done_index]),
                            T.address_of(counters[epoch_index]),
                            T.if_then_else(s == num_chunks - 1, 1, 0),
                            dtype="handle",
                        )

        return oneshot_p2p_send

    return program


def _build_recv(world_size: int, rank: int, threads: int, slots: int, flag_stride: int, hca_count: int):
    import tilelang
    import tilelang.language as T

    source = render_source(
        world_size=world_size,
        rank=rank,
        slots=slots,
        flag_stride=flag_stride,
        hca_count=hca_count,
        neighbor_lanes=hca_count,
    )
    resident = _resident_blocks(threads)

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_buf, n_counters = T.dynamic("n_buf, n_counters")

        @T.prim_func
        def oneshot_p2p_recv(
            out: T.Tensor((n_buf,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            total_packs: T.int64,
            chunk_packs: T.int32,
            num_chunks: T.int32,
            peer_slot: T.int32,
            src_rank: T.int32,
            recv_off: T.int64,
            flag_off: T.int64,
            ack_ctrl_off: T.int64,
            ctrl_off: T.int64,
            slot_bytes: T.int64,
            epoch_index: T.int32,
            ack_index: T.int32,
            arrival_index: T.int32,
            chunk_done_index: T.int32,
            grid_x: T.int32,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(source)
                tx = T.get_thread_binding()
                epoch = T.alloc_var(T.uint32)
                epoch = T.call_extern("roce_ld_relaxed_gpu_u32", T.address_of(counters[epoch_index]), dtype=T.uint32)
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                peer_flags_addr = T.alloc_var(T.int64)
                peer_flags_addr = (
                    region_base
                    + flag_off
                    + T.cast(peer_slot, T.int64) * T.int64(slots * hca_count * flag_stride)
                )
                ack_ctrl_addr = T.alloc_var(T.int64)
                ack_ctrl_addr = region_base + ack_ctrl_off + T.cast(peer_slot, T.int64) * T.int64(flag_stride)
                ctrl_addr = T.alloc_var(T.int64)
                ctrl_addr = region_base + ctrl_off

                for s in T.serial(num_chunks):
                    seq = T.alloc_var(T.uint32)
                    seq = epoch + T.cast(s, T.uint32) + T.uint32(1)
                    if s > 0:
                        if bx == 0:
                            if tx == 0:
                                T.call_extern(
                                    "roce_wait_device_eq",
                                    T.address_of(counters[chunk_done_index]),
                                    epoch + T.cast(s, T.uint32),
                                    T.call_extern("roce_ptr", ctrl_addr, dtype="handle"),
                                    dtype="handle",
                                )
                        T.sync_threads()
                    # 1. block 0 waits for every stripe flag of this chunk; other blocks wait on arrival
                    T.call_extern(
                        "roce_p2p_wait_rx_flags",
                        bx,
                        tx,
                        T.call_extern("roce_ptr", peer_flags_addr, dtype="handle"),
                        seq,
                        src_rank,
                        T.call_extern("roce_ptr", ctrl_addr, dtype="handle"),
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
                    # 2. unpack recv[peer_slot][seq & (slots - 1)] into out with system-scope loads
                    slot = T.alloc_var(T.int64)
                    slot = T.cast(seq & T.uint32(slots - 1), T.int64)
                    recv_slot_addr = T.alloc_var(T.int64)
                    recv_slot_addr = region_base + recv_off + (T.cast(peer_slot, T.int64) * T.int64(slots) + slot) * slot_bytes
                    pack_base = T.alloc_var(T.int64)
                    pack_base = T.cast(s, T.int64) * T.cast(chunk_packs, T.int64)
                    rem_packs = T.alloc_var(T.int64)
                    rem_packs = total_packs - pack_base
                    this_packs = T.alloc_var(T.int32)
                    this_packs = T.if_then_else(rem_packs < T.cast(chunk_packs, T.int64), T.cast(rem_packs, T.int32), chunk_packs)
                    count = T.alloc_var(T.int32)
                    count = T.max(0, (this_packs - index + stride - 1) // stride)
                    for i in T.serial(count):
                        local_pack = T.cast(index + i * stride, T.int64)
                        T.call_extern(
                            "roce_copy_pack_from_sys_at",
                            T.call_extern("roce_ptr", recv_slot_addr, dtype="handle"),
                            local_pack,
                            T.address_of(out[0]),
                            pack_base + local_pack,
                            dtype="handle",
                        )
                    T.sync_threads()
                    # 3. last block publishes rx_ack_seq for the proxy and advances epoch on final chunk
                    if tx == 0:
                        T.call_extern(
                            "roce_p2p_ack_doorbell",
                            T.address_of(counters[ack_index]),
                            T.cast(grid_x, T.uint32),
                            T.call_extern("roce_ptr", ack_ctrl_addr, dtype="handle"),
                            seq,
                            T.address_of(counters[chunk_done_index]),
                            T.address_of(counters[epoch_index]),
                            T.if_then_else(s == num_chunks - 1, 1, 0),
                            dtype="handle",
                        )

        return oneshot_p2p_recv

    return program


def _process_key(world_size, rank, threads, slots, flag_stride, hca_count, device_index):
    return (int(world_size), int(rank), int(threads), int(slots), int(flag_stride), int(hca_count), int(device_index))


def is_send_launcher_prepared(*key) -> bool:
    return _process_key(*key) in _PREPARED_SEND_LAUNCHERS


def is_recv_launcher_prepared(*key) -> bool:
    return _process_key(*key) in _PREPARED_RECV_LAUNCHERS


@functools.cache
def get_send_launcher(
    world_size: int,
    rank: int,
    threads: int,
    slots: int,
    flag_stride: int,
    hca_count: int,
    device_index: int,
) -> Callable[[P2PLaunch], None]:
    if int(threads) < int(hca_count):
        raise ValueError(f"RoCE P2P kernels need threads >= hca_count, got threads={threads} hca_count={hca_count}")
    process_key = _process_key(world_size, rank, threads, slots, flag_stride, hca_count, device_index)
    cache_key = process_key[:-1]
    raise_if_kernel_resolution_frozen("tilelang.jit", target="oneshot_p2p_send", cache_key=cache_key)
    started = time.monotonic()
    kernel = _build_send(world_size, rank, threads, slots, flag_stride, hca_count)()
    logger.info("compiled tilelang oneshot.p2p_send %s in %.1f s", cache_key, time.monotonic() - started)

    def run(launch: P2PLaunch) -> None:
        kernel(
            launch.tensor,
            launch.counters,
            int(launch.region.data_ptr()),
            int(launch.total_packs),
            int(launch.chunk_packs),
            int(launch.num_chunks),
            int(launch.peer_slot),
            int(launch.peer_rank),
            int(launch.data_off),
            int(launch.signal_off),
            int(launch.doorbell_off),
            int(launch.ctrl_off),
            int(launch.slot_bytes),
            int(launch.epoch_index),
            int(launch.barrier_index),
            int(launch.ready_index),
            int(launch.done_index),
            int(launch.grid_x),
        )

    _PREPARED_SEND_LAUNCHERS.add(process_key)
    return run


@functools.cache
def get_recv_launcher(
    world_size: int,
    rank: int,
    threads: int,
    slots: int,
    flag_stride: int,
    hca_count: int,
    device_index: int,
) -> Callable[[P2PLaunch], None]:
    if int(threads) < int(hca_count):
        raise ValueError(f"RoCE P2P kernels need threads >= hca_count, got threads={threads} hca_count={hca_count}")
    process_key = _process_key(world_size, rank, threads, slots, flag_stride, hca_count, device_index)
    cache_key = process_key[:-1]
    raise_if_kernel_resolution_frozen("tilelang.jit", target="oneshot_p2p_recv", cache_key=cache_key)
    started = time.monotonic()
    kernel = _build_recv(world_size, rank, threads, slots, flag_stride, hca_count)()
    logger.info("compiled tilelang oneshot.p2p_recv %s in %.1f s", cache_key, time.monotonic() - started)

    def run(launch: P2PLaunch) -> None:
        kernel(
            launch.tensor,
            launch.counters,
            int(launch.region.data_ptr()),
            int(launch.total_packs),
            int(launch.chunk_packs),
            int(launch.num_chunks),
            int(launch.peer_slot),
            int(launch.peer_rank),
            int(launch.data_off),
            int(launch.signal_off),
            int(launch.doorbell_off),
            int(launch.ctrl_off),
            int(launch.slot_bytes),
            int(launch.epoch_index),
            int(launch.barrier_index),
            int(launch.ready_index),
            int(launch.done_index),
            int(launch.grid_x),
        )

    _PREPARED_RECV_LAUNCHERS.add(process_key)
    return run


__all__ = [
    "PACK_BYTES",
    "get_recv_launcher",
    "get_send_launcher",
    "is_recv_launcher_prepared",
    "is_send_launcher_prepared",
]
