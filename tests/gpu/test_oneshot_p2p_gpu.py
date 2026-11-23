"""Multi-node GPU correctness and fail-stop tests for one-shot RoCE P2P (``sparknet.oneshot.P2P``).

Run with torchrun on two or more directly cabled or switched Spark nodes::

    torchrun --nnodes=2 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29651 \\
        -m pytest -x tests/gpu/test_oneshot_p2p_gpu.py
"""

from __future__ import annotations

import os
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="requires a torchrun launch with WORLD_SIZE >= 2",
)


@pytest.fixture(scope="module")
def p2p_runtime():
    from sparknet import oneshot as roce

    if not roce.is_supported():
        pytest.skip("RoCE P2P needs an integrated GPU with an active RDMA device")
    if os.environ.get("SPARKNET_ROCE_TOPOLOGY") == "ring4" and not os.environ.get("SPARKNET_ROCE_PP_PEER_HCAS"):
        pytest.skip("ring4 without SPARKNET_ROCE_PP_PEER_HCAS has no opposite-rank direct cable")
    if not dist.is_initialized():
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    rt = roce.P2P.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        slot_bytes=1 << 20,  # 1 MiB slots so 8 MiB exercises > 4-slot credit backpressure
        pack_bytes=4 << 20,
    )
    rt.prepare()
    yield rt
    rt.close()
    dist.barrier()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.int64, torch.uint8])
@pytest.mark.parametrize("numel", [1, 7, 16, 256, 4096])
def test_p2p_send_recv_small_and_unaligned_sizes(p2p_runtime, dtype, numel):
    """Exact bit-for-bit transfer across dtypes and non-16B-multiple sizes."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    dst = (rank + 1) % world
    src = (rank - 1) % world

    torch.manual_seed(100 + rank * 31 + numel)
    if dtype in (torch.int32, torch.int64, torch.uint8):
        high = 250 if dtype == torch.uint8 else 100_000
        send_t = torch.randint(0, high, (numel,), dtype=dtype, device=p2p_runtime.device)
    else:
        send_t = torch.randn(numel, dtype=dtype, device=p2p_runtime.device)
    recv_t = torch.zeros_like(send_t)

    # Reference over gloo CPU so we don't need NCCL initialized.
    gathered = [None] * world
    dist.all_gather_object(gathered, send_t.cpu())
    expected = gathered[src].to(p2p_runtime.device)

    p2p_runtime.send(send_t, dst=dst)
    p2p_runtime.recv(recv_t, src=src)
    torch.cuda.synchronize()
    p2p_runtime.check_health()
    assert torch.equal(recv_t, expected)


def test_p2p_multichunk_streaming_and_credit_backpressure(p2p_runtime):
    """Stream 10 MiB through 1 MiB 4-slot ring (10 chunks > 4 credits) bidirectionally."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    dst = (rank + 1) % world
    src = (rank - 1) % world
    # 10 MiB = 5M bf16 elements = 10 chunks of 1 MiB (> 4 credit slots)
    numel = (10 << 20) // 2
    torch.manual_seed(777 + rank)
    send_t = torch.randn(numel, dtype=torch.bfloat16, device=p2p_runtime.device)
    recv_t = torch.empty_like(send_t)

    # Launch 6 back-to-back sends/recvs to verify credit wrap and streaming stability.
    for step in range(6):
        val = send_t + float(step)
        if rank % 2 == 0:
            p2p_runtime.send(val, dst=dst)
            p2p_runtime.recv(recv_t, src=src)
        else:
            p2p_runtime.recv(recv_t, src=src)
            p2p_runtime.send(val, dst=dst)
        torch.cuda.synchronize()
        p2p_runtime.check_health()
        torch.manual_seed(777 + src)
        expected = torch.randn(numel, dtype=torch.bfloat16, device=p2p_runtime.device) + float(step)
        assert torch.equal(recv_t, expected)


def test_p2p_unaligned_pointer_interoperates_with_aligned(p2p_runtime):
    """Unaligned sender view matches aligned receiver buffer and vice versa."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    dst = (rank + 1) % world
    src = (rank - 1) % world
    # > 1 slot (2.5 MiB) to exercise chunked alignment staging across slot boundaries.
    nbytes = (5 << 19) + 16
    base = torch.arange(nbytes + 32, dtype=torch.uint8, device=p2p_runtime.device)
    base = ((base + rank * 37) % 251).to(torch.uint8)
    send_view = base[3 : 3 + nbytes]  # data_ptr % 16 != 0
    recv_aligned = torch.empty(nbytes, dtype=torch.uint8, device=p2p_runtime.device)
    assert send_view.data_ptr() % 16 != 0
    assert recv_aligned.data_ptr() % 16 == 0

    if rank % 2 == 0:
        p2p_runtime.send(send_view, dst=dst)
        p2p_runtime.recv(recv_aligned, src=src)
    else:
        p2p_runtime.recv(recv_aligned, src=src)
        p2p_runtime.send(send_view, dst=dst)
    torch.cuda.synchronize()
    p2p_runtime.check_health()

    ref = ((torch.arange(nbytes + 32, dtype=torch.uint8, device=p2p_runtime.device) + src * 37) % 251).to(torch.uint8)
    assert torch.equal(recv_aligned, ref[3 : 3 + nbytes])


def test_p2p_send_recv_tensors_coalesced_and_large(p2p_runtime):
    """Multi-tensor coalesced decode (<= pack_bytes) and large zero-copy prefill (> pack_bytes)."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    dst = (rank + 1) % world
    src = (rank - 1) % world

    for rows in (6, 96, 1024):  # ~300 KiB (coalesced), ~4.9 MiB (> 4 MiB pack -> zero-copy), ~52 MiB
        torch.manual_seed(900 + rows + rank)
        s_tensors = [
            torch.randn(rows, 5120, dtype=torch.bfloat16, device=p2p_runtime.device),
            torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device=p2p_runtime.device),
            torch.randn(rows, 4, 16, dtype=torch.float32, device=p2p_runtime.device),
            torch.randint(0, 1000, (rows,), dtype=torch.int32, device=p2p_runtime.device),
        ]
        r_tensors = [torch.empty_like(t) for t in s_tensors]
        if rank % 2 == 0:
            p2p_runtime.send_tensors(s_tensors, dst=dst)
            p2p_runtime.recv_tensors(r_tensors, src=src)
        else:
            p2p_runtime.recv_tensors(r_tensors, src=src)
            p2p_runtime.send_tensors(s_tensors, dst=dst)
        torch.cuda.synchronize()
        p2p_runtime.check_health()

        torch.manual_seed(900 + rows + src)
        expected = [
            torch.randn(rows, 5120, dtype=torch.bfloat16, device=p2p_runtime.device),
            torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device=p2p_runtime.device),
            torch.randn(rows, 4, 16, dtype=torch.float32, device=p2p_runtime.device),
            torch.randint(0, 1000, (rows,), dtype=torch.int32, device=p2p_runtime.device),
        ]
        for got, exp in zip(r_tensors, expected, strict=True):
            assert torch.equal(got, exp)


def test_p2p_broadcast(p2p_runtime):
    """Broadcast from last rank to all other ranks (pipeline sampled/draft token feedback)."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    src = world - 1
    buf = torch.arange(128, dtype=torch.int64, device=p2p_runtime.device) if rank == src else torch.full((128,), -1, dtype=torch.int64, device=p2p_runtime.device)
    p2p_runtime.broadcast(buf, src=src)
    torch.cuda.synchronize()
    p2p_runtime.check_health()
    assert torch.equal(buf, torch.arange(128, dtype=torch.int64, device=p2p_runtime.device))


def test_p2p_cuda_graph_capture_and_replay(p2p_runtime):
    """CUDA-graph captured send and recv replay cleanly across > 4 slots with changing inputs."""
    rank = dist.get_rank()
    world = dist.get_world_size()
    dst = (rank + 1) % world
    src = (rank - 1) % world

    send_buf = torch.zeros(6 * 5120, dtype=torch.bfloat16, device=p2p_runtime.device)
    recv_buf = torch.zeros_like(send_buf)
    stream = torch.cuda.Stream(device=p2p_runtime.device)

    # Warmup on capture stream
    with torch.cuda.stream(stream):
        if rank % 2 == 0:
            p2p_runtime.send(send_buf, dst=dst)
            p2p_runtime.recv(recv_buf, src=src)
        else:
            p2p_runtime.recv(recv_buf, src=src)
            p2p_runtime.send(send_buf, dst=dst)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), p2p_runtime.capture(stream=stream):
        if rank % 2 == 0:
            p2p_runtime.send(send_buf, dst=dst)
            p2p_runtime.recv(recv_buf, src=src)
        else:
            p2p_runtime.recv(recv_buf, src=src)
            p2p_runtime.send(send_buf, dst=dst)

    for step in range(32):
        send_buf.fill_(float(rank * 100 + step))
        recv_buf.fill_(-1.0)
        graph.replay()
        torch.cuda.synchronize()
        p2p_runtime.check_health()
        assert torch.all(recv_buf == float(src * 100 + step))


def test_p2p_dual_stripe_byte_balance(p2p_runtime):
    """Dual-stripe P2P balances posted bytes evenly across both HCAs."""
    if p2p_runtime.stripe_count < 2:
        pytest.skip("requires two RoCE stripes")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dst = (rank + 1) % world
    src = (rank - 1) % world
    nbytes = 256 * 1024
    before = p2p_runtime.stats()["bytes_posted_per_hca"]
    t = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=p2p_runtime.device)
    out = torch.empty_like(t)
    if rank % 2 == 0:
        p2p_runtime.send(t, dst=dst)
        p2p_runtime.recv(out, src=src)
    else:
        p2p_runtime.recv(out, src=src)
        p2p_runtime.send(t, dst=dst)
    torch.cuda.synchronize()
    deadline = time.monotonic() + 5.0
    while True:
        after = p2p_runtime.stats()["bytes_posted_per_hca"]
        deltas = [a - b for a, b in zip(after, before, strict=True)]
        if sum(deltas) >= nbytes:
            break
        assert time.monotonic() < deadline, deltas
        time.sleep(0.001)
    assert deltas == [nbytes // 2, nbytes // 2]
