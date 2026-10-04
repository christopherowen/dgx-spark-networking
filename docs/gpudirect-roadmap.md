# GPU-initiated transport: the roadmap

Today the host proxy posts every RDMA write: a C thread polls the doorbell,
stripes the payload, posts work requests and, in `ring4`, forwards the
neighbours' fragments. Its measured cost on the critical path is small
(local posting under 1 us per collective, relay phase 6 to 87 us dominated
by the transfer itself) but it is a thread per rank, a doorbell poll and a
host-side ordering point. The pivot moves the posting side into the kernel
while keeping the protocol, geometry and wire layout, so the kernels, the
flags, the relay semantics and the fail-stop contract do not change.

The seam is `sparknet.transport.Transport`. `Geometry` carries everything a
posting implementation needs; the runtime constructs the transport through
a factory and calls `local_blob`, `connect`, `start`, `failed`, `error`,
`stats` and `close` in that order. The host proxy is the first
implementation; `GpuNetIOTransport` is the staged second one.

## Stage 1: GPUNetIO with the CPU proxy handler

DOCA GPUNetIO (open source, pinned in `native/gpunetio`) exposes verbs
queues to CUDA kernels: `doca_gpu_dev_verbs_wqe_prepare_write`,
`doca_gpu_dev_verbs_submit`, `doca_gpu_dev_verbs_poll_cq`. With
`NIC_HANDLER_CPU_PROXY` the kernel builds the write and flag WQEs right after
staging and publishes them; a host thread only rings the NIC doorbell
(`doca_gpu_verbs_cpu_proxy_progress`). This is NVIDIA's recommended mode on
Spark (CPU/GPU shared memory, CPU-proxy transmissions).

Qualified on the fleet (spark3 `2026-10-03-relay-progress`): built for SM121
with CUDA 13.0 and rdma-core 50, no DOCA SDK, no driver change. Two sample
adaptations were necessary: queues and data buffers in explicit CPU/GPU
shared host memory registered through the CPU pointer, and a system-scope
release when publishing GPU-written host work queues (the sample's GPU-scope
fence left the proxy reading stale WQEs; both attempts timed out). With
them, all ten sizes from 1 byte to 16 KiB completed on both lanes of every
cable: half round trip 4.1 to 4.5 us at one byte, 6.7 to 7.0 us at 16 KiB.
These are ping-pong samples, not collectives.

What the adapter needs: a per-peer, per-lane QP pair created through
`doca_gpu_verbs_create_qp_hl` and connected like the proxy's RC QPs; the
region registered once per HCA; the kernel's doorbell step replaced by WQE
construction for every stripe (data write then inline flag write on the same
QP, keeping the flag-after-data ordering); the relay step for `ring4` either
kept on the host (the intermediate's forwarding is host work anyway) or
moved into a small forwarding kernel; `stats` from the device counters; the
fail-stop poison on a CQE error. Acceptance: the C simulator's protocol
cases re-expressed against the device API, the GPU test, and the collective
probe at equal correctness, bidirectional routing, graph execution and mixed
load; it must beat the host proxy, not merely match it.

## Stage 2: GPU doorbell

`NIC_HANDLER_GPU_SM_DB` lets the kernel ring the UAR itself. The one attempt
on the Sparks set up successfully but no size completed; dgx1 rebooted
during the run and dgx2's GPU stayed at 96 percent utilization until a
reboot, with no classified driver or firmware failure. Before any further
attempt: a host-level investigation (journal, pstore, UAR mapping on this
driver, `doca_gpu_verbs_can_gpu_register_uar`), on one idle node, with the
owner.

## Stage 3: device-memory registration

With unified memory the pinned host region already serves the GPU at full
bandwidth, so this stage is about placement, not necessity. rdma-core 50 on
the fleet exports `ibv_reg_dmabuf_mr`; whether the GB10 driver exports a
dma-buf for a device allocation (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`) is
what `sparknet probe gpudirect` reports. `nvidia-peermem` ships with driver
580 and is present but not loaded on the fleet; loading it is a host
decision.

## What stays the same

Pinned host slots and per-lane flags, the two-slot lifetime, fixed-rank
reduction, the epoch, the fail-stop contract, the node map and the policy.
The NCCL side is untouched by the pivot.
