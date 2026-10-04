# DOCA GPUNetIO on DGX Spark

Pinned source: [NVIDIA-DOCA/gpunetio](https://github.com/NVIDIA-DOCA/gpunetio)
at `586453728bcab2d4c50574924dc6cf43543c9ed4` (library build 4.0.1,
BSD-3-Clause). `build.sh` clones it into `.work/`, applies the series and
builds the library plus the one-sided write-latency example for SM121 with
CUDA 13.0. No DOCA SDK, GDRCopy, driver change or root package is needed.

The series adapts the sample to [NVIDIA's Spark guidance](https://networking-docs.nvidia.com/doca/archive/3-5-0/doca-gpunetio#general-performance-and-best-practices)
(CPU/GPU shared memory and CPU-proxy transmissions):

1. `0001-spark-shared-host-memory.patch`: queues and data buffers in explicit
   CPU/GPU shared host memory, registered through the CPU pointer; unequal
   CPU/GPU aliases are rejected instead of assumed.
2. `0002-progress-diagnostic.patch`, `0003-error-diagnostic.patch`: producer
   indices and CQE error syndromes in the logs, so a stall is attributable.
3. `0004-system-scope-fence.patch`: GPU-written host work queues are published
   with a system-scope release before the CPU proxy is signalled. With the
   sample's GPU-scope fence both bounded pair attempts timed out; with system
   scope all ten sizes (1 byte to 16 KiB) completed on both lanes of every
   cable of the four Sparks (half round trip 4.1 to 4.5 us at one byte, 6.7 to
   7.0 us at 16 KiB).

The GPU-doorbell handler (`NIC_HANDLER_GPU_SM_DB`) is excluded: its one
attempt ended in a host-level failure (dgx1 rebooted, dgx2's GPU needed a
reboot). See `docs/gpudirect-roadmap.md`.
