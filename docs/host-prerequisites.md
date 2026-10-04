# Host prerequisites

Checked by `sparknet probe doctor` and `sparknet probe gpudirect`; changed
only by the owner's fleet tooling.

- DGX Spark OS 26.09 or later, kernel `7.0.0-1019-nvidia` (4 KiB or 64 KiB
  pages) with `kho=off`: without it `ibv_reg_mr` fails with ENOMEM under
  memory pressure, which breaks both NCCL and one-shot.
- NVIDIA driver 580.178.04 (open kernel module); ConnectX-7 firmware
  28.45.4028; rdma-core 50 with `libibverbs-dev` and the mlx5 provider
  headers; a C compiler (the proxy is built at first use or at image build).
- Both PCIe functions of each cabled port active at 200 Gb/s, MTU 9000, one
  static IPv4 address per cable stripe, no IPv6, GID index 3 holding the
  IPv4 RoCE v2 GID.
- Locked-memory limit unlimited for the serving user and inside the
  container (`--ulimit memlock=-1:-1`, `--cap-add IPC_LOCK`,
  `--device /dev/infiniband`).
- Docker with the NVIDIA container runtime; `--network=host`.
- SRP target discovery off, NVMe interrupt coalescing off, snap off, no
  desktop: the fleet's host policy, which keeps memory and interrupts
  predictable for the memory guards and the proxy thread.
- A memory guard beside the serving process: the fleet's watermark-boost
  and MemAvailable guards (spark3 `memguard`) are outside this library.
