#!/usr/bin/env bash
# Build the pinned DOCA GPUNetIO open-source tree with the Spark patches, in an isolated .work tree.
#
#   native/gpunetio/build.sh [stock|system]     (default: system = all four patches)
#
# Needs git, make, CUDA 13 (CUDA_HOME) and the rdma-core development packages. The library and the
# one-sided latency example are built for SM121. Nothing is installed system-wide; point
# SPARKNET_GPUNETIO_DIR at the resulting tree for `sparknet probe gpudirect`.
set -euo pipefail
ARM=${1:-system}
case "$ARM" in stock|system) ;; *) echo "arm must be stock or system" >&2; exit 2 ;; esac
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REV=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["upstream"])' "$ROOT/native/gpunetio/sources.json")
SRC="$ROOT/.work/gpunetio-${REV:0:12}"
mkdir -p "$ROOT/.work"
if [[ ! -d "$SRC/.git" ]]; then
  git clone --quiet https://github.com/NVIDIA-DOCA/gpunetio.git "$SRC"
fi
# configure rewrites this tracked generated header; reset only that reproducible output.
git -C "$SRC" restore --source=HEAD -- include/doca_gpunetio_config.h 2>/dev/null || true
git -C "$SRC" checkout --quiet --detach "$REV"
if [[ "$ARM" == system ]]; then
  while read -r patch; do
    [[ -z "$patch" || "$patch" == \#* ]] && continue
    git -C "$SRC" -c user.name=sparknet -c user.email=sparknet@localhost am --quiet --committer-date-is-author-date "$ROOT/native/gpunetio/$patch"
  done < "$ROOT/native/gpunetio/series"
fi
echo "source head: $(git -C "$SRC" rev-parse HEAD)"
# The upstream all target can link examples before the library under -j.
make -C "$SRC" -j4 lib CUDA_ARCH=121 CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
make -C "$SRC" -j4 examples CUDA_ARCH=121 CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
echo "export SPARKNET_GPUNETIO_DIR=$SRC"
