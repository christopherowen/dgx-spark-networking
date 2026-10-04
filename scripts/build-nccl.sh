#!/usr/bin/env bash
# Rebuild NCCL 2.30.7 for SM121 with patches/nccl and leave libnccl.so.2 plus its sha256 in OUT.
#
#   scripts/build-nccl.sh [OUT=./build/nccl] [JOBS=4]
#
# Needs git, make, CUDA (CUDA_HOME, default /usr/local/cuda) and nvcc. Runs on a Spark or in the
# vLLM base image; the serving Dockerfile does the same steps in its nccl-builder stage.
set -euo pipefail
OUT=${1:-build/nccl}
JOBS=${2:-4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOCK="$ROOT/upstreams.lock.json"
REV=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sources"]["nccl"]["revision"])' "$LOCK")
URL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sources"]["nccl"]["upstream"])' "$LOCK")
WORK="$ROOT/.work/nccl-$REV"
mkdir -p "$ROOT/.work" "$OUT"
if [[ ! -d "$WORK/.git" ]]; then
  git clone --quiet "$URL" "$WORK"
fi
git -C "$WORK" checkout --quiet --detach "$REV"
git -C "$WORK" reset --quiet --hard
while read -r patch; do
  [[ -z "$patch" || "$patch" == \#* ]] && continue
  git -C "$WORK" -c user.name=sparknet -c user.email=sparknet@localhost am --quiet --committer-date-is-author-date "$ROOT/patches/nccl/$patch"
done < "$ROOT/patches/nccl/series"
echo "patch head: $(git -C "$WORK" rev-parse HEAD)  tree: $(git -C "$WORK" rev-parse 'HEAD^{tree}')"
make -C "$WORK" -j"$JOBS" src.build CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
  NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
cp -L "$WORK/build/lib/libnccl.so.2" "$OUT/libnccl.so.2"
sha256sum "$OUT/libnccl.so.2" | cut -d' ' -f1 > "$OUT/libnccl.so.2.sha256"
echo "built $OUT/libnccl.so.2 ($(cat "$OUT/libnccl.so.2.sha256"))"
