#!/usr/bin/env bash
# The data-phase sweep of README.md: readphase.cu built against two checkouts' generated headers
# (rank 0 of a three- and a four-rank clique) and run on the grid each one's runtime chooses.
#
#   run.sh OLD_CHECKOUT NEW_CHECKOUT OUT     (inside the serving image on an idle GB10)
set -euo pipefail
OLD=$1; NEW=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for world in 3 4; do
  for arm in old new; do
    src=$([ "$arm" = old ] && echo "$OLD" || echo "$NEW")
    python3 - "$src/sparknet/oneshot/_device.py" "$world" > "$OUT/header-$arm-w$world.h" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("device", sys.argv[1])
device = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device)
print(device.render_source(world_size=int(sys.argv[2]), rank=0, slots=2, flag_stride=64, hca_count=2, neighbor_lanes=2))
PY
    flags=()
    if [ "$arm" = new ]; then flags=(-DPEERS_ONLY_GATHER); fi
    nvcc -O3 -std=c++17 -arch=sm_121a -DHDR="\"$OUT/header-$arm-w$world.h\"" "${flags[@]}" -o "$OUT/readphase-$arm-w$world" "$HERE/readphase.cu"
  done
  "$OUT/readphase-old-w$world" 8 2 > "$OUT/old-w$world.tsv"    # 0.4.0: at most 8 blocks, about two packs a thread
  "$OUT/readphase-new-w$world" 64 1 > "$OUT/new-w$world.tsv"   # one pack a thread, at most 64 blocks
done
