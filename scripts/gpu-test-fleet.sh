#!/usr/bin/env bash
# Run the one-shot GPU test suite (tests/gpu/test_oneshot_gpu.py) on every node of a map:
# one torchrun process per node inside the serving image, the map's rendered environment for
# the transport and profile, and this checkout's sparknet package and tests mounted over the
# image's. Run it inside a cluster window with serving stopped.
#
#   scripts/gpu-test-fleet.sh MAP TRANSPORT PROFILE IMAGE [pytest args...]
#
# Environment: SSH_USER (default: the map's ssh_user), CHECKOUT (path of this checkout on the
# nodes, default ~/sparknet-port), PACKAGE_TARGET (the image's sparknet directory), EXTRA_ENV
# ("KEY=VALUE KEY2=VALUE2", e.g. SPARKNET_ROCE_KERNELS=tilelang), MASTER_PORT, OUT (log directory).
# Each node's output lands in $OUT/<node>.log; the exit status is non-zero if any rank failed.
set -euo pipefail
MAP=$1; TRANSPORT=$2; PROFILE=$3; IMAGE=$4; shift 4
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHECKOUT=${CHECKOUT:-/home/\$USER/sparknet-port}
PACKAGE_TARGET=${PACKAGE_TARGET:-/usr/local/lib/python3.12/dist-packages/sparknet}
MASTER_PORT=${MASTER_PORT:-29651}
OUT=${OUT:-$ROOT/evidence/private/gpu-test-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
read -r MASTER NNODES SSH_DEFAULT < <(python3 - "$MAP" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(next(n["management_ip"] for n in d["nodes"] if n["head"]), len(d["nodes"]), d.get("ssh_user") or "")
PY
)
SSH_USER=${SSH_USER:-$SSH_DEFAULT}
PYTEST_ARGS=$(python3 -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@")
pids=()
while read -r node rank; do
  envflags=$(python3 -m sparknet.cli topology render "$MAP" "$node" --transport "$TRANSPORT" --profile "$PROFILE" --json \
    | python3 -c 'import json, shlex, sys; env = json.load(sys.stdin); env.setdefault("NCCL_DEBUG", "WARN"); print(" ".join("--env " + shlex.quote(f"{k}={v}") for k, v in sorted(env.items())))')
  extra=""
  for e in ${EXTRA_ENV:-}; do extra="$extra --env $(printf %q "$e")"; done
  checkout=${CHECKOUT//\$USER/$SSH_USER}
  cmd="docker run --rm --name=sparknet-gpu-test --network=host --gpus=all --memory=16g --memory-swap=16g \
    --device=/dev/infiniband:/dev/infiniband:rwm --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --shm-size=2g \
    $envflags $extra \
    --volume $checkout/sparknet:$PACKAGE_TARGET:ro --volume $checkout/tests:/sparknet-tests:ro \
    --entrypoint /usr/bin/timeout $IMAGE --signal=TERM --kill-after=15s 1500s \
    python3 -m torch.distributed.run --nnodes=$NNODES --nproc-per-node=1 --node-rank=$rank --master-addr=$MASTER --master-port=$MASTER_PORT \
    -m pytest -x -q -p no:cacheprovider /sparknet-tests/gpu/test_oneshot_gpu.py $PYTEST_ARGS"
  target=${SSH_USER:+$SSH_USER@}$node
  echo "[$node rank $rank] $cmd" > "$OUT/$node.log"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$target" "$cmd" >> "$OUT/$node.log" 2>&1 &
  pids+=($!)
done < <(python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); [print(n["name"], n["rank"]) for n in sorted(d["nodes"], key=lambda n: n["rank"])]' "$MAP")
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
while read -r node rank; do
  printf '%s (rank %s): ' "$node" "$rank"
  grep -E '[0-9]+ passed|[0-9]+ failed|[0-9]+ error' "$OUT/$node.log" | tail -1 || echo "no pytest summary (see $OUT/$node.log)"
done < <(python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); [print(n["name"], n["rank"]) for n in sorted(d["nodes"], key=lambda n: n["rank"])]' "$MAP")
echo "logs: $OUT"
exit $rc
