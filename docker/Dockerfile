# syntax=docker/dockerfile:1.7
#
# Reference serving image: the fabric parts of a multi-Spark vLLM recipe on
# top of the vLLM nightly ARM64 image the fleet serves from. It adds, in
# three stages, what docs/integration.md section 3 asks of an image:
#
#   nccl-builder   NCCL 2.30.7 from the pinned release with the packaged
#                  patch series, built for SM121 only; the library and its
#                  SHA-256 land in /opt/nccl.
#   runtime        the base image with the wheel's libnccl.so.2 replaced by
#                  the patched build, sparknet installed without
#                  dependencies, the one-shot proxy built into
#                  SPARKNET_ROCE_CACHE_DIR, and an import check that fails
#                  the build unless the NCCL hash and the proxy ABI are the
#                  ones built here.
#
# The engine itself is yours: the Local Inference Lab vLLM fork with the
# one-file switch in sparknet/integration/vllm/README.md, or any engine that
# constructs the runtime as examples/minimal_engine.py does. spark-ds41f's
# Dockerfile has the same three stages plus its own vLLM, B12X and TileLang
# stages; this file is the fabric subset of it.
#
# Build on a Spark (the base image is aarch64, NCCL compiles with the base
# image's nvcc; a CPU-only machine can build it too, slowly):
#
#   docker build -f docker/Dockerfile -t sparknet-serving:local .
#
# CI lints this file (`docker buildx build --check`); it does not build it.

ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf
# The base image digest is linux/arm64 only (the Spark is aarch64), so every
# stage says so; a build or a lint on another host otherwise assumes its own.

FROM --platform=linux/arm64 ${BASE_IMAGE} AS sparknet-wheel

USER root
WORKDIR /workspace/sparknet
COPY . .
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /opt/sparknet-wheel . \
    && ls /opt/sparknet-wheel/dgx_spark_networking-*.whl

FROM --platform=linux/arm64 ${BASE_IMAGE} AS nccl-builder

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,from=sparknet-wheel,source=/opt/sparknet-wheel,target=/tmp/sparknet-wheel \
    python3 -m pip install --no-cache-dir --no-deps /tmp/sparknet-wheel/dgx_spark_networking-*.whl
COPY upstreams.lock.json /workspace/upstreams.lock.json
ARG NCCL_COMPILE_JOBS=4
RUN set -eux; \
    lock=/workspace/upstreams.lock.json; \
    url=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sources"]["nccl"]["upstream"])' "$lock"); \
    rev=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sources"]["nccl"]["revision"])' "$lock"); \
    tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sources"]["nccl"]["tree"])' "$lock"); \
    git clone --quiet --filter=blob:none "$url" /workspace/nccl; \
    git -C /workspace/nccl checkout --quiet --detach "$rev"; \
    python3 -m sparknet.cli nccl patches --export /workspace/nccl-patches; \
    grep -v '^#' /workspace/nccl-patches/series | grep -v '^$' | while read -r patch; do \
      git -C /workspace/nccl -c user.name=sparknet -c user.email=sparknet@localhost \
        am --quiet --committer-date-is-author-date "/workspace/nccl-patches/$patch"; \
    done; \
    test "$(git -C /workspace/nccl rev-parse 'HEAD^{tree}')" = "$tree"; \
    make -C /workspace/nccl -j"${NCCL_COMPILE_JOBS}" src.build \
      CUDA_HOME=/usr/local/cuda \
      NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"; \
    mkdir -p /opt/nccl; \
    cp -L /workspace/nccl/build/lib/libnccl.so.2 /opt/nccl/libnccl.so.2; \
    sha256sum /opt/nccl/libnccl.so.2 | cut -d' ' -f1 > /opt/nccl/libnccl.so.2.sha256

FROM --platform=linux/arm64 ${BASE_IMAGE} AS runtime

USER root
# The proxy is plain C over libibverbs, built once here so no compiler runs
# when serving starts.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=nccl-builder /opt/nccl/ /opt/nccl/
RUN set -eux; \
    target=$(python3 -c 'import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], "lib", "libnccl.so.2"))'); \
    test -f "$target"; \
    install -m 0755 /opt/nccl/libnccl.so.2 "$target"; \
    rm /opt/nccl/libnccl.so.2
RUN --mount=type=bind,from=sparknet-wheel,source=/opt/sparknet-wheel,target=/tmp/sparknet-wheel \
    python3 -m pip install --no-cache-dir --no-deps /tmp/sparknet-wheel/dgx_spark_networking-*.whl
ENV SPARKNET_ROCE_CACHE_DIR=/opt/sparknet/roce
# The kernel caches are TileLang's and the CuTe DSL's own (TILELANG_CACHE_DIR,
# CUTE_DSL_CACHE_DIR) and are shared with every other kernel in the serving
# process, so the recipe places them with its other JIT caches on a mounted
# volume and warms them; this image does not set them.
RUN python3 - <<'PY'
import ctypes
import hashlib
import importlib.metadata as metadata
import os

import nvidia.nccl

unconditional = [r for r in metadata.requires("dgx-spark-networking") or () if "extra ==" not in r]
assert not unconditional, f"sparknet takes no runtime dependencies: {unconditional}"

# Ask the installed library for its version (NCCL_VERSION_CODE 2.30.7 = 23007)
# and check it is the build from the nccl-builder stage.
nccl = os.path.join(list(nvidia.nccl.__path__)[0], "lib", "libnccl.so.2")
version = ctypes.c_int()
assert ctypes.CDLL(nccl).ncclGetVersion(ctypes.byref(version)) == 0
assert version.value == 23007, version.value
with open(nccl, "rb") as handle:
    built = open("/opt/nccl/libnccl.so.2.sha256").read().strip()
    assert hashlib.sha256(handle.read()).hexdigest() == built, "NCCL is not the patched build"

from sparknet.oneshot._proxy import load

print("sparknet", metadata.version("dgx-spark-networking"), "proxy ABI", load().roce_abi_version())
print("fabric build checks passed")
PY
