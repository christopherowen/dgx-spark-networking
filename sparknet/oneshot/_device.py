"""The one-shot protocol's device side as CUDA C, for kernels that are generated as CUDA source.

The TileLang kernels (``_oneshot_tilelang``, ``_allgather_tilelang``) inject
this header with ``T.import_source`` and call its functions by name. It is
the CuTe DSL intrinsics of ``_cute_intrinsics.py`` spelled as inline PTX in
C, plus the five protocol phases, so that the two kernel families execute
the same memory-ordering instructions and the same fixed-order float32
accumulation and produce bit-identical output. The GPU posting of RDMA work
(the roadmap's stage 1) goes into this header too, called from the doorbell
phase.

Everything that is a compile-time constant of one launcher specialization
(world size, rank, slot count, flag lanes, flag stride) is emitted as a
``#define`` by ``render_source`` so the source loops unroll. The functions
take pointers: the TileLang kernels obtain them with ``T.address_of`` on their
device buffers and with ``roce_ptr`` on the pinned region's base address.
"""

from __future__ import annotations

PACK_BYTES = 16

# Functions the kernels call. Kept in one place so a test can check the
# header defines each of them.
FUNCTIONS = (
    "roce_ld_relaxed_gpu_u32",
    "roce_ld_relaxed_sys_u32",
    "roce_atomic_add_relaxed_gpu_u32",
    "roce_st_release_gpu_u32",
    "roce_st_relaxed_sys_u32",
    "roce_fence_sc_sys",
    "roce_fence_sc_gpu",
    "roce_spin_until_eq_acquire_sys",
    "roce_ptr",
    "roce_copy_pack",
    "roce_doorbell",
    "roce_wait_flags",
    "roce_tail",
    "roce_reduce_pack_f32",
    "roce_reduce_pack_f16",
    "roce_reduce_pack_bf16",
    "roce_gather_pack",
)

_HEADER = r"""
// sparknet one-shot collectives: device protocol helpers (generated per launcher).
// Payload slots and flags live in pinned host memory that the NIC writes and
// the GPU reads in place, so every access to them carries system scope. The
// functions take pointers; the TileLang kernels pass T.address_of of their
// buffer arguments.
#define ROCE_PACK_BYTES 16

typedef unsigned int roce_u32;
typedef unsigned char roce_byte;
struct roce_pack { roce_u32 w[4]; };

// The pinned host region reaches the kernel as a base address (TileLang's FFI
// takes only device tensors as buffer arguments); this turns it into a pointer.
__device__ __forceinline__ roce_byte *roce_ptr(long long address) { return (roce_byte *)(size_t)address; }

__device__ __forceinline__ roce_u32 roce_ld_relaxed_gpu_u32(const void *p) {
    roce_u32 v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ roce_u32 roce_ld_relaxed_sys_u32(const void *p) {
    roce_u32 v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ roce_u32 roce_atomic_add_relaxed_gpu_u32(void *p, roce_u32 x) {
    roce_u32 prior;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(prior) : "l"(p), "r"(x) : "memory");
    return prior;
}

__device__ __forceinline__ void roce_st_release_gpu_u32(void *p, roce_u32 x) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(x) : "memory");
}

__device__ __forceinline__ void roce_st_relaxed_sys_u32(void *p, roce_u32 x) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(p), "r"(x) : "memory");
}

__device__ __forceinline__ void roce_fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

__device__ __forceinline__ void roce_fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

// Spin until the word at p equals expected (system scope). Returns 0 on
// success and 1 after limit polls without a match, so a dead peer or proxy
// surfaces as an error instead of a hung kernel.
__device__ __forceinline__ roce_u32 roce_spin_until_eq_acquire_sys(const void *p, roce_u32 expected, roce_u32 limit) {
    roce_u32 polls = 0;
    for (;;) {
        roce_u32 seen;
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(p) : "memory");
        if (seen == expected) {
            return 0u;
        }
        polls += 1u;
        if (polls >= limit) {
            return 1u;
        }
    }
}

__device__ __forceinline__ roce_pack roce_ld_global_pack(const void *src) {
    roce_pack p;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(p.w[0]), "=r"(p.w[1]), "=r"(p.w[2]), "=r"(p.w[3]) : "l"(src) : "memory");
    return p;
}

// One 16-byte pack from NIC-written pinned memory, never cached.
__device__ __forceinline__ roce_pack roce_ld_relaxed_sys_pack(const void *src) {
    roce_pack p;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(p.w[0]), "=r"(p.w[1]), "=r"(p.w[2]), "=r"(p.w[3]) : "l"(src) : "memory");
    return p;
}

__device__ __forceinline__ void roce_st_global_pack(void *dst, roce_pack p) {
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(dst), "r"(p.w[0]), "r"(p.w[1]), "r"(p.w[2]), "r"(p.w[3]) : "memory");
}

__device__ __forceinline__ void roce_copy_pack(const void *src, void *dst) {
    roce_st_global_pack(dst, roce_ld_global_pack(src));
}

// Conversions spelled exactly as the CuTe DSL kernels spell them, so both
// families round the same way.
__device__ __forceinline__ float roce_u32_as_f32(roce_u32 w) { return __uint_as_float(w); }
__device__ __forceinline__ roce_u32 roce_f32_as_u32(float f) { return __float_as_uint(f); }

__device__ __forceinline__ void roce_unpack_bf16x2(roce_u32 w, float &lo, float &hi) {
    asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.bf16 %0, l;\n\tcvt.f32.bf16 %1, h;\n}"
        : "=f"(lo), "=f"(hi) : "r"(w));
}

__device__ __forceinline__ void roce_unpack_f16x2(roce_u32 w, float &lo, float &hi) {
    asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %2;\n\tcvt.f32.f16 %0, l;\n\tcvt.f32.f16 %1, h;\n}"
        : "=f"(lo), "=f"(hi) : "r"(w));
}

__device__ __forceinline__ roce_u32 roce_pack_f32x2_to_bf16x2(float lo, float hi) {
    roce_u32 w;
    asm("{\n\t.reg .b16 l, h;\n\tcvt.rn.bf16.f32 l, %1;\n\tcvt.rn.bf16.f32 h, %2;\n\tmov.b32 %0, {l, h};\n}"
        : "=r"(w) : "f"(lo), "f"(hi));
    return w;
}

__device__ __forceinline__ roce_u32 roce_pack_f32x2_to_f16x2(float lo, float hi) {
    roce_u32 w;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(w) : "f"(lo), "f"(hi));
    return w;
}

// Phase 2: the last block to finish staging publishes nbytes (per slot) and
// then seq in the control record the proxy thread polls.
__device__ __forceinline__ void roce_doorbell(void *stage_counter, roce_u32 grid, roce_byte *ctrl,
                                              roce_u32 nbytes, roce_u32 seq) {
    roce_fence_sc_sys();
    roce_u32 prior = roce_atomic_add_relaxed_gpu_u32(stage_counter, 1u);
    if ((prior + 1u) % grid == 0u) {
        long long slot = (long long)(seq & 1u);
        roce_st_relaxed_sys_u32(ctrl + 4, nbytes);
        roce_st_relaxed_sys_u32(ctrl + 16 + slot * 4, nbytes);
        roce_fence_sc_sys();
        roce_st_relaxed_sys_u32(ctrl, seq);
    }
}

// Phase 3: one thread per (peer, flag lane) waits for that lane's flag. A
// timeout records seq, the peer and the lane in the control record and
// poisons the runtime.
__device__ __forceinline__ void roce_wait_flags(int tidx, const roce_byte *flags, roce_u32 seq, roce_u32 spin_limit,
                                                roce_byte *ctrl, void *poison) {
    if (tidx < ROCE_WORLD * ROCE_HCA_COUNT) {
        int peer = tidx / ROCE_HCA_COUNT;
        int hca = tidx - peer * ROCE_HCA_COUNT;
        bool active = peer != ROCE_RANK;
#if ROCE_NEIGHBOR_LANES != ROCE_HCA_COUNT
        active = active && ((peer == (ROCE_RANK + 2) % ROCE_WORLD) || (hca < ROCE_NEIGHBOR_LANES));
#endif
        if (active) {
            long long slot = (long long)(seq & 1u);
            const roce_byte *flag = flags + (((long long)peer * ROCE_SLOTS + slot) * ROCE_HCA_COUNT + hca) * ROCE_FLAG_STRIDE;
            roce_u32 timed_out = roce_spin_until_eq_acquire_sys(flag, seq, spin_limit);
            if (timed_out != 0u) {
                roce_st_relaxed_sys_u32(ctrl + 12, (roce_u32)peer);
                roce_st_relaxed_sys_u32(ctrl + 24, (roce_u32)hca);
                roce_st_relaxed_sys_u32(ctrl + 8, seq);
                roce_st_release_gpu_u32(poison, seq);
            }
        }
    }
}

// Phase 5: the last block to finish publishes the next epoch, unless a
// timeout was recorded (a failed sequence keeps the epoch).
__device__ __forceinline__ void roce_tail(void *tail_counter, roce_u32 grid, const roce_byte *ctrl,
                                          void *epoch, roce_u32 seq) {
    roce_u32 prior = roce_atomic_add_relaxed_gpu_u32(tail_counter, 1u);
    if ((prior + 1u) % grid == 0u) {
        roce_fence_sc_gpu();
        if (roce_ld_relaxed_sys_u32(ctrl + 8) == 0u) {
            roce_st_release_gpu_u32(epoch, seq);
        }
    }
}

// The pack at byte offset ``offset`` of ``source``: the local input, or the peer's receive slot.
__device__ __forceinline__ roce_pack roce_source_pack(int source, const void *input, const void *recv,
                                                      long long slot_bytes, roce_u32 seq, long long offset) {
    if (source == ROCE_RANK) {
        return roce_ld_global_pack((const roce_byte *)input + offset);
    }
    long long slot = (long long)(seq & 1u);
    return roce_ld_relaxed_sys_pack((const roce_byte *)recv + ((long long)source * ROCE_SLOTS + slot) * slot_bytes + offset);
}

// Phase 4 (all-reduce): sum the local input and every peer slot in fixed rank
// order with float32 accumulation, so every rank stores identical bits.
__device__ __forceinline__ void roce_reduce_pack_f32(const void *input, const void *recv, long long slot_bytes,
                                                     roce_u32 seq, void *output, long long offset) {
    float acc[4];
#pragma unroll
    for (int source = 0; source < ROCE_WORLD; source++) {
        roce_pack p = roce_source_pack(source, input, recv, slot_bytes, seq, offset);
#pragma unroll
        for (int w = 0; w < 4; w++) {
            float value = roce_u32_as_f32(p.w[w]);
            if (source == 0) {
                acc[w] = value;
            } else {
                acc[w] = acc[w] + value;
            }
        }
    }
    roce_pack out;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        out.w[w] = roce_f32_as_u32(acc[w]);
    }
    roce_st_global_pack((roce_byte *)output + offset, out);
}

// Phase 4 (all-reduce, float16): the same fixed-order float32 accumulation
// on pairs of 16-bit values.
__device__ __forceinline__ void roce_reduce_pack_f16(const void *input, const void *recv, long long slot_bytes,
                                       roce_u32 seq, void *output, long long offset) {
    float acc[8];
#pragma unroll
    for (int source = 0; source < ROCE_WORLD; source++) {
        roce_pack p = roce_source_pack(source, input, recv, slot_bytes, seq, offset);
#pragma unroll
        for (int w = 0; w < 4; w++) {
            float lo, hi;
            roce_unpack_f16x2(p.w[w], lo, hi);
            if (source == 0) {
                acc[2 * w] = lo;
                acc[2 * w + 1] = hi;
            } else {
                acc[2 * w] = acc[2 * w] + lo;
                acc[2 * w + 1] = acc[2 * w + 1] + hi;
            }
        }
    }
    roce_pack out;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        out.w[w] = roce_pack_f32x2_to_f16x2(acc[2 * w], acc[2 * w + 1]);
    }
    roce_st_global_pack((roce_byte *)output + offset, out);
}

// Phase 4 (all-reduce, bfloat16): the same fixed-order float32 accumulation
// on pairs of 16-bit values.
__device__ __forceinline__ void roce_reduce_pack_bf16(const void *input, const void *recv, long long slot_bytes,
                                       roce_u32 seq, void *output, long long offset) {
    float acc[8];
#pragma unroll
    for (int source = 0; source < ROCE_WORLD; source++) {
        roce_pack p = roce_source_pack(source, input, recv, slot_bytes, seq, offset);
#pragma unroll
        for (int w = 0; w < 4; w++) {
            float lo, hi;
            roce_unpack_bf16x2(p.w[w], lo, hi);
            if (source == 0) {
                acc[2 * w] = lo;
                acc[2 * w + 1] = hi;
            } else {
                acc[2 * w] = acc[2 * w] + lo;
                acc[2 * w + 1] = acc[2 * w + 1] + hi;
            }
        }
    }
    roce_pack out;
#pragma unroll
    for (int w = 0; w < 4; w++) {
        out.w[w] = roce_pack_f32x2_to_bf16x2(acc[2 * w], acc[2 * w + 1]);
    }
    roce_st_global_pack((roce_byte *)output + offset, out);
}

// Phase 4 (all-gather): shard s of pack copy_index lands at column block s of
// its output row (row_packs == shard_packs for a dim-0 concatenation).
__device__ __forceinline__ void roce_gather_pack(const void *input, const void *recv, long long slot_bytes,
                                                 roce_u32 seq, int copy_index, int row_packs, void *output) {
    int row = copy_index / row_packs;
    int col = copy_index - row * row_packs;
    long long out_row_packs = (long long)ROCE_WORLD * row_packs;
    long long offset = (long long)copy_index * ROCE_PACK_BYTES;
#pragma unroll
    for (int source = 0; source < ROCE_WORLD; source++) {
        roce_pack p = roce_source_pack(source, input, recv, slot_bytes, seq, offset);
        roce_byte *dest = (roce_byte *)output + ((long long)row * out_row_packs + (long long)source * row_packs + col) * ROCE_PACK_BYTES;
        roce_st_global_pack(dest, p);
    }
}
"""


def render_source(*, world_size: int, rank: int, slots: int, flag_stride: int, hca_count: int, neighbor_lanes: int) -> str:
    """The header for one launcher specialization, constants first."""
    defines = {
        "ROCE_WORLD": int(world_size),
        "ROCE_RANK": int(rank),
        "ROCE_SLOTS": int(slots),
        "ROCE_FLAG_STRIDE": int(flag_stride),
        "ROCE_HCA_COUNT": int(hca_count),
        "ROCE_NEIGHBOR_LANES": int(neighbor_lanes),
    }
    return "".join(f"#define {name} {value}\n" for name, value in defines.items()) + _HEADER


__all__ = ["FUNCTIONS", "PACK_BYTES", "render_source"]
