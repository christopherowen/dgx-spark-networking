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

``render_source(fused=...)`` adds the fused halves (``roce_fused_*``): the
send half (stage into the send slot, ring the doorbell) for a producer kernel
and the receive half (wait, reduce, advance the epoch) for a consumer kernel,
with one runtime's region layout and counter indices as constants. The
runtime's ``device_header()`` renders it; ``docs/oneshot.md`` gives the
calling contract.
"""

from __future__ import annotations

from collections.abc import Mapping

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
    "roce_globaltimer",
    "roce_trace_record",
    "roce_trace_store",
    "roce_trace_record_if",
    "roce_trace_begin",
    "roce_trace_stamp",
)

# The fused halves, present only in a header rendered with ``fused=``.
FUSED_FUNCTIONS = (
    "roce_fused_seq",
    "roce_fused_poisoned",
    "roce_fused_send_slot",
    "roce_fused_send_slot_at",
    "roce_fused_arrive_last",
    "roce_fused_send_commit",
    "roce_fused_wait",
    "roce_fused_sum_pack_f32",
    "roce_fused_sum_pack_f16",
    "roce_fused_sum_pack_bf16",
    "roce_fused_reduce_pack_f32",
    "roce_fused_reduce_pack_f16",
    "roce_fused_reduce_pack_bf16",
    "roce_fused_receive_tail",
)

# The constants ``fused=`` must carry: the runtime's region layout and counter indices.
FUSED_CONSTANTS = (
    "ROCE_SEND_OFF",
    "ROCE_RECV_OFF",
    "ROCE_FLAG_OFF",
    "ROCE_CTRL_OFF",
    "ROCE_SLOT_BYTES",
    "ROCE_POISON_INDEX",
    "ROCE_FUSED_STAGE_INDEX",
    "ROCE_FUSED_TAIL_INDEX",
    "ROCE_SPIN_LIMIT",
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

__device__ __forceinline__ void roce_st_relaxed_gpu_u32(void *p, roce_u32 x) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(x) : "memory");
}

// Trace (SPARKNET_ROCE_TRACE): %globaltimer stamps into the op's record of the
// pinned trace file, as the CuTe kernels write them (sparknet.oneshot.trace).
__device__ __forceinline__ unsigned long long roce_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

#if ROCE_TRACE
__device__ __forceinline__ roce_byte *roce_trace_record(long long trace_base, roce_u32 seq) {
    return roce_ptr(trace_base) + ROCE_TRACE_HEADER_BYTES
        + (long long)(seq & (roce_u32)(ROCE_TRACE_RECORDS - 1)) * ROCE_TRACE_RECORD_BYTES;
}

__device__ __forceinline__ void roce_trace_store(roce_byte *record, int word, unsigned long long value) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(record + 8 * (long long)word), "l"(value) : "memory");
}

// Block 0's record for the flag arrivals, nothing for the other blocks.
__device__ __forceinline__ roce_byte *roce_trace_record_if(long long trace_base, roce_u32 seq, int take) {
    return take ? roce_trace_record(trace_base, seq) : nullptr;
}

// The op's first words: start time, sequence, and nbytes | grid << 32.
__device__ __forceinline__ void roce_trace_begin(long long trace_base, roce_u32 seq, roce_u32 nbytes, roce_u32 grid) {
    roce_byte *record = roce_trace_record(trace_base, seq);
    roce_trace_store(record, ROCE_TRACE_W_START, roce_globaltimer());
    roce_trace_store(record, ROCE_TRACE_W_SEQ, (unsigned long long)seq);
    roce_trace_store(record, ROCE_TRACE_W_META, (unsigned long long)nbytes | ((unsigned long long)grid << 32));
}

// The current time into one word of the op's record.
__device__ __forceinline__ void roce_trace_stamp(long long trace_base, roce_u32 seq, int word) {
    roce_trace_store(roce_trace_record(trace_base, seq), word, roce_globaltimer());
}
#endif

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

// The doorbell itself: nbytes (per slot), then seq, in the control record the
// proxy thread polls.
__device__ __forceinline__ void roce_ring(roce_byte *ctrl, roce_u32 nbytes, roce_u32 seq) {
    long long slot = (long long)(seq & 1u);
    roce_st_relaxed_sys_u32(ctrl + 4, nbytes);
    roce_st_relaxed_sys_u32(ctrl + 16 + slot * 4, nbytes);
    roce_fence_sc_sys();
    roce_st_relaxed_sys_u32(ctrl, seq);
}

// Phase 2: the last block to finish staging rings the doorbell; returns 1 in
// that block.
__device__ __forceinline__ roce_u32 roce_doorbell(void *stage_counter, roce_u32 grid, roce_byte *ctrl,
                                                  roce_u32 nbytes, roce_u32 seq) {
    roce_fence_sc_sys();
    roce_u32 prior = roce_atomic_add_relaxed_gpu_u32(stage_counter, 1u);
    if ((prior + 1u) % grid == 0u) {
        roce_ring(ctrl, nbytes, seq);
        return 1u;
    }
    return 0u;
}

// Phase 3: one thread per (peer, flag lane) waits for that lane's flag. A
// timeout records seq, the peer and the lane in the control record and
// poisons the runtime. A traced kernel passes block 0's trace record, which
// gets every lane's arrival time.
__device__ __forceinline__ void roce_wait_flags(int tidx, const roce_byte *flags, roce_u32 seq, roce_u32 spin_limit,
                                                roce_byte *ctrl, void *poison, roce_byte *trace_record = nullptr) {
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
#if ROCE_TRACE
            if (trace_record != nullptr && timed_out == 0u) {
                roce_trace_store(trace_record, ROCE_TRACE_W_FLAGS + peer * ROCE_HCA_COUNT + hca, roce_globaltimer());
            }
#endif
            if (timed_out != 0u) {
                roce_st_relaxed_sys_u32(ctrl + 12, (roce_u32)peer);
                roce_st_relaxed_sys_u32(ctrl + 24, (roce_u32)hca);
                roce_st_relaxed_sys_u32(ctrl + 8, seq);
                roce_st_release_gpu_u32(poison, seq);
            }
        }
    }
}

// The epoch advance: every block's timeout store precedes its tail arrival,
// so the error word is final here, and a failed sequence keeps the epoch.
__device__ __forceinline__ void roce_publish_epoch(const roce_byte *ctrl, void *epoch, roce_u32 seq) {
    roce_fence_sc_gpu();
    if (roce_ld_relaxed_sys_u32(ctrl + 8) == 0u) {
        roce_st_release_gpu_u32(epoch, seq);
    }
}

// Phase 5: the last block to finish publishes the next epoch, unless a
// timeout was recorded; returns 1 in that block.
__device__ __forceinline__ roce_u32 roce_tail(void *tail_counter, roce_u32 grid, const roce_byte *ctrl,
                                              void *epoch, roce_u32 seq) {
    roce_u32 prior = roce_atomic_add_relaxed_gpu_u32(tail_counter, 1u);
    if ((prior + 1u) % grid == 0u) {
        roce_publish_epoch(ctrl, epoch, seq);
        return 1u;
    }
    return 0u;
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
// order with float32 accumulation, so every rank stores identical bits. The
// sum functions return the reduced pack; the reduce functions store it.
__device__ __forceinline__ roce_pack roce_sum_pack_f32(const void *input, const void *recv, long long slot_bytes,
                                                       roce_u32 seq, long long offset) {
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
    return out;
}

// Phase 4 (all-reduce, float16): the same fixed-order float32 accumulation
// on pairs of 16-bit values.
__device__ __forceinline__ roce_pack roce_sum_pack_f16(const void *input, const void *recv, long long slot_bytes,
                                                       roce_u32 seq, long long offset) {
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
    return out;
}

// Phase 4 (all-reduce, bfloat16): the same fixed-order float32 accumulation
// on pairs of 16-bit values.
__device__ __forceinline__ roce_pack roce_sum_pack_bf16(const void *input, const void *recv, long long slot_bytes,
                                                        roce_u32 seq, long long offset) {
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
    return out;
}

__device__ __forceinline__ void roce_reduce_pack_f32(const void *input, const void *recv, long long slot_bytes,
                                                     roce_u32 seq, void *output, long long offset) {
    roce_st_global_pack((roce_byte *)output + offset, roce_sum_pack_f32(input, recv, slot_bytes, seq, offset));
}

__device__ __forceinline__ void roce_reduce_pack_f16(const void *input, const void *recv, long long slot_bytes,
                                                     roce_u32 seq, void *output, long long offset) {
    roce_st_global_pack((roce_byte *)output + offset, roce_sum_pack_f16(input, recv, slot_bytes, seq, offset));
}

__device__ __forceinline__ void roce_reduce_pack_bf16(const void *input, const void *recv, long long slot_bytes,
                                                      roce_u32 seq, void *output, long long offset) {
    roce_st_global_pack((roce_byte *)output + offset, roce_sum_pack_bf16(input, recv, slot_bytes, seq, offset));
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

#ifdef ROCE_FUSED
// Fused halves of the one-shot all-reduce, for kernels outside sparknet. A
// producer kernel runs the send half in place of the standalone kernel's
// staging; a consumer kernel runs the receive half in place of its wait and
// reduce. Between the two halves no other collective of this runtime runs:
// the runtime's fused_send()/receive() enforce that on the host. Both halves
// take the sequence from the epoch, which only the receive half advances.
// Their arrival counters reset themselves, so any grid size works.

// This op's sequence number (the epoch is the last completed one).
__device__ __forceinline__ roce_u32 roce_fused_seq(const void *counters) {
    return roce_ld_relaxed_gpu_u32(counters) + 1u;
}

// 1 once a wait timed out anywhere in this runtime. Both halves then skip the
// protocol; the host raises at its next check.
__device__ __forceinline__ roce_u32 roce_fused_poisoned(const void *counters) {
    return roce_ld_relaxed_gpu_u32((const roce_u32 *)counters + ROCE_POISON_INDEX) != 0u ? 1u : 0u;
}

// Where the producer writes the op's payload (16-byte aligned), in the input
// dtype's layout: the bytes the standalone kernel would have staged.
__device__ __forceinline__ roce_byte *roce_fused_send_slot(long long region_base, roce_u32 seq) {
    return roce_ptr(region_base) + ROCE_SEND_OFF + (long long)(seq & 1u) * ROCE_SLOT_BYTES;
}

// The send slot at byte offset ``offset``, for generators without pointer arithmetic.
__device__ __forceinline__ roce_byte *roce_fused_send_slot_at(long long region_base, roce_u32 seq, long long offset) {
    return roce_fused_send_slot(region_base, seq) + offset;
}

// One block's arrival at a self-resetting counter: the last of grid blocks
// resets it and gets 1. Stream order separates one op's arrivals from the next.
__device__ __forceinline__ roce_u32 roce_fused_arrive_last(void *counter, roce_u32 grid) {
    roce_u32 prior = roce_atomic_add_relaxed_gpu_u32(counter, 1u);
    if (prior + 1u == grid) {
        roce_st_relaxed_gpu_u32(counter, 0u);
        return 1u;
    }
    return 0u;
}

// Send half: one thread of every producer block, after a __syncthreads() that
// follows the block's last store into the send slot. The last block rings the
// doorbell and gets 1.
__device__ __forceinline__ roce_u32 roce_fused_send_commit(void *counters, roce_u32 grid, long long region_base,
                                                          roce_u32 nbytes, roce_u32 seq) {
    roce_fence_sc_sys();
    if (roce_fused_arrive_last((roce_u32 *)counters + ROCE_FUSED_STAGE_INDEX, grid)) {
        roce_ring(roce_ptr(region_base) + ROCE_CTRL_OFF, nbytes, seq);
        return 1u;
    }
    return 0u;
}

// Receive half, wait: every thread of every consumer block, then
// __syncthreads(). A block that then sees roce_fused_poisoned() skips the
// slots (its output is garbage; the host raises).
__device__ __forceinline__ void roce_fused_wait(int tidx, long long region_base, roce_u32 seq, void *counters,
                                                roce_byte *trace_record = nullptr) {
    roce_wait_flags(tidx, roce_ptr(region_base) + ROCE_FLAG_OFF, seq, (roce_u32)ROCE_SPIN_LIMIT,
                    roce_ptr(region_base) + ROCE_CTRL_OFF, (roce_u32 *)counters + ROCE_POISON_INDEX, trace_record);
}

// Receive half, reduce: the all-reduced pack at byte offset ``offset``, summed
// in fixed rank order and rounded to the dtype: the bits the standalone
// all-reduce stores there. Use it in registers or store it.
__device__ __forceinline__ roce_pack roce_fused_sum_pack_f32(long long region_base, roce_u32 seq, long long offset) {
    return roce_sum_pack_f32(roce_fused_send_slot(region_base, seq), roce_ptr(region_base) + ROCE_RECV_OFF,
                             ROCE_SLOT_BYTES, seq, offset);
}

__device__ __forceinline__ roce_pack roce_fused_sum_pack_f16(long long region_base, roce_u32 seq, long long offset) {
    return roce_sum_pack_f16(roce_fused_send_slot(region_base, seq), roce_ptr(region_base) + ROCE_RECV_OFF,
                             ROCE_SLOT_BYTES, seq, offset);
}

__device__ __forceinline__ roce_pack roce_fused_sum_pack_bf16(long long region_base, roce_u32 seq, long long offset) {
    return roce_sum_pack_bf16(roce_fused_send_slot(region_base, seq), roce_ptr(region_base) + ROCE_RECV_OFF,
                              ROCE_SLOT_BYTES, seq, offset);
}

__device__ __forceinline__ void roce_fused_reduce_pack_f32(long long region_base, roce_u32 seq, void *output,
                                                           long long offset) {
    roce_st_global_pack((roce_byte *)output + offset, roce_fused_sum_pack_f32(region_base, seq, offset));
}

__device__ __forceinline__ void roce_fused_reduce_pack_f16(long long region_base, roce_u32 seq, void *output,
                                                           long long offset) {
    roce_st_global_pack((roce_byte *)output + offset, roce_fused_sum_pack_f16(region_base, seq, offset));
}

__device__ __forceinline__ void roce_fused_reduce_pack_bf16(long long region_base, roce_u32 seq, void *output,
                                                            long long offset) {
    roce_st_global_pack((roce_byte *)output + offset, roce_fused_sum_pack_bf16(region_base, seq, offset));
}

// Receive half, tail: one thread of every consumer block after the block's
// last read of the slots, a roce_fence_sc_gpu() and a __syncthreads(). The
// last block advances the epoch (unless a wait timed out) and gets 1.
__device__ __forceinline__ roce_u32 roce_fused_receive_tail(void *counters, roce_u32 grid, long long region_base,
                                                           roce_u32 seq) {
    if (roce_fused_arrive_last((roce_u32 *)counters + ROCE_FUSED_TAIL_INDEX, grid)) {
        roce_publish_epoch(roce_ptr(region_base) + ROCE_CTRL_OFF, counters, seq);
        return 1u;
    }
    return 0u;
}
#endif
"""


def render_source(*, world_size: int, rank: int, slots: int, flag_stride: int, hca_count: int, neighbor_lanes: int,
                  trace: Mapping[str, int] | None = None, fused: Mapping[str, int] | None = None) -> str:
    """The header for one launcher specialization, constants first.

    ``trace`` is the trace record layout (``sparknet.oneshot.trace.device_defines()``)
    for a traced kernel; ``fused`` carries every name of ``FUSED_CONSTANTS`` and
    adds the fused halves.
    """
    defines = {
        "ROCE_WORLD": int(world_size),
        "ROCE_RANK": int(rank),
        "ROCE_SLOTS": int(slots),
        "ROCE_FLAG_STRIDE": int(flag_stride),
        "ROCE_HCA_COUNT": int(hca_count),
        "ROCE_NEIGHBOR_LANES": int(neighbor_lanes),
        "ROCE_TRACE": 1 if trace else 0,
    }
    if trace:
        defines.update({name: int(value) for name, value in trace.items()})
    if fused is not None:
        missing = [name for name in FUSED_CONSTANTS if name not in fused]
        if missing:
            raise ValueError(f"fused header constants missing: {missing}")
        defines["ROCE_FUSED"] = 1
        defines.update({name: int(fused[name]) for name in FUSED_CONSTANTS})
    return "".join(f"#define {name} {value}\n" for name, value in defines.items()) + _HEADER


__all__ = ["FUNCTIONS", "FUSED_CONSTANTS", "FUSED_FUNCTIONS", "PACK_BYTES", "render_source"]
