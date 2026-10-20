"""The one-shot protocol's device side as CUDA C, for kernels that are generated as CUDA source.

The TileLang kernels (``_oneshot_tilelang``, ``_allgather_tilelang``) inject
this header with ``T.import_source`` and call its functions by name: the
protocol's memory-ordering instructions as inline PTX, its fixed-order float32
accumulation and conversions, and the five protocol phases. (Until 0.4.0 a
vendored CuTe DSL pair executed the same instructions bit for bit.) The GPU
posting of RDMA work
(the roadmap's stage 1) goes into this header too, called from the doorbell
phase.

Everything that is a compile-time constant of one launcher specialization
(world size, rank, slot count, flag lanes, flag stride) is emitted as a
``#define`` by ``render_source`` so the source loops unroll. The functions
take pointers: the TileLang kernels obtain them with ``T.address_of`` on their
device buffers and with ``roce_ptr`` on the pinned region's base address.
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
    "roce_wait_until_eq_acquire_sys",
    "roce_trap",
    "roce_ptr",
    "roce_copy_pack",
    "roce_doorbell",
    "roce_wait_flags",
    "roce_wait_device_eq",
    "roce_wait_peers",
    "roce_tail",
    "roce_reduce_pack_f32",
    "roce_reduce_pack_f16",
    "roce_reduce_pack_bf16",
    "roce_gather_dest",
    "roce_stage_gather_pack",
    "roce_gather_pack",
    "roce_globaltimer",
    "roce_trace_record",
    "roce_trace_store",
    "roce_trace_record_if",
    "roce_trace_begin",
    "roce_trace_stamp",
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

// Trace (SPARKNET_ROCE_TRACE): %globaltimer stamps into the op's record of the
// pinned trace file (sparknet.oneshot.trace).
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

// Control-record words beside the doorbell's: the wait timeout the host sets
// (microseconds), and the time a timed-out wait had waited.
#define ROCE_CTRL_TIMEOUT_US 28
#define ROCE_CTRL_WAITED_US 32

// Wait until the word at p equals expected (system scope). Returns 0 on a
// match, else the microseconds waited (at least 1) once the timeout in the
// control record has passed. The clock and the timeout are read only after 64
// unanswered polls, so a wait that ends quickly costs nothing extra.
__device__ __forceinline__ roce_u32 roce_wait_until_eq_acquire_sys(const void *p, roce_u32 expected,
                                                                   const roce_byte *ctrl) {
    unsigned long long start = 0ull, limit = 0ull;
    for (roce_u32 polls = 1u;; polls++) {
        roce_u32 seen;
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(p) : "memory");
        if (seen == expected) {
            return 0u;
        }
        if ((polls & 63u) == 0u) {
            unsigned long long now = roce_globaltimer();
            if (start == 0ull) {
                start = now;
                limit = 1000ull * (unsigned long long)roce_ld_relaxed_sys_u32(ctrl + ROCE_CTRL_TIMEOUT_US);
            } else if (now - start >= limit) {
                roce_u32 waited = (roce_u32)((now - start) / 1000ull);
                return waited == 0u ? 1u : waited;
            }
        }
    }
}

// Stop the GPU: a timed-out collective leaves this rank's data unusable, and
// nothing after it may run on it. The host reads the control record, which
// the caller wrote and fenced first.
__device__ __forceinline__ void roce_trap() { asm volatile("trap;"); }

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

// Conversions: round-to-nearest-even packing of the float32 sums.
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
// then seq in the control record the proxy thread polls; returns 1 in that
// block.
__device__ __forceinline__ roce_u32 roce_doorbell(void *stage_counter, roce_u32 grid, roce_byte *ctrl,
                                                  roce_u32 nbytes, roce_u32 seq) {
    roce_fence_sc_sys();
    roce_u32 prior = roce_atomic_add_relaxed_gpu_u32(stage_counter, 1u);
    if ((prior + 1u) % grid == 0u) {
        long long slot = (long long)(seq & 1u);
        roce_st_relaxed_sys_u32(ctrl + 4, nbytes);
        roce_st_relaxed_sys_u32(ctrl + 16 + slot * 4, nbytes);
        roce_fence_sc_sys();
        roce_st_relaxed_sys_u32(ctrl, seq);
        return 1u;
    }
    return 0u;
}

// Phase 3: one thread per (peer, flag lane) waits for that lane's flag. A
// timeout writes the missing peer, its lane and the time waited, then the
// failed sequence, into the control record (the proxy stops relaying when it
// sees that sequence, and the host reports the record), and stops the GPU. A
// traced kernel passes block 0's trace record, which gets every lane's
// arrival time.
__device__ __forceinline__ void roce_wait_flags(int tidx, const roce_byte *flags, roce_u32 seq, roce_byte *ctrl,
                                                roce_byte *trace_record = nullptr) {
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
            roce_u32 waited = roce_wait_until_eq_acquire_sys(flag, seq, ctrl);
#if ROCE_TRACE
            if (trace_record != nullptr && waited == 0u) {
                roce_trace_store(trace_record, ROCE_TRACE_W_FLAGS + peer * ROCE_HCA_COUNT + hca, roce_globaltimer());
            }
#endif
            if (waited != 0u) {
                roce_st_relaxed_sys_u32(ctrl + 12, (roce_u32)peer);
                roce_st_relaxed_sys_u32(ctrl + 24, (roce_u32)hca);
                roce_st_relaxed_sys_u32(ctrl + ROCE_CTRL_WAITED_US, waited);
                roce_fence_sc_sys();
                roce_st_relaxed_sys_u32(ctrl + 8, seq);
                roce_fence_sc_sys();
                roce_trap();
            }
        }
    }
}

// Wait until a device-memory word equals seq (gpu scope), bounded by the
// control record's timeout like the flag waits; a timed-out wait stops the GPU.
__device__ __forceinline__ void roce_wait_device_eq(const void *word, roce_u32 seq, const roce_byte *ctrl) {
    unsigned long long start = 0ull, limit = 0ull;
    for (roce_u32 polls = 1u;; polls++) {
        roce_u32 seen;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(word) : "memory");
        if (seen == seq) {
            return;
        }
        if ((polls & 63u) == 0u) {
            unsigned long long now = roce_globaltimer();
            if (start == 0ull) {
                start = now;
                limit = 1000ull * (unsigned long long)roce_ld_relaxed_sys_u32(ctrl + ROCE_CTRL_TIMEOUT_US);
            } else if (now - start >= limit) {
                roce_trap();
            }
        }
    }
}

// Phase 3 for every block: block 0 polls the peers' flags, and only once the
// doorbell has rung (``rung``, set by the block that rang it). A system-scope
// poll of the pinned region slows the system-scope stores of blocks still
// staging, and every poller is one more, so no other block polls it: they wait
// in device memory for block 0's ``arrival`` word, which block 0 sets after its
// barrier. Block 0's flag wait records a missing peer and stops the GPU; the
// device-memory waits share its timeout and stop the GPU only after it.
__device__ __forceinline__ void roce_wait_peers(int block, int tidx, const roce_byte *flags, roce_u32 seq,
                                                roce_byte *ctrl, const void *rung, const void *arrival,
                                                roce_byte *trace_record = nullptr) {
    if (block == 0) {
        if (tidx < ROCE_WORLD * ROCE_HCA_COUNT) {
            roce_wait_device_eq(rung, seq, ctrl);
            roce_wait_flags(tidx, flags, seq, ctrl, trace_record);
        }
    } else if (tidx == 0) {
        roce_wait_device_eq(arrival, seq, ctrl);
    }
}

// Phase 5: the last block to finish publishes the next epoch; returns 1 in
// that block.
__device__ __forceinline__ roce_u32 roce_tail(void *tail_counter, roce_u32 grid, void *epoch, roce_u32 seq) {
    roce_u32 prior = roce_atomic_add_relaxed_gpu_u32(tail_counter, 1u);
    if ((prior + 1u) % grid == 0u) {
        roce_fence_sc_gpu();
        roce_st_release_gpu_u32(epoch, seq);
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

// The output pack of source ``source`` for gather pack ``copy_index``: shard s
// occupies column block s of every output row (row_packs == shard_packs for a
// dim-0 concatenation).
__device__ __forceinline__ roce_byte *roce_gather_dest(void *output, int copy_index, int row_packs, int source) {
    int row = copy_index / row_packs;
    int col = copy_index - row * row_packs;
    return (roce_byte *)output + (((long long)row * ROCE_WORLD + source) * row_packs + col) * ROCE_PACK_BYTES;
}

// Phase 1 (all-gather): stage one pack of the local shard into the send slot
// and write it to this rank's column block of the output, so the data phase
// reads only the peers.
__device__ __forceinline__ void roce_stage_gather_pack(const void *input, void *send, int copy_index, int row_packs,
                                                       void *output) {
    long long offset = (long long)copy_index * ROCE_PACK_BYTES;
    roce_pack p = roce_ld_global_pack((const roce_byte *)input + offset);
    roce_st_global_pack((roce_byte *)send + offset, p);
    roce_st_global_pack(roce_gather_dest(output, copy_index, row_packs, ROCE_RANK), p);
}

// Phase 4 (all-gather): copy pack copy_index of every peer's shard to its
// column block. The loads of a group of peers are all issued before its first
// store: a store orders the loads after it, so a load-store pair per peer
// would make every peer a round trip of its own to the pinned region. Groups
// of four bound the registers.
#define ROCE_GATHER_GROUP 4
__device__ __forceinline__ void roce_gather_pack(const void *recv, long long slot_bytes, roce_u32 seq, int copy_index,
                                                 int row_packs, void *output) {
    long long offset = (long long)copy_index * ROCE_PACK_BYTES;
    long long slot = (long long)(seq & 1u);
#pragma unroll
    for (int first = 0; first < ROCE_WORLD - 1; first += ROCE_GATHER_GROUP) {
        roce_pack p[ROCE_GATHER_GROUP];
#pragma unroll
        for (int k = 0; k < ROCE_GATHER_GROUP; k++) {
            int peer = (ROCE_RANK + 1 + first + k) % ROCE_WORLD;
            if (first + k < ROCE_WORLD - 1) {
                p[k] = roce_ld_relaxed_sys_pack((const roce_byte *)recv + ((long long)peer * ROCE_SLOTS + slot) * slot_bytes + offset);
            }
        }
#pragma unroll
        for (int k = 0; k < ROCE_GATHER_GROUP; k++) {
            int peer = (ROCE_RANK + 1 + first + k) % ROCE_WORLD;
            if (first + k < ROCE_WORLD - 1) {
                roce_st_global_pack(roce_gather_dest(output, copy_index, row_packs, peer), p[k]);
            }
        }
    }
}
"""


def render_source(*, world_size: int, rank: int, slots: int, flag_stride: int, hca_count: int, neighbor_lanes: int,
                  trace: Mapping[str, int] | None = None) -> str:
    """The header for one launcher specialization, constants first.

    ``trace`` is the trace record layout (``sparknet.oneshot.trace.device_defines()``)
    for a traced kernel.
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
    return "".join(f"#define {name} {value}\n" for name, value in defines.items()) + _HEADER


__all__ = ["FUNCTIONS", "PACK_BYTES", "render_source"]
