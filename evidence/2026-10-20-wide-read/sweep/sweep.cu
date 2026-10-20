// Read-phase microbenchmark for the one-shot reduce: the sparknet device header's own
// roce_reduce_pack_bf16 (relaxed sys-scope loads from cudaHostAlloc memory, the local input
// from device memory), swept over grid, block size and packs in flight per thread, with and
// without an L2 bulk prefetch running beside it on another stream.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include HDR

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ unsigned long long gt() { return roce_globaltimer(); }

__device__ __forceinline__ void stamp_begin(unsigned long long *st) {
    if (threadIdx.x == 0) atomicMin(&st[0], gt());
}
__device__ __forceinline__ void stamp_end(unsigned long long *st) {
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&st[1], gt());
}

// v0: the kernel as committed: a grid-stride loop, one pack (all sources) at a time.
template <int T>
__global__ void __launch_bounds__(T, 1536 / T) k_pack(const void *inp, const roce_byte *recv, long long slot_bytes,
                                                      roce_u32 seq, void *out, int size_packs, unsigned long long *st) {
    stamp_begin(st);
    int stride = gridDim.x * T;
    for (int pack = blockIdx.x * T + threadIdx.x; pack < size_packs; pack += stride)
        roce_reduce_pack_bf16(inp, recv, slot_bytes, seq, out, (long long)pack * ROCE_PACK_BYTES);
    stamp_end(st);
}

// The same with ld.global.cv instead of the relaxed system-scope load (information only).
__device__ __forceinline__ roce_pack ld_cv_pack(const void *src) {
    roce_pack p;
    asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(p.w[0]), "=r"(p.w[1]), "=r"(p.w[2]), "=r"(p.w[3]) : "l"(src) : "memory");
    return p;
}

template <bool CV>
__device__ __forceinline__ roce_pack src_pack(int source, const void *input, const void *recv, long long slot_bytes,
                                              roce_u32 seq, long long offset) {
    if (source == ROCE_RANK) return roce_ld_global_pack((const roce_byte *)input + offset);
    long long slot = (long long)(seq & 1u);
    const void *p = (const roce_byte *)recv + ((long long)source * ROCE_SLOTS + slot) * slot_bytes + offset;
    return CV ? ld_cv_pack(p) : roce_ld_relaxed_sys_pack(p);
}

__device__ __forceinline__ void acc_bf16(float *acc, roce_pack p, bool first) {
#pragma unroll
    for (int w = 0; w < 4; w++) {
        float lo, hi;
        roce_unpack_bf16x2(p.w[w], lo, hi);
        acc[2 * w] = first ? lo : acc[2 * w] + lo;
        acc[2 * w + 1] = first ? hi : acc[2 * w + 1] + hi;
    }
}

__device__ __forceinline__ void store_bf16(void *out, long long offset, const float *acc) {
    roce_pack o;
#pragma unroll
    for (int w = 0; w < 4; w++) o.w[w] = roce_pack_f32x2_to_bf16x2(acc[2 * w], acc[2 * w + 1]);
    roce_st_global_pack((roce_byte *)out + offset, o);
}

// vK: K packs a thread, every source's loads for all K issued before any is consumed.
template <int T, int K, bool CV>
__global__ void __launch_bounds__(T, 1536 / T) k_multi(const void *inp, const roce_byte *recv, long long slot_bytes,
                                                       roce_u32 seq, void *out, int size_packs, unsigned long long *st) {
    stamp_begin(st);
    int stride = gridDim.x * T;
    for (int base = blockIdx.x * T + threadIdx.x; base < size_packs; base += K * stride) {
        roce_pack v[K][ROCE_WORLD];
#pragma unroll
        for (int s = 0; s < ROCE_WORLD; s++)
#pragma unroll
            for (int k = 0; k < K; k++)
                if (base + k * stride < size_packs)
                    v[k][s] = src_pack<CV>(s, inp, recv, slot_bytes, seq, (long long)(base + k * stride) * ROCE_PACK_BYTES);
#pragma unroll
        for (int k = 0; k < K; k++) {
            if (base + k * stride < size_packs) {
                float acc[8];
#pragma unroll
                for (int s = 0; s < ROCE_WORLD; s++) acc_bf16(acc, v[k][s], s == 0);
                store_bf16(out, (long long)(base + k * stride) * ROCE_PACK_BYTES, acc);
            }
        }
    }
    stamp_end(st);
}


// Gathers (dim 0: row_packs == shard_packs): the committed per-source load/store, and all loads first.
template <int T>
__global__ void __launch_bounds__(T, 1536 / T) k_gather(const void *inp, const roce_byte *recv, long long slot_bytes,
                                                        roce_u32 seq, void *out, int size_packs, unsigned long long *st) {
    stamp_begin(st);
    int stride = gridDim.x * T;
    for (int pack = blockIdx.x * T + threadIdx.x; pack < size_packs; pack += stride)
        roce_gather_pack(inp, recv, slot_bytes, seq, pack, size_packs, out);
    stamp_end(st);
}

template <int T>
__global__ void __launch_bounds__(T, 1536 / T) k_gather_first(const void *inp, const roce_byte *recv, long long slot_bytes,
                                                              roce_u32 seq, void *out, int size_packs, unsigned long long *st) {
    stamp_begin(st);
    int stride = gridDim.x * T;
    for (int pack = blockIdx.x * T + threadIdx.x; pack < size_packs; pack += stride) {
        roce_pack v[ROCE_WORLD];
#pragma unroll
        for (int s = 0; s < ROCE_WORLD; s++) v[s] = src_pack<false>(s, inp, recv, slot_bytes, seq, (long long)pack * ROCE_PACK_BYTES);
#pragma unroll
        for (int s = 0; s < ROCE_WORLD; s++)
            roce_st_global_pack((roce_byte *)out + ((long long)s * size_packs + pack) * ROCE_PACK_BYTES, v[s]);
    }
    stamp_end(st);
}

// L2 bulk prefetch beside the reduce, as the model's prefetch fires right before each all-reduce.
__global__ void k_prefetch(const char *buf, long long bytes, int chunk) {
    if (threadIdx.x != 0) return;
    for (long long off = (long long)blockIdx.x * chunk; off + chunk <= bytes; off += (long long)gridDim.x * chunk)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(buf + off), "r"(chunk) : "memory");
}

typedef void (*kfn)(const void *, const roce_byte *, long long, roce_u32, void *, int, unsigned long long *);

struct Variant { const char *name; int threads; kfn fn; };

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "%s SMs=%d maxThr/SM=%d L2=%d\n", prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, prop.l2CacheSize);
    const long long slot_bytes = 2 << 20;
    const size_t recv_bytes = (size_t)ROCE_WORLD * ROCE_SLOTS * slot_bytes;
    // A large pinned region, written by the host first so the slots are long out of the CPU caches.
    const size_t pinned_bytes = 512ull << 20;
    roce_byte *pinned;
    CHECK(cudaHostAlloc((void **)&pinned, pinned_bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < pinned_bytes; i += 4) *(unsigned *)(pinned + i) = 0x3f803f80u;  // bf16 1.0 pairs
    roce_byte *recv_dev, *inp, *out;
    CHECK(cudaMalloc(&recv_dev, recv_bytes));
    CHECK(cudaMemset(recv_dev, 0x3f, recv_bytes));
    CHECK(cudaMalloc(&inp, slot_bytes));
    CHECK(cudaMemset(inp, 0x3f, slot_bytes));
    CHECK(cudaMalloc(&out, ROCE_WORLD * slot_bytes));
    const long long pf_total = 1ll << 30;  // prefetch source: 1 GiB of device memory, a fresh 20 MiB window each time
    char *pf;
    CHECK(cudaMalloc(&pf, pf_total));
    CHECK(cudaMemset(pf, 1, pf_total));
    unsigned long long *st;
    CHECK(cudaMalloc(&st, 2 * sizeof(unsigned long long)));
    cudaStream_t s1, s2;
    CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    std::vector<Variant> variants = {
        {"pack", 512, k_pack<512>}, {"pack", 256, k_pack<256>}, {"pack", 128, k_pack<128>},
        {"m2", 512, k_multi<512, 2, false>}, {"m2", 256, k_multi<256, 2, false>}, {"m2", 128, k_multi<128, 2, false>},
        {"m4", 256, k_multi<256, 4, false>},
        {"cv1", 256, k_multi<256, 1, true>},
        {"gather", 512, k_gather<512>}, {"gfirst", 512, k_gather_first<512>},
    };
    if (argc > 2) {  // only the variants named in argv[2] (comma list of name/threads)
        std::vector<Variant> keep;
        for (auto &v : variants) { char key[32]; snprintf(key, sizeof key, ",%s/%d,", v.name, v.threads);
            char list[256]; snprintf(list, sizeof list, ",%s,", argv[2]); if (strstr(list, key)) keep.push_back(v); }
        variants = keep;
    }
    for (auto &v : variants) {
        cudaFuncAttributes a;
        CHECK(cudaFuncGetAttributes(&a, v.fn));
        fprintf(stderr, "%s/%d regs=%d local=%zu\n", v.name, v.threads, a.numRegs, a.localSizeBytes);
    }
    const int sizes_kib[] = {10, 30, 60, 160, 480, 960};
    const int grids[] = {1, 2, 4, 8, 16, 32, 64, 128};
    const char *mems[] = {"pinned", "device"};
    printf("W\tmem\tprefetch\tvariant\tthreads\tKiB\tgrid\tp50_us\tp10_us\tp90_us\n");
    unsigned long long init[2] = {~0ull, 0ull}, host[2];
    long long pf_off = 0;
    for (int m = 0; m < 2; m++) {
        const roce_byte *recv = m == 0 ? pinned : recv_dev;
        for (int pfon = 0; pfon < 2; pfon++) {
            if (m == 1 && pfon == 1) continue;
            for (auto &v : variants) {
                for (int kib : sizes_kib) {
                    int packs = kib * 1024 / 16;
                    for (int g : grids) {
                        if ((long long)g * v.threads > 4ll * packs) continue;  // a quarter pack a thread is enough
                        std::vector<double> t;
                        for (int it = 0; it < iters + 10; it++) {
                            CHECK(cudaMemcpyAsync(st, init, sizeof(init), cudaMemcpyHostToDevice, s1));
                            CHECK(cudaStreamSynchronize(s1));
                            if (pfon) {
                                k_prefetch<<<4, 32, 0, s2>>>(pf + pf_off, 20ll << 20, 64 << 10);
                                pf_off = (pf_off + (20ll << 20)) % (pf_total - (20ll << 20));
                            }
                            v.fn<<<g, v.threads, 0, s1>>>(inp, recv, slot_bytes, (roce_u32)(it + 1), out, packs, st);
                            CHECK(cudaGetLastError());
                            CHECK(cudaMemcpyAsync(host, st, sizeof(host), cudaMemcpyDeviceToHost, s1));
                            CHECK(cudaStreamSynchronize(s1));
                            CHECK(cudaStreamSynchronize(s2));
                            if (it >= 10) t.push_back((host[1] - host[0]) / 1000.0);
                        }
                        std::sort(t.begin(), t.end());
                        size_t n = t.size();
                        printf("%d\t%s\t%d\t%s\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\n", ROCE_WORLD, mems[m], pfon, v.name, v.threads,
                               kib, g, t[n / 2], t[n / 10], t[n * 9 / 10]);
                        fflush(stdout);
                    }
                }
            }
        }
    }
    return 0;
}
