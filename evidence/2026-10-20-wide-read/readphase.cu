// The one-shot data phase alone on one GB10: the generated header's own reduce and gather
// functions over peer slots in cudaHostAlloc memory (sparknet's region comes from
// torch.zeros(pin_memory=True)), the local input in device memory, on the grid the runtime
// would choose, with and without an L2 bulk prefetch of 20 MiB launched just before it on
// another stream (the model's prefetch fires right before each all-reduce).
//
//   nvcc -O3 -std=c++17 -arch=sm_121a -DHDR='"header.h"' [-DPEERS_ONLY_GATHER] readphase.cu
//   ./a.out MAX_BLOCKS PACKS_PER_THREAD
//
// PEERS_ONLY_GATHER selects the gather signature of headers whose staging writes the own
// shard (the data phase reads the peers only). Times are %globaltimer spans from the first
// block's start to the last block's end, median and p10/p90 of 200 launches.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include HDR

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int THREADS = 512;

__device__ __forceinline__ void stamp_begin(unsigned long long *st) {
    if (threadIdx.x == 0) atomicMin(&st[0], roce_globaltimer());
}
__device__ __forceinline__ void stamp_end(unsigned long long *st) {
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&st[1], roce_globaltimer());
}

__global__ void __launch_bounds__(THREADS, 1536 / THREADS) k_reduce(const void *inp, const roce_byte *recv, long long slot_bytes,
        roce_u32 seq, void *out, int size_packs, unsigned long long *st) {
    stamp_begin(st);
    for (int pack = blockIdx.x * THREADS + threadIdx.x; pack < size_packs; pack += gridDim.x * THREADS)
        roce_reduce_pack_bf16(inp, recv, slot_bytes, seq, out, (long long)pack * ROCE_PACK_BYTES);
    stamp_end(st);
}

__global__ void __launch_bounds__(THREADS, 1536 / THREADS) k_gather(const void *inp, const roce_byte *recv, long long slot_bytes,
        roce_u32 seq, void *out, int size_packs, unsigned long long *st) {
    stamp_begin(st);
    for (int pack = blockIdx.x * THREADS + threadIdx.x; pack < size_packs; pack += gridDim.x * THREADS) {
#ifdef PEERS_ONLY_GATHER
        roce_gather_pack(recv, slot_bytes, seq, pack, size_packs, out);
#else
        roce_gather_pack(inp, recv, slot_bytes, seq, pack, size_packs, out);
#endif
    }
    stamp_end(st);
}

__global__ void k_prefetch(const char *buf, long long bytes, int chunk) {
    if (threadIdx.x != 0) return;
    for (long long off = (long long)blockIdx.x * chunk; off + chunk <= bytes; off += (long long)gridDim.x * chunk)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(buf + off), "r"(chunk) : "memory");
}

typedef void (*kfn)(const void *, const roce_byte *, long long, roce_u32, void *, int, unsigned long long *);

// The runtime's grid: a power of two of blocks holding packs_per_thread packs a thread, capped.
static int grid_blocks(int packs, int packs_per_thread, int max_blocks) {
    int per_block = packs_per_thread * THREADS;
    int required = std::max(1, (packs + per_block - 1) / per_block), grid = 1;
    while (grid < required) grid <<= 1;
    return std::min(grid, max_blocks);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s MAX_BLOCKS PACKS_PER_THREAD\n", argv[0]); return 2; }
    const int max_blocks = atoi(argv[1]), packs_per_thread = atoi(argv[2]), iters = 200;
    const long long slot_bytes = 2 << 20;
    // A large pinned region written by the host first, so the slots are long out of the CPU caches.
    const size_t pinned_bytes = 512ull << 20;
    roce_byte *pinned, *inp, *out;
    CHECK(cudaHostAlloc((void **)&pinned, pinned_bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < pinned_bytes; i += 4) *(unsigned *)(pinned + i) = 0x3f803f80u;
    CHECK(cudaMalloc(&inp, slot_bytes));
    CHECK(cudaMemset(inp, 0x3f, slot_bytes));
    CHECK(cudaMalloc(&out, ROCE_WORLD * slot_bytes));
    const long long pf_total = 1ll << 30, pf_bytes = 20ll << 20;
    char *pf;
    CHECK(cudaMalloc(&pf, pf_total));
    CHECK(cudaMemset(pf, 1, pf_total));
    unsigned long long *st, init[2] = {~0ull, 0ull}, host[2];
    CHECK(cudaMalloc(&st, sizeof(init)));
    cudaStream_t s1, s2;
    CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    const struct { const char *name; kfn fn; } kernels[] = {{"reduce", k_reduce}, {"gather", k_gather}};
    const int sizes_kib[] = {10, 30, 60, 160, 480, 960};
    long long pf_off = 0;
    printf("world\tkernel\tprefetch\tKiB\tgrid\tp50_us\tp10_us\tp90_us\n");
    for (auto &k : kernels) {
        for (int prefetch = 0; prefetch < 2; prefetch++) {
            for (int kib : sizes_kib) {
                int packs = kib * 1024 / ROCE_PACK_BYTES, grid = grid_blocks(packs, packs_per_thread, max_blocks);
                std::vector<double> t;
                for (int it = 0; it < iters + 10; it++) {
                    CHECK(cudaMemcpyAsync(st, init, sizeof(init), cudaMemcpyHostToDevice, s1));
                    CHECK(cudaStreamSynchronize(s1));
                    if (prefetch) {
                        k_prefetch<<<4, 32, 0, s2>>>(pf + pf_off, pf_bytes, 64 << 10);
                        pf_off = (pf_off + pf_bytes) % (pf_total - pf_bytes);
                    }
                    k.fn<<<grid, THREADS, 0, s1>>>(inp, pinned, slot_bytes, (roce_u32)(it + 1), out, packs, st);
                    CHECK(cudaGetLastError());
                    CHECK(cudaMemcpyAsync(host, st, sizeof(host), cudaMemcpyDeviceToHost, s1));
                    CHECK(cudaStreamSynchronize(s1));
                    CHECK(cudaStreamSynchronize(s2));
                    if (it >= 10) t.push_back((host[1] - host[0]) / 1000.0);
                }
                std::sort(t.begin(), t.end());
                size_t n = t.size();
                printf("%d\t%s\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\n", ROCE_WORLD, k.name, prefetch, kib, grid, t[n / 2], t[n / 10],
                       t[n * 9 / 10]);
                fflush(stdout);
            }
        }
    }
    return 0;
}
