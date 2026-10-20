# 2026-10-20: a wider data phase

The one-shot kernels' data phase reads every peer's slot from the pinned
host region (`torch.zeros(pin_memory=True)`, i.e. `cudaHostAlloc`) with
system-scope loads. Until this change the launch grid held about two 16-byte
packs a thread on at most 8 blocks of 512 threads, and the all-gather issued
a load and a store per source.

## The data phase alone (`readphase/`)

`readphase.cu` runs the generated header's own `roce_reduce_pack_bf16` and
`roce_gather_pack` over peer slots in `cudaHostAlloc` memory on one GB10
(dgx4, idle, r6x image `vllm-ds41f-kkref:19f2c20ed4d6-r6x`), on the grid each
version's runtime chooses, with and without an L2 bulk prefetch of 20 MiB
launched just before it on another stream (the model's prefetch fires right
before each all-reduce). `run.sh OLD NEW OUT` builds it against two
checkouts' headers: `old` is origin/main (0.4.0, at most 8 blocks, two packs
a thread), `new` this branch (one pack a thread, at most 64 blocks; the
gather reads the peers only, staging has written the own shard). Median of
200 launches, first block's start to last block's end; the grid in brackets.

Three ranks, beside the prefetch:

| Per rank | Reduce, old | Reduce, new | Gather, old | Gather, new |
| ---: | ---: | ---: | ---: | ---: |
| 10 KiB | 9.7 us (1) | 6.3 us (2) | 15.6 us (1) | 5.9 us (2) |
| 30 KiB | 10.8 (2) | 6.4 (4) | 17.3 (2) | 6.1 (4) |
| 60 KiB | 10.7 (4) | 6.7 (8) | 17.9 (4) | 6.8 (8) |
| 160 KiB | 15.6 (8) | 8.3 (32) | 26.1 (8) | 8.0 (32) |
| 480 KiB | 33.9 (8) | 10.2 (64) | 39.3 (8) | 10.8 (64) |
| 960 KiB | 40.5 (8) | 16.6 (64) | 47.6 (8) | 16.6 (64) |

Three ranks, no prefetch: reduce 1.3 to 0.8 us at 10 KiB, 1.7 to 1.1 at
60 KiB, 6.4 to 4.5 at 480 KiB; gather 2.4 to 0.8, 2.6 to 1.1, 9.7 to 4.5.
Four ranks, beside the prefetch: reduce 11.6 to 7.5 us at 60 KiB and 35.6
to 12.3 at 480 KiB; gather 24.7 to 7.5 and 43.2 to 13.6 (`new-w4.tsv`,
`old-w4.tsv`).

What it shows: beside the prefetch a single round trip to the pinned region
costs about 6 us (the 10 KiB floor, whatever the grid) against under 1 us
without it, and the time is the number of serial round trips a thread makes.
The old grid gave each thread two packs, and the old gather a round trip per
peer (a store orders the loads after it). A first sweep over block size (128,
256, 512 threads) and packs in flight a thread (one, two or four packs'
loads issued before use) found the thread count is what matters: 256-thread
blocks at twice the grid matched 512, and issuing two or four packs' loads
together was no faster and spilled at four ranks within 40 registers
(`sweep/`: `sweep.cu` over grids of 1 to 128 blocks, its reduce variants in
`w3.tsv`/`w4.tsv` and the two gather forms in `gather-w*.tsv`, which also
hold device-memory rows for comparison).

## Registers

Every launcher (2, 3, 4, 8 and 16 ranks; float32, float16 and bfloat16;
ring4; traced) compiles to 40 registers on sm_121a, as before. The
all-reduce kernels are unchanged (the grid is a launch argument; the
eight-rank ones keep the single 8-byte spill they had in 0.4.0). The
all-gather kernels use no local memory: the first version of the loads-first
gather spilled scalars at three and four ranks (8 to 16 bytes of stack), which
writing the own shard while staging removed (the data phase then holds only
the peers' packs). Checked by extracting each cubin from TileLang's cache and
counting `LDL`/`STL` in its SASS.
