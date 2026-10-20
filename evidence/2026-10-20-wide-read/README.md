# 2026-10-20: a wider data phase with one poller

The one-shot kernels' data phase reads every peer's slot from the pinned
host region (`torch.zeros(pin_memory=True)`, i.e. `cudaHostAlloc`) with
system-scope loads. Until this change the launch grid held about two 16-byte
packs a thread on at most 8 blocks of 512 threads, every block polled the
peers' flags as soon as it had staged, and the all-gather issued a load and a
store per source. The change: one pack a thread up to 64 blocks, only block 0
polls (after the doorbell), the gather's loads first. The first half was
measured on one GPU (below), the whole on the triangle (`fabric/`).

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

## On the triangle (`fabric/`)

dgx1-3, serving stopped in spark-ds41f lab windows, r6x image with each
arm's sparknet checkout mounted over the image's package (origin/main
f7fcc1e is code-identical to the image's). Every receipt is one node's final
probe JSON (`passed`, timings: five samples of 16 graph-replayed calls x 16);
`tables.py DIR` prints the slowest rank's median per run, median over runs.
RDMA error counters (retransmissions, sequence errors, CNPs) stayed zero in
every arm.

1. `probe-adapter/` (spark-ds41f `scripts/probe_collectives.py`, the vLLM
   adapter's path; main and wide-read a68d0ad, three runs each): the wide grid
   with every block polling was much slower from 60 KiB up, BF16 all-reduce
   18.6 to 20.7 us at 60 KiB, 31.9 to 45.6 at 160 KiB, 62.5 to 98.9 at
   480 KiB. Fail-stop passed; the GPU suite stopped at its first test,
   `test_payload_is_striped_across_every_hca`, which expects every HCA to
   carry every peer: on the triangle each HCA serves one peer (the proxy's
   `writes_completed_per_hca` equals `ops_posted` on all four), a test bug the
   triangle had not run into before, unrelated to the kernels. Later suites
   deselect it (`gpu-tests.txt`).
2. `diag/` (`sparknet probe collectives`, two runs each): the wide-read grid
   capped at 8, 16, 32 and 64 blocks, and `poll8`, an experiment in which only
   the first 8 of 64 blocks poll. The cost grows with the polling blocks from
   16 on, and `poll8` is level with 0.4.0. The traced runs
   (`trace-summary.json`, phases on each rank's GPU clock) put it in staging:
   blocks that had staged polled while the others still stored to the pinned
   region, and the doorbell came late (480 KiB: staging 5.4 to 23 to 29 us).
3. `diag2/` (three runs each): pollers gated on the doorbell, with 8 polling
   blocks (`v2p8`) or one (`v2p1`, the committed design, measured as
   experiment commit 2b841df). One poller is fastest: eight gated pollers
   still waited longer at 60 KiB (13.0 against 11.7 us), so polling also
   competes with the NIC's writes. The GPU suite (71 passed, 1 skipped) and
   fail-stop passed for both.

| BF16, per rank | All-reduce, main | All-reduce, v2p1 | All-gather, main | All-gather, v2p1 |
| ---: | ---: | ---: | ---: | ---: |
| 10 KiB | 10.8 us | 10.7 us | 12.8 us | 10.5 us |
| 30 KiB | 15.3 us | 14.3 us | 15.6 us | 13.7 us |
| 60 KiB | 19.6 us | 17.6 us | 19.6 us | 17.5 us |
| 160 KiB | 32.5 us | 28.7 us | 35.3 us | 29.5 us |
| 480 KiB | 62.2 us | 59.0 us | 69.2 us | 65.0 us |
| 960 KiB | 108.3 us | 102.6 us | 118.7 us | 112.7 us |

4. `qualify/` (the committed design, aa01019; spark-ds41f's adapter-path probe,
   three runs each against main): GPU suite 71 passed, 1 skipped and
   fail-stop passed on every rank (`gpu-tests.txt`); BF16 all-reduce 11.1 to
   10.8 us at 10 KiB, 14.0 to 13.8 at 30, 18.8 to 17.7 at 60, 32.5 to 28.8 at
   160, 62.2 to 59.1 at 480, 107.7 to 104.2 at 960; all-gather 12.4 to 10.4,
   14.8 to 13.5, 19.3 to 18.0, 33.8 to 30.5, 69.5 to 63.9, 119.4 to 113.3.

Traced phases, all-reduce p50 on rank 0 (staging / wait / data phase, us):
main 2.3 / 11.8 / 2.5 at 60 KiB, 5.4 / 46.8 / 7.1 at 480 KiB; v2p1 1.8 /
11.7 / 2.3 and 4.0 / 45.6 / 5.9. These probes run without the model's L2
prefetch, where the data phase gained most on one GPU.

## Serving (`serving/`)

spark-ds41f lab run `wide-read-serving` on the triangle, 2026-10-20 22:16 to
23:27 UTC (owner-approved after the qualification): servebench suites
`screen-tp3` and `single-stream` on r6x TP3 at 1M context
(`r6x-tp3-1m-dev-1017` with the dev-mode collective RPC for the engine grid),
the sparknet checkout mounted over the image's package. Arms in order:
`wide` (aa01019), `main` (f7fcc1e), `wide-again`. servebench counted no
foreign requests in any run, and nothing outside dgx1 connected to the
serving port during the window.

- Outputs: greedy responses identical in 128 of 128 prompts, one output hash
  for all three arms, teacher-forced distributions identical (KL 0).
- Engine grid (decode step time at fixed request and row counts, the suites'
  text-independent step cost): geometric mean over 15 shapes -1.38 percent
  (`wide`) and -1.82 percent (`wide-again`) against `main`, each +-0.16
  percent; the two `wide` arms differ by -0.44 percent, so with `main`
  between them the change is about -1.6 percent. Per shape -1.5 to -2.6
  percent from four requests up, -0.1 to -1.7 percent at one request.
- Closed loop at 8 streams: output +0.34 and +0.40 percent (+-1.26), one
  stream +0.16 and +0.46 percent (+-7.2), prefill at 32K and 256K level: all
  within their intervals.

Verdict against the 1 percent bar: the step cost clears it (about 1.6
percent, outside every interval); the throughput workloads cannot resolve 1
percent and read level, in the same direction.
