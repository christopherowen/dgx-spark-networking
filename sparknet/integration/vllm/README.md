# vLLM integration

`SparknetRoceAllReduce` is a drop-in replacement for the `B12xRoceAllReduce`
class in Local Inference Lab's vLLM fork
(`vllm/distributed/device_communicators/b12x_roce_all_reduce.py`, pinned by
spark3-vllm-ds41f at `04c30fa98e79`). It keeps the constructor keywords, the
`should_custom_ar`/`custom_all_reduce`/`should_all_gather`/`all_gather`/
`capture`/`check_health`/`close` methods and the three limit properties that
the explicit-collective-policy patch (spark3 vLLM patch 0027) added.

## Switching the fork to sparknet

One vLLM patch, on top of the spark3 series:

1. In `cuda_communicator.py`, import `SparknetRoceAllReduce` from
   `sparknet.integration.vllm` where `B12xRoceAllReduce` is imported, and
   construct it with the same arguments plus
   `in_the_same_node=in_the_same_node_as(group, source_rank=0)`.
2. Keep the policy guards of patch 0027 (`_require_roce_policy`, no
   symmetric-memory or AITER paths while the RoCE policy is selected,
   `--disable-custom-all-reduce` rejected when RoCEnante is requested).
3. Drop the b12x preparation-unit registration (`register_b12x_unit_provider`,
   `get_b12x_preparation_units`): the adapter prepares its launchers in its
   constructor, before model load and graph capture.
4. `sp_prefill.py` keeps reading `all_reduce_capacity_bytes` for the
   sequence-parallel floor; nothing else changes.

The environment stays the same: `VLLM_ENABLE_ROCE_ALLREDUCE=1`,
`VLLM_ROCE_ALLREDUCE_MAX_SIZE`, `VLLM_ROCE_ALLGATHER_MAX_SIZE` and the
`B12X_ROCE_*` names the current profiles set are all read; the
`SPARKNET_ROCE_*` names take precedence when both are present
(`sparknet nccl env --profile tp4-ring` prints both).

Serving images must build the proxy and warm the CuTe kernels at image build
time (one import plus `prepare` in a build step), then call
`sparknet.rocenante.freeze_kernel_resolution()` after the engine warm-up so
no compilation can happen inside a step.

## Other engines

Any engine with a tensor-parallel process group can use the runtime directly:
construct `sparknet.rocenante.AllReduce.from_exchange_group` over a CPU (gloo)
group, `prepare` the dtypes it will reduce, decide dispatch with
`sparknet.policy.CollectivePolicy`, and call `check_health` after each step's
own device-to-host synchronization.
