# FastAFD on GB200 — measurement package

You are running **FastAFD** (`hao-ai-lab/FastAFD`) on GB200 NVL72 and reporting a
handful of numbers back. The point is not to tune FastAFD. The point is to
**measure it** so we can check a performance model (NVIDIA AIConfigurator) that
currently predicts AFD behaviour without any GB200 AFD measurements behind it.

Everything you need is in this package. The FastAFD source is bundled under
`fastafd/` (working tree, no `.git`), so you do not need GitHub access.

---

## The one number that matters most

FastAFD's blog publishes exactly one absolute latency:

> the Qwen3-235B 4-node 8K step grows from **32.826 ms** to 42.612 ms
> [with cross-step overlap disabled]

So **32.826 ms** is the decode-step latency at `AFD_TOTAL_NODES=4`, 8K context,
96 requests per attention GPU, `mb=2`.

AIC predicts **30.412 ms** for that same point.

**Measure it.** If you do nothing else, run that one configuration and report
its steady-state decode-step latency.

---

## Hard requirements — do not substitute

| Requirement | Why |
|---|---|
| **GB200 / Blackwell (sm100)** | The AFD path is `sm100_mega_moe_m2n.hpp` and `sm100_fp8_fp4_mega_moe.hpp`. On Hopper (sm90) it will not build. Do not fall back to H100/H200/H20. |
| **aarch64 + CUDA 13.0** | The only platform FastAFD declares validated; the install path ships an `aarch64` vLLM wheel. GB200 is aarch64 Grace, so this is satisfied. |
| **>= 2 nodes** for the alignment run, **>= 4 nodes** for the 235B preset | 4 GPUs/node. The preset puts the **last** Ray node on MLP and the rest on attention. |
| **NVLink/NVSwitch or IB between nodes** | DeepEP asserts `ncclCommQueryProperties(...).railedGinType != NCCL_GIN_TYPE_NONE`. Allocate nodes on a shared fabric. |
| **Shared scratch holding the weights** | Multi-node runs need a local `MODEL_PATH` snapshot; per-rank HuggingFace cache resolution does not work. |

If any of these cannot be met, **stop and say so** rather than substituting.
A measurement on the wrong hardware is worse than no measurement.

---

## Contents

```
README.md                      this file
RUN_PROTOCOL.md                exact commands, in order
aic_predictions.csv            AIC's predictions; fill in the measured_* columns
computelab_gb200_fastafd.sh    helper for NVIDIA ComputeLab (cdb/crun/slurm)
fastafd/                       FastAFD source tree
aic_reference/                 how the predictions were produced, for context
```

## Metric definition — use FastAFD's own

From the blog, so both sides count the same way:

> per-GPU decode throughput: tokens generated per step, divided by the step
> latency and by the **total GPU count of the run, including, for FastAFD, the
> FFN GPUs that host no requests**.

So for `AFD_TOTAL_NODES=N`: attention GPUs = `(N-1)*4`, FFN GPUs = `4`, and the
denominator is `N*4`. Tokens per step = `(N-1)*4 * PER_ATTN_GPU_BSZ`.

Measure **steady-state decode** only — after prefill, over a window of
synchronized decode steps with a fixed batch and CUDA-graph replay. Exclude
prefill and any warm-up steps.

## What to report back

1. `aic_predictions.csv` with `measured_step_ms` and `measured_tok_s_gpu`
   filled in for every row you ran. Leave the rest blank.
2. The `reports/` directory each run produces (`$RUN_DIR`).
3. The **best observed M:N** per workload, and the step latency at that point.
   FastAFD's blog claims 7–11:1 for Qwen3-235B and 17:1 for MiniMax-M2.5.
4. Anything that failed to build or run, verbatim. Negative results are useful;
   guesses are not.

## What we expect to learn

AIC and FastAFD agree that Qwen3 becomes FFN-limited as M:N grows, but they
disagree about **where**. AIC's predicted step latency at 8K turns over after
`N=6` (5:1); FastAFD reports its best in 7–11:1. At 16K AIC predicts nearly flat
latency out to `N=12` (11:1), which matches. The measurement settles it.
