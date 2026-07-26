# Reproducing the FastAFD Workload with AIC AFD and Dynamo Mocker

This guide covers one path only:

`FastAFD workload and measured F-stage time → AIC AFD domain decode service → Dynamo Mocker`

The example uses Qwen3-235B-A22B-FP8 on GB200 NVL72 with a `17A:1F`
split. AIC and Mocker do not require GPUs. GPUs are required only when
remeasuring the FastAFD F-stage time.

## 1. Pin the Code Versions

```bash
mkdir -p ~/afd-repro && cd ~/afd-repro

git clone --branch afd-moe-timing \
  git@github.com:liz-badada/aiconfigurator.git
git clone --branch afd-moe-timing \
  git@github.com:liz-badada/dynamo.git
git clone https://github.com/hao-ai-lab/FastAFD.git
git -C FastAFD checkout 3c7161949310b6d59d6b4cf9bf997a4935c8113b
```

Feature baseline commits:

- AIC: `d862321fb56cb0d15b8ec775eca9da751e54d06c`
- Dynamo: `818501a6d21bc2669ae8ef275e606db82a918441`
- FastAFD: `3c7161949310b6d59d6b4cf9bf997a4935c8113b`

The Dynamo branch has no AFD-specific patch. Mocker already provides the
standard profile interface required by this workflow, so adding a second
AFD/MoE-specific timing interface would duplicate functionality.

## 2. Install AIC and Dynamo

```bash
cd ~/afd-repro/aiconfigurator
python3 -m uv sync --extra dev
git lfs pull

cd ~/afd-repro/dynamo
uv venv .venv
source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --uv --release)
uv pip install -e lib/gpu_memory_service
uv pip install -e .
```

The Dynamo Python source and Rust binding must come from the same checkout.
Do not mount current Python source over an older Dynamo image or binding.

## 3. Define the FastAFD Case and F-stage Input

The FastAFD NVL72 8K case can be launched as follows. It requires an active
18-node Ray cluster:

```bash
cd ~/afd-repro/FastAFD

MODEL_PATH=/path/to/Qwen3-235B-A22B-FP8 \
AFD_TOTAL_NODES=18 \
RUN_VLLM_ALIGNMENT=0 \
NSYS=1 \
bash scripts/experiments/afd/qwen3_235b/\
run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
```

The workload contract is:

| Parameter | 8K | 16K |
|---|---:|---:|
| Resident batch per A GPU | 96 | 48 |
| Number of microbatches | 2 | 2 |
| Batch per microbatch per A GPU | 48 | 24 |
| A GPUs across 17 A nodes | 68 | 68 |
| Domain-wide resident requests | 6528 | 3264 |

`F_STAGE_MS` must be measured for the same case. It is the F-stage wall time
for the full resident batch across all 94 layers and both microbatches. It
includes F-side MoE compute and A-to-F/F-to-A stage communication, but excludes
the A-side router. Do not pass the complete FastAFD decode-step time as
`F_STAGE_MS`.

The `42.9378303527832 ms` value below is an existing reduced-topology B200
measurement used to validate the software path. It is not a topology-exact
GB200 NVL72 measurement. Replace it with the matching NVL72 measurement when
one is available.

## 4. Run AIC and Generate the Mocker Profile

```bash
export AIC=~/afd-repro/aiconfigurator
export CONTEXT=8192
export OUTPUT_TOKENS=16
export A_NODES=17
export F_NODES=1
export A_TP=1
export PER_A_GPU_BATCH=96
export F_STAGE_MS=42.9378303527832
export PROFILE=/tmp/afd_8k_17a1f.npz

cd "$AIC"
"$AIC/.venv/bin/python" - <<'PY'
import logging
import os

import numpy as np

from aiconfigurator.cli.api import cli_estimate

logging.disable(logging.CRITICAL)

context = int(os.environ["CONTEXT"])
output_tokens = int(os.environ["OUTPUT_TOKENS"])
a_nodes = int(os.environ["A_NODES"])
f_nodes = int(os.environ["F_NODES"])
a_tp = int(os.environ["A_TP"])
per_a_gpu_batch = int(os.environ["PER_A_GPU_BATCH"])
f_stage_ms = float(os.environ["F_STAGE_MS"])
profile = os.environ["PROFILE"]

# --a-batch-size is per A worker, and one A worker spans a_tp GPUs.
a_batch_size = per_a_gpu_batch * a_tp
a_workers = a_nodes * 4 // a_tp
resident = a_workers * a_batch_size

result = cli_estimate(
    model_path="Qwen/Qwen3-235B-A22B-FP8",
    system_name="gb200",
    mode="afd",
    backend_name="sglang",
    backend_version="0.5.10",
    database_mode="HYBRID",
    isl=context - 1,
    osl=2,
    tp_size=4,
    n_a_nodes=a_nodes,
    n_f_nodes=f_nodes,
    a_tp_size=a_tp,
    a_batch_size=a_batch_size,
    f_moe_ep_size=4,
    num_microbatches=2,
    pipeline_model="conservative",
    afd_phase="decode",
    afd_combined_with_pd=False,
    kvcache_quant_mode="bfloat16",
    comm_quant_mode="half",
    afd_moe_time_ms=f_stage_ms,
)

service_ms = float(result.raw["decode_batch_service_time_ms"])
np.savez(
    profile,
    prefill_isl=np.array([0.0, float(resident * context)]),
    prefill_ttft_ms=np.array([0.0, 1.0]),
    decode_active_kv_tokens=np.array(
        [0.0, float(resident * (context + output_tokens))]
    ),
    decode_context_length=np.array(
        [float(context), float(context + output_tokens)]
    ),
    decode_itl=np.full((2, 2), service_ms),
)

print(f"resident_requests={resident}")
print(f"microbatch_period_ms={result.tpot:.9f}")
print(f"decode_batch_service_time_ms={service_ms:.9f}")
print(f"profile={profile}")
PY
```

The 8K example should produce:

```text
resident_requests=6528
microbatch_period_ms=21.469000000
decode_batch_service_time_ms=42.937830353
```

Mocker must use `decode_batch_service_time_ms`. Do not export `result.tpot`,
which represents one microbatch pipeline period.

## 5. Run Mocker

One logical Mocker worker represents the complete A+F domain. Therefore this
case uses one worker with a domain-wide batch of 6528, not a batch of 96.

```bash
export DYNAMO=~/afd-repro/dynamo
export B_TOTAL=$((A_NODES * 4 * PER_A_GPU_BATCH))
export TOTAL_GPUS=$(((A_NODES + F_NODES) * 4))
export PREFILL_TOKENS=$((B_TOTAL * CONTEXT))
export REQUIRED_TOKENS=$((B_TOTAL * (CONTEXT + OUTPUT_TOKENS)))
export NUM_BLOCKS=$(((REQUIRED_TOKENS + 63) / 64 + 2 * B_TOTAL))
export REPORT=/tmp/afd_8k_17a1f_mocker.json

ENGINE_ARGS=$(printf \
  '{"engine_type":"vllm","num_gpu_blocks":%d,"block_size":64,"max_num_seqs":%d,"max_num_batched_tokens":%d,"enable_prefix_caching":false,"enable_chunked_prefill":false,"planner_profile_data":"%s"}' \
  "$NUM_BLOCKS" "$B_TOTAL" "$PREFILL_TOKENS" "$PROFILE")

cd "$DYNAMO"
"$DYNAMO/.venv/bin/python" -m dynamo.replay \
  --input-tokens "$CONTEXT" \
  --output-tokens "$OUTPUT_TOKENS" \
  --request-count "$B_TOTAL" \
  --replay-concurrency "$B_TOTAL" \
  --num-workers 1 \
  --replay-mode offline \
  --router-mode round_robin \
  --extra-engine-args "$ENGINE_ARGS" \
  --report-json "$REPORT"
```

Check the steady-state result:

```bash
python - <<'PY'
import json
import os

report = json.load(open(os.environ["REPORT"], encoding="utf-8"))
batch = int(os.environ["B_TOTAL"])
gpus = int(os.environ["TOTAL_GPUS"])
itl_ms = float(report["mean_itl_ms"])

print(f"completed_requests={report['completed_requests']}")
print(f"mocker_mean_itl_ms={itl_ms:.9f}")
print(f"steady_tokens_s_gpu={batch * 1000 / (itl_ms * gpus):.3f}")
PY
```

The 8K example should produce approximately:

```text
completed_requests=6528
mocker_mean_itl_ms=42.937830000
steady_tokens_s_gpu=2111.580
```

Mocker's finite-length `Output Token Throughput` includes startup and shutdown
boundaries. For a steady-state comparison with FastAFD or AIC, use the
tokens/s/GPU value reconstructed from `mean_itl_ms` above.

For the 16K case, set `CONTEXT=16384` and `PER_A_GPU_BATCH=48`, then provide
the matching `F_STAGE_MS`. Every A/F ratio must use its own measured F-stage
time. Do not reuse one ratio's F-stage time across an entire sweep.
