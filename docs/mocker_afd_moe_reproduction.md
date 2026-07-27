# Reproducing AFD With and Without MTP

This guide runs the same fixed-resident decode case in four modes:

| Mode | A/F split | MTP |
|---|---:|---:|
| `agg` | no | off |
| `afd` | yes | off |
| `agg_mtp` | no | on |
| `afd_mtp` | yes | on |

Two execution paths are supported:

1. **AIC direct** compares the compute service time of one resident decode
   round.
2. **Dynamo Mocker** consumes the same AIC service time and adds request
   arrivals, scheduling, batching, KV-cache pressure, queueing, and output-token
   completion.

Both paths are CPU-only. A GPU is needed only to obtain measured operation
times supplied through `afd_moe_time_ms`.

## 1. Code

```bash
git clone --branch afd-moe-timing \
  git@github.com:liz-badada/aiconfigurator.git
git clone --branch afd-moe-timing \
  git@github.com:liz-badada/dynamo.git

cd aiconfigurator
python3 -m uv sync --extra dev
git lfs pull

cd ../dynamo
uv venv .venv
source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --uv --release)
uv pip install -e lib/gpu_memory_service
uv pip install -e .
```

## 2. MTP Contract

For draft depth `nextn=N`, AIC models one target-verification round with
`q=N+1` query tokens per active request.

If `r_i` is the conditional acceptance probability of draft position `i`,

```text
E[accepted drafts] = r1 + r1*r2 + ... + r1*r2*...*rN
E[output tokens per round] = 1 + E[accepted drafts]
effective TPOT = raw verification-round period / E[output tokens per round]
```

AIC direct applies the last division. A Mocker profile must instead contain the
**raw verification-round wall time**; Mocker samples the accepted output-token
burst and therefore applies progress itself.

The same conditional rates must be used to generate the AIC profile and to run
Mocker. Do not combine MTP with `decode_speedup_ratio`.

## 3. Generate Four AIC Results and Mocker Profiles

The example below is a 72-GPU GB200 domain with `17A:1F`, attention TP1, and a
resident batch of 96 per A GPU. Adjust the aggregate topology to the deployment
being compared.

`MTP_ACCEPT_RATES` must come from a serving trace when distribution-sensitive
latency is required. The example value is an equal-conditional-rate surrogate
whose mean progress is 2.28125 tokens per round; it is not a measured
per-position distribution.

```bash
export CONTEXT=8192
export OUTPUT_TOKENS=256
export MTP_NEXTN=3
export MTP_ACCEPT_RATES=0.631245693194,0.631245693194,0.631245693194

cd aiconfigurator
.venv/bin/python - <<'PY'
import json
import math
import os
from pathlib import Path

import numpy as np

from aiconfigurator.cli.api import cli_estimate

MODEL = "Qwen/Qwen3-235B-A22B-FP8"
SYSTEM = "gb200"
BACKEND = "sglang"
BACKEND_VERSION = "0.5.10"
DATABASE_MODE = "HYBRID"

CONTEXT = int(os.environ["CONTEXT"])
OUTPUT_TOKENS = int(os.environ["OUTPUT_TOKENS"])
NEXTN = int(os.environ["MTP_NEXTN"])
RATES = tuple(float(x) for x in os.environ["MTP_ACCEPT_RATES"].split(","))

A_NODES = 17
F_NODES = 1
GPUS_PER_NODE = 4
A_TP = 1
PER_A_GPU_BATCH = 96
A_BATCH_SIZE = PER_A_GPU_BATCH * A_TP
NUM_MICROBATCHES = 2
GLOBAL_BATCH = A_NODES * GPUS_PER_NODE // A_TP * A_BATCH_SIZE

# Same 72-GPU aggregate baseline: 18 independent four-GPU replicas.
STATIC_REPLICAS = 18
STATIC_TP = 4
STATIC_ATTN_DP = 1
STATIC_MOE_TP = 1
STATIC_MOE_EP = 4
STATIC_BATCH_PER_REPLICA = math.ceil(GLOBAL_BATCH / STATIC_REPLICAS)


def expected_accepted(rates):
    survival = 1.0
    total = 0.0
    for rate in rates:
        survival *= rate
        total += survival
    return total


def mtp_kwargs(enabled):
    if not enabled:
        return {}
    if len(RATES) != NEXTN:
        raise ValueError("MTP_ACCEPT_RATES must contain exactly MTP_NEXTN values")
    return {
        "nextn": NEXTN,
        "nextn_accepted": expected_accepted(RATES),
    }


def common():
    return {
        "model_path": MODEL,
        "system_name": SYSTEM,
        "backend_name": BACKEND,
        "backend_version": BACKEND_VERSION,
        "database_mode": DATABASE_MODE,
        "isl": CONTEXT - 1,
        "osl": 2,
        "kvcache_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
    }


def run_agg(mtp):
    kwargs = common() | mtp_kwargs(mtp)
    kwargs.update(
        mode="static_gen",
        batch_size=STATIC_BATCH_PER_REPLICA,
        tp_size=STATIC_TP,
        attention_dp_size=STATIC_ATTN_DP,
        moe_tp_size=STATIC_MOE_TP,
        moe_ep_size=STATIC_MOE_EP,
    )
    return cli_estimate(**kwargs)


def run_afd(mtp):
    kwargs = common() | mtp_kwargs(mtp)
    kwargs.update(
        mode="afd",
        tp_size=STATIC_TP,
        n_a_nodes=A_NODES,
        n_f_nodes=F_NODES,
        a_tp_size=A_TP,
        a_batch_size=A_BATCH_SIZE,
        f_moe_ep_size=GPUS_PER_NODE,
        num_microbatches=NUM_MICROBATCHES,
        pipeline_model="conservative",
        afd_phase="decode",
        afd_combined_with_pd=False,
    )

    # A measured override must match the active verification width. A q=1
    # measurement must not be reused for q=NEXTN+1.
    q = NEXTN + 1 if mtp else 1
    measured = os.environ.get(f"F_STAGE_Q{q}_MS")
    if measured:
        kwargs["afd_moe_time_ms"] = float(measured)
    return cli_estimate(**kwargs)


def raw_service_ms(result, mtp):
    raw = result.raw.get("decode_batch_service_time_ms")
    if raw is not None:
        return float(raw)
    progress = 1.0 + (expected_accepted(RATES) if mtp else 0.0)
    return float(result.tpot) * progress


summary = {}
for topology, runner in (("agg", run_agg), ("afd", run_afd)):
    for mtp in (False, True):
        name = topology + ("_mtp" if mtp else "")
        result = runner(mtp)
        service_ms = raw_service_ms(result, mtp)
        summary[name] = {
            "effective_tpot_ms": float(result.tpot),
            "raw_decode_round_ms": service_ms,
            "tokens_s": float(result.raw.get("tokens/s", 0.0))
            * (STATIC_REPLICAS if topology == "agg" else 1),
            "tokens_s_gpu": float(result.raw.get("tokens/s/gpu", 0.0)),
        }

        # Fixed-case profile. Both axes bracket only this workload envelope.
        # Use a denser grid before replaying a broad batch/context sweep.
        profile_batch = (
            STATIC_BATCH_PER_REPLICA if topology == "agg" else GLOBAL_BATCH
        )
        profile = Path(f"/tmp/{name}.npz")
        np.savez(
            profile,
            prefill_isl=np.array([0.0, float(profile_batch * CONTEXT)]),
            prefill_ttft_ms=np.array([0.0, 1.0]),
            decode_active_kv_tokens=np.array(
                [0.0, float(profile_batch * (CONTEXT + OUTPUT_TOKENS))]
            ),
            decode_context_length=np.array(
                [float(CONTEXT), float(CONTEXT + OUTPUT_TOKENS)]
            ),
            decode_itl=np.full((2, 2), service_ms),
        )

print(json.dumps(summary, indent=2, sort_keys=True))
print(f"global_batch={GLOBAL_BATCH}")
PY
```

The fixed profile intentionally isolates decode. Its 1 ms prefill placeholder
must not be used to report TTFT. Replace the prefill table with measured or AIC
prefill points when TTFT is part of the experiment.

## 4. Replay Any of the Four Modes

```bash
export DYNAMO=$PWD/../dynamo
export GLOBAL_BATCH=$((17 * 4 * 96))
export BLOCK_SIZE=64

run_case () {
  mode=$1
  profile=/tmp/${mode}.npz
  if [[ "$mode" == agg* ]]; then
    workers=18
    worker_batch=$(((GLOBAL_BATCH + workers - 1) / workers))
  else
    workers=1
    worker_batch=$GLOBAL_BATCH
  fi
  worker_blocks=$(
    GLOBAL_BATCH="$worker_batch" python - <<'PY'
import os
b = int(os.environ["GLOBAL_BATCH"])
s = int(os.environ["CONTEXT"]) + int(os.environ["OUTPUT_TOKENS"])
block = int(os.environ["BLOCK_SIZE"])
print((b * s + block - 1) // block + 2 * b)
PY
  )
  mtp_json=
  if [[ "$mode" == *_mtp ]]; then
    mtp_json=$(printf \
      ',"aic_nextn":%d,"aic_nextn_accept_rates":"%s"' \
      "$MTP_NEXTN" "$MTP_ACCEPT_RATES")
  fi

  engine_args=$(printf \
    '{"engine_type":"vllm","num_gpu_blocks":%d,"block_size":%d,"max_num_seqs":%d,"max_num_batched_tokens":%d,"enable_prefix_caching":false,"enable_chunked_prefill":false%s,"aic_mtp_seed":42,"planner_profile_data":"%s"}' \
    "$worker_blocks" "$BLOCK_SIZE" "$worker_batch" \
    "$((worker_batch * CONTEXT))" "$mtp_json" "$profile")

  "$DYNAMO/.venv/bin/python" -m dynamo.replay \
    --input-tokens "$CONTEXT" \
    --output-tokens "$OUTPUT_TOKENS" \
    --request-count "$GLOBAL_BATCH" \
    --replay-concurrency "$GLOBAL_BATCH" \
    --num-workers "$workers" \
    --replay-mode offline \
    --router-mode round_robin \
    --extra-engine-args "$engine_args" \
    --report-json "/tmp/${mode}_mocker.json"
}

run_case agg
run_case afd
run_case agg_mtp
run_case afd_mtp
```

Use the AIC-direct results for fixed-resident compute comparison. Use the
Mocker reports for request-level throughput and latency under the configured
scheduler. The two comparisons must use the same global batch, hardware
budget, context, verification width, and acceptance-rate contract.
