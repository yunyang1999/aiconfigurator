# AIC AFD + Dynamo Mocker 复现 FastAFD workload

这份说明只复现一条链路：

`FastAFD workload / F-stage 实测时间 → AIC AFD 整域 decode service → Dynamo Mocker`

示例采用 Qwen3-235B-A22B-FP8、GB200 NVL72、`17A:1F`。AIC 和
Mocker 不需要 GPU；只有重新测 FastAFD 的 F-stage 时间需要 GPU。

## 1. 固定代码版本

```bash
mkdir -p ~/afd-repro && cd ~/afd-repro

git clone --branch afd-moe-timing \
  git@github.com:liz-badada/aiconfigurator.git
git clone --branch afd-moe-timing \
  git@github.com:liz-badada/dynamo.git
git clone https://github.com/hao-ai-lab/FastAFD.git
git -C FastAFD checkout 3c7161949310b6d59d6b4cf9bf997a4935c8113b
```

对应提交：

- AIC：`6e7442b2bb86d7553fbaf27c843da5713d2a69a7`
- Dynamo：`9d01867c8a7ff4e0f043b214e3728d2b9e835e90`
- FastAFD：`3c7161949310b6d59d6b4cf9bf997a4935c8113b`

Dynamo 分支没有额外补丁。Mocker 已有标准 profile 接口，不应再增加
AFD/MoE 专用逻辑。

## 2. 安装 AIC 和 Dynamo

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

必须让 Dynamo 的 Python 源码和 Rust binding 来自同一个 checkout；不要把新
源码挂到旧 Dynamo 镜像的 binding 上。

## 3. FastAFD case 和 F-stage 输入

FastAFD 的 NVL72 8K case 可这样启动；需要已经运行的 18 节点 Ray 集群：

```bash
cd ~/afd-repro/FastAFD

MODEL_PATH=/path/to/Qwen3-235B-A22B-FP8 \
AFD_TOTAL_NODES=18 \
RUN_VLLM_ALIGNMENT=0 \
NSYS=1 \
bash scripts/experiments/afd/qwen3_235b/\
run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
```

该 case 的口径是：

| 参数 | 8K | 16K |
|---|---:|---:|
| 每张 A GPU 驻留 batch | 96 | 48 |
| microbatch 数 | 2 | 2 |
| 每张 A GPU 每个 microbatch | 48 | 24 |
| 17 个 A 节点的 A GPU 数 | 68 | 68 |
| 全域驻留请求数 | 6528 | 3264 |

传给 AIC 的 `F_STAGE_MS` 必须是同一 case 下，完整 resident batch 穿过全部
94 层和两个 microbatch 的 F-stage wall time。它包括 F 侧 MoE 计算以及
A↔F stage 通信，不包括 A 侧 router。不要把 FastAFD 整个 decode-step 时间
直接当成 `F_STAGE_MS`。

下面的 `42.9378303527832 ms` 是现有 B200 缩比实测数据，用于验证软件链路；
它不是拓扑严格等价的 GB200 NVL72 实测。拿到 NVL72 同 case 实测后只需替换
这个值。

## 4. AIC 计算并生成 Mocker profile

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

# --a-batch-size 是每个 A worker 的 batch；一个 A worker 占 a_tp 张 GPU。
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

8K 示例应得到：

```text
resident_requests=6528
microbatch_period_ms=21.469000000
decode_batch_service_time_ms=42.937830353
```

Mocker 要使用 `decode_batch_service_time_ms`，不能使用只代表一个
microbatch pipeline period 的 `result.tpot`。

## 5. 运行 Mocker

一个 Mocker logical worker 代表完整的 A+F domain，因此这里是一个 worker、
全域 batch 6528，不是 batch 96。

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

核对稳态结果：

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

8K 示例应得到约：

```text
completed_requests=6528
mocker_mean_itl_ms=42.937830000
steady_tokens_s_gpu=2111.580
```

Mocker 表里的有限长度 `Output Token Throughput` 包含启动和结束边界；同
FastAFD/AIC 做稳态对比时，应使用上面由 `mean_itl_ms` 重建的
tokens/s/GPU。

16K 只需把 `CONTEXT=16384`、`PER_A_GPU_BATCH=48`，并换成同 case 的
`F_STAGE_MS`。每个 A/F ratio 都必须使用该 ratio 自己实测的 F-stage
时间，不能把一个 ratio 的时间复用于整个 sweep。
