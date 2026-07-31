# Run protocol

Run the phases in order. Phase 2 is a correctness gate: if outputs do not align
with vLLM, the timings from later phases are not trustworthy.

---

## Phase 0 — confirm the hardware

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
uname -m                       # expect aarch64
nvcc --version | tail -2       # expect CUDA 13.x
```

Expect `compute_cap = 10.x` (Blackwell). **If it is 9.x, stop** — the MegaMoE
path is SM100-only and will not build.

For multi-node, confirm the nodes share a fabric and that Ray can see them all.

## Phase 1 — environment

```bash
cd fastafd
conda env create -f environment.cuda130.yml     # creates minisgl-cuda130
./scripts/enter_clean.sh minisgl-cuda130
pip install -e .

# vLLM, for the alignment check and the colocated baseline
python -m pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu130

# runtime overrides the AFD/DeepEP path needs
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "nvidia-nccl-cu13==2.30.4" "setuptools==80.10.2" "fsspec==2026.2.0"
```

Validated versions: `torch 2.10.0+cu130`, `triton 3.6.0`,
`nvidia-nccl-cu13 2.30.4`, `vllm 0.19.0+cu130`.

DeepEP and DeepGEMM are JIT-built on first use. The first run is slow; that is
expected, and that compile time must not be counted in any measurement.

## Phase 2 — correctness gate (2 nodes, 8 GPUs)

```bash
./scripts/validate/qwen3_30b_a3b_alignment.sh          # mini-sgl vs vLLM, single node, 4 GPUs
./scripts/validate/qwen3_30b_a3b_fastafd_alignment.sh  # FastAFD serve path vs vLLM, 8 GPUs
```

The second needs a running Ray cluster with >= 8 GPUs (attention TP4 + MLP TP4).

**Do not continue if this fails.** Report the failure instead.

## Phase 3 — the headline point (4 nodes, 16 GPUs)

This is the configuration whose latency the blog publishes as 32.826 ms.

```bash
export MODEL_PATH=/shared/scratch/Qwen3-235B-A22B-FP8   # a local snapshot, not an HF id
MODEL_PATH=$MODEL_PATH AFD_TOTAL_NODES=4 RUN_VLLM_ALIGNMENT=0 NSYS=0 \
  bash scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
```

Record from `$RUN_DIR` (printed by the script, under `reports/`):

* steady-state **decode-step latency** in ms
* **per-GPU decode throughput**, using the denominator defined in README.md
  (all 16 GPUs, including the 4 FFN GPUs)

Enter both into `aic_predictions.csv` on the row
`model=Qwen3-235B, isl=8192, total_nodes=4`.

## Phase 4 — sweep M:N

Same command, varying `AFD_TOTAL_NODES`. Node `N` means `(N-1)` attention nodes
and 1 MLP node.

```bash
for N in 4 6 8 10 12 14 18; do
  MODEL_PATH=$MODEL_PATH AFD_TOTAL_NODES=$N RUN_VLLM_ALIGNMENT=0 NSYS=0 \
    bash scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
done
```

Run as many `N` as your allocation allows; more points are better, but **4, 8
and 12 are the priority** — they bracket the claimed 7–11:1 optimum.

For 16K, use the `..._16k_b48_...` preset script instead (48 requests per
attention GPU).

If `MiniMax-M2.5` weights are available, repeat with the presets under
`scripts/experiments/afd/minimax_m25/`. Its claimed optimum is 17:1, which needs
18 nodes.

## Phase 5 — colocated baseline

The baseline is vLLM on **one 4-GPU node**, DP4 + EP4. The preset already
encodes the exact arguments; enable it with:

```bash
MODEL_PATH=$MODEL_PATH AFD_TOTAL_NODES=4 RUN_VLLM_ALIGNMENT=1 NSYS=0 \
  bash scripts/experiments/afd/qwen3_235b/run_afd_qwen3_235b_a22b_fp8_8k_b96_dynamicnode_mb2_nsys_alignment.sh
```

Its per-GPU throughput is unchanged at larger node counts (colocated DP nodes
serve independently), so **one measurement per (model, context length)** is
enough. Enter it on the `kind=colocated-baseline` row.

## Phase 6 — optional, only if time allows

Re-run Phase 3 with cross-step overlap disabled. The blog reports the step
growing 32.826 -> 42.612 ms; reproducing that 23% delta is a good check that the
harness is measuring the same thing the blog measured.

---

## Notes

* Long-context prompt caches ship in `fastafd/prompts/`. Do not regenerate them
  unless a run demands a size that is not there.
* `NSYS=1` plus `MINISGL_RAY_NSYS_BIN=/path/to/nsys` captures traces. Useful if
  a number looks wrong, but it perturbs timing — never measure with it on.
* The MoE backend defaults to `deepep`; these presets override it to
  `megamoe_m2n`. Leave that alone. If you must disable MegaMoE for a build
  reason, set `MINISGL_AFD_MOE_BACKEND=deepep`, **say so in the report**, and
  mark those rows — the two backends are not interchangeable for timing.
