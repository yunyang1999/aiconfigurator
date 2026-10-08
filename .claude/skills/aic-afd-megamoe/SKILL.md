---
name: aic-afd-megamoe
description: How the AIC (aiconfigurator) AFD vs non-AFD (colocated) decode sweep on GB200 NVL72 is built and run -- measured FastAFD MegaMoE stage times injected into AIC, a per-layer routing-skew coefficient on both arms, the configuration grid, the scripts and their order, and how the workbooks pick and compare deployments. Use when running or extending that sweep, or explaining its numbers.
---

# AIC AFD vs non-AFD sweep on GB200 NVL72 (measured MegaMoE)

Decode-phase comparison of AFD (attention-FFN disaggregation) against
colocated serving, in tokens/s per GPU on a 72-GPU NVL72 domain. AIC models
attention, KV, memory and the pipeline; **the MoE stage is a GB200
measurement**, looked up per configuration and handed to AIC.

- **Code**: branch `afd/gb200-megamoe-measured` of
  `https://github.com/yunyang1999/aiconfigurator`, directory
  `tools/afd_megamoe_gb200/` (imports `tools/afd_megamoe_b200/`).
- **Data**: the bundle `afd_megamoe_data.zip`; set `AFD_MEGAMOE_ROOT` to its
  unpacked `afd_megamoe_data/` (contains `results/`, routing JSONs,
  `reference/gb200_perlayer_routing.csv`).
- **Where**: GB200 (aarch64) compute nodes via Slurm, with `aiconfigurator`
  and `aiconfigurator_core` installed (aic-core carries a Rust extension built
  by maturin at install time).

## The grid

| Dimension | Values |
|---|---|
| Models | Qwen3-235B-A22B, MiniMax-M2.5, MiniMax-M3, DeepSeek-V4-Flash, DeepSeek-V4-Pro (all FP4 MoE) |
| ISL | 4K, 8K, 16K, 32K, 64K, 128K |
| OSL | 256 / 1024 / 4096 at MTP 0; 1024 at MTP 1, 2 |
| MTP (nextN) | 0, 1, 2 -- accepted tokens 0.85 and 1.53 |
| AFD splits (A:F, one 72-GPU unit) | 68A4F, 64A8F, 56A16F, 40A32F; DeepSeek-V4-Pro 64A8F, 60A12F, 56A16F, 48A24F |
| AFD attention TP | 1, 2, 4 -- kept only where `ag` = A/TP was measured |
| AFD microbatches | 1, 2, 4 |
| AFD batch | sequences per A GPU, on the measured rungs (8 .. 1024) |
| Colocated | EP 1/2/4/8 for Qwen3 and MiniMax-M2.5; 2/4/8 for MiniMax-M3 and DeepSeek-V4-Flash (EP=1 crashes the MegaMoE kernel there); 4/8/24 for DeepSeek-V4-Pro -- x attention TP 1/2/4; tokens per rank = batch / TP |

About 49.6k configurations. Each row ends `ok`, `oom` (does not fit HBM) or
`illegal` (outside the measured envelope -- returned, never extrapolated).

## How each arm is computed

**AFD arm** -- `cli_estimate(mode="afd")` on system `gb200_g{1,2,4,8}` (1/2/4/8
GPUs per AIC node, since AFD pools are sized in nodes), sglang 0.5.12,
`pipeline_model="conservative"`, with
`afd_moe_time_ms` = measured stage x skew coefficient. The measured stage comes
from `results/megamoe-m2n-nvl72`, keyed by (model, nextN, microbatches, experts
per GPU = experts/F, sequences per A GPU, `ag`). Given it, AIC replaces the
whole F side and zeroes every A<->F communication term, because the measured
stage already spans quant, dispatch, wait, combine and expert compute. Then:

```
t_f_layer = afd_moe_time_ms / m / L          # m microbatches, L layers
t_cycle   = max(t_a_layer, t_f_layer)
tpot      = pipeline_fill + (m * L - 1) * t_cycle
```

**Non-AFD (colocated) arm** -- AIC `static_gen` (decode worker) with the MoE
op substituted by the measured stage from `results/megamoe-colocated-nvl72`,
keyed by (model, nextN, EP, tokens per rank), times its own skew coefficient.
HBM capacity is enforced against GB200's usable 185 GiB.

The two measured tables are different kernel paths and must not stand in for
each other: colocated is an all-to-all inside one EP group whose ranks also run
attention (indexed by tokens per rank, 8..1024); AFD is an M2N transfer between
two disjoint pools (indexed by sequences per A GPU and `ag`; 10..104448 tokens
per F rank). At matched shard and tokens they differ by a median 4.2%, up to 43%.

## Routing-skew coefficient (both arms)

```
coefficient = 1 + f * (I - 1)
```

- `I`: per layer, busiest-rank load / mean-rank load (experts in contiguous
  rank blocks), **averaged over layers**. Summing layers first cancels the skew,
  because the busiest rank changes almost every layer.
- `f`: token-proportional share of the stage at this batch, from a
  least-squares fit `t = fixed + slope * batch` over the measured balanced
  curve: `f = slope*batch / t`. Usually 0.2-0.4, since decode MoE is dominated
  by a per-call fixed cost.
- Validated on 39 paired GB200 runs (real vs uniform routing): median error
  +0.003, mean absolute 0.020.
- Recorded routing exists only for Qwen3-235B; the other four use synthetic
  per-layer draws at Qwen3's measured per-layer CV.
- Per row it shows in `moe_provenance` as `x1.xxx`; on AFD rows also
  `moe_ms / moe_ms_g1` (`_g1` = no skew).

## How deployments are compared

For each (model, ISL, OSL, MTP) each arm independently picks its best
deployment -- highest tokens/s/GPU, F GPUs included in the denominator --
without a latency limit and under TPOT limits 10/15/20/30/50/100/150 ms. The
ratio is AFD / colocated. A gain is admitted to `AFD_Perf_Gain > 10%` only if it
holds across 0.9x-1.2x of the limit, since both arms' throughput is a staircase
in the limit. Memory walls (largest batch that fits) are computed separately and
cached.

## How to run

```bash
export AFD_MEGAMOE_ROOT=<unpacked afd_megamoe_data>
export AIC_PY=<python of the env with aiconfigurator installed>
export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<GB200 partition>
cd tools/afd_megamoe_gb200

bash make_overlay.sh                       # once: overlay for EP=24 (see below)
sbatch add_moeperf_ep24.sbatch
sbatch add_ep24.sbatch

sbatch verify_repro.sbatch                 # optional: sample vs the bundled reference -> RESULT: PASS
sbatch sweep_perlayer.sbatch               # -> handoff/gb200_perlayer_routing.csv (~1.5 h)
sbatch build_perlayer_deliverables.sbatch  # -> handoff/*.xlsx
```

Outputs: `gb200_perlayer_routing.csv` (one row per configuration),
`AIC_AFD_MegaMoE_GB200_NVL72_perlayer.xlsx` (best deployments, SLA sheets,
A:F sheet, GPU-budget sheet), `AFD_RoutingSkew_PerLayer_vs_LayerSummed.xlsx`.

**EP=24 overlay.** AIC ships MoE data only at power-of-two EP, so DeepSeek-V4-Pro
EP=24 needs extra rows. `make_overlay.sh` mirrors the checkout's systems data
into `$AFD_MEGAMOE_ROOT/aic_overlay` and links `tools/afd_megamoe_gb200/data` to
it; the two `add_*` jobs write the EP=24 rows there. The link matters: AIC
resolves a system's data relative to the directory holding its yaml, and the
`gb200_g*` yamls live in `tools/afd_megamoe_gb200/`.

## Files

| `tools/afd_megamoe_gb200/` | Role |
|---|---|
| `run_gb200_measured.py` | the grid, both arms, the AIC calls |
| `megamoe_table.py` | measured tables, lookups, skew coefficient |
| `afd_paths.py` | reads `AFD_MEGAMOE_ROOT` |
| `gb200_g{1,2,4,8}.yaml` | GB200 system specs |
| `capacity_wall.py`, `precompute_walls.py` | memory walls |
| `build_gb200_style.py`, `build_skew_comparison.py` | workbooks |
| `build_perlayer_distributions.py` | routing JSONs from FastAFD `expert_load_*.json` dumps |
| `make_overlay.sh`, `add_*.sbatch` | EP=24 overlay |
| `sweep_perlayer.sbatch`, `build_perlayer_deliverables.sbatch`, `verify_repro.sbatch` | jobs |

## Things that bite

- Run AIC on the GB200 node, not an x86 login node: there the aarch64 env fails
  every config with a numpy import error.
- `afd_t_attn_us` / `afd_t_ffn_us` in the workbooks are milliseconds per layer
  per microbatch.
- `AFD_ROUTING_AGGREGATED=1` reproduces the older layer-summed skew, for
  comparison only.

## Limits

Against end-to-end FastAFD runs (Qwen3-235B, 4K/1K, MTP 0) the AFD/colocated
ratio and the SLA crossover agree; absolute throughput is optimistic (about 7%
at a fixed batch, 10-30% at a fixed TPOT limit). AIC has no host/coordinator
overhead, which grows with the number of A GPUs, and the AFD arm has only
72-GPU units -- end to end, several small units in parallel (16A8F, 12A4F) do
better than one large unit. Four of five models' routing is synthetic.
