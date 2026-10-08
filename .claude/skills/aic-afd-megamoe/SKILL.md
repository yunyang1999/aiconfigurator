---
name: aic-afd-megamoe
description: Use when running, reproducing or extending the AIC (aiconfigurator) sweep that compares AFD (attention-FFN disaggregation) against colocated MoE serving on GB200 NVL72 using measured FastAFD MegaMoE stage times and a per-layer routing-skew coefficient -- setting it up from scratch, running the sweep, building the workbooks, checking that a checkout reproduces the reference results, or explaining how the model computes TPOT and the skew coefficient.
---

# AIC AFD x MegaMoE sweep on GB200 NVL72

Decode-phase comparison of AFD against colocated serving, in tokens/s per GPU on
a 72-GPU NVL72 domain, for Qwen3-235B-A22B, MiniMax-M2.5, MiniMax-M3,
DeepSeek-V4-Flash and DeepSeek-V4-Pro. AIC models everything except the MoE
stage; the MoE stage is a measurement taken on GB200 hardware.

You need three things, and nothing else:

1. **Code** -- branch `afd/gb200-megamoe-measured` of
   `https://github.com/yunyang1999/aiconfigurator`.
2. **Data** -- the bundle `afd_megamoe_data.zip` (measurements, routing
   distributions, a reference result). It unpacks to `afd_megamoe_data/`.
3. **Compute** -- a Slurm cluster with GB200 (aarch64) nodes and outbound
   network for `pip`.

## Quick start

Run from a GB200 compute node (an interactive allocation is fine). Replace the
three `<...>` values; everything else is copy-paste.

```bash
# 0. Where to work, and how jobs are submitted on your cluster
export WORK=<a directory with ~2 GB free>
export SBATCH_ACCOUNT=<your slurm account>
export SBATCH_PARTITION=<a GB200 partition>
cd "$WORK"

# 1. Code
git clone -b afd/gb200-megamoe-measured https://github.com/yunyang1999/aiconfigurator.git aic

# 2. Python environment -- create it ON a GB200 node (aarch64 wheels)
python3 -m venv venv
venv/bin/pip install -e aic/aic-core -e aic openpyxl

# 3. Data
unzip -q /path/to/afd_megamoe_data.zip -d "$WORK"

# 4. Environment the scripts read
export AFD_MEGAMOE_ROOT=$WORK/afd_megamoe_data
export AIC_PY=$WORK/venv/bin/python
export AFD_TOOLDIR=$WORK/aic/tools/afd_megamoe_gb200

# 5. AIC systems overlay (EP=24 rows AIC does not ship)
bash "$AFD_TOOLDIR/make_overlay.sh"
cd "$AFD_TOOLDIR"
sbatch --wait add_moeperf_ep24.sbatch
sbatch --wait add_ep24.sbatch

# 6. Prove the setup reproduces the reference -- must print "RESULT: PASS"
sbatch --wait verify_repro.sbatch && cat afd-megamoe-verify_*.out

# 7. Full sweep (~49.6k configurations, ~1.5 h) and workbooks
sbatch --wait sweep_perlayer.sbatch
sbatch --wait build_perlayer_deliverables.sbatch
ls handoff/
```

Outputs land in `$AFD_TOOLDIR/handoff/`:
`gb200_perlayer_routing.csv` (one row per configuration),
`AIC_AFD_MegaMoE_GB200_NVL72_perlayer.xlsx` (best deployment per scenario),
`AFD_RoutingSkew_PerLayer_vs_LayerSummed.xlsx` (skew-method comparison).
Slurm logs land in the submit directory as `<job-name>_<jobid>.out`.

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `AFD_MEGAMOE_ROOT` | yes | unpacked data bundle; must contain `results/` |
| `AIC_PY` | yes, for `.sbatch` | the venv's python |
| `AFD_TOOLDIR` | no | `tools/afd_megamoe_gb200` of the checkout; defaults to the submit directory |
| `SBATCH_ACCOUNT`, `SBATCH_PARTITION` | per cluster | read by `sbatch` itself; the job scripts carry no account or partition |
| `AFD_VERIFY_CSV` | no | reference for `verify_repro`; default `$AFD_MEGAMOE_ROOT/reference/gb200_perlayer_routing.csv` |
| `AFD_ROUTING_AGGREGATED=1` | no | reproduce the older layer-summed skew instead of per-layer |
| `AFD_ROUTING_RUN` | only to rebuild routing | FastAFD run dir whose `ray_logs/` holds `expert_load_*.json` |

## Data bundle

```
afd_megamoe_data/
  results/megamoe-m2n-nvl72/*.json         AFD arm: MegaMoE stage per (model, A:F split, seq per A rank, microbatches, nextN)
  results/megamoe-colocated-nvl72/*.json   colocated arm: MegaMoE stage per (model, EP, tokens per rank, nextN)
  results/megamoe-colocated-ep24/*.json    DeepSeek-V4-Pro at EP=24 (input to add_ep24.sbatch)
  real_routing_qwen3_235b_perlayer.json    recorded routing, 94 layers x 128 experts   (used)
  synthetic_perlayer_{128,256,384}.json    assumed routing for the other four models   (used)
  real_routing_qwen3_235b.json             layer-summed variants, only for AFD_ROUTING_AGGREGATED=1
  synthetic_cvmeasured_{128,256,384}.json
  measured_imbalance_coefficients.json     paired real-vs-uniform GB200 runs behind the skew validation
  projected_layout_spec.json               column layout of the projected workbook
  reference/gb200_perlayer_routing.csv     published sweep, the target verify_repro checks against
  README.md
```

`aic_overlay/` is created inside it by `make_overlay.sh`; it is not shipped,
because it must point at your own checkout.

Each measurement JSON's `stage_cuda.p50_ms` is the MoE stage for all layers of
one decode step: max over ranks per iteration (so straggler cost is included),
median over 40 timed iterations after 30 warmups.

## How the model works

### MoE stage: measured, not modelled

| Arm | Measured table | Lookup key | Entry into AIC |
|---|---|---|---|
| AFD | `megamoe-m2n-nvl72` | model, nextN, microbatches, experts per GPU (experts / F), sequences per A rank, `ag` = A / attention-TP | `cli_estimate(mode="afd", afd_moe_time_ms=...)` |
| Colocated | `megamoe-colocated-nvl72` | model, nextN, EP, tokens per rank | substituted into the `static_gen` op graph |

With `afd_moe_time_ms` set, AIC replaces the whole F side with it and sets every
A<->F communication term to zero: the measured stage already spans quant,
dispatch, wait, combine and the expert kernels, so nothing is billed twice.
Lookups outside the measured envelope return `illegal`; nothing is extrapolated.

The AFD pipeline (`pipeline_model="conservative"`) runs `m x L` cycles for `m`
microbatches and `L` layers:

```
t_f_layer = afd_moe_time_ms / m / L
t_cycle   = max(t_a_layer, t_f_layer)
tpot      = pipeline_fill + (m * L - 1) * t_cycle
```

`m` both shrinks each cycle's F time and multiplies the cycle count, so total
MoE work is conserved; what `m` changes is how much of it overlaps attention.

**The two measured tables cannot stand in for each other.** Colocated is an
all-to-all inside one EP group whose ranks also run attention, indexed by tokens
per rank (8..1024). AFD is an M2N transfer between two disjoint pools, indexed
by sequences per A rank and by `ag`, which colocated has no notion of. At
matched (experts per GPU, tokens per rank) they differ by a median 4.2% and up
to 43%. Converted to tokens per F rank, the AFD table covers 10..104448.

### Routing-skew coefficient, applied to both arms

```
coefficient = 1 + f * (I - 1)          # multiplies the measured balanced stage
```

- `I`, load imbalance: per layer, sum each rank's experts (contiguous blocks)
  and take busiest / mean; then **average over layers**. Summing the layers
  first is wrong: the busiest rank changes almost every layer, so the peaks
  cancel (at 32 ranks the layer-summed value is 1.31, the per-layer one 2.90).
- `f`, token-proportional share of the stage at this batch: least-squares fit
  `t = fixed + slope * batch` to the measured balanced curve, `f = slope*batch / t`.
  Typically 0.2-0.4 because decode MoE is dominated by a per-call fixed cost
  (~45 us per call, twice that at `m=2`) -- which is why a 2-3x load imbalance
  costs only ~1.1-1.4x in time.
- Validated against 39 paired GB200 runs (real vs uniform routing, alternating
  in one allocation): median error +0.003, mean absolute 0.020.
- Only Qwen3-235B's routing is recorded (a FastAFD serving run, decode only).
  The other four models use 94 independent synthetic layers at Qwen3's measured
  per-layer CV (136%). Their coefficients are assumptions.

Read a row's coefficient from `moe_provenance`, e.g.
`measured mb=2 ag=16 (a_tp=4) x1.166 (routing skew recorded, F=8)` -- `x1.166`.
On AFD rows, `moe_ms / moe_ms_g1` is the same number (`_g1` = without skew).

## Output columns worth knowing

- `status`: `ok`, `oom`, or `illegal` (outside the measured envelope).
- `tokens_s_gpu`: per GPU of the deployment unit, F GPUs included.
  `tpsgpu_domain72`: also charges GPUs left idle when the unit does not tile 72.
- `afd_t_attn_us`, `afd_t_ffn_us` in the workbooks are **milliseconds per layer
  per microbatch**, despite the suffix.
- Workbooks keep each arm's best deployment per (model, ISL, OSL, MTP[, SLA]).

## Files

| `tools/afd_megamoe_gb200/` | Role |
|---|---|
| `afd_paths.py` | reads `AFD_MEGAMOE_ROOT`; every script gets the data root from here |
| `megamoe_table.py` | `MeasuredStage`: measured tables, lookups, skew coefficient |
| `run_gb200_measured.py` | sweep driver; `ROUTING_DIST` selects the distributions |
| `gb200_g{1,2,4,8}.yaml` | GB200 system specs with 1/2/4/8 GPUs per AIC node |
| `make_overlay.sh` | builds `$AFD_MEGAMOE_ROOT/aic_overlay` and the `data` link |
| `add_moeperf_ep24.sbatch`, `add_ep24.sbatch` | write the EP=24 rows into the overlay |
| `verify_repro.py`, `verify_repro.sbatch` | re-run a sample, diff against the reference |
| `sweep_perlayer.sbatch` | the sweep |
| `capacity_wall.py`, `precompute_walls.py` | memory walls, cached in `handoff/capacity_walls.json` |
| `build_gb200_style.py`, `build_skew_comparison.py`, `build_afd_benefit_workbook.py`, `build_clean_workbook.py` | workbooks |
| `build_perlayer_deliverables.sbatch` | walls (if missing) + both workbooks |
| `build_perlayer_distributions.py` | rebuild routing JSONs from a FastAFD capture |

Library changes this branch carries (all required):
`afd_moe_time_ms` in `src/aiconfigurator/sdk/inference_session.py` and the CLI;
`enforce_head_tp_divisibility=False` for the F-worker (`aic-core` `config.py`,
`config_builders.py`, `models/base.py`, `src/aiconfigurator/cli/api.py`) --
the F-worker runs no attention, and DeepSeek-V4-Pro 48A24F fails the head-count
check without it; the DeepSeek-V4 entries in
`tools/afd_megamoe_b200/colocated_measured.py`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every config fails with `Unable to import required dependency numpy` | running on an x86 login node with an aarch64 venv | run on a GB200 node |
| `AFD_MEGAMOE_ROOT is not set` / `has no results/` | data root not exported, or pointed one level too high/low | point it at the directory that contains `results/` |
| `verify_repro` shows `status_changed` on `ep24` rows | overlay missing or `data` link not pointing at it | rerun step 5; `readlink $AFD_TOOLDIR/data` must end in `aic_overlay/data` |
| `make_overlay.sh` says the overlay exists | rebuilding over an old one | `rm -rf $AFD_MEGAMOE_ROOT/aic_overlay`, rerun step 5 |
| AFD "token range only reaches 1024" | an AFD table built from the colocated data | use `megamoe-m2n-nvl72` for AFD |
| `KeyError` on a DeepSeek-V4 model / head-count assertion at 48A24F | an AIC checkout without this branch's changes | use this branch |
| `sbatch: error: ... account/partition` | `SBATCH_ACCOUNT` / `SBATCH_PARTITION` unset | export them, or pass `-A` / `-p` |

Two rules behind the overlay, both learned the hard way: AIC resolves a
system's `data_dir` relative to the systems root where that system's yaml was
found, so for `gb200_g*` the overlay only takes effect through
`$AFD_TOOLDIR/data`; and writing to a symlinked file edits whatever it points
at, so the builders unlink before writing.

## What the results say, and their limits

Compared with end-to-end FastAFD runs (Qwen3-235B, 4K/1K, MTP 0), AIC gets the
**AFD/colocated ratio and the SLA crossover right**, and is **optimistic in
absolute throughput** (about 7% at a fixed batch, 10-30% at a fixed TPOT limit;
up to 36% for a 72-GPU AFD unit at large batch). Known gaps:

- No host/coordinator overhead; FastAFD's grows with the number of A GPUs.
- The AFD arm only has 72-GPU units (40A32F, 56A16F, 64A8F, 68A4F). End to end,
  several small units in parallel (16A8F, 12A4F) beat one large unit and keep
  AFD level with colocated up to ~40 ms TPOT; AIC's crossover sits near 25 ms.
- The batch grid is coarse at the low end (8 -> 16); at tight limits quote
  medians, not the extreme multiples.
- Four of five models' routing is synthetic.
