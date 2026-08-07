---
name: afd-megamoe-sweep
description: Runbook for the homogeneous AFD-vs-colocated sweep on B200 with measured FastAFD MegaMoE times injected into AIConfigurator. Use when asked to run, extend, re-run or debug this sweep, to add a model/split/batch rung, or when a MegaMoE benchmark or AIC estimate arm needs re-measuring. Covers the sbatch launch, both AIC arms, the workbook build, and the traps that produced wrong numbers before.
---

# AFD vs colocated on B200, with measured MegaMoE

Produces `AIC_AFD_Homogenous_MegaMoE_B200.xlsx` — the same seven sheets and
column names as the GB200 `AIC_AFD_Homogeneous_GB200_MultiModel_v0514.xlsx`
reference, with every MoE stage time replaced by a direct FastAFD benchmark.

Three models: Qwen3-235B-A22B, MiniMax-M2.5, MiniMax-M3.
Precision `w4a8_mxfp4_mxfp8` — E2M1 weights, UE8M0 block-32 scales, E4M3
activations. **This is not NVFP4; do not describe it as NVFP4.**

## Why the measurement is mandatory

`b200_sxm`'s SILICON database has no `w4a8_mxfp4_mxfp8` MoE rows, so AIC cannot
produce this precision on this system at all without injection. And where it
does model MoE, it is wrong in a direction that matters: AIC's generic MoE grows
2.7x from batch 8 to 32 while the measured kernel moves 4%, because MegaMoE is
weight-bandwidth bound. A generic sweep overstates colocated cost at large batch.

## Layout

| path | what |
| --- | --- |
| `/home/scratch.yuny_wwfo_1/megamoe` | working root — **not** `~`, local disk is too small |
| `FastAFD/` | benchmark repo, branch `megamoe-multimodel-b200` |
| `results/megamoe-*/` | raw benchmark JSONs |
| `handoff/` | intermediate CSVs and the workbook |
| `/home/scratch.yuny_wwfo_1/aic/aic-yy` | AIConfigurator, branch `pr1323-afd-megamoe-pr1479` |
| `/home/scratch.yuny_wwfo_1/aic/venv-aic/bin/python` | **runs AIC** (has pandas + the SDK) |
| `/bin/python3` | **builds workbooks** (has openpyxl, no SDK) |

Two interpreters, and they are not interchangeable. Using `/bin/python3` for an
AIC run fails with `ModuleNotFoundError: pandas`.

## The live pipeline

Only these files are current. Everything else in the working root is exploratory
and should be ignored: `build_workbook.py`, `build_main_result.py`,
`build_measured_only.py`, `build_measured_sla.py`, `build_pareto.py`,
`run_aic_colocated.py`, `run_aic_scaleout.py`, `run_aic_serial.py`,
`run_aic_generic.py`, `agg_decode_only.py`, `megamoe_time_model.py`.

```
sbatch benchmarks  ->  results/megamoe-*/            (FastAFD, on B200)
        |
        +-- run_aic_sweeps.py      -> handoff/aic_afd_sweeps_final.csv       (AFD arm)
        +-- colocated_measured.py  -> handoff/aic_colocated_measured.csv     (colocated arm)
                    |
                    +-- build_v0514_style.py -> handoff/AIC_AFD_MegaMoE_B200_v0514style.xlsx
```

Shared helpers: `stability.py` (which points to accept), `colo_metrics.py`
(decode-only pricing), `megamoe_extrapolate.py` (physical model, only used where
a rung was never measured).

---

# Step 1 — Benchmark MegaMoE on B200

## 1.1 Submit

`submit.sh` wraps FastAFD's sbatch and overrides two things the upstream file
hardcodes: `--account=beta-users` (we do not hold it; we use `wwfo-china`) and
`--qos=interactive-isolated-short` (caps concurrency; we use `batch-short`).
Both are overridden on the command line so the repo tree stays clean — the
sbatch records `source_dirty` in every result JSON and that should stay `false`.

```sh
cd /home/scratch.yuny_wwfo_1/megamoe
RUN_SCRIPT=run_megamoe_m2n_model_suite.sh \
RESULTS_DIR=/workspace/results/megamoe-m2n-<tag> \
SEQUENCES_PER_AG_RANK_GRID="1 2 4 6 8 16 32 48 64 96 128 192 256 384" \
AFD_SPLIT_GRID="4:4 5:1 6:2 7:1" \
MICROBATCH_GRID="1" \
MODELS="qwen3_235b_fp4" \
MTP_NEXTN_GRID="0 1 2" \
BACKEND_GRID=mega \
WARMUPS=30 ITERATIONS=40 \
TIME_LIMIT=00:50:00 \
JOB_NAME=mm-qwen3 \
  ./submit.sh
```

For the colocated kernel use `RUN_SCRIPT=run_megamoe_colocated_model_suite.sh`,
`TOKENS_PER_RANK_GRID=...` and `EP_SIZE=1|2|4|8`.

## 1.2 Four launch traps

**`SLURM_CPU_BIND=none` is required.** The upstream srun sets no `--cpu-bind`,
and on nodes whose allocation is NUMA-split (umb-b200-262 hands out two 48-CPU
blocks) the default binding lands outside the allocation and srun dies in
6 seconds with exit 64, *"Unable to satisfy cpu bind request"*. Other nodes have
a contiguous mask and work, so the failure looks random. `submit.sh` exports it;
do not remove that line.

**Size TIME_LIMIT to the work.** Comparable jobs run 3–17 minutes. Asking for
the queue maximum of 3:55 left jobs sitting in `(Resources)` for over an hour
because a window that long is hard for the backfill scheduler to find; the same
jobs started within minutes at 40–50 minutes. Give 3–5x headroom, not 20x.

**One job per model.** These OOM per-model, and a single combined job loses
everything when one model fails.

**`BACKEND_GRID=mega`.** The default is `mega deepep`, and the DeepEP reference
arm refuses EP=1 outright (*"Unsupported number of EP ranks: 1"*), failing the
whole job.

## 1.3 Known impossible configurations

- **MiniMax-M3 with F=1** (5A1F, 7A1F) OOMs at the *smallest* batch — one GPU
  cannot hold all 128 experts alongside everything else (171.84 GiB allocated,
  2.27 GiB requested). Do not retry; record as absent.
- **F must divide the expert count**, so F ∈ {1, 2, 4}.
- **A + F ≤ 8.** MegaMoE's dispatch/combine buffer is
  `torch.distributed._symmetric_memory` and the kernel dereferences one raw peer
  pointer per rank (`mega_m2n.hpp:181`), so every rank must sit in one NVLink
  domain. There is no cross-node MegaMoE on B200 SXM.

## 1.4 Check the results

```sh
/bin/python3 -c "
import sys,json,glob,collections; sys.path.insert(0,'/home/scratch.yuny_wwfo_1/megamoe')
from stability import accept
c=collections.Counter()
for p in glob.glob('/home/scratch.yuny_wwfo_1/megamoe/results/megamoe-*/*.json'):
    j=json.load(open(p)); c['total']+=1; c['accepted']+=accept(j)
print(dict(c))"
```

**The acceptance gate is ours, not FastAFD's.** FastAFD stamps
`stable = cv_percent <= 3.0 and all(finite)`
(`benchmark_megamoe_colocated_models.py:873`), where CV is the spread across the
40 timed iterations of one run — not across repeated jobs. `stability.py` raises
the dispersion limit to **4%** and keeps the finiteness check untouched, because
3% was discarding small-batch rungs whose values sit exactly on the flat trend
their neighbours describe (Qwen3 ep=8 nextN=0: batch 4 at 7.020 ms CV 3.45%,
batch 6 at 7.102 ms CV 3.63%, between measured 6.528 and 6.919). A short kernel
at small batch carries more launch jitter — a property of the measurement, not
of the kernel. Dispersion is about repeatability; non-finite output is about
validity, so only the first is relaxed.

---

# Step 2 — AIC estimates, AFD arm

```sh
cd /home/scratch.yuny_wwfo_1/megamoe
/home/scratch.yuny_wwfo_1/aic/venv-aic/bin/python run_aic_sweeps.py \
    --workers 14 --out handoff/aic_afd_sweeps_final.csv
```

~4700 configurations, ~25 min at 14 workers. Driven by the measured index, so
new benchmark rungs appear automatically once their directory is listed in
`SPLIT_DIRS`.

Injection is `--afd-moe-time-ms`, which `inference_session.py:1425` turns into
`t_f_step_i = afd_moe_time_ms / num_microbatches`, replacing **all F-side
compute and the stage communication** (`T_a2f` goes to zero). Attention, KV and
the pipeline model stay AIC's.

Four flags that are not optional:

| flag | why |
| --- | --- |
| `--no-afd-combined-with-pd` | makes the AFD arm a decode worker; AIC then reports TTFT 0 |
| `--nextn-accepted 0.85 / 1.53` | AIC refuses `nextn>0` without it. Values taken from the GB200 sheet's own Configuration tab, so the two studies stay comparable |
| `--database-mode HYBRID` for M3 | MiniMax Sparse Attention has no silicon rows; AIC says *"MSA has no silicon data"*. Only M3's **attention** falls back — its MoE is still the injected measurement |
| `--systems-paths default,aic_systems` + `b200_sxm_subnode` | `b200_sxm` declares 8 GPUs/node so a sub-node split like 6A2F cannot be expressed. The custom system sets 1 GPU per AIC-node and raises `inter_node_bw` to the NVLink figure, which is what those links physically are inside one node. **Valid only while total GPUs ≤ 8** |

Capture stderr: AIC raises OOM as a traceback there, and OOM is a real outcome
(the sheet calls it `batch_limited_by=capacity`), not a tooling failure. It also
carries the pipeline-degradation warning described next, which is easy to miss
if only OOM is matched.

## 2.1 Pipeline model, and the mb=1 correction

Run `--pipeline-model conservative`. With the measured stage injected
`t_a2f = t_f2a = 0`, so `max(t_a, t_f, t_c)` and `max(t_a + t_a2f, t_f + t_f2a)`
are numerically identical — verified to the digit. Conservative is the weaker
claim: nothing in this setup demonstrates three in-flight microbatches hiding a
network stage, so do not assert it.

`pipeline_model` and `num_microbatches` are **independent knobs**, and mb=1/2/3
do **not** map to serial/conservative/optimistic. mb affects `t_cycle` in exactly
one place — the optimistic branch's `min_m` gate — and otherwise only scales the
slot count in
`t_global_step = pipeline_fill + t_cycle × (mb × layers − 1)`.

That formula makes more microbatches **worse**, not better, once `t_a` becomes
the binding stage. Holding everything else fixed on MiniMax-M3 4A4F batch 16:

| mb | T_a/layer | T_f/layer | TPOT | concurrency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 145 µs | 188 µs | 11.4 ms | 64 |
| 2 | 136 | 94 | 16.4 | 64 |
| 4 | 124 | 47 | 29.8 | 64 |

`t_f` is the injected time divided by mb, so it shrinks; the slot count grows
proportionally; and once `t_a > t_f` the cadence is pinned at `t_a` while the
slots keep multiplying. On top of that the *measured* mb=2 stage is not half the
mb=1 one — MegaMoE re-streams the expert shard per lane, so Qwen3 4A4F at 384
tokens measures 10.145 ms at mb=1 and **16.765 ms** at mb=2.

**The correction.** AIC's K=3 and K=2 cadences declare `N_min` of 3 and 2
in-flight microbatches, but only the optimistic branch enforced its threshold —
and its fallback landed on conservative, which needs two as well. A single
in-flight microbatch cannot overlap anything: layer i+1's A input **is** layer
i's F output, so the pools strictly alternate with nothing to fill the gaps.
This branch was added to `_pipeline_tcycle`, before the optimistic and
conservative branches:

```python
if num_microbatches < 2:
    return t_a + t_a2f + t_f + t_f2a, False
```

Only mb=1 changes; mb≥2 reproduces bit-for-bit. On the M3 config above, mb=1
goes from 11.4 ms to 20.0 ms.

It moved the headline: the no-latency-limit ratio went from **1.000 (AFD ahead
30/60)** to **0.751 (AFD ahead 0/60)**, and `AFD_Perf_Gain > 10%` from 143 rows
to 26.

**This is pending confirmation with the PR #1323 authors** — the missing `N_min`
check may be an omission or an intentional fine-grained-overlap assumption. The
pre-correction sweep is kept at
`handoff/aic_afd_sweeps_optimistic_mb1max.csv.bak` for comparison.

Note what did *not* happen: the optimiser still picks mb=1 in 400 of 739 rows
after the correction (was 442). Making mb=1 1.75x more expensive did not push it
to mb=2, because mb=2 is worse for a different reason. AFD has no cheap overlap
available on this hardware — mb=1 gets none, mb=2 gets some but pays 65% more
MoE for it.

---

# Step 3 — AIC estimates, colocated arm

```sh
/home/scratch.yuny_wwfo_1/aic/venv-aic/bin/python colocated_measured.py \
    --workers 14 --out handoff/aic_colocated_measured.csv
```

~9100 configurations, ~30 min. The CLI refuses `--afd-moe-time-ms` outside AFD
mode, so the substitution happens on the op breakdown:

```
new_step = generation_step
         − (generic MoE / shared-expert ops)
         + measured MegaMoE colocated stage
         + residual for layer-equivalents the measurement does not cover
```

`generation_router_gemm` is **kept** — FastAFD's measured stage excludes the
router, so removing it too would double-subtract.

## 3.1 Use `static_gen`, not `agg` — this is the memory fix

Both give the same decode step: agg's `genonly_step` total and static_gen's TPOT
agree to the digit (14.914 ms on Qwen3, isl=4096, batch 8). They size
**activation memory** differently, and only static_gen sizes it for decode.

AIC's agg path takes activations from `BuildConfig.max_num_tokens`, which
`cli_estimate` defaults to `isl` (`api.py:941`), and `memory.py:273` states that
`max_batch_size` does not affect non-KV memory. So the agg colocated arm was
charged a prefill chunk's activations — 2.016 GiB at isl=4096 rising to
**32.203 GiB at isl=65536 with the batch unchanged** — while the AFD arm, run
prefill-disaggregated, pays none of it. That is 18% of a 180 GiB card taken from
the colocated KV budget for work a decode worker never does, and it forced OOM
at batches that fit: isl=131072 batch 8 reads 184.05 GiB under agg and
119.75 GiB under static_gen.

Two consequences of using static_gen, both handled in `colocated_measured.py`:

- **Its op latencies are cumulative over the generation phase.** Divide by
  `osl - 1` to recover per-step. Verified: the cumulative sum over `osl-1`
  reproduces its own TPOT exactly.
- **It runs no OOM check at all.** isl=131072 batch 64 returns a result
  reporting 782.89 GiB on a 180 GiB card. Capacity is enforced ourselves against
  `CAPACITY_GIB = 180.0`.

Do **not** reach for `res.per_ops_data` here — it is `None` in static_gen mode.
The breakdown is on `res.summary.get_generation_latency_dict()`. Concluding
"static_gen has no op breakdown" from the first field alone cost a day.

## 3.2 Price the residual at the measured rate

AIC bills `q·layers + nextN` layer-equivalents of MoE work (`q = nextN+1`); the
measurement covers `q·moe_layers`. Charge the remainder at the **measured**
per-layer rate, not AIC's generic one — the substitution exists precisely because
the generic rate is wrong, and charging it here reintroduces that bias, worst at
large batch.

| model | layers / MoE layers | residual, nextN=0 | nextN=2 |
| --- | --- | ---: | ---: |
| Qwen3-235B | 94 / 94 | **0.00%** | 0.70% |
| MiniMax-M2.5 | 62 / 62 | **0.00%** | 1.06% |
| MiniMax-M3 | 60 / 57 | 5.00% | 6.04% |

Qwen3 and M2.5 are all-MoE, so at nextN=0 the step is pure measurement plus
AIC's attention. Only M3 carries a real residual — 3 dense layers priced with
the MoE proxy, the same proxy AIC itself applies.

---

# Step 4 — Build the workbook

```sh
/bin/python3 build_v0514_style.py
cp handoff/AIC_AFD_MegaMoE_B200_v0514style.xlsx \
   /home/yuny/kernel_arena/AIC_AFD_Homogenous_MegaMoE_B200.xlsx
```

Seven sheets, column names identical to the GB200 reference:
`Configuration`, `AFD_Perf_Gain > 10% `, `Model`, `Context`, `ISL_x_OSL`,
`SLA_Latency`, `AF_Split`.

## 4.1 The two conventions that define `ratio`

Both copied from the GB200 sheet and verified against its own rows:

```
ratio          = afd_tpsgpu / colo_tpsgpu     (3653/4935.5 -> 0.74;  732.9/648.7 -> 1.13)
tokps_per_user = tpsgpu * unit_gpus / conc    (reproduces its 8.15 and 10.28)
```

`ratio` is **tokens/s/GPU, not latency**, and **each arm searches its own best
batch** — concurrency is deliberately not held equal. Both arms are also priced
decode-only, which the GB200 sheet does too: its rows reproduce as
`conc / TPOT / gpus` with no prefill term.

## 4.2 batch_limited_by

Four mutually exclusive categories, evaluated **inside the winner's own
topology**, looking only at the next rung:

| value | meaning |
| --- | --- |
| `capacity` | the next rung does not fit |
| `latency` | the next rung fits but breaks the TPOT ceiling |
| `compute` | the next rung fits and is allowed, but does not raise tokens/s/GPU |
| `grid` | there is no next rung — an open end, not a limit |

Pooling OOM records across topologies is wrong and was a real bug: a 7A1F OOM at
batch 32 says nothing about 4A4F, and an ep=4 OOM at tokens-per-rank 8 is a
different total load from an ep=8 one. Pooling made every no-SLA row report
`capacity`.

Expect `capacity` to dominate and `compute` to be nearly absent. That is
physical, not a bug: the measured stage is flat in batch while tokens grow, so
tokens/s/GPU rises monotonically until memory runs out.

## 4.3 Do not add columns

The deliverable matches the reference column-for-column. Columns that were tried
and removed: `afd_comm_hidden` and the three `colo_*memory*` columns (blank —
the estimate path does not report per-GPU weight memory), the TBO ceiling group,
and the `colo_any_*` provenance group. Only the SLA-sensitivity trio stays:
`ratio_sla_plus10`, `sla_edge_sensitive`, `gain_survives_sla_plus10`.

Verify parity before shipping:

```sh
/bin/python3 -c "
import openpyxl
a=openpyxl.load_workbook('/home/yuny/kernel_arena/AIC_AFD_Homogenous_MegaMoE_B200 (1).xlsx')
b=openpyxl.load_workbook('/home/scratch.yuny_wwfo_1/megamoe/handoff/AIC_AFD_MegaMoE_B200_v0514style.xlsx')
for n in a.sheetnames:
    if n=='Configuration': continue
    ca=[c.value for c in a[n][2] if c.value]; cb=[c.value for c in b[n][2] if c.value]
    print(n, 'OK' if ca==cb else f'DIFF extra={[x for x in cb if x not in ca]}')"
```

---

# Traps that produced wrong numbers

Each of these shipped a wrong result before being caught. They are ordered by
how much they moved the answer.

**Prefill amortised into colocated TPOT.** The AFD arm is prefill-disaggregated;
AIC's agg TPOT is not. Comparing them charges colocated for work AFD never does,
by 3.185x at osl=256, 1.532x at osl=1024, 1.117x at osl=4096. Price colocated on
its gen-only step. *(Fixed. `colo_metrics.decode_only`.)*

**Prefill activations in colocated memory.** Section 3.1. *(Fixed by
static_gen.)*

**Colocated batch multiplied by EP.** AIC's `--batch-size` with
`--attention-dp-size 8` is **already per attention-DP rank** — a run with
`--batch-size 96 --attention-dp-size 8` reports `Concurrency: 768`. Multiplying
by ep silently runs 8x the intended tokens per rank.

**Asymmetric batch grids between arms.** When the colocated kernel had 1/2/4/6
measured and the split kernel stopped at 8, colocated could reach TPOT ceilings
AFD could not, and the 15 ms row read 3.09x purely from that. Keep both grids
identical.

**Rescaling instead of re-optimising under a ceiling.** Any what-if that changes
the cost model must re-run the configuration search, not scale the old winner —
31% of rows change winning configuration.

**mb=1 was silently given an overlap it cannot sustain.** See §2.1. The
pipeline is across MICROBATCHES, not across layers — `min_m` in the source is
literally counted in "in-flight microbatches". An earlier reading of this as a
cross-layer pipeline was wrong, and wrong in the direction that flatters AFD by
1.75x on the affected rows.

**Microbatching is net negative on this hardware.** Qwen3 4A4F, 384 tokens:
mb=1 gives 10.145 ms injected / 3166 tok/s/GPU; mb=2 gives 16.765 ms / 1986.
MegaMoE is weight-bound, so two lanes re-stream the whole expert shard per layer
while only halving the tokens. Keep `MICROBATCH_GRID="1"` unless the question is
specifically about microbatching.

**batch = 6 is a kernel anomaly, not a data error.** It is *slower than batch 8*
on 6 of 10 (model, split) combinations — 1.5x the work taking 1.9x the time,
CV 1.03%, reproducible. Probably tile/wave quantisation. It matters because at
isl=131072 only 4–7 sequences per rank fit, so batch 6 sits in the window long
context is forced into. The optimiser routes around it; interpolating through it
does not.

**Cross-tree numeric comparisons prove nothing.** Two AIC checkouts differed in
324 files. To attribute a difference to one commit, copy only that commit's
files into one tree and confirm the numbers reproduce bit-for-bit. That is how
the `perf_interp` adoption (liz-badada@23bd13d5, joint-log2 kNN past the
frontier) was verified: attention 21.680 / 24.930 / 22.385 / 14.401 ms on four
configs. Headline impact was **+0.7% median** on the ratio, not the −35% the
hand-picked corners suggested.

**MiniMax-M3 has no generic-MoE arm.** Without injection AIC cannot price its
A→F stage transfer: *"No empirical utilisation data to estimate this op at
query=(tokens × 6144)"*. It fails at every split and batch, so any
generic-vs-measured control covers Qwen3 and M2.5 only.

---

# Adding a model, split or batch rung

1. Benchmark it (Step 1). Give the results directory a new `megamoe-*` name.
2. Add that directory to `SPLIT_DIRS` in `run_aic_sweeps.py` (split kernel) or
   to `measured_index()` and the extrapolator's `raw_dirs` in
   `colocated_measured.py` (colocated kernel).
3. Add the rung to `MEASURED_RUNGS` in `colocated_measured.py` if it is a new
   batch value — `TPR_GRID` is derived from it.
4. Re-run Steps 2–4. Both drivers are driven by the measured index, so nothing
   else needs editing.

For a new model also add it to `MODELS` (path, `layers`, `moe_layers`) in
`colocated_measured.py`, to `MODEL_PATH` in `run_aic_sweeps.py`, and to
`MODEL_LABEL` / `ATTN` in `build_v0514_style.py`. Set `DATABASE_MODE` to
`HYBRID` if its attention has no silicon coverage.

# Background jobs survive the session, waiters do not

Long runs must be launched detached, or they die when the session ends:

```sh
(setsid nohup <cmd> > handoff/<name>.log 2>&1 &)
```

A `until [ -f ... ]; do sleep; done` waiter started in the foreground is killed
on session teardown even though the job it watches survives. For a chain, put
the whole sequence in a small `sh` script and `setsid` that.
