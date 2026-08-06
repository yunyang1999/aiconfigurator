---
name: aic-afd-homogeneous-sweep
description: Use when sweeping AFD (attention-FFN disaggregation) against a colocated baseline on one homogeneous rack in AIC and turning the result into a decision workbook. Covers arm symmetry, the split/batch/context grid, sharded execution, workbook assembly, the two-batch-overlap ceiling column, and the traps that have each silently flipped a conclusion.
---

# AIC Homogeneous AFD Sweep

Compare an AFD deployment against a colocated one on the same rack, over the
same models and workloads, and produce a workbook a reader can act on without
re-deriving anything.

The reference implementation is in `scripts/` and was used to build the
GB200 / SGLang 0.5.14 workbooks. Treat it as a starting point to edit, not a
black box: the model list, contexts and system are hard-coded on purpose so
every run declares its own scope.

## What a run produces

Two arms, per grid cell:

- **AFD** — `cli_estimate(mode="afd", ...)`: an A pool of `n_a_nodes` and an F
  pool of `n_f_nodes`, swept over splits, `tp_a`, and batch.
- **Colocated** — `cli_estimate(mode="static_gen", ...)`: the same model on one
  undisaggregated pool, swept over `(tp, dp, moe_tp, moe_ep)` and batch.

A cell is `(model, precision, isl, osl, mtp_nextn)`. Both arms pick their own
best configuration inside the cell under the same constraint, then
`ratio = afd_tokens_s_gpu / colo_tokens_s_gpu`.

Sheets: one per axis the reader varies (context, ISL×OSL, model×precision), one
per constraint (latency SLA, per-user speed target), one enumerating the A:F
splits, one fixed-GPU-budget sheet, plus `Configuration` defining every column.

## Phase 0 — pin the engine, then prove it is pinned

Three things silently change results. Assert all three before the first shard.

1. **Which package got imported.** A relative `PYTHONPATH` loads the *installed*
   wheel, not the worktree. Print and check:
   ```python
   print(aiconfigurator.__file__, aiconfigurator_core.__file__)
   ```
   Both must be under the worktree. Use absolute paths in `PYTHONPATH`.
2. **Which perf-data version.** For `gb200/sglang` the versions differ in
   *content*, not recency: `0.5.10` is the only complete 24-table set; `0.5.9`
   has 9 tables and much more optimistic large-token MoE data, which flatters
   AFD roughly twice as much as colocated; `0.5.12` carries only
   `SHARED_LAYER_REUSE.txt`. `sorted()` returns `0.5.9` as the max
   (lexicographic), so "pick the newest" selects the least complete. Name the
   version explicitly.
3. **Which comm model.** Pass `afd_comm_model=` explicitly. The AFD cross-pool
   comm model is an active area of change; a run that does not record the model
   name cannot be compared against an earlier one.

If the Python interpolation engine changed, no rebuild is needed — the compiled
extension exports only `AicEngine`, `RustForwardPassPerfModel` and
`engine_spec_bincode_from_json`, and `perf_interp.query` resolves to
`perf_interp/engine.py`. If the *Rust* interpolation source changed, the `.so`
must be rebuilt (`maturin build --release`) or the change is inert.

## Phase 1 — make the two arms comparable

Every item here has produced a wrong headline at least once.

- **`afd_combined_with_pd=True`.** With `False` the other phase is neither
  modelled nor its GPUs counted, so half a system is compared to a whole one.
  Alternatively run the baseline as `disagg` so it is PD-vs-PD on both sides —
  but then the AFD arm must be disagg too, and per-role fields
  (`prefill_model_path`, `decode_moe_quant_mode`, ...) are required; top-level
  equivalents are rejected.
- **Set `agg_num_gpu_candidates` explicitly.** The colocated search defaults to
  a maximum of 8 GPUs. Left alone it reports ratios above 20 — it is comparing a
  64-GPU AFD deployment against an 8-GPU baseline.
- **Do not let the baseline sit at `tp>1` attention by default.** A tp>1
  attention baseline manufactured a 2.5x AFD win in one run; sweep
  `tp ∈ {1,2,4,8}` with matching `dp` and let the baseline choose.
- **Raise `afd_max_af_ratio`** (default 4.0) if the intended split range exceeds
  it, or the split search is clipped without saying so.
- **The colocated arm's communication is billed fully serial.** Real SGLang runs
  two-batch overlap. Give the baseline that credit in a separate column rather
  than leaving the exposure unstated — see Phase 4.

## Phase 2 — the grid

- **Batch grid must be dense, not powers of two.** With a power-of-2 grid, 96→128
  skips every batch HBM actually allows in between; densifying to 53 rungs moved
  a named case to batch 112 and changed its ratio. In one full sweep **69 of 72
  optima landed on non-power-of-2 batches**. Use the rung list in
  `scripts/sweep_afd.py`.
- **Splits**: `[(a,f) for f in (1,2,4,8) for a in range(1,18) if (a+f)*gpus_per_node <= rack]`.
- **Contexts and OSL differ per workbook.** Confirm the OSL grid per workbook
  before assuming; one backfill was needed because the multi-model book carried
  `osl ∈ {256,1024,4096}` while a derived run used only two.
- **MTP**: `nextn ∈ {0,1,2}` with `nextn_accepted` 0.85 / 1.53. Verify the MTP
  path is actually engaged — at `nextn=0` an MTP fix is inert by construction,
  which makes it a poor smoke test.
- **Memory cap** (`MEM_CAP`) is the admission filter; state its value in the
  workbook, since it decides which configurations exist at all.

## Phase 3 — run sharded, merge, build

Shard over `(model, context)` round-robin so each worker gets a mix of cheap and
expensive cells rather than one worker owning every long context.

```bash
for S in $(seq 0 47); do
  setsid nohup env WORKBOOK=multi SHARD=$S NSHARD=48 OUT=$D/out_$S.csv \
    PYTHONPATH=$ABS python3 scripts/sweep_afd.py > $D/log_$S.log 2>&1 < /dev/null &
done
```

- Track progress by **row count**, not file count — CSVs are created immediately
  and written incrementally.
- Never `pkill -f <script>`: the pattern matches your own shell and kills the
  session (exit 144). Match on `/proc/<pid>/environ` instead.
- A backslash inside an f-string is a `SyntaxError`; the whole script then never
  runs and the shard silently produces nothing. Compile before launching:
  `python3 -c "import ast; ast.parse(open(f).read())"`.

Then `scripts/build_workbook.py <merged.csv> <out.xlsx> <title> <book> fixed`.

## Phase 4 — derived sheets

- `scripts/add_budget_sheet.py` — fixes a GPU budget, lets each arm pick its own
  unit and replicate, and reports absolute totals plus stranded GPUs. Ratio-only
  sheets assume infinite replication and hide both.
- `scripts/patch_limited_by.py` — recomputes `batch_limited_by` in place on a
  finished workbook, rewriting only those cells and the `Configuration` line
  that defines them. Prefer this over rebuilding when the correction is to one
  derived column: a rebuild re-runs every downstream step and re-opens every
  formatting decision, which makes the diff impossible to review.
- `scripts/sweep_tbo.py` + `scripts/tbo_probe.py` + `scripts/add_tbo_column.py` —
  re-runs the colocated arm recording the compute/communication split, then adds
  a column comparing AFD against a colocated arm whose step time is
  `max(compute, comm)` instead of `compute + comm`: the ceiling of two-batch
  overlap. The AFD arm is left alone — its A/F pipeline already overlaps.

  Two things make this correct rather than decorative:

  - **Reach inside `OverlapOp`.** GQA models (Qwen3, MiniMax-M2.5) put
    `moe_pre_dispatch` / `moe_post_dispatch` at the top level. MLA/MSA models
    (DeepSeek-V3, Kimi-K2.5, MiniMax-M3) wrap the routed path in
    `generation_moe_overlap`, which computes `max(routed, shared)` — that is
    *shared-expert* two-stream overlap, not TBO, and the dispatch→GEMM→combine
    chain inside the routed group is still serial. Reading only the top level
    reports these models as having zero communication.
  - **Re-optimise the colocated configuration under the ceiling**, do not rescale
    the old winner. Hiding the collectives rewards wide-EP layouts that serial
    billing penalised; 57% of cells changed winner in one run. Rescaling
    understates the baseline and flatters AFD.

  The TBO sweep must be aimed at **exactly** the slice the workbook covers. It
  joins on `(model, precision, isl, osl, mtp_nextn)` and a slice it did not run
  yields an empty column, not an error — check the reported match count, and
  confirm the OSL grid in particular, which has differed between a book and a
  derived run.

  Report the ceiling with its caveat: it grants perfect overlap, free
  micro-batch splitting and unchanged kernel efficiency at half the batch. It is
  most generous exactly where least attainable — at long context and small
  batch the all-to-all is small-payload latency-bound, so splitting into two
  micro-batches nearly doubles its total latency. More useful than the ceiling
  alone: sweep the hidden fraction `f` from 0 to 1 with
  `step = compute + (1-f)*comm` and report at what `f` the AFD wins disappear.

## Reading the perf data — what these fields are not

- In a colocated `raw` dict, `gemm` / `kvcache` / `fmha` / `moe` / `comm` are
  **quantisation-mode names, not latencies**. Emitting them as timing columns
  produces a plausible-looking table of nonsense.
- `per_ops_data` is `None` on the `static_gen` path (hard-coded). The per-op
  breakdown is `summary._generation_latency_dict`.
- That dict totals the **whole generation phase**, not one step. A probe that
  records a single query is in per-step units; rescale by the ratio the dict's
  own entry gives rather than assuming the step count.
- The workbook labels attention as `GQA 64:4` while a sweep typically writes
  `GQA`. Do not put `attn` in a join key — the model already determines it.

## Workbook conventions

- **`batch_limited_by` has four states, not three.** Comparing the winning batch
  only against the largest batch that fits HBM reports an SLA-bound point as
  `compute` — reading as saturated compute when the constraint was latency.
  Correct states: `capacity` (no larger batch fits), `latency` (larger batches
  fit but breach the sheet's constraint), `compute` (larger batches fit and
  comply yet do not raise throughput), `grid` (the rung list ran out).
- **Freeze panes must cover the constraint column**, not just the identity
  columns, or scrolling right loses the SLA the row is conditioned on.
- Objective wording only: state backend and hardware versions; no
  "customer" / "yours" / "ours" / "the PR version" framing.
- Green fill for `ratio > 1`.
- Normalise display labels once, at build time (e.g. `MSA (linear)` → `MSA`).

## Reading the result

The AFD gain is a **balance** effect, not a bottleneck effect:

```
ratio ≈ (t_a + t_f) / max(t_a, t_f)      maximised at t_f / t_a ≈ 1
```

Moving away from `t_f/t_a ≈ 1` in *either* direction hurts. This is why making
the MoE faster can make AFD worse, and why a per-model kernel speedup cannot be
assumed to help AFD. The step model is
`step = L × [M × max(t_a, t_f) + t_c]` with `M` micro-batches and `L` layers;
it reproduces reported step times to within 1% on >99% of samples, so a residual
above that is a modelling bug, not noise.

Decompose a win before reporting it: a capacity-driven win (weights crowding
HBM, so the baseline cannot hold batch) and a rate-driven win (`t_f/t_a ≈ 1`)
have different implications for whether the win survives a hardware change.

## Discipline

- Re-runs must be compared point-by-point against the previous run, not by
  headline. Report how many points changed and the per-arm median, since a
  change that speeds up *both* arms can still move the ratio.
- Any bound that favours one arm gets an explicit, quantified caveat in the
  workbook itself, not only in the chat that produced it.
- State what was excluded and why. A model dropped for sparse perf data is a
  finding, not an omission.
