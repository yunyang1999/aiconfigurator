> **Superseded.** This runbook describes the first version, which projected
> B200 MegaMoE measurements onto GB200. The sweep now uses MegaMoE stage times
> **measured on GB200 NVL72** and a per-layer routing-skew coefficient on both
> arms. For how it works and how to run it, see
> [`.claude/skills/aic-afd-megamoe/SKILL.md`](../../.claude/skills/aic-afd-megamoe/SKILL.md).
> The injection mechanism in section 2 below is still accurate; the data
> provenance and the "nothing here is a GB200 measurement" statement are not.

# GB200 NVL72 projection from B200 MegaMoE measurements

Projects the AFD-versus-colocated decode comparison onto a 72-GPU NVL72 domain
using MegaMoE stage times measured on B200. Sibling of `../afd_megamoe_b200`,
which it imports from (`colocated_measured`, `stability`) and which must stay on
the path.

Nothing here is a GB200 measurement. The MoE stage is B200 hardware; attention,
KV and memory are AIC's `gb200` model. No end-to-end AFD serving run on GB200 has
been made, which is why the delivered workbook is named `..._Projected_...`.

---

## 1. Why B200 kernel data transfers

MegaMoE runs entirely inside one NVLink domain, and `gb200.yaml` and
`b200_sxm.yaml` give it the same 900 GB/s. GB200 has 3.9% more memory bandwidth
and 11.1% more FP4 throughput, so reusing a B200 time for a weight-bound stage is
conservative: the real kernel would be a few percent faster.

What changes is the domain: 8 GPUs to 72. That is a deployment-shape change, not
a kernel change. A 68A4F split cannot exist on B200 and is ordinary on NVL72.

## 2. What is injected where

AIC's AFD path sums `a_partition.attn_ops` for the A side and replaces
`f_partition.ffn_ops` wholesale with `--afd-moe-time-ms`, setting `T_a2f` to
zero. The benchmark's `measurement_boundary` -- A-side FP8 quant and dispatch
through F-side expert compute and combine -- is exactly that span, so nothing is
double counted and nothing is dropped.

The colocated arm gets the same measured stage through the op-level substitution
in `../afd_megamoe_b200/colocated_measured.py`. Comparing a measured kernel
against AIC's generic MoE model would not be a comparison of deployments.

## 3. The lookup key, and the one that was wrong

    stage = base(model, nextN, microbatches, experts per F GPU, sequences per A rank)
            x g(A / a_tp)

`base` is an exact lookup into the measured grid. No interpolation: the batch
grid IS the measured grid.

The key is NOT tokens per F GPU. The measured stage includes per-A-rank work, so
it tracks sequences per A rank. MiniMax-M2.5, 32 experts per F GPU, 64 sequences
per A rank:

    ag=1  ->  16 tokens per F GPU  ->  7.89 ms
    ag=2  ->  32                   ->  7.91
    ag=3  ->  48                   ->  7.89

The independent variable moves 3x and the stage does not. An earlier version
keyed on tokens per F GPU, which overstated the F-pool time for wide A pools and
penalised AFD throughout.

`g(ag) = ag ** 0.053` (`ag ** 0.193` below 4 sequences per rank) is the only
extrapolated term: fitted on ag = 1..7, used to ag = 68 where it reaches 1.25.
Leave-one-out over 758 measured points gives median error 0.8%, p90 7.0%, no bias
by ag. `g > 1`, so it always makes AFD look worse, and every row also carries
`ratio_domain72_g1` with `g` switched off.

## 4. Files

| file | role |
|---|---|
| `megamoe_table.py` | the measured surface plus `g(ag)`; directory allowlist and protocol gate |
| `run_gb200_measured.py` | the sweep: both arms, both microbatch counts, single writer, row-count self-check |
| `capacity_wall.py` | bisects the MEMORY model for the true batch wall |
| `precompute_walls.py` | enumerates the walls each sheet needs, computes them in parallel, caches to JSON |
| `build_gb200_style.py` | the workbook, v0514 layout plus Summary / band / capacity columns |
| `shard_model.py` | bound on what a finer expert shard would buy; used to justify excluding F >= 8 |
| `gb200_g{1,2,4,8}.yaml` | one GPU-per-node override each, so every (A, F, a_tp) is expressible |
| `submit_batchfine{,2}.sh` | the two batch-refinement measurement rounds |
| `finish_gb200_d.sbatch` | walls plus workbook as a Slurm job |

## 5. Running it

    export PYTHONPATH=../afd_megamoe_b200:.
    python run_gb200_measured.py --workers 14 --out handoff/gb200_measured.csv
    python precompute_walls.py --workers 14
    python build_gb200_style.py

Or as one Slurm job, which is what to use if the session may end:

    sbatch --account=... --qos=batch-short --partition=<any cpu-ish partition> \
        finish_gb200_d.sbatch

The sweep is AIC modelling, not a benchmark, so it needs CPU only. It was run on
an `a100 1gpu-16cpu` node after probing that the venv there returns TPOT 19.507
for the same configuration as the login node, to the digit.

## 6. Traps that produced wrong numbers

**Do not run the sweep as a background process.** It is not a Slurm job, so it
dies with the session. One run died at 75000/307524 with no CSV; an attempt to
detach it with `setsid nohup` was killed by a tool timeout and left a CSV that
could not be reproduced from the code on disk. A workbook was once built from a
stale CSV because two processes were writing the same path and `until [ -f ... ]`
waited for the loser. Hence: single writer, atomic rename, and the row count
verified against the log inside the script.

**`systems_paths=` on every `cli_estimate` call costs 6 seconds.** It calls
`set_systems_paths` twice (install, then restore in a `finally`) and each call
runs `clear_all_op_caches()`, evicting the GEMM perf table. Set the process-global
path once per worker and pass `systems_paths=None`: 6.3 s becomes 0.016 s per
configuration, verified to give identical numbers.

**Microbatching is a lever, not a detail.** At mb=1 AIC falls back to serial --
one in-flight microbatch cannot overlap anything, since layer i+1's A input IS
layer i's F output -- so the step is `t_a + t_f` and the F pool idles while the A
pool works. mb=2 buys `max(t_a + t_a2f, t_f + t_f2a)` at a measured 1.79x on the
stage, because each lane re-streams the expert weights. At long context, where
attention dominates, that trade is worth taking: Qwen3 at isl=131072 goes from
parity to 1.14x. Sweeping only mb=1 hides the entire long-context result.

**Both arms need the same levers.** An early sweep gave attention tensor
parallelism to the AFD A pool and pinned the colocated arm at tp=1. That alone
produced a median 1.53 gain under a 15 ms limit; with colocated allowed the same
tp, it fell to 0.795. Sharding attention is what buys latency headroom, so giving
it to one side decides the comparison by construction.

**Throughput is a staircase in the latency limit.** The MoE stage is
weight-bound, so TPOT barely moves with batch, and a single-threshold ratio
measures which arm happened to step most recently. MiniMax-M2.5 at isl=32768
reads 1.68 at 10 ms, 3.31 at 11 ms and 1.01 at 12 ms. Hence the band: re-optimise
both arms at 0.9x..1.2x the limit and admit a gain only if the LOWER edge clears
1.10. A one-sided +10% probe is not enough -- it moves in either direction
depending on whose rung is nearer.

**The batch grid quantises capacity.** It is the measured grid, which is right
for throughput and wrong for any statement about how much batch fits. Qwen3 at
isl=65536 read a 41.7% AFD concurrency advantage where the true walls (batch 108
against 118) give 3.2%: the colocated arm was not out of memory at batch 64, it
was out of rungs at 62% of the card. `capacity_wall.py` finds the real wall from
the memory model alone, and `submit_batchfine{,2}.sh` refined the ladder around
it. Round 1 refined the wrong end -- `logical_tokens_per_rank` IS the colocated
per-GPU concurrency, so its wall scales as 1/isl and sits near 27 at isl=65536,
not near 100 -- and made the comparison worse before round 2 fixed it.

**`nextn_accepted` must accompany `nextn`.** `capacity_wall._colo_memory` passed
only the latter, AIC raised, and a bare `except` reported that as "no wall". All
184 unresolved walls were colocated and every one was nextN 1 or 2 -- which is
where AFD wins most often, so the rows that most needed the ladder check were
exactly the ones silently missing it, and `verdict` skipped the check when the
column was empty. Three rows moved from `gain` to `gain, ladder-inflated` once
fixed.

**A low CV does not redeem a short run.** The multimodel smoke suite runs
warmups=2 / iterations=10 and reported 0.190 ms for a Qwen3 point that measures
6.919 ms under the real protocol -- 36x off, with cv=3.55%, which clears a
dispersion gate on its own. `stability.accept` now also requires warmups >= 30
and iterations >= 40.

## 7. What this cannot answer

* **DeepSeek-V3 / V3.1 / Kimi-K2.5** -- never benchmarked with MegaMoE. The two
  DeepSeek-V4 files in the results tree are single smoke runs and fail the gate.
* **FP8-block / INT4-wo / NVFP4** -- only `w4a8_mxfp4_mxfp8` was measured, and it
  is NOT NVFP4. A modelled sweep can vary precision freely; an injected kernel
  cannot.
* **Expert shards finer than 16 per GPU** -- eight GPUs cannot produce them.
  `shard_model.py` bounds what they would buy and offers the bound to BOTH arms;
  no extrapolated configuration wins anywhere. The reason is arithmetic:
  72 = 2^3 x 3^2, so an EP dividing both the expert count (a power of two here)
  and the rack can only be 1, 2, 4 or 8. ep16 strands 8 GPUs and the 11% penalty
  cancels the cheaper stage.
* **MiniMax-M3's attention** -- MSA has no silicon rows, so it comes from SOL plus
  an empirical factor, and that term is 42% to 61% of the M3 decode step. M3's
  numbers are softer than the other two models'.

## 8. Kimi-K3 on upstream main

K3 (KDA linear attention + MLA, LatentMoE with 896 experts) exists only on
upstream main, so its slice runs this tooling against that checkout instead of
this branch's engine; the other models' results are untouched.

* **Engine.** An upstream-main checkout with the measured-stage AFD commits
  (`afd_moe_time_ms`, serial m=1, F-worker head check) plus two K3 fixes: ops
  outside the measured stage stay on the A pool (router, dense layer, draft
  model, and for LatentMoE the latent projections and shared experts -- the
  stage covers only routed experts in the 3584 latent space), and K3's KDA
  head check skips the F-worker so F=28 builds.
* **Systems.** `make_upstream_systems.sh` writes
  `$AFD_MEGAMOE_ROOT/aic_systems_upstream`: the granularity files, plus
  `gb200_nvl72` (the 72-GPU NVLink domain as one node) for the colocated arm,
  because upstream gates EP > 4 on DeepEP tables K3 does not have. At EP=4 the
  two files agree op for op.
* **Version.** sglang 0.5.16 for every model in the run: K3's KDA tables exist
  only there; all other families backward-fill from 0.5.14.
* **Measured stage.** `kimi_k3_fp4` in FastAFD's profiles, routed experts only
  (latent 3584, inter 3072, top-16, no shared expert in the stage). Splits
  64A8F / 56A16F / 44A28F / 40A32F with tp 2/4 fan-ins where node-aligned;
  colocated EP 8/16/32/64. Routing skew is assumed (`build_k3_routing.py`).
* **Run.** `sbatch sweep_k3_upstream.sbatch` with `AIC_PY`, `AFD_MEGAMOE_ROOT`
  and `AFD_AIC_ROOT` set. Qwen3-235B rides along as an anchor on the same engine.

Read K3 with these in mind:

* **Engine offset.** Same injected stage, same memory -- but on upstream's
  0.5.14 tables Qwen3's non-attention A-side work is cheaper than on this
  branch's 0.5.12 ones (TPOT -7 to -15% AFD, -8 to -10% colocated at isl 4096).
  Compare K3 with the Qwen3 anchor from the same run, not with this branch's CSV.
* **KDA state.** Upstream charges 5 radix-cache state slots per request
  (~2.2 GB per request at tp=1), so K3 is memory-bound in both arms well before
  the stage saturates.
* **MTP.** Upstream widens generation attention by nextN+1 (no shared-KV
  query_len path), and K3's sglang draft attention (64q/16kv, head 64) has no
  GB200 rows: it is priced as 32q/8kv at head 128 -- same widths, bytes, FLOPs
  and GQA ratio.
