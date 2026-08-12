#!/usr/bin/env python3
"""GB200 NVL72 sweep with the measured MegaMoE stage injected into both arms.

F side measured, A side AIC. That split is exact in AIC: the AFD path sums
`a_partition.attn_ops` for the A side and replaces `f_partition.ffn_ops` wholesale
with `--afd-moe-time-ms`, setting T_a2f to zero. The benchmark's stage boundary --
A-side FP8 quant and dispatch through F-side expert compute and combine -- is
exactly the span AIC hands over, so nothing is double-counted and nothing is
dropped. The colocated arm gets the same measured stage through the op-level
substitution the B200 study used, because comparing a measured kernel against
AIC's generic MoE model would not be a comparison of deployments.

The grid is chosen so the injected value is always a real measurement:

    AFD          F in {1, 2, 4}   -> shards 128/64/32 (Qwen3, M3), 256/128/64 (M2.5)
                 batch on the measured seq rungs
                 A and a_tp free; the A pool enters only through g(A/a_tp)
    colocated    EP in {1, 2, 4, 8}, all measured and all dividing 72 exactly
                 tokens per expert rank = batch / tp, on the measured rungs

F >= 8 is deliberately absent: it needs shard 16, which only the colocated kernel
has run, and the two kernels differ by a median 4.2% and up to 43% at matched
(shard, tokens per rank). That is too loose to stand in for a measurement, so
that region is left empty rather than modelled.

Both arms sweep attention tensor parallelism over {1, 2, 4}. Sharding attention
is what buys latency headroom under a tight limit, so giving it to one arm only
would decide the comparison by construction. The cap is 4 on both sides because
AIC requires tp to divide gpus_per_node and expresses AFD pools in node units, so
a_tp = 8 would force F to be a multiple of 8 -- outside the measured shards.

Every AFD row is computed twice: once with the fitted g(ag) and once with g == 1,
which is the most AFD-favourable choice available. Any conclusion that holds at
both ends does not rest on the one extrapolated term.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/scratch.yuny_wwfo_1/megamoe")
AIC = Path("/home/scratch.yuny_wwfo_1/aic/aic-yy")
sys.path.insert(0, str(ROOT))

SYSPATHS = f"default,{ROOT}/aic_systems"
DOMAIN = 72
CAPACITY_GIB = 198674743296 / 1024 ** 3     # gb200.yaml mem_capacity, 185.02 GiB
GIB = 1024 ** 3

MODEL_PATH = {"qwen3_235b_fp4": "Qwen/Qwen3-235B-A22B-FP8",
              "minimax_m25_fp4": "MiniMaxAI/MiniMax-M2.5",
              "minimax_m3_fp4": "MiniMaxAI/MiniMax-M3"}
DB_MODE = {"minimax_m3_fp4": "HYBRID"}
NEXTN_ACCEPTED = {0: None, 1: 0.85, 2: 1.53}

F_VALUES = (1, 2, 4)
A_LADDER = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32,
            36, 40, 44, 48, 52, 56, 60, 64, 66, 68, 70, 71)
TP = (1, 2, 4)
COLO_EP = (1, 2, 4, 8)
# v0514's grid. nextN 1 and 2 carry full measured coverage -- same shards, same
# 14 batch rungs, same colocated EPs as nextN 0 -- so they cost nothing but AIC
# time. isl and osl are pure AIC dimensions: the injected MoE stage does not
# depend on either.
ISLS = (4096, 8192, 16384, 32768, 65536, 131072)
MTPS = (0, 1, 2)
# osl is swept only at nextN=0, which is how v0514's ISL_x_OSL sheet is built.
OSLS_BY_MTP = {0: (256, 1024, 4096), 1: (1024,), 2: (1024,)}
# Microbatching is a real lever, not a detail. With mb=1 AIC falls back to serial
# (one in-flight microbatch cannot overlap anything), so the AFD step is
# t_a + t_f and the F pool idles while the A pool works. mb=2 buys the overlap --
# the step becomes max(t_a + t_a2f, t_f + t_f2a) -- at the cost of a MEASURED
# 1.79x on the stage itself, because each lane re-streams the expert weights and
# MegaMoE is weight-bound. Which way that trade lands depends on the balance:
# where attention dominates (long context, wide A pool) mb=2 wins outright, and
# on Qwen3 at isl=131072 it takes the AFD arm from parity to 1.14x. Both counts
# are measured, so both are swept and each cell keeps whichever is better.
MBS = (1, 2)

_T = None
_EST = None


def table():
    global _T
    if _T is None:
        from megamoe_table import MeasuredStage
        _T = MeasuredStage(ROOT)
    return _T


def _est():
    """cli_estimate with the systems path installed ONCE per worker.

    Passing systems_paths= per call costs 6 s: cli_estimate calls
    set_systems_paths twice (install, then restore in a finally) and each call
    runs clear_all_op_caches(), evicting the GEMM perf table. Setting the
    process-global path once leaves the cache intact -- 0.016 s per call instead
    of 6.3 s. Verified to give identical numbers either way.
    """
    global _EST
    if _EST is None:
        sys.path.insert(0, str(AIC / "src"))
        from aiconfigurator.cli.api import cli_estimate
        from aiconfigurator_core.sdk.perf_database import set_systems_paths
        set_systems_paths(SYSPATHS)
        _EST = cli_estimate
    return _EST


def domain_units(unit_gpus):
    n = DOMAIN // unit_gpus
    return n, DOMAIN - n * unit_gpus


def _afd_once(model, a, fq, tp, g, batch, isl, osl, mtp, moe_ms, mb=1):
    kw = dict(model_path=MODEL_PATH[model], system_name=f"gb200_g{g}",
              backend_name="sglang", backend_version="0.5.12", mode="afd",
              moe_quant_mode="w4a8_mxfp4_mxfp8",
              database_mode=DB_MODE.get(model, "HYBRID"),
              isl=isl, osl=osl, nextn=mtp, afd_combined_with_pd=False,
              n_a_nodes=a // g, n_f_nodes=fq // g, a_tp_size=tp,
              a_batch_size=batch, num_microbatches=mb,
              f_moe_ep_size=fq, pipeline_model="conservative",
              afd_moe_time_ms=moe_ms)
    acc = NEXTN_ACCEPTED.get(mtp)
    if acc is not None:
        kw["nextn_accepted"] = acc
    return _est()(**kw)


def run_afd(cfg):
    T = table()
    model, a, fq, tp, g = cfg["model"], cfg["a"], cfg["f"], cfg["a_tp"], cfg["gran"]
    mb = cfg["mb"]
    row = {k: cfg[k] for k in ("model", "a", "f", "batch", "mtp", "isl", "osl", "a_tp", "gran", "mb")}
    row.update(arm="afd", unit_gpus=a + fq,
               split=f"{a}A{fq}F" + (f"/tp_a{tp}" if tp > 1 else "")
                     + (f"/mb{mb}" if mb > 1 else ""))
    ms, prov = T.afd(model, fq, cfg["batch"], a, tp, cfg["mtp"], mb)
    row["moe_provenance"] = prov
    if ms is None:
        row["status"] = "illegal"
        return row
    ms1 = T.afd_ungained(model, fq, cfg["batch"], cfg["mtp"], mb)
    row["moe_ms"] = round(ms, 6)
    row["moe_ms_g1"] = round(ms1, 6)
    row["ag_effective"] = a / tp
    row["base_spread_pct"] = round(
        T.base_spread.get((model, cfg["mtp"], mb, T.experts[model] // fq, cfg["batch"]), 0.0), 2)
    try:
        r = _afd_once(model, a, fq, tp, g, cfg["batch"], cfg["isl"], cfg["osl"], cfg["mtp"], ms, mb)
    except Exception as e:
        row["status"] = "oom" if "OOM" in str(e) else "error"
        row["error"] = str(e)[:120]
        return row
    n, idle = domain_units(a + fq)
    if r.num_total_gpus != a + fq:
        row["status"] = "error"
        row["error"] = f"unit size mismatch: {a+fq} vs AIC {r.num_total_gpus}"
        return row
    row.update(status="ok", tpot_ms=r.tpot, concurrency=r.concurrency,
               tokens_s_gpu=r.tokens_per_second_per_gpu, memory_gb=r.memory,
               T_a_layer=r.raw.get("decode_t_a_layer"),
               T_f_layer=r.raw.get("decode_t_f_layer"),
               balance=r.raw.get("decode_balance_ratio"),
               units_in_72=n, gpus_idle=idle,
               tpsgpu_domain72=round(r.tokens_per_second_per_gpu * (a + fq) * n / DOMAIN, 2))
    # Sensitivity end-point: the same deployment with the extrapolated term off.
    try:
        r1 = _afd_once(model, a, fq, tp, g, cfg["batch"], cfg["isl"], cfg["osl"], cfg["mtp"], ms1, mb)
        row["tpot_ms_g1"] = r1.tpot
        row["tokens_s_gpu_g1"] = r1.tokens_per_second_per_gpu
        row["tpsgpu_domain72_g1"] = round(
            r1.tokens_per_second_per_gpu * (a + fq) * n / DOMAIN, 2)
    except Exception:
        pass
    return row


def run_colo(cfg):
    """Colocated arm with the same measured stage substituted into the op graph.

    static_gen rather than agg: both give the same decode step, but agg sizes
    activations from a prefill chunk (max_num_tokens defaults to isl) and this
    study compares two DECODE workers. static_gen runs no OOM check of its own,
    so capacity is enforced here against GB200's 185.0 GiB.
    """
    from colocated_measured import (MODELS, op_group, substitute,
                                    unmeasured_moe_fraction)
    T = table()
    model, ep, tp = cfg["model"], cfg["ep"], cfg["tp"]
    dp = ep // tp
    m = MODELS[model]
    n, idle = domain_units(ep)
    row = {k: cfg[k] for k in ("model", "batch", "mtp", "isl", "osl")}
    row.update(arm="colo", split=f"ep{ep}" + (f"/tp{tp}" if tp > 1 else ""),
               unit_gpus=ep, a=ep, f=ep, a_tp=tp, gran=1,
               units_in_72=n, gpus_idle=idle)
    # Tokens each expert rank contributes: total in-flight (batch * dp) spread
    # over the ep expert shards. At tp=1 that is just the batch, which is how
    # the colocated benchmark was parameterised.
    if cfg["batch"] % tp:
        row["status"] = "illegal"
        row["moe_provenance"] = "batch not divisible by tp"
        return row
    tpr = cfg["batch"] // tp
    ms, prov = T.colo(model, ep, tpr, cfg["mtp"])
    row["moe_provenance"] = prov
    row["tokens_per_expert_rank"] = tpr
    if ms is None:
        row["status"] = "illegal"
        return row
    row["moe_ms"] = round(ms, 6)
    kw = dict(model_path=m["path"], system_name="gb200_g4", backend_name="sglang",
              backend_version="0.5.12", mode="static_gen",
              moe_quant_mode="w4a8_mxfp4_mxfp8",
              database_mode=DB_MODE.get(model, "HYBRID"),
              isl=cfg["isl"], osl=cfg["osl"], nextn=cfg["mtp"],
              tp_size=tp, attention_dp_size=dp, moe_ep_size=ep,
              batch_size=cfg["batch"])
    acc = NEXTN_ACCEPTED.get(cfg["mtp"])
    if acc is not None:
        kw["nextn_accepted"] = acc
    try:
        r = _est()(**kw)
    except Exception as e:
        row["status"] = "oom" if "OOM" in str(e) else "error"
        row["error"] = str(e)[:120]
        return row

    steps = max(cfg["osl"] - 1, 1)
    raw_gen = r.summary.get_generation_latency_dict() or {}
    if not raw_gen:
        row["status"] = "error"
        row["error"] = "no generation latency dict"
        return row
    gen = {k: v / steps for k, v in raw_gen.items()}
    ctx = {k: v / steps for k, v in (r.summary.get_context_latency_dict() or {}).items()}
    mem = r.summary.get_memory() or {}
    tot = float(mem.get("total") or 0.0)
    row["memory_gb"] = round(tot / GIB if tot > 1e6 else tot, 3)
    if row["memory_gb"] > CAPACITY_GIB:
        row["status"] = "oom"
        return row
    try:
        new_total, generic_moe, _r, residual, fused, comm_share = substitute(
            gen, ctx, measured_ms=ms, layers=m["layers"],
            moe_layers=m["moe_layers"], nextn=cfg["mtp"])
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"substitution failed: {str(e)[:100]}"
        return row
    per_step = 1.0 + {0: 0.0, 1: 0.85, 2: 1.53}.get(cfg["mtp"], 0.0)
    tpot = new_total / per_step
    # Concurrency and GPU count from AIC, not local arithmetic: with tp > 1 the
    # in-flight count is batch * dp, and recomputing it by hand is how an earlier
    # version inflated colocated throughput by a factor of tp.
    conc, gpus = r.concurrency, r.num_total_gpus
    if gpus != ep:
        row["status"] = "error"
        row["error"] = f"unit size mismatch: {ep} vs AIC {gpus}"
        return row
    tps = conc * 1000.0 / tpot / gpus
    row.update(status="ok", tpot_ms=round(tpot, 4), concurrency=conc,
               tokens_s_gpu=round(tps, 2), moe_ms_generic=round(generic_moe, 4),
               residual_ms=round(residual, 4), fused_moe_op=fused,
               comm_share=round(comm_share, 4) if comm_share is not None else None,
               attention_ms=round(sum(v for k, v in gen.items()
                                      if op_group(k) == "attention"), 4),
               unmeasured_moe_fraction=round(
                   unmeasured_moe_fraction(m["layers"], m["moe_layers"], cfg["mtp"]), 4),
               tpsgpu_domain72=round(tps * gpus * n / DOMAIN, 2))
    return row


def dispatch(c):
    try:
        return run_afd(c) if c["kind"] == "afd" else run_colo(c)
    except Exception as e:
        return {**{k: v for k, v in c.items() if k != "kind"}, "arm": c["kind"],
                "status": "error", "error": f"{type(e).__name__}: {e}"[:140]}


def build_configs():
    T = table()
    cfgs, seen = [], set()
    for mtp in MTPS:
      env = T.envelope(mtp)
      for osl in OSLS_BY_MTP[mtp]:
       for model in sorted(MODEL_PATH):
        e = env.get(model)
        if not e:
            continue
        ne = T.experts[model]
        for isl in ISLS:
            for batch in e["seqs"]:
                for fq in F_VALUES:
                    if ne % fq or ne // fq not in e["shards"]:
                        continue
                    for g in (1, 2, 4):
                        if fq % g:
                            continue
                        for tp in TP:
                            if g % tp:
                                continue
                            for a in A_LADDER:
                                if a % g or a % tp or a + fq > DOMAIN:
                                    continue
                                # g only gates which tp is reachable; the four
                                # granularity files agree to the digit elsewhere,
                                # so one run per distinct (a, f, tp) is enough.
                                for mb in MBS:
                                    if (ne // fq, batch) not in e[f"mb{mb}"]:
                                        continue
                                    key = (model, mtp, osl, isl, batch, a, fq, tp, mb)
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    cfgs.append(dict(kind="afd", model=model, a=a, f=fq,
                                                     batch=batch, mtp=mtp, isl=isl,
                                                     osl=osl, a_tp=tp, gran=g, mb=mb))
            for batch in e["colo_batches"]:
                for ep in COLO_EP:
                    if ep not in e["eps"]:
                        continue
                    for tp in TP:
                        if ep % tp or batch % tp:
                            continue
                        cfgs.append(dict(kind="colo", model=model, ep=ep, tp=tp,
                                         batch=batch, mtp=mtp, isl=isl, osl=osl))
    # Group by (system, model): the GEMM table caches per system and the op graph
    # per model, so a worker that jumps between them pays 2.0 s and 0.46 s instead
    # of 0.016 s. ex.map hands out contiguous slices.
    cfgs.sort(key=lambda c: (c["kind"], c.get("gran", 0), c["model"], c["mtp"], c["osl"], c["isl"]))
    return cfgs


COLS = ["model", "arm", "split", "a", "f", "a_tp", "gran", "unit_gpus", "units_in_72",
        "gpus_idle", "batch", "mb", "mtp", "isl", "osl", "status", "moe_ms", "moe_ms_g1",
        "moe_provenance", "ag_effective", "base_spread_pct", "tokens_per_expert_rank",
        "T_a_layer", "T_f_layer", "balance", "tpot_ms", "concurrency", "tokens_s_gpu",
        "tpsgpu_domain72", "tpot_ms_g1", "tokens_s_gpu_g1", "tpsgpu_domain72_g1",
        "memory_gb", "attention_ms", "moe_ms_generic", "residual_ms", "comm_share",
        "fused_moe_op", "unmeasured_moe_fraction", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", type=Path, default=ROOT / "handoff" / "gb200_measured.csv")
    args = ap.parse_args()

    cfgs = build_configs()
    n_afd = sum(1 for c in cfgs if c["kind"] == "afd")
    print(f"configs: {len(cfgs)}  (afd {n_afd}, colo {len(cfgs) - n_afd})", flush=True)

    import collections
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(dispatch, cfgs, chunksize=64), 1):
            rows.append(row)
            if i % 5000 == 0 or i == len(cfgs):
                print(f"  {i}/{len(cfgs)} "
                      f"{dict(collections.Counter(r.get('status') for r in rows))}", flush=True)

    # Atomic write, then verify. An earlier run had two processes writing the same
    # path and a workbook was built from the loser's output.
    tmp = args.out.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, args.out)
    with args.out.open() as fh:
        written = sum(1 for _ in fh) - 1
    assert written == len(rows), f"wrote {written} rows, expected {len(rows)}"
    print(f"wrote {len(rows)} -> {args.out}  (verified)")


if __name__ == "__main__":
    main()
