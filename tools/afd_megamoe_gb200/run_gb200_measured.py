#!/usr/bin/env python3
"""GB200 NVL72 sweep with the REAL lyris-measured MegaMoE stage injected into
both arms. Not a B200 + g(ag) projection -- see megamoe_table.py's REAL_SPLITS
for why this data needs no extrapolation for the configs it covers.

F side measured, A side AIC. That split is exact in AIC: the AFD path sums
`a_partition.attn_ops` for the A side and replaces `f_partition.ffn_ops` wholesale
with `--afd-moe-time-ms`, setting T_a2f to zero. The benchmark's stage boundary --
A-side FP8 quant and dispatch through F-side expert compute and combine -- is
exactly the span AIC hands over, so nothing is double-counted and nothing is
dropped. The colocated arm gets the same measured stage through the op-level
substitution the B200 study used, because comparing a measured kernel against
AIC's generic MoE model would not be a comparison of deployments.

The grid is chosen so the injected value is always a real measurement:

    AFD          (A, F) PAIRED as one deployment split per F, exactly as
                 measured on lyris: 68A4F/64A8F/56A16F/40A32F for four of the
                 five models, and 64A8F/60A12F/56A16F/48A24F for
                 deepseek_v4_pro (68A4F does not fit its HBM budget).
                 a_tp over {1, 2, 4}, but ONLY where the resulting dispatcher
                 count ag = A/tp was itself measured -- see REAL_AGS in
                 megamoe_table.py. There is no g(ag) scaling any more: ag is
                 part of the lookup key, and an unmeasured ag is dropped rather
                 than priced from a neighbouring one. batch on the measured
                 seq rungs.
    colocated    EP at whatever lyris actually measured per model (1/2/4/8 for
                 qwen3_235b/minimax_m25; 2/4/8 for minimax_m3/deepseek_v4_flash
                 and deepseek_v4_pro's own 4/8/24 -- EP=1 crashes FastAFD's
                 MegaMoE kernel for those three models, a real kernel
                 limitation, not a gap in this sweep)
                 tokens per expert rank = batch / tp, on the measured rungs

F now reaches 16 and 32 (24 for deepseek_v4_pro) because the split kernel was
measured directly at those shards on lyris -- the old B200-projection tool
capped F at {1, 2, 4} because only the colocated kernel had ever run wider
shards there, which was too loose to stand in for a measurement.

Both arms sweep attention tensor parallelism over {1, 2, 4}. Sharding attention
is what buys latency headroom under a tight limit, so giving it to one arm only
would decide the comparison by construction.

Every AFD row's `moe_ms` and `moe_ms_g1` are identical: with ag in the key
there is no extrapolated term left to switch off. The two columns are kept so
downstream tooling built for the B200 projection's sensitivity check still runs.

68A4F has NO tp>1 support, and that is a measurement gap rather than a choice:
FastAFD's launcher requires nnodes*4 == world_size, so ag=34 (38 GPUs) and
ag=17 (21 GPUs) at F=4 cannot be benchmarked at all. Those rows come back
`illegal`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from afd_paths import data_root

# Data root: the unpacked data bundle, from AFD_MEGAMOE_ROOT (required).
ROOT = data_root()
# Repository root, derived rather than hard-coded so a fresh checkout runs
# from wherever it was cloned. AFD_AIC_ROOT points the sweep at another AIC
# checkout instead -- Kimi-K3 runs on upstream main, which carries the K3 model
# and its GB200 data.
AIC = Path(os.environ.get("AFD_AIC_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT))

TOOLDIR = Path(__file__).resolve().parent
# gb200_g{1,2,4,8}.yaml live next to this script, not in a ROOT/aic_systems dir
# -- that path is a leftover from the old B200-projection tool and does not
# exist on lyris.
# Overlay first. It mirrors the shipped systems tree with symlinks and replaces
# exactly one file: the DeepSeek-V4 MoE table, with measured EP=24 rows added.
# Every moe_ep_size AIC ships is a power of two, so EP=24 could not be priced
# for any model and DSV4-Pro's colocated arm rested on EP=8 alone.
# The systems ROOT, not its data/ subdirectory: "default" resolves to
# .../aiconfigurator_core/systems and the loader appends data/ itself. Pointing
# one level too deep resolved cleanly and silently found nothing.
#
# The overlay mirrors THIS branch's systems tree, so a run on another checkout
# (AFD_AIC_ROOT) replaces the whole path with AFD_AIC_SYSTEMS -- for upstream,
# "default,$AFD_MEGAMOE_ROOT/aic_systems_upstream" from make_upstream_systems.sh.
# AFD_COLO_SYSTEM names the colocated arm's system there (gb200_nvl72; see that
# script for why).
OVERLAY = str(ROOT / "aic_overlay")
SYSPATHS = os.environ.get("AFD_AIC_SYSTEMS") or f"{OVERLAY},default,{TOOLDIR}"
COLO_SYSTEM = os.environ.get("AFD_COLO_SYSTEM", "gb200_g4")
DOMAIN = 72
CAPACITY_GIB = 198674743296 / 1024 ** 3     # gb200.yaml mem_capacity, 185.02 GiB
GIB = 1024 ** 3

MODEL_PATH = {"qwen3_235b_fp4": "Qwen/Qwen3-235B-A22B-FP8",
              "minimax_m25_fp4": "MiniMaxAI/MiniMax-M2.5",
              "minimax_m3_fp4": "MiniMaxAI/MiniMax-M3",
              "deepseek_v4_flash_fp4": "deepseek-ai/DeepSeek-V4-Flash",
              "deepseek_v4_pro_fp4": "deepseek-ai/DeepSeek-V4-Pro",
              "kimi_k3_fp4": "moonshotai/Kimi-K3"}
DB_MODE = {"minimax_m3_fp4": "HYBRID"}
# K3 runs on upstream main, whose queryable sglang versions are 0.5.14 (current)
# and 0.5.16 (next). Its KDA kernels exist only at 0.5.16; MoE, MLA and GEMM
# backward-fill from 0.5.14. AFD_BACKEND_VERSION sets every other model's
# version for such a run (0.5.16 puts a Qwen3 anchor on the same data as K3).
BACKEND_VERSION = {"kimi_k3_fp4": "0.5.16"}
DEFAULT_BACKEND_VERSION = os.environ.get("AFD_BACKEND_VERSION", "0.5.12")
NEXTN_ACCEPTED = {0: None, 1: 0.85, 2: 1.53}

# A is not swept independently: real lyris data measured one (A, F) split per F.
# Attention TP then divides it -- tp=k means k A-GPUs dispatch once, so the
# physical condition is ag = A/k -- and REAL_AGS in megamoe_table.py is the
# single source of truth for which of those are measured. F reaches 16 and 32
# (24 for deepseek_v4_pro), which the old B200-projection F_VALUES=(1,2,4) cap
# excluded because only the colocated kernel had ever run those shards there.
TP = (1, 2, 4)
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
# on Qwen3 at isl=131072 it takes the AFD arm from parity to 1.14x. lyris also
# measured mb=4 (further overlap, further weight-restream cost); all three
# counts are measured, so all three are swept and each cell keeps whichever
# is best.
MBS = (1, 2, 3, 4)   # m = 3 joins where its stage was measured (rungs divisible by 3)

_T = None
_EST = None


# Recorded routing distributions, one per model, keyed as the sweep keys models.
# Only what a real router on real prompts actually produced goes in here; a model
# with no recording is simply absent, and its stage times stay balanced-only and
# are marked as such in the output rather than corrected from a neighbour's shape.
# The coefficient is proportional to the skew, so borrowing another model's
# distribution would move the answer by the full difference in skew.
#
# PER LAYER. The distributions below keep counts_per_layer, and the imbalance
# is averaged over layers rather than computed on the layer-sum. A decode step
# waits for each layer's own busiest rank, and that rank moves nearly every
# layer, so summing the layers first cancels the skew away: on the recorded
# Qwen3 routing the layer-summed max/mean is 1.31 at 32 ranks where the
# per-layer figure is 2.90. Set AFD_ROUTING_AGGREGATED=1 to reproduce the
# earlier layer-summed numbers for comparison.
AGGREGATED = os.environ.get("AFD_ROUTING_AGGREGATED") == "1"

ROUTING_DIST = {
    # Recorded from a real router on real prompts, 94 layers kept separate.
    "qwen3_235b_fp4": ROOT / "real_routing_qwen3_235b_perlayer.json",
    # Synthetic: 94 independent layers, each drawn lognormal at the per-layer CV
    # Qwen3's real router produced (136%). These four models' own routing has
    # not been recorded -- minisgl implements none of their architectures except
    # MiniMax-M2's, and that capture has not yet completed. Without an entry
    # here the correction bails before it reaches the MEASURED coefficient, so
    # their rows would carry the uniform-routing time. Stating the assumed skew
    # is more informative than that.
    "minimax_m25_fp4": ROOT / "synthetic_perlayer_256.json",
    "minimax_m3_fp4": ROOT / "synthetic_perlayer_128.json",
    # Recorded: DeepSeek-V4-Flash real weights, 1152 real prompts, 43 layers kept
    # separate (hash-routed layers 0-2 included as routed).
    "deepseek_v4_flash_fp4": ROOT / "real_routing_dsv4_flash_perlayer.json",
    "deepseek_v4_pro_fp4": ROOT / "synthetic_perlayer_384.json",
    # 92 LatentMoE layers x 896 experts, same generator (build_k3_routing.py).
    "kimi_k3_fp4": ROOT / "synthetic_perlayer_896.json",
}
ROUTING_DIST_AGGREGATED = {
    "qwen3_235b_fp4": ROOT / "real_routing_qwen3_235b.json",
    "minimax_m25_fp4": ROOT / "synthetic_cvmeasured_256.json",
    "minimax_m3_fp4": ROOT / "synthetic_cvmeasured_128.json",
    "deepseek_v4_flash_fp4": ROOT / "synthetic_cvmeasured_256.json",
    "deepseek_v4_pro_fp4": ROOT / "synthetic_cvmeasured_384.json",
}
# Which models' skew is recorded rather than assumed; the workbook says so.
ROUTING_IS_REAL = {"qwen3_235b_fp4", "deepseek_v4_flash_fp4"}


def _routing_distributions():
    src = ROUTING_DIST_AGGREGATED if AGGREGATED else ROUTING_DIST
    key = "counts_total" if AGGREGATED else "counts_per_layer"
    out = {}
    for model, path in src.items():
        try:
            out[model] = json.loads(Path(path).read_text())[key]
        except (OSError, KeyError, ValueError):
            continue
    return out


def table():
    global _T
    if _T is None:
        from megamoe_table import MeasuredStage
        _T = MeasuredStage(ROOT, imbalance_dist=_routing_distributions(),
                           imbalance_measured=ROOT / "measured_imbalance_coefficients.json",
                           real_skew=ROUTING_IS_REAL)
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
        _k3_overrides()
        _EST = cli_estimate
    return _EST


def _k3_overrides():
    """Kimi-K3 adjustments applied in every worker before any model is built.

    Draft attention: price Kimi-K3's DSPARK draft attention on a shape GB200
    has data for.

    The sglang draft is GQA 64q/16kv at head_dim 64, and GB200's sglang tables
    have no row at that shape (or its tp=2/4 shards) -- the only K3 op without
    data, and only under MTP. 32q/8kv at head_dim 128 keeps the q and kv widths
    (4096 / 1024), so the draft GEMMs, KV bytes per token, attention FLOPs and
    GQA ratio are all unchanged; only the kernel's head granularity differs.
    """
    try:
        from aiconfigurator_core.sdk.models.kimi_k3 import KimiK3Model
    except ImportError:          # a checkout without K3
        return
    m = KimiK3Model
    assert (m.DRAFT_NUM_HEADS, m.DRAFT_NUM_KV_HEADS, m.DRAFT_HEAD_DIM) in ((64, 16, 64), (32, 8, 128))
    m.DRAFT_NUM_HEADS, m.DRAFT_NUM_KV_HEADS, m.DRAFT_HEAD_DIM = 32, 8, 128
    # KDA state slots (sensitivity): upstream charges KDA_STATE_SLOTS_PER_REQUEST
    # (5) radix-cache state slots per request, ~2.2 GB at tp=1; a decode worker
    # without prefix caching holds one. AFD_K3_KDA_SLOTS overrides the count.
    slots = os.environ.get("AFD_K3_KDA_SLOTS")
    if slots:
        m.KDA_STATE_SLOTS_PER_REQUEST = int(slots)


def domain_units(unit_gpus):
    n = DOMAIN // unit_gpus
    return n, DOMAIN - n * unit_gpus


def _afd_once(model, a, fq, tp, g, batch, isl, osl, mtp, moe_ms, mb=1):
    kw = dict(model_path=MODEL_PATH[model], system_name=f"gb200_g{g}",
              backend_name="sglang", backend_version=BACKEND_VERSION.get(model, DEFAULT_BACKEND_VERSION), mode="afd",
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
    ms1 = T.afd_ungained(model, fq, cfg["batch"], cfg["mtp"], mb, a_gpus=a, a_tp=tp)
    row["moe_ms"] = round(ms, 6)
    row["moe_ms_g1"] = round(ms1, 6)
    row["ag_effective"] = a / tp
    row["base_spread_pct"] = round(
        T.base_spread.get((model, cfg["mtp"], mb, T.experts[model] // fq, cfg["batch"],
                           a // tp), 0.0), 2)
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
    kw = dict(model_path=m["path"], system_name=COLO_SYSTEM, backend_name="sglang",
              backend_version=BACKEND_VERSION.get(model, DEFAULT_BACKEND_VERSION), mode="static_gen",
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
    billing = {k: m[k] for k in ("aic_moe_layers", "draft_moe_layers") if k in m}
    try:
        new_total, generic_moe, _r, residual, fused, comm_share = substitute(
            gen, ctx, measured_ms=ms, layers=m["layers"],
            moe_layers=m["moe_layers"], nextn=cfg["mtp"], **billing)
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
                   unmeasured_moe_fraction(m["layers"], m["moe_layers"], cfg["mtp"], **billing), 4),
               tpsgpu_domain72=round(tps * gpus * n / DOMAIN, 2))
    return row


def dispatch(c):
    try:
        return run_afd(c) if c["kind"] == "afd" else run_colo(c)
    except Exception as e:
        return {**{k: v for k, v in c.items() if k != "kind"}, "arm": c["kind"],
                "status": "error", "error": f"{type(e).__name__}: {e}"[:140]}


def build_configs(models=None):
    T = table()
    from megamoe_table import REAL_AGS, REAL_SPLITS
    cfgs, seen = [], set()
    for mtp in MTPS:
      env = T.envelope(mtp)
      for osl in OSLS_BY_MTP[mtp]:
       for model in sorted(models or MODEL_PATH):
        e = env.get(model)
        if not e:
            continue
        ne = T.experts[model]
        for isl in ISLS:
            for batch in e["seqs"]:
                for fq, a in sorted(REAL_SPLITS.get(model, {}).items()):
                    if ne % fq or ne // fq not in e["shards"]:
                        continue
                    assert a + fq == DOMAIN, f"{model} F={fq}: split {a}+{fq} != {DOMAIN}"
                    measured_ags = set(e.get("ags", {}).get(ne // fq, ()))
                    for g in (1, 2, 4):
                        if fq % g or a % g:
                            continue
                        for tp in TP:
                            if g % tp or a % tp:
                                continue
                            # ag = A/tp must itself be a measured condition; an
                            # unmeasured ag is dropped, never priced from another
                            if a // tp not in measured_ags:
                                continue
                            # a is the real measured split for this (model, fq),
                            # fixed rather than swept -- see module docstring and
                            # REAL_SPLITS in megamoe_table.py. g only gates which
                            # tp is reachable; the four granularity files agree
                            # to the digit elsewhere, so one run per distinct
                            # (a, f, tp) is enough.
                            for mb in MBS:
                                if (ne // fq, batch, a // tp) not in e[f"mb{mb}"]:
                                    continue
                                key = (model, mtp, osl, isl, batch, a, fq, tp, mb)
                                if key in seen:
                                    continue
                                seen.add(key)
                                cfgs.append(dict(kind="afd", model=model, a=a, f=fq,
                                                 batch=batch, mtp=mtp, isl=isl,
                                                 osl=osl, a_tp=tp, gran=g, mb=mb))
            for batch in e["colo_batches"]:
                for ep in sorted(e["eps"]):
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
    ap.add_argument("--models", nargs="+", choices=sorted(MODEL_PATH),
                    help="restrict the sweep to these models (default: all)")
    args = ap.parse_args()

    cfgs = build_configs(args.models)
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
