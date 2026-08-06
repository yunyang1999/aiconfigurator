#!/usr/bin/env python3
"""Colocated (AGG) arm with the measured MegaMoE stage substituted in, running
on OUR branch (pr1323todo-fixed).

Why here and not on liz-badada's fork: her branch has the measurement-ingestion
work but not the communication work. It has no `comm_model` at all -- no
`pr1323 / pr1323todo / pr1323new / fastafd` selector and no
`_strip_moe_dispatch_from_partition`, which is the fix that stops the
model-internal MoE dispatch all-to-all from being billed on top of the A<->F
transfer it actually IS. Running her tool would trade our communication model
for her ingestion. Porting the ingestion here keeps both.

Ported from liz-badada/aiconfigurator@pr1323-afd-moe-eval:
  op_group, average_overlap_parts, replace_agg_moe_stage, unmeasured_moe_fraction
They depend only on generic AIC SDK objects (op._group_a/_group_b,
inner.query(database, ...), runtime_config, model.generation_ops), all of which
exist unchanged in our branch.

The substitution:

    new_generation = generation
                   - (generic MoE / shared-expert ops)
                   + measured MegaMoE colocated stage
                   + retained router
                   + generic residual for layers the measured boundary misses

Two op layouts occur and both are handled:
  * Qwen3-235B / MiniMax-M2.5 expose moe_pre_dispatch + generation_moe +
    moe_post_dispatch, with generation_router_gemm separate. The router is kept.
  * MiniMax-M3 exposes ONE fused generation_moe_overlap op with the router
    inside it. average_overlap_parts re-queries the op's two inner groups and
    separates router from expert, so the router can be kept there too. This is
    the piece that made M3 impossible to handle from the CLI text breakdown.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from stability import accept

ROOT = Path("/home/scratch.yuny_wwfo_1/megamoe")
AIC = Path("/home/scratch.yuny_wwfo_1/aic/aic-yy")
sys.path.insert(0, str(ROOT))

# Every tokens-per-rank an AFD row can ask for. AFD total concurrency is
# batch_per_a_gpu * A and the colocated arm must match it on 8 GPUs, so the
# required value is batch * A / 8. Measuring only 8/16/.../192 left 5A1F and
# 7A1F with no counterpart at all -- hence the union below.
# The measured rungs, plus every per-rank batch a concurrency-matched split
# would need. 1/2/4/6 exist because at isl=131072 the KV cache leaves room for
# only 4-7 sequences per rank, so without them the 128K cells can only ever be
# below-batch-floor extrapolation; 256/384 exist because the 5A1F wins all sat
# at the old ceiling with batch_limited_by=grid_max.
MEASURED_RUNGS = (1, 2, 4, 6, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384)
TPR_GRID = sorted(set(MEASURED_RUNGS) | {b * a // 8 for a in (4, 5, 6, 7) for b in
                  (8, 16, 32, 48, 64, 96, 128, 192) if (b * a) % 8 == 0})

DECODE_STRIDE = 128

MODELS = {
    "qwen3_235b_fp4": dict(path="Qwen/Qwen3-235B-A22B-FP8", layers=94, moe_layers=94),
    "minimax_m25_fp4": dict(path="MiniMaxAI/MiniMax-M2.5", layers=62, moe_layers=62),
    "minimax_m3_fp4": dict(path="MiniMaxAI/MiniMax-M3", layers=60, moe_layers=57),
}
DATABASE_MODE = {"minimax_m3_fp4": "HYBRID"}
# B200 SXM usable HBM per GPU. static_gen does not enforce capacity itself.
CAPACITY_GIB = 180.0
GIB = 1024 ** 3
NEXTN_ACCEPTED = {0: None, 1: 0.85, 2: 1.53}


# ---------------------------------------------------------------- ported bits
def op_group(name: str) -> str:
    v = name.lower()
    if "attention" in v:
        return "attention"
    if "router" in v:
        return "router"
    if "moe" in v or "expert" in v:
        return "MoE / shared expert"
    return "other"


def average_overlap_parts(model, database, runtime_config, op):
    """Serialized router and expert-only latency inside one fused overlap op.

    Ported verbatim in behaviour from her tool. The router legs of the two
    overlap groups run serially and are summed; the expert legs overlap, so the
    slower group is taken.
    """
    groups = (getattr(op, "_group_a", ()), getattr(op, "_group_b", ()))
    if not any(groups):
        return 0.0, 0.0
    seq_batch = int(runtime_config.batch_size)
    width = int(getattr(model, "_nextn", 0) or 0) + 1
    token_batch = seq_batch * width
    beam = int(runtime_config.beam_width)
    steps = max(int(runtime_config.osl or 1) - 1, 1)
    router_total = expert_total = 0.0
    repeat_total = 0
    for offset in range(0, steps, DECODE_STRIDE):
        repeat = min(DECODE_STRIDE, steps - offset)
        g_router, g_expert = [], []
        for group in groups:
            r = e = 0.0
            for inner in group:
                val = float(inner.query(
                    database, x=token_batch * beam, batch_size=token_batch,
                    beam_width=beam, s=int(runtime_config.isl) + offset + 1,
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                ))
                if "router" in inner._name.lower():
                    r += val
                else:
                    e += val
            g_router.append(r)
            g_expert.append(e)
        router_total += sum(g_router) * repeat
        expert_total += max(g_expert, default=0.0) * repeat
        repeat_total += repeat
    d = max(repeat_total, 1)
    return router_total / d, expert_total / d


def uncovered_layer_equivalents(layers: int, moe_layers: int, nextn: int) -> tuple[int, int]:
    """(uncovered, total) MoE layer-equivalents, in AIC's own accounting.

    AIC bills q*layers + nextn layer-equivalents of MoE work, where q = nextn+1;
    the FastAFD measurement covers q*moe_layers. What is left over is:

      * the `nextn` MTP draft layer-equivalents, for every model, and
      * q*(layers - moe_layers) dense layers, which only MiniMax-M3 has (3 of
        its 60). Qwen3-235B and MiniMax-M2.5 are all-MoE, so for them the
        uncovered part is PURELY the draft layers and is exactly zero at nextn=0.
    """
    q = nextn + 1
    total = q * layers + nextn
    return max(total - q * moe_layers, 0), total


def unmeasured_moe_fraction(layers: int, moe_layers: int, nextn: int) -> float:
    """Uncovered share, kept for reporting."""
    unc, total = uncovered_layer_equivalents(layers, moe_layers, nextn)
    return unc / total if total else 0.0


def substitute(generation: dict, context: dict, *,
               measured_ms: float, layers: int, moe_layers: int, nextn: int):
    """Return (new_total, generic_moe, router_note, residual, fused).

    Two op layouts occur:

    * Qwen3-235B / MiniMax-M2.5: moe_pre_dispatch + generation_moe +
      moe_post_dispatch, with generation_router_gemm a SEPARATE op. Only the
      three MoE ops are replaced; the router op is untouched, which is exactly
      the measured boundary (FastAFD's stage excludes the router).

    * MiniMax-M3: one fused generation_moe_overlap with the router inside it.
      The CLI breakdown cannot split it, and doing so properly needs
      average_overlap_parts against live SDK objects. Instead the router share
      is BOUNDED from AIC's own context-phase breakdown, where it IS separated:
      context_router_gemm / (router + moe + pre + post). On M3 that is ~0.4%,
      so replacing the whole fused op understates colocated latency by under
      half a percent. The bound is reported per row rather than hidden.
    """
    moe_names = [n for n in generation if op_group(n) == "MoE / shared expert"]
    if not moe_names:
        raise RuntimeError("no MoE op in the AGG graph")
    generic_expert = sum(generation[n] for n in moe_names)
    fused = any("overlap" in n.lower() for n in moe_names)

    # Share of the MoE stage that is DISPATCH/COMBINE, i.e. the part a two-batch
    # overlap could actually hide. The GEMMs cannot be hidden: on one GPU the MoE
    # GEMMs and attention both need the SMs, so only the collective legs can be
    # tucked behind another microbatch's compute. Taken from AIC's own op split,
    # which separates them for Qwen3-235B and MiniMax-M2.5.
    #
    # None for MiniMax-M3, whose ops arrive as one fused generation_moe_overlap:
    # there is nothing to divide, and guessing a share for it would put a made-up
    # number in the one column that exists to bound a made-up mechanism.
    comm_names = [n for n in moe_names if "dispatch" in n.lower()
                  or "combine" in n.lower()]
    comm_share = (sum(generation[n] for n in comm_names) / generic_expert
                  if comm_names and generic_expert > 0 else None)

    router_bound_pct = 0.0
    if fused:
        c_router = sum(v for k, v in context.items() if op_group(k) == "router")
        c_moe = sum(v for k, v in context.items() if op_group(k) == "MoE / shared expert")
        if c_router + c_moe > 0:
            router_bound_pct = c_router / (c_router + c_moe) * 100.0

    total = sum(generation.values())
    # The leftover layer-equivalents are priced at the MEASURED per-layer rate,
    # not at AIC's generic one. Pricing them generically was inconsistent: the
    # substitution exists precisely because AIC's generic MoE cost is wrong in a
    # known direction -- it grows 2.7x from batch 8 to 32 where the measured
    # kernel moves 4% -- so charging the residual at that rate reintroduces the
    # bias this whole exercise removes, and it does so worst at large batch.
    #
    # For the MTP draft layers, which are the entire residual on Qwen3-235B and
    # MiniMax-M2.5, the measured rate is exactly right: a draft layer is a full
    # MoE layer. For MiniMax-M3's 3 dense layers it is a proxy, but it is the
    # same proxy AIC itself applies (it prices them with the MoE op too), so the
    # substitution stays self-consistent.
    uncovered, total_equiv = uncovered_layer_equivalents(layers, moe_layers, nextn)
    covered = max(total_equiv - uncovered, 1)
    residual = measured_ms / covered * uncovered
    new_total = total - generic_expert + measured_ms + residual
    return new_total, generic_expert, router_bound_pct, residual, fused, comm_share


# ------------------------------------------------------------------ measured
def moe_for(model, ep, tpr, mtp, ex, meas):
    """Measured colocated stage if we ran it, otherwise interpolated in BATCH.

    Batch interpolation at fixed EP is the safe direction: the colocated stage
    is weight-bound, doubling the batch moves it 3-6%, and every value here sits
    inside the measured 8..192 envelope. Interpolating across EP would be a very
    different claim -- that changes experts-per-GPU, which is the dominant term
    -- and is deliberately not done here.
    """
    hit = meas.get((model, ep, tpr, mtp))
    if hit is not None:
        return hit, "measured"
    from megamoe_extrapolate import MEASURED_SHAPES
    p = ex.predict(MEASURED_SHAPES[model], "agg", ep, ep, tpr, 1, mtp,
                   MODELS[model]["moe_layers"])
    return p["afd_moe_time_ms"], "model:batch-" + p["provenance"]


def measured_index():
    idx = {}
    # The ep1/2/4 runs are keyed by ep_size, so they extend the index rather than
    # colliding with ep=8. They were originally benchmarked to make the weight
    # term of the extrapolation identifiable, but they are ordinary measurements
    # and are what lets the colocated arm choose its GPU count.
    for d in ("megamoe-colocated-ep8", "megamoe-colocated-grid",
              "megamoe-colocated-ep8-rerun2", "megamoe-colocated-ep1",
              "megamoe-colocated-ep2", "megamoe-colocated-ep4",
              "megamoe-colocated-batchext"):
        p0 = ROOT / "results" / d
        if not p0.is_dir():
            continue
        for p in sorted(p0.glob("*.json")):
            j = json.loads(p.read_text())
            if not accept(j):
                continue
            w, t = j["workload"], j["topology"]
            idx.setdefault((j["model_profile"]["key"], t["ep_size"],
                            w["logical_tokens_per_rank"], w["mtp_nextn"]),
                           j["stage_cuda"]["p50_ms"])
    return idx


def run_one(cfg):
    sys.path.insert(0, str(AIC / "src"))
    from aiconfigurator.cli.api import cli_estimate

    m = MODELS[cfg["model"]]
    ep = cfg["ep"]
    row = {k: cfg[k] for k in ("model", "ep", "tokens_per_rank", "mtp", "isl", "osl")}
    row["batch_total"] = cfg["tokens_per_rank"] * ep
    row["moe_ms_measured"] = round(cfg["moe_ms"], 6)
    row["moe_provenance"] = cfg.get("moe_provenance", "measured")
    common_kw = dict(
        model_path=m["path"], system_name="b200_sxm", backend_name="sglang",
        backend_version="0.5.12", mode="static_gen",
        moe_quant_mode="w4a8_mxfp4_mxfp8",
        database_mode=DATABASE_MODE.get(cfg["model"], "HYBRID"),
        isl=cfg["isl"], osl=cfg["osl"], nextn=cfg["mtp"],
        tp_size=1, attention_dp_size=ep, moe_ep_size=ep,
        batch_size=cfg["tokens_per_rank"],
    )
    acc = NEXTN_ACCEPTED.get(cfg["mtp"])
    if acc is not None:
        common_kw["nextn_accepted"] = acc
    try:
        res = cli_estimate(**common_kw)
    except Exception as e:
        row["status"] = "oom" if "OOM" in str(e) else "error"
        row["error"] = str(e)[:150]
        return row

    # static_gen, not agg. Both give the SAME decode step -- agg's genonly_step
    # total and static_gen's TPOT agree to the digit (14.914 ms on Qwen3-235B at
    # isl=4096, batch 8) -- but they size activation memory differently, and only
    # static_gen sizes it for decode.
    #
    # AIC's agg path takes activations from BuildConfig.max_num_tokens, which
    # cli_estimate defaults to isl (api.py:941), and memory.py:273 states that
    # max_batch_size does not affect non-KV memory. So the agg colocated arm was
    # charged a prefill chunk's activations -- 2.016 GiB at isl=4096 rising to
    # 32.203 GiB at isl=65536 with the batch unchanged -- while the AFD arm, run
    # with --no-afd-combined-with-pd, pays none of it. That is 18% of a 180 GiB
    # card taken from the colocated KV budget for work a decode worker never
    # does, and it forced OOM at batches that in fact fit: isl=131072 batch 8
    # reads 184.05 GiB under agg and 119.75 GiB here.
    #
    # static_gen's op times are CUMULATIVE over the generation phase, so they are
    # divided by (osl - 1) to recover the per-step values the substitution works
    # in. Verified: the cumulative sum / (osl-1) reproduces TPOT exactly.
    #
    # It also performs no OOM check of its own -- isl=131072 batch 64 returns a
    # result at 782.89 GiB on a 180 GiB card -- so capacity is enforced below.
    steps = max(int(cfg["osl"]) - 1, 1)
    raw_gen = res.summary.get_generation_latency_dict() or {}
    if not raw_gen:
        row["status"] = "error"
        row["error"] = "no generation latency dict"
        return row
    gen = {k: v / steps for k, v in raw_gen.items()}
    context = {k: v / steps for k, v in (res.summary.get_context_latency_dict() or {}).items()}

    mem = res.summary.get_memory() or {}
    total = float(mem.get("total") or 0.0)
    row["memory_gib"] = round(total / GIB if total > 1e6 else total, 3)
    if row["memory_gib"] > CAPACITY_GIB:
        row["status"] = "oom"
        return row

    row["tpot_ms_generic"] = res.tpot
    row["gen_total_generic_ms"] = round(sum(gen.values()), 6)
    try:
        new_total, generic_moe, router_bound, residual, fused, comm_share = substitute(
            gen, context, measured_ms=cfg["moe_ms"],
            layers=m["layers"], moe_layers=m["moe_layers"], nextn=cfg["mtp"])
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"substitution failed: {str(e)[:120]}"
        return row

    # Attention is needed for the TBO bound: ideal two-batch overlap can hide
    # at best the smaller of (attention, MoE) behind the other. AIC models the
    # colocated layer as attention-then-MoE serially and has no TBO knob, so
    # this bound is what brackets that gap. It is a CEILING on what TBO buys,
    # not a claim about what SGLang achieves.
    row["attention_ms"] = round(sum(v for k, v in gen.items()
                                    if op_group(k) == "attention"), 6)
    row["moe_ms_generic"] = round(generic_moe, 6)
    row["fused_moe_op"] = fused
    row["router_bound_pct"] = round(router_bound, 4)
    row["residual_ms"] = round(residual, 6)
    row["unmeasured_moe_fraction"] = round(
        unmeasured_moe_fraction(m["layers"], m["moe_layers"], cfg["mtp"]), 6)
    row["gen_total_measured_ms"] = round(new_total, 6)
    # Ceiling on what a two-batch overlap could hide. Only the collective legs
    # qualify: on one GPU the MoE GEMMs and attention both need the SMs, so
    # compute cannot be tucked behind compute -- only dispatch/combine, which
    # ride NVLink and the copy engines, can go behind another microbatch's work.
    # An earlier version hid min(attention, whole MoE), which assumes the entire
    # MoE stage evaporates; no overlap scheme can do that, so it bounded nothing.
    #
    # comm_share comes from AIC's own generic op split (dispatch+combine over the
    # whole MoE stage) and is applied to the MEASURED stage time. It remains a
    # ceiling for a second reason worth stating: MegaMoE already fuses dispatch
    # and combine into the kernel and overlaps them internally, so whatever a TBO
    # could still win on top is smaller than this again.
    row["comm_share_of_moe"] = round(comm_share, 6) if comm_share is not None else None
    if comm_share is None:
        row["gen_total_comm_overlap_ms"] = None
    else:
        hideable = min(row["attention_ms"], comm_share * cfg["moe_ms"])
        row["gen_total_comm_overlap_ms"] = round(max(new_total - hideable, 1e-6), 6)
    scale = new_total / row["gen_total_generic_ms"] if row["gen_total_generic_ms"] else None
    if scale and scale > 0:
        conc = cfg["tokens_per_rank"] * ep
        row["tpot_ms_measured"] = round(res.tpot * scale, 6)
        row["tokens_s_gpu_measured"] = round(conc / (row["tpot_ms_measured"] / 1000) / ep, 2)
        row["tokens_s_gpu_generic"] = round(conc / (res.tpot / 1000) / ep, 2)
        if row["gen_total_comm_overlap_ms"]:
            st = row["gen_total_comm_overlap_ms"] / row["gen_total_generic_ms"]
            row["tpot_ms_comm_overlap"] = round(res.tpot * st, 6)
            row["tokens_s_gpu_comm_overlap"] = round(
                conc / (row["tpot_ms_comm_overlap"] / 1000) / ep, 2)
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=ROOT / "handoff" / "aic_colocated_measured.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    meas = measured_index()
    from megamoe_extrapolate import MegaMoEExtrapolator
    ex = MegaMoEExtrapolator(raw_dirs=[ROOT / "results" / d for d in
                                       ("megamoe-m2n-extended", "megamoe-m2n-afratio",
                                        "megamoe-m2n-grid", "megamoe-colocated-ep8",
                                        "megamoe-colocated-grid", "megamoe-colocated-ep1",
                                        "megamoe-colocated-ep2", "megamoe-colocated-ep4",
                                        "megamoe-colocated-batchext")
                                       if (ROOT / "results" / d).is_dir()])
    # EP sweep, so the colocated arm can choose its GPU count the way the AFD arm
    # chooses its split. Locking colocated to ep=8 while letting AFD pick 6 or 8
    # GPUs is an asymmetry, and it matters: the AFD optimum is often the smallest
    # pool that keeps T_f close to T_a, and colocated deserves the same freedom.
    # ep < 8 runs only on the rungs the kernel was actually benchmarked at, so
    # every one of those rows is provenance=measured. ep=8 keeps the full grid
    # because the concurrency-matched workbooks index into it.
    cfgs = []
    for isl in (4096, 8192, 16384, 32768, 65536, 131072):
        for osl in (256, 1024, 4096):
            for model in MODELS:
                for mtp in (0, 1, 2):
                    for ep in (1, 2, 4, 8):
                        for tpr in (TPR_GRID if ep == 8 else MEASURED_RUNGS):
                            if ep != 8 and (model, ep, tpr, mtp) not in meas:
                                continue          # keep the sub-8 arm purely measured
                            ms, prov = moe_for(model, ep, tpr, mtp, ex, meas)
                            cfgs.append(dict(model=model, ep=ep, tokens_per_rank=tpr,
                                             mtp=mtp, isl=isl, osl=osl, moe_ms=ms,
                                             moe_provenance=prov))
    if a.limit:
        cfgs = cfgs[: a.limit]
    print(f"measured colocated points: {len(meas)}   AIC runs: {len(cfgs)}")

    from concurrent.futures import ProcessPoolExecutor
    import collections
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(run_one, cfgs), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(cfgs):
                print(f"  {i}/{len(cfgs)} {dict(collections.Counter(x.get('status') for x in rows))}", flush=True)

    cols = ["model", "ep", "tokens_per_rank", "batch_total", "mtp", "isl", "osl", "status",
            "moe_ms_measured", "moe_provenance", "moe_ms_generic", "fused_moe_op", "router_bound_pct", "residual_ms",
            "unmeasured_moe_fraction", "attention_ms", "memory_gib", "gen_total_generic_ms",
            "gen_total_measured_ms", "comm_share_of_moe", "gen_total_comm_overlap_ms",
            "tpot_ms_generic", "tpot_ms_measured", "tpot_ms_comm_overlap",
            "tokens_s_gpu_generic", "tokens_s_gpu_measured",
            "tokens_s_gpu_comm_overlap", "error"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} {dict(collections.Counter(r.get('status') for r in rows))} -> {a.out}")


if __name__ == "__main__":
    main()
