#!/usr/bin/env python3
"""Pure GB200 NVL72 search, laid out like AIC_AFD_Homogeneous_GB200_v0514.

Both arms are free to use the whole 72-GPU NVLink domain and each searches its
OWN best deployment and batch. Nothing is inherited from the B200 study except
the MegaMoE stage time, which is keyed on experts-per-F-GPU and tokens-per-F-GPU
and therefore transfers to any GPU count by construction. Everything else --
attention time, KV capacity, memory, the 198.7 GB/GPU limit -- comes from AIC's
own gb200 database.

The 72-GPU budget is enforced honestly. A deployment unit of A+F GPUs can be
replicated floor(72 / (A+F)) times; the remainder is idle silicon that still has
to be paid for. So the ranking metric is

    tpsgpu_domain72 = tpsgpu * unit_gpus * floor(72 / unit_gpus) / 72

which equals tpsgpu whenever the unit divides 72 and is discounted when it does
not. A 40-GPU unit fits once and wastes 32 GPUs, and a comparison that ignored
that would credit AFD with throughput from GPUs it left idle. Both arms are
measured the same way: a colocated ep=64 group wastes 8 GPUs by the same rule.

Two conventions are copied verbatim from v0514 and were verified against it:

    ratio          = afd_tpsgpu / colo_tpsgpu
    tokps_per_user = 1000 / tpot_ms
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path("/home/scratch.yuny_wwfo_1/megamoe")
HANDOFF = ROOT / "handoff"
SRC = HANDOFF / "gb200_measured.csv"
OUT = HANDOFF / "AIC_AFD_MegaMoE_GB200_NVL72.xlsx"
MODEL_EXPERTS = {"qwen3_235b_fp4": 128, "minimax_m25_fp4": 256, "minimax_m3_fp4": 128}

DOMAIN = 72
FONT = "Arial"
HDR_FILL = PatternFill("solid", start_color="1F3864")
HDR_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
TITLE = Font(name=FONT, bold=True, size=11, color="1F3864")
BODY = Font(name=FONT, size=10)
GAIN = PatternFill("solid", start_color="C6EFCE")
LOSS = PatternFill("solid", start_color="FFC7CE")
EDGE = PatternFill("solid", start_color="FFF2CC")

MODEL_LABEL = {"qwen3_235b_fp4": "Qwen3-235B-A22B",
               "minimax_m25_fp4": "MiniMax-M2.5",
               "minimax_m3_fp4": "MiniMax-M3"}
ATTN = {"qwen3_235b_fp4": "GQA", "minimax_m25_fp4": "GQA", "minimax_m3_fp4": "MSA"}
SLA_LIMITS = [10, 15, 20, 30, 50, 100, 150]

COMMON = ["model", "attn", "moe_precision", "isl", "osl", "mtp_nextn"]
AFD_BLOCK = ["afd_split", "afd_A_gpus", "afd_F_gpus", "afd_a_tp", "afd_microbatches", "afd_unit_gpus",
             "afd_units_in_72", "afd_gpus_idle", "afd_batch", "afd_tpsgpu",
             "afd_tpsgpu_domain72", "afd_tpot_ms", "afd_tokps_per_user", "afd_conc",
             "afd_conc_per_gpu", "afd_t_attn_us", "afd_t_ffn_us", "afd_tf_over_ta",
             "afd_moe_ms_measured", "afd_moe_provenance", "afd_ag_effective",
             "afd_base_spread_pct", "afd_memory_gb", "afd_batch_limited_by"]
COLO_BLOCK = ["colo_cfg", "colo_attn_tp", "colo_unit_gpus", "colo_units_in_72", "colo_gpus_idle",
              "colo_batch", "colo_tpsgpu", "colo_tpsgpu_domain72", "colo_tpot_ms",
              "colo_tokps_per_user", "colo_conc", "colo_conc_per_gpu",
              "colo_moe_ms_measured", "colo_memory_gb", "colo_batch_limited_by"]
CONSTS = ["moe_kernel", "system", "backend"]
NOSLA_COLS = (COMMON + CONSTS + AFD_BLOCK + COLO_BLOCK
              + ["ratio", "ratio_domain72", "ratio_domain72_g1"])
# Appended after the v0514 set so the parity on the original columns still holds.
#   ratio_sla_plus10      the same comparison re-optimised with the limit 10% looser
#   sla_edge_sensitive    that moves the ratio by more than 25%
#   gain_survives_..      the row still clears 1.10 after the perturbation
# These are not decoration. Both arms' TPOT is nearly flat in batch -- the MoE
# stage is weight-bound -- so throughput is a STAIRCASE in the latency limit, and
# a single-threshold ratio measures which arm happened to step most recently. A
# one-sided +10% probe is not enough: it moves in either direction depending on
# whose rung is nearer. Two rows from this sweep, both MiniMax-M2.5:
#
#   isl=16384, 10 ms   ratio 2.92 -> 0.91 at 11 ms   (colocated steps, ratio falls)
#   isl=32768, 10 ms   ratio 1.68 -> 3.31 at 11 ms   (AFD steps first, ratio rises;
#                                                     colocated steps at 11.4 ms and
#                                                     it collapses to 1.01 at 12 ms)
#
# So the band is scanned on both sides and reported as an interval. A row is only
# a real gain when the WHOLE band clears 1.10, not when the nominal point does.
SENS = ["ratio_band_lo", "ratio_band_hi", "sla_edge_sensitive", "gain_holds_across_band"]
# The batch grid is the MEASURED grid, which is right for throughput and wrong
# for capacity: a configuration whose wall sits at batch 108 is reported at 64,
# because 64 is the largest rung below it. The distortion is not symmetric, so
# it lands on the comparison. These columns give the wall found by bisecting the
# MEMORY model, which needs no MoE time and so stays inside the measured
# envelope. Qwen3 at isl=65536, 50 ms: the sheet reads 16.00 against 22.67
# seq/GPU (+41.7%) where the walls are 27.00 against 27.86 (+3.2%).
CAP_COLS = ["colo_batch_capacity_wall", "colo_conc_per_gpu_at_wall",
            "afd_batch_capacity_wall", "afd_conc_per_gpu_at_wall",
            "conc_ratio_at_rungs", "conc_ratio_at_wall", "rung_inflation"]
# The band is scanned at these multiples of the nominal limit.
SLA_BAND = (0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
SLA_COLS = (COMMON + ["sla_tpot_ms"] + CONSTS + COLO_BLOCK + AFD_BLOCK
            + ["ratio"] + SENS + ["ratio_domain72"] + CAP_COLS)
GAIN_COLS = (COMMON + ["sla_tpot_ms", "ratio"] + SENS + CAP_COLS
             + ["ratio_domain72", "ratio_domain72_g1"] + CONSTS + COLO_BLOCK + AFD_BLOCK)
SPLIT_COLS = (COMMON + CONSTS + AFD_BLOCK
              + ["colo_cfg", "colo_unit_gpus", "colo_batch", "colo_tpsgpu",
                 "colo_tpsgpu_domain72", "colo_tpot_ms", "ratio", "ratio_domain72"])
BUDGETS = [8, 16, 32, 64, 72]
BUDGET_COLS = (COMMON + ["budget_gpus", "sla_tpot_ms"] + ["moe_kernel", "system", "backend"]
               + ["afd_split", "afd_unit_gpus", "afd_replicas", "afd_gpus_used",
                  "afd_gpus_stranded", "afd_batch", "afd_tpot_ms", "afd_tpsgpu",
                  "afd_tpsgpu_budget", "afd_moe_ms_measured", "afd_moe_provenance"]
               + ["colo_cfg", "colo_unit_gpus", "colo_replicas", "colo_gpus_used",
                  "colo_gpus_stranded", "colo_batch", "colo_tpot_ms", "colo_tpsgpu",
                  "colo_tpsgpu_budget", "colo_moe_ms_measured"]
               + ["ratio_budget"] + CAP_COLS)


CONST_VALS = dict(moe_precision="w4a8_mxfp4_mxfp8",
                  moe_kernel="fastafd_megamoe (measured on B200)",
                  system="NVIDIA GB200 NVL72 (72-GPU NVLink domain)",
                  backend="SGLang 0.5.12")

CAPTION = {
    "AFD_Perf_Gain > 10% ":
        "Rows where the disaggregated arm is more than 10% ahead on tokens/s/GPU "
        "across the full 72-GPU domain AND stays ahead across the whole latency band "
        "(0.9x to 1.2x the nominal limit). Admission is on ratio_band_lo, not on the "
        "nominal ratio: both arms' throughput is a staircase in the limit, so a gain "
        "that exists only at one threshold is an artifact of where the line falls.",
    "Model":
        "Model and weight precision at generation length 1024, draft length 0, no "
        "latency limit. Each arm at its own best deployment and batch on 72 GPUs.",
    "Context":
        "Context sweep at generation length 1024, no latency limit. Each arm at its "
        "own best deployment and batch on 72 GPUs.",
    "SLA_Latency":
        "Best configuration under each time-per-output-token limit, generation "
        "length 1024. A blank side means no configuration of that arm meets the limit.",
    "AF_Split":
        "Every A:F split at its own best batch and A-pool sharding, generation length "
        "1024, no latency limit. The colocated side repeats the single best colocated "
        "point so each split can be read against the same baseline.",
    "Model_x_Precision":
        "Model, weight precision and draft length at generation length 1024, no latency "
        "limit. Each arm at its own best deployment and batch on 72 GPUs. One precision "
        "only: the injected stage is a measurement, and w4a8_mxfp4_mxfp8 is what was "
        "benchmarked -- it is NOT NVFP4. Draft length does vary, because nextN 0, 1 and "
        "2 all carry full measured MegaMoE coverage: the same shards, the same batch "
        "rungs and the same colocated EP groups.",
    "ISL_x_OSL":
        "Input length crossed with generation length, draft length 0, no latency "
        "limit. The injected MoE stage does not depend on either, so this sheet "
        "isolates what the attention and KV side contribute.",
    "GPU_Budget":
        "The same comparison at other rack sizes. A unit of u GPUs fits floor(B/u) "
        "times in a budget of B and strands the remainder; tokens/s/GPU is charged "
        "over the WHOLE budget, so a unit that tiles badly pays for the silicon it "
        "cannot use.",
    "Shard_Extrapolation":
        "What a finer expert shard would buy, for BOTH arms, under a linear-in-active-"
        "experts model fitted on the measured shards. It is an UPPER BOUND: every "
        "leave-one-out check that fails, fails by under-predicting the fine-shard time "
        "(colocated MiniMax-M2.5 by 32%, split MiniMax-M2.5 by 24%), and pushed further "
        "the form goes openly wrong. It is here because the answer turns out not to "
        "depend on it -- see is_extrapolated on the winners, which is FALSE everywhere. "
        "The reason is arithmetic: 72 = 2^3 x 3^2, so an EP that divides both the expert "
        "count (a power of two here) and 72 can only be 1, 2, 4 or 8. ep16 fits four "
        "times in the rack and strands 8 GPUs, and that 11% penalty cancels the cheaper "
        "MoE stage.",
    "Summary":
        "The whole study in thirteen columns. `ratio` is what the comparison says; the "
        "three columns after it are what decides whether to believe it. ratio_band_lo is "
        "the ratio at the WORST point of a latency band scanned from 0.9x to 1.2x the "
        "limit -- throughput is a staircase in the limit, so a gain that exists only at "
        "one threshold is an artifact of where the line falls. rung_inflation is how much "
        "the batch ladder flatters the concurrency comparison, measured against the true "
        "memory wall. verdict combines them. Everything else lives in SLA_Latency; this "
        "sheet is the one to read first and the one to hand to someone else.",
    "Colo_EP_Frontier":
        "The colocated side broken out per EP group, so the choice the search makes "
        "is visible rather than asserted. The search does consider every EP and every "
        "attention tp; ep8 wins because a finer expert shard lowers the MoE stage and "
        "that stage is nearly flat in batch, so it acts as a floor no batch choice can "
        "get under -- on Qwen3 at 8 tokens per rank it is 14.27 ms at ep1, 9.59 at ep4 "
        "and 6.92 at ep8. EP above 8 was never benchmarked and does not divide 72; it "
        "would only strengthen the colocated side, so its absence does not flatter it.",
    "Matched_Shard":
        "Both arms holding the SAME number of experts per GPU: an AFD F pool of f GPUs "
        "against a colocated EP group of f GPUs. This separates the architectural "
        "question (does splitting pay?) from the sharding question (how finely are the "
        "experts spread?), which the free search mixes together because the colocated "
        "arm can reach ep8 while the AFD F pool stops at 4. Both sides fully measured.",
}


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def units(unit_gpus):
    """(replicas that fit in 72, GPUs left idle)."""
    n = DOMAIN // unit_gpus
    return n, DOMAIN - n * unit_gpus


def load():
    """cell -> {'afd': [...], 'colo': [...], 'oom': {topology -> [batches]}}."""
    cells = defaultdict(lambda: {"afd": [], "colo": [], "oom": defaultdict(list)})
    for r in csv.DictReader(SRC.open()):
        cell = (r["model"], int(r["isl"]), int(r.get("osl") or 1024), int(r["mtp"]))
        arm, topo = r["arm"], r["split"]
        if r["status"] == "oom":
            cells[cell]["oom"][topo].append(int(r["batch"]))
            continue
        if r["status"] != "ok" or not f(r.get("tpot_ms")):
            continue
        rec = {"topo": topo, "batch": int(r["batch"]),
               "unit_gpus": int(r["unit_gpus"]), "units": int(r["units_in_72"]),
               "idle": int(r["gpus_idle"]), "tpsgpu": f(r["tokens_s_gpu"]),
               "tpsgpu72": f(r["tpsgpu_domain72"]), "tpot_ms": f(r["tpot_ms"]),
               "conc": f(r["concurrency"]), "moe_ms": f(r["moe_ms"]),
               "prov": r.get("moe_provenance", ""), "mem": f(r.get("memory_gb")),
               "tp": int(r.get("a_tp") or 1),
               "tpsgpu72_g1": f(r.get("tpsgpu_domain72_g1")),
               "tpot_g1": f(r.get("tpot_ms_g1"))}
        if arm == "afd":
            rec.update(a=int(r["a"]), fq=int(r["f"]), a_tp=int(r.get("a_tp") or 1),
                       t_a=f(r.get("T_a_layer")), t_f=f(r.get("T_f_layer")),
                       balance=f(r.get("balance")),
                       mb=int(r.get("mb") or 1),
                       ag_eff=f(r.get("ag_effective")),
                       spread=f(r.get("base_spread_pct")))
        cells[cell][arm].append(rec)
    return {k: v for k, v in cells.items() if v["afd"] and v["colo"]}


def limited_by(pt, same_topo, oom_batches, sla):
    """Why this configuration could not use a larger batch.

    capacity  the next batch rung does not fit in memory
    latency   it fits but breaks the latency limit
    compute   it fits and is legal, but throughput stopped rising
    grid      there is no next rung -- the sweep ended, not the hardware
    """
    b = pt["batch"]
    ok = {c["batch"]: c for c in same_topo}
    rungs = sorted(x for x in set(ok) | set(oom_batches) if x > b)
    if not rungs:
        return "grid"
    nxt = rungs[0]
    if nxt not in ok:
        return "capacity"
    if sla is not None and ok[nxt]["tpot_ms"] > sla:
        return "latency"
    return "compute"


def best_budget(pts, oom, budget, sla=None):
    """Best point when only `budget` GPUs exist, charged over the whole budget."""
    cands = [p for p in pts
             if p["unit_gpus"] <= budget and (sla is None or p["tpot_ms"] <= sla)]
    if not cands:
        return None
    def eff(p):
        n = budget // p["unit_gpus"]
        return p["tpsgpu"] * n * p["unit_gpus"] / budget
    w = max(cands, key=lambda p: (eff(p), -p["tpot_ms"]))
    same = [p for p in pts if p["topo"] == w["topo"] and p["unit_gpus"] <= budget]
    w = dict(w)
    w["limited_by"] = limited_by(w, same, oom.get(w["topo"], []), sla)
    return w


def best(pts, oom, sla=None):
    """Highest tokens/s/GPU in the deployment unit, subject to the limit.

    v0514's convention, and the one `ratio` reports: tokens/s divided by every GPU
    in the deployment, AFD FFN GPUs included. The rack-tiling question -- a unit
    that does not divide 72 strands the remainder -- is answered on its own terms
    by the GPU_Budget sheet, and every row here carries afd_units_in_72 and
    afd_gpus_idle so the stranding is visible where it happens.

    The two conventions disagree on 3.2% of selections and always the same way:
    the AFD arm takes a unit that strands 6 to 16 GPUs for about 4% more per-unit
    throughput. ratio_domain72 is kept alongside for exactly those rows.
    """
    cands = [p for p in pts if sla is None or p["tpot_ms"] <= sla]
    if not cands:
        return None
    w = max(cands, key=lambda p: (p["tpsgpu"], -p["tpot_ms"]))
    # limited_by gets the UNFILTERED points for this topology. Handing it the
    # SLA-filtered list would hide every rung that exists but is too slow, and
    # those rungs would then fall through to the OOM branch and be reported as
    # `capacity` -- claiming the batch did not fit when in fact it fit and missed
    # the deadline. That is the difference between "buy more memory" and "relax
    # the SLA", so it has to be right.
    same = [p for p in pts if p["topo"] == w["topo"]]
    w = dict(w)
    w["limited_by"] = limited_by(w, same, oom.get(w["topo"], []), sla)
    return w


def emit(cell, a, c, sla=None):
    """One comparison row, both blocks filled from the winners handed in."""
    model, isl, osl, mtp = cell
    row = {"model": MODEL_LABEL[model], "attn": ATTN[model], "isl": isl, "osl": osl,
           "mtp_nextn": mtp, **CONST_VALS}
    if sla is not None:
        row["sla_tpot_ms"] = sla
    if a:
        row.update({
            "afd_split": a["topo"], "afd_A_gpus": a["a"], "afd_F_gpus": a["fq"],
            "afd_a_tp": a["a_tp"], "afd_microbatches": a.get("mb", 1),
            "afd_unit_gpus": a["unit_gpus"],
            "afd_units_in_72": a["units"], "afd_gpus_idle": a["idle"],
            "afd_batch": a["batch"], "afd_tpsgpu": a["tpsgpu"],
            "afd_tpsgpu_domain72": a["tpsgpu72"], "afd_tpot_ms": round(a["tpot_ms"], 3),
            "afd_tokps_per_user": round(1000.0 / a["tpot_ms"], 2),
            "afd_conc": int(a["conc"]),
            "afd_conc_per_gpu": round(a["conc"] / a["unit_gpus"], 2),
            "afd_t_attn_us": round(a["t_a"], 2) if a["t_a"] else None,
            "afd_t_ffn_us": round(a["t_f"], 2) if a["t_f"] else None,
            "afd_tf_over_ta": round(a["balance"], 4) if a["balance"] else None,
            "afd_moe_ms_measured": round(a["moe_ms"], 3),
            "afd_moe_provenance": a["prov"],
            "afd_ag_effective": a.get("ag_eff"),
            "afd_base_spread_pct": a.get("spread"),
            "afd_memory_gb": round(a["mem"], 1) if a["mem"] else None,
            "afd_batch_limited_by": a["limited_by"]})
    if c:
        row.update({
            "colo_cfg": c["topo"], "colo_attn_tp": c["tp"],
            "colo_unit_gpus": c["unit_gpus"],
            "colo_units_in_72": c["units"], "colo_gpus_idle": c["idle"],
            "colo_batch": c["batch"], "colo_tpsgpu": c["tpsgpu"],
            "colo_tpsgpu_domain72": c["tpsgpu72"], "colo_tpot_ms": round(c["tpot_ms"], 3),
            "colo_tokps_per_user": round(1000.0 / c["tpot_ms"], 2),
            "colo_conc": int(c["conc"]),
            "colo_conc_per_gpu": round(c["conc"] / c["unit_gpus"], 2),
            "colo_moe_ms_measured": round(c["moe_ms"], 3),
            "colo_memory_gb": round(c["mem"], 1) if c["mem"] else None,
            "colo_batch_limited_by": c["limited_by"]})
    if a and c:
        row["ratio"] = round(a["tpsgpu"] / c["tpsgpu"], 4)
        row["ratio_domain72"] = round(a["tpsgpu72"] / c["tpsgpu72"], 4)
        # Sensitivity end-point: the same row with the ONE extrapolated term
        # (the A-pool width factor g) switched off, which is the most
        # AFD-favourable value available. A conclusion that holds at both ends
        # does not rest on the extrapolation.
        if a.get("tpsgpu72_g1"):
            row["ratio_domain72_g1"] = round(a["tpsgpu72_g1"] / c["tpsgpu72"], 4)
    return row


def sheet(wb, name, cols, rows, caption=None):
    ws = wb.create_sheet(name[:31])
    top = 1
    if caption:
        ws.cell(row=1, column=1, value=caption).font = TITLE
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(len(cols), 8))
        ws.cell(row=1, column=1).alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[1].height = 30
        top = 3
    for i, n in enumerate(cols, 1):
        c = ws.cell(row=top, column=i, value=n)
        c.font, c.fill = HDR_FONT, HDR_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = ws.cell(row=top + 1, column=1)
    ratio_cols = {i for i, n in enumerate(cols, 1)
                  if n in ("ratio", "ratio_domain72", "ratio_domain72_g1",
                           "ratio_band_lo", "ratio_band_hi", "ratio_budget")}
    sens_col = next((i for i, n in enumerate(cols, 1) if n == "sla_edge_sensitive"), None)
    surv_col = next((i for i, n in enumerate(cols, 1) if n == "gain_holds_across_band"), None)
    for r, row in enumerate(rows, top + 1):
        for i, n in enumerate(cols, 1):
            c = ws.cell(row=r, column=i, value=row.get(n))
            c.font = BODY
            if i in ratio_cols and isinstance(row.get(n), (int, float)):
                v = row[n]
                c.fill = GAIN if v >= 1.10 else LOSS if v < 0.95 else EDGE
            if i == sens_col and row.get(n) == "YES":
                c.fill = EDGE
            if i == surv_col and row.get(n) == "NO":
                c.fill = LOSS
    for col in ws.columns:
        w = max((len(str(x.value)) for x in col if x.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(9, min(28, w + 2))
    return ws


def sheet_config(wb, cells, nrows):
    """Item / Detail, laid out like the v0514 Configuration sheet.

    Same two-column shape and the same topics, plus the rows that only exist
    because this study injects a measured kernel instead of modelling one.
    """
    ws = wb.create_sheet("Configuration", 0)
    prov = defaultdict(int)
    for d in cells.values():
        for pt in d["afd"] + d["colo"]:
            key = ("measured" if pt["prov"].startswith("measured") and "g(" not in pt["prov"]
                   else "measured x g(ag), ag within the measured 1..7"
                   if "beyond" not in pt["prov"]
                   else "measured x g(ag), ag beyond 7")
            prov[key] += 1
    tot = sum(prov.values()) or 1
    prov_txt = "; ".join(f"{k}: {v/tot*100:.1f}%" for k, v in sorted(prov.items()))

    lines = [
        ("Item", "Detail"),
        ("Subject", "Homogeneous GPU-to-GPU Attention-FFN Disaggregation versus colocated "
                    "serving, decode phase, with the MoE stage taken from FastAFD MegaMoE "
                    "measurements on B200 rather than from a performance model."),
        ("Hardware", "NVIDIA GB200 NVL72. 72 GPUs in one NVLink domain, 185.0 GiB usable HBM "
                     "per GPU (gb200.yaml mem_capacity 198,674,743,296 B)."),
        ("Why B200 numbers apply", "MegaMoE runs entirely inside one NVLink domain and both "
                                   "machines give it 900 GB/s (gb200.yaml and b200_sxm.yaml "
                                   "agree to the digit). GB200 has 3.9% more memory bandwidth "
                                   "and 11.1% more FP4 throughput, so reusing a B200 time for "
                                   "a weight-bound stage is conservative."),
        ("Serving framework", "SGLang 0.5.12 packaged performance tables."),
        ("Estimation engine", "AIConfigurator. AFD arm in afd mode, colocated arm in "
                              "static_gen mode. Every deployment and batch is enumerated "
                              "explicitly and estimated one at a time, in process."),
        ("Database mode", "HYBRID for all three, but it does different work in each arm. "
                          "The AFD arm passes --afd-moe-time-ms, which bypasses AIC's MoE op "
                          "query entirely, so no fallback ever fires: 10 of 10 sampled AFD rows "
                          "return bit-identical TPOT under SILICON. The colocated arm needs it "
                          "to run at all -- AIC must build a full generation breakdown before "
                          "the substitution can overwrite the MoE ops, and gb200's silicon "
                          "tables have no power_law_1.2 routing slice for the Qwen3 MoE shape, "
                          "so SILICON aborts on a number this study then discards."),
        ("MiniMax-M3 caveat", "M3 is the one model where HYBRID supplies a quantity that is "
                              "KEPT. MSA has no silicon rows, so its attention comes from "
                              "SOL plus an empirical factor, borrowing utilisation from DSA "
                              "through AIC's cross-op transfer. That term is 42% to 61% of the "
                              "M3 decode step (rising with batch), against a measured MoE "
                              "stage. Qwen3-235B and MiniMax-M2.5 keep nothing model-filled: "
                              "their attention, KV, dense and router GEMM are silicon-backed "
                              "under either mode."),
        ("Models", "MiniMax-M2.5, MiniMax-M3, Qwen3-235B-A22B. These are the models with a "
                   "MegaMoE benchmark; DeepSeek-V4 flash and pro exist in the results tree "
                   "as one smoke file each and fail the protocol gate."),
        ("Attention", "Qwen3-235B-A22B: GQA 64:4; MiniMax-M2.5: GQA 48:8; MiniMax-M3: MSA."),
        ("MoE topology", "Qwen3-235B: 128 experts, top-8, 94 of 94 layers MoE. "
                         "MiniMax-M2.5: 256 experts, top-8, 62 of 62. "
                         "MiniMax-M3: 128 experts, top-4, 57 of 60."),
        ("", ""),
        ("MoE stage source", "FastAFD MegaMoE on B200 SXM, measured. BOTH arms get it: the AFD "
                             "arm through --afd-moe-time-ms, the colocated arm through an "
                             "op-level substitution into the generation breakdown. Comparing a "
                             "measured kernel against a modelled one would not be a comparison "
                             "of deployments."),
        ("Injection point", "AIC sums a_partition.attn_ops for the A side and replaces "
                            "f_partition.ffn_ops wholesale with the injected value, setting "
                            "T_a2f to zero. The benchmark's measurement_boundary -- A-side FP8 "
                            "quant and dispatch through F-side expert compute and combine -- is "
                            "exactly that span, so nothing is double counted or dropped."),
        ("Lookup key", "(model, nextN, microbatches, experts per F GPU, sequences per A rank). "
                       "Exact key, no interpolation; the batch grid IS the measured grid. NOT "
                       "tokens per F GPU: the stage includes per-A-rank work, and on "
                       "MiniMax-M2.5 at 32 experts/GPU and 64 sequences per A rank the tokens "
                       "per F GPU move 16 -> 32 -> 48 while the stage stays at "
                       "7.89 / 7.91 / 7.89 ms."),
        ("Modelled term", "One: g(ag) = ag ** 0.053 (ag ** 0.193 below 4 sequences per rank), "
                          "the rise as the A pool widens at fixed shard and batch. Fitted on "
                          "ag = 1..7, used to ag = 68 where it reaches 1.25. g > 1, so it "
                          "always makes AFD look worse. Every row also carries "
                          "ratio_domain72_g1, the same comparison with g switched off."),
        ("Data provenance", prov_txt + ". The extrapolated part is the g factor only; the "
                            "stage itself is an exact measured lookup in every row."),
        ("", ""),
        ("moe_precision (column)", "w4a8_mxfp4_mxfp8, the only precision benchmarked. It is NOT "
                                   "NVFP4. Unlike a modelled sweep, precision cannot be varied "
                                   "here without new measurements."),
        ("moe_kernel (column)", "fastafd_megamoe, measured. Attention, KV cache and dense "
                                "layers come from AIC's gb200 tables."),
        ("Input length", "4096, 8192, 16384, 32768, 65536, 131072"),
        ("Generation length", "1024 throughout; 256 and 4096 additionally at draft length 0 "
                              "(ISL_x_OSL sheet)."),
        ("Speculative decoding", "Draft lengths 0, 1, 2 with accepted lengths 0.85 and 1.53. "
                                 "All three carry full measured MegaMoE coverage."),
        ("Microbatching", "1 and 2, both measured. At mb=1 AIC falls back to serial -- one "
                          "in-flight microbatch cannot overlap anything, since layer i+1's A "
                          "input IS layer i's F output -- so the step is t_a + t_f. mb=2 buys "
                          "max(t_a + t_a2f, t_f + t_f2a) at a measured 1.79x on the stage, "
                          "because each lane re-streams the expert weights. Each cell keeps "
                          "whichever is better."),
        ("Pipeline model", "conservative."),
        ("", ""),
        ("Colocated search", "EP in 1, 2, 4, 8 crossed with attention tensor parallelism 1, 2, "
                             "4; attention data parallelism is the remainder. EP is capped at 8 "
                             "by measurement, and 72 = 2^3 x 3^2 means an EP that divides both "
                             "the expert count and the rack can only be 1, 2, 4 or 8 anyway -- "
                             "ep16 strands 8 GPUs."),
        ("AFD search", "F pools of 1, 2, 4 GPUs (the measured shards), A pools of 1 to 71 with "
                       "A + F <= 72, A-pool tensor parallelism 1, 2, 4, microbatches 1 and 2. "
                       "Ratios to 71:1. F >= 8 needs a shard only the colocated kernel has run, "
                       "and the two kernels differ by a median 7.0% and up to 120% at matched "
                       "load, so that region is left empty rather than modelled."),
        ("Node granularity", "AIC expresses AFD pools in node units and requires tp_a to divide "
                             "gpus_per_node, so four copies of gb200.yaml are swept at 1, 2, 4 "
                             "and 8 GPUs per node and unioned. They agree to the digit where "
                             "they overlap; the override is bandwidth- and latency-neutral "
                             "inside one rack."),
        ("Batch selection", "The measured rungs: 1, 2, 4, 6, 8, 16, 32, 48, 64, 96, 128, 192, "
                            "256, 384 sequences per rank. Not a synthetic grid -- every rung is "
                            "a benchmark point, which is what keeps the injected value exact."),
        ("Domain accounting", "A unit of u GPUs is replicated floor(72/u) times and the "
                              "remainder is idle but still paid for: tpsgpu_domain72 = tpsgpu * "
                              "u * floor(72/u) / 72. Both arms measured the same way."),
        ("Evaluation rule", "Each arm is swept over its own configuration space; points over "
                            "the HBM limit are discarded; the highest tokens/s/GPU across the "
                            "whole 72-GPU domain wins, subject to the latency limit."),
        ("Throughput metric", "Decode tokens per second divided by every GPU in the "
                              "deployment unit, including AFD FFN GPUs that host no requests. "
                              "A unit that does not divide 72 strands the remainder; that is "
                              "reported per row as afd_gpus_idle and priced explicitly on the "
                              "GPU_Budget sheet."),
        ("Capacity wall", "The batch grid is the measured grid, so a configuration whose "
                          "memory wall falls between two rungs is reported at the lower one. "
                          "The *_capacity_wall columns bisect the MEMORY model to find the "
                          "real wall -- no MoE time is involved, so nothing leaves the "
                          "measured envelope -- and rung_inflation is how much the ladder "
                          "flatters the concurrency comparison. Qwen3 at isl=65536 under a "
                          "50 ms limit reads 16.00 against 22.67 seq/GPU, a 41.7% AFD "
                          "advantage; the walls are 27.00 against 27.86, which is 3.2%. The "
                          "colocated arm was not out of memory at batch 64, it was out of "
                          "rungs at 62% of the card. Refining the ladder around the wall is "
                          "submitted separately; until it lands this bounds the artefact "
                          "rather than removing it, and the THROUGHPUT columns are unaffected "
                          "because they still come from exact lookups."),
        ("Latency band", "Every SLA row is re-optimised at 0.9x, 0.95x, 1.0x, 1.05x, 1.1x and "
                         "1.2x the nominal limit and the resulting ratios are reported as "
                         "[ratio_band_lo, ratio_band_hi]. Throughput is a STAIRCASE in the "
                         "limit -- the MoE stage is weight-bound, so TPOT barely moves with "
                         "batch -- and a single-threshold ratio measures which arm happened "
                         "to step most recently. sla_edge_sensitive marks a band wider than "
                         "25% of the nominal; gain_holds_across_band requires the LOWER edge "
                         "to clear 1.10, which is what the AFD_Perf_Gain sheet admits on."),
        ("Ratio", "AFD tokens/s/GPU divided by colocated tokens/s/GPU at the same model, "
                  "context, generation length, draft length and latency limit. Per "
                  "deployment unit, v0514's convention; ratio_domain72 restates it over the "
                  "whole rack and the band columns are computed on `ratio`.",
         "0.95 <= ratio < 1.1", "ratio >= 1.1"),
        ("batch_limited_by", "What stopped the winning batch from growing. capacity: the next "
                             "rung does not fit in HBM. latency: it fits but breaks the limit. "
                             "compute: it fits and is legal but throughput stopped rising. "
                             "grid: the sweep ended, not the hardware."),
        ("Latency limits", "10, 15, 20, 30, 50, 100, 150 ms time per output token."),
        ("", ""),
        ("Stability gate", "CV <= 4% across the timed iterations, all ranks finite, AND the "
                           "campaign protocol (warmups >= 30, iterations >= 40). The last "
                           "clause is load-bearing: a smoke run at warmups=2 / iterations=10 "
                           "reported 0.190 ms for a point that measures 6.919 ms properly, with "
                           "a CV of 3.55% that clears the dispersion gate on its own."),
        ("Reproducibility", "Explicit directory allowlist, sorted file iteration, duplicate "
                            "points reduced by median, atomic write, and a row-count check "
                            "against the log. 16 randomly chosen rows were recomputed in a "
                            "fresh process and matched to the last digit."),
        ("Validation", "g(ag) leave-one-out over 758 measured points: median error 0.8%, p90 "
                       "7.0%, no bias by ag. Split versus colocated kernel at matched load: "
                       "median 7.0%, max 120% -- the basis for excluding F >= 8. Shard "
                       "extrapolation offered to BOTH arms: no extrapolated configuration wins "
                       "anywhere, so the conclusion does not rest on it."),
        ("Not covered", "DeepSeek-V3 / V3.1 / Kimi-K2.5 (never benchmarked with MegaMoE); "
                        "FP8-block, INT4-wo and NVFP4 (only w4a8_mxfp4_mxfp8 was measured); "
                        "expert shards finer than 16 per GPU (8 GPUs cannot produce them)."),
        ("Validation status", "Simulation. The MoE stage in every row is a hardware "
                              "measurement; the attention, KV and memory sides are AIC model "
                              "output. No end-to-end AFD serving run on GB200 has been "
                              "measured, and no figure here has been confirmed against a "
                              "GB200 system."),
        ("Configuration points", f"{307524:,} enumerated; {nrows:,} comparison rows."),
    ]
    for r, row in enumerate(lines, 1):
        for c, val in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=val)
            if r == 1:
                cell.font, cell.fill = HDR_FONT, HDR_FILL
            elif c == 1:
                cell.font = Font(name=FONT, bold=True, size=10)
            else:
                cell.font = BODY
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        if row[0] == "Ratio" and len(row) > 2:
            ws.cell(row=r, column=3).fill = EDGE
            ws.cell(row=r, column=4).fill = GAIN
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 108
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 14
    ws.freeze_panes = "A2"
    return ws


# Walls are precomputed in parallel and cached on disk by precompute_walls.py.
# Computing them inline is what ran the finishing job out of its Slurm limit once
# the sweep reached 764k rows: each wall is a bisection of 10-40 AIC calls, and
# the builder was doing them one at a time.
_WALL_DISK = {}
_WALL_PATH = HANDOFF / "capacity_walls.json"
if _WALL_PATH.exists():
    import json as _json
    _WALL_DISK = _json.loads(_WALL_PATH.read_text())
_WALL_CACHE = {}


SKIP_CAP = bool(os.environ.get("SKIP_CAPACITY_WALL"))


def capacity_walls(cell, a, c):
    """(colo wall, colo seq/GPU, afd wall, afd seq/GPU) at the memory limit.

    Memory only -- no MoE stage time is involved -- so this stays inside the
    measured envelope even though the batch it lands on was never benchmarked.
    Cached per (cell, topology): bisection costs a handful of AIC calls and the
    same topology recurs across every latency limit.
    """
    if SKIP_CAP:
        return {}
    import json as _json
    import capacity_wall as CW
    import run_gb200_measured as R
    from colocated_measured import MODELS
    model, isl, osl, mtp = cell
    out = {}
    if c:
        key = ("colo", model, isl, osl, mtp, c["unit_gpus"], c["tp"])
        hit = _WALL_DISK.get(_json.dumps(list(key)))
        if isinstance(hit, list):
            _WALL_CACHE[key] = (hit[0], hit[1])
        elif key not in _WALL_CACHE:
            _WALL_CACHE[key] = CW.colo_wall(R._est(), MODELS[model]["path"],
                                            c["unit_gpus"], c["tp"], isl, osl, mtp,
                                            R.CAPACITY_GIB)
        b, conc = _WALL_CACHE[key]
        if b:
            out["colo_batch_capacity_wall"] = b
            out["colo_conc_per_gpu_at_wall"] = round(conc / c["unit_gpus"], 2)
    if a:
        g = a["a_tp"] if a["a_tp"] in (1, 2, 4) else 1
        while g > 1 and (a["a"] % g or a["fq"] % g):
            g //= 2
        key = ("afd", model, isl, osl, mtp, a["a"], a["fq"], a["a_tp"], g,
               a.get("mb", 1))
        hit = _WALL_DISK.get(_json.dumps(list(key)))
        if isinstance(hit, list):
            _WALL_CACHE[key] = (hit[0], hit[1])
        elif key not in _WALL_CACHE:
            _WALL_CACHE[key] = CW.afd_wall(R._afd_once, model, a["a"], a["fq"],
                                           a["a_tp"], max(g, 1), isl, osl, mtp,
                                           a.get("mb", 1))
        b, conc = _WALL_CACHE[key]
        if b:
            out["afd_batch_capacity_wall"] = b
            out["afd_conc_per_gpu_at_wall"] = round(conc / a["unit_gpus"], 2)
    if a and c and a["conc"] and c["conc"]:
        r_rung = (a["conc"] / a["unit_gpus"]) / (c["conc"] / c["unit_gpus"])
        out["conc_ratio_at_rungs"] = round(r_rung, 4)
        if "afd_conc_per_gpu_at_wall" in out and out.get("colo_conc_per_gpu_at_wall"):
            r_wall = out["afd_conc_per_gpu_at_wall"] / out["colo_conc_per_gpu_at_wall"]
            out["conc_ratio_at_wall"] = round(r_wall, 4)
            out["rung_inflation"] = round(r_rung / r_wall, 4) if r_wall else None
    return out


def main():
    cells = load()
    wb = Workbook()
    wb.remove(wb.active)
    order = sorted(cells, key=lambda c: (MODEL_LABEL[c[0]], c[3], c[2], c[1]))
    # v0514 pins osl=1024 and nextN=0 on the headline sheets and gives osl its own
    # sheet; the SLA and budget sheets carry every nextN.
    base_1024 = [c for c in order if c[2] == 1024]
    mtp0 = [c for c in base_1024 if c[3] == 0]

    # ---- no-limit sheets --------------------------------------------------
    def pair(cell, sla=None):
        d = cells[cell]
        return emit(cell, best(d["afd"], d["oom"], sla), best(d["colo"], d["oom"], sla), sla)

    nolimit = []
    for c in base_1024:
        row = pair(c)
        row.update(capacity_walls(c, best(cells[c]["afd"], cells[c]["oom"]),
                                  best(cells[c]["colo"], cells[c]["oom"])))
        nolimit.append(row)
    NOSLA_FULL = NOSLA_COLS + CAP_COLS
    sheet(wb, "Model_x_Precision", NOSLA_FULL, nolimit, CAPTION["Model_x_Precision"])
    sheet(wb, "Context", NOSLA_FULL, nolimit, CAPTION["Context"])
    iso = []
    for c in [x for x in order if x[3] == 0]:
        row = pair(c)
        row.update(capacity_walls(c, best(cells[c]["afd"], cells[c]["oom"]),
                                  best(cells[c]["colo"], cells[c]["oom"])))
        iso.append(row)
    sheet(wb, "ISL_x_OSL", NOSLA_FULL, iso, CAPTION["ISL_x_OSL"])

    # ---- SLA sweep --------------------------------------------------------
    sla_rows = []
    for cell in base_1024:
        d = cells[cell]
        for sla in SLA_LIMITS:
            if not best(d["afd"], d["oom"], sla) and not best(d["colo"], d["oom"], sla):
                continue
            row = pair(cell, sla)
            band = []
            for k in SLA_BAND:
                ak = best(d["afd"], d["oom"], sla * k)
                ck = best(d["colo"], d["oom"], sla * k)
                if ak and ck and ck["tpsgpu"]:
                    band.append(ak["tpsgpu"] / ck["tpsgpu"])
            base = row.get("ratio")
            if band:
                lo, hi = min(band), max(band)
                row["ratio_band_lo"] = round(lo, 4)
                row["ratio_band_hi"] = round(hi, 4)
                if base:
                    row["sla_edge_sensitive"] = "YES" if (hi - lo) / base > 0.25 else "no"
                    row["gain_holds_across_band"] = "YES" if lo > 1.10 else "NO"
            row.update(capacity_walls(cell, best(d["afd"], d["oom"], sla),
                                      best(d["colo"], d["oom"], sla)))
            sla_rows.append(row)
    sheet(wb, "SLA_Latency", SLA_COLS, sla_rows, CAPTION["SLA_Latency"])

    # ---- Summary: the minimum needed to read the result ---------------------
    SUM_COLS = ["model", "isl", "osl", "mtp_nextn", "sla_tpot_ms",
                "colo_cfg", "colo_batch", "afd_split", "afd_batch",
                "ratio", "ratio_band_lo", "rung_inflation", "verdict"]
    summary = []
    for r in sla_rows:
        lo = r.get("ratio_band_lo")
        hi = r.get("ratio_band_hi")
        infl = r.get("rung_inflation")
        base = r.get("ratio")
        if base is None:
            v = "one arm infeasible"
        elif lo is None:
            v = "no band"
        elif lo <= 1.10 < (hi or lo):
            v = "threshold noise"          # wins only at some limits
        elif lo <= 1.10:
            v = "no gain"
        elif infl is not None and infl >= 1.25:
            v = "gain, ladder-inflated"    # real but overstated by the batch grid
        else:
            v = "gain"
        summary.append({**{k: r.get(k) for k in SUM_COLS if k != "verdict"},
                        "verdict": v})
    order_v = {"gain": 0, "gain, ladder-inflated": 1, "threshold noise": 2,
               "no gain": 3, "no band": 4, "one arm infeasible": 5}
    summary.sort(key=lambda r: (order_v.get(r["verdict"], 9),
                                -(r.get("ratio") or 0)))
    ws_s = sheet(wb, "Summary", SUM_COLS, summary, CAPTION["Summary"])
    vi = SUM_COLS.index("verdict") + 1
    for rr in range(4, ws_s.max_row + 1):
        v = ws_s.cell(row=rr, column=vi).value
        if v == "gain":
            ws_s.cell(row=rr, column=vi).fill = GAIN
        elif v in ("gain, ladder-inflated", "threshold noise"):
            ws_s.cell(row=rr, column=vi).fill = EDGE
    wb.move_sheet("Summary", offset=-(len(wb.sheetnames) - 2))

    # Admission is on the BAND, not the nominal point: a gain that exists only at
    # one threshold is a staircase artifact, not a property of the deployment.
    gain = [r for r in sla_rows if (r.get("ratio_band_lo") or 0) > 1.10]
    sheet(wb, "AFD_Perf_Gain > 10% ", GAIN_COLS, gain, CAPTION["AFD_Perf_Gain > 10% "])

    # ---- per-split sheet --------------------------------------------------
    split_rows = []
    for cell in mtp0:
        d = cells[cell]
        c = best(d["colo"], d["oom"])
        for topo in sorted({p["topo"] for p in d["afd"]},
                           key=lambda s: (int(s.split("A")[0]), s)):
            # topo already carries a_tp ("32A4F/tp_a2"), so each sharding of the
            # same A:F is its own row rather than being collapsed into one.
            pts = [p for p in d["afd"] if p["topo"] == topo]
            a = best(pts, d["oom"])
            if a:
                # No capacity columns here: see precompute_walls.needed.
                split_rows.append(emit(cell, a, c))
    sheet(wb, "AF_Split", SPLIT_COLS, split_rows, CAPTION["AF_Split"])

    # ---- matched-shard sheet ----------------------------------------------
    matched = []
    for cell in base_1024:
        d = cells[cell]
        for fq in (1, 2, 4):
            A = [p for p in d["afd"] if p["fq"] == fq]
            C = [p for p in d["colo"] if p["unit_gpus"] == fq]
            if not A or not C:
                continue
            a = best(A, d["oom"])
            c = best(C, d["oom"])
            row = emit(cell, a, c)
            row["experts_per_gpu"] = MODEL_EXPERTS[cell[0]] // fq
            row.update(capacity_walls(cell, a, c))
            matched.append(row)
    sheet(wb, "Matched_Shard", COMMON + ["experts_per_gpu"] + CONSTS + AFD_BLOCK
          + COLO_BLOCK + ["ratio", "ratio_domain72", "ratio_domain72_g1"] + CAP_COLS,
          matched, CAPTION["Matched_Shard"])

    # ---- per-EP colocated frontier ----------------------------------------
    ep_rows = []
    for cell in base_1024:
        d = cells[cell]
        for ep in sorted({p["unit_gpus"] for p in d["colo"]}):
            pts = [p for p in d["colo"] if p["unit_gpus"] == ep]
            for sla in (None, 10, 15, 20, 30, 50):
                c = best(pts, d["oom"], sla)
                if not c:
                    continue
                row = emit(cell, None, c, sla)
                row["colo_experts_per_gpu"] = MODEL_EXPERTS[cell[0]] // ep
                row["is_search_winner"] = (best(d["colo"], d["oom"], sla) or {}).get("topo") == c["topo"]
                row.update(capacity_walls(cell, None, c))
                ep_rows.append(row)
    sheet(wb, "Colo_EP_Frontier",
          COMMON + ["sla_tpot_ms", "colo_experts_per_gpu", "is_search_winner"]
          + CONSTS + COLO_BLOCK
          + ["colo_batch_capacity_wall", "colo_conc_per_gpu_at_wall"],
          ep_rows, CAPTION["Colo_EP_Frontier"])

    # ---- GPU_Budget --------------------------------------------------------
    # Same question at other rack sizes. A unit of u GPUs fits floor(B/u) times in
    # a budget of B and strands the rest; tokens/s/GPU is charged over the WHOLE
    # budget, so a unit that tiles badly pays for the silicon it cannot use.
    budget_rows = []
    for cell in base_1024:
        d = cells[cell]
        for B in BUDGETS:
            for sla in (None, 20, 50):
                a = best_budget(d["afd"], d["oom"], B, sla)
                c = best_budget(d["colo"], d["oom"], B, sla)
                if not a and not c:
                    continue
                row = emit(cell, a, c, sla)
                row["budget_gpus"] = B
                for arm, w in (("afd", a), ("colo", c)):
                    if w:
                        n = B // w["unit_gpus"]
                        row[f"{arm}_replicas"] = n
                        row[f"{arm}_gpus_used"] = n * w["unit_gpus"]
                        row[f"{arm}_gpus_stranded"] = B - n * w["unit_gpus"]
                        row[f"{arm}_tpsgpu_budget"] = round(
                            w["tpsgpu"] * n * w["unit_gpus"] / B, 2)
                if a and c and row.get("colo_tpsgpu_budget"):
                    row["ratio_budget"] = round(
                        row["afd_tpsgpu_budget"] / row["colo_tpsgpu_budget"], 4)
                row.update(capacity_walls(cell, a, c))
                budget_rows.append(row)
    sheet(wb, "GPU_Budget", BUDGET_COLS, budget_rows, CAPTION["GPU_Budget"])

    sheet_config(wb, cells, len(nolimit) + len(sla_rows))
    wb.save(OUT)
    print(f"wrote {OUT}")
    return nolimit, sla_rows


if __name__ == "__main__":
    main()
