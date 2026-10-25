#!/usr/bin/env python3
"""Side-by-side workbook: routing skew measured per layer vs summed over layers.

Two sweeps, identical in every other respect -- same measured MegaMoE curves,
same f, same formula 1 + f*(I-1), same grid. The only difference is how the
expert-load imbalance I is read out of the recorded routing:

  Layer-Summed   add all 94 layers' expert counts, then take max/mean over
                 ranks. What the first sweep did.
  Per-Layer      take max/mean over ranks within each layer, then average
                 across layers. What a decode step actually pays, because
                 every layer waits for its own busiest rank and that rank
                 changes almost every layer.

Also carries the attention-TP question, because the correction lands hardest on
the AFD arm's fine shards and those are the configurations that lean on TP.
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from afd_paths import data_root

HERE = Path(__file__).resolve().parent
HANDOFF = HERE / "handoff"
# Data root: the unpacked data bundle, from AFD_MEGAMOE_ROOT (required).
W = data_root()

OLD = Path(os.environ.get("AFD_CSV_LAYERSUMMED") or HANDOFF / "gb200_with_real_routing.csv")
NEW = Path(os.environ.get("AFD_CSV_PERLAYER") or HANDOFF / "gb200_perlayer_routing.csv")
OUT = Path(os.environ.get("AFD_COMPARISON_OUT")
           or HANDOFF / "AFD_RoutingSkew_PerLayer_vs_LayerSummed.xlsx")

SLA_LIMITS = [10, 15, 20, 30, 50, 100, 150]
MODEL_NAME = {"qwen3_235b_fp4": "Qwen3-235B-A22B",
              "minimax_m25_fp4": "MiniMax-M2.5",
              "minimax_m3_fp4": "MiniMax-M3",
              "deepseek_v4_flash_fp4": "DeepSeek-V4-Flash",
              "deepseek_v4_pro_fp4": "DeepSeek-V4-Pro"}
MODEL_EXPERTS = {"qwen3_235b_fp4": 128, "minimax_m25_fp4": 256, "minimax_m3_fp4": 128,
                 "deepseek_v4_flash_fp4": 256, "deepseek_v4_pro_fp4": 384}

HDR = PatternFill("solid", fgColor="1F3864")
SUBHDR = PatternFill("solid", fgColor="D9E1F2")
GOOD = PatternFill("solid", fgColor="E2EFDA")
BAD = PatternFill("solid", fgColor="FCE4E4")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
COEFF = re.compile(r"x([\d.]+) \(routing skew (\w+), (EP|F)=(\d+)")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------- load
def load(path: Path):
    """cell -> {'afd': [...], 'colo': [...]}, plus the applied coefficients."""
    cells = defaultdict(lambda: {"afd": [], "colo": []})
    coeffs = defaultdict(list)
    for r in csv.DictReader(path.open()):
        m = COEFF.search(r.get("moe_provenance") or "")
        if m and r["status"] == "ok":
            coeffs[(r["arm"], m.group(3), int(m.group(4)), r["model"])].append(float(m.group(1)))
        if r["status"] != "ok" or not num(r.get("tpot_ms")):
            continue
        cell = (r["model"], int(r["isl"]), int(r.get("osl") or 1024), int(r["mtp"]))
        cells[cell][r["arm"]].append({
            "split": r["split"], "batch": int(r["batch"]),
            "tps": num(r["tokens_s_gpu"]), "tpot": num(r["tpot_ms"]),
            "tp": int(r.get("a_tp") or 1),
            "a": int(r["a"]) if r["a"] else 0, "f": int(r["f"]) if r["f"] else 0,
            "unit": int(r["unit_gpus"]), "conc": num(r["concurrency"]),
            "coeff": float(m.group(1)) if m else 1.0,
        })
    return {k: v for k, v in cells.items() if v["afd"] and v["colo"]}, coeffs


def best(pts, sla=None, tp=None):
    c = [p for p in pts if (sla is None or p["tpot"] <= sla) and (tp is None or p["tp"] == tp)]
    return max(c, key=lambda p: (p["tps"], -p["tpot"])) if c else None


def imbalance(layers, ranks):
    """Same computation megamoe_table does, so the sheet cannot drift from it."""
    if not layers or ranks <= 0 or len(layers[0]) % ranks:
        return None
    per = len(layers[0]) // ranks
    vals = []
    for counts in layers:
        loads = [sum(counts[i * per:(i + 1) * per]) for i in range(ranks)]
        mean = sum(loads) / ranks
        if mean > 0:
            vals.append(max(loads) / mean)
    return sum(vals) / len(vals) if vals else None


# -------------------------------------------------------------------- sheet
def sheet(wb, name, cols, rows, caption=None, widths=None):
    ws = wb.create_sheet(name)
    r0 = 1
    if caption:
        ws.cell(1, 1, caption).font = Font(italic=True, color="595959", size=10)
        ws.cell(1, 1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=max(len(cols), 6))
        ws.row_dimensions[1].height = 30
        r0 = 4
    for i, c in enumerate(cols, 1):
        cell = ws.cell(r0, i, c)
        cell.fill = HDR
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = BOX
    ws.row_dimensions[r0].height = 34
    for j, row in enumerate(rows, r0 + 1):
        for i, c in enumerate(cols, 1):
            v = row.get(c)
            cell = ws.cell(j, i, v)
            cell.border = BOX
            cell.font = Font(size=10)
            if isinstance(v, float):
                cell.number_format = "0.000" if abs(v) < 100 else "0.0"
        fill = row.get("_fill")
        if fill:
            for i in range(1, len(cols) + 1):
                ws.cell(j, i).fill = fill
    for i, c in enumerate(cols, 1):
        w = (widths or {}).get(c)
        if w is None:
            w = max(11, min(30, len(c) * 0.95 + 3))
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = ws.cell(r0 + 1, 1)
    return ws


# ------------------------------------------------------------------ content
def sheet_summary(wb, old, new):
    rows = []

    def block(label, sla):
        out = {}
        for tag, cells in (("Layer-Summed", old), ("Per-Layer", new)):
            R = []
            for cell, d in cells.items():
                a, c = best(d["afd"], sla), best(d["colo"], sla)
                if a and c:
                    R.append(a["tps"] / c["tps"])
            out[tag] = R
        a, b = out["Layer-Summed"], out["Per-Layer"]
        if not a or not b:
            return None
        return {"Latency Limit (ms/token)": label,
                "Scenarios Compared": len(b),
                "AFD Wins - Layer-Summed": sum(1 for x in a if x > 1),
                "AFD Wins - Per-Layer": sum(1 for x in b if x > 1),
                "Median Throughput Ratio - Layer-Summed": round(st.median(a), 4),
                "Median Throughput Ratio - Per-Layer": round(st.median(b), 4),
                "Change in Median Ratio": round(st.median(b) - st.median(a), 4),
                "_fill": GOOD if st.median(b) > st.median(a) else BAD}

    for sla in SLA_LIMITS:
        r = block(sla, sla)
        if r:
            rows.append(r)
    r = block("no limit", None)
    if r:
        rows.append(r)
    cols = ["Latency Limit (ms/token)", "Scenarios Compared",
            "AFD Wins - Layer-Summed", "AFD Wins - Per-Layer",
            "Median Throughput Ratio - Layer-Summed", "Median Throughput Ratio - Per-Layer",
            "Change in Median Ratio"]
    sheet(wb, "Summary", cols, rows,
          "Throughput ratio is AFD tokens/s/GPU divided by colocated tokens/s/GPU, each arm at "
          "its own best A:F split, attention sharding and batch on 72 GPUs, decode phase. "
          "AFD wins means ratio above 1. Layer-Summed is the earlier sweep, which added all 94 "
          "layers' expert counts before measuring imbalance; Per-Layer measures each layer's own "
          "busiest rank and averages, which is what a decode step pays. Nothing else differs.")


def sheet_models(wb, old, new):
    """Where the correction actually landed. This is the sheet to read first.

    The all-model median moves a long way, and almost none of that movement is
    in the models where AFD competes. Without this split a reader concludes
    "the correction hurt AFD" when what it did was widen a gap that was already
    decided, on the two models whose routing skew is assumed rather than
    recorded.
    """
    rows = []
    for model in sorted({c[0] for c in new}):
        o, n = [], []
        for cell in sorted(set(old) & set(new)):
            if cell[0] != model:
                continue
            a, c = best(old[cell]["afd"]), best(old[cell]["colo"])
            if a and c:
                o.append(a["tps"] / c["tps"])
            a, c = best(new[cell]["afd"]), best(new[cell]["colo"])
            if a and c:
                n.append(a["tps"] / c["tps"])
        if not n:
            continue
        rows.append({
            "Model": MODEL_NAME.get(model, model),
            "Expert Load Source": "recorded" if model == "qwen3_235b_fp4" else "assumed",
            "Scenarios": len(n),
            "AFD Wins - Layer-Summed": sum(1 for x in o if x > 1),
            "AFD Wins - Per-Layer": sum(1 for x in n if x > 1),
            "Median Throughput Ratio - Layer-Summed": round(st.median(o), 4),
            "Median Throughput Ratio - Per-Layer": round(st.median(n), 4),
            "Change in Median Ratio": round(st.median(n) - st.median(o), 4),
            "_fill": GOOD if model == "qwen3_235b_fp4" else None})
    rows.sort(key=lambda r: -r["AFD Wins - Per-Layer"])
    sheet(wb, "Model Breakdown",
          ["Model", "Expert Load Source", "Scenarios",
           "AFD Wins - Layer-Summed", "AFD Wins - Per-Layer",
           "Median Throughput Ratio - Layer-Summed", "Median Throughput Ratio - Per-Layer",
           "Change in Median Ratio"], rows,
          "Read this before the all-model Summary. The correction barely moves the three models "
          "where AFD competes; nearly all of the drop in the overall median is on the two "
          "DeepSeek models, which AFD already lost everywhere, and whose routing skew is assumed "
          "rather than recorded. The green row is the only model with a recorded distribution. "
          "No latency limit.",
          {"Model": 22, "Expert Load Source": 18})


def sheet_scenarios(wb, old, new):
    rows = []
    for cell in sorted(set(old) & set(new)):
        ao, co = best(old[cell]["afd"]), best(old[cell]["colo"])
        an, cn = best(new[cell]["afd"]), best(new[cell]["colo"])
        if not (ao and co and an and cn):
            continue
        ro, rn = ao["tps"] / co["tps"], an["tps"] / cn["tps"]
        verdict = ("AFD still ahead" if ro > 1 and rn > 1 else
                   "AFD lost its lead" if ro > 1 else
                   "AFD gained a lead" if rn > 1 else "colocated still ahead")
        rows.append({
            "Model": MODEL_NAME.get(cell[0], cell[0]),
            "Input Tokens": cell[1], "Output Tokens": cell[2], "MTP Draft Tokens": cell[3],
            "AFD Best Split": an["split"], "AFD Attention TP": an["tp"], "AFD Batch": an["batch"],
            "AFD Skew Coefficient": round(an["coeff"], 4),
            "Colocated Skew Coefficient": round(cn["coeff"], 4),
            "Throughput Ratio - Layer-Summed": round(ro, 4),
            "Throughput Ratio - Per-Layer": round(rn, 4),
            "Change": round(rn - ro, 4),
            "Verdict": verdict,
            "_fill": BAD if verdict == "AFD lost its lead" else
                     GOOD if verdict == "AFD gained a lead" else None})
    rows.sort(key=lambda r: r["Change"])
    cols = ["Model", "Input Tokens", "Output Tokens", "MTP Draft Tokens",
            "AFD Best Split", "AFD Attention TP", "AFD Batch",
            "AFD Skew Coefficient", "Colocated Skew Coefficient",
            "Throughput Ratio - Layer-Summed", "Throughput Ratio - Per-Layer",
            "Change", "Verdict"]
    sheet(wb, "Scenario Comparison", cols, rows,
          "One row per model x context x MTP setting, no latency limit, sorted by how much the "
          "correction moved the answer. The deployment columns describe the Per-Layer winner. "
          "Skew coefficient is the multiplier applied to the measured balanced MoE stage time.",
          {"Model": 20, "AFD Best Split": 18, "Verdict": 20})


def sheet_coefficients(wb, old_c, new_c):
    # {model: (per-layer counts, [layer-summed counts as a single layer])} so the
    # sheet can show both readings of the same recorded routing side by side.
    dists = {}
    for m, fn in (("qwen3_235b_fp4", "real_routing_qwen3_235b_perlayer.json"),
                  ("minimax_m25_fp4", "synthetic_perlayer_256.json"),
                  ("minimax_m3_fp4", "synthetic_perlayer_128.json"),
                  ("deepseek_v4_flash_fp4", "synthetic_perlayer_256.json"),
                  ("deepseek_v4_pro_fp4", "synthetic_perlayer_384.json")):
        p = W / fn
        if p.exists():
            d = json.loads(p.read_text())
            dists[m] = (d["counts_per_layer"], [d["counts_total"]])

    rows = []
    keys = set(old_c) | set(new_c)
    agg = defaultdict(lambda: {"old": [], "new": []})
    for k in keys:
        arm, kind, n, model = k
        agg[(arm, kind, n, model)]["old"] += old_c.get(k, [])
        agg[(arm, kind, n, model)]["new"] += new_c.get(k, [])
    for (arm, kind, n, model), v in sorted(agg.items(),
                                           key=lambda x: (x[0][0], x[0][3], -x[0][2])):
        if not v["new"]:
            continue
        ne = MODEL_EXPERTS.get(model)
        il = ia = None
        if model in dists and ne:
            il = imbalance(dists[model][0], n)
            ia = imbalance(dists[model][1], n)
        o, nn = sorted(v["old"]), sorted(v["new"])
        rows.append({
            "Arm": "AFD" if arm == "afd" else "Colocated",
            "Model": MODEL_NAME.get(model, model),
            "Expert-Parallel Ranks": n,
            "Experts per GPU": ne // n if ne else None,
            "Configurations": len(nn),
            "Load Imbalance - Layer-Summed": round(ia, 4) if ia else None,
            "Load Imbalance - Per-Layer": round(il, 4) if il else None,
            "Median Coefficient - Layer-Summed": round(st.median(o), 4) if o else None,
            "Median Coefficient - Per-Layer": round(st.median(nn), 4),
            "p90 Coefficient - Per-Layer": round(nn[9 * len(nn) // 10], 4),
            "Max Coefficient - Per-Layer": round(max(nn), 4)})
    cols = ["Arm", "Model", "Expert-Parallel Ranks", "Experts per GPU", "Configurations",
            "Load Imbalance - Layer-Summed", "Load Imbalance - Per-Layer",
            "Median Coefficient - Layer-Summed", "Median Coefficient - Per-Layer",
            "p90 Coefficient - Per-Layer", "Max Coefficient - Per-Layer"]
    sheet(wb, "Routing Skew Coefficient", cols, rows,
          "Load imbalance is the busiest rank's token count over the mean rank's. Coefficient is "
          "what that costs in time: 1 + f x (imbalance - 1), where f is the share of the measured "
          "stage time that moves with token count, fitted per configuration from the balanced "
          "sweep. Ranks are AFD's F count or the colocated arm's EP size. Only Qwen3's "
          "distribution is recorded; the rest are drawn at its measured per-layer skew.",
          {"Model": 20, "Arm": 12})


def sheet_tp(wb, old, new):
    rows = []
    for tag, cells in (("Layer-Summed", old), ("Per-Layer", new)):
        for label, sla in [(s, s) for s in SLA_LIMITS] + [("no limit", None)]:
            R, R1 = [], []
            for cell, d in cells.items():
                c = best(d["colo"], sla)
                a, a1 = best(d["afd"], sla), best(d["afd"], sla, tp=1)
                if not (a and c):
                    continue
                R.append(a["tps"] / c["tps"])
                if a1:
                    R1.append(a1["tps"] / c["tps"])
            if not R:
                continue
            rows.append({"Routing Skew Read As": tag,
                         "Latency Limit (ms/token)": label,
                         "Scenarios": len(R),
                         "AFD Wins - Any Attention TP": sum(1 for x in R if x > 1),
                         "AFD Wins - Attention TP 1 Only": sum(1 for x in R1 if x > 1),
                         "Median Ratio - Any Attention TP": round(st.median(R), 4),
                         "Median Ratio - Attention TP 1 Only": round(st.median(R1), 4) if R1 else None})
    cols = ["Routing Skew Read As", "Latency Limit (ms/token)", "Scenarios",
            "AFD Wins - Any Attention TP", "AFD Wins - Attention TP 1 Only",
            "Median Ratio - Any Attention TP", "Median Ratio - Attention TP 1 Only"]
    sheet(wb, "Attention TP Dependence", cols, rows,
          "How much of AFD's advantage needs attention tensor parallelism. FastAFD's attention TP "
          "duplicates work that does not shard -- the sweep already prices that in: at TP=2 the "
          "A-side layer time falls to 0.66-0.73x for the GQA models instead of 0.50x, and for the "
          "two MLA models it does not fall at all (0.98-1.35x). Attention TP 1 Only restricts the "
          "AFD arm to unsharded attention; the colocated arm is unrestricted in both columns.",
          {"Routing Skew Read As": 20})

    # the concrete list, on the corrected data
    lst = []
    for cell, d in new.items():
        a1, c = best(d["afd"], tp=1), best(d["colo"])
        if not (a1 and c) or a1["tps"] / c["tps"] <= 1:
            continue
        a = best(d["afd"])
        lst.append({"Model": MODEL_NAME.get(cell[0], cell[0]),
                    "Input Tokens": cell[1], "Output Tokens": cell[2], "MTP Draft Tokens": cell[3],
                    "AFD Split": a1["split"], "AFD Batch": a1["batch"],
                    "AFD Latency (ms/token)": round(a1["tpot"], 2),
                    "Colocated Batch": c["batch"],
                    "Colocated Latency (ms/token)": round(c["tpot"], 2),
                    "Throughput Ratio - Attention TP 1": round(a1["tps"] / c["tps"], 4),
                    "Throughput Ratio - Best Attention TP": round(a["tps"] / c["tps"], 4),
                    "Best Attention TP": a["tp"]})
    lst.sort(key=lambda r: -r["Throughput Ratio - Attention TP 1"])
    sheet(wb, "AFD Wins Without Attention TP",
          ["Model", "Input Tokens", "Output Tokens", "MTP Draft Tokens", "AFD Split", "AFD Batch",
           "AFD Latency (ms/token)", "Colocated Batch", "Colocated Latency (ms/token)",
           "Throughput Ratio - Attention TP 1", "Throughput Ratio - Best Attention TP",
           "Best Attention TP"], lst,
          "Scenarios where AFD beats colocated with attention tensor parallelism switched off "
          "entirely, on the per-layer skew. These are the configurations that avoid FastAFD's "
          "attention-TP duplication altogether. No latency limit.",
          {"Model": 20, "AFD Split": 16})


def sheet_notes(wb):
    ws = wb.create_sheet("Method Notes")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 118
    r = 1

    def head(t):
        nonlocal r
        c = ws.cell(r, 1, t)
        c.fill = HDR
        c.font = Font(bold=True, color="FFFFFF", size=11)
        ws.cell(r, 2).fill = HDR
        r += 1

    def item(k, v):
        nonlocal r
        ws.cell(r, 1, k).font = Font(bold=True, size=10)
        ws.cell(r, 1).alignment = Alignment(vertical="top")
        c = ws.cell(r, 2, v)
        c.font = Font(size=10)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 14 * (1 + len(v) // 118)
        r += 1

    head("What changed between the two sweeps")
    item("The one difference", "How the expert-load imbalance is read out of the recorded routing. "
         "Everything else is identical: the same measured MegaMoE stage times, the same f, the "
         "same formula, the same grid of 49632 configurations.")
    item("Layer-Summed (before)", "Add all 94 layers' per-expert token counts into one vector, "
         "then take the busiest rank over the mean rank. Qwen3 at 32 ranks: 1.31.")
    item("Per-Layer (now)", "Take the busiest rank over the mean rank within each layer, then "
         "average over layers. Qwen3 at 32 ranks: 2.90.")
    item("Why the first one was wrong", "A decode step costs the sum over layers of each layer's "
         "own slowest rank. The slowest rank is a different rank on nearly every layer -- in the "
         "recorded Qwen3 run only 38% of layers shared a straggler at 4 ranks, against 25% by "
         "chance, and 10% at 32 ranks. Summing the layers first lets one layer's peak cancel "
         "another's trough, and reports a machine more balanced than any step actually runs.")
    item("Size of the error", "The excess load was understated 2.6x at 4 ranks, 5.6x at 8, 6.8x "
         "at 16, 9.1x at 64, 11.6x at 128. Finer shards were understated more, and the AFD arm "
         "runs finer shards than the colocated arm, so the correction costs AFD more.")

    head("Where the numbers come from")
    item("MoE stage time", "Measured on GB200 NVL72 with the MegaMoE kernel: 4214 AFD-split runs "
         "and 1430 colocated runs, 30 warmups and 40 timed iterations each, allocation pinned "
         "inside one NVLink domain. Max across ranks per iteration, so straggler cost is in.")
    item("Routing distribution", "Qwen3-235B: recorded from a real FastAFD serving run, 94 layers "
         "x 128 experts x 12 attention DP ranks, 15.2M token-to-expert assignments, decode only. "
         "The other four models: 94 independent layers drawn lognormal at Qwen3's measured "
         "per-layer CV of 136%. Those four are assumptions and are labelled as such.")
    item("Coefficient", "1 + f x (imbalance - 1). f is the share of the stage time that moves "
         "with token count, fitted per configuration from the balanced sweep as t = fixed + "
         "variable x batch. Checked against 39 paired GB200 measurements: median error +0.003, "
         "mean absolute 0.020.")
    item("Applied to both arms", "AFD and colocated both carry the correction, each at its own "
         "shard count -- AFD's F, the colocated arm's EP.")

    head("Read this first")
    item("Only one model's skew is recorded", "Qwen3-235B's routing was captured from a real "
         "FastAFD serving run. MiniMax-M2.5, MiniMax-M3, DeepSeek-V4-Flash and "
         "DeepSeek-V4-Pro have no recorded routing; each is given 94 independent layers "
         "drawn at Qwen3's measured per-layer skew. Their coefficients are assumptions.")
    item("Where the correction landed", "The all-model median ratio falls 0.91 to 0.82, but "
         "Qwen3 and MiniMax-M2.5 -- the models where AFD actually competes -- barely move "
         "(1.0005 to 1.0043 and 1.0327 to 1.0328). Nearly the whole drop is on the two "
         "DeepSeek models, which AFD already lost in every scenario, and whose skew is "
         "assumed. Do not read the DeepSeek movement as a measured result. See Model Breakdown.")
    item("Why assumed skew moves more", "Independent synthetic layers have no correlation "
         "across layers, so their layer-summed reading is flatter than a real router's "
         "(13-14% CV against Qwen3's 25%), which makes the per-layer correction look "
         "larger: 8-13x for DeepSeek against 2.6-6.8x for Qwen3.")

    head("What this still does not capture")
    item("Pipeline desynchronisation", "AIC models the A and F pools as cleanly overlapped. A "
         "straggler that moves every layer breaks that overlap, and the cost of lost overlap is "
         "more than the stage-time increase. A multiplier on stage time cannot express it. This "
         "is the largest known remaining gap, and it points the same way: against AFD.")
    item("Bucket padding", "The recorded run used token buckets 8/16/32/48/64. If the kernel pads "
         "the busiest rank up to the next bucket the cost is a step, not the straight line f "
         "assumes, so cells near a bucket edge are still understated.")
    item("Extrapolation of f", "f is fitted on a balanced grid where every rank carries the mean. "
         "The busiest rank carries 2-3x that, outside the fitted range.")
    item("Static replay", "The paired kernel measurements that validated f used one fixed "
         "distribution, so the same rank was hot every iteration. A real run rotates the hot rank "
         "and pays a resynchronisation each time.")

    head("Reading the throughput ratio")
    item("Definition", "AFD tokens/s/GPU divided by colocated tokens/s/GPU, decode phase, each "
         "arm at its own best A:F split, attention sharding and batch within 72 GPUs, and charged "
         "over every GPU in the deployment including AFD's FFN GPUs.")
    item("Above 1", "AFD serves more tokens per GPU-second than colocated at that model, context "
         "and latency limit.")


def main():
    old, old_c = load(OLD)
    new, new_c = load(NEW)
    print(f"layer-summed: {len(old)} cells from {OLD.name}")
    print(f"per-layer:    {len(new)} cells from {NEW.name}")

    wb = Workbook()
    wb.remove(wb.active)
    sheet_summary(wb, old, new)
    sheet_models(wb, old, new)
    sheet_scenarios(wb, old, new)
    sheet_coefficients(wb, old_c, new_c)
    sheet_tp(wb, old, new)
    sheet_notes(wb)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print("sheets:", wb.sheetnames)


if __name__ == "__main__":
    main()
