#!/usr/bin/env python3
"""AIC sweep result, in as few columns as the answer actually needs.

build_gb200_style.py reproduces the v0514 layout: 11 sheets, 30-60 columns,
several of them different slices of the same rows. That is the right shape for
auditing and the wrong shape for reading. This is the reading version -- three
sheets, and every column earns its place.

Nothing is recomputed here: the winner selection, the latency band and the
verdict are imported from build_gb200_style so the two workbooks cannot drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLDIR))
sys.path.insert(0, str(TOOLDIR.parent / "afd_megamoe_b200"))

import os
os.environ.setdefault("SKIP_CAPACITY_WALL", "")

from openpyxl import Workbook                                    # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill         # noqa: E402
from openpyxl.utils import get_column_letter                     # noqa: E402

import build_gb200_style as B                                    # noqa: E402

OUT = TOOLDIR / "handoff" / "AIC_AFD_vs_Colocated_GB200.xlsx"
FONT = "Arial"
HDR = PatternFill("solid", start_color="1F3864")
HDRF = Font(name=FONT, bold=True, color="FFFFFF", size=10)
BODY = Font(name=FONT, size=10)
TITLE = Font(name=FONT, bold=True, size=11, color="1F3864")
GAIN = PatternFill("solid", start_color="C6EFCE")
LOSS = PatternFill("solid", start_color="FFC7CE")
EDGE = PatternFill("solid", start_color="FFF2CC")

# (column, one-line note). The note is the whole point: a column nobody can
# explain in one line does not belong in a reading copy.
MAIN = [
    ("model", ""),
    ("isl", "input length"),
    ("osl", "generation length"),
    ("draft_len", "speculative draft length (nextN)"),
    ("tpot_limit_ms", "per-token latency limit; blank = unlimited"),
    ("afd_config", "winning AFD deployment: A:F split, attention TP, microbatches"),
    ("afd_batch", "sequences per attention rank"),
    ("afd_tpot_ms", "AIC time per output token"),
    ("afd_tok_s_gpu", "AIC tokens/s per GPU across the 72-GPU domain"),
    ("colo_config", "winning colocated deployment: EP group, attention TP"),
    ("colo_batch", "logical tokens per rank"),
    ("colo_tpot_ms", "AIC time per output token"),
    ("colo_tok_s_gpu", "AIC tokens/s per GPU across the 72-GPU domain"),
    ("ratio", "afd_tok_s_gpu / colo_tok_s_gpu. >1 means AFD ahead"),
    ("ratio_worst_in_band", "the same ratio at the worst point of a 0.9x-1.2x latency band"),
    ("grid_inflation", "how much the measured batch ladder flatters the comparison; >1.25 means overstated"),
    ("verdict", "gain / gain but ladder-inflated / threshold noise / no gain"),
]

SPLIT = [
    ("model", ""),
    ("isl", "input length"),
    ("draft_len", "speculative draft length (nextN)"),
    ("afd_config", "A:F split, attention TP, microbatches"),
    ("A_gpus", "attention GPUs"),
    ("F_gpus", "expert GPUs"),
    ("afd_batch", "sequences per attention rank"),
    ("afd_tpot_ms", "AIC time per output token"),
    ("afd_tok_s_gpu", "AIC tokens/s per GPU across the 72-GPU domain"),
    ("moe_stage_ms", "the MEASURED MegaMoE stage injected for this row"),
    ("moe_provenance", "which measurement, exactly"),
    ("ratio_vs_best_colo", "against the best colocated deployment for the same context"),
]


def _sheet(wb, name, cols, rows, caption):
    ws = wb.create_sheet(name)
    ws.cell(row=1, column=1, value=caption).font = TITLE
    ws.cell(row=1, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=min(len(cols), 10))
    ws.row_dimensions[1].height = 44
    for c, (h, note) in enumerate(cols, 1):
        ws.cell(row=2, column=c, value=note).font = Font(name=FONT, size=8, italic=True,
                                                         color="666666")
        ws.cell(row=2, column=c).alignment = Alignment(wrap_text=True, vertical="top")
        cell = ws.cell(row=3, column=c, value=h)
        cell.fill, cell.font = HDR, HDRF
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[2].height = 40
    names = [h for h, _ in cols]
    for r, row in enumerate(rows, 4):
        for c, h in enumerate(names, 1):
            v = row.get(h)
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = BODY
            if isinstance(v, float):
                cell.number_format = "0.000"
            if h == "verdict":
                cell.fill = (GAIN if v == "gain" else
                             EDGE if v in ("gain, ladder-inflated", "threshold noise") else LOSS)
    for c, (h, _) in enumerate(cols, 1):
        w = 26 if h in ("afd_config", "colo_config", "moe_provenance", "verdict") else \
            max(10, min(len(h) + 3, 18))
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:{get_column_letter(len(cols))}{3 + len(rows)}"



def _cached_inflation(cell, a, c):
    """rung_inflation from the precomputed wall cache ONLY.

    B.capacity_walls falls through to computing a missing wall live, which needs
    AIC and therefore a compute node. The cache was built for osl=1024 winners,
    so other osl values are simply absent -- report that as blank rather than
    either crashing or silently substituting. A blank grid_inflation means "not
    computed", never "no inflation".
    """
    import json as _j
    model, isl, osl, mtp = cell
    if not (a and c and a.get("conc") and c.get("conc")):
        return None
    ck = ("colo", model, isl, osl, mtp, c["unit_gpus"], c["tp"])
    g = a["a_tp"] if a["a_tp"] in (1, 2, 4) else 1
    while g > 1 and (a["a"] % g or a["fq"] % g):
        g //= 2
    ak = ("afd", model, isl, osl, mtp, a["a"], a["fq"], a["a_tp"], max(g, 1), a.get("mb", 1))
    cv = B._WALL_DISK.get(_j.dumps(list(ck)))
    av = B._WALL_DISK.get(_j.dumps(list(ak)))
    if not (isinstance(cv, list) and isinstance(av, list)):
        return None
    colo_at_wall = cv[1] / c["unit_gpus"]
    afd_at_wall = av[1] / a["unit_gpus"]
    if not (colo_at_wall and afd_at_wall):
        return None
    r_rungs = (a["conc"] / a["unit_gpus"]) / (c["conc"] / c["unit_gpus"])
    r_wall = afd_at_wall / colo_at_wall
    return r_rungs / r_wall if r_wall else None


def main():
    cells = B.load()
    main_rows, split_rows = [], []

    for cell in sorted(cells, key=lambda c: (B.MODEL_LABEL[c[0]], c[3], c[2], c[1])):
        d = cells[cell]
        model, isl, osl, mtp = cell
        for sla in [None] + list(B.SLA_LIMITS):
            a = B.best(d["afd"], d["oom"], sla)
            c = B.best(d["colo"], d["oom"], sla)
            if not a and not c:
                continue
            base = (a["tpsgpu"] / c["tpsgpu"]) if (a and c and c["tpsgpu"]) else None
            band = []
            if sla is not None:
                for k in B.SLA_BAND:
                    ak = B.best(d["afd"], d["oom"], sla * k)
                    ck = B.best(d["colo"], d["oom"], sla * k)
                    if ak and ck and ck["tpsgpu"]:
                        band.append(ak["tpsgpu"] / ck["tpsgpu"])
            lo = min(band) if band else None
            hi = max(band) if band else None
            infl = _cached_inflation(cell, a, c)
            if base is None:
                v = "one arm infeasible"
            elif lo is None:
                v = "no band" if sla is not None else ("gain" if base > 1.10 else "no gain")
            elif lo <= 1.10 < (hi or lo):
                v = "threshold noise"
            elif lo <= 1.10:
                v = "no gain"
            elif infl is not None and infl >= 1.25:
                v = "gain, ladder-inflated"
            else:
                v = "gain"
            main_rows.append({
                "model": B.MODEL_LABEL[model], "isl": isl, "osl": osl, "draft_len": mtp,
                "tpot_limit_ms": sla,
                "afd_config": a["topo"] if a else None,
                "afd_batch": a["batch"] if a else None,
                "afd_tpot_ms": round(a["tpot_ms"], 3) if a else None,
                "afd_tok_s_gpu": a["tpsgpu72"] if a else None,
                "colo_config": c["topo"] if c else None,
                "colo_batch": c["batch"] if c else None,
                "colo_tpot_ms": round(c["tpot_ms"], 3) if c else None,
                "colo_tok_s_gpu": c["tpsgpu72"] if c else None,
                "ratio": round(base, 3) if base else None,
                "ratio_worst_in_band": round(lo, 3) if lo else None,
                "grid_inflation": round(infl, 3) if infl else None,
                "verdict": v,
            })

        # per-split view, no latency limit, so the split space is visible
        cbest = B.best(d["colo"], d["oom"], None)
        by_f = {}
        for x in d["afd"]:
            k = (x["a"], x["fq"], x["a_tp"], x.get("mb", 1))
            if k not in by_f or x["tpsgpu"] > by_f[k]["tpsgpu"]:
                by_f[k] = x
        for k, x in sorted(by_f.items()):
            split_rows.append({
                "model": B.MODEL_LABEL[model], "isl": isl, "draft_len": mtp,
                "afd_config": x["topo"], "A_gpus": x["a"], "F_gpus": x["fq"],
                "afd_batch": x["batch"], "afd_tpot_ms": round(x["tpot_ms"], 3),
                "afd_tok_s_gpu": x["tpsgpu72"],
                "moe_stage_ms": round(x["moe_ms"], 3) if x.get("moe_ms") else None,
                "moe_provenance": x.get("prov"),
                "ratio_vs_best_colo": round(x["tpsgpu"] / cbest["tpsgpu"], 3)
                if cbest and cbest["tpsgpu"] else None,
            })

    wb = Workbook(); wb.remove(wb.active)
    ws = wb.create_sheet("README")
    for i, (t, f) in enumerate([
        ("AFD vs colocated MoE on GB200 NVL72 — AIC simulation with measured MegaMoE injected", TITLE),
        ("", BODY),
        ("The MoE stage is a hardware measurement on GB200 (lyris). Attention, KV and memory", BODY),
        ("come from AIC's gb200 model. BOTH arms receive the same measured kernel.", BODY),
        ("Raw measurements are in MegaMoE_GB200_measurements.xlsx; this file is what AIC did", BODY),
        ("with them.", BODY),
        ("", BODY),
        ("How to read Comparison", TITLE),
        ("  `ratio` is the answer; the two columns after it decide whether to believe it.", BODY),
        ("  ratio_worst_in_band — throughput is a STAIRCASE in the latency limit, so a gain", BODY),
        ("    that appears at one threshold and vanishes at the next is an artifact. This is", BODY),
        ("    the ratio at the worst point of a 0.9x-1.2x band around the limit.", BODY),
        ("  grid_inflation — the measured batch ladder quantises capacity. This is how much", BODY),
        ("    it flatters the comparison, measured against the true memory wall. >1.25 means", BODY),
        ("    the gain is real but overstated.", BODY),
        ("  verdict combines them. Only `gain` is admitted without qualification.", BODY),
        ("", BODY),
        ("Scope — read before quoting any number", TITLE),
        ("  BALANCED ROUTING ONLY. Every injected measurement uses perfectly uniform expert", BODY),
        ("  load (dispersion exactly 0, generated arithmetically, no real router involved).", BODY),
        ("  Production routing is learned and skewed. Measured worst-case skew costs AFD", BODY),
        ("  about twice what it costs colocated (median 1.98x against 1.09x on the MoE", BODY),
        ("  stage), so the real comparison is less favourable to AFD than this file shows.", BODY),
        ("", BODY),
        ("  68A4F has no attention-TP rows: FastAFD's launcher needs nnodes*4 == world_size,", BODY),
        ("  and TP=2/TP=4 at F=4 imply 38 and 21 GPUs. Not a choice, a measurement gap.", BODY),
        ("", BODY),
        ("  MiniMax-M3's attention has no silicon rows (MSA); it is SOL plus an empirical", BODY),
        ("  factor and is 42-61% of its decode step. Its numbers are softer than the rest.", BODY),
        ("", BODY),
        ("Validation", TITLE),
        ("  A real FastAFD serving run at N=4 measured 32.216 ms per decode step against", BODY),
        ("  FastAFD's own published 32.826 ms — 1.9% apart. AIC predicts 27.896 ms for the", BODY),
        ("  same configuration, i.e. 15.5% optimistic.", BODY),
    ], 1):
        ws.cell(row=i, column=1, value=t).font = f
    ws.column_dimensions["A"].width = 98

    _sheet(wb, "Comparison", MAIN, main_rows,
           "One row per (model, context, latency limit). Each arm at its own best deployment "
           "and batch on the 72-GPU domain. Blank tpot_limit_ms = no latency constraint.")
    _sheet(wb, "AFD_by_split", SPLIT, split_rows,
           "Every A:F split at its own best batch, no latency limit, with the measured MoE "
           "stage that was injected. Shows which split the search picked and why.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}  (Comparison {len(main_rows)}, AFD_by_split {len(split_rows)})")


if __name__ == "__main__":
    main()
