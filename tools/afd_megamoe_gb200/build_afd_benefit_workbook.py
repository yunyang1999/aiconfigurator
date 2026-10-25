#!/usr/bin/env python3
"""AFD vs colocated on GB200 NVL72, priced with real expert routing.

Both arms are DECODE workers: every latency and throughput here is a decode
step, not a prefill chunk and not an end-to-end average.

Every MoE stage time carries the cost of real, skewed routing. The kernel
benchmark that produced the stage times routes tokens uniformly, which no
router does; the correction is measured, and the column that carries it is
visible in the split sheet so a reader can see which rows got one.

Column names are written out. A reader should not have to hold a legend in
their head to know whether a number is a latency or a rate, or which arm it
belongs to.

Selection logic is imported from build_gb200_style rather than repeated, so
this workbook and the audit workbook cannot disagree about who won.
"""
from __future__ import annotations

import json
import re
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
import build_clean_workbook as C                                 # noqa: E402
from afd_paths import data_root

OUT = TOOLDIR / "handoff" / "AFD_vs_Colocated_GB200_NVL72.xlsx"
ROUTING_FILES = [
    # (label, path, whether the load was recorded rather than assumed)
    ("Qwen3-235B-A22B", "real_routing_qwen3_235b.json", True),
    ("MiniMax-M2.5", "synthetic_cvmeasured_256.json", False),
    ("MiniMax-M3", "synthetic_cvmeasured_128.json", False),
    ("DeepSeek-V4-Flash", "synthetic_cvmeasured_256.json", False),
    ("DeepSeek-V4-Pro", "synthetic_cvmeasured_384.json", False),
]
# Data root: the unpacked data bundle, from AFD_MEGAMOE_ROOT (required).
MEGAMOE = data_root()
MEASURED = MEGAMOE / "measured_imbalance_coefficients.json"

FONT = "Arial"
HDR = PatternFill("solid", start_color="1F3864")
HDRF = Font(name=FONT, bold=True, color="FFFFFF", size=10)
BODY = Font(name=FONT, size=10)
TITLE = Font(name=FONT, bold=True, size=11, color="1F3864")
GAIN = PatternFill("solid", start_color="C6EFCE")
LOSS = PatternFill("solid", start_color="FFC7CE")
EDGE = PatternFill("solid", start_color="FFF2CC")

_MULT = re.compile(r"x(\d+\.\d+)")

# (key, printed heading, one-line note). A column that cannot be explained in
# one line does not belong in a workbook meant to be read.
COMPARISON = [
    ("model",            "Model", ""),
    ("isl",              "Input Length (tokens)", ""),
    ("osl",              "Output Length (tokens)", ""),
    ("draft",            "Speculative Draft Length", "0 = speculation off"),
    ("limit",            "Decode Latency Limit (ms/token)", "blank = no limit imposed"),
    ("afd_cfg",          "AFD Deployment", "attention:expert GPU split, attention TP, microbatches"),
    ("afd_batch",        "AFD Batch (sequences per attention GPU)", ""),
    ("afd_tpot",         "AFD Decode Latency (ms/token)", ""),
    ("afd_tps",          "AFD Decode Throughput (tokens/s/GPU)", "averaged over all 72 GPUs, idle ones included"),
    ("colo_cfg",         "Colocated Deployment", "expert-parallel group size, attention TP"),
    ("colo_batch",       "Colocated Batch (tokens per GPU)", ""),
    ("colo_tpot",        "Colocated Decode Latency (ms/token)", ""),
    ("colo_tps",         "Colocated Decode Throughput (tokens/s/GPU)", "averaged over all 72 GPUs, idle ones included"),
    ("ratio",            "AFD Throughput Advantage", "AFD divided by colocated; above 1.00 means AFD is ahead"),
    ("ratio_worst",      "Advantage at Worst Point in Latency Band", "same ratio re-derived across a 0.90x-1.20x sweep of the limit"),
    ("inflation",        "Batch Ladder Inflation", "how much the discrete batch rungs flatter AFD; at or above 1.25 the advantage is a rung artefact"),
    ("verdict",          "Assessment", ""),
]

SPLITS = [
    ("model",            "Model", ""),
    ("isl",              "Input Length (tokens)", ""),
    ("draft",            "Speculative Draft Length", ""),
    ("afd_cfg",          "AFD Deployment", ""),
    ("a_gpus",           "Attention GPUs", ""),
    ("f_gpus",           "Expert GPUs", ""),
    ("shard",            "Experts per Expert GPU", "what decides how much routing skew a single GPU absorbs"),
    ("afd_batch",        "Batch (sequences per attention GPU)", ""),
    ("afd_tpot",         "Decode Latency (ms/token)", ""),
    ("afd_tps",          "Decode Throughput (tokens/s/GPU)", ""),
    ("moe_ms",           "MoE Stage (ms)", "dispatch, expert compute and combine, for all MoE layers"),
    ("imb",              "Routing Imbalance Factor", "what real skewed routing costs over uniform routing; blank = not corrected"),
    ("moe_src",          "MoE Stage Source", ""),
    ("vs_colo",          "Advantage vs Best Colocated", ""),
]

COEFF_COLS = [
    ("arm",      "Deployment Style", "split = AFD, colocated = experts on the attention GPUs"),
    ("shard",    "Experts per GPU", ""),
    ("tpe",      "Tokens per Expert", "what decides how many GEMM tiles an expert needs"),
    ("skew",     "Expert Load Source", "recorded from a serving run, or synthetic at a stated skew"),
    ("bal",      "Uniform Load (ms)", "same sampler, flat expert probabilities"),
    ("rep",      "Skewed Load (ms)", "same sampler, the skewed load in the previous column"),
    ("coef",     "Cost of Skewed Routing", "skewed divided by uniform; above 1.00 means skew costs time"),
    ("rounds",   "Paired Rounds", "alternating rounds inside one allocation"),
]

ROUTING_COLS = [
    ("model",   "Model", ""),
    ("source",  "Expert Load Source", "recorded from a serving run, or synthetic at a stated skew"),
    ("experts", "Routed Experts", ""),
    ("peak",    "Busiest Expert vs Average", "per-expert skew, before any grouping onto GPUs"),
    ("cv",      "Expert Load Variation (%)", "coefficient of variation across experts"),
    ("s64",     "Load Imbalance at 64 Experts/GPU", "max over mean of per-GPU load; a LOAD ratio, not a time ratio"),
    ("s32",     "Load Imbalance at 32 Experts/GPU", ""),
    ("s16",     "Load Imbalance at 16 Experts/GPU", ""),
    ("s8",      "Load Imbalance at 8 Experts/GPU", ""),
    ("s4",      "Load Imbalance at 4 Experts/GPU", ""),
]


_EXPERTS = None


def _experts():
    """Routed-expert count per model, from the measurement table itself rather
    than a second copy that could drift from it."""
    global _EXPERTS
    if _EXPERTS is None:
        from megamoe_table import MeasuredStage
        _EXPERTS = dict(MeasuredStage().experts)
    return _EXPERTS


def _imbalance_factor(prov: str):
    """Pull the applied factor out of a provenance string, or None."""
    # Match on "routing skew", which is what the table writes. This matched
    # "real routing" until the provenance was reworded to distinguish recorded
    # from assumed skew, and the rename silently zeroed the column rather than
    # failing -- the workbook reported 0 corrections while 18780 rows carried one.
    if not prov or "routing skew" not in prov:
        return None
    m = _MULT.search(prov)
    return round(float(m.group(1)), 4) if m else None


def _num(v, places=3):
    """A plain Python float, rounded, or None.

    Written at the SOURCE rather than at the cell, because rounding at the cell
    did not stick: the values arrive as numpy scalars, which satisfy
    isinstance(v, float) and still serialise as 8.239000000000001 for a number
    already rounded to three places. float() breaks that.
    """
    if v is None:
        return None
    try:
        return round(float(v), places)
    except (TypeError, ValueError):
        return v


def _round_cell(v):
    """Numbers rounded on the way into the sheet, whatever type they arrive as.

    An isinstance(v, float) guard was not enough: values reach here from CSV
    parsing and arithmetic and some are numeric STRINGS, which slipped past it
    and landed in the sheet as 8.239000000000001. Reading as false precision is
    the visible symptom; the real problem is that the guard silently did nothing
    for 1021 cells while appearing to work.
    """
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int,)):
        return v
    # float(v) first, deliberately. Values arrive as numpy scalars from the CSV
    # reader, and those pass isinstance(v, float) yet serialise with the full
    # binary expansion -- 8.239000000000001 for a value already rounded to three
    # decimals. Converting to a plain Python float is what makes round() stick.
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return v


def _sheet(wb, name, cols, rows, caption):
    ws = wb.create_sheet(name)
    ws.cell(row=1, column=1, value=caption).font = TITLE
    for i, (_, head, note) in enumerate(cols, 1):
        c = ws.cell(row=3, column=i, value=head)
        c.fill, c.font = HDR, HDRF
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        if note:
            ws.cell(row=2, column=i, value=note).font = Font(
                name=FONT, size=8, italic=True, color="808080")
            ws.cell(row=2, column=i).alignment = Alignment(wrap_text=True, vertical="bottom")
    for r, row in enumerate(rows, 4):
        for i, (key, _, _) in enumerate(cols, 1):
            v = _round_cell(row.get(key))
            c = ws.cell(row=r, column=i, value=v)
            c.font = BODY
            # Display format as well as stored value: the reader should never be
            # shown more precision than the measurement carries, whatever the
            # float happens to serialise as.
            if isinstance(v, float):
                # This is what the reader sees. openpyxl serialises floats with
                # %.16g, so the stored text can read 8.239000000000001 for a
                # value of 8.239; the format code is what Excel renders.
                c.number_format = "0.000" if abs(v) < 100 else "0.0"
        v = row.get("verdict")
        if v:
            fill = {"gain": GAIN, "gain, ladder-inflated": EDGE,
                    "threshold noise": EDGE}.get(v, LOSS if v == "no gain" else None)
            if fill:
                ws.cell(row=r, column=len(cols)).fill = fill
    ws.row_dimensions[2].height = 30
    ws.row_dimensions[3].height = 44
    for i, (_, head, _) in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(i)].width = min(34, max(12, len(head) * 0.62 + 6))
    ws.freeze_panes = "A4"
    return ws


def _coeff_rows():
    """The paired measurements the correction is read from.

    Worth a sheet of its own because it overturns the obvious expectation.
    Uniform routing is not a neutral baseline: grouped-GEMM work is quantised
    into tiles, so a uniform load sitting just past a tile boundary pays for a
    whole extra tile on every expert, while a skewed load of the same mean drops
    many experts back under it. At 72 tokens per expert the real router measured
    FASTER than uniform; by 192 it measured slower.
    """
    try:
        rows = json.loads(MEASURED.read_text())["rows"]
    except (OSError, KeyError, ValueError):
        return []
    out = []
    for r in sorted(rows, key=lambda x: (x["arm"], -x["experts_per_gpu"], x["tokens_per_expert"])):
        raw = r.get("skew", "real")
        out.append({"arm": "split" if r["arm"] == "afd" else "colocated",
                    "skew": ("recorded" if raw == "real"
                             else raw.replace("synthetic ", "synthetic, ")),
                    "shard": r["experts_per_gpu"],
                    "tpe": round(r["tokens_per_expert"]),
                    "bal": r["uniform_ms"], "rep": r["replay_ms"],
                    "coef": r["coefficient"], "rounds": r.get("rounds")})
    return out


def _routing_rows():
    """One row per model, saying which expert load its coefficients rest on.

    Every model gets a row, not just the recorded one. 1348 of the corrected
    rows in this workbook rest on an assumed skew, and a reader cannot judge
    them without seeing what was assumed.

    The imbalance columns are LOAD ratios -- what share of the work the busiest
    GPU holds. They are not time ratios and are much larger than the measured
    cost; see Cost of Routing Skew for what the kernel actually pays.
    """
    out = []
    for label, fname, recorded in ROUTING_FILES:
        try:
            d = json.loads((MEGAMOE / fname).read_text())
        except OSError:
            continue
        counts = d["counts_total"]
        ne = len(counts)
        total = sum(counts)
        mean = total / ne
        var = sum((c - mean) ** 2 for c in counts) / ne
        cv = 100 * var ** 0.5 / mean if mean else 0.0

        def imb(per_gpu):
            if per_gpu > ne or ne % per_gpu:
                return None
            g = ne // per_gpu
            loads = [sum(counts[i * per_gpu:(i + 1) * per_gpu]) for i in range(g)]
            m = sum(loads) / g
            return round(max(loads) / m, 4) if m else None

        out.append({
            "model": label,
            "source": ("recorded from a serving run" if recorded
                       else f"synthetic, shaped to {cv:.1f}% variation"),
            "experts": ne,
            "peak": round(max(counts) / mean, 4) if mean else None,
            "cv": round(cv, 2),
            "s64": imb(64), "s32": imb(32), "s16": imb(16),
            "s8": imb(8), "s4": imb(4),
        })
    return out


def main():
    cells = B.load()
    comparison, splits = [], []

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
            lo, hi = (min(band), max(band)) if band else (None, None)
            infl = C._cached_inflation(cell, a, c)
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
            comparison.append({
                "model": B.MODEL_LABEL[model], "isl": isl, "osl": osl, "draft": mtp,
                "limit": sla,
                "afd_cfg": a["topo"] if a else None,
                "afd_batch": a["batch"] if a else None,
                "afd_tpot": _num(a["tpot_ms"]) if a else None,
                "afd_tps": _num(a["tpsgpu72"]) if a else None,
                "colo_cfg": c["topo"] if c else None,
                "colo_batch": c["batch"] if c else None,
                "colo_tpot": _num(c["tpot_ms"]) if c else None,
                "colo_tps": _num(c["tpsgpu72"]) if c else None,
                "ratio": _num(base) if base else None,
                "ratio_worst": _num(lo) if lo else None,
                "inflation": _num(infl) if infl else None,
                "verdict": v,
            })

        cbest = B.best(d["colo"], d["oom"], None)
        by_f = {}
        for x in d["afd"]:
            k = (x["a"], x["fq"], x["a_tp"], x.get("mb", 1))
            if k not in by_f or x["tpsgpu"] > by_f[k]["tpsgpu"]:
                by_f[k] = x
        experts = _experts().get(model)
        for k, x in sorted(by_f.items()):
            splits.append({
                "model": B.MODEL_LABEL[model], "isl": isl, "draft": mtp,
                "afd_cfg": x["topo"], "a_gpus": x["a"], "f_gpus": x["fq"],
                "shard": (experts // x["fq"]) if experts and x["fq"] else None,
                "afd_batch": x["batch"], "afd_tpot": _num(x.get("tpot_ms")),
                "afd_tps": _num(x.get("tpsgpu72")),
                "moe_ms": _num(x.get("moe_ms")),
                "imb": _imbalance_factor(x.get("prov", "")),
                "moe_src": x.get("prov"),
                "vs_colo": _num(x["tpsgpu"] / cbest["tpsgpu"])
                if cbest and cbest["tpsgpu"] else None,
            })

    wb = Workbook()
    wb.remove(wb.active)
    _sheet(wb, "AFD vs Colocated", COMPARISON, comparison,
           "Best AFD deployment against best colocated deployment, per model, "
           "context and decode latency limit. Both arms are decode workers.")
    _sheet(wb, "AFD Split Detail", SPLITS, splits,
           "Every attention:expert split at its own best batch, no latency limit, "
           "with the MoE stage time that produced the row.")
    rr = _routing_rows()
    if rr:
        _sheet(wb, "Expert Routing", ROUTING_COLS, rr,
               "Expert load recorded from a real serving run, and the per-GPU "
               "imbalance it implies at each expert shard size.")
    cr = _coeff_rows()
    if cr:
        _sheet(wb, "Cost of Routing Skew", COEFF_COLS, cr,
               "Paired measurements of the same MoE stage under uniform and "
               "skewed expert load, alternating inside one allocation. Expert "
               "Load Source says whether the skew was recorded or assumed.")
    _readme(wb, comparison, splits)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    corrected = sum(1 for s in splits if s["imb"])
    print(f"wrote {OUT}")
    print(f"  comparison rows {len(comparison)}, split rows {len(splits)}, "
          f"of which {corrected} carry a routing-imbalance correction")


def _readme(wb, comparison, splits):
    ws = wb.create_sheet("Notes")
    corrected = sum(1 for s in splits if s["imb"])
    factors = [s["imb"] for s in splits if s["imb"]]
    lines = [
        ("What this workbook answers", TITLE),
        ("Whether splitting attention and experts onto separate GPU pools (AFD) "
         "beats running them together (colocated), on one GB200 NVL72 domain.", BODY),
        ("", BODY),
        ("Both arms are decode workers", TITLE),
        ("Every latency and throughput column is a decode step. Prefill is not "
         "in any number here.", BODY),
        ("Throughput is averaged over all 72 GPUs in the domain, including any a "
         "deployment leaves idle, so a split that strands GPUs is charged for them.", BODY),
        ("", BODY),
        ("Where the MoE stage time comes from", TITLE),
        ("Hardware measurement on GB200, not a model. Attention, KV and memory come "
         "from AIC's own database.", BODY),
        ("", BODY),
        ("What real routing costs", TITLE),
        ("The kernel benchmark can give every expert exactly the same number of "
         "tokens. No router does. Expert load was recorded from a real serving "
         "run and replayed through the same kernel, paired against a uniform "
         "reference inside one allocation, three alternating rounds per point.", BODY),
        ("The cost depends almost entirely on how many experts a GPU holds. At 48 "
         "or more it is 1.000 -- a GPU holding that many experts averages the "
         "skew away. It rises as the shard narrows: median 1.044 at 16 experts "
         "per GPU and 1.050 at 4, reaching 1.137 at the busiest point measured.", BODY),
        ("Two references were tried and the first was wrong. The benchmark's "
         "built-in balanced mode gives each token top_k CONSECUTIVE expert ids, "
         "which puts a token on one rank and strides each expert's tokens by a "
         "power of two. Against that, real routing measured up to 29% FASTER, "
         "which prices a cache-access artefact rather than load. The reference "
         "used here is a flat distribution through the SAME sampler, so the "
         "access pattern is held fixed and only the load varies.", BODY),
        (f"{corrected} of {len(splits)} split rows carry a correction"
         + (f"; the factors run {min(factors):.3f} to {max(factors):.3f}." if factors else "."), BODY),
        ("", BODY),
        ("Recorded skew versus assumed skew", TITLE),
        ("MoE Stage Source says which. Qwen3-235B's expert load was recorded from "
         "a real serving run. The other four models' routing has not been captured "
         "-- the serving stack implements none of their architectures -- so their "
         "rows use a skew shaped to the coefficient of variation Qwen3's real "
         "router produced, 25.1%.", BODY),
        ("That assumption is second order. Between 14% and 25% skew the "
         "coefficient moved about 1%, and above 48 experts per GPU it is 1.000 "
         "whatever the skew. It matters only for the narrow-shard rows.", BODY),
        ("", BODY),
        ("How much skew there was", TITLE),
        ("The busiest expert took about 1.62x the average. Grouped onto GPUs that "
         "falls to roughly 1.08x at 32 experts per GPU and 1.31x at 4. Those are "
         "LOAD ratios, not time ratios: the measured time cost is far smaller, "
         "because a decode MoE stage streams every expert's weights whether or "
         "not tokens land on them.", BODY),
        ("", BODY),
        ("Reading the Assessment column", TITLE),
        ("gain — AFD ahead across the whole latency band.", BODY),
        ("gain, ladder-inflated — ahead, but the discrete batch rungs account for "
         "enough of it that the margin should not be quoted.", BODY),
        ("threshold noise — the winner changes inside the band.", BODY),
        ("no gain — colocated at least matches AFD.", BODY),
    ]
    for i, (text, font) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = font
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 118


if __name__ == "__main__":
    main()
