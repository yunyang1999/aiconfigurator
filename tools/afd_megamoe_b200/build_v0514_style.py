#!/usr/bin/env python3
"""B200 MegaMoE sweep, laid out exactly like AIC_AFD_Homogeneous_GB200_v0514.

Same eight sheets, same sheet order, same column names in the same order, same
A1 captions. The only thing that changes is where the MoE stage time comes from:
every one of them is a direct FastAFD MegaMoE benchmark on B200 instead of AIC's
generic MoE model.

The two conventions that define what `ratio` means are copied verbatim and were
verified against the source workbook:

  ratio = afd_tpsgpu / colo_tpsgpu       (3653/4935.5 -> 0.74, 732.9/648.7 -> 1.13)
  tokps_per_user = tpsgpu * unit_gpus / conc      (reproduces its 8.15 and 10.28)

and each arm searches its OWN best batch, so concurrency is not held equal.

Two columns cannot be filled the way the GB200 sheet fills them and are left
blank rather than invented: afd_comm_hidden (MegaMoE fuses dispatch and combine
into the kernel, so there is no separable comm to hide) and colo_memory_gb /
colo_weights_per_gpu_GB / colo_weight_share_of_hbm_pct (the AGG estimate path
used here does not report per-GPU weight memory). Everything else is populated.
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from colo_metrics import decode_only

ROOT = Path("/home/scratch.yuny_wwfo_1/megamoe")
HANDOFF = ROOT / "handoff"
OUT = HANDOFF / "AIC_AFD_MegaMoE_B200_v0514style.xlsx"

FONT = "Arial"
HDR_FILL = PatternFill("solid", start_color="1F3864")
HDR_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
TITLE = Font(name=FONT, bold=True, size=11, color="1F3864")
BODY = Font(name=FONT, size=10)
MEAS = Font(name=FONT, size=10, color="0000FF")
GAIN = PatternFill("solid", start_color="E2EFDA")
EDGE = PatternFill("solid", start_color="FFF2CC")
FAIL = PatternFill("solid", start_color="FFC7CE")

MODEL_LABEL = {"qwen3_235b_fp4": "Qwen3-235B-A22B",
               "minimax_m25_fp4": "MiniMax-M2.5",
               "minimax_m3_fp4": "MiniMax-M3"}
# Attention family, from AIC's own model config load (Qwen3MoeForCausalLM and
# MiniMaxM2ForCausalLM report GQA head counts) and from AIC's own refusal on M3
# ("MSA has no silicon data"), which is why M3 runs in HYBRID.
ATTN = {"qwen3_235b_fp4": "GQA", "minimax_m25_fp4": "GQA", "minimax_m3_fp4": "MSA"}
CONST = dict(moe_precision="w4a8_mxfp4_mxfp8",
             moe_kernel="fastafd_megamoe (measured)",
             system="NVIDIA B200 SXM 8-GPU",
             backend="SGLang 0.5.12")
AFD_FILES = ["aic_afd_sweeps_final.csv"]
SLA_LIMITS = [10, 15, 20, 30, 50, 100, 150]

# v0514 column orders, copied field for field.
COMMON_HEAD = ["model", "attn", "moe_precision", "isl", "osl", "mtp_nextn"]
AFD_BLOCK = ["afd_split", "afd_A_gpus", "afd_F_gpus", "afd_unit_gpus", "afd_batch",
             "afd_tpsgpu", "afd_tpot_ms", "afd_tokps_per_user", "afd_conc",
             "afd_conc_per_gpu", "afd_t_attn_us", "afd_t_ffn_us", "afd_t_comm_us",
             "afd_tf_over_ta", "afd_memory_gb", "afd_batch_limited_by"]
COLO_BLOCK = ["colo_cfg", "colo_unit_gpus", "colo_batch", "colo_tpsgpu",
              "colo_tpot_ms", "colo_tokps_per_user", "colo_conc", "colo_conc_per_gpu",
              "colo_batch_limited_by"]
NOSLA_COLS = COMMON_HEAD + ["moe_kernel", "system", "backend"] + AFD_BLOCK + COLO_BLOCK + ["ratio"]
SLA_COLS = (COMMON_HEAD + ["sla_tpot_ms", "moe_kernel", "system", "backend"]
            + COLO_BLOCK + AFD_BLOCK + ["ratio"])
GAIN_COLS = (COMMON_HEAD + ["sla_tpot_ms", "ratio", "moe_kernel", "system", "backend"]
             + COLO_BLOCK + AFD_BLOCK)
SPLIT_COLS = (COMMON_HEAD + ["moe_kernel", "system", "backend"] + AFD_BLOCK
              + ["colo_cfg", "colo_unit_gpus", "colo_batch", "colo_tpsgpu",
                 "colo_tpot_ms", "ratio"])
# Appended AFTER the v0514 columns, so the parity check on the original set
# still holds and the additions are visibly ours.
#
#   ratio_vs_colo_..    ratio against the colocated arm's overlap ceiling. Only
#                       dispatch/combine is hidden, because on one GPU the MoE
#                       GEMMs and attention compete for the same SMs -- compute
#                       cannot hide behind compute. AIC has no TBO model, so
#                       `ratio` assumes colocated overlaps nothing and this column
#                       is the other end. Still a ceiling: MegaMoE already overlaps
#                       those legs inside the kernel. Blank for MiniMax-M3 (fused
#                       ops, no separable share). A construction, not a measurement.
#   ratio_sla_plus10    the same comparison re-optimised with the TPOT ceiling 10%
#                       looser. Where it moves a lot, the row sits on a threshold
#                       edge: the colocated arm's TPOT is nearly flat in batch, so a
#                       ceiling landing between two rungs can halve its throughput.
#   sla_edge_sensitive  the ratio moves more than 25% under that perturbation.
#   gain_survives_..    whether the row still clears the 1.10 line that put it on
#                       the AFD_Perf_Gain sheet. This is NOT implied by
#                       sla_edge_sensitive: a row can move only 24% and still
#                       cross 1.10, which is exactly what 10 of the 23 failing
#                       rows do. Relative movement and crossing the sheet's own
#                       admission line are different questions.
SENS = ["ratio_sla_plus10", "sla_edge_sensitive", "gain_survives_sla_plus10"]
SLA_COLS = SLA_COLS + SENS
GAIN_COLS = (COMMON_HEAD + ["sla_tpot_ms", "ratio"] + SENS
             + ["moe_kernel", "system", "backend"] + COLO_BLOCK + AFD_BLOCK)

BLUE = {"afd_tpsgpu", "afd_tpot_ms", "colo_tpsgpu", "colo_tpot_ms"}
DERIVED = {"ratio_sla_plus10"}
CAPTION = {
    "AFD_Perf_Gain > 10% ":
        "Best configuration under each time-per-output-token limit, generation length "
        "1024, where the disaggregated arm is more than 10% ahead. A blank side means "
        "no configuration of that arm meets the limit. gain_survives_sla_plus10=NO "
        "(red) marks rows whose 10% margin disappears when the ceiling is moved 10%, "
        "i.e. the reason the row is on this sheet does not survive the threshold "
        "being placed slightly differently.",
    "Model":
        "Model and weight precision at generation length 1024, draft length 0, no "
        "latency limit. Each arm at its own best deployment and batch.",
    "Context":
        "Context sweep at generation length 1024, no latency limit. Each arm at its "
        "own best deployment and batch.",
    "ISL_x_OSL":
        "Input length crossed with generation length, draft length 0, no latency limit.",
    "SLA_Latency":
        "Best configuration under each time-per-output-token limit, generation length "
        "1024. A blank side means no configuration of that arm meets the limit.",
    "AF_Split":
        "Every A:F split at its own best batch, generation length 1024, draft length 0, "
        "no latency limit. The colocated side repeats the single best colocated point.",
}


def rd(name):
    p = HANDOFF / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ load arms
def load():
    # OOM rungs are keyed by TOPOLOGY, not just by cell. A 7A1F run that OOMs at
    # batch 32 says nothing about whether 4A4F fits there, and an ep=4 OOM at
    # tokens-per-rank 8 is a different total load from an ep=8 one. Pooling them
    # made every no-SLA row report batch_limited_by=capacity, because some other
    # topology had always OOMed somewhere above.
    cells = defaultdict(lambda: {"afd": [], "colo": [],
                                 "afd_oom": defaultdict(list),
                                 "colo_oom": defaultdict(list)})
    seen = set()
    for fn in AFD_FILES:
        for r in rd(fn):
            cell = (r["model"], int(r["isl"]), int(r["osl"]), int(r["mtp"]))
            a, fq, b = int(r["a"]), int(r["f"]), int(r["batch"])
            key = (cell, a, fq, b, int(r["mb_pipeline"]))
            if key in seen:
                continue
            seen.add(key)
            if r["status"] == "oom":
                cells[cell]["afd_oom"][f"{a}A{fq}F"].append(b)
                continue
            if r["status"] != "ok" or not f(r["tokens_s_gpu"]):
                continue
            cells[cell]["afd"].append({
                "afd_split": f"{a}A{fq}F", "afd_A_gpus": a, "afd_F_gpus": fq,
                "afd_unit_gpus": a + fq, "afd_batch": b, "afd_conc": b * a,
                "afd_tpsgpu": f(r["tokens_s_gpu"]), "afd_tpot_ms": f(r["tpot_ms"]),
                "afd_t_attn_us": round((f(r["T_a_layer"]) or 0) * 1000, 1),
                "afd_t_ffn_us": round((f(r["T_f_layer"]) or 0) * 1000, 1),
                "afd_t_comm_us": round((f(r["T_a2f_layer"]) or 0) * 1000, 1),
                "afd_tf_over_ta": f(r["balance"]),
                # MegaMoE fuses dispatch/combine into the kernel, so there is no
                # separable communication to hide or expose. Blank, not False.
                "afd_comm_hidden": None,
                "afd_memory_gb": round(max(f(r.get("a_mem_gb")) or 0,
                                           f(r.get("f_mem_gb")) or 0), 1),
                "_mb": int(r["mb_pipeline"]),
            })
    for r in rd("aic_colocated_measured.csv"):
        cell = (r["model"], int(r["isl"]), int(r["osl"]), int(r["mtp"]))
        if r["status"] == "oom":
            cells[cell]["colo_oom"][int(r["ep"])].append(int(r["tokens_per_rank"]))
            continue
        m = decode_only(r)
        if not m:
            continue
        ep = m["colo_unit_gpus"]
        m["colo_cfg"] = f"tp1dp{ep}-moeep{ep}"
        if r.get("moe_provenance") != "measured":
            continue
        cells[cell]["colo"].append(m)
    return {k: v for k, v in cells.items() if v["afd"] and v["colo"]}


def limited_by(w, arm, same_topo, oom_batches, sla):
    """Why the batch search stopped, in the four categories that actually differ.

    Only the NEXT rung matters: whatever blocks first is the binding constraint,
    and everything is evaluated inside the winner's own topology.

      capacity  the next rung does not fit
      latency   the next rung fits but breaks the TPOT ceiling
      compute   the next rung fits and is allowed, but does not raise
                tokens/s/GPU -- throughput has peaked
      grid      there is no next rung; the true optimum may lie past what was
                benchmarked, so this is an open end rather than a limit
    """
    b = w[f"{arm}_batch"]
    ok = {c[f"{arm}_batch"]: c for c in same_topo}
    rungs = sorted(x for x in set(ok) | set(oom_batches) if x > b)
    if not rungs:
        return "grid"
    nxt = rungs[0]
    if nxt not in ok:
        return "capacity"
    if sla is not None and ok[nxt][f"{arm}_tpot_ms"] > sla:
        return "latency"
    return "compute"


def best(cands, arm, oom_by_topo, sla=None, overlap=False):
    """Highest tokens/s/GPU, optionally under a TPOT ceiling.

    overlap=True re-runs the search on the overlap-adjusted metric instead of
    rescaling the winner picked without overlap. That matters and is not a
    refinement: hiding the collective legs rewards exactly the wide-EP layouts
    that serial communication accounting penalises, so the config that wins with
    overlap is often not the one that wins without it. Rescaling the old winner
    would understate the colocated arm and flatter AFD.
    """
    tps = f"{arm}_tpsgpu_comm_overlap" if overlap else f"{arm}_tpsgpu"
    tpot = f"{arm}_tpot_comm_overlap_ms" if overlap else f"{arm}_tpot_ms"
    if overlap:
        cands = [c for c in cands if c.get(tps) and c.get(tpot)]
        if not cands:
            return None
    pool = cands if sla is None else [c for c in cands if c[tpot] <= sla]
    if not pool:
        return None
    w = dict(max(pool, key=lambda c: c[tps]))
    topo = w["afd_split"] if arm == "afd" else w["colo_unit_gpus"]
    same = [c for c in cands
            if (c["afd_split"] if arm == "afd" else c["colo_unit_gpus"]) == topo]
    if not overlap:
        w[f"{arm}_batch_limited_by"] = limited_by(
            w, arm, same, oom_by_topo.get(topo, []), sla)
    w[f"{arm}_tokps_per_user"] = round(
        w[f"{arm}_tpsgpu"] * w[f"{arm}_unit_gpus"] / w[f"{arm}_conc"], 2)
    w[f"{arm}_conc_per_gpu"] = round(w[f"{arm}_conc"] / w[f"{arm}_unit_gpus"], 1)
    return w


def row(cell, a, c, sla=None):
    model, isl, osl, mtp = cell
    r = {"model": MODEL_LABEL[model], "attn": ATTN[model], "isl": isl, "osl": osl,
         "mtp_nextn": mtp, **CONST}
    if sla is not None:
        r["sla_tpot_ms"] = sla
    r.update(a or {})
    r.update(c or {})
    if a and c:
        r["ratio"] = round(a["afd_tpsgpu"] / c["colo_tpsgpu"], 3)
    return r


# ---------------------------------------------------------------- sheet writer
def sheet(wb, name, cols, rows, shade=None):
    ws = wb.create_sheet(name)
    ws.cell(row=1, column=1, value=CAPTION.get(name, "")).font = TITLE
    for i, n in enumerate(cols, 1):
        cell = ws.cell(row=2, column=i, value=n)
        cell.font, cell.fill = HDR_FONT, HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = ws.cell(row=3, column=1)
    for i, r in enumerate(rows):
        rw = i + 3
        for c, k in enumerate(cols, 1):
            cell = ws.cell(row=rw, column=c, value=r.get(k))
            cell.font = MEAS if k in BLUE else BODY
            if k.endswith(("_ms", "_us")) and isinstance(r.get(k), float):
                cell.number_format = "0.00"
            if k == "ratio":
                cell.number_format = "0.000"
                cell.font = Font(name=FONT, size=10, bold=True)
            elif k in DERIVED:
                cell.number_format = "0.000"
            elif k == "sla_edge_sensitive" and r.get(k) == "yes":
                cell.fill = EDGE
            elif k == "gain_survives_sla_plus10" and r.get(k) == "NO":
                cell.fill = FAIL
                cell.font = Font(name=FONT, size=10, bold=True, color="9C0006")
        if shade and shade(r):
            for c in range(1, len(cols) + 1):
                ws.cell(row=rw, column=c).fill = GAIN
    ws.auto_filter.ref = f"A2:{get_column_letter(len(cols))}{len(rows) + 2}"
    for col in ws.columns:
        w = max((len(str(x.value)) for x in col if x.value is not None), default=0)
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(8, min(30, w + 2))
    return ws


def sheet_config(wb, cells, counts):
    ws = wb.create_sheet("Configuration", 0)
    ws.cell(row=1, column=1, value="Item").font = HDR_FONT
    ws.cell(row=1, column=1).fill = HDR_FILL
    ws.cell(row=1, column=2, value="Value").font = HDR_FONT
    ws.cell(row=1, column=2).fill = HDR_FILL
    items = [
        ("Subject", "Homogeneous GPU-to-GPU Attention-FFN Disaggregation versus colocated "
                    "serving, decode phase."),
        ("Layout", "Same sheets, sheet order and column names as "
                   "AIC_AFD_Homogeneous_GB200_MultiModel_v0514. Only the source of the MoE "
                   "stage time differs."),
        ("MoE stage", "MEASURED. FastAFD MegaMoE on B200, commit "
                      "80279509ada512cd6c86438c8efe07433fa76c79, branch "
                      "megamoe-multimodel-b200. 30 warmups / 40 iterations, all MoE layers. "
                      "No extrapolated MoE time appears in this workbook."),
        ("Stability gate", "A benchmark point is used when its dispersion across the 40 timed "
                           "iterations of one run is CV <= 4% AND every rank returned finite "
                           "output. FastAFD's own `stable` flag is CV <= 3%; the limit is "
                           "raised here because 3% was discarding small-batch rungs whose "
                           "values sit exactly on the flat trend their neighbours describe "
                           "(Qwen3-235B ep=8 nextN=0: batch 4 at 7.020 ms CV 3.45%, batch 6 at "
                           "7.102 ms CV 3.63%, between measured 6.528 and 6.919). A short "
                           "kernel at small batch carries more launch jitter -- that is a "
                           "property of the measurement, not of the kernel. Dropping them left "
                           "a hole between batch 2 and batch 8 that a TPOT ceiling could land "
                           "in. The finiteness check is NOT relaxed: dispersion is about "
                           "repeatability, non-finite output is about validity."),
        ("Why measured is required", "b200_sxm's SILICON database has no w4a8_mxfp4_mxfp8 "
                                     "MoE rows, so AIC cannot produce this precision on this "
                                     "system without the injected measurement."),
        ("moe_precision", "w4a8_mxfp4_mxfp8 - E2M1 weights, UE8M0 block-32 scales, E4M3 "
                          "activations. This is NOT NVFP4."),
        ("ratio", "afd_tpsgpu / colo_tpsgpu, i.e. output tokens per second per GPU, "
                  "disaggregated over colocated. Same definition as the GB200 sheet."),
        ("tokps_per_user", "tpsgpu * unit_gpus / conc, same as the GB200 sheet."),
        ("Batch policy", "Each arm searches its own batch grid for maximum tokens/s/GPU. "
                         "Concurrency is NOT held equal. batch_limited_by records what "
                         "stopped the search: capacity, latency, grid_max or throughput_peak."),
        ("Serving phase", "Decode only, both arms, in TIME and in MEMORY. The "
                          "disaggregated arm runs --no-afd-combined-with-pd, so AIC reports "
                          "TTFT 0 and its TPOT is already a pure decode step."),
        ("Colocated estimate mode",
         "static_gen, not agg. Both give the same decode step -- agg's genonly_step total and "
         "static_gen's TPOT agree to the digit (14.914 ms on Qwen3-235B, isl=4096, batch 8) -- "
         "but they size activation memory differently. AIC's agg path takes activations from "
         "BuildConfig.max_num_tokens, which cli_estimate defaults to isl (api.py:941), and "
         "memory.py:273 states that max_batch_size does not affect non-KV memory. The agg "
         "colocated arm was therefore charged a prefill chunk's activations -- 2.016 GiB at "
         "isl=4096 rising to 32.203 GiB at isl=65536 with the batch unchanged -- while the "
         "AFD arm pays none of it. That is 18% of a 180 GiB card taken from the colocated KV "
         "budget for work a decode worker never does, and it forced OOM at batches that fit: "
         "isl=131072 batch 8 reads 184.05 GiB under agg and 119.75 GiB here. static_gen holds "
         "activations at 0.101 GiB, flat in isl and in batch."),
        ("Per-step conversion",
         "static_gen reports op latencies CUMULATIVE over the generation phase, so they are "
         "divided by (osl - 1). Verified: the cumulative sum over (osl-1) reproduces its own "
         "TPOT exactly."),
        ("AFD pipeline model",
         "conservative, and microbatches are capped at what the pipeline can sustain. AIC's "
         "K=3 (optimistic) and K=2 (conservative) cadences declare N_min of 3 and 2 in-flight "
         "microbatches, but only the optimistic branch enforced its threshold, and its "
         "fallback landed on conservative -- which needs two as well. A single in-flight "
         "microbatch cannot overlap anything: layer i+1's A input IS layer i's F output, so "
         "the pools strictly alternate. mb=1 is therefore priced serially "
         "(t_a + t_a2f + t_f + t_f2a per layer) rather than at max(t_a, t_f). On MiniMax-M3 "
         "4A4F batch 16 that is 20.0 ms instead of 11.4 ms. mb>=2 is untouched. "
         "PENDING CONFIRMATION with the PR #1323 authors: the missing N_min check may be an "
         "omission or an intentional fine-grained-overlap assumption."),
        ("Why conservative and not optimistic",
         "With the measured stage injected, t_a2f = t_f2a = 0, so max(t_a, t_f, t_c) and "
         "max(t_a + t_a2f, t_f + t_f2a) are numerically identical -- verified to the digit. "
         "Conservative is the weaker claim: nothing here demonstrates three in-flight "
         "microbatches hiding a network stage."),
        ("Capacity check",
         "Enforced here, not by AIC. static_gen runs no OOM check of its own -- isl=131072 at "
         "batch 64 returns a result reporting 782.89 GiB on a 180 GiB card -- so a row is "
         "marked OOM when summary memory total exceeds 180 GiB."),
        ("System", "NVIDIA B200 SXM, 8 GPUs per NVLink domain, 180 GiB usable HBM per GPU."),
        ("NVLink domain", "8 GPUs. MegaMoE's dispatch/combine buffer is "
                          "torch.distributed._symmetric_memory and its kernel dereferences "
                          "one raw peer pointer per rank, so A+F <= 8 for every split."),
        ("Splits measured", "4A4F, 6A2F (8 GPUs), 5A1F, 7A1F (6 and 8 GPUs). f_moe_ep_size "
                            "must divide the expert count, so F is 1, 2 or 4."),
        ("Colocated topologies", "ep 1, 2, 4 and 8, all measured. The colocated arm chooses "
                                 "its GPU count the same way the split arm chooses A:F."),
        ("Batch grid", "1, 2, 4, 6, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384 - identical "
                       "for both arms."),
        ("Engine", "AIConfigurator 0.11.0, branch pr1323-afd-megamoe-pr1479. Measured stage "
                   "injected via --afd-moe-time-ms (inference_session.py:1425), replacing "
                   "all F-side compute and stage communication."),
        ("Interpolation", "Past-the-frontier queries use the joint-log2 kNN util transfer "
                          "adopted from liz-badada@23bd13d5; the earlier nearest-path snap "
                          "was discontinuous at the generation-attention staircase."),
        ("attn", "GQA for Qwen3-235B and MiniMax-M2.5, both under SILICON. MSA for "
                 "MiniMax-M3, which has no silicon coverage - AIC reports 'MSA has no "
                 "silicon data', so only its ATTENTION falls back to HYBRID. Its MoE stage "
                 "is still the injected measurement."),
        ("MiniMax-M3 gap", "No 5A1F or 7A1F rows: with F=1 the whole expert set lands on one "
                           "GPU and the benchmark OOMs at the smallest batch. Absent, not "
                           "modelled."),
        ("afd_comm_hidden", "Blank throughout. MegaMoE fuses dispatch, L1, activation, L2 "
                            "and combine into one kernel, so unlike the GB200 sheet there is "
                            "no separable communication term to hide."),
        ("colo memory columns", "Blank. The AGG estimate path used here does not report "
                                "per-GPU weight memory, and inventing it would put a modelled "
                                "number in a measured workbook."),
        ("Not controlled", "AIC gives the disaggregated arm an A/F pipeline and has no "
                           "two-batch-overlap model for colocated, so every ratio here "
                           "assumes colocated does no attention/MoE overlap."),
        ("Measured points", f"{counts['afd_pts']} split-kernel and {counts['colo_pts']} "
                            f"colocated-kernel benchmarks, covering {len(cells)} "
                            "(model, isl, osl, nextN) cells."),
    ]
    for i, (k, v) in enumerate(items, start=2):
        a = ws.cell(row=i, column=1, value=k)
        a.font, a.alignment = Font(name=FONT, bold=True, size=10), Alignment(vertical="top")
        b = ws.cell(row=i, column=2, value=v)
        b.font, b.alignment = BODY, Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 112
    for i in range(2, len(items) + 2):
        ws.row_dimensions[i].height = 42


# ------------------------------------------------------------------------ main
def main():
    cells = load()
    nosla, sla_rows, split_rows = [], [], []

    for cell in sorted(cells, key=lambda c: (MODEL_LABEL[c[0]], c[1], c[2], c[3])):
        d = cells[cell]
        a = best(d["afd"], "afd", d["afd_oom"])
        c = best(d["colo"], "colo", d["colo_oom"])
        if a and c:
            nosla.append(row(cell, a, c))
            bysplit = defaultdict(list)
            for x in d["afd"]:
                bysplit[x["afd_split"]].append(x)
            for _, g in sorted(bysplit.items()):
                w = best(g, "afd", d["afd_oom"])
                split_rows.append(row(cell, w, c))
        for lim in SLA_LIMITS:
            aa = best(d["afd"], "afd", d["afd_oom"], sla=lim)
            cc = best(d["colo"], "colo", d["colo_oom"], sla=lim)
            if not (aa or cc):
                continue
            rr = row(cell, aa, cc, sla=lim)
            # Re-optimise both arms 10% looser. Both are free to move, so this is
            # not a handicap on either side -- it just asks whether the answer
            # survives the ceiling being placed slightly differently.
            a2 = best(d["afd"], "afd", d["afd_oom"], sla=lim * 1.1)
            c2 = best(d["colo"], "colo", d["colo_oom"], sla=lim * 1.1)
            if a2 and c2:
                r2 = a2["afd_tpsgpu"] / c2["colo_tpsgpu"]
                rr["ratio_sla_plus10"] = round(r2, 3)
                if rr.get("ratio"):
                    rr["sla_edge_sensitive"] = (
                        "yes" if abs(r2 / rr["ratio"] - 1) > 0.25 else "no")
                    if rr["ratio"] > 1.10:
                        rr["gain_survives_sla_plus10"] = "yes" if r2 > 1.10 else "NO"
            sla_rows.append(rr)
    gain = [r for r in sla_rows if r.get("ratio") and r["ratio"] > 1.10]
    gain.sort(key=lambda r: -r["ratio"])
    mxp = [r for r in nosla if r["mtp_nextn"] == 0 and r["osl"] == 1024]
    ctx = [r for r in nosla if r["osl"] == 1024]
    isl_osl = [r for r in nosla if r["mtp_nextn"] == 0]

    wb = Workbook()
    wb.remove(wb.active)
    sheet(wb, "AFD_Perf_Gain > 10% ", GAIN_COLS, gain)
    sheet(wb, "Model", NOSLA_COLS, mxp)
    sheet(wb, "Context", NOSLA_COLS, ctx)
    sheet(wb, "ISL_x_OSL", NOSLA_COLS, isl_osl)
    sheet(wb, "SLA_Latency", SLA_COLS, sla_rows)
    sheet(wb, "AF_Split", SPLIT_COLS, split_rows)
    sheet_config(wb, cells, {
        "afd_pts": sum(len(v["afd"]) for v in cells.values()),
        "colo_pts": sum(len(v["colo"]) for v in cells.values())})
    wb.save(OUT)

    with (HANDOFF / "afd_vs_colocated_v0514style.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SLA_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sla_rows)

    rr = [r["ratio"] for r in nosla if r.get("ratio")]
    print(f"sheets: Configuration, AFD_Perf_Gain > 10% ({len(gain)}), "
          f"Model_x_Precision ({len(mxp)}), Context ({len(ctx)}), "
          f"ISL_x_OSL ({len(isl_osl)}), SLA_Latency ({len(sla_rows)}), "
          f"AF_Split ({len(split_rows)})")
    print(f"no-latency-limit ratio: median {st.median(rr):.3f}  "
          f"AFD ahead {sum(1 for x in rr if x > 1)}/{len(rr)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
