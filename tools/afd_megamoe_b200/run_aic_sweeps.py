#!/usr/bin/env python3
"""Drive AIConfigurator AFD sweeps with measured MegaMoE stage times.

Three sweeps:

  A  A:F ratio x batch x MTP x microbatches, every MoE time measured.
  B  Overlap bracket. AIC gives the AFD arm a ping-pong pipeline while the
     colocated arm has no overlap model at all, which biases the comparison.
     Re-running the same points at num_microbatches=1 (no overlap) and 2, and
     under both pipeline models, brackets the AFD advantage between a floor and
     a ceiling that are BOTH backed by measurement -- we measured mb=1 and mb=2
     separately, so neither end is modelled.
  C  ISL/OSL grid at the winning split, to see how the A:F optimum moves with
     context length.

Why a custom system: AIC sizes the F pool as n_f_nodes * gpus_per_node, and
b200_sxm declares 8 GPUs/node, so a sub-node split like 6A2F cannot be
expressed. aic_systems/b200_sxm_subnode.yaml sets 1 GPU per AIC-node and
raises inter_node_bw to the NVLink figure, which is what those links physically
are inside one 8-GPU node. Valid only while total GPUs <= 8.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from stability import accept

ROOT = Path("/home/scratch.yuny_wwfo_1/megamoe")
AIC_PY = "/home/scratch.yuny_wwfo_1/aic/venv-aic/bin/python"
SYSPATHS = f"default,{ROOT}/aic_systems"
SYSTEM = "b200_sxm_subnode"

# Draft-acceptance schedule taken verbatim from the AIC_AFD_Homogeneous sheet's
# Configuration tab: "Conditional acceptance falls with draft position
# (0.85, 0.80), so draft length 1 accepts 0.85 tokens per step on average and
# draft length 2 accepts 1.53." AIC refuses nextn>0 without this -- it has no
# built-in acceptance assumption -- and using the sheet's own numbers keeps this
# sweep comparable to it.
NEXTN_ACCEPTED = {0: None, 1: 0.85, 2: 1.53}

MODEL_PATH = {
    "qwen3_235b_fp4": "Qwen/Qwen3-235B-A22B-FP8",
    "minimax_m25_fp4": "MiniMaxAI/MiniMax-M2.5",
    "minimax_m3_fp4": "MiniMaxAI/MiniMax-M3",
}
# megamoe-m2n-batchext extends the grid to sequences-per-AG-rank 1/2/4/6 and
# 256/384. The low end exists because at isl=131072 only 4-7 sequences per rank
# fit, so without it the 128K cells can only be below-batch-floor extrapolation;
# the high end because every 5A1F win sat at the old ceiling with
# batch_limited_by=grid_max, leaving its failure point unobserved.
SPLIT_DIRS = ["megamoe-m2n-extended", "megamoe-m2n-afratio", "megamoe-m2n-grid",
              "megamoe-m2n-batchext"]

# MiniMax-M3 uses MiniMax Sparse Attention, which has no silicon coverage
# ("Error: MSA has no silicon data; use HYBRID or EMPIRICAL"). Only its
# ATTENTION falls back to the hybrid model -- the MoE stage is still the
# injected measurement, so the F side stays fully measured either way.
DATABASE_MODE = {"minimax_m3_fp4": "HYBRID"}


def load_measured() -> dict:
    """(model, A, F, batch_per_A, microbatches, mtp) -> stage ms, all layers+lanes."""
    out = {}
    for d in SPLIT_DIRS:
        for p in sorted((ROOT / "results" / d).glob("*.json")):
            j = json.loads(p.read_text())
            if not accept(j):
                continue
            w, t = j["workload"], j["topology"]
            if "ag_size" not in t:
                continue
            key = (j["model_profile"]["key"], t["ag_size"], t["eg_size"],
                   w["sequences_per_ag_rank"], w["microbatches"], w["mtp_nextn"])
            out.setdefault(key, j["stage_cuda"]["p50_ms"])
    return out


FIELDS = {
    "T_a_layer": r"T_a_layer:\s*([\d.]+)",
    "T_f_layer": r"T_f_layer:\s*([\d.]+)",
    "T_a2f_layer": r"T_a2f_layer:\s*([\d.]+)",
    "balance": r"Balance Ratio:\s*([\d.]+)",
    "tpot_ms": r"TPOT:\s*([\d.]+)",
    "tokens_s": r"tokens/s:\s*([\d,]+\.?\d*)",
    "tokens_s_gpu": r"tokens/s/gpu:\s*([\d,]+\.?\d*)",
    "tokens_s_user": r"tokens/s/user:\s*([\d,]+\.?\d*)",
    "concurrency": r"Concurrency:\s*([\d,]+)",
    "a_mem_gb": r"\(a\) Memory:\s*([\d.]+)",
    "f_mem_gb": r"\(f\) Memory:\s*([\d.]+)",
    "b_total": r"B_total:\s*([\d,]+)",
}


def run_one(cfg: dict) -> dict:
    cmd = [
        AIC_PY, "-m", "aiconfigurator.cli.main", "estimate",
        "--model-path", MODEL_PATH[cfg["model"]], "--system", SYSTEM,
        "--backend", "sglang", "--perf-db-version", "0.5.12",
        "--systems-paths", SYSPATHS,
        "--estimate-mode", "afd",
        "--moe-quant-mode", "w4a8_mxfp4_mxfp8",
        "--database-mode", DATABASE_MODE.get(cfg["model"], "SILICON"),
        "--isl", str(cfg["isl"]), "--osl", str(cfg["osl"]),
        "--nextn", str(cfg["mtp"]), "--no-afd-combined-with-pd",
        "--n-a-nodes", str(cfg["a"]), "--n-f-nodes", str(cfg["f"]),
        "--a-tp-size", "1", "--a-batch-size", str(cfg["batch"]),
        "--num-microbatches", str(cfg["mb_pipeline"]),
        "--f-moe-ep-size", str(cfg["f"]),
        "--pipeline-model", cfg.get("pipeline", "conservative"),
        "--afd-moe-time-ms", f"{cfg['moe_ms']:.6f}", "--no-color",
    ]
    accepted = NEXTN_ACCEPTED.get(cfg["mtp"])
    if accepted is not None:
        cmd += ["--nextn-accepted", str(accepted)]
    row = {k: cfg[k] for k in ("sweep", "model", "a", "f", "batch", "mtp",
                               "mb_measured", "mb_pipeline", "isl", "osl")}
    row["pipeline"] = cfg.get("pipeline", "conservative")
    row["moe_ms_measured"] = round(cfg["moe_ms"], 6)
    row["database_mode"] = DATABASE_MODE.get(cfg["model"], "SILICON")
    try:
        # stderr matters: AIC raises OOM as a traceback there, and that is a
        # real outcome (the sheet calls it batch_limited_by=capacity), not a
        # tooling failure.
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        return row
    for name, pat in FIELDS.items():
        m = re.search(pat, out)
        row[name] = float(m.group(1).replace(",", "")) if m else None
    if row["tpot_ms"] is not None:
        row["status"] = "ok"
    elif "OOM:" in out:
        row["status"] = "oom"            # legitimate: config does not fit
    else:
        err = re.search(r"^(?:Error|RuntimeError): (.+)$", out, re.M)
        row["status"] = "error"
        row["error"] = err.group(1)[:140] if err else "unparsed"
    return row


def build_configs(measured: dict, only: set[str]) -> list[dict]:
    cfgs = []
    # ---- Sweep A: full measured grid, AIC's default pipeline ----------------
    if "A" in only:
        for (model, a, f, batch, mb, mtp), ms in sorted(measured.items()):
            cfgs.append(dict(sweep="A", model=model, a=a, f=f, batch=batch, mtp=mtp,
                             mb_measured=mb, mb_pipeline=mb, isl=4096, osl=1024, moe_ms=ms))
    # ---- Sweep B: overlap bracket ------------------------------------------
    # Only where BOTH mb=1 and mb=2 were measured, so the floor and the ceiling
    # each rest on their own measurement rather than on a pipeline assumption.
    if "B" in only:
        for (model, a, f, batch, mb, mtp), ms in sorted(measured.items()):
            if mtp != 0 or batch not in (48, 96):
                continue
            for pipeline in ("optimistic", "conservative"):
                cfgs.append(dict(sweep="B", model=model, a=a, f=f, batch=batch, mtp=mtp,
                                 mb_measured=mb, mb_pipeline=mb, isl=4096, osl=1024,
                                 moe_ms=ms, pipeline=pipeline))
    # ---- Sweep C: context sweep, searching batch at every point -------------
    # The batch is swept rather than pinned. At long context a fixed batch just
    # OOMs and tells you nothing; the sheet searches the batch grid and reports
    # what stopped it (batch_limited_by=capacity). Sweeping every measured
    # batch rung reproduces that: the best feasible batch per (isl, osl, split)
    # falls out, and the OOM rows mark the capacity wall.
    if "C" in only:
        for isl in (4096, 8192, 16384, 32768, 65536, 131072):
            for osl in (256, 1024, 4096):
                for (model, a, f, batch, mb, mtp), ms in sorted(measured.items()):
                    if mtp != 0:
                        continue
                    # BOTH microbatch counts: mb=1 is the only setting that can be
                    # compared fairly against colocated (neither side overlapped),
                    # and it is also AFD's faster configuration on this hardware.
                    cfgs.append(dict(sweep="C", model=model, a=a, f=f, batch=batch,
                                     mtp=mtp, mb_measured=mb, mb_pipeline=mb,
                                     isl=isl, osl=osl, moe_ms=ms))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default="A,B,C")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "handoff" / "aic_afd_sweeps.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    measured = load_measured()
    cfgs = build_configs(measured, set(a.sweeps.split(",")))
    if a.limit:
        cfgs = cfgs[: a.limit]
    print(f"measured MegaMoE points: {len(measured)}")
    print(f"AIC runs queued        : {len(cfgs)}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, row in enumerate(ex.map(run_one, cfgs), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(cfgs):
                ok = sum(1 for r in rows if r.get("status") == "ok")
                oom = sum(1 for r in rows if r.get("status") == "oom")
                print(f"  {i}/{len(cfgs)}  ok={ok} oom={oom}", flush=True)

    cols = ["sweep", "model", "a", "f", "batch", "mtp", "mb_measured", "mb_pipeline",
            "pipeline", "isl", "osl", "database_mode", "moe_ms_measured", "status",
            *FIELDS.keys(), "error"]
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    import collections
    c = collections.Counter(r.get("status") for r in rows)
    print(f"wrote {len(rows)} rows {dict(c)} -> {a.out}")


if __name__ == "__main__":
    main()
