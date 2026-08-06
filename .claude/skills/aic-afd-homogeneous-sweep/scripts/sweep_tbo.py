"""Re-run the colocated arm, recording the compute/communication split per point.

The colocated arm is billed in AIC as a strict sum: every all-to-all and
all-reduce lands on the critical path. Real SGLang runs two-batch overlap
(TBO), which issues one micro-batch's DeepEP dispatch/combine on a side stream
while the other micro-batch computes. The ceiling of that technique is
``max(compute, comm)`` instead of ``compute + comm`` -- every byte hidden, at
no cost. That ceiling is what this records, so the AFD comparison can be
restated against a colocated arm that is given the benefit of the doubt.

The bound is deliberately unreachable. It grants perfect overlap, ignores that
splitting the batch in two shrinks every GEMM's M dimension, ignores the extra
activation buffers, and ignores the sync points a real implementation keeps.
A colocated arm that still loses under it loses for structural reasons.
"""
import csv
import itertools
import os
import re
import time
import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("aiconfigurator").setLevel(logging.CRITICAL)

import aiconfigurator
import aiconfigurator_core
import tbo_probe
from aiconfigurator.cli.api import cli_estimate

RECORD = tbo_probe.install()

SYS, BE, VER = "gb200", "sglang", "0.5.14"
MEM_CAP = 186.0
WORKBOOK = os.environ.get("WORKBOOK", "multi")
SHARD = int(os.environ.get("SHARD", 0))
NSHARD = int(os.environ.get("NSHARD", 1))
OUT = os.environ["OUT"]

# The compute/comm split is delegated to tbo_probe, which also reaches inside
# an OverlapOp so the GQA and MLA/MSA op layouts are measured the same way.


ATTN = {"Qwen3-235B-A22B": "GQA", "MiniMax-M2.5": "GQA", "DeepSeek-V3": "MLA",
        "DeepSeek-V3.1": "MLA", "Kimi-K2.5": "MLA", "MiniMax-M3": "MSA"}

if WORKBOOK == "m3":
    MODELS = [("MiniMax-M3", "MiniMaxAI/MiniMax-M3", "FP8-block", "fp8_block", "HYBRID"),
              ("MiniMax-M3", "MiniMaxAI/MiniMax-M3", "NVFP4", "nvfp4", "HYBRID")]
    CONTEXTS = [4096, 16384, 65536, 131072, 262144, 524288, 1048576]
    OSLS = [256, 1024]
else:
    MODELS = [("Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-FP8", "FP8-block", "fp8_block", "SILICON"),
              ("Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-FP8", "NVFP4", "nvfp4", "SILICON"),
              ("MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.5", "FP8-block", "fp8_block", "SILICON"),
              ("MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.5", "NVFP4", "nvfp4", "SILICON"),
              ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3", "FP8-block", "fp8_block", "SILICON"),
              ("DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1", "NVFP4", "nvfp4", "SILICON"),
              ("Kimi-K2.5", "moonshotai/Kimi-K2.5", "INT4-wo", "int4_wo", "SILICON"),
              ("Kimi-K2.5", "moonshotai/Kimi-K2.5", "NVFP4", "nvfp4", "SILICON")]
    CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072]
    OSLS = [256, 1024, 4096]

# Same narrowing knobs as sweep_afd.py, so both sweeps can be aimed at an
# identical slice. The TBO column joins on (model, precision, isl, osl, mtp);
# a slice this sweep does not cover yields no column rather than an error.
if os.environ.get("ISL_ONLY"):
    keep = {int(v) for v in os.environ["ISL_ONLY"].split(",")}
    CONTEXTS = [c for c in CONTEXTS if c in keep]
if os.environ.get("OSL_ONLY"):
    OSLS = [int(v) for v in os.environ["OSL_ONLY"].split(",")]

MTP = [("0", {}), ("1", dict(nextn=1, nextn_accepted=0.85)),
       ("2", dict(nextn=2, nextn_accepted=1.53))]

BATCH = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128,
         144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512,
         576, 640, 704, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2560, 3072,
         3584, 4096]

CO_UNITS = [2, 4, 8, 16, 32, 64]


def colo_configs():
    out = []
    for n in CO_UNITS:
        for tp in (1, 2, 4, 8):
            if n % tp:
                continue
            dp = n // tp
            for ep in (n, n // 2, n // 4):
                if ep < 1 or n % ep:
                    continue
                out.append((tp, dp, n // ep, ep, n))
    return sorted(set(out))


NONAFD = colo_configs()
print(f"app :{aiconfigurator.__file__}\ncore:{aiconfigurator_core.__file__}", flush=True)

TASKS = [(m, c) for m in MODELS for c in CONTEXTS]
TASKS = [t for i, t in enumerate(TASKS) if i % NSHARD == SHARD]
print(f"workbook={WORKBOOK} cells={len(TASKS)} colo={len(NONAFD)} batch={len(BATCH)}", flush=True)

COLS = ["model", "attn", "precision", "isl", "osl", "mtp_nextn", "config", "unit_gpus",
        "batch_per_unit", "tokens_s_gpu", "tpot_ms", "tokens_s_user", "memory_gb",
        "compute_ms", "comm_ms", "comm_ops"]

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, COLS)
    w.writeheader()
    n_rows = 0
    t0 = time.time()
    for ((mname, mp, prec, mq, dbm), isl), osl, (mtag, mkw) in itertools.product(TASKS, OSLS, MTP):
        base = dict(model_path=mp, system_name=SYS, backend_name=BE, backend_version=VER,
                    database_mode=dbm, isl=isl, osl=osl, moe_quant_mode=mq)
        common = dict(model=mname, attn=ATTN[mname], precision=prec, isl=isl, osl=osl,
                      mtp_nextn=mtag)
        for (tp, dp, mtp_, mep, n) in NONAFD:
            for bs in BATCH:
                RECORD.clear()
                try:
                    r = cli_estimate(mode="static_gen", batch_size=bs, tp_size=tp,
                                     attention_dp_size=dp, moe_tp_size=mtp_,
                                     moe_ep_size=mep, **mkw, **base)
                    d = r.raw
                except Exception:
                    continue
                if d.get("memory", 1e9) > MEM_CAP or d.get("tokens/s/gpu") is None:
                    continue
                lat = getattr(r.summary, "_generation_latency_dict", None) or {}
                comp, comm = tbo_probe.split_step(lat, RECORD)
                names = "|".join(sorted(k for k in lat if tbo_probe.is_comm(k)))
                if RECORD:
                    names = names + ("|" if names else "") + "overlap:" + ",".join(sorted(RECORD))
                w.writerow(dict(config=f"tp{tp}dp{dp}-moetp{mtp_}ep{mep}", unit_gpus=n,
                                batch_per_unit=bs,
                                tokens_s_gpu=round(d["tokens/s/gpu"], 4),
                                tpot_ms=round(d.get("tpot", 0), 4),
                                tokens_s_user=round(d.get("tokens/s/user", 0), 4),
                                memory_gb=round(d.get("memory", 0), 2),
                                compute_ms=round(comp, 5), comm_ms=round(comm, 5),
                                comm_ops=names, **common))
                n_rows += 1
        fh.flush()
    print(f"done rows={n_rows} {time.time()-t0:.0f}s", flush=True)
