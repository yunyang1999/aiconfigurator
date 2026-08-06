"""AFD vs colocated sweep engine, modelled on the reference workbook sheets.

Produces one long-form CSV per (comm_model, workbook). Downstream the CSV is
reduced into the result sheets.

Adapting this to a new sweep
----------------------------
The scope is hard-coded on purpose, so every CSV declares what it covers. Edit,
in this order: ``SYS, BE, VER`` (system and backend version -- see the
perf-data version trap in SKILL.md), ``MODELS`` (one entry per model x
precision, with the database_mode each one needs), ``CONTEXTS`` / ``OSLS`` /
``MTP``, then ``MEM_CAP`` for the target HBM. ``SPLITS`` and ``CO_UNITS`` follow
the rack: splits must satisfy ``(a + f) * gpus_per_node <= rack``.

Leave ``BATCH`` alone unless the memory envelope changed; it is the grid that
made non-power-of-2 optima visible.

Batch grid
----------
Steps alternate x1.5 and x1.33 instead of doubling. Doubling overshoots: a
capacity-limited point that stops at 96 may well fit 112 or 144, and the extra
resident requests are exactly where an AFD attention pool has room that a
colocated GPU does not.
"""
import csv, itertools, logging, os, sys, time

logging.getLogger("aiconfigurator").setLevel(logging.CRITICAL)
import aiconfigurator, aiconfigurator_core
from aiconfigurator.cli.api import cli_estimate

COMM_MODEL = os.environ["COMM_MODEL"]
WORKBOOK = os.environ["WORKBOOK"]            # "multi" | "m3"
OUT = os.environ["OUT"]

SYS, BE, VER = "gb200", "sglang", "0.5.14"

ATTN = {"Qwen3-235B-A22B": "GQA 64:4", "MiniMax-M2.5": "GQA 48:8",
        "DeepSeek-V3": "MLA", "DeepSeek-V3.1": "MLA", "Kimi-K2.5": "MLA",
        "MiniMax-M3": "MSA"}
GPN, MEM_CAP, RACK = 4, 185.0, 72

# 1.5x-ish geometric grid, reaching down to 1 for 128K and up for short context.
# Steps stay near x1.15 through the mid range. The coarser x1.33 rungs used
# earlier skipped the band where a capacity-limited attention pool still has
# HBM left: Qwen3-235B-A22B NVFP4 at 65536 tops out near batch 119, but a grid
# stepping 96 -> 128 stops at 96 and leaves 35 GB of the 185 GB budget unused.
BATCH = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96,
         112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384,
         416, 448, 480, 512, 576, 640, 704, 768, 896, 1024, 1152, 1280, 1536,
         1792, 2048, 2304, 2560, 3072, 3584, 4096]

if WORKBOOK == "multi":
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
else:
    # MiniMax-M3 has no collected operators of its own; NVFP4 MoE plus the
    # HYBRID database lets neighbouring kernels stand in.
    MODELS = [("MiniMax-M3", "MiniMaxAI/MiniMax-M3", "NVFP4", "nvfp4", "HYBRID"),
              ("MiniMax-M3", "MiniMaxAI/MiniMax-M3", "FP8-block", "fp8_block", "HYBRID")]
    # M3 attention is GenerationMSAModule -- linear/state attention, so the
    # per-sequence state does not grow with context and million-token inputs
    # stay reachable. That is precisely the regime worth probing here.
    CONTEXTS = [4096, 16384, 65536, 131072, 262144, 524288, 1048576]
    OSLS = [256, 1024, 4096]

# nextn_accepted is the average number of ACCEPTED DRAFT tokens per step
# (0 <= nextn_accepted <= nextn), not the per-round token progress.
# Conditional acceptance falls with draft position: 0.85 at the first, 0.80 at
# the second, 0.60 at the third. The expectation is the running product sum,
# so draft length 1 accepts 0.85 and draft length 2 accepts 0.85 + 0.85*0.80.
ACCEPT = [0.85, 0.80, 0.60]


def expected_accepted(n):
    e = p = 0.0, 1.0
    tot, run = 0.0, 1.0
    for i in range(n):
        run *= ACCEPT[i]
        tot += run
    return round(tot, 6)


MTP = [("0", {})] + [(str(n), {"nextn": n, "nextn_accepted": expected_accepted(n)})
                     for n in (1, 2)]
SPLITS = [(a, f) for f in (1, 2, 4, 8) for a in range(1, 18)
          if (a + f) * GPN <= RACK]
TP_A = [1, 2, 4]
CO_UNITS = [2, 4, 8, 16, 32, 64]

FIELDS = ["arm", "model", "attn", "precision", "moe_kernel", "system", "backend", "database_mode",
          "isl", "osl", "mtp_nextn", "config", "unit_gpus", "a_gpus", "f_gpus",
          "batch_per_unit", "concurrency", "tokens_s_gpu", "tpot_ms", "tokens_s_user",
          "memory_gb", "t_attn_us", "t_ffn_us", "t_comm_us",
          # decomposition, mirroring the reference workbook's column set
          "t_a2f_us", "t_f2a_us", "comm_hidden", "tf_over_ta", "balance_ratio",
          "t_step_ms", "a_memory_gb", "f_memory_gb"]


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
# Shard over (model, context) pairs, round-robin, so each worker gets a mix of
# cheap and expensive cells rather than one worker owning every long context.
# Optional narrowing, used to re-run a slow slice on its own.
if os.environ.get("ISL_ONLY"):
    keep = {int(v) for v in os.environ["ISL_ONLY"].split(",")}
    CONTEXTS = [c for c in CONTEXTS if c in keep]
if os.environ.get("OSL_ONLY"):
    OSLS = [int(v) for v in os.environ["OSL_ONLY"].split(",")]

TASKS = [(m, c) for m in MODELS for c in CONTEXTS]
SHARD, NSHARD = os.environ.get("SHARD"), int(os.environ.get("NSHARD", "1"))
if SHARD is not None:
    TASKS = TASKS[int(SHARD)::NSHARD]

print(f"comm_model={COMM_MODEL} workbook={WORKBOOK} cells={len(TASKS)*len(OSLS)*len(MTP)} "
      f"splits={len(SPLITS)} tp_a={len(TP_A)} colo={len(NONAFD)} batch={len(BATCH)}", flush=True)

def _f(v, nd=2):
    """Some raw fields arrive as preformatted strings; coerce or blank out."""
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return ""


fh = open(OUT, "w", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
w.writeheader()
t0 = time.time()
n_rows = 0

for ((mname, mp, prec, mq, dbm), isl), osl, (mtag, mkw) in itertools.product(TASKS, OSLS, MTP):
    base = dict(model_path=mp, system_name=SYS, backend_name=BE, backend_version=VER,
                database_mode=dbm, isl=isl, osl=osl, moe_quant_mode=mq)
    kern = "sglang_flashinfer_trtllm_moe"
    common_row = dict(model=mname, attn=ATTN[mname], precision=prec, moe_kernel=kern, system="NVIDIA GB200 NVL72",
                      backend=f"SGLang {VER}", database_mode=dbm, isl=isl, osl=osl, mtp_nextn=mtag)
    cell = time.time()

    for (tp, dp, mtp_, mep, n) in NONAFD:
        for bs in BATCH:
            try:
                d = cli_estimate(mode="static_gen", batch_size=bs, tp_size=tp, attention_dp_size=dp,
                                 moe_tp_size=mtp_, moe_ep_size=mep, **mkw, **base).raw
            except Exception:
                continue
            if d.get("memory", 1e9) > MEM_CAP or d.get("tokens/s/gpu") is None:
                continue
            w.writerow(dict(arm="colocated", config=f"tp{tp}dp{dp}-moetp{mtp_}ep{mep}",
                            unit_gpus=n, a_gpus="", f_gpus="", batch_per_unit=bs,
                            concurrency=d.get("concurrency", ""),
                            tokens_s_gpu=round(d["tokens/s/gpu"], 2), tpot_ms=round(d.get("tpot", 0), 3),
                            tokens_s_user=round(d.get("tokens/s/user", 0), 3),
                            memory_gb=round(d.get("memory", 0), 2), t_attn_us="", t_ffn_us="",
                            t_comm_us="", **common_row))
            n_rows += 1

    for (na, nf) in SPLITS:
        for tpa in TP_A:
            if GPN % tpa:
                continue
            for bs in BATCH:
                try:
                    d = cli_estimate(mode="afd", n_a_nodes=na, n_f_nodes=nf, a_tp_size=tpa,
                                     a_batch_size=bs, f_moe_ep_size=nf * GPN, num_microbatches=2,
                                     pipeline_model="conservative", afd_phase="decode",
                                     afd_combined_with_pd=False, afd_comm_model=COMM_MODEL,
                                     **mkw, **base).raw
                except Exception:
                    continue
                if d.get("memory", 1e9) > MEM_CAP or d.get("tokens/s/gpu") is None:
                    continue
                lbl = f"{na*GPN}A{nf*GPN}F" + (f"/tp_a{tpa}" if tpa > 1 else "")
                w.writerow(dict(arm="afd", config=lbl, unit_gpus=(na + nf) * GPN,
                                a_gpus=na * GPN, f_gpus=nf * GPN, batch_per_unit=bs,
                                concurrency=d.get("concurrency", ""),
                                tokens_s_gpu=round(d["tokens/s/gpu"], 2),
                                tpot_ms=round(d.get("tpot", 0), 3),
                                tokens_s_user=round(d.get("tokens/s/user", 0), 3),
                                memory_gb=round(d.get("memory", 0), 2),
                                t_attn_us=round(d.get("t_a_layer", 0) * 1000, 1),
                                t_ffn_us=round(d.get("t_f_layer", 0) * 1000, 1),
                                t_comm_us=round(d.get("t_c_layer", 0) * 1000, 1),
                                t_a2f_us=round(d.get("t_a2f_layer", 0) * 1000, 1),
                                t_f2a_us=round(d.get("t_f2a_layer", 0) * 1000, 1),
                                comm_hidden=bool(d.get("comm_hidden", False)),
                                tf_over_ta=(round(d["t_f_layer"] / d["t_a_layer"], 3)
                                            if d.get("t_a_layer") else ""),
                                balance_ratio=_f(d.get("balance_ratio"), 3),
                                t_step_ms=_f(d.get("t_step"), 3),
                                a_memory_gb=_f(d.get("(a)memory")),
                                f_memory_gb=_f(d.get("(f)memory")), **common_row))
                n_rows += 1
    fh.flush()
    print(f"[{mname:>16} {prec:>9} isl={isl:>6} osl={osl:>5} mtp={mtag}] "
          f"rows={n_rows} ({time.time()-cell:.0f}s, total {time.time()-t0:.0f}s)", flush=True)

fh.close()
print(f"DONE rows={n_rows} {time.time()-t0:.0f}s -> {OUT}")
