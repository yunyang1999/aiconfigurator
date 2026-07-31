"""AIC AFD vs FastAFD measured, with the MoE kernel as a CONTROLLED variable.

Every arm (AFD and colocated) uses the same model / system / backend / version /
data mode / MoE kernel; only the swept columns differ.

  model    Qwen/Qwen3-235B-A22B-FP8    system gb200 (NVL72, 4 GPU/node)
  backend  sglang 0.5.10               data   SILICON
  decode only, osl 1024, num_microbatches 2, pipeline conservative
  A pool tp=1 pure DP; F pool moe_tp=1 pure EP   (= FastAFD's ATTN_TP=1 / MLP_TP=1)

Swept: isl, A:F split, batch, comm_model {pr1323, fastafd}, moe kernel
  moe kernel  fp8_block -> sglang_fused_moe_triton      (the FP8 checkpoint dtype)
              nvfp4     -> sglang_flashinfer_cutedsl_moe (fast Blackwell kernel,
                           the closest AIC has to FastAFD's DeepGEMM/MegaMoE)
"""
import csv, logging, time

logging.getLogger("aiconfigurator").setLevel(logging.CRITICAL)
import aiconfigurator, aiconfigurator_core
from aiconfigurator.cli.api import cli_estimate

print("app :", aiconfigurator.__file__, flush=True)
print("core:", aiconfigurator_core.__file__, flush=True)

MODELS = [("Qwen3-235B", "Qwen/Qwen3-235B-A22B-FP8"), ("MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.5")]
SYSTEM, BACKEND, VERSION, DBMODE = "gb200", "sglang", "0.5.10", "SILICON"
GPN, MEM_CAP, RACK = 4, 185.0, 72

ISLS = [8192, 16384]
SPLITS = [(1, 1), (2, 1), (3, 1), (5, 1), (7, 1), (9, 1), (11, 1), (17, 1),
          (1, 2), (2, 2), (4, 2), (6, 2), (10, 2), (14, 2),
          (1, 3), (2, 4), (4, 4), (8, 4), (12, 4),
          (2, 8), (4, 8), (8, 8)]
A_BATCH = [16, 32, 48, 64, 96, 128, 192, 256]
CO_BATCH = [8, 16, 32, 48, 64, 96, 128, 192, 256]
COMM_MODELS = ["pr1323", "fastafd"]
MOE_KERNELS = [("bfloat16", "triton-bf16"), ("fp8_block", "triton-fp8"), ("nvfp4", "cutedsl-nvfp4")]
# FastAFD published per-GPU decode throughput gain over colocated, and best M:N
FASTAFD = {"Qwen3-235B": {8192: 1.41, 16384: 1.44, "mn": "7-11:1"},
           "MiniMax-M2.5": {8192: 1.45, 16384: 1.35, "mn": "17:1"}}

rows, t0 = [], time.time()
best_colo, best_afd = {}, {}

for mname, MODEL in MODELS:
  for mq, kname in MOE_KERNELS:
    for isl in ISLS:
        for bs in CO_BATCH:
            try:
                d = cli_estimate(model_path=MODEL, system_name=SYSTEM, backend_name=BACKEND,
                                 backend_version=VERSION, database_mode=DBMODE, mode="static_gen",
                                 isl=isl, osl=1024, batch_size=bs, tp_size=1,
                                 attention_dp_size=4, moe_tp_size=1, moe_ep_size=4,
                                 moe_quant_mode=mq).raw
            except Exception:
                continue
            if d.get("memory", 1e9) > MEM_CAP or d.get("tokens/s/gpu") is None:
                continue
            tg = d["tokens/s/gpu"]
            rows.append(dict(kind="colocated", model=mname, moe=kname, comm_model="-", isl=isl,
                             split="4A+F(dp4ep4)", gpus=4, batch=bs, tpot=d.get("tpot"),
                             tps_gpu=tg, mem=d.get("memory"), bal=None, t_a=None, t_f=None, t_c=None, a2f=None, f2a=None))
            k = (mname, kname, isl)
            if k not in best_colo or tg > best_colo[k]["tps_gpu"]:
                best_colo[k] = dict(tps_gpu=tg, batch=bs, tpot=d.get("tpot"))
        print(f"[colo] {mname} moe={kname} isl={isl} -> {best_colo.get((mname,kname,isl))} ({time.time()-t0:.0f}s)", flush=True)

for mname, MODEL in MODELS:
  for mq, kname in MOE_KERNELS:
    for cm in COMM_MODELS:
        for isl in ISLS:
            for (na, nf) in SPLITS:
                if (na + nf) * GPN > RACK:
                    continue
                for ab in A_BATCH:
                    try:
                        d = cli_estimate(model_path=MODEL, system_name=SYSTEM, backend_name=BACKEND,
                                         backend_version=VERSION, database_mode=DBMODE, mode="afd",
                                         isl=isl, osl=1024, n_a_nodes=na, n_f_nodes=nf, a_tp_size=1,
                                         a_batch_size=ab, f_moe_ep_size=nf * GPN, num_microbatches=2,
                                         pipeline_model="conservative", afd_phase="decode",
                                         afd_combined_with_pd=False, afd_comm_model=cm,
                                         moe_quant_mode=mq).raw
                    except Exception:
                        continue
                    if d.get("memory", 1e9) > MEM_CAP or d.get("tokens/s/gpu") is None:
                        continue
                    tg = d["tokens/s/gpu"]
                    rows.append(dict(kind="afd", model=mname, moe=kname, comm_model=cm, isl=isl,
                                     split=f"{na*GPN}A:{nf*GPN}F", gpus=(na + nf) * GPN, batch=ab,
                                     tpot=d.get("tpot"), tps_gpu=tg, mem=d.get("memory"),
                                     bal=d.get("decode_balance_ratio"), t_a=d.get("t_a_layer"),
                                     t_f=d.get("t_f_layer"), t_c=d.get("t_c_layer"),
                                     a2f=d.get("t_a2f_layer"), f2a=d.get("t_f2a_layer")))
                    k = (mname, kname, cm, isl)
                    if k not in best_afd or tg > best_afd[k]["tps_gpu"]:
                        best_afd[k] = dict(tps_gpu=tg, split=f"{na*GPN}A:{nf*GPN}F", batch=ab,
                                           tpot=d.get("tpot"), bal=d.get("decode_balance_ratio"),
                                           t_a=d.get("t_a_layer"), t_f=d.get("t_f_layer"),
                                           t_c=d.get("t_c_layer"))
            print(f"[afd] {mname} moe={kname} comm={cm} isl={isl} -> {best_afd.get((mname,kname,cm,isl))} ({time.time()-t0:.0f}s)", flush=True)

out = "/root/.claude/jobs/9f54bfc2/tmp/exp4_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\n" + "=" * 124)
print(f"{'model':>13} {'moe kernel':>11} {'isl':>6} {'comm_model':>11} | {'colo t/s/gpu':>13} | "
      f"{'AFD t/s/gpu':>12} {'split':>9} {'bal':>5} | {'AIC':>7} | {'FastAFD':>8} {'best M:N':>9}")
print("=" * 124)
for mname, _ in MODELS:
    for mq, kname in MOE_KERNELS:
        for isl in ISLS:
            c = best_colo.get((mname, kname, isl))
            for cm in COMM_MODELS:
                a = best_afd.get((mname, kname, cm, isl))
                if not c or not a:
                    print(f"{mname:>13} {kname:>11} {isl:>6} {cm:>11} |  (missing)"); continue
                print(f"{mname:>13} {kname:>11} {isl:>6} {cm:>11} | {c['tps_gpu']:>13.1f} | "
                      f"{a['tps_gpu']:>12.1f} {a['split']:>9} {a['bal']:>5.2f} | "
                      f"{a['tps_gpu']/c['tps_gpu']:>6.2f}x | {FASTAFD[mname][isl]:>7.2f}x {FASTAFD[mname]['mn']:>9}")
print("=" * 124)
print(f"rows={len(rows)} csv={out} elapsed={time.time()-t0:.0f}s")
