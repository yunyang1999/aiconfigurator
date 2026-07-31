"""Full GB200 homogeneous AFD sweep for DeepSeek-V3, native data only.

Controlled: model deepseek-ai/DeepSeek-V3, system gb200 (NVL72, 4 GPU/node),
backend sglang 0.5.10, database SILICON (no SOL/empirical fallback), decode
phase, osl 1024, num_microbatches 2, conservative pipeline, A pool tp=1 pure DP,
F pool moe_tp=1 pure EP, comm model "fastafd".

DeepSeek-V3's shape (h=7168, inter=2048, E=256, topk=8) matches the collected
`deepepmoe` wide-EP table exactly, so the F pool is priced with the real DeepEP
MoE kernel rather than a generic one. No afd_moe_time_ms anywhere.

Swept: A:F split, per-A-GPU batch, and MTP on/off.
Baseline: colocated (non-AFD) decode at the same per-GPU accounting.
"""
import csv, logging, time

logging.getLogger("aiconfigurator").setLevel(logging.CRITICAL)
import aiconfigurator, aiconfigurator_core
from aiconfigurator.cli.api import cli_estimate

print("app :", aiconfigurator.__file__, flush=True)
print("core:", aiconfigurator_core.__file__, flush=True)

MODEL = "deepseek-ai/DeepSeek-V3"
SYS, BE, VER, DB = "gb200", "sglang", "0.5.10", "SILICON"
GPN, CAP, RACK = 4, 185.0, 72
ISL, OSL = 8192, 1024
MQ = "fp8_block"           # the FP8 checkpoint dtype; matches the deepepmoe table
MTP = [("no-mtp", {}), ("mtp-n3", {"nextn": 3, "nextn_accepted": 2.28125})]

SPLITS = [(a, f) for f in (2, 4, 8) for a in (1, 2, 3, 5, 7, 9, 11, 13, 15, 17)
          if (a + f) * GPN <= RACK]
A_BATCH = [32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
# colocated: DSV3 FP8 weights do not fit on one 4-GPU node, so the baseline is
# the smallest colocated replica that holds the model, swept over tp/ep.
# DP attention (tp=1, attention_dp=N, moe_ep=N) -- the layout real DSV3 serving
# uses. tp>1 with attention_dp=1 replicates MLA's latent KV N times and would
# hand AFD an unearned win, since the A pool runs tp_a=1 pure DP.
CO = [4, 8, 16, 32, 64]
CO_BATCH = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]

rows, t0 = [], time.time()
best_co, best_afd = {}, {}

for tag, mkw in MTP:
    for n in CO:
        for bs in CO_BATCH:
            try:
                d = cli_estimate(model_path=MODEL, system_name=SYS, backend_name=BE,
                                 backend_version=VER, database_mode=DB, mode="static_gen",
                                 isl=ISL, osl=OSL, batch_size=bs, tp_size=1,
                                 attention_dp_size=n, moe_tp_size=1, moe_ep_size=n,
                                 moe_quant_mode=MQ, **mkw).raw
            except Exception:
                continue
            if d.get("memory", 1e9) > CAP or d.get("tokens/s/gpu") is None:
                continue
            tg = d["tokens/s/gpu"]
            rows.append(dict(kind="colocated", mtp=tag, split=f"dp{n}-ep{n}", gpus=n,
                             batch=bs, tpot=d.get("tpot"), tps_gpu=tg, mem=d.get("memory"),
                             bal=None, t_a=None, t_f=None, t_c=None))
            if tag not in best_co or tg > best_co[tag]["tps_gpu"]:
                best_co[tag] = dict(tps_gpu=tg, split=f"dp{n}-ep{n}", batch=bs, tpot=d["tpot"],
                                    mem=d["memory"])
    print(f"[colo] {tag} -> {best_co.get(tag)} ({time.time()-t0:.0f}s)", flush=True)

for tag, mkw in MTP:
    for (na, nf) in SPLITS:
        for ab in A_BATCH:
            try:
                d = cli_estimate(model_path=MODEL, system_name=SYS, backend_name=BE,
                                 backend_version=VER, database_mode=DB, mode="afd",
                                 isl=ISL, osl=OSL, n_a_nodes=na, n_f_nodes=nf, a_tp_size=1,
                                 a_batch_size=ab, f_moe_ep_size=nf * GPN, num_microbatches=2,
                                 pipeline_model="conservative", afd_phase="decode",
                                 afd_combined_with_pd=False, afd_comm_model="fastafd",
                                 moe_quant_mode=MQ, **mkw).raw
            except Exception:
                continue
            if d.get("memory", 1e9) > CAP or d.get("tokens/s/gpu") is None:
                continue
            tg = d["tokens/s/gpu"]
            rows.append(dict(kind="afd", mtp=tag, split=f"{na*GPN}A:{nf*GPN}F",
                             gpus=(na + nf) * GPN, batch=ab, tpot=d.get("tpot"), tps_gpu=tg,
                             mem=d.get("memory"), bal=d.get("decode_balance_ratio"),
                             t_a=d.get("t_a_layer"), t_f=d.get("t_f_layer"), t_c=d.get("t_c_layer")))
            if tag not in best_afd or tg > best_afd[tag]["tps_gpu"]:
                best_afd[tag] = dict(tps_gpu=tg, split=f"{na*GPN}A:{nf*GPN}F", batch=ab,
                                     tpot=d["tpot"], bal=d.get("decode_balance_ratio"),
                                     t_a=d.get("t_a_layer"), t_f=d.get("t_f_layer"),
                                     t_c=d.get("t_c_layer"), gpus=(na + nf) * GPN)
    print(f"[afd] {tag} -> {best_afd.get(tag)} ({time.time()-t0:.0f}s)", flush=True)

out = "/root/.claude/jobs/9f54bfc2/tmp/exp7c_dsv3_results.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("\n" + "=" * 108)
print(f"{'MTP':>8} | {'colo t/s/gpu':>13} {'cfg':>10} {'bs':>5} | "
      f"{'AFD t/s/gpu':>12} {'split':>9} {'bs':>5} {'bal':>5} {'gpus':>5} | {'AFD/colo':>9}")
print("=" * 108)
for tag, _ in MTP:
    c, a = best_co.get(tag), best_afd.get(tag)
    if not c or not a:
        print(f"{tag:>8} | (missing)"); continue
    print(f"{tag:>8} | {c['tps_gpu']:>13.1f} {c['split']:>10} {c['batch']:>5} | "
          f"{a['tps_gpu']:>12.1f} {a['split']:>9} {a['batch']:>5} {a['bal']:>5.2f} {a['gpus']:>5} | "
          f"{a['tps_gpu']/c['tps_gpu']:>8.2f}x")
print("=" * 108)
if "no-mtp" in best_afd and "mtp-n3" in best_afd:
    print(f"MTP gain, AFD:  {best_afd['mtp-n3']['tps_gpu']/best_afd['no-mtp']['tps_gpu']:.2f}x")
    print(f"MTP gain, colo: {best_co['mtp-n3']['tps_gpu']/best_co['no-mtp']['tps_gpu']:.2f}x")
print(f"rows={len(rows)} csv={out} elapsed={time.time()-t0:.0f}s")
