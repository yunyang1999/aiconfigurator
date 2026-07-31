"""AIC predictions for exactly the configs FastAFD's published presets run.

Preset semantics, read from
scripts/experiments/afd/qwen3_235b/run_afd_..._8k_b96_dynamicnode_mb2_...sh:
  AFD_TOTAL_NODES=N, last Ray node = MLP, the rest = attention
  AFD_GPUS_PER_NODE=4, ATTN_TP_SIZE=1 (pure DP), MLP_TP_SIZE=1, AFD_MLP_EP_SIZE=4
  8K preset PER_ATTN_GPU_BSZ=96, 16K preset 48, AFD_NUM_MB=2
Baseline (mocker doc): independent four-GPU replicas, dp4/ep4.
"""
import csv, logging
logging.getLogger("aiconfigurator").setLevel(logging.CRITICAL)
import aiconfigurator, aiconfigurator_core
from aiconfigurator.cli.api import cli_estimate

print("app :", aiconfigurator.__file__)
print("core:", aiconfigurator_core.__file__)

SYS, BE, VER, DB = "gb200", "sglang", "0.5.10", "SILICON"
GPN, CAP = 4, 185.0
MODELS = [("Qwen3-235B", "Qwen/Qwen3-235B-A22B-FP8"),
          ("MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.5")]
WORKLOADS = [(8192, 96), (16384, 48)]
TOTAL_NODES = [4, 6, 8, 10, 12, 14, 18]
KERNELS = [("fp8_block", "triton-fp8"), ("nvfp4", "cutedsl-nvfp4")]

rows = []
base = {}
for mname, mp in MODELS:
    for mq, kname in KERNELS:
        for isl, _b in WORKLOADS:
            best = None
            for bs in (32, 64, 96, 128, 160, 192, 256):
                try:
                    d = cli_estimate(model_path=mp, system_name=SYS, backend_name=BE,
                                     backend_version=VER, database_mode=DB, mode="static_gen",
                                     isl=isl, osl=1024, batch_size=bs, tp_size=1,
                                     attention_dp_size=4, moe_tp_size=1, moe_ep_size=4,
                                     moe_quant_mode=mq).raw
                except Exception:
                    continue
                if d.get("memory", 1e9) > CAP or d.get("tokens/s/gpu") is None:
                    continue
                if best is None or d["tokens/s/gpu"] > best["tokens/s/gpu"]:
                    best = dict(d); best["_bs"] = bs
            if best:
                base[(mname, kname, isl)] = best
                rows.append(dict(kind="colocated-baseline", model=mname, moe_kernel=kname,
                                 isl=isl, total_nodes=1, split="4GPU dp4/ep4",
                                 gpus=4, batch_per_gpu=best["_bs"],
                                 pred_step_ms=round(best["tpot"], 3),
                                 pred_tok_s_gpu=round(best["tokens/s/gpu"], 1),
                                 pred_ratio_vs_colo=1.0, pred_t_a=None, pred_t_f=None,
                                 pred_t_c=None, pred_balance=None,
                                 pred_mem_gb=round(best["memory"], 1),
                                 measured_step_ms="", measured_tok_s_gpu="", notes=""))

for mname, mp in MODELS:
    for mq, kname in KERNELS:
        for isl, ab in WORKLOADS:
            b = base.get((mname, kname, isl))
            for N in TOTAL_NODES:
                na, nf = N - 1, 1
                if (na + nf) * GPN > 72:
                    continue
                try:
                    d = cli_estimate(model_path=mp, system_name=SYS, backend_name=BE,
                                     backend_version=VER, database_mode=DB, mode="afd",
                                     isl=isl, osl=1024, n_a_nodes=na, n_f_nodes=nf,
                                     a_tp_size=1, a_batch_size=ab, f_moe_ep_size=nf * GPN,
                                     num_microbatches=2, pipeline_model="conservative",
                                     afd_phase="decode", afd_combined_with_pd=False,
                                     afd_comm_model="fastafd", moe_quant_mode=mq).raw
                except Exception as e:
                    rows.append(dict(kind="afd", model=mname, moe_kernel=kname, isl=isl,
                                     total_nodes=N, split=f"{na*GPN}A:{nf*GPN}F",
                                     gpus=(na + nf) * GPN, batch_per_gpu=ab,
                                     pred_step_ms="", pred_tok_s_gpu="", pred_ratio_vs_colo="",
                                     pred_t_a="", pred_t_f="", pred_t_c="", pred_balance="",
                                     pred_mem_gb="", measured_step_ms="", measured_tok_s_gpu="",
                                     notes=f"AIC failed: {type(e).__name__}"))
                    continue
                oom = d.get("memory", 0) > CAP
                rows.append(dict(kind="afd", model=mname, moe_kernel=kname, isl=isl,
                                 total_nodes=N, split=f"{na*GPN}A:{nf*GPN}F",
                                 gpus=(na + nf) * GPN, batch_per_gpu=ab,
                                 pred_step_ms=round(d["tpot"], 3),
                                 pred_tok_s_gpu=round(d["tokens/s/gpu"], 1),
                                 pred_ratio_vs_colo=(round(d["tokens/s/gpu"] / b["tokens/s/gpu"], 3)
                                                     if b else ""),
                                 pred_t_a=d.get("t_a_layer"), pred_t_f=d.get("t_f_layer"),
                                 pred_t_c=d.get("t_c_layer"),
                                 pred_balance=d.get("decode_balance_ratio"),
                                 pred_mem_gb=round(d.get("memory", 0), 1),
                                 measured_step_ms="", measured_tok_s_gpu="",
                                 notes="PREDICTED OOM" if oom else ""))

out = "/root/.claude/jobs/9f54bfc2/tmp/aic_predictions.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

hdr = f"{'model':>13} {'kernel':>14} {'isl':>6} {'N':>3} {'split':>9} {'bs':>4} | {'step ms':>8} {'tok/s/gpu':>10} {'vs colo':>8} {'bal':>5}"
print("\n" + hdr); print("-" * len(hdr))
for r in rows:
    if r["kind"] != "afd" or r["moe_kernel"] != "cutedsl-nvfp4" or r["model"] != "Qwen3-235B":
        continue
    print(f"{r['model']:>13} {r['moe_kernel']:>14} {r['isl']:>6} {r['total_nodes']:>3} "
          f"{r['split']:>9} {r['batch_per_gpu']:>4} | {str(r['pred_step_ms']):>8} "
          f"{str(r['pred_tok_s_gpu']):>10} {str(r['pred_ratio_vs_colo']):>8} {str(r['pred_balance']):>5}")
print(f"\nrows={len(rows)} -> {out}")
for k, v in base.items():
    print(f"baseline {k}: {v['tokens/s/gpu']:.1f} tok/s/gpu @bs={v['_bs']} step={v['tpot']:.2f}ms")
