"""Absolute check against FastAFD's one published step latency, plus the
speedup identity decomposed factor by factor.

FastAFD blog states:
  * Qwen3-235B, 4-node (3 attn + 1 FFN node = 12A:4F), 8K: decode step = 32.826 ms
  * per-GPU speedup = M/(M+N)  x  B_AFD/B_vLLM  x  T_vLLM/T_AFD
  * "attention GPUs shed more than 90% of the weights, carry 1.5x more
    resident requests"  -> the batch factor is ~1.5 and the latency factor ~1
  * best M:N for Qwen3 is 7-11:1; gain 1.41x @8K, 1.44x @16K
"""
import logging
import aiconfigurator, aiconfigurator_core
logging.getLogger("aiconfigurator").setLevel(logging.CRITICAL)
from aiconfigurator.cli.api import cli_estimate

print("app :", aiconfigurator.__file__)
print("core:", aiconfigurator_core.__file__)

MODEL = "Qwen/Qwen3-235B-A22B-FP8"
BASE = dict(model_path=MODEL, system_name="gb200", backend_name="sglang",
            backend_version="0.5.10", database_mode="SILICON", isl=8192, osl=1024)
MEM_CAP = 185.0
MEASURED_STEP_MS = 32.826   # FastAFD, Qwen3-235B, 4-node, 8K

def afd(na, nf, ab, mq, cm="fastafd", **kw):
    return cli_estimate(mode="afd", n_a_nodes=na, n_f_nodes=nf, a_tp_size=1,
                        a_batch_size=ab, f_moe_ep_size=nf * 4, num_microbatches=2,
                        pipeline_model="conservative", afd_phase="decode",
                        afd_combined_with_pd=False, afd_comm_model=cm,
                        moe_quant_mode=mq, **kw, **BASE).raw

def colo(bs, mq):
    return cli_estimate(mode="static_gen", batch_size=bs, tp_size=1, attention_dp_size=4,
                        moe_tp_size=1, moe_ep_size=4, moe_quant_mode=mq, **BASE).raw

# ---------------------------------------------------------------- 1. absolute step latency
print(f"\n=== [1] AIC decode-step latency at FastAFD's exact 4-node 8K point ===")
print(f"    (3 attn nodes + 1 FFN node = 12A:4F, batch 96 per A-GPU, mb=2)")
print(f"    FastAFD measured: {MEASURED_STEP_MS} ms\n")
print(f"{'moe kernel':>14} {'comm model':>11} {'AIC step ms':>12} {'ratio to measured':>18}")
for mq, kname in (("fp8_block", "triton-fp8"), ("nvfp4", "cutedsl-nvfp4")):
    for cm in ("pr1323", "fastafd"):
        try:
            d = afd(3, 1, 96, mq, cm)
            t = d["tpot"]
            print(f"{kname:>14} {cm:>11} {t:>12.3f} {t/MEASURED_STEP_MS:>17.2f}x")
        except Exception as e:
            print(f"{kname:>14} {cm:>11}   FAILED {type(e).__name__}")

# ---------------------------------------------------------------- 2. is the baseline KV-limited?
print(f"\n=== [2] does AIC's colocated baseline hit the KV wall? (cap {MEM_CAP} GB) ===")
print(f"{'moe':>14} {'batch/GPU':>10} {'mem GB':>8} {'tpot ms':>9} {'tok/s/gpu':>10}")
for mq, kname in (("fp8_block", "triton-fp8"), ("nvfp4", "cutedsl-nvfp4")):
    for bs in (64, 96, 128, 192, 256, 384, 512):
        try:
            d = colo(bs, mq)
        except Exception:
            print(f"{kname:>14} {bs:>10}   (error)"); continue
        flag = "  <-- OOM" if d.get("memory", 0) > MEM_CAP else ""
        print(f"{kname:>14} {bs:>10} {d.get('memory', 0):>8.1f} {d.get('tpot', 0):>9.2f} "
              f"{d.get('tokens/s/gpu', 0):>10.1f}{flag}")
    print()

# ---------------------------------------------------------------- 3. decompose the identity
print("=== [3] speedup identity, AIC vs FastAFD ===")
print("    speedup = M/(M+N) x B_AFD/B_vLLM x T_vLLM/T_AFD\n")
print(f"{'moe':>14} {'split':>9} {'M/(M+N)':>8} {'B_AFD':>6} {'B_vLLM':>7} {'batch f':>8} "
      f"{'T_vLLM':>8} {'T_AFD':>8} {'lat f':>7} {'product':>8} {'actual':>7}")
for mq, kname in (("fp8_block", "triton-fp8"), ("nvfp4", "cutedsl-nvfp4")):
    # best colocated under the memory cap
    best_c = None
    for bs in (32, 64, 96, 128, 192, 256, 384, 512):
        try:
            d = colo(bs, mq)
        except Exception:
            continue
        if d.get("memory", 1e9) > MEM_CAP or d.get("tokens/s/gpu") is None:
            continue
        if best_c is None or d["tokens/s/gpu"] > best_c["tokens/s/gpu"]:
            best_c = dict(tokens_per_s_gpu=d["tokens/s/gpu"], bs=bs, tpot=d["tpot"],
                          **{"tokens/s/gpu": d["tokens/s/gpu"]})
    for (na, nf) in ((3, 1), (5, 1), (7, 1), (9, 1), (11, 1), (17, 1)):
        best_a = None
        for ab in (32, 64, 96, 128, 192, 256, 384):
            try:
                d = afd(na, nf, ab, mq)
            except Exception:
                continue
            if d.get("memory", 1e9) > MEM_CAP or d.get("tokens/s/gpu") is None:
                continue
            if best_a is None or d["tokens/s/gpu"] > best_a["tokens/s/gpu"]:
                best_a = dict(d); best_a["_bs"] = ab
        if best_a is None or best_c is None:
            continue
        M, N = na, nf
        tax = M / (M + N)
        bf = best_a["_bs"] / best_c["bs"]
        lf = best_c["tpot"] / best_a["tpot"]
        prod = tax * bf * lf
        actual = best_a["tokens/s/gpu"] / best_c["tokens/s/gpu"]
        print(f"{kname:>14} {na*4}A:{nf*4}F".rjust(24)
              + f" {tax:>8.3f} {best_a['_bs']:>6} {best_c['bs']:>7} {bf:>8.2f} "
                f"{best_c['tpot']:>8.2f} {best_a['tpot']:>8.2f} {lf:>7.2f} {prod:>8.2f} {actual:>6.2f}x")
    print()
print("FastAFD reference: Qwen3-235B 1.41x @8K at best M:N in 7-11:1,")
print("with batch factor ~1.5 (attention GPUs shed >90% of weights) and latency factor ~1.")
