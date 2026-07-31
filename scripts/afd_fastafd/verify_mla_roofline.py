"""Checks on the MLA verification-roofline scale + wrapper propagation."""
import itertools
import sys

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.mla import GenerationMLA
from aiconfigurator_core.sdk.operations import contains_speculative_attention
from aiconfigurator.sdk.perf_database import get_database

print("mla.py:", sys.modules["aiconfigurator_core.sdk.operations.mla"].__file__)
db = get_database("gb200", "sglang", "0.5.10")

# ---- 1. anchor: q=1 -> exactly 1.0 ----
bad = []
for b, s, n, kv in itertools.product(
        [1, 8, 96, 512], [2, 128, 8192, 131072], [16, 64, 128],
        [common.KVCacheQuantMode.bfloat16, common.KVCacheQuantMode.fp8]):
    op = GenerationMLA("gen_mla", 1.0, n, kv)
    sc = op._verification_roofline_scale(db, batch_size=b, s=s, query_len=1)
    if abs(sc - 1.0) > 1e-12:
        bad.append((b, s, n, kv.value.name, sc))
print(f"\n[1] anchor q=1 -> scale==1.0 exactly: {'PASS' if not bad else 'FAIL'} "
      f"({len(bad)} mismatches of 96)")
for x in bad[:4]:
    print("    ", x)

# ---- 2. bounded and monotone ----
op = GenerationMLA("gen_mla", 1.0, 128, common.KVCacheQuantMode.fp8)
viol = []
for s in (1024, 8192, 32768):
    prev = 1.0
    for q in range(1, 9):
        sc = op._verification_roofline_scale(db, batch_size=96, s=s, query_len=q)
        if sc < prev - 1e-12 or sc < 1.0 - 1e-12 or sc > q + 1e-9:
            viol.append((s, q, sc))
        prev = sc
print(f"[2] monotone in q and within [1, q]: {'PASS' if not viol else 'FAIL'} {viol[:3]}")

# ---- 3. physics ----
print("\n[3] MLA scale vs context (DeepSeek-V3 shape: 128 heads, fp8 latent KV), b=96")
print(f"{'s':>8} " + " ".join(f"q={q:<7}" for q in (2, 4, 8)))
for s in (2, 8, 32, 128, 512, 2048, 8192, 32768):
    row = [op._verification_roofline_scale(db, batch_size=96, s=s, query_len=q) for q in (2, 4, 8)]
    print(f"{s:>8} " + " ".join(f"{v:<9.3f}" for v in row))
print("    batch-widening would give exactly q on every row.")

# ---- 4. does the predicate see through the DeepSeek MLA FallbackOp? ----
from aiconfigurator_core.sdk import config as cfgmod
from aiconfigurator.sdk.models import get_model
mc = cfgmod.ModelConfig(
    gemm_quant_mode=common.GEMMQuantMode.fp8_block, kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    fmha_quant_mode=common.FMHAQuantMode.bfloat16, moe_quant_mode=common.MoEQuantMode.fp8_block,
    comm_quant_mode=common.CommQuantMode.half, tp_size=4, pp_size=1, attention_dp_size=1,
    moe_tp_size=1, moe_ep_size=4)
m = get_model("deepseek-ai/DeepSeek-V3", mc, "sglang")
print("\n[4] contains_speculative_attention() over DeepSeek-V3 generation ops:")
for o in m.generation_ops:
    print(f"    {type(o).__name__:<22} {getattr(o, '_name', '?'):<28} -> {contains_speculative_attention(o)}")
