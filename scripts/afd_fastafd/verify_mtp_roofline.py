"""Correctness checks on the MTP verification-roofline scale."""
import itertools
import sys

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.attention import GenerationAttention
from aiconfigurator.sdk.perf_database import get_database

print("attention.py:", sys.modules["aiconfigurator_core.sdk.operations.attention"].__file__)
db = get_database("gb200", "sglang", "0.5.10")

def make(n=64, n_kv=4, head_size=128, window=0, kv=common.KVCacheQuantMode.bfloat16):
    return GenerationAttention("gen_attn", 1.0, n, n_kv, kv, window_size=window, head_size=head_size)

# ---- 1. anchor: q=1 must give exactly 1.0 ----
bad = []
for b, s, w, kv in itertools.product(
        [1, 8, 96, 512], [2, 128, 8192, 131072], [0, 4096],
        [common.KVCacheQuantMode.bfloat16, common.KVCacheQuantMode.fp8]):
    op = make(window=w, kv=kv)
    sc = op._verification_roofline_scale(db, batch_size=b, s=s, query_len=1)
    if abs(sc - 1.0) > 1e-12:
        bad.append((b, s, w, kv.value.name, sc))
print(f"\n[1] anchor q=1 -> scale==1.0 exactly: {'PASS' if not bad else 'FAIL'} "
      f"({len(bad)} mismatches out of 64)")
for x in bad[:5]:
    print("    ", x)

# ---- 2. monotone in q, and bounded by [1, q] ----
op = make()
viol = []
for b, s in itertools.product([96], [1024, 8192, 32768]):
    prev = 1.0
    for q in range(1, 9):
        sc = op._verification_roofline_scale(db, batch_size=b, s=s, query_len=q)
        if sc < prev - 1e-12 or sc < 1.0 - 1e-12 or sc > q + 1e-9:
            viol.append((b, s, q, sc))
        prev = sc
print(f"[2] monotone non-decreasing in q and within [1, q]: {'PASS' if not viol else 'FAIL'}")
for x in viol[:5]:
    print("    ", x)

# ---- 3. physics: memory-bound long context -> scale ~1; compute-bound short -> ~q ----
print("\n[3] scale vs context length, Qwen3-235B head config (n=64, n_kv=4, hs=128, bf16 KV), b=96")
print(f"{'s':>8} " + " ".join(f"q={q:<6}" for q in (2, 4, 8)))
for s in (2, 8, 32, 128, 512, 2048, 8192, 32768, 131072):
    row = [op._verification_roofline_scale(db, batch_size=96, s=s, query_len=q) for q in (2, 4, 8)]
    print(f"{s:>8} " + " ".join(f"{v:<8.3f}" for v in row))

print("\n    batch-widening (what PR#1323 / base_backend did) would give exactly q "
      "for every row above.")

# ---- 4. sliding window: union of prefix positions must not exceed the window ----
print("\n[4] SWA window=4096, s=32768, b=96")
opw = make(window=4096)
for q in (1, 2, 4, 8):
    print(f"    q={q}: scale={opw._verification_roofline_scale(db, batch_size=96, s=32768, query_len=q):.4f}")

# ---- 5. which attention classes actually receive query_len? ----
import inspect
from aiconfigurator_core.sdk import operations as ops_mod
print("\n[5] classes whose query() accepts query_len:")
for name in dir(ops_mod):
    cls = getattr(ops_mod, name)
    if inspect.isclass(cls) and hasattr(cls, "query"):
        try:
            src = inspect.getsource(cls.query)
        except (OSError, TypeError):
            continue
        if "query_len" in src:
            print("    ", name)
