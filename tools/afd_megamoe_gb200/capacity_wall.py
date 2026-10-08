#!/usr/bin/env python3
"""Where the capacity wall actually is, independent of the batch ladder.

The batch grid is the MEASURED grid -- 1, 2, 4, 6, 8, 16, 32, 48, 64, 96, 128,
192, 256, 384 sequences per rank -- because the injected MoE stage has to be an
exact lookup. That is right for throughput, but it distorts any statement about
CAPACITY: a configuration whose real wall sits at batch 108 is reported at 64,
because 64 is the largest rung below it and the next one is 128.

The distortion is not symmetric between the arms, so it lands directly on the
comparison. Qwen3-235B, isl=65536, 50 ms limit:

                      best rung   memory     seq/GPU     true wall   seq/GPU
    colocated ep8/tp4   64        115.3 GiB    16.00        108        27.00
    AFD 68A4F/tp_a4     96        150.1 GiB    22.67        118        27.86
                                          ratio  1.417                 1.032

The sheet reads a 41.7% concurrency advantage where the hardware gives 3.2%. The
colocated arm was never short of memory at batch 64 -- it was short of RUNGS,
sitting at 62% of the card.

This module finds the true wall by bisection. It needs only the MEMORY model, so
it is free of the ladder: static_gen reports a memory breakdown at any batch, and
the AFD path has AIC raise its own OOM. No MoE time is involved, so nothing here
leaves the measured envelope -- the throughput numbers in the workbook are
untouched and still come from exact lookups.

What it cannot do is tell you the THROUGHPUT at the true wall: that would need
the MoE stage at batch 108, which is a measurement we have not taken. Refining
the ladder around the wall is submitted separately (submit_batchfine.sh); until
those land, this sheet bounds the ladder artefact rather than removing it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from afd_paths import data_root

# Data root: the unpacked data bundle, from AFD_MEGAMOE_ROOT (required).
ROOT = data_root()
sys.path.insert(0, str(ROOT))

GIB = 1024 ** 3


# Accepted draft lengths, same table the sweep uses. Omitting this is what made
# every nextN>0 colocated wall fail: AIC needs the acceptance rate alongside the
# draft length, the call raised, and colo_wall's except clause turned that into
# "no wall". 184 of 184 unresolved walls were colocated and every one of them was
# nextN 1 or 2, which is exactly the regime where AFD wins most often -- so the
# rows that most needed the ladder check were the ones silently missing it.
NEXTN_ACCEPTED = {0: None, 1: 0.85, 2: 1.53}


def _colo_memory(est, model_path, ep, tp, batch, isl, osl, mtp):
    kw = {}
    acc = NEXTN_ACCEPTED.get(mtp)
    if acc is not None:
        kw["nextn_accepted"] = acc
    r = est(model_path=model_path, system_name="gb200_g4", backend_name="sglang",
            backend_version="0.5.12", mode="static_gen",
            moe_quant_mode="w4a8_mxfp4_mxfp8", database_mode="HYBRID",
            isl=isl, osl=osl, nextn=mtp, tp_size=tp, attention_dp_size=ep // tp,
            moe_ep_size=ep, batch_size=batch, **kw)
    m = r.summary.get_memory() or {}
    t = float(m.get("total") or 0.0)
    return (t / GIB if t > 1e6 else t), r.concurrency


def colo_wall(est, model_path, ep, tp, isl, osl, mtp, cap, hi=4096):
    """Largest batch whose memory fits, by bisection. Returns (batch, concurrency)."""
    step = tp                      # batch must divide into tp-way groups
    lo = step
    # A raise here used to be swallowed as "no wall". It is now re-raised so the
    # caller records it as an error rather than silently dropping the column.
    if _colo_memory(est, model_path, ep, tp, lo, isl, osl, mtp)[0] > cap:
        return None, None
    while True:
        nxt = lo * 2
        if nxt > hi:
            break
        try:
            if _colo_memory(est, model_path, ep, tp, nxt, isl, osl, mtp)[0] > cap:
                break
        except Exception:
            break
        lo = nxt
    high = min(lo * 2, hi)
    while high - lo > step:
        mid = ((lo + high) // 2 // step) * step
        if mid <= lo:
            break
        try:
            fits = _colo_memory(est, model_path, ep, tp, mid, isl, osl, mtp)[0] <= cap
        except Exception:
            fits = False
        if fits:
            lo = mid
        else:
            high = mid
    _, conc = _colo_memory(est, model_path, ep, tp, lo, isl, osl, mtp)
    return lo, conc


def afd_wall(run_once, model, a, f, tp, g, isl, osl, mtp, mb, hi=4096):
    """Largest batch AIC accepts before raising OOM. Returns (batch, concurrency)."""
    step = mb                      # batch must divide by the microbatch count
    lo = step

    def ok(b):
        try:
            run_once(model, a, f, tp, g, b, isl, osl, mtp, 10.0, mb)
            return True
        except Exception as e:
            return "OOM" not in str(e)

    if not ok(lo):
        return None, None
    while True:
        nxt = lo * 2
        if nxt > hi or not ok(nxt):
            break
        lo = nxt
    high = min(lo * 2, hi)
    while high - lo > step:
        mid = ((lo + high) // 2 // step) * step
        if mid <= lo:
            break
        if ok(mid):
            lo = mid
        else:
            high = mid
    return lo, lo * (a // tp)


if __name__ == "__main__":
    import run_gb200_measured as R
    R._est()
    from colocated_measured import MODELS
    cap = R.CAPACITY_GIB
    print(f"capacity {cap:.2f} GiB, Qwen3-235B isl=65536 osl=1024 nextN=0\n")
    b, c = colo_wall(R._est(), MODELS["qwen3_235b_fp4"]["path"], 8, 4, 65536, 1024, 0, cap)
    print(f"  colocated ep8/tp4  wall batch {b}  concurrency {c}  {c/8:.2f} seq/GPU")
    b2, c2 = afd_wall(R._afd_once, "qwen3_235b_fp4", 68, 4, 4, 4, 65536, 1024, 0, 2)
    print(f"  AFD 68A4F/tp_a4    wall batch {b2}  concurrency {c2}  {c2/72:.2f} seq/GPU")
    print(f"\n  continuous-capacity concurrency ratio {(c2/72)/(c/8):.3f}")
