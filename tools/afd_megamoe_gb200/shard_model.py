#!/usr/bin/env python3
"""What a finer expert shard would cost -- a BOUND, not a measurement.

B200 puts at most 8 GPUs in one NVLink domain, so the finest expert shard the
campaign could ever produce is num_experts/8: 16 on Qwen3 and MiniMax-M3, 32 on
MiniMax-M2.5. GB200 NVL72 has 72 GPUs in the domain, so ep16/32/64 and AFD F
pools of 8 or 16 are legal there and were never benchmarked here.

Expanding the DOMAIN does not by itself require a finer shard -- 9 replicas of
ep8, or one 68A4F unit, fill 72 GPUs using only measured per-GPU loads, and that
is what the main sweep does. This module answers the separate question: what if
you also spread the experts more thinly?

THE MODEL
At a fixed number of sequences per rank, changing ep changes the expert shard
but NOT the assignments each rank computes: assignments = tokens_per_rank *
top_k, independent of ep. So along that axis the only moving part is how many
experts a rank must read, and with the balanced routing the benchmark uses that
is exactly

    active_experts_per_rank = (num_experts / ep) * min(1, total_tokens * top_k / num_experts)

The stage is then fitted linearly in that quantity, per (model, tokens per rank).

WHY IT IS A BOUND AND NOT AN ANSWER
Hold out the finest measured shard, fit on the rest, predict it:

    colocated   Qwen3        median  2.9%      MiniMax-M3   median  3.8%
                MiniMax-M2.5 median 32.3%
    split       Qwen3        median  1.7%  (p90 46.2%)
                MiniMax-M2.5 median 24.4%

and the split-kernel extrapolation to eg=8 lands 7-36% BELOW the colocated
measurement at the same shard. Every check that fails, fails in the same
direction: the linear form under-predicts the fine-shard time, because the curve
flattens onto a floor the coarse-shard points cannot see. Pushed further it goes
openly wrong -- MiniMax-M3 at 8 tokens/rank extrapolates to 0.71 ms at ep64,
which is 12 microseconds for a 57-layer MoE stage.

So these numbers OVERSTATE the benefit of finer sharding. That matters for the
conclusion, because a finer shard is exactly what would make an AFD F pool cheap:
the F pool is stuck at 32 experts/GPU while colocated reaches 16. Treating this
sheet as a result would manufacture an AFD advantage the data does not support.
Read it as "the benefit is at most this much", and settle it with GB200 runs.
"""
from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

from megamoe_table import COLO_DIRS, SPLIT_DIRS, ROOT
from stability import accept


def _active(ne, topk, ep, total_tokens):
    return (ne / ep) * min(1.0, total_tokens * topk / ne)


def _fit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return None
    a = (n * sxy - sx * sy) / den
    return a, (sy - a * sx) / n


class ShardModel:
    """Per (model, tokens-per-rank) linear fit in active experts, both kernels."""

    def __init__(self, root: Path = ROOT):
        self.mp = {}
        self.colo = collections.defaultdict(dict)   # (model, tpr) -> {ep: ms}
        self.split = collections.defaultdict(dict)  # (model, seq, ag) -> {eg: ms}
        for d in COLO_DIRS:
            p0 = root / "results" / d
            if not p0.is_dir():
                continue
            for p in sorted(p0.glob("*.json")):
                j = json.loads(p.read_text())
                if not accept(j):
                    continue
                w, t, mp = j["workload"], j["topology"], j["model_profile"]
                if "ep_size" not in t or w["mtp_nextn"]:
                    continue
                self.mp[mp["key"]] = mp
                self.colo[(mp["key"], w["logical_tokens_per_rank"])][t["ep_size"]] = \
                    j["stage_cuda"]["p50_ms"]
        for d in SPLIT_DIRS:
            p0 = root / "results" / d
            if not p0.is_dir():
                continue
            for p in sorted(p0.glob("*.json")):
                j = json.loads(p.read_text())
                if not accept(j):
                    continue
                w, t, mp = j["workload"], j["topology"], j["model_profile"]
                if "ag_size" not in t or w.get("microbatches") != 1 or w["mtp_nextn"]:
                    continue
                self.mp.setdefault(mp["key"], mp)
                self.split[(mp["key"], w["sequences_per_ag_rank"], t["ag_size"])][
                    t["eg_size"]] = j["stage_cuda"]["p50_ms"]
        self.loo = self._validate()

    def _validate(self):
        """Hold out the finest measured shard, fit on the rest, predict it."""
        out = {}
        for kernel, store, tot in (("colo", self.colo, lambda k, e: k[1] * e),
                                   ("split", self.split, lambda k, e: k[1] * k[2])):
            err = collections.defaultdict(list)
            for k, d in store.items():
                if len(d) < 3:
                    continue
                mp = self.mp[k[0]]
                hold = max(d)
                tr = [e for e in d if e != hold]
                f = _fit([_active(mp["num_routed_experts"], mp["routed_top_k"], e, tot(k, e))
                          for e in tr], [d[e] for e in tr])
                if not f:
                    continue
                a, c = f
                pred = a * _active(mp["num_routed_experts"], mp["routed_top_k"],
                                   hold, tot(k, hold)) + c
                err[k[0]].append(abs(pred - d[hold]) / d[hold] * 100)
            out[kernel] = {m: dict(n=len(v), median=st.median(v),
                                   p90=sorted(v)[int(len(v) * 0.9)], max=max(v))
                           for m, v in err.items() if v}
        return out

    def predict_colo(self, model, ep, tpr):
        d = self.colo.get((model, tpr))
        if not d or len(d) < 2:
            return None
        if ep in d:
            return d[ep]
        mp = self.mp[model]
        f = _fit([_active(mp["num_routed_experts"], mp["routed_top_k"], e, tpr * e)
                  for e in sorted(d)], [d[e] for e in sorted(d)])
        if not f:
            return None
        a, c = f
        v = a * _active(mp["num_routed_experts"], mp["routed_top_k"], ep, tpr * ep) + c
        return v if v > 0 else None

    def predict_split(self, model, eg, seq, ag):
        d = self.split.get((model, seq, ag))
        if not d or len(d) < 2:
            return None
        if eg in d:
            return d[eg]
        mp = self.mp[model]
        tot = seq * ag
        f = _fit([_active(mp["num_routed_experts"], mp["routed_top_k"], e, tot)
                  for e in sorted(d)], [d[e] for e in sorted(d)])
        if not f:
            return None
        a, c = f
        v = a * _active(mp["num_routed_experts"], mp["routed_top_k"], eg, tot) + c
        return v if v > 0 else None


if __name__ == "__main__":
    M = ShardModel()
    print("leave-one-out on the finest measured shard")
    for kernel, per in M.loo.items():
        for m, s in sorted(per.items()):
            print(f"  {kernel:<6} {m:<18} n={s['n']:<3} median {s['median']:5.1f}%  "
                  f"p90 {s['p90']:5.1f}%  max {s['max']:5.1f}%")
    print("\ncolocated stage, measured (*) and extrapolated, tokens/rank=64")
    for m in sorted(M.mp):
        row = []
        for ep in (1, 2, 4, 8, 16, 32, 64):
            v = M.predict_colo(m, ep, 64)
            if v is None:
                continue
            star = "*" if ep in M.colo.get((m, 64), {}) else ""
            row.append(f"ep{ep}={v:.2f}{star}")
        print(f"  {m:<18} " + "  ".join(row))
