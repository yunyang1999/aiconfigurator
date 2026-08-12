#!/usr/bin/env python3
"""Measured MegaMoE stage, as a surface the GB200 sweep can inject.

WHAT THE MEASUREMENT DEPENDS ON
-------------------------------
The benchmark's own measurement_boundary is "A-side fused block-32 FP8 quant,
dispatch, wait, top-k combine and routed-output scaling; F-side persistent fused
FP8xFP4 expert L1, model activation, requant, L2 and return". So the stage
includes per-A-rank work, and the data says that work dominates:

    MiniMax-M2.5, 32 experts per F GPU, 64 sequences per A rank
        ag=1  ->  16 tokens per F GPU  ->  7.89 ms
        ag=2  ->  32                   ->  7.91
        ag=3  ->  48                   ->  7.89

Tokens per F GPU move 3x and the stage does not. An earlier version of this
study keyed on tokens per F GPU and was wrong for exactly that reason.

The two variables that do determine it are

    shard = num_routed_experts / F      experts held by one F rank
    seq                                 sequences per A rank  (= a_batch_size)

with the A-pool width a weak third term. Fixing (model, shard, seq) and growing
the A pool gives a small, systematic, model- and shard-independent rise:

    ag         2      3      4      5      6      7
    MiniMax-M2.5   1.006  1.012  1.021  1.031  1.034  1.039
    MiniMax-M3     1.008  1.017  1.019  1.048  1.078    -
    Qwen3-235B     1.017  1.027  1.029  1.062  1.072  1.057

which a single power law g(ag) = ag**p captures (p fitted below, ~0.053 for
seq >= 4). Below seq = 4 the exponent is much larger, because an under-fed F
pool has spare capacity that extra A ranks fill with real work, so the two
regimes are fitted separately.

WHAT IS MEASURED AND WHAT IS MODELLED
-------------------------------------
    base(model, nextN, shard, seq)   100% measured, exact key, no interpolation
    g(ag)                            the only extrapolation, and it is bounded:
                                     g(64) = 1.25 against a trend measured over
                                     ag = 1..7

g > 1, so the modelled term always makes AFD look WORSE. A sweep can therefore
be run twice -- with g and with g == 1 -- and any conclusion that holds at both
ends does not rest on the extrapolation.

Callers must stay inside the measured envelope on `shard` and `seq`; lookups
outside it return None rather than a guess. For the split kernel that means
F in {1, 2, 4} (shards 128/64/32 on Qwen3 and MiniMax-M3, 256/128/64 on
MiniMax-M2.5) and seq on the measured rungs. F = 8 would need shard 16, which
only the colocated kernel has ever run; the two kernels differ by a median 4.2%
and up to 43% at matched (shard, tokens per rank), which is too loose to stand
in for a measurement.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from stability import accept

ROOT = Path("/home/scratch.yuny_wwfo_1/megamoe")

# Explicit allowlists. Globbing results/megamoe-* is how the multimodel smoke
# suite (warmups=2, iterations=10) leaked in once; the protocol gate in
# stability.accept now catches that too, but naming the directories keeps a new
# scratch directory from silently joining the study.
SPLIT_DIRS = ["megamoe-m2n-extended", "megamoe-m2n-afratio", "megamoe-m2n-grid",
              "megamoe-m2n-batchext", "megamoe-m2n-splitfill",
              # Rungs 72..224, added because the 64 -> 96 -> 128 gap straddles the
              # GB200 capacity wall and was quantising the concurrency comparison:
              # Qwen3 at isl=65536 read a 41.7% AFD advantage where the walls
              # (batch 108 against 118) give 3.2%.
              "megamoe-m2n-batchfine"]
COLO_DIRS = ["megamoe-colocated-ep1", "megamoe-colocated-ep2", "megamoe-colocated-ep4",
             "megamoe-colocated-ep8", "megamoe-colocated-ep8-rerun2",
             "megamoe-colocated-grid", "megamoe-colocated-batchext",
             "megamoe-colocated-batchfine"]

SMALL_SEQ = 4          # below this the A pool feeds an under-loaded F pool


class MeasuredStage:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.experts: dict[str, int] = {}
        self.layers: dict[str, int] = {}
        # (model, nextN, microbatches, shard, seq) -> [(ag, ms), ...]
        self._split: dict = defaultdict(list)
        # (model, nextN, ep, tokens_per_rank) -> [ms, ...]
        self._colo: dict = defaultdict(list)
        self._load()
        self.p_big, self.p_small, self.fit = self._fit_ag()
        self.base, self.base_spread = self._build_base()

    # ---------------------------------------------------------------- loading
    def _load(self):
        for d in SPLIT_DIRS:
            p0 = self.root / "results" / d
            if not p0.is_dir():
                continue
            for p in sorted(p0.glob("*.json")):          # sorted: reproducible
                j = json.loads(p.read_text())
                if not accept(j):
                    continue
                w, t, mp = j["workload"], j["topology"], j["model_profile"]
                if "ag_size" not in t or w.get("microbatches") not in (1, 2):
                    continue
                self.experts[mp["key"]] = mp["num_routed_experts"]
                self.layers[mp["key"]] = mp["num_moe_layers"]
                shard = mp["num_routed_experts"] // t["eg_size"]
                self._split[(mp["key"], w["mtp_nextn"], w["microbatches"], shard,
                             w["sequences_per_ag_rank"])].append(
                                 (t["ag_size"], j["stage_cuda"]["p50_ms"]))
        for d in COLO_DIRS:
            p0 = self.root / "results" / d
            if not p0.is_dir():
                continue
            for p in sorted(p0.glob("*.json")):
                j = json.loads(p.read_text())
                if not accept(j):
                    continue
                w, t, mp = j["workload"], j["topology"], j["model_profile"]
                if "ep_size" not in t:
                    continue
                self.experts.setdefault(mp["key"], mp["num_routed_experts"])
                self.layers.setdefault(mp["key"], mp["num_moe_layers"])
                self._colo[(mp["key"], w["mtp_nextn"], t["ep_size"],
                            w["logical_tokens_per_rank"])].append(
                                j["stage_cuda"]["p50_ms"])

    # ------------------------------------------------------------- ag scaling
    def _fit_ag(self):
        """g(ag) = ag**p, fitted separately above and below SMALL_SEQ.

        Only groups that contain an ag=1 point contribute, so every ratio is
        measured against a real single-A-rank baseline rather than against
        another extrapolation.
        """
        pts = {"big": [], "small": []}
        for (model, n, mb, shard, seq), v in self._split.items():
            by_ag = defaultdict(list)
            for ag, ms in v:
                by_ag[ag].append(ms)
            if 1 not in by_ag:
                continue
            b = st.median(by_ag[1])
            for ag, mss in by_ag.items():
                if ag == 1 or b <= 0:
                    continue
                bucket = "big" if seq >= SMALL_SEQ else "small"
                pts[bucket].append((math.log(ag), math.log(st.median(mss) / b)))
        out, info = {}, {}
        for k, v in pts.items():
            if len(v) < 4:
                out[k] = 0.0
                info[k] = dict(n=len(v), p=0.0, resid_median=None)
                continue
            n = len(v)
            sx = sum(a for a, _ in v); sy = sum(b for _, b in v)
            sxx = sum(a * a for a, _ in v); sxy = sum(a * b for a, b in v)
            p = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            res = [abs(math.exp(b) - math.exp(p * a)) / math.exp(p * a) * 100 for a, b in v]
            out[k] = p
            info[k] = dict(n=n, p=p, resid_median=st.median(res),
                           resid_p90=sorted(res)[int(n * 0.9)])
        return out["big"], out["small"], info

    def g(self, ag: float, seq: float) -> float:
        if ag <= 1:
            return 1.0
        return ag ** (self.p_big if seq >= SMALL_SEQ else self.p_small)

    # -------------------------------------------------------------- base grid
    def _build_base(self):
        """Every measurement normalised to ag=1 and averaged.

        Using the median of ms / g(ag) rather than only the ag=1 point keeps all
        the data in play while staying consistent with the multiplicative form.
        The within-group spread after normalisation is kept: it is the honest
        error bar on the surface and is carried into every row.
        """
        base, spread = {}, {}
        for k, v in self._split.items():
            seq = k[4]
            norm = [ms / self.g(ag, seq) for ag, ms in v]
            base[k] = st.median(norm)
            spread[k] = (max(norm) - min(norm)) / min(norm) * 100 if len(norm) > 1 else 0.0
        return base, spread

    # --------------------------------------------------------------- lookups
    def afd(self, model, f_gpus, batch, a_gpus, a_tp, nextn=0, mb=1):
        """(stage ms for all MoE layers, provenance) for one AFD deployment.

        The A pool contributes through its number of INDEPENDENT replicas,
        a_gpus / a_tp: a tensor-parallel group cooperates on one set of
        sequences, which is also how AIC counts concurrency (batch * A / a_tp).
        """
        ne = self.experts.get(model)
        if ne is None:
            return None, "model not measured"
        if ne % f_gpus:
            return None, f"F={f_gpus} does not divide {ne} experts"
        shard = ne // f_gpus
        key = (model, nextn, mb, shard, batch)
        if key not in self.base:
            return None, (f"outside measured envelope "
                          f"(shard={shard}, seq={batch}, mb={mb})")
        ag = a_gpus / a_tp
        gv = self.g(ag, batch)
        tag = "" if mb == 1 else f" mb={mb}"
        prov = ("measured" + tag) if ag <= 1 else (
            f"measured{tag} x g(ag={ag:g})={gv:.3f}"
            + ("" if ag <= 7 else " [ag beyond measured 1..7]"))
        return self.base[key] * gv, prov

    def afd_ungained(self, model, f_gpus, batch, nextn=0, mb=1):
        """The same stage with g == 1: the most AFD-favourable choice."""
        ne = self.experts.get(model)
        if ne is None or ne % f_gpus:
            return None
        return self.base.get((model, nextn, mb, ne // f_gpus, batch))

    def colo(self, model, ep, batch, nextn=0):
        """(stage ms, provenance) for a colocated EP group. Exact key only."""
        v = self._colo.get((model, nextn, ep, batch))
        if not v:
            return None, f"outside measured envelope (ep={ep}, tokens/rank={batch})"
        return st.median(v), "measured"

    # ----------------------------------------------------------- descriptions
    def envelope(self, nextn=0):
        """{model: {'shards': [...], 'seqs': [...], 'eps': [...]}}"""
        out = defaultdict(lambda: {"shards": set(), "seqs": set(), "eps": set(),
                                   "colo_batches": set(), "mb1": set(), "mb2": set()})
        for (m, n, mb, shard, seq) in self.base:
            if n == nextn:
                out[m]["shards"].add(shard); out[m]["seqs"].add(seq)
                out[m][f"mb{mb}"].add((shard, seq))
        for (m, n, ep, b) in self._colo:
            if n == nextn:
                out[m]["eps"].add(ep); out[m]["colo_batches"].add(b)
        # mb1 / mb2 stay as sets of (shard, seq): the sweep asks "is THIS pair
        # measured at this microbatch count", and the two counts do not cover the
        # same grid -- mb=2 was run on a subset.
        return {m: {k: (v if k in ("mb1", "mb2") else sorted(v))
                    for k, v in d.items()} for m, d in out.items()}


if __name__ == "__main__":
    T = MeasuredStage()
    print("g(ag) = ag**p")
    for k, v in T.fit.items():
        lab = f"seq >= {SMALL_SEQ}" if k == "big" else f"seq < {SMALL_SEQ}"
        print(f"  {lab:<12} p={v['p']:.4f}  n={v['n']}  "
              f"residual median {v['resid_median']:.1f}%  p90 {v['resid_p90']:.1f}%")
    print(f"  extrapolated: " + "  ".join(f"g({a})={T.g(a, 8):.3f}" for a in (8, 16, 32, 64, 68)))
    print(f"\nbase surface: {len(T.base)} measured cells")
    sp = sorted(T.base_spread.values())
    print(f"  within-cell spread after ag-normalisation: "
          f"median {sp[len(sp)//2]:.1f}%  p90 {sp[int(len(sp)*0.9)]:.1f}%  max {max(sp):.1f}%")
    print("\nmeasured envelope at nextN=0")
    for m, d in sorted(T.envelope().items()):
        print(f"  {m}")
        print(f"    split shards {d['shards']}  seq {d['seqs']}")
        print(f"    colocated ep {d['eps']}  batches {d['colo_batches']}")
