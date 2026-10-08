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

import os
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from stability import accept
from afd_paths import data_root

# Data root: the unpacked data bundle, from AFD_MEGAMOE_ROOT (required).
ROOT = data_root()

# Explicit allowlists. Globbing results/megamoe-* is how the multimodel smoke
# suite (warmups=2, iterations=10) leaked in once; the protocol gate in
# stability.accept now catches that too, but naming the directories keeps a new
# scratch directory from silently joining the study.
#
# GB200 NVL72 (lyris), real hardware, not a B200 projection. The old
# B200 tree this file originally pointed at is on a
# different cluster and is not reachable from lyris, so there is no
# B200 + g(ag) fallback here -- see REAL_SPLITS below for why none is needed
# for the configs this data covers.
SPLIT_DIRS = ["megamoe-m2n-nvl72"]
COLO_DIRS = ["megamoe-colocated-nvl72"]

# (model, F) -> the ag_size values actually measured on lyris.
#
# ag is the number of INDEPENDENT dispatchers reaching the F pool. Attention TP
# reduces it: tp=k means k A-GPUs cooperate on one batch and dispatch ONCE, so
# an AIC config of (A GPUs, tp=k) is physically ag = A/k. That makes ag a real
# axis of the measurement, and it MUST be part of the lookup key.
#
# It was not, and that was a live bug: the first port of this file keyed on
# (model, nextn, mb, shard, seq) only, with ag entering through g(ag) -- which
# collapses to 1.000 on real data because there is no ag=1 baseline to fit. So
# every a_tp got the ag=A measurement: a_tp=2 over-charged ~48%, a_tp=4 ~119%,
# across 54 of 150 headline winners. The provenance string said so all along
# ("measured mb=2 x g(ag=17)=1.000") and was read as confirmation rather than
# as a warning.
#
# The deployment splits are 68A4F / 64A8F / 56A16F / 40A32F (64A8F / 60A12F /
# 56A16F / 48A24F for deepseek_v4_pro, whose 68A4F does not fit HBM); the
# smaller ag values are those same splits under tp=2 and tp=4.
#
# Gaps are real and are NOT filled by substitution. FastAFD's launcher requires
# nnodes * 4 == world_size ("the A:F split must be node-aligned"), so any ag
# where ag+F is not a multiple of 4 cannot be benchmarked at all: ag=34 and 17
# at F=4 (38 and 21 GPUs), 14 at F=16, 10 at F=32, 30 and 15 at F=12. That
# costs 68A4F its TP support entirely. A config whose ag is absent here returns
# None from afd() and is reported `illegal`, never priced from a neighbour --
# pricing an unmeasured condition with a measured number is the bug above.
REAL_AGS = {
    "qwen3_235b_fp4":        {4: (68,), 8: (64, 32, 16), 16: (56, 28), 32: (40, 20)},
    "minimax_m25_fp4":       {4: (68,), 8: (64, 32, 16), 16: (56, 28), 32: (40, 20)},
    "minimax_m3_fp4":        {4: (68,), 8: (64, 32, 16), 16: (56, 28), 32: (40, 20)},
    "deepseek_v4_flash_fp4": {4: (68,), 8: (64, 32, 16), 16: (56, 28), 32: (40, 20)},
    "deepseek_v4_pro_fp4":   {8: (64, 32, 16), 12: (60,), 16: (56, 28), 24: (48, 24, 12)},
}
# Kept as the deployment-split view (F -> the tp=1 ag), for callers that want
# "what is the full-width split for this F" rather than "what is measurable".
REAL_SPLITS = {m: {f: max(ags) for f, ags in d.items()} for m, d in REAL_AGS.items()}

SMALL_SEQ = 4          # below this the A pool feeds an under-loaded F pool


def _load_measured(path):
    """{(arm, experts_per_gpu): {tokens_per_expert: coefficient}} from the paired
    measurement file, or empty when none is supplied."""
    if not path:
        return {}
    try:
        rows = json.loads(Path(path).read_text())["rows"]
    except (OSError, KeyError, ValueError):
        return {}
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[(r["arm"], int(r["experts_per_gpu"]))][round(float(r["tokens_per_expert"]), 3)].append(
            float(r["coefficient"]))
    # several (seq, mb) cells can land on the same tokens-per-expert; the median
    # keeps one outlier from setting the value for that point.
    return {k: {x: st.median(v) for x, v in d.items()} for k, d in acc.items()}


class MeasuredStage:
    def __init__(self, root: Path = ROOT, imbalance_dist=None,
                 imbalance_measured=None, real_skew=None):
        """`imbalance_dist`: {model: [per-expert token counts]} recorded from a
        real serving run. When given, every stage time returned by afd() and
        colo() is the time REAL skewed routing would take, not the balanced time
        the kernel benchmark measures.

        The correction is a lookup on the measured curve, not a model. A stage
        ends when its busiest rank finishes; that rank holds `imb x batch`
        tokens; and a rank holding `imb x batch` tokens takes exactly what the
        balanced sweep already measured AT that batch. So the coefficient is
        t(imb*batch)/t(batch), read off measured points with the other key
        dimensions held fixed. imb lands between measured batches, so this
        interpolates and never extrapolates.

        Known bias, upward: at the balanced point t(imb*batch) EVERY rank carries
        imb*batch tokens, whereas real skew keeps the mean at batch and only the
        hottest rank reaches imb*batch. Collective traffic is therefore priced
        for more tokens than actually move, so this is a tightened upper bound
        rather than an unbiased estimate. Replay measurement quantifies the gap.

        Default None reproduces the balanced-only behaviour bit for bit.
        """
        self.root = root
        # Stored as {model: [[per-expert counts] for each layer]}. A flat list
        # is accepted and wrapped as a single layer, which is what the first
        # pass did implicitly -- and getting exactly that wrong is why this is
        # now a list of layers. See _rank_imbalance.
        self._imbalance_dist = {
            m: (list(c) if c and isinstance(c[0], (list, tuple)) else [list(c)])
            for m, c in (imbalance_dist or {}).items() if c
        }
        self._measured = _load_measured(imbalance_measured)
        # Models whose skew was RECORDED. The rest use an assumed skew, and
        # a reader must be able to tell which without reading the code.
        self.real_skew = set(real_skew or ())
        self.experts: dict[str, int] = {}
        self.topk: dict[str, int] = {}
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
                if "ag_size" not in t or w.get("microbatches") not in (1, 2, 4):
                    continue
                if w.get("routing") != "balanced":
                    continue        # hotset rows live in the same directory
                if t["ag_size"] not in REAL_AGS.get(mp["key"], {}).get(t["eg_size"], ()):
                    continue        # not a legitimate (A/tp, F) point for this model
                self.experts[mp["key"]] = mp["num_routed_experts"]
                self.topk[mp["key"]] = mp.get("routed_top_k")
                self.layers[mp["key"]] = mp["num_moe_layers"]
                shard = mp["num_routed_experts"] // t["eg_size"]
                self._split[(mp["key"], w["mtp_nextn"], w["microbatches"], shard,
                             w["sequences_per_ag_rank"], t["ag_size"])].append(
                                 j["stage_cuda"]["p50_ms"])
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
                if w.get("routing") != "balanced":
                    continue        # hotset rows live in the same directory
                self.experts.setdefault(mp["key"], mp["num_routed_experts"])
                self.topk.setdefault(mp["key"], mp.get("routed_top_k"))
                self.layers.setdefault(mp["key"], mp["num_moe_layers"])
                self._colo[(mp["key"], w["mtp_nextn"], t["ep_size"],
                            w["logical_tokens_per_rank"])].append(
                                j["stage_cuda"]["p50_ms"])

    # ------------------------------------------------------------- ag scaling
    def _fit_ag(self):
        """Retired. ag is now part of the lookup key, so nothing is scaled.

        The B200 study needed `g(ag) = ag**p` because it swept the A pool at
        fixed F and had to reach ag values it had not measured. On real GB200
        data every usable ag is measured outright, so the honest answer for an
        unmeasured ag is "no data", not "the nearest one times a power law".
        Kept as a stub returning zero exponents so the reporting block and any
        external caller of `g()` still run; `g()` is now identically 1.0.

        This is also where the ag/tp bug lived: with the fit degenerate on real
        data, `g()` returned 1.0 and silently made every a_tp share one
        measurement. Making ag part of the key is what actually fixes it --
        neutering g() alone would not have.
        """
        return 0.0, 0.0, {"big": dict(n=0, p=0.0, resid_median=None),
                          "small": dict(n=0, p=0.0, resid_median=None)}

    def g(self, ag: float, seq: float) -> float:
        """Identically 1.0: ag is a key, not a scale factor. See _fit_ag."""
        return 1.0

    # -------------------------------------------------------------- base grid
    def _build_base(self):
        """Every measurement normalised to ag=1 and averaged.

        Each cell is now an exact (model, nextn, mb, shard, seq, ag) key, so the
        list being reduced is repeat measurements of the SAME condition rather
        than different ag values normalised onto a common one. The spread is
        therefore true run-to-run dispersion and is carried into every row.
        """
        base, spread = {}, {}
        for k, v in self._split.items():
            base[k] = st.median(v)
            spread[k] = (max(v) - min(v)) / min(v) * 100 if len(v) > 1 else 0.0
        return base, spread

    # --------------------------------------------------------------- lookups
    def afd(self, model, f_gpus, batch, a_gpus, a_tp, nextn=0, mb=1):
        """(stage ms for all MoE layers, provenance) for one AFD deployment.

        Attention TP reduces the number of INDEPENDENT dispatchers: a tp group
        cooperates on one set of sequences and dispatches once, so this config
        is physically ag = a_gpus / a_tp, which is also how AIC counts
        concurrency (batch * A / a_tp). ag is part of the key -- an unmeasured
        ag returns None rather than borrowing another ag's number.
        """
        ne = self.experts.get(model)
        if ne is None:
            return None, "model not measured"
        if ne % f_gpus:
            return None, f"F={f_gpus} does not divide {ne} experts"
        if a_gpus % a_tp:
            return None, f"A={a_gpus} not divisible by a_tp={a_tp}"
        shard = ne // f_gpus
        ag = a_gpus // a_tp
        key = (model, nextn, mb, shard, batch, ag)
        if key not in self.base:
            allowed = REAL_AGS.get(model, {}).get(f_gpus, ())
            why = (f"ag={ag} not measured at F={f_gpus} (have {list(allowed)})"
                   if ag not in allowed else
                   f"outside measured envelope (shard={shard}, seq={batch}, mb={mb}, ag={ag})")
            return None, why
        tag = "" if mb == 1 else f" mb={mb}"
        prov = f"measured{tag} ag={ag}" + (f" (a_tp={a_tp})" if a_tp > 1 else "")
        ms = self.base[key]
        hit = self._coeff_afd(key, shard)
        if hit is not None:
            mult, note = hit
            return ms * mult, f"{prov} x{mult:.3f} ({note})"
        return ms, prov

    def afd_ungained(self, model, f_gpus, batch, nextn=0, mb=1, a_gpus=None, a_tp=1):
        """Same cell. Retained for callers built around the old g(ag) sensitivity
        end-point; with ag in the key there is no extrapolated term left to turn
        off, so this equals afd()'s value whenever afd() returns one."""
        ne = self.experts.get(model)
        if ne is None or ne % f_gpus or a_gpus is None or a_gpus % a_tp:
            return None
        return self.base.get((model, nextn, mb, ne // f_gpus, batch, a_gpus // a_tp))

    def colo(self, model, ep, batch, nextn=0):
        """(stage ms, provenance) for a colocated EP group. Exact key only."""
        v = self._colo.get((model, nextn, ep, batch))
        if not v:
            return None, f"outside measured envelope (ep={ep}, tokens/rank={batch})"
        ms = st.median(v)
        # The colocated arm gets the SAME treatment as the split arm. Correcting
        # only one of them would not add imbalance to AIC's search, it would tilt
        # the comparison by exactly the correction. And the two arms do not share
        # a coefficient even at equal experts-per-rank: the split arm's F ranks
        # absorb every A rank's tokens, so they sit closer to compute-bound and
        # pay MORE of the skew, while a colocated GPU is almost purely
        # weight-bound and absorbs most of it. Measured on this grid, at 32
        # experts per rank the split arm pays about 6% and colocated about 1%.
        hit = self._coeff_colo(model, nextn, ep, batch)
        if hit is not None:
            mult, note = hit
            return ms * mult, f"measured x{mult:.3f} ({note})"
        return ms, "measured"

    # ------------------------------------------------------ imbalance lookup
    def _measured_coeff(self, arm: str, experts_per_gpu: int, tokens_per_expert: float):
        """Coefficient read off PAIRED balanced/replay runs, or None.

        Preferred over the curve-lookup model, which paired measurement showed to
        be wrong in a structural way rather than a noisy one: it predicted about
        1.05 at 32 experts per GPU where the measurement says 0.99, and 1.13 at 4
        where the measurement says 0.97.

        Keyed on tokens per expert because that, not the skew, is what decides
        the answer. Grouped-GEMM work comes in tiles, so a uniform load sitting
        just past a tile boundary pays an extra tile on every expert while a
        skewed load of the same mean drops many experts back under it. Measured:
        at 72 tokens per expert skew ran 5.4% FASTER than uniform; by 192 it ran
        5.4% slower.

        Interpolates between measured points and refuses outside them.
        """
        pts = self._measured.get((arm, experts_per_gpu))
        if not pts:
            return None
        xs = sorted(pts)
        if tokens_per_expert < xs[0] or tokens_per_expert > xs[-1]:
            return None
        for lo, hi in zip(xs, xs[1:]):
            if lo <= tokens_per_expert <= hi:
                if hi == lo:
                    return pts[lo]
                w = (tokens_per_expert - lo) / (hi - lo)
                return pts[lo] * (1 - w) + pts[hi] * w
        return pts[xs[-1]]


    def _rank_imbalance(self, model: str, ranks: int):
        """Mean over layers of max/mean per-rank load, for the recorded routing.

        Experts are assigned to ranks in contiguous blocks, which is what the
        runtime does, so this is the imbalance the deployment actually sees --
        not an average over hypothetical expert placements.

        Averaged PER LAYER, then across layers -- not computed on the routing
        summed over layers. A decode step pays `sum over layers of max over
        ranks`: every layer waits for its own busiest rank, and that rank is a
        different one almost every layer. In the recorded Qwen3 run only 38% of
        layers shared a straggler at 4 ranks, against 25% by chance, and 10% at
        32 ranks. Summing the layers first therefore lets the peaks cancel
        against the troughs and reports a deployment far more balanced than any
        step actually runs: aggregated max/mean is 1.31 at 32 ranks where the
        per-layer figure is 2.90. The excess load is understated 2.6x to 11.6x,
        worse the finer the shard.

        A flat (single-layer) distribution still works and gives the old value,
        so the two can be diffed directly.
        """
        layers = self._imbalance_dist.get(model)
        if not layers or ranks <= 0 or len(layers[0]) % ranks:
            return None
        per = len(layers[0]) // ranks
        vals = []
        for counts in layers:
            loads = [sum(counts[i * per:(i + 1) * per]) for i in range(ranks)]
            mean = sum(loads) / ranks
            if mean > 0:
                vals.append(max(loads) / mean)
        return sum(vals) / len(vals) if vals else None

    @staticmethod
    def _monotone(curve: dict) -> dict:
        """The measured curve, made non-decreasing in batch.

        More tokens on a rank cannot make that rank finish sooner, so a dip
        between adjacent batches is run-to-run noise. Both ends of the
        coefficient must come from THIS curve, never one from here and one from
        the raw points: a raw denominator sitting in a noise dip while the
        numerator is read off the lifted curve inflates the ratio outright. That
        mismatch scored 1.2795 at F=8 batch=16 -- a 28% penalty for a 9% rise in
        tokens, on a stage that only moves 1.4x-2.6x across a 16x batch range.
        """
        run = float("-inf")
        out = {}
        for b in sorted(curve):
            run = out[b] = max(run, curve[b])
        return out

    @staticmethod
    def _interp(curve: dict, x: float):
        """Linear interpolation on the measured points. Returns None off the
        ends -- an imbalance correction read off an extrapolated stage time
        would be indistinguishable from one read off a measurement, so refuse
        instead. Expects an already-monotone curve."""
        xs = sorted(curve)
        if not xs or x < xs[0] or x > xs[-1]:
            return None
        for lo, hi in zip(xs, xs[1:]):
            if lo <= x <= hi:
                if hi == lo:
                    return curve[lo]
                return curve[lo] + (curve[hi] - curve[lo]) * (x - lo) / (hi - lo)
        return curve[xs[-1]]

    @staticmethod
    def _token_fraction(curve: dict, batch: int):
        """Share of the stage time that moves with token count, at this batch.

        Fitted from the BALANCED grid already measured, so it costs nothing and
        assumes nothing: t = fixed + variable*batch, and the fraction is
        variable*batch / t. It is the answer to "how compute-bound is this
        stage" -- 1.0 would mean every extra token costs proportional time,
        0.0 that the stage is pure weight streaming. Measured here it runs
        0.05 to 0.87, which is why treating the stage as compute-bound and
        charging the full load imbalance overstates the cost several fold.
        """
        xs = sorted(curve)
        if len(xs) < 4:
            return None
        n = len(xs)
        sx = sum(xs)
        sy = sum(curve[x] for x in xs)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * curve[x] for x in xs)
        den = n * sxx - sx * sx
        if den == 0:
            return None
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        if a <= 0 or b <= 0:
            return None
        var = b * batch
        return var / (a + var)

    def _skew_cost(self, imb, curve, batch):
        """1 + f*(imb-1): the load imbalance, discounted by how much of the
        stage actually responds to token count.

        Checked against 39 paired GB200 measurements: median error +0.003,
        mean absolute 0.020, worst +0.079 -- and that worst point is one whose
        two measured repeats themselves differ by 0.077. Good to about two
        points on a quantity that ranges 1.00 to 1.14.

        Preferred over reading the measured table directly because it covers
        every cell rather than the 43% that were benchmarked, and because
        adding a configuration later costs arithmetic instead of GPU hours.
        """
        if imb is None or imb <= 1.0:
            return None
        f = self._token_fraction(curve, batch)
        if f is None:
            return None
        return 1.0 + f * (imb - 1.0)

    def _coeff_afd(self, key, shard: int):
        model, nextn, mb, _shard, batch, ag = key
        ne = self.experts.get(model)
        if not ne or shard <= 0:
            return None
        imb = self._rank_imbalance(model, ne // shard)
        curve = {b: v for (m, n, mbb, sh, b, a), v in self.base.items()
                 if (m, n, mbb, sh, a) == (model, nextn, mb, shard, ag)}
        hit = self._skew_cost(imb, curve, batch)
        if hit is None:
            return None
        kind = "recorded" if model in self.real_skew else "assumed"
        return hit, f"routing skew {kind}, F={ne // shard}"

    def _coeff_colo(self, model, nextn, ep, batch):
        imb = self._rank_imbalance(model, ep)
        curve = {b: st.median(v) for (m, n, e, b), v in self._colo.items()
                 if (m, n, e) == (model, nextn, ep) and v}
        hit = self._skew_cost(imb, curve, batch)
        if hit is None:
            return None
        kind = "recorded" if model in self.real_skew else "assumed"
        return hit, f"routing skew {kind}, EP={ep}"

    # ----------------------------------------------------------- descriptions
    def envelope(self, nextn=0):
        """{model: {'shards', 'seqs', 'eps', 'colo_batches', 'mb1/2/4', 'ags'}}

        `mbN` are sets of (shard, seq, ag) triples -- the sweep asks "is THIS
        exact condition measured at this microbatch count", and ag belongs in
        that question. `ags` maps shard -> the measured ag values, which is what
        lets the sweep enumerate only the (A, tp) pairs that can be priced.
        """
        out = defaultdict(lambda: {"shards": set(), "seqs": set(), "eps": set(),
                                   "colo_batches": set(), "mb1": set(), "mb2": set(),
                                   "mb4": set(), "ags": defaultdict(set)})
        for (m, n, mb, shard, seq, ag) in self.base:
            if n == nextn:
                out[m]["shards"].add(shard); out[m]["seqs"].add(seq)
                out[m]["ags"][shard].add(ag)
                out[m][f"mb{mb}"].add((shard, seq, ag))
        for (m, n, ep, b) in self._colo:
            if n == nextn:
                out[m]["eps"].add(ep); out[m]["colo_batches"].add(b)
        return {m: {k: (v if k in ("mb1", "mb2", "mb4") else
                        ({s: sorted(a) for s, a in v.items()} if k == "ags" else sorted(v)))
                    for k, v in d.items()} for m, d in out.items()}


if __name__ == "__main__":
    T = MeasuredStage()
    print(f"base surface: {len(T.base)} measured cells "
          f"(key = model, nextn, mb, shard, seq, ag)")
    sp = sorted(T.base_spread.values())
    print(f"  within-cell dispersion (repeat runs of the SAME condition): "
          f"median {sp[len(sp)//2]:.1f}%  p90 {sp[int(len(sp)*0.9)]:.1f}%  max {max(sp):.1f}%")
    print("\nmeasured envelope at nextN=0")
    for m, d in sorted(T.envelope().items()):
        print(f"  {m}")
        print(f"    split shards {d['shards']}  seq {d['seqs']}")
        print(f"    ag per shard {dict(d['ags'])}")
        print(f"    colocated ep {d['eps']}  batches {d['colo_batches']}")
