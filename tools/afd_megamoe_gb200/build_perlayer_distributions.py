#!/usr/bin/env python3
"""Rebuild the routing distributions keeping the per-LAYER structure.

The first pass collapsed all 94 layers into one 128-vector (`counts_total`).
That cancels most of the skew, because the straggler rank is a different rank
on almost every layer -- at EP=4 only 38% of layers share the same worst rank,
barely above the 25% you would get by chance. A real decode step pays
`sum over layers of max over ranks`, so the layer axis has to survive.

Two outputs:

  real_routing_qwen3_235b_perlayer.json
      counts_per_layer[94][128], summed over the 12 per-rank dumps. Summing
      across DP ranks is correct and stays: every attention DP rank dispatches
      into the same expert-parallel group, so their tokens genuinely land on
      the same expert. (They are also exactly equal -- 1,263,360 assignments
      each -- so nothing is lost there either way.)

  synthetic_perlayer_{128,256,384}.json
      For the four models with no recorded routing. Each layer is an
      independent draw at qwen3's measured per-layer CV, which is the right
      generator given how weakly the straggler persists across layers.
      These stay clearly labelled as assumed, not recorded.
"""
import json
import math
import os
import random
import statistics as st
from pathlib import Path
from afd_paths import data_root

# Output directory: the same data root the sweep reads (AFD_MEGAMOE_ROOT).
W = data_root()
# The FastAFD serving run whose ray_logs/expert_load_*.json dumps are merged.
_run = os.environ.get("AFD_ROUTING_RUN")
if not _run:
    raise SystemExit("set AFD_ROUTING_RUN to a FastAFD run directory whose ray_logs/ "
                     "holds expert_load_*.json dumps (only needed to rebuild the "
                     "distributions; the bundle already ships them)")
RUN = Path(_run)


def cv(v):
    m = sum(v) / len(v)
    return st.pstdev(v) / m if m else 0.0


def main():
    dumps = sorted((RUN / "ray_logs").glob("expert_load_*.json"))
    if not dumps:
        raise SystemExit(f"no dumps under {RUN}/ray_logs")

    per_layer = None
    for fn in dumps:
        d = json.load(open(fn))
        cl = d["counts_per_layer"]
        if per_layer is None:
            per_layer = [[0] * len(cl[0]) for _ in cl]
        for li, row in enumerate(cl):
            for e, c in enumerate(row):
                per_layer[li][e] += c
    nL, ne = len(per_layer), len(per_layer[0])
    total = [sum(per_layer[li][e] for li in range(nL)) for e in range(ne)]

    layer_cvs = [cv(r) for r in per_layer]
    agg_cv = cv(total)
    print(f"{len(dumps)} dumps -> {nL} layers x {ne} experts, "
          f"{sum(total):,} assignments")
    print(f"  aggregated CV      {agg_cv*100:6.2f}%   <- what the first pass used")
    print(f"  per-layer CV mean  {st.mean(layer_cvs)*100:6.2f}%  "
          f"median {st.median(layer_cvs)*100:6.2f}%  "
          f"min {min(layer_cvs)*100:.1f}%  max {max(layer_cvs)*100:.1f}%")

    out = W / "real_routing_qwen3_235b_perlayer.json"
    json.dump({"schema": "fastafd.expert-load.v2",
               "note": f"per-layer routing merged from {len(dumps)} per-rank dumps under {RUN}",
               "num_experts": ne, "num_layers": nL,
               "source_dumps": [f.name for f in dumps],
               "counts_per_layer": per_layer,
               "counts_total": total,
               "summary": {"assignments": sum(total),
                           "aggregated_cv_percent": agg_cv * 100,
                           "per_layer_cv_mean_percent": st.mean(layer_cvs) * 100,
                           "per_layer_cv_median_percent": st.median(layer_cvs) * 100}},
              open(out, "w"))
    print(f"  wrote {out.name}")

    # ---- synthetic, for the models with no recorded routing ----------------
    # A lognormal per-expert weight reproduces a target CV and keeps every
    # expert active, which is what the measured layers look like: skewed, but
    # no expert ever at zero.
    target = st.mean(layer_cvs)
    sigma = math.sqrt(math.log(1 + target ** 2))
    rng = random.Random(20260901)
    for n in (128, 256, 384):
        layers = []
        for _ in range(nL):
            w = [rng.lognormvariate(-sigma * sigma / 2, sigma) for _ in range(n)]
            s = sum(w)
            layers.append([x / s * 1_000_000 for x in w])
        got = st.mean(cv(r) for r in layers)
        tot = [sum(layers[li][e] for li in range(nL)) for e in range(n)]
        p = W / f"synthetic_perlayer_{n}.json"
        json.dump({"schema": "fastafd.expert-load.v2",
                   "note": (f"ASSUMED, not recorded. {nL} independent layers at "
                            f"qwen3's measured per-layer CV {target*100:.1f}%."),
                   "num_experts": n, "num_layers": nL,
                   "counts_per_layer": layers, "counts_total": tot,
                   "summary": {"per_layer_cv_mean_percent": got * 100,
                               "aggregated_cv_percent": cv(tot) * 100}},
                  open(p, "w"))
        print(f"  wrote {p.name}: per-layer CV {got*100:.1f}% "
              f"(target {target*100:.1f}%), aggregated CV {cv(tot)*100:.1f}%")

    # ---- what this changes, in one table ----------------------------------
    def imb(counts, ranks):
        per = len(counts) // ranks
        loads = [sum(counts[i * per:(i + 1) * per]) for i in range(ranks)]
        m = sum(loads) / ranks
        return max(loads) / m if m else 1.0

    print("\n  ranks   aggregated I   per-layer I   excess ratio")
    for r in (2, 4, 8, 16, 32, 64, 128):
        if ne % r:
            continue
        ia = imb(total, r)
        il = st.mean(imb(per_layer[li], r) for li in range(nL))
        k = (il - 1) / (ia - 1) if ia > 1.000001 else float("nan")
        print(f"  {r:>5}   {ia:>12.4f}   {il:>11.4f}   {k:>12.2f}x")


if __name__ == "__main__":
    main()
