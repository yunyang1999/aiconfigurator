#!/usr/bin/env python3
"""Synthetic per-layer routing for Kimi-K3 (896 routed experts, 92 MoE layers).

K3's router has not been recorded. Same generator as the other unrecorded
models in build_perlayer_distributions.py: each layer an independent lognormal
draw at Qwen3's measured per-layer CV, read from the recorded distribution in
the data bundle so the two stay in step. The output is labelled as assumed.

usage: AFD_MEGAMOE_ROOT=<bundle> build_k3_routing.py
"""
import json
import math
import random
import statistics as st

from afd_paths import data_root

EXPERTS = 896
LAYERS = 92          # layer 0 is dense; 92 LatentMoE layers


def cv(v):
    m = sum(v) / len(v)
    return st.pstdev(v) / m if m else 0.0


def main():
    W = data_root()
    ref = json.loads((W / "real_routing_qwen3_235b_perlayer.json").read_text())
    target = ref["summary"]["per_layer_cv_mean_percent"] / 100
    sigma = math.sqrt(math.log(1 + target ** 2))
    rng = random.Random(20261019)
    layers = []
    for _ in range(LAYERS):
        w = [rng.lognormvariate(-sigma * sigma / 2, sigma) for _ in range(EXPERTS)]
        s = sum(w)
        layers.append([x / s * 1_000_000 for x in w])
    got = st.mean(cv(r) for r in layers)
    tot = [sum(layers[li][e] for li in range(LAYERS)) for e in range(EXPERTS)]
    out = W / f"synthetic_perlayer_{EXPERTS}.json"
    json.dump({"schema": "fastafd.expert-load.v2",
               "note": (f"ASSUMED, not recorded. {LAYERS} independent layers at "
                        f"qwen3's measured per-layer CV {target*100:.1f}%."),
               "num_experts": EXPERTS, "num_layers": LAYERS,
               "counts_per_layer": layers, "counts_total": tot,
               "summary": {"per_layer_cv_mean_percent": got * 100,
                           "aggregated_cv_percent": cv(tot) * 100}},
              open(out, "w"))
    print(f"wrote {out.name}: per-layer CV {got*100:.1f}% (target {target*100:.1f}%), "
          f"aggregated CV {cv(tot)*100:.1f}%")

    def imb(counts, ranks):
        per = len(counts) // ranks
        loads = [sum(counts[i * per:(i + 1) * per]) for i in range(ranks)]
        return max(loads) / (sum(loads) / ranks)

    print("  F ranks   experts/rank   per-layer max/mean")
    for r in (8, 16, 28, 32, 56, 64):
        if EXPERTS % r == 0:
            print(f"  {r:>7}   {EXPERTS // r:>12}   {st.mean(imb(l, r) for l in layers):>18.4f}")


if __name__ == "__main__":
    main()
