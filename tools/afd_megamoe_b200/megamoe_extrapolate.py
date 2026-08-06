#!/usr/bin/env python3
"""Extrapolate MegaMoE MoE-stage latency to unmeasured model shapes and
parallel configurations, and emit `afd_moe_time_ms` values for AIConfigurator.

Why a physical model rather than a per-model curve fit
------------------------------------------------------
A per-model fit (t = a*experts_per_F + b*assignments + c) cannot say anything
about a model shape it never saw. Putting the SHAPE into the features and
letting the coefficients be HARDWARE EFFICIENCIES fixes that: the same
coefficients then apply to any MoE shape on the same GPU.

Per MoE layer, per microbatch, per F GPU:

    weight_bytes = (n_routed/F * 3*h*i_moe + n_shared * 3*h*i_shared) * bytes_per_elem
    flops        = (tok*topk/F) * 6*h*i_moe + (tok*n_shared/F) * 6*h*i_shared
    t            = weight_bytes/(BW_peak*eff_w) + flops/(FLOPS_peak*eff_c) + const

fitted against the measured points. The fit recovers physically sensible
constants, which is the main reason to believe it generalises:

    afd (split M2N kernel) : eff_w 69.1% (5.32 TB/s of 7.7), eff_c 27.8%, 18.2 us/layer/lane
    agg (colocated kernel) : eff_w 73.3% (5.65 TB/s of 7.7), eff_c 41.8%, 45.2 us/layer

The agg coefficients are only identifiable because EP 1/2/4 were measured
alongside EP 8. With EP=8 alone, experts-per-GPU is constant across the whole
colocated set, the solver folds the fixed overhead into the weight term, and it
reports 105.5% of peak HBM bandwidth -- unphysical, and a clear signal the fit
was unidentified rather than merely noisy.

The split stage is weight-bound (70-88%), which is why doubling the batch moves
it only 3-6%. The COLOCATED stage at EP=8 is not: measured weight share is just
29% for Qwen3-235B, 64% for MiniMax-M2.5, 55% for MiniMax-M3, because each GPU
holds only 16-32 experts and fixed per-layer cost dominates. That is why naive
1/EP scaling of the colocated stage overstates the speedup by 1.6-2.7x
(measured EP 1->8: 3.01x for Qwen3, 4.72x for M2.5, against an ideal 8x).

The fit is restricted to sequences-per-lane >= MIN_SEQ_PER_LANE; see that
constant for why, and for the measured error-vs-batch curve that sets it.

How far to trust it
-------------------
`accuracy()` runs leave-one-MODEL-out cross validation -- fit on two model
shapes, predict the third. That is the honest proxy for "predict a shape we
never measured":

    afd: 8.3% / 8.4% / 10.5% mean absolute error (max ~29%), n=389
    agg: 7.2% / 7.5% / 9.2% mean absolute error (max ~39%), n=223

So budget roughly +-10% on an unmeasured shape, and treat anything tagged
`extrapolated:*` as indicative. Never merge a non-`measured` row into an
`exact-only` AIC profile -- that is precisely what `--require-measured-moe`
exists to prevent.

Known gap: CROSS-NODE F pools
-----------------------------
Every measured point lives inside one 8-GPU NVLink domain, because MegaMoE's
symmetric buffer requires it. The model therefore contains NO inter-node
communication term. For F > 8 (or A+F > 8) the weight term still extrapolates
on physics (it falls as 1/F), but the A->F transfer would run over IB instead
of NVLink and is simply absent from the fit. Those rows are tagged
`extrapolated:cross-node` and should be treated as a LOWER BOUND on latency.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from stability import accept

DEFAULT_PROFILE = Path("/home/scratch.yuny_wwfo_1/megamoe/deliverable/afd_moe_stage_profile.json")

# B200 SXM datasheet, as used by AIC's own b200_sxm.yaml
MEM_BW_PEAK = 7.7e12      # B/s
FP4_FLOPS_PEAK = 9.0e15   # FLOP/s
GPUS_PER_NVLINK_DOMAIN = 8

# Validity floor for the linear model, established empirically. Fit error by
# sequences-per-lane over 337 measured split points:
#   4 -> 51.7% mean, 8 -> 23.6%, 16 -> 10.2%, 24 -> 4.7%, 32..192 -> 6.6-8.3%
# Below ~16 the kernel is launch/latency-bound rather than weight-bound and a
# single constant term cannot absorb the regime change. Those batches ARE
# measured, so the model is simply not used there.
MIN_SEQ_PER_LANE = 16

# E2M1 weight + UE8M0 block-32 scale -> 4 bits + 8 bits per 32 elements
BYTES_PER_WEIGHT_ELEM = 0.5 + 1.0 / 32.0


@dataclass(frozen=True)
class Shape:
    """MoE shape. Field names mirror AIC's get_model_config_from_model_path."""
    name: str
    hidden_size: int
    moe_inter_size: int
    num_experts: int
    topk: int
    moe_layers: int
    n_shared: int = 0
    shared_inter_size: int = 0

    @property
    def shared_i(self) -> int:
        return self.shared_inter_size or self.moe_inter_size

    @property
    def structure(self) -> tuple:
        """Shape identity ignoring the label. Used to recognise a measured shape
        that arrives under a different name -- the sheet calls it
        'Qwen3-235B-A22B', the FastAFD registry calls it 'qwen3_235b_fp4'."""
        return (self.hidden_size, self.moe_inter_size, self.num_experts,
                self.topk, self.n_shared, self.shared_i)


# Shapes for the AIC_AFD_Homogeneous sweep, read from AIC itself so the numbers
# match whatever the consumer's AIC run will assume. n_shared is not exposed by
# that helper; DeepSeek/Kimi carry one shared expert, the others none.
SHEET_SHAPES = {
    "DeepSeek-V3":      Shape("DeepSeek-V3", 7168, 2048, 256, 8, 58, n_shared=1),
    "DeepSeek-V3.1":    Shape("DeepSeek-V3.1", 7168, 2048, 256, 8, 58, n_shared=1),
    "Kimi-K2.5":        Shape("Kimi-K2.5", 7168, 2048, 384, 8, 60, n_shared=1),
    "MiniMax-M2.5":     Shape("MiniMax-M2.5", 3072, 1536, 256, 8, 62),
    "Qwen3-235B-A22B":  Shape("Qwen3-235B-A22B", 4096, 1536, 128, 8, 94),
}

# The three shapes we actually measured, keyed by their FastAFD profile name.
MEASURED_SHAPES = {
    "qwen3_235b_fp4":  Shape("qwen3_235b_fp4", 4096, 1536, 128, 8, 94),
    "minimax_m25_fp4": Shape("minimax_m25_fp4", 3072, 1536, 256, 8, 62),
    "minimax_m3_fp4":  Shape("minimax_m3_fp4", 6144, 3072, 128, 4, 57, n_shared=1, shared_inter_size=3072),
}


def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-30:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _lstsq(X, y):
    n = len(X[0])
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(n)] for i in range(n)]
    b = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(n)]
    return _solve(A, b)


def features(shape: Shape, a_gpus: int, f_gpus: int, seq_per_source_rank: int,
             microbatches: int, mtp_nextn: int) -> list[float]:
    """Per-layer, per-microbatch physical work, normalised to peak -> milliseconds."""
    h, i_m, i_s = shape.hidden_size, shape.moe_inter_size, shape.shared_i
    tokens = (seq_per_source_rank / microbatches) * (mtp_nextn + 1) * a_gpus
    weight_bytes = ((shape.num_experts / f_gpus) * 3 * h * i_m
                    + shape.n_shared * 3 * h * i_s) * BYTES_PER_WEIGHT_ELEM
    flops = ((tokens * shape.topk / f_gpus) * 6 * h * i_m
             + (tokens * shape.n_shared / f_gpus) * 6 * h * i_s)
    return [weight_bytes / MEM_BW_PEAK * 1e3, flops / FP4_FLOPS_PEAK * 1e3, 1.0]


class MegaMoEExtrapolator:
    def __init__(self, profile_path: Path = DEFAULT_PROFILE,
                 raw_dirs: list[Path] | None = None):
        """Fit from measured points.

        `raw_dirs` reads the benchmark's own result JSONs instead of the
        exported AIC profile. That matters: the exporter applies gate 5, which
        drops a split point whose workload has no matched colocated partner.
        That gate is an AIC-consumption rule, not a statement about measurement
        validity -- such a point is still a perfectly good latency sample and
        belongs in the fit. Stability (CV <= 3%) IS still required.
        """
        self.rows: list[tuple[str, str, list[float], float, int, int]] = []
        self.measured: dict[tuple, float] = {}
        if raw_dirs:
            self._load_raw(raw_dirs)
        else:
            self._load_profile(profile_path)
        self._fit()

    def _record(self, model_key, stage, a, f, seq, mb, nextn, layers, latency_ms):
        shape = MEASURED_SHAPES[model_key]
        x = features(shape, a, f, seq, mb, nextn)
        # seq/lane is carried so _fit can drop the launch-bound regime.
        self.rows.append((model_key, stage, x, latency_ms / layers / mb, a, f, seq / mb))
        # Exact-hit index, so a config we actually measured is returned
        # verbatim instead of being re-predicted by the fit.
        self.measured[(shape.structure, stage, a, f, seq, mb, nextn, layers)] = latency_ms

    def _load_profile(self, profile_path: Path):
        for e in json.loads(Path(profile_path).read_text())["entries"]:
            if e["moe_backend"] != "megamoe":
                continue
            if e["stage"] == "agg":
                a = f = int(e["topology"].lower().lstrip("ep"))
            else:
                sa, sf = e["topology"].upper().split("A")
                a, f = int(sa), int(sf.rstrip("F"))
            self._record(e["model_profile"], e["stage"], a, f,
                         e["logical_batch_per_source_rank"], e["microbatches"],
                         e["mtp_nextn"], e["moe_layers"], e["latency_ms"])

    def _load_raw(self, dirs: list[Path]):
        for d in dirs:
            for p in sorted(Path(d).glob("*.json")):
                j = json.loads(p.read_text())
                if not accept(j):
                    continue
                key = j["model_profile"]["key"]
                if key not in MEASURED_SHAPES:
                    continue
                w, topo = j["workload"], j["topology"]
                if "ep_size" in topo:  # colocated
                    stage, a, f = "agg", topo["ep_size"], topo["ep_size"]
                    seq, mb = w["logical_tokens_per_rank"], 1
                else:
                    stage, a, f = "afd", topo["ag_size"], topo["eg_size"]
                    seq, mb = w["sequences_per_ag_rank"], w["microbatches"]
                self._record(key, stage, a, f, seq, mb, w["mtp_nextn"],
                             w["layers"], j["stage_cuda"]["p50_ms"])

    def _fit(self):
        self.coef = {}
        self.env = {}
        for stage in ("afd", "agg"):
            sub = [r for r in self.rows if r[1] == stage and r[6] >= MIN_SEQ_PER_LANE]
            if len(sub) < 4:
                continue
            c = _lstsq([r[2] for r in sub], [r[3] for r in sub])
            if c is None:
                continue
            self.coef[stage] = c
            self.env[stage] = {
                "f_gpus": sorted({r[5] for r in sub}),
                "structures": {MEASURED_SHAPES[r[0]].structure for r in sub},
                "n": len(sub),
            }

    def efficiencies(self, stage: str) -> dict:
        c = self.coef[stage]
        return {
            "weight_bw_efficiency": 1.0 / c[0],
            "effective_bw_TBps": MEM_BW_PEAK / c[0] / 1e12,
            "compute_efficiency": 1.0 / c[1],
            "fixed_overhead_us_per_layer_per_lane": c[2] * 1e3,
        }

    def predict(self, shape: Shape, stage: str, a_gpus: int, f_gpus: int,
                seq_per_source_rank: int, microbatches: int, mtp_nextn: int,
                moe_layers: int | None = None) -> dict:
        """Return the AIC `afd_moe_time_ms` (full resident batch, all layers,
        all microbatches) plus provenance."""
        layers = moe_layers if moe_layers is not None else shape.moe_layers
        hit = self.measured.get((shape.structure, stage, a_gpus, f_gpus,
                                 seq_per_source_rank, microbatches, mtp_nextn, layers))
        if hit is not None:
            return {"afd_moe_time_ms": hit, "provenance": "measured",
                    "per_layer_per_lane_us": hit / layers / microbatches * 1e3,
                    "weight_term_pct": None, "note": None}
        c = self.coef[stage]
        x = features(shape, a_gpus, f_gpus, seq_per_source_rank, microbatches, mtp_nextn)
        per_layer_lane = sum(x[i] * c[i] for i in range(3))
        if per_layer_lane <= 0:
            raise ValueError("non-positive prediction; inputs far outside the fit")
        total = per_layer_lane * layers * microbatches

        env = self.env[stage]
        tags = []
        if shape.structure not in env["structures"]:
            tags.append("shape")
        # For the colocated stage the A and F roles are the SAME GPUs, so the
        # NVLink domain only has to hold f_gpus of them, not a_gpus + f_gpus.
        if seq_per_source_rank / microbatches < MIN_SEQ_PER_LANE:
            tags.append("below-batch-floor")
        domain_gpus = f_gpus if stage == "agg" else a_gpus + f_gpus
        if domain_gpus > GPUS_PER_NVLINK_DOMAIN:
            tags.append("cross-node")
        elif f_gpus not in env["f_gpus"]:
            tags.append("f-pool")
        prov = "interpolated" if not tags else "extrapolated:" + "+".join(tags)
        return {
            "afd_moe_time_ms": total,
            "provenance": prov,
            "per_layer_per_lane_us": per_layer_lane * 1e3,
            "weight_term_pct": x[0] * c[0] / per_layer_lane * 100,
            "note": ("no inter-node communication term is fitted; treat as a lower bound"
                     if "cross-node" in prov else None),
        }

    def accuracy(self, stage: str = "afd") -> dict:
        """Leave-one-MODEL-out CV: the honest proxy for an unmeasured shape."""
        sub = [r for r in self.rows if r[1] == stage and r[6] >= MIN_SEQ_PER_LANE]
        out = {}
        for held in sorted({r[0] for r in sub}):
            tr = [r for r in sub if r[0] != held]
            te = [r for r in sub if r[0] == held]
            ct = _lstsq([r[2] for r in tr], [r[3] for r in tr])
            if ct is None:
                continue
            errs = [abs(sum(r[2][i] * ct[i] for i in range(3)) - r[3]) / r[3] * 100 for r in te]
            out[held] = {"n": len(te), "mean_abs_pct": sum(errs) / len(errs),
                         "max_abs_pct": max(errs)}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--out", type=Path, help="write the generated table here (JSON)")
    ap.add_argument("--raw", action="store_true",
                    help="fit from raw benchmark JSONs (uses points that export gate 5 drops)")
    ap.add_argument("--include-cross-node", action="store_true",
                    help="also emit F pools that leave the 8-GPU NVLink domain (lower bounds only)")
    a = ap.parse_args()

    RAW = [Path("/home/scratch.yuny_wwfo_1/megamoe/results")/d for d in
           ("megamoe-m2n-extended","megamoe-m2n-afratio","megamoe-m2n-grid",
            "megamoe-colocated-ep8","megamoe-colocated-grid",
            # EP 1/2/4 were added specifically to identify the colocated weight
            # coefficient. With EP=8 alone, experts-per-GPU is constant across
            # the whole colocated set and the solver folded the fixed overhead
            # into the weight term, yielding an unphysical 105.5% of peak HBM
            # bandwidth. Four EP points fix that.
            "megamoe-colocated-ep1","megamoe-colocated-ep2","megamoe-colocated-ep4")]
    ex = MegaMoEExtrapolator(a.profile, raw_dirs=[d for d in RAW if d.is_dir()] if a.raw else None)
    print("fitted hardware efficiencies")
    for stage in ("afd", "agg"):
        if stage not in ex.coef:
            continue
        eff = ex.efficiencies(stage)
        print(f"  {stage}: BW {eff['weight_bw_efficiency']*100:.1f}% "
              f"({eff['effective_bw_TBps']:.2f} TB/s)  compute {eff['compute_efficiency']*100:.1f}%  "
              f"fixed {eff['fixed_overhead_us_per_layer_per_lane']:.1f} us   n={ex.env[stage]['n']}")
    print("\nleave-one-model-out cross validation")
    for stage in ("afd", "agg"):
        if stage not in ex.coef:
            continue
        for m, v in ex.accuracy(stage).items():
            print(f"  {stage} hold out {m:18} n={v['n']:2}  mean {v['mean_abs_pct']:5.2f}%  max {v['max_abs_pct']:5.2f}%")

    if not a.out:
        return

    # Batch rungs mirroring the AIC sweep's per-A-GPU batch range, plus the
    # microbatch and draft-length grid the spreadsheet actually uses.
    batches = [8, 16, 32, 48, 64, 96, 128, 192, 256, 352, 512]
    splits = [(7, 1), (6, 2), (5, 1), (4, 4)]
    if a.include_cross_node:
        splits += [(4, 8), (8, 8), (8, 16), (16, 8), (16, 16), (32, 8), (32, 16), (16, 32)]
    table = []
    for name, shape in SHEET_SHAPES.items():
        for stage, cfgs in (("afd", splits), ("agg", [(8, 8)])):
            for A, F in cfgs:
                for B in batches:
                    for mb in (1, 2):
                        if B % mb:
                            continue
                        for n in (0, 1, 2):
                            r = ex.predict(shape, stage, A, F, B, mb, n)
                            table.append({
                                "model": name, "stage": stage,
                                "topology": f"ep{F}" if stage == "agg" else f"{A}A{F}F",
                                "a_gpus": A, "f_gpus": F,
                                "batch_per_source_rank": B, "microbatches": mb,
                                "mtp_nextn": n, "moe_layers": shape.moe_layers,
                                **r,
                            })
    payload = {
        "schema": "megamoe.generated-stage-latency.v1",
        "system": "b200_sxm",
        "moe_precision": "w4a8_mxfp4_mxfp8",
        "generator": "megamoe_extrapolate.py",
        "basis": "fitted to measured MegaMoE points; see accuracy[] for leave-one-model-out error",
        "efficiencies": {s: ex.efficiencies(s) for s in ex.coef},
        "accuracy_leave_one_model_out": {s: ex.accuracy(s) for s in ex.coef},
        "warning": ("Rows whose provenance is not 'measured' are model output. Do not merge them "
                    "into an exact-only AIC profile. cross-node rows have no inter-node comm term "
                    "and are lower bounds."),
        "rows": table,
    }
    a.out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {len(table)} rows -> {a.out}")


if __name__ == "__main__":
    main()
