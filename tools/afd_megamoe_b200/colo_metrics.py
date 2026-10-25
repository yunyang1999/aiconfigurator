#!/usr/bin/env python3
"""Decode-only metrics for the colocated arm.

Why this module exists. The AFD arm is run with --no-afd-combined-with-pd, so it
is prefill-disaggregated: AIC reports TTFT 0.000 and its TPOT is a pure decode
step. The colocated arm runs in agg mode, where the SAME GPUs serve prefill and
decode, so AIC amortises the prefill into TPOT. Comparing those two TPOTs
charges colocated for work the AFD arm never does, and the size of the charge
scales like 1/osl:

    colocated TPOT inflation vs its own gen-only step, measured over 2717 rows
      osl=256   3.185x
      osl=1024  1.532x
      osl=4096  1.117x

That is an artefact of the serving mode, not a property of either design, and it
was inflating every ratio in the first cut of both workbooks.

The fix is to price the colocated arm on its gen-only step, which AIC reports
separately and which is nearly osl-independent (26.5 ms at osl=256 vs 28.0 ms at
osl=1024 for the same config -- the small residue is average KV length, which is
real). Both arms are then decode-only and the comparison is like-for-like.

Giving BOTH arms prefill would be the other way to make it symmetric, but AFD
with combined PD is a different deployment and our MegaMoE stage was measured in
decode, so decode-only on both sides is the defensible choice.
"""
from __future__ import annotations

# Draft-acceptance schedule, verbatim from the GB200 sheet's Configuration tab.
# A step emits 1 + accepted tokens, so per-token time is step / (1 + accepted) --
# the same division AIC applies internally (verified: AFD TPOT at nextn=1 and 2
# differs from the un-drafted step by exactly 1/1.85 and 1/2.53).
NEXTN_ACCEPTED = {0: 0.0, 1: 0.85, 2: 1.53}


def decode_only(row: dict) -> dict | None:
    """Colocated decode-only TPOT and tokens/s/GPU, plus an overlap ceiling.

    The overlap ceiling hides only dispatch/combine, not the whole MoE stage.
    On one GPU the MoE GEMMs and attention both need the SMs, so a two-batch
    overlap can only tuck the collective legs behind another microbatch's
    compute. It stays a CEILING because MegaMoE already fuses and overlaps those
    legs inside the kernel, so the increment still available to a TBO is less
    again. Blank for MiniMax-M3, whose ops arrive fused and cannot be split.

    `row` is one record of aic_colocated_measured.csv. Returns None unless the
    row is usable.
    """
    if row.get("status") != "ok":
        return None
    try:
        ep = int(row["ep"])
        conc = int(row["tokens_per_rank"]) * ep
        gen = float(row["gen_total_measured_ms"])
        gen_ov = float(row["gen_total_comm_overlap_ms"] or 0)
        mtp = int(row["mtp"])
    except (KeyError, TypeError, ValueError):
        return None
    if gen <= 0 or conc <= 0:
        return None
    per_step_tokens = 1.0 + NEXTN_ACCEPTED.get(mtp, 0.0)
    tpot = gen / per_step_tokens
    tpot_ov = gen_ov / per_step_tokens if gen_ov > 0 else None
    out = {
        "colo_unit_gpus": ep,
        "colo_batch": int(row["tokens_per_rank"]),
        "colo_conc": conc,
        "colo_tpot_ms": round(tpot, 4),
        "colo_tpsgpu": round(conc * 1000.0 / tpot / ep, 2),
        "colo_gen_step_ms": round(gen, 4),
        "colo_moe_ms_measured": float(row["moe_ms_measured"]),
        "colo_moe_provenance": row["moe_provenance"],
        "colo_attn_ms": float(row["attention_ms"]) if row.get("attention_ms") else None,
    }
    if tpot_ov:
        out["colo_tpot_comm_overlap_ms"] = round(tpot_ov, 4)
        out["colo_tpsgpu_comm_overlap"] = round(conc * 1000.0 / tpot_ov / ep, 2)
        out["colo_comm_share_of_moe"] = (float(row["comm_share_of_moe"])
                                         if row.get("comm_share_of_moe") else None)
    return out


def generic_decode_only(row: dict) -> dict | None:
    """Same pricing, but on AIC's own generic MoE op instead of the measurement.

    This is the control arm for megamoe_effect. It must use the identical
    decode-only convention, otherwise the ratio-of-ratios would be comparing a
    prefill-amortised number against a decode-only one and the artefact would
    survive the division instead of cancelling.
    """
    if row.get("status") != "ok":
        return None
    try:
        ep = int(row["ep"])
        conc = int(row["tokens_per_rank"]) * ep
        gen = float(row["gen_total_generic_ms"])
        mtp = int(row["mtp"])
    except (KeyError, TypeError, ValueError):
        return None
    if gen <= 0 or conc <= 0:
        return None
    tpot = gen / (1.0 + NEXTN_ACCEPTED.get(mtp, 0.0))
    return {
        "colo_unit_gpus": ep, "colo_batch": int(row["tokens_per_rank"]),
        "colo_conc": conc,
        "colo_tpot_ms": round(tpot, 4),
        "colo_tpsgpu": round(conc * 1000.0 / tpot / ep, 2),
        "colo_gen_step_ms": round(gen, 4),
        "colo_moe_ms_generic": float(row["moe_ms_generic"]) if row.get("moe_ms_generic") else None,
    }
