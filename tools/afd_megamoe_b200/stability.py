#!/usr/bin/env python3
"""Which benchmark points this study accepts.

FastAFD stamps every result with

    stable = cv_percent <= 3.0 and all(all_outputs_finite_by_rank)

(`benchmark_megamoe_colocated_models.py:873`). That 3% is FastAFD's own gate,
not AIC's and not this study's, and CV there is the spread across the 40 timed
iterations of ONE run -- not across repeated jobs.

We raise the dispersion limit to 4% and keep the finiteness check untouched.
Those are different kinds of failure: dispersion is about how repeatable a
number is, non-finite output is about whether it is a number at all.

Why 4%. The 3% line was discarding small-batch rungs whose values are visibly
correct. Qwen3-235B colocated at ep=8, nextN=0:

    batch   p50 ms   CV
      2      6.528   0.33%   kept
      4      7.020   3.45%   dropped
      6      7.102   3.63%   dropped
      8      6.919   0.17%   kept

7.020 and 7.102 sit exactly on the flat trend the neighbours describe and land
within 3% of the physical model's own estimate for those rungs. What made them
noisy is that the kernel is short (~7 ms) at small batch, so fixed launch and
scheduling jitter is a larger share of it -- a property of the measurement, not
of the kernel's speed.

The cost of dropping them was not cosmetic. A measured-only grid then jumped
from batch 2 straight to batch 8, and where a TPOT ceiling landed in that hole
the colocated arm was pinned at the low rung: at Qwen3 isl=16384 / osl=4096
under 15 ms it was held at concurrency 16 when batch 4 would in fact have fit at
14.96 ms, turning a 3.0x ratio into 5.8x.

Everything admitted between 3% and 4% carries a wider error bar than the rest.
`cv_percent` is preserved per row so that is visible rather than implied.
"""
from __future__ import annotations

CV_LIMIT = 4.0
FASTAFD_CV_LIMIT = 3.0     # what `stable` in the JSON means
# The campaign's measurement protocol. Anything shorter is a plumbing check, not
# a measurement, and a low CV does not redeem it: the multimodel smoke suite runs
# warmups=2 / iterations=10 and reported 0.190 ms for a Qwen3 point that measures
# 6.919 ms under the real protocol -- 36x off, with cv=3.55% that sails through
# the dispersion gate. A cold kernel is reproducibly cold.
MIN_WARMUPS = 30
MIN_ITERATIONS = 40


def accept(payload: dict, cv_limit: float = CV_LIMIT) -> bool:
    """True if a benchmark JSON is usable at this study's dispersion limit."""
    if (payload.get("warmups") or 0) < MIN_WARMUPS:
        return False
    if (payload.get("iterations") or 0) < MIN_ITERATIONS:
        return False
    finite = payload.get("all_outputs_finite_by_rank")
    if finite is not None and not all(finite):
        return False
    stage = payload.get("stage_cuda") or {}
    cv = stage.get("cv_percent")
    if cv is None:
        # Nothing to judge on: fall back to whatever FastAFD decided.
        return bool(payload.get("stable"))
    return float(cv) <= cv_limit


def relaxed(payload: dict) -> bool:
    """True for points this study admits that FastAFD's own gate would not."""
    return accept(payload) and not payload.get("stable", False)
