"""Split a colocated step into compute and communication, through OverlapOp.

Two op layouts appear in the sweep. GQA models (Qwen3, MiniMax-M2.5) place
``moe_pre_dispatch`` / ``moe_post_dispatch`` at the top level, so the
all-to-all sits plainly on the critical path. MLA/MSA models (DeepSeek-V3,
Kimi-K2.5, MiniMax-M3) wrap the routed path in an ``OverlapOp`` that already
reports ``max(routed, shared)`` -- the shared expert runs on a second stream.
That is not two-batch overlap: inside the routed group the dispatch, the
grouped GEMM and the combine are still serial, so the all-to-all is still
billed. Reaching into the group is what makes the two layouts comparable.

The ceiling reported here hides *all* communication:

    step_tbo = max(compute_critical_path, total_comm)

with an ``OverlapOp`` contributing ``max(a_compute, b_compute)`` to the compute
path and ``a_comm + b_comm`` to the comm path. No overlap technique can beat
this -- it assumes perfect hiding, free micro-batch splitting, and unchanged
kernel efficiency at half the batch.
"""
from __future__ import annotations

COMM_TOKENS = {"ar", "dispatch", "alltoall", "a2a", "allgather", "reducescatter",
               "ag", "rs", "combine"}


def is_comm(name: str) -> bool:
    """True when an underscore-delimited token of ``name`` names a collective."""
    return bool(set(name.split("_")) & COMM_TOKENS)


_RECORD: dict = {}


def install() -> dict:
    """Patch OverlapOp.query to record its inner compute/comm split.

    Returns the dict the recorder writes into; callers clear it between
    queries. The wrapper re-queries the inner ops, which is cheap because every
    op memoises its database lookup, and returns the original result unchanged
    so the estimate itself is untouched.
    """
    from aiconfigurator_core.sdk.operations import OverlapOp

    orig = OverlapOp.__dict__["query"]

    def q(self, database, **kwargs):
        r = orig(self, database, **kwargs)
        acc = {"a_comp": 0.0, "a_comm": 0.0, "b_comp": 0.0, "b_comm": 0.0}
        for tag, group in (("a", self._group_a), ("b", self._group_b)):
            for op in group:
                v = float(op.query(database, **kwargs))
                acc[f"{tag}_{'comm' if is_comm(op._name) else 'comp'}"] += v
        acc["reported"] = float(r)
        _RECORD[self._name] = acc
        return r

    OverlapOp.query = q
    return _RECORD


def split_step(latency_dict: dict, record: dict) -> tuple[float, float]:
    """-> (compute critical path, total comm) in the units of ``latency_dict``.

    ``latency_dict`` totals the whole generation phase, while the recorder sees
    one query. The two are rescaled onto the dict's units by the ratio the
    OverlapOp's own entry gives, so the split never depends on knowing the step
    count.
    """
    comp = comm = 0.0
    for name, v in latency_dict.items():
        v = float(v)
        if v <= 0:
            continue
        inner = record.get(name)
        if inner is not None:
            rep = inner["reported"]
            k = v / rep if rep > 0 else 0.0
            comp += k * max(inner["a_comp"], inner["b_comp"])
            comm += k * (inner["a_comm"] + inner["b_comm"])
        elif is_comm(name):
            comm += v
        else:
            comp += v
    return comp, comm
