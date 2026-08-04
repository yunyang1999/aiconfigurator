# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD communication ops — cross-pool P2P transfer and intra-pool collectives.

Four ops model the full AFD communication path:

* ``AFDTransfer`` — unidirectional cross-pool P2P (A→F or F→A)
* ``AFDFAllGather`` — F-node intra-node AllGather along the token dimension
* ``AFDFReduceScatter`` — F-node intra-node ReduceScatter after F compute
* ``AFDCombine`` — A-side cross-EP local HBM reduce-add
"""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, Optional

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase


def _afd_send_prob(num_experts: int, topk: int, num_f_nodes: int) -> float:
    """Probability that a token must be dispatched to a given F-node.

    For MoE with expert parallelism (num_experts > 0, topk > 0,
    num_f_nodes > 1): uses the combinatorial formula
    ``P_send = 1 - C(E - E/Nf, k) / C(E, k)`` -- the probability that
    at least one of a token's top-k experts resides on the target F-node.

    For dense models or degenerate configs: returns ``1 / num_f_nodes``
    (uniform distribution of tokens across F-nodes).
    """
    # dense model not verified yet
    if num_experts <= 0 or topk <= 0 or num_f_nodes <= 1:  # degenerate configs
        return 1.0 / max(num_f_nodes, 1)
    experts_per_node = num_experts // num_f_nodes
    if experts_per_node <= 0:
        return 1.0 / max(num_f_nodes, 1)

    n_other = num_experts - experts_per_node
    if topk > n_other:
        return 1.0
    return 1.0 - comb(n_other, topk) / comb(num_experts, topk)


# --------------------------------------------------------------------------- #
# Cross-pool communication model
#

# A->F: FP8 e4m3 payload plus one UE8M0 scale per 32 elements.
_FASTAFD_DISPATCH_BYTES_PER_ELEM = 1.0 + 1.0 / 32.0
# F->A: BF16 partial results.
_FASTAFD_COMBINE_BYTES_PER_ELEM = 2.0


def _afd_reach(num_experts: int, topk: int, num_units: int) -> float:
    """Expected number of distinct units a token must be sent to.

    ``num_units`` is whatever the experts are sharded over -- ranks for the
    NVLink-direct path, nodes for the RDMA hop of the hybrid path.  Equals
    ``num_units * P(unit holds >=1 of the token's top-k experts)`` and is
    therefore capped by ``min(topk, num_units)``.
    """
    units = max(int(num_units), 1)
    if units <= 1:
        return 1.0
    if num_experts <= 0 or topk <= 0:  # dense: tokens are partitioned, not replicated
        return 1.0
    return units * _afd_send_prob(num_experts, topk, units)


class AFDTransfer(Operation):
    """Unidirectional cross-pool P2P transfer (A→F **or** F→A).

    Construct with ``direction="a2f"`` or ``direction="f2a"`` to declare
    which leg of the round-trip this instance models.  ``query()``
    returns the single-direction latency.

    A-side operates in DP mode: each A-rank holds its own independent
    tokens and sends/receives them with the full ``hidden_size`` per
    token.  The per-link payload is therefore symmetric for both
    directions.
    """

    _VALID_DIRECTIONS = ("a2f", "f2a")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        direction: str,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        comm_overhead_factor: float = 1.0,
    ) -> None:
        super().__init__(name, scale_factor)
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(f"AFDTransfer: direction must be one of {self._VALID_DIRECTIONS}, got {direction!r}")
        self._direction = direction
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._comm_overhead_factor = float(comm_overhead_factor or 1.0)
        self._weights = 0.0

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def num_f_nodes(self) -> int:
        """Physical F-node count: ``ceil(n_f_workers / gpus_per_node)``."""
        return max((self._n_f_workers + self._gpus_per_node - 1) // self._gpus_per_node, 1)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        nf = self.num_f_nodes
        p_send = _afd_send_prob(self._num_experts, self._topk, nf)
        bpe = self._comm_quant_mode.value.memory
        # x = tokens held by a single A-rank; per_link_bytes is
        # the volume on one A-rank → F-rank P2P connection.
        per_link_bytes = int(p_send * x * self._hidden_size * bpe)
        if per_link_bytes <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = database.query_p2p(per_link_bytes)
        lat = float(result) * self._comm_overhead_factor
        return PerformanceResult(
            lat * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights


class AFDFAllGather(Operation):
    """F-node intra-node AllGather along the **token** dimension before F compute.

    Each F-GPU within a node receives a subset of tokens from A-side P2P.
    The AllGather collects all token subsets across the ``gpus_per_node``
    GPUs so that every F-GPU sees the complete token batch needed for
    FFN/MoE computation.

    Returns 0 when the node has only 1 GPU or under broadcast rank mapping.
    """

    _VALID_RANK_MAPPINGS = ("one_to_one", "broadcast")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        rank_mapping: str = "one_to_one",
        full_buffer: bool = False,
        participants: int = 0,
        f_moe_tp_size: int = 0,
    ) -> None:
        super().__init__(name, scale_factor)
        if rank_mapping not in self._VALID_RANK_MAPPINGS:
            raise ValueError(
                f"AFDFAllGather: rank_mapping must be one of {self._VALID_RANK_MAPPINGS}, got {rank_mapping!r}"
            )
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._rank_mapping = rank_mapping
        # ``full_buffer`` selects the repo-wide ``query_nccl`` contract, where
        # ``message_size`` is the FULL gathered buffer: the SOL is
        # ``bpe * message_size * (N-1)/N / bw`` and model-side callers pass
        # ``x * hidden`` with the undivided token count. The PR #1323 default
        # divides by the rank count, which under-reports by that factor.
        # ``participants`` overrides the collective width (the F-side TP group)
        # when it is not simply the GPUs in a node.
        self._full_buffer = bool(full_buffer)
        self._participants = max(int(participants), 0)
        # Existence gate only -- it does not touch sizing. A token-dimension
        # collective presupposes a TP group on the F side; under pure EP
        # (moe_tp == 1) every F rank owns its own experts and there is nothing
        # to gather along tokens. The historical gate was
        # ``min(n_f_workers, gpus_per_node) > 1``, a property of the fabric,
        # which fires whether or not such a group exists. 0 means "caller did
        # not say", which preserves the historical behaviour exactly.
        self._f_moe_tp_size = max(int(f_moe_tp_size), 0)
        self._weights = 0.0

    @property
    def has_tp_group(self) -> bool:
        return self._f_moe_tp_size != 1

    @property
    def num_f_nodes(self) -> int:
        return max(
            (self._n_f_workers + self._gpus_per_node - 1) // self._gpus_per_node,
            1,
        )

    @property
    def f_gpus_in_node(self) -> int:
        """Collective width: the F-side TP group, capped at one node."""
        if self._participants:
            return max(min(self._participants, self._gpus_per_node), 1)
        return min(self._n_f_workers, self._gpus_per_node)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        f_local = self.f_gpus_in_node
        if f_local <= 1 or self._rank_mapping != "one_to_one" or not self.has_tp_group:
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        total = x * self._n_a_workers
        nf = self.num_f_nodes
        p_send = _afd_send_prob(self._num_experts, self._topk, nf)
        tokens_per_f_node = p_send * total
        per_rank_elements = int(
            tokens_per_f_node * self._hidden_size / (1 if self._full_buffer else f_local)
        )
        if per_rank_elements <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = database.query_nccl(self._comm_quant_mode, f_local, "all_gather", per_rank_elements)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights


class AFDFReduceScatter(Operation):
    """F-node intra-node NCCL ReduceScatter after F compute.

    After MoE/FFN, every F-GPU within a node holds results for *all*
    tokens that were AllGathered earlier.  Because A-rank <-> F-rank is
    one-to-one mapped, a ReduceScatter along the **token** dimension
    places each A-rank's tokens onto the corresponding F-GPU, ready
    for the F->A P2P transfer.

    The number of participants is ``min(n_f_workers, gpus_per_node)``
    -- the intra-node F-GPU count -- regardless of TP or EP configuration.
    Returns 0 when the node has only 1 F-GPU or under broadcast rank
    mapping.
    """

    _VALID_RANK_MAPPINGS = ("one_to_one", "broadcast")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        rank_mapping: str = "one_to_one",
        full_buffer: bool = False,
        participants: int = 0,
        f_moe_tp_size: int = 0,
    ) -> None:
        super().__init__(name, scale_factor)
        if rank_mapping not in self._VALID_RANK_MAPPINGS:
            raise ValueError(
                f"AFDFReduceScatter: rank_mapping must be one of {self._VALID_RANK_MAPPINGS}, got {rank_mapping!r}"
            )
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._rank_mapping = rank_mapping
        # ``full_buffer`` selects the repo-wide ``query_nccl`` contract, where
        # ``message_size`` is the FULL gathered buffer: the SOL is
        # ``bpe * message_size * (N-1)/N / bw`` and model-side callers pass
        # ``x * hidden`` with the undivided token count. The PR #1323 default
        # divides by the rank count, which under-reports by that factor.
        # ``participants`` overrides the collective width (the F-side TP group)
        # when it is not simply the GPUs in a node.
        self._full_buffer = bool(full_buffer)
        self._participants = max(int(participants), 0)
        # Existence gate only -- it does not touch sizing. A token-dimension
        # collective presupposes a TP group on the F side; under pure EP
        # (moe_tp == 1) every F rank owns its own experts and there is nothing
        # to gather along tokens. The historical gate was
        # ``min(n_f_workers, gpus_per_node) > 1``, a property of the fabric,
        # which fires whether or not such a group exists. 0 means "caller did
        # not say", which preserves the historical behaviour exactly.
        self._f_moe_tp_size = max(int(f_moe_tp_size), 0)
        self._weights = 0.0

    @property
    def has_tp_group(self) -> bool:
        return self._f_moe_tp_size != 1

    @property
    def num_f_nodes(self) -> int:
        return max((self._n_f_workers + self._gpus_per_node - 1) // self._gpus_per_node, 1)

    @property
    def f_gpus_in_node(self) -> int:
        """Collective width: the F-side TP group, capped at one node."""
        if self._participants:
            return max(min(self._participants, self._gpus_per_node), 1)
        return min(self._n_f_workers, self._gpus_per_node)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        f_local = self.f_gpus_in_node
        if f_local <= 1 or self._rank_mapping != "one_to_one" or not self.has_tp_group:
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        total = x * self._n_a_workers
        nf = self.num_f_nodes
        p_send = _afd_send_prob(self._num_experts, self._topk, nf)
        tokens_per_f_node = p_send * total
        per_rank_elements = int(
            tokens_per_f_node * self._hidden_size / (1 if self._full_buffer else f_local)
        )
        if per_rank_elements <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = database.query_nccl(self._comm_quant_mode, f_local, "reduce_scatter", per_rank_elements)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights


class AFDCombine(Operation):
    """A-side cross-EP combine: local HBM reduce-add of partial results.

    When F-side uses expert parallelism (``f_moe_ep_size > 1``), each
    A-rank receives ``f_moe_ep_size`` partial results from different
    EP partitions and reduces them locally.  Each partial result carries
    the full ``hidden_size`` per token (F->A transfers full hidden).
    Returns 0 for dense FFN (``f_moe_ep_size <= 1``).
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        tp_a: int = 1,
        f_moe_ep_size: int = 1,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = int(hidden_size)
        self._tp_a = max(int(tp_a), 1)
        self._f_moe_ep_size = max(int(f_moe_ep_size), 1)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        if self._f_moe_ep_size <= 1:  # no expert parallelism
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        tokens_per_a_rank = (x + self._tp_a - 1) // self._tp_a
        bpe = self._comm_quant_mode.value.memory
        total_bytes = int((self._f_moe_ep_size + 1) * tokens_per_a_rank * self._hidden_size * bpe)
        if total_bytes <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = database.query_mem_op(total_bytes)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights




class _AFDTopology:
    """Single source of truth for AFD deployment shape.

    Every op derives from this, so they cannot disagree about whether a count
    is GPUs or workers -- the conflation behind defects 2, 3 and 11.
    """

    def __init__(self, *, n_a_workers, n_f_workers, gpus_per_node, tp_a,
                 f_moe_ep_size, num_experts, topk):
        self.n_a_workers = max(int(n_a_workers), 1)      # DP replicas
        self.tp_a = max(int(tp_a), 1)                    # GPUs per A replica
        self.a_gpus = self.n_a_workers * self.tp_a
        self.f_gpus = max(int(n_f_workers), 1)           # GPUs
        self.gpus_per_node = max(int(gpus_per_node), 1)
        self.num_experts = max(int(num_experts), 0)
        self.topk = max(int(topk), 0)
        # Defect 3/9: the destination unit is the expert partition. A dense
        # model has one partition -- under the Phase-1 F-DP=1 invariant the F
        # pool is a single replica, so a dense token goes to one place and is
        # then spread by the F-side collective, not split across nodes.
        ep = max(int(f_moe_ep_size), 1) if self.num_experts > 0 else 1
        self.ep_width = max(min(ep, self.f_gpus), 1)
        self.moe_tp = max(self.f_gpus // self.ep_width, 1)

    @property
    def total_gpus(self) -> int:
        return self.a_gpus + self.f_gpus

    @property
    def reach(self) -> float:
        """Expected number of expert partitions one token must be sent to."""
        return _afd_reach(self.num_experts, self.topk, self.ep_width)

    def bandwidth(self, database: PerfDatabase) -> float:
        """Defect 8: choose the tier the traffic actually crosses.

        ``query_p2p`` hardcodes ``inter_node_bw`` (it carries a standing TODO
        saying so). ``SystemSpec.get_p2p_bandwidth`` already implements the
        node / rack / inter-rack ladder; feed it the deployment's GPU count.
        """
        return float(database.system_spec.get_p2p_bandwidth(self.total_gpus))




