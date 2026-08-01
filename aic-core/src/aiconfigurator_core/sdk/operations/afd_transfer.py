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
# FastAFD-calibrated communication model
#
# Structure taken from hao-ai-lab/FastAFD (minisgl + vendored DeepEP V2 /
# DeepGEMM), which is a measured serving system, not an estimator:
#
#   * experts are partitioned per RANK, not per node
#       moe/deepep_m2n_union.py   experts_per_union_rank = ceil(E / eg_size)
#       impls/dispatch_deterministic_prologue.cuh
#                                 kNumExpertsPerRank = E / kNumScaleupRanks
#                                 map_expert_to_rank_idx(e) = e / kNumExpertsPerRank
#   * one message per (token, destination rank), deduplicated -- same kernel:
#       "Avoid duplicate messages to a single rank" / is_unique(rank_idx)
#   * the two legs carry DIFFERENT payloads
#       apis/mega_m2n.hpp  fp8_token_layout  = Data(hidden)      # A->F, e4m3
#                          fp8_sf_layout     = Data(hidden / 32) # per-32 scales
#                          bf16_token_layout = Data(hidden * 2)  # F->A
#   * within one NVLink/NVSwitch domain DeepEP takes the DIRECT path
#       elastic/buffer.hpp  `if (num_scaleout_ranks == 1) { // Direct dispatch }`
#     so a token crosses the fabric once per destination RANK.  Only when the
#     pool spans RDMA does the hybrid path kick in, where the token crosses the
#     slow link once per destination NODE and is then forwarded over NVLink
#       elastic/buffer.hpp  scaleout_send_buffer multiplicity == 1
# --------------------------------------------------------------------------- #

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
        self._weights = 0.0

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
        if f_local <= 1 or self._rank_mapping != "one_to_one":
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
        self._weights = 0.0

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
        if f_local <= 1 or self._rank_mapping != "one_to_one":
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


class AFDTransferFastAFD(Operation):
    """Cross-pool transfer priced the way FastAFD actually moves bytes.

    Differences from :class:`AFDTransfer` (see the module-level note for the
    FastAFD source lines behind each one):

    1. **Sums over destinations.** ``AFDTransfer`` charges the volume to a
       single destination against one GPU's fabric port; a token actually
       replicates to every unit holding one of its top-k experts.
    2. **Dedups per rank, not per node.** Experts are sharded per rank, so the
       fan-out is ``reach_rank``; the node-level count only bounds the RDMA hop
       of the hybrid path.
    3. **Direction-specific payload.** A->F is FP8 + per-32 scales, F->A is
       BF16 -- roughly a 2x asymmetry that the symmetric model cannot express.
    4. **Models incast.** An F-rank receives from every A-worker, so the F-side
       ingress can dominate in the asymmetric M:N layouts AFD is built for.
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
        tp_a: int = 1,
        f_moe_ep_size: int = 0,
        f_moe_tp_size: int = 1,
        comm_overhead_factor: float = 1.0,
    ) -> None:
        super().__init__(name, scale_factor)
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"AFDTransferFastAFD: direction must be one of {self._VALID_DIRECTIONS}, got {direction!r}"
            )
        self._direction = direction
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._tp_a = max(int(tp_a), 1)
        # Expert-shard width. FastAFD runs MLP_TP=1 so every F rank owns a
        # distinct expert slice and this equals n_f_workers, but AIC's default
        # sweep also enumerates moe_tp > 1. Under TP a token is addressed to an
        # EP group once and that group's AllGather fans it out, so the
        # cross-pool leg scales with the EP width, not the rank count.
        self._f_moe_ep_size = max(int(f_moe_ep_size), 0) or self._n_f_workers
        self._f_moe_tp_size = max(int(f_moe_tp_size), 1)
        self._comm_overhead_factor = float(comm_overhead_factor or 1.0)
        self._weights = 0.0

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def expert_shard_width(self) -> int:
        """Number of independent expert partitions the A side must address."""
        return max(min(self._f_moe_ep_size, self._n_f_workers), 1)

    @property
    def num_f_nodes(self) -> int:
        return max((self._n_f_workers + self._gpus_per_node - 1) // self._gpus_per_node, 1)

    @property
    def bytes_per_elem(self) -> float:
        """FP8+scales outbound, BF16 inbound -- the legs are not symmetric."""
        return (
            _FASTAFD_DISPATCH_BYTES_PER_ELEM
            if self._direction == "a2f"
            else _FASTAFD_COMBINE_BYTES_PER_ELEM
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")

        node = database.system_spec["node"]
        gpn = self._gpus_per_node
        rack_gpus = node.get("num_gpus_per_rack", float("inf"))
        latency = float(node.get("p2p_latency", 0.0))

        ep_width = self.expert_shard_width
        reach_rank = _afd_reach(self._num_experts, self._topk, ep_width)
        # Node-level dedup for the RDMA hop is bounded by the EP width too: a
        # node cannot hold more distinct partitions than it has EP groups.
        reach_node = _afd_reach(self._num_experts, self._topk, min(self.num_f_nodes, ep_width))
        payload = self._hidden_size * self.bytes_per_elem  # bytes per token per copy
        total_tokens = x * self._n_a_workers               # global tokens this microbatch

        a_gpus = self._n_a_workers * self._tp_a
        total_gpus = a_gpus + self._n_f_workers
        fabric_bw = float(database.system_spec.get_p2p_bandwidth(total_gpus))

        if total_gpus <= rack_gpus:
            # One NVLink/NVSwitch domain -> DeepEP's direct path: a token
            # crosses the fabric once per destination RANK.
            a_bytes = reach_rank * x * payload / self._tp_a          # per A-GPU
            f_bytes = total_tokens * (reach_rank / ep_width) * payload / self._f_moe_tp_size  # per F-GPU
            seconds = max(a_bytes, f_bytes) / fabric_bw + latency
        else:
            # Hybrid path: the slow link carries one copy per destination NODE,
            # then NVLink fans out inside that node.  The two hops overlap.
            nvl_bw = float(node["intra_node_bw"])
            a_slow = reach_node * x * payload / self._tp_a
            f_slow = total_tokens * (reach_node / max(min(self.num_f_nodes, ep_width), 1)) * payload / gpn
            t_slow = max(a_slow, f_slow) / fabric_bw + latency
            f_nvl = total_tokens * ((reach_rank - reach_node) / ep_width) * payload
            seconds = max(t_slow, f_nvl / nvl_bw)

        lat_ms = seconds * 1000.0 * self._comm_overhead_factor
        return PerformanceResult(lat_ms * self._scale_factor, energy=0.0, source="empirical")

    def get_weights(self, **kwargs):
        return self._weights


class AFDCombineFastAFD(Operation):
    """A-side local reduce of the partials that actually come back.

    FastAFD's combine buffer reserves one slot per top-k expert
    (``mega_m2n.hpp`` ``combine_token_buffer``); DeepEP's elastic path
    reserves ``min(num_ranks, num_topk)`` under ``allow_multiple_reduction``.
    Either way the count follows top-k reach over the *expert-holding ranks* --
    never ``f_moe_ep_size``, which in AIC defaults to 1 and silently zeroes the
    op.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        n_f_workers: int,
        tp_a: int = 1,
        num_experts: int = 0,
        topk: int = 0,
        f_moe_ep_size: int = 0,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = int(hidden_size)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._tp_a = max(int(tp_a), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._f_moe_ep_size = max(int(f_moe_ep_size), 0) or self._n_f_workers
        self._weights = 0.0

    @property
    def num_partials(self) -> float:
        """One partial per expert partition that holds >=1 of the token's top-k."""
        return _afd_reach(self._num_experts, self._topk,
                          max(min(self._f_moe_ep_size, self._n_f_workers), 1))

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        partials = self.num_partials
        if partials <= 1.0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        tokens_per_a_rank = (x + self._tp_a - 1) // self._tp_a
        # read `partials` BF16 buffers, write one reduced BF16 result
        total_bytes = int(
            (partials + 1.0)
            * tokens_per_a_rank
            * self._hidden_size
            * _FASTAFD_COMBINE_BYTES_PER_ELEM
        )
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


# --------------------------------------------------------------------------- #
# "pr1323new": PR #1323's model with the eleven audited defects fixed
#
# Scope is deliberately narrow -- it corrects what was measurably wrong and
# changes nothing else. In particular it keeps ``comm_quant_mode`` symmetric
# for both legs (the FP8-out/BF16-back asymmetry real stacks show is a separate
# question) and charges no buffer memory, so it stays comparable to ``pr1323``
# term by term.
#
#  1 sum over destinations instead of billing one
#  2 model incast -- an F rank receives from every A worker
#  3 shard experts over EP groups, not over physical nodes
#  4 drop the F-side collective when the F pool has no TP group
#  5 size that collective with the full-buffer query_nccl contract
#  6 let its participant count be the real TP width, uncapped
#  7 stop billing the model-internal MoE dispatch as F-pool compute
#  8 pick the fabric tier from the GPU count instead of always inter_node_bw
#  9 a dense token reaches one F replica, it is not split across nodes
# 10 count arriving partials, not the EP width
# 11 one definition of ``x`` shared by every op (tokens per A worker)
# --------------------------------------------------------------------------- #


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


class AFDTransferPR1323New(Operation):
    """Cross-pool transfer, summed over destinations and incast-aware."""

    _VALID_DIRECTIONS = ("a2f", "f2a")

    def __init__(self, name: str, scale_factor: float, *, direction: str,
                 hidden_size: int, n_a_workers: int, n_f_workers: int,
                 gpus_per_node: int = 8, num_experts: int = 0, topk: int = 0,
                 tp_a: int = 1, f_moe_ep_size: int = 0,
                 comm_quant_mode: Optional[common.CommQuantMode] = None,
                 comm_overhead_factor: float = 1.0) -> None:
        super().__init__(name, scale_factor)
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"AFDTransferPR1323New: direction must be one of {self._VALID_DIRECTIONS}, got {direction!r}"
            )
        self._direction = direction
        self._hidden_size = int(hidden_size)
        self._topo = _AFDTopology(n_a_workers=n_a_workers, n_f_workers=n_f_workers,
                                  gpus_per_node=gpus_per_node, tp_a=tp_a,
                                  f_moe_ep_size=f_moe_ep_size, num_experts=num_experts,
                                  topk=topk)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._comm_overhead_factor = float(comm_overhead_factor or 1.0)
        self._weights = 0.0

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def topology(self) -> _AFDTopology:
        return self._topo

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        x = int(kwargs.get("x", 0))                       # defect 11: tokens per A worker
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        t = self._topo
        payload = self._hidden_size * self._comm_quant_mode.value.memory
        reach = t.reach

        # Defect 1: every destination's copy leaves the same port.
        a_egress = reach * x * payload / t.tp_a
        # Defect 2: an F rank is fed by all A workers. Tokens landing on one
        # partition = total x reach/ep_width; the partition's TP ranks share it.
        f_ingress = (x * t.n_a_workers) * (reach / t.ep_width) * payload / t.moe_tp

        bw = t.bandwidth(database)
        latency = float(database.system_spec["node"].get("p2p_latency", 0.0))
        seconds = max(a_egress, f_ingress) / bw + latency
        lat_ms = seconds * 1000.0 * self._comm_overhead_factor
        return PerformanceResult(lat_ms * self._scale_factor, energy=0.0, source="empirical")

    def get_weights(self, **kwargs):
        return self._weights


class AFDGroupCollectivePR1323New(Operation):
    """F-side token-dimension collective, present only when a TP group exists."""

    _VALID = ("all_gather", "reduce_scatter")

    def __init__(self, name: str, scale_factor: float, *, collective: str,
                 hidden_size: int, n_a_workers: int, n_f_workers: int,
                 gpus_per_node: int = 8, num_experts: int = 0, topk: int = 0,
                 tp_a: int = 1, f_moe_ep_size: int = 0,
                 comm_quant_mode: Optional[common.CommQuantMode] = None) -> None:
        super().__init__(name, scale_factor)
        if collective not in self._VALID:
            raise ValueError(f"AFDGroupCollectivePR1323New: collective must be one of {self._VALID}")
        self._collective = collective
        self._hidden_size = int(hidden_size)
        self._topo = _AFDTopology(n_a_workers=n_a_workers, n_f_workers=n_f_workers,
                                  gpus_per_node=gpus_per_node, tp_a=tp_a,
                                  f_moe_ep_size=f_moe_ep_size, num_experts=num_experts,
                                  topk=topk)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        t = self._topo
        # Defect 4: pure EP has no TP group, so no token-dimension collective.
        # The old gate was min(n_f_workers, gpus_per_node) > 1 -- a property of
        # the fabric, which fires whether or not a TP group exists.
        if t.moe_tp <= 1:
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        tokens_in_group = (x * t.n_a_workers) * (t.reach / t.ep_width)
        # Defect 5: repo-wide contract -- message_size is the FULL buffer.
        # Defect 6: the width is the TP group, not the GPUs in a node; passing
        # the true count also makes query_nccl select the right bandwidth tier.
        elements = int(tokens_in_group * self._hidden_size)
        if elements <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = database.query_nccl(self._comm_quant_mode, t.moe_tp, self._collective, elements)
        return PerformanceResult(float(result) * self._scale_factor,
                                 energy=result.energy * self._scale_factor,
                                 source=getattr(result, "source", "silicon"))

    def get_weights(self, **kwargs):
        return self._weights


class AFDCombinePR1323New(Operation):
    """A-side reduce over the partials that actually arrive."""

    def __init__(self, name: str, scale_factor: float, *, hidden_size: int,
                 n_a_workers: int, n_f_workers: int, gpus_per_node: int = 8,
                 num_experts: int = 0, topk: int = 0, tp_a: int = 1,
                 f_moe_ep_size: int = 0,
                 comm_quant_mode: Optional[common.CommQuantMode] = None) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = int(hidden_size)
        self._topo = _AFDTopology(n_a_workers=n_a_workers, n_f_workers=n_f_workers,
                                  gpus_per_node=gpus_per_node, tp_a=tp_a,
                                  f_moe_ep_size=f_moe_ep_size, num_experts=num_experts,
                                  topk=topk)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._weights = 0.0

    @property
    def num_partials(self) -> float:
        # Defect 10: one partial per partition holding >=1 of the token's
        # top-k, i.e. the same reach the transfer uses -- capped by topk, not
        # equal to the EP width (which also made ep=1 return exactly zero).
        return self._topo.reach

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        partials = self.num_partials
        if partials <= 1.0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        # Defect 11: x is tokens per A worker in every op; the A worker's GPUs
        # split the reduce.
        tokens_per_a_gpu = (x + self._topo.tp_a - 1) // self._topo.tp_a
        bpe = self._comm_quant_mode.value.memory
        total_bytes = int((partials + 1.0) * tokens_per_a_gpu * self._hidden_size * bpe)
        if total_bytes <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = database.query_mem_op(total_bytes)
        return PerformanceResult(float(result) * self._scale_factor,
                                 energy=result.energy * self._scale_factor,
                                 source=getattr(result, "source", "silicon"))

    def get_weights(self, **kwargs):
        return self._weights
