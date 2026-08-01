# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Operations package — one file per operation family with class-owned data
loading and querying.

This package replaces the prior monolithic ``operations.py``. Each family
lives in its own module; ``Operation`` and ``clear_all_op_caches`` come
from ``base.py``.

Public surface preserves the prior import pattern:

    from aiconfigurator_core.sdk.operations import GEMM, ContextAttention, ...
"""

from __future__ import annotations

from aiconfigurator_core.sdk.operations.afd_transfer import (
    AFDCombine,
    AFDCombineFastAFD,
    AFDCombinePR1323New,
    AFDFAllGather,
    AFDFReduceScatter,
    AFDTransfer,
    AFDGroupCollectivePR1323New,
    AFDTransferFastAFD,
    AFDTransferPR1323New,
    _afd_send_prob,
)
from aiconfigurator_core.sdk.operations.attention import ContextAttention, EncoderAttention, GenerationAttention
from aiconfigurator_core.sdk.operations.base import Operation, clear_all_op_caches, warm_all_op_data
from aiconfigurator_core.sdk.operations.communication import NCCL, P2P, CustomAllReduce
from aiconfigurator_core.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule
from aiconfigurator_core.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    DeepSeekV4MegaMoEModule,
    DeepSeekV4MHCModule,
    GenerationDeepSeekV4AttentionModule,
    _BaseDeepSeekV4AttentionModule,
)
from aiconfigurator_core.sdk.operations.elementwise import ElementWise
from aiconfigurator_core.sdk.operations.embedding import Embedding
from aiconfigurator_core.sdk.operations.gemm import GEMM
from aiconfigurator_core.sdk.operations.mamba import GDNKernel, Mamba2, Mamba2Kernel
from aiconfigurator_core.sdk.operations.mla import (
    ContextMLA,
    GenerationMLA,
    MLABmm,
    MLAModule,
    WideEPContextMLA,
    WideEPGenerationMLA,
)
from aiconfigurator_core.sdk.operations.moe import MoE, MoEDispatch, TrtLLMWideEPMoE, TrtLLMWideEPMoEDispatch
from aiconfigurator_core.sdk.operations.msa import ContextMSAModule, GenerationMSAModule
from aiconfigurator_core.sdk.operations.overlap import FallbackOp, OverlapOp

# Re-export commonly-imported names that the prior monolithic operations.py
# exposed at module level. Some test files and external callers do
# ``from aiconfigurator_core.sdk.operations import PerformanceResult``.
from aiconfigurator_core.sdk.performance_result import PerformanceResult


# --------------------------------------------------------------------------- #
# Speculative verification: which ops price a shared KV prefix
# --------------------------------------------------------------------------- #

#: Leaf ops whose cost depends on the *sequence* batch and the verification
#: width, not on the widened token batch. Under speculative decoding the
#: ``query_len`` verification tokens share one KV prefix, so charging them as
#: independent sequences rereads that prefix ``query_len`` times.
SPECULATIVE_ATTENTION_OPS = (GenerationAttention, GenerationMLA)


def contains_speculative_attention(op) -> bool:
    """True if ``op`` is, or wraps, an op that needs ``query_len``.

    Attention is not always a bare op: DeepSeek-family models emit the whole
    MLA block as a ``FallbackOp`` whose fallback list holds the GEMMs, the
    BMMs and ``GenerationMLA``. An ``isinstance`` test on the outer op misses
    those, which is how MLA models kept getting batch-widened verification.
    """
    if isinstance(op, SPECULATIVE_ATTENTION_OPS):
        return True
    inner = (
        list(getattr(op, "_group_a", []))
        + list(getattr(op, "_group_b", []))
        + list(getattr(op, "_fallback", []))
    )
    primary = getattr(op, "_primary", None)
    if primary is not None:
        inner.append(primary)
    return any(contains_speculative_attention(child) for child in inner)

__all__ = [
    "GEMM",
    "NCCL",
    "P2P",
    "AFDCombine",
    "AFDCombineFastAFD",
    "AFDCombinePR1323New",
    "AFDFAllGather",
    "AFDFReduceScatter",
    "AFDTransfer",
    "AFDGroupCollectivePR1323New",
    "AFDTransferFastAFD",
    "AFDTransferPR1323New",
    "ContextAttention",
    "ContextDSAModule",
    "ContextDeepSeekV4AttentionModule",
    "ContextMLA",
    "ContextMSAModule",
    "CustomAllReduce",
    "DeepSeekV4MHCModule",
    "DeepSeekV4MegaMoEModule",
    "ElementWise",
    "Embedding",
    "EncoderAttention",
    "FallbackOp",
    "GDNKernel",
    "GenerationAttention",
    "GenerationDSAModule",
    "GenerationDeepSeekV4AttentionModule",
    "GenerationMLA",
    "GenerationMSAModule",
    "MLABmm",
    "MLAModule",
    "Mamba2",
    "Mamba2Kernel",
    "MoE",
    "MoEDispatch",
    "Operation",
    "OverlapOp",
    "PerformanceResult",
    "TrtLLMWideEPMoE",
    "TrtLLMWideEPMoEDispatch",
    "WideEPContextMLA",
    "SPECULATIVE_ATTENTION_OPS",
    "WideEPGenerationMLA",
    "_BaseDeepSeekV4AttentionModule",
    "_afd_send_prob",
    "clear_all_op_caches",
    "contains_speculative_attention",
    "warm_all_op_data",
]
