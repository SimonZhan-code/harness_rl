"""Gamma registry — the seed context-management policies for the Step 1 probe.

These four variants are the seed items of the C1 provenance playbook. Register new
Gamma variants here so the eval runner can sweep over them by name.
"""
from __future__ import annotations

from harness_rl.gamma.base import BaseContextManager, ContextManager, count_tokens
from harness_rl.gamma.g0_truncate import G0Truncate
from harness_rl.gamma.g1_retrieval import G1Retrieval
from harness_rl.gamma.g2_summarize import G2Summarize
from harness_rl.gamma.g3_structured_memory import G3StructuredMemory

GAMMA_REGISTRY: dict[str, type[BaseContextManager]] = {
    G0Truncate.name: G0Truncate,
    G1Retrieval.name: G1Retrieval,
    G2Summarize.name: G2Summarize,
    G3StructuredMemory.name: G3StructuredMemory,
}


def make_gamma(name: str, **kwargs) -> BaseContextManager:
    if name not in GAMMA_REGISTRY:
        raise KeyError(f"unknown gamma variant {name!r}; known: {list(GAMMA_REGISTRY)}")
    return GAMMA_REGISTRY[name](**kwargs)


__all__ = [
    "ContextManager",
    "BaseContextManager",
    "GAMMA_REGISTRY",
    "make_gamma",
    "count_tokens",
    "G0Truncate",
    "G1Retrieval",
    "G2Summarize",
    "G3StructuredMemory",
]
