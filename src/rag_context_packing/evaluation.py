from __future__ import annotations
from rag_context_packing.models import Chunk, SelectionResult
from rag_context_packing.scoring import average_pairwise_redundancy
from rag_context_packing.selectors import (
    select_dynamic_programming,
    select_greedy_density,
    select_redundancy_aware_greedy,
    select_top_k,
)

def evaluate_result(result: SelectionResult, budget: int) -> dict[str, float | int | str]:
    usage_ratio = result.total_tokens / budget if budget > 0 else 0.0
    return {
        "method": result.method,
        "budget": budget,
        "total_tokens": result.total_tokens,
        "budget_usage_ratio": usage_ratio,
        "total_relevance": result.total_relevance,
        "selected_count": len(result.selected_chunks),
        "avg_pairwise_redundancy": average_pairwise_redundancy(result.selected_chunks),
        "runtime_ms": result.runtime_ms,
    }

def run_all_methods(chunks: list[Chunk], budget: int) -> list[dict[str, float | int | str]]:
    selectors = [
        select_top_k,
        select_greedy_density,
        select_dynamic_programming,
        select_redundancy_aware_greedy,
    ]
    return [evaluate_result(selector(chunks, budget), budget) for selector in selectors]

def run_budget_sweep(
    chunks: list[Chunk],
    budgets: list[int],
) -> list[dict[str, float | int | str]]:

    rows: list[dict[str, float | int | str]] = []
    for budget in budgets:
        rows.extend(run_all_methods(chunks, budget))
    return rows