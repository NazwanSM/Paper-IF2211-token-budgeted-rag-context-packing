from __future__ import annotations
import time
from math import isclose
from rag_context_packing.models import Chunk, SelectionResult
from rag_context_packing.scoring import jaccard_similarity

EPSILON = 1e-12

def _validate_budget(budget: int) -> None:
    if budget < 0:
        raise ValueError("Budget must be non-negative.")

def _build_result(
    method: str,
    selected_chunks: list[Chunk],
    started_at: float,
) -> SelectionResult:
    total_tokens = sum(chunk.tokens for chunk in selected_chunks)
    total_relevance = sum(chunk.relevance for chunk in selected_chunks)
    runtime_ms = (time.perf_counter() - started_at) * 1000
    return SelectionResult(
        method=method,
        selected_chunks=selected_chunks,
        total_tokens=total_tokens,
        total_relevance=total_relevance,
        runtime_ms=runtime_ms,
    )

def _fits(chunk: Chunk, used_tokens: int, budget: int) -> bool:
    return used_tokens + chunk.tokens <= budget

def select_top_k(chunks: list[Chunk], budget: int) -> SelectionResult:
    _validate_budget(budget)
    started_at = time.perf_counter()
    selected: list[Chunk] = []
    used_tokens = 0

    ordered_chunks = sorted(chunks, key=lambda chunk: (-chunk.relevance, chunk.chunk_id))
    for chunk in ordered_chunks:
        if _fits(chunk, used_tokens, budget):
            selected.append(chunk)
            used_tokens += chunk.tokens

    return _build_result("topk", selected, started_at)

def select_greedy_density(chunks: list[Chunk], budget: int) -> SelectionResult:
    _validate_budget(budget)
    started_at = time.perf_counter()
    selected: list[Chunk] = []
    used_tokens = 0

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (
            -(chunk.relevance / chunk.tokens),
            -chunk.relevance,
            chunk.tokens,
            chunk.chunk_id,
        ),
    )
    for chunk in ordered_chunks:
        if _fits(chunk, used_tokens, budget):
            selected.append(chunk)
            used_tokens += chunk.tokens

    return _build_result("greedy-density", selected, started_at)

def _is_better(
    candidate_value: float,
    candidate_tokens: int,
    current_value: float,
    current_tokens: int,
) -> bool:
    if candidate_value > current_value + EPSILON:
        return True
    if isclose(candidate_value, current_value, abs_tol=EPSILON):
        return candidate_tokens < current_tokens
    return False

def select_dynamic_programming(chunks: list[Chunk], budget: int) -> SelectionResult:
    _validate_budget(budget)
    started_at = time.perf_counter()
    ordered_chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    item_count = len(ordered_chunks)

    values = [[0.0] * (budget + 1) for _ in range(item_count + 1)]
    token_usage = [[0] * (budget + 1) for _ in range(item_count + 1)]
    keep = [[False] * (budget + 1) for _ in range(item_count + 1)]

    for item_index, chunk in enumerate(ordered_chunks, start=1):
        previous_index = item_index - 1
        for capacity in range(budget + 1):
            best_value = values[previous_index][capacity]
            best_tokens = token_usage[previous_index][capacity]

            if chunk.tokens <= capacity:
                candidate_value = (
                    values[previous_index][capacity - chunk.tokens] + chunk.relevance
                )
                candidate_tokens = (
                    token_usage[previous_index][capacity - chunk.tokens] + chunk.tokens
                )
                if _is_better(candidate_value, candidate_tokens, best_value, best_tokens):
                    values[item_index][capacity] = candidate_value
                    token_usage[item_index][capacity] = candidate_tokens
                    keep[item_index][capacity] = True
                    continue

            values[item_index][capacity] = best_value
            token_usage[item_index][capacity] = best_tokens

    best_capacity = 0
    best_value = 0.0
    best_tokens = 0
    for capacity in range(budget + 1):
        candidate_value = values[item_count][capacity]
        candidate_tokens = token_usage[item_count][capacity]
        if _is_better(candidate_value, candidate_tokens, best_value, best_tokens):
            best_capacity = capacity
            best_value = candidate_value
            best_tokens = candidate_tokens

    selected: list[Chunk] = []
    capacity = best_capacity
    for item_index in range(item_count, 0, -1):
        if keep[item_index][capacity]:
            chunk = ordered_chunks[item_index - 1]
            selected.append(chunk)
            capacity -= chunk.tokens

    selected.reverse()
    return _build_result("dp", selected, started_at)

def select_redundancy_aware_greedy(
    chunks: list[Chunk],
    budget: int,
    lambda_: float = 0.75,
) -> SelectionResult:

    _validate_budget(budget)
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be between 0 and 1.")

    started_at = time.perf_counter()
    remaining = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    selected: list[Chunk] = []
    used_tokens = 0

    while remaining:
        feasible = [chunk for chunk in remaining if _fits(chunk, used_tokens, budget)]
        if not feasible:
            break

        scored_candidates: list[tuple[float, float, Chunk]] = []
        for chunk in feasible:
            max_similarity = (
                max(jaccard_similarity(chunk.text, chosen.text) for chosen in selected)
                if selected
                else 0.0
            )
            adjusted_score = lambda_ * chunk.relevance - (1.0 - lambda_) * max_similarity
            density_score = adjusted_score / chunk.tokens
            scored_candidates.append((density_score, adjusted_score, chunk))

        _, _, chosen = sorted(
            scored_candidates,
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2].relevance,
                item[2].tokens,
                item[2].chunk_id,
            ),
        )[0]
        selected.append(chosen)
        used_tokens += chosen.tokens
        remaining.remove(chosen)

    return _build_result("redundancy-aware", selected, started_at)