from __future__ import annotations
import argparse
import re
from collections.abc import Callable
from rag_context_packing.data import load_chunks
from rag_context_packing.models import Chunk, SelectionResult
from rag_context_packing.selectors import (
    select_dynamic_programming,
    select_greedy_density,
    select_redundancy_aware_greedy,
    select_top_k,
)

Selector = Callable[[list[Chunk], int], SelectionResult]

SELECTORS: dict[str, Selector] = {
    "dp": select_dynamic_programming,
    "topk": select_top_k,
    "greedy-density": select_greedy_density,
    "redundancy-aware": select_redundancy_aware_greedy,
}

def _snippet(text: str, max_length: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select RAG context chunks under a fixed token budget.",
    )
    parser.add_argument(
        "--data",
        default="data/stratrag_chunks.jsonl",
        help="Path to local chunk JSONL data.",
    )
    parser.add_argument("--budget", type=int, default=1500, help="Token budget.")
    parser.add_argument(
        "--method",
        choices=sorted(SELECTORS),
        default="dp",
        help="Selection method to run.",
    )
    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    chunks = load_chunks(args.data)
    selector = SELECTORS[args.method]
    result = selector(chunks, args.budget)

    usage_ratio = result.total_tokens / args.budget if args.budget > 0 else 0.0
    selected_ids = ", ".join(chunk.chunk_id for chunk in result.selected_chunks)

    print(f"Selected method: {result.method}")
    print(f"Total relevance: {result.total_relevance:.3f}")
    print(f"Total tokens: {result.total_tokens} / {args.budget}")
    print(f"Budget usage: {usage_ratio:.2%}")
    print(f"Runtime: {result.runtime_ms:.3f} ms")
    print(f"Selected chunk IDs: {selected_ids}")
    print("Top 5 selected chunk snippets:")
    for chunk in result.selected_chunks[:5]:
        print(f"- {chunk.chunk_id} ({chunk.tokens} tokens, rel={chunk.relevance:.3f}): {_snippet(chunk.text)}")

if __name__ == "__main__":
    main()