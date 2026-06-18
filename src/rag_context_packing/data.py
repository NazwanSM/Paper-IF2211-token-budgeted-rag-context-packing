from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any
from rag_context_packing.models import Chunk

TOPICS = [
    "Retrieval-Augmented Generation",
    "context window limits",
    "token budget",
    "chunk selection",
    "dynamic programming",
    "greedy algorithms",
    "redundancy in retrieval",
    "hallucination reduction",
    "prompt construction",
    "AI engineering",
]

BASE_TEMPLATES = [
    (
        "Retrieval-Augmented Generation systems retrieve candidate chunks before "
        "building a prompt. A context packing step decides which chunks are useful "
        "enough to spend scarce context tokens on."
    ),
    (
        "Context windows are finite, so long chunks can crowd out shorter evidence. "
        "A token budget makes the selection problem measurable and suitable for "
        "algorithmic comparison."
    ),
    (
        "The context selection task can be mapped to 0/1 knapsack. Each chunk has a "
        "token cost, a relevance value, and a binary decision about whether it should "
        "enter the final prompt."
    ),
    (
        "Greedy retrieval baselines are attractive because they are simple and fast. "
        "However, sorting only by relevance may waste capacity when the best chunks "
        "are very large."
    ),
    (
        "Relevance density divides relevance by token count. This heuristic often "
        "selects compact chunks that provide useful evidence per token, although it "
        "does not guarantee optimality."
    ),
    (
        "Dynamic programming computes the optimal relevance under an integer token "
        "budget. The method is slower than greedy selection but gives a useful upper "
        "bound for evaluating heuristics."
    ),
    (
        "Redundant retrieval results can repeat the same evidence. Measuring overlap "
        "between selected chunks helps explain whether a context is diverse or merely "
        "filled with near-duplicates."
    ),
    (
        "Better context packing can reduce hallucination risk by prioritizing relevant "
        "evidence. It cannot guarantee factual generation, but it can improve the "
        "quality of information supplied to a downstream model."
    ),
    (
        "Prompt construction is an engineering trade-off between breadth, depth, and "
        "budget. A reproducible selector makes the trade-off explicit instead of "
        "leaving it to ad hoc top-k settings."
    ),
    (
        "AI engineering prototypes should be small, deterministic, and easy to test. "
        "Classic algorithms are useful when the goal is to isolate system behavior "
        "without calling external model APIs."
    ),
]

DETAIL_SENTENCES = [
    "The query asks for practical evidence about packing retrieved documents.",
    "The chunk is intended to represent a paragraph from technical notes.",
    "The example favors clarity over realistic corpus noise.",
    "The passage includes enough repeated terminology to support redundancy checks.",
    "The selection score is evaluated independently from answer generation.",
    "The experiment compares speed, relevance, token usage, and overlap.",
]

def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

def _coerce_chunk(raw: dict[str, Any], line_number: int) -> Chunk:
    text = str(raw.get("text", "")).strip()
    tokens_raw = raw.get("tokens")
    tokens = int(tokens_raw) if tokens_raw is not None else _estimate_tokens(text)
    if tokens <= 0:
        tokens = _estimate_tokens(text)

    return Chunk(
        chunk_id=str(raw.get("chunk_id", f"c{line_number:03d}")),
        doc_id=str(raw.get("doc_id", "doc_unknown")),
        text=text,
        tokens=tokens,
        relevance=float(raw.get("relevance", 0.0)),
    )

def load_chunks(path: str) -> list[Chunk]:
    dataset_path = Path(path)
    chunks: list[Chunk] = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            chunks.append(_coerce_chunk(raw, line_number))

    return chunks

def _build_text(index: int, rng: random.Random) -> str:
    topic = TOPICS[index % len(TOPICS)]
    template = BASE_TEMPLATES[index % len(BASE_TEMPLATES)]
    detail = DETAIL_SENTENCES[index % len(DETAIL_SENTENCES)]
    query_terms = rng.sample(TOPICS, k=3)

    if index % 10 in {8, 9}:
        duplicate_source = BASE_TEMPLATES[(index - 1) % len(BASE_TEMPLATES)]
        return (
            f"{duplicate_source} This near-duplicate chunk repeats evidence about "
            f"{topic}, {query_terms[0]}, and {query_terms[1]} with minor wording changes."
        )

    return (
        f"{template} Topic focus: {topic}. {detail} Related terms include "
        f"{query_terms[0]}, {query_terms[1]}, and {query_terms[2]}."
    )

def write_synthetic_chunks(path: str, n: int = 100, seed: int = 42) -> None:
    if n <= 0:
        raise ValueError("Synthetic dataset size must be positive.")

    rng = random.Random(seed)
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("w", encoding="utf-8") as file:
        for index in range(n):
            text = _build_text(index, rng)
            base_tokens = _estimate_tokens(text)
            topic_bonus = 0.08 * ((index % len(TOPICS)) / max(1, len(TOPICS) - 1))
            duplicate_penalty = 0.04 if index % 10 in {8, 9} else 0.0
            relevance = min(
                0.99,
                max(0.15, rng.uniform(0.35, 0.92) + topic_bonus - duplicate_penalty),
            )
            record = {
                "chunk_id": f"c{index + 1:03d}",
                "doc_id": f"doc_{(index // 5) + 1:02d}",
                "text": text,
                "tokens": base_tokens + rng.randint(20, 140),
                "relevance": round(relevance, 3),
            }
            file.write(json.dumps(record, ensure_ascii=True) + "\n")

def ensure_dataset(path: str) -> list[Chunk]:
    dataset_path = Path(path)
    if not dataset_path.exists() or dataset_path.stat().st_size == 0:
        write_synthetic_chunks(str(dataset_path))
    return load_chunks(str(dataset_path))
