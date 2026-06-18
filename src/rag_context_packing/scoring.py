from __future__ import annotations
import re
from itertools import combinations
from rag_context_packing.models import Chunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

def normalize_text(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())

def jaccard_similarity(a: str, b: str) -> float:
    words_a = set(normalize_text(a))
    words_b = set(normalize_text(b))
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)

def average_pairwise_redundancy(chunks: list[Chunk]) -> float:
    if len(chunks) < 2:
        return 0.0

    similarities = [
        jaccard_similarity(left.text, right.text)
        for left, right in combinations(chunks, 2)
    ]
    return sum(similarities) / len(similarities)
