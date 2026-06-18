from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    tokens: int
    relevance: float

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError("Chunk token count must be positive.")
        if not isfinite(self.relevance):
            raise ValueError("Chunk relevance must be a finite number.")

@dataclass(slots=True)
class SelectionResult:
    method: str
    selected_chunks: list[Chunk]
    total_tokens: int
    total_relevance: float
    runtime_ms: float
