from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from rag_context_packing.models import Chunk

def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

def _coerce_chunk(raw: dict[str, Any], line_number: int) -> Chunk:
    text = str(raw.get("text", "")).strip()
    tokens_raw = raw.get("tokens")
    tokens = int(tokens_raw) if tokens_raw is not None else estimate_tokens(text)
    if tokens <= 0:
        tokens = estimate_tokens(text)

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
