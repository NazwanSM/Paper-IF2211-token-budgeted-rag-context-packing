from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_context_packing.data import estimate_tokens, load_chunks
from rag_context_packing.scoring import normalize_text

STRATRAG_DATASET = "Aryanp088/StratRAG"
STRATRAG_CONFIG = "default"
STRATRAG_ROWS_URL = "https://datasets-server.huggingface.co/rows"

def _fetch_stratrag_rows(
    split: str = "validation",
    offset: int = 0,
    length: int = 100,
) -> list[dict[str, Any]]:
    if length <= 0:
        return []

    params = urlencode(
        {
            "dataset": STRATRAG_DATASET,
            "config": STRATRAG_CONFIG,
            "split": split,
            "offset": offset,
            "length": min(length, 100),
        }
    )
    with urlopen(f"{STRATRAG_ROWS_URL}?{params}", timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = payload.get("rows", [])
    return [row["row"] for row in rows if "row" in row]

def _lexical_relevance(query: str, text: str) -> float:
    query_terms = set(normalize_text(query))
    text_terms = set(normalize_text(text))
    if not query_terms or not text_terms:
        return 0.05

    overlap_ratio = len(query_terms & text_terms) / len(query_terms)
    return round(0.05 + (0.45 * overlap_ratio), 3)

def stratrag_row_to_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    query_id = str(row.get("id", "stratrag_unknown"))
    query = str(row.get("query", "")).strip()
    doc_pool = row.get("doc_pool", [])
    gold_indices = set(row.get("gold_doc_indices", []))
    records: list[dict[str, Any]] = []

    for doc_index, document in enumerate(doc_pool):
        if not isinstance(document, dict):
            continue

        text = str(document.get("text", "")).strip()
        source = str(document.get("source", "")).strip()
        if not text or text == "__no_content__" or source == "__pad__":
            continue

        is_gold = doc_index in gold_indices
        relevance = 1.0 if is_gold else _lexical_relevance(query, text)
        title_prefix = f"{source}. " if source and not text.startswith(source) else ""
        full_text = f"{title_prefix}{text}"

        records.append(
            {
                "chunk_id": f"{query_id}_d{doc_index:02d}",
                "doc_id": str(document.get("doc_id", f"{query_id}_doc_{doc_index:02d}")),
                "text": full_text,
                "tokens": estimate_tokens(full_text),
                "relevance": relevance,
                "query_id": query_id,
                "query": query,
                "source": source,
                "is_gold": is_gold,
                "source_dataset": STRATRAG_DATASET,
            }
        )

    return records

def write_stratrag_chunks(
    path: str,
    split: str = "validation",
    examples: int = 12,
    offset: int = 0,
) -> None:
    
    if examples <= 0:
        raise ValueError("Number of StratRAG examples must be positive.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    current_offset = offset
    remaining = examples
    while remaining > 0:
        batch = _fetch_stratrag_rows(
            split=split,
            offset=current_offset,
            length=min(remaining, 100),
        )
        if not batch:
            break

        for row in batch:
            records.extend(stratrag_row_to_records(row))

        current_offset += len(batch)
        remaining -= len(batch)

    if not records:
        raise RuntimeError("No StratRAG rows were fetched from Hugging Face.")

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download StratRAG examples and convert them to data/*.jsonl chunks.",
    )
    parser.add_argument(
        "--output",
        default="data/stratrag_chunks.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"],
        help="StratRAG split to import.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=12,
        help="Number of StratRAG query examples to import.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Starting row offset in the selected split.",
    )
    return parser

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    write_stratrag_chunks(
        path=args.output,
        split=args.split,
        examples=args.examples,
        offset=args.offset,
    )
    chunks = load_chunks(args.output)

    print(f"Imported {len(chunks)} chunks from {STRATRAG_DATASET}")
    print(f"Split: {args.split}")
    print(f"Examples: {args.examples}")
    print(f"Output: {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
