from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_context_packing.data import load_chunks
from rag_context_packing.evaluation import run_budget_sweep
from rag_context_packing.visualization import create_all_plots

BUDGETS = [500, 1000, 1500, 2000, 3000]

def _parse_budgets(raw_budgets: str) -> list[int]:
    budgets = [int(value.strip()) for value in raw_budgets.split(",") if value.strip()]
    if not budgets:
        raise ValueError("At least one budget must be provided.")
    return budgets

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run context packing experiments on local chunk JSONL data.",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "stratrag", "custom"],
        default="stratrag",
        help="Dataset label and default JSONL path to evaluate.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to local JSONL data. Defaults depend on --dataset.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for CSV, summary, and plots.",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(budget) for budget in BUDGETS),
        help="Comma-separated token budgets.",
    )
    return parser

def _write_summary(frame: pd.DataFrame, output_path: Path, dataset_label: str) -> None:
    best_relevance = frame.loc[frame["total_relevance"].idxmax()]
    fastest_method = frame.groupby("method")["runtime_ms"].mean().idxmin()
    lowest_redundancy_method = (
        frame.groupby("method")["avg_pairwise_redundancy"].mean().idxmin()
    )
    average_relevance = frame.groupby("method")["total_relevance"].mean().sort_values(
        ascending=False
    )

    lines = [
        "# Experiment Summary",
        "",
        (
            "This experiment compares token-budgeted RAG context selection methods "
            f"on the {dataset_label} dataset."
        ),
        "",
        "## Main Findings",
        "",
        (
            f"- Highest single-run relevance: {best_relevance['method']} at budget "
            f"{int(best_relevance['budget'])} with total relevance "
            f"{best_relevance['total_relevance']:.3f}."
        ),
        f"- Fastest method on average: {fastest_method}.",
        f"- Lowest average redundancy: {lowest_redundancy_method}.",
        "",
        "## Average Relevance by Method",
        "",
        "| Method | Mean total relevance |",
        "|---|---:|",
    ]
    for method, value in average_relevance.items():
        lines.append(f"| {method} | {value:.3f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Dynamic programming provides the optimal relevance for the stated "
                "knapsack objective, so it is the strongest reference point when the "
                "budget is not too large. Greedy methods are faster and easier to "
                "explain, but they can miss combinations of medium-sized chunks that "
                "fit better together. The redundancy-aware greedy method may sacrifice "
                "some relevance density to avoid repeatedly selecting overlapping text."
            ),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _load_experiment_chunks(args: argparse.Namespace) -> tuple[Path, str]:
    if args.dataset == "stratrag":
        data_path = Path(args.data) if args.data else ROOT / "data" / "stratrag_chunks.jsonl"
        _validate_data_path(data_path)
        return data_path, "StratRAG"

    if args.dataset == "custom":
        if not args.data:
            raise ValueError("--data is required when --dataset custom.")
        data_path = Path(args.data)
        _validate_data_path(data_path)
        return data_path, "custom JSONL"

    data_path = Path(args.data) if args.data else ROOT / "data" / "chunks.jsonl"
    _validate_data_path(data_path)
    return data_path, "deterministic synthetic"

def _validate_data_path(data_path: Path) -> None:
    if not data_path.exists() or data_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Dataset file not found or empty: {data_path}. "
            "Run scripts/import_stratrag.py first for StratRAG data, or pass --data."
        )

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    budgets = _parse_budgets(args.budgets)
    data_path, dataset_label = _load_experiment_chunks(args)
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(str(data_path))
    rows = run_budget_sweep(chunks, budgets)
    frame = pd.DataFrame(rows)

    csv_path = results_dir / "experiment_results.csv"
    frame.to_csv(csv_path, index=False)

    _write_summary(frame, results_dir / "summary.md", dataset_label)
    create_all_plots(rows, str(results_dir))

    print(f"Dataset: {dataset_label}")
    print(f"Loaded {len(chunks)} chunks from {data_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {results_dir / 'summary.md'}")
    print(f"Wrote plots to {results_dir}")

if __name__ == "__main__":
    main()
