from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ExperimentRow = dict[str, float | int | str]

def _plot_metric(
    rows: Sequence[ExperimentRow],
    metric: str,
    ylabel: str,
    output_path: str,
) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Cannot plot empty experiment results.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    _, axis = plt.subplots(figsize=(8, 5))
    for method, group in frame.groupby("method", sort=False):
        ordered = group.sort_values("budget")
        axis.plot(ordered["budget"], ordered[metric], marker="o", label=method)

    axis.set_xlabel("Token budget")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title(f"{ylabel} vs token budget")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()

def plot_relevance_vs_budget(rows: Sequence[ExperimentRow], output_path: str) -> None:
    _plot_metric(rows, "total_relevance", "Total relevance", output_path)

def plot_runtime_vs_budget(rows: Sequence[ExperimentRow], output_path: str) -> None:
    _plot_metric(rows, "runtime_ms", "Runtime (ms)", output_path)

def plot_redundancy_vs_budget(rows: Sequence[ExperimentRow], output_path: str) -> None:
    _plot_metric(rows, "avg_pairwise_redundancy", "Average pairwise redundancy", output_path)


def create_all_plots(rows: Sequence[ExperimentRow], results_dir: str) -> None:
    output_dir = Path(results_dir)
    plot_relevance_vs_budget(rows, str(output_dir / "relevance_vs_budget.png"))
    plot_runtime_vs_budget(rows, str(output_dir / "runtime_vs_budget.png"))
    plot_redundancy_vs_budget(rows, str(output_dir / "redundancy_vs_budget.png"))