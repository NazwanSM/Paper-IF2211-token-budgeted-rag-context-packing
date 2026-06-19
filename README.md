# Token-Budgeted RAG Context Packing

This repository is a small, reproducible Python prototype for studying **Token-Budgeted Context Packing for Retrieval-Augmented Generation Using Dynamic Programming and Greedy Relevance Baselines**.

The project focuses on the algorithmic part of Retrieval-Augmented Generation (RAG): given a query and many retrieved chunks, choose the subset of chunks that fits inside a fixed token budget while maximizing total relevance.

It does not implement a chatbot and does not call OpenAI, Gemini, Anthropic, or other LLM APIs.

## Why This Matters

RAG systems often retrieve more text than can fit into the final prompt. A simple top-k strategy may select highly relevant chunks that are too long, leaving little room for other useful evidence. A token-budgeted selector makes this trade-off explicit:

- long chunks are expensive,
- relevance is valuable,
- context windows are limited,
- redundant chunks can waste budget.

This is a useful mini-project because it connects AI engineering practice with a classic Informatics optimization problem.

## Knapsack Formulation

The context packing task is modeled as a 0/1 knapsack problem:

| RAG concept | Knapsack concept |
|---|---|
| Document chunk | Item |
| Token count | Weight / cost |
| Relevance score | Value / profit |
| Context token budget | Capacity |
| Selected context | Selected item subset |

The objective is:

```text
maximize   sum(relevance(ci))
subject to sum(tokens(ci)) <= budget
```

## Implemented Algorithms

This prototype implements four selectors:

1. **Top-K by Relevance Baseline**
   Sorts chunks by relevance and adds every chunk that still fits.

2. **Greedy by Relevance Density**
   Sorts chunks by `relevance / tokens`, then adds every chunk that still fits.

3. **Dynamic Programming 0/1 Knapsack**
   Finds the optimal subset for total relevance under the token budget with `O(n * B)` complexity.

4. **Redundancy-Aware Greedy Selection**
   Uses a simple MMR-inspired score with Jaccard similarity over normalized word sets.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── chunks.jsonl
│   └── stratrag_chunks.jsonl
├── experiments/
│   └── run_experiments.py
├── results/
│   ├── experiment_results.csv
│   ├── summary.md
│   ├── relevance_vs_budget.png
│   ├── runtime_vs_budget.png
│   └── redundancy_vs_budget.png
└── src/
      └── rag_context_packing/
         ├── __init__.py
         ├── cli.py
         ├── data.py
         ├── evaluation.py
         ├── models.py
         ├── scoring.py
         ├── selectors.py
         └── visualization.py
```

## Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows PowerShell
pip install -r requirements.txt
```

The repository can run the module commands below directly from the project root. For use from another working directory, install the local package once:

```bash
pip install -e .
```

## Run the CLI

```bash
python -m rag_context_packing.cli --budget 1500 --method dp
python -m rag_context_packing.cli --data data/stratrag_chunks.jsonl --budget 1500 --method topk
python -m rag_context_packing.cli --data data/stratrag_chunks.jsonl --budget 1500 --method greedy-density
python -m rag_context_packing.cli --data data/stratrag_chunks.jsonl --budget 1500 --method redundancy-aware
```

The CLI prints the selected method, total relevance, total tokens, budget usage, selected chunk IDs, and the top five selected snippets.

## Run Experiments

```bash
python experiments/run_experiments.py
```

The experiment runner evaluates all methods on these budgets:

```text
500, 1000, 1500, 2000, 3000
```

It writes:

- `results/experiment_results.csv`
- `results/summary.md`
- `results/relevance_vs_budget.png`
- `results/runtime_vs_budget.png`
- `results/redundancy_vs_budget.png`

By default, the experiment runner reads `data/stratrag_chunks.jsonl`. It does not download data automatically; use the importer below if the StratRAG JSONL needs to be regenerated.

## Use StratRAG Real Data

This project can also import a small real retrieval benchmark sample from Hugging Face: `Aryanp088/StratRAG`. The importer uses Python's standard library and the public Hugging Face Dataset Viewer API, so no extra heavyweight dependency is required. Internet access is only needed the first time the local JSONL file is generated.

Import 12 validation examples, producing roughly 100+ clean candidate chunks after placeholder documents are removed:

```bash
python scripts/import_stratrag.py --split validation --examples 12 --output data/stratrag_chunks.jsonl
```

Run the CLI on the imported real dataset:

```bash
python -m rag_context_packing.cli --data data/stratrag_chunks.jsonl --budget 500 --method dp
```

Run experiments directly on StratRAG and store the outputs separately:

```bash
python experiments/run_experiments.py --dataset stratrag --results-dir results_stratrag
```

The StratRAG conversion maps each document in `doc_pool` into the project chunk schema. Documents listed in `gold_doc_indices` receive relevance `1.0`, while distractors receive a deterministic lexical-overlap relevance score against the query.

## Run Tests

```bash
pytest -q
```

The tests cover:

- DP correctness on a known knapsack instance,
- token budget constraints for all selectors,
- predictable greedy density behavior,
- redundancy metric bounds,
- evaluation row fields.

## Example Result Table

With the imported StratRAG sample, the budget 1500 rows look like this:

| method | budget | total_tokens | total_relevance | selected_count | avg_pairwise_redundancy | runtime_ms |
|---|---:|---:|---:|---:|---:|---:|
| topk | 1500 | 1497 | 19.187 | 20 | 0.077 | 0.061 |
| greedy-density | 1500 | 1496 | 21.121 | 29 | 0.115 | 0.065 |
| dp | 1500 | 1496 | 21.121 | 29 | 0.115 | 51.059 |
| redundancy-aware | 1500 | 1491 | 20.462 | 22 | 0.077 | 1049.107 |

## Expected Interpretation

Dynamic Programming should achieve the best or tied-best relevance because it optimizes the formal knapsack objective exactly. Greedy methods are usually faster and may be strong baselines, especially when relevance density aligns well with the token budget. The redundancy-aware method may select slightly less relevance in exchange for lower overlap among chunks.

This makes the project suitable as a compact AI Engineer / Software Engineer portfolio artifact: the implementation is lightweight, deterministic, testable, and grounded in a clear algorithmic formulation.
