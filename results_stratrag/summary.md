# Experiment Summary

This experiment compares token-budgeted RAG context selection methods on the StratRAG dataset.

## Main Findings

- Highest single-run relevance: dp at budget 3000 with total relevance 29.676.
- Fastest method on average: topk.
- Lowest average redundancy: redundancy-aware.

## Average Relevance by Method

| Method | Mean total relevance |
|---|---:|
| dp | 20.525 |
| greedy-density | 20.504 |
| redundancy-aware | 19.876 |
| topk | 17.720 |

## Interpretation

Dynamic programming provides the optimal relevance for the stated knapsack objective, so it is the strongest reference point when the budget is not too large. Greedy methods are faster and easier to explain, but they can miss combinations of medium-sized chunks that fit better together. The redundancy-aware greedy method may sacrifice some relevance density to avoid repeatedly selecting overlapping text.
