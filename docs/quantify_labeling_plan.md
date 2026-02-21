# Quantify the Labeling Problem

## Motivation

Edge labels in tree search evaluation data are assigned by checking whether each
edge's clade exists in the larch DAG (which contains only maximum parsimony
trees). If phangorn finds trees with parsimony scores **equal to or better** than
larch's optimum, those trees may have edges incorrectly labeled as non-MP. This
is the "labeling concern" described in `docs/treesearch_training_ideas.md`.

Before investing in training data improvements (issue #49), we need to quantify
how much of the observed AUROC degradation in the late-search regime is due to
actual model failure vs. corrupted ground truth labels.

## Approach

Create a script `dpvtex/log_tree_searches/quantify_labeling_problem.py` that
systematically compares phangorn's best parsimony scores against larch's optimum
across all datasets, start types, and replicates.

### Key metrics per replicate

| Metric | Description |
|--------|-------------|
| `score_gap` | phangorn best − larch MP (negative = phangorn beats larch) |
| `num_at_or_below_mp` | Trees with parsimony score ≤ larch MP |
| `num_below_mp` | Trees that strictly beat larch |
| `frac_late_search_at_mp` | Fraction of last 20% of trees at/below MP score |
| `total_nonmp_labels_in_mp_trees` | Non-MP labels in trees that should be fully MP — these are suspect |
| `frac_suspect_labels` | Suspect labels / total edges in those trees |

### Data

For each of 4 viral datasets × 2 start types (random/NJ) × 3-5 replicates:

- **DAG**: `shared_data/viral/treesearch/{dataset}/larch-output.pb`
- **FASTA**: `shared_data/viral/treesearch/{dataset}/input.fasta`
- **Pickles**: `shared_data/treesearch/{start_type}/{dataset}/{dataset}_rep*_tree_search.p`

### Implementation

Single new file that:

1. Reuses `get_parsimony_scores()` and `get_dag_mp_score()` from the adjacent
   `compare_parsimony_scores.py`
2. Iterates over all datasets, start types, and replicates (auto-discovered via
   glob)
3. For each replicate: loads trees from pickle, computes parsimony scores,
   identifies trees at/below larch's MP score, counts potentially mislabeled
   non-MP edges
4. Caches DAG loading per dataset (same DAG shared across start types)
5. Outputs a CSV summary table and a printed report with per-dataset aggregates

### CLI

```bash
cd dpvtex/log_tree_searches
python quantify_labeling_problem.py \
    --data-root ../../shared_data \
    --output-csv labeling_problem_summary.csv
```

Optional flags: `--datasets`, `--start-types` to analyze subsets for faster
iteration.

### Core analysis logic

For each replicate:

1. Load tree→labels dict from pickle
2. Compute parsimony scores for all trees using `get_parsimony_scores()`
3. Compare each score against larch's MP score
4. For trees at/below MP: count non-MP labels (positions 2+ in preorder
   traversal, since positions 0-1 are masked root/first-child)
5. Compute late-search statistics (last 20% of trees)

### Verification

1. Run on a single replicate first: `--datasets rotavirusA_H_H2 --start-types
   random_starting`
2. Cross-check against running `compare_parsimony_scores.py` on the same data
3. Verify that `frac_suspect_labels > 0` only when `score_gap ≤ 0`

### Runtime

Estimated 1-3 hours for the full analysis (parsimony score computation via
`historydag.parsimony` is the bottleneck). Progress is printed per replicate.

## Possible outcomes and next steps

- **If phangorn rarely beats larch**: The labeling problem is minor, and the
  AUROC degradation is primarily a real model failure. Proceed with issue #49
  (varied training proportions) with confidence.
- **If phangorn frequently beats larch**: A significant portion of the AUROC
  drop may be a measurement artifact. Before issue #49, implement idea B (seed
  larch with phangorn's best tree) to fix labels.
- **Mixed results**: May vary by dataset. Report per-dataset to inform which
  datasets are reliable for evaluation.
