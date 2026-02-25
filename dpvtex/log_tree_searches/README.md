# Log Tree Searches Pipeline

This directory contains a Snakemake pipeline to:

1. Run Maximum Parsimony tree searches for a set of input alignments
2. Save intermediate trees during the tree search
3. Run larch-usher to get a collection of maximum parsimony trees
4. Extract intermediate trees and label edges as MP or non-MP for use as DPVT
   testing input

## Prerequisites

- The `dpvtex` package from the base directory of this repo must be installed
- R with packages `optparse` and `remotes` (included in pixi environment)
- A modified version of the `phangorn` package (version 2.12.1) from
  [this GitHub repo](https://github.com/lenacoll/phangorn/tree/log-mp-search-trees),
  which allows logging trees along a maximum parsimony tree search. **Install it
  by running `pixi run install-phangorn` once after setting up the pixi
  environment.**
- The `larch-usher` software
  ([installation guide](https://github.com/matsengrp/larch))

## Running the Pipeline

### Configuration

Create or modify `config.yaml` with the following settings:

```yaml
input_dir: "path/to/alignment/directory" # Directory containing input alignments
output_dir: "path/to/output/directory" # Directory for output pickle files
larch_build: "path/to/larch-usher" # Path to larch-usher binary
nicknames: "path/to/data_nicknames.json" # Path to dataset nicknames file
num_replicates: 3 # Number of tree search replicates (default: 3)
start_tree_type: "random" # Starting tree type: "random" or "nj" (default: "random")
```

### Input Directory Structure

Input alignments must be organized as:

```
input_dir/
├── dataset1/
│   └── dataset1.fasta
├── dataset2/
│   └── dataset2.fasta
└── ...
```

Each alignment must be in a subdirectory with the same name as the FASTA file
(without extension).

### Running

From this directory, run:

```bash
# Using default config.yaml
snakemake -c4

# Using a custom config file
snakemake -c4 --configfile my_config.yaml

# Dry run to see what will be executed
snakemake -n
```

## Pipeline Steps

### 1. Prepare Larch Input (`prepare_larch_input`)

Copies the input FASTA file to `input.fasta` in the alignment directory.

### 2. Preprocess for Larch (`preprocess_larch`)

Converts the FASTA alignment to larch-usher input format, producing:

- `output.pb` - Protocol buffer file
- `output.txt` - Reference sequence
- `output.vcf` - VCF file

### 3. Run Larch-Usher (`run_larch_usher`)

Runs `larch-usher` to generate a collection of all maximum parsimony trees,
saved as `larch-output.pb`.

### 4. MP Tree Search (`log_mp_tree_search`)

Uses the modified `phangorn` R package to perform maximum parsimony tree
searches, logging intermediate trees to `{basename}_rep{N}_log.trees`. Multiple
replicates are run with different random seeds.

### 5. Compute Labels (`compute_labels`)

Labels edges in the intermediate trees as MP or non-MP using the collection of
MP trees from larch-usher. Outputs pickle files in DPVT format.

### 6. Add Nicknames (`add_to_dataset_nicknames`)

Adds glob pattern entries to the nicknames JSON file for use in DPVT
training/testing. Creates one entry per alignment using the format
`{prefix}_{alignment_name}`, where the prefix identifies the dataset and start
tree type. For example:

```json
{
  "random_treesearch_my_dataset_alignment1": "treesearch/my_dataset/random_starting/alignment1/*_tree_search.p",
  "random_treesearch_my_dataset_alignment2": "treesearch/my_dataset/random_starting/alignment2/*_tree_search.p"
}
```

The glob patterns automatically discover all replicates (`rep1`, `rep2`, etc.)
within each alignment directory, expanding to nicknames like
`random_treesearch_my_dataset_alignment1_rep1_tree_search`.

## Output

The pipeline produces:

- `{output_dir}/{start_tree_type}_starting/{basename}/{basename}_rep{N}_tree_search.p` -
  Per-replicate pickle files
- Updated nicknames JSON file with glob pattern entries for automatic replicate
  discovery

## Standalone Scripts

### `quantify_phangorn_larch_comparison.py`

Compares phangorn's best parsimony scores against larch's DAG optimum to
identify trees where edge labels may be incorrect. For each replicate, it
computes the score gap between phangorn's last tree and larch's MP score, and
measures the fraction of edges not supported by the DAG (non-DAG edges).

```bash
python quantify_phangorn_larch_comparison.py \
  --data-root ../../shared_data \
  --output-dir ../../shared_data/treesearch \
  --datasets influenzaC_fluC_M rotavirusA_H_H2 \
  --start-types nj random
```

The output CSV is named automatically based on the datasets (e.g.
`phangorn_larch_comparison_influenzaC_fluC_M_rotavirusA_H_H2.csv`).

Use `--all-trees` to analyze every intermediate tree in the search (not just the
final tree). This produces a CSV with one row per tree per replicate, including
`tree_index` and `normalized_tree_index` columns:

```bash
python quantify_phangorn_larch_comparison.py \
  --data-root ../../shared_data \
  --output-dir ../../shared_data/treesearch \
  --datasets influenzaC_fluC_M \
  --start-types nj random --all-trees
```

## Next Steps

After generating treesearch data, use `train/treesearch.snakefile` to train
models and evaluate their performance on the intermediate trees. See the
"Tree Search Evaluation Workflow" section of the main
[README](../../README.md) for details.
