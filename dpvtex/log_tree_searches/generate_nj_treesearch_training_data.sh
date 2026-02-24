#!/bin/bash
# Run the full treesearch training data pipeline for OrthoMaM NJ starting trees.
# Usage: bash generate_nj_treesearch_training_data.sh
set -euo pipefail

cd "$(dirname "$0")"

# Step 1: Run phangorn tree searches + labeling via Snakemake
echo "=== Step 1: Running phangorn tree searches + labeling ==="
snakemake -s Snakefile --configfile config_training_nj.yaml -c8

# Step 2: Merge per-alignment pickles into a single training dataset
echo "=== Step 2: Merging per-alignment pickles ==="
python merge_treesearch_pickles.py \
  ../../shared_data/treesearch_training/nj_starting \
  ../../shared_data/orthomam_treesearch_nj_training.p

# Step 3: Register merged pickle in training nicknames
echo "=== Step 3: Registering in training nicknames ==="
python -c "
import json
path = '../../train/my_treesearch_nicknames.json'
with open(path) as f:
    data = json.load(f)
data['orthomam_treesearch_nj_training'] = 'orthomam_treesearch_nj_training.p'
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print(f'Added orthomam_treesearch_nj_training to {path}')
"
