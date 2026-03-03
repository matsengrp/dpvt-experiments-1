#!/bin/bash
# Run the full treesearch training data pipeline.
# Edit the variables below to configure for different datasets or start tree types.
# Usage: bash generate_treesearch_training_data.sh
set -euo pipefail

cd "$(dirname "$0")"

CONFIG=config_orthomam_treesearch_training_nj.yaml
START_TYPE=nj
OUTPUT_DIR=../../shared_data/treesearch_training
MERGED_PICKLE=../../shared_data/orthomam_treesearch_nj_training.p
NICKNAMES_FILE=../../train/my_treesearch_nicknames.json
NICKNAME_KEY=orthomam_treesearch_nj_training

# Step 1: Run phangorn tree searches + labeling via Snakemake
echo "=== Step 1: Running phangorn tree searches + labeling ==="
snakemake -s Snakefile --configfile "$CONFIG" -c8

# Step 2: Merge per-alignment pickles into a single training dataset
echo "=== Step 2: Merging per-alignment pickles ==="
python merge_treesearch_pickles.py \
  "${OUTPUT_DIR}/${START_TYPE}_starting" \
  "$MERGED_PICKLE"

# Step 3: Register merged pickle in training nicknames.
# Note: add_to_dataset_nicknames.py registers per-alignment glob patterns;
# here we register the single merged pickle, so we update the JSON directly.
echo "=== Step 3: Registering in training nicknames ==="
python -c "
import json, sys
path = sys.argv[1]
key = sys.argv[2]
value = sys.argv[3]
with open(path) as f:
    data = json.load(f)
data[key] = value
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print(f'Added {key} to {path}')
" "$NICKNAMES_FILE" "$NICKNAME_KEY" "$(basename "$MERGED_PICKLE")"
