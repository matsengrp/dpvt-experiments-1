#!/usr/bin/env python3
"""
Generate config files for the three-phase DPVT pipeline.

Usage:
    python scripts/generate_configs.py <input_data_path> <config_prefix> [output_dir]

Arguments:
    input_data_path: Path to the input alignment directory
    config_prefix: Prefix for the generated config filenames (e.g., "simulated_15seq_20sites_100algnmnts")
    output_dir: Directory where config files will be created (default: "configs")

Example:
    python scripts/generate_configs.py ../../../data/simulated_alignments/alisim_alignment_15_seq_20_sites_100_algnmnts simulated_15seq_20sites_100algnmnts configs/

    This will generate:
        - configs/simulated_15seq_20sites_100algnmnts_prepare.yaml
        - configs/simulated_15seq_20sites_100algnmnts_train.yaml
        - configs/simulated_15seq_20sites_100algnmnts_test.yaml
"""

import os
import sys
from pathlib import Path


def generate_prepare_config(
    input_data,
    dataset_name,
    max_ambiguous_site_frac_per_seq=0.2,
    min_frac_sites_retained=0.8,
    test_fraction=0.2,
):
    """Generate Phase 1 & 2 config (prepare)."""
    return f"""# Config for {dataset_name} - Phases 1 & 2
# This config is for preprocessing and dataset preparation

# ============================================================================
# Dataset Identification
# ============================================================================
dataset_name: "{dataset_name}"
input_data: "{input_data}"

# ============================================================================
# Phase 1: Preprocessing Settings (sequence-level filtering)
# ============================================================================
max_ambiguous_site_frac_per_seq: {max_ambiguous_site_frac_per_seq}     # Remove sequences with >{int(max_ambiguous_site_frac_per_seq*100)}% gaps/ambiguous chars
remove_duplicate_site_patterns: False    # Whether to deduplicate columns

# ============================================================================
# Phase 2: Dataset Preparation Settings (alignment-level filtering)
# ============================================================================
output_datasets: "../../data"
min_frac_sites_retained: {min_frac_sites_retained}             # Exclude alignments that lost >{int((1-min_frac_sites_retained)*100)}% of sites

# Train/test splitting
create_train_test_split: true
test_fraction: {test_fraction}

# ============================================================================
# Phase 3: Training Data Generation Settings
# ============================================================================
larch_command: "larch"                   # Command to run larch (can be "larch", "larch-phylo", or a full path)
larch_output: "../../data"
edge_distribution: "constant"            # Options: constant, uniform, treesearch_mimic, random_subtree
num_cores: 8

# Tree extraction parameters (optional - defaults shown)
# max_trees: 200                         # Max trees to extract per alignment
# max_spr_moves: 100                     # Max SPR moves per tree
# spr_move_divisor: 10                   # Divisor for constant SPR distribution
# subtree_max_attempts: 100              # Max attempts for subtree replacement
# subtree_target_non_mp_proportion: 0.167  # Target non-MP edge proportion (~1/6)
"""


def generate_phase3_config(dataset_name, split_type, min_frac_sites_retained=0.8):
    """Generate Phase 3 config for train or test split.

    Args:
        dataset_name: Name of the dataset
        split_type: Either "train" or "test"
        min_frac_sites_retained: Minimum fraction of sites retained (default: 0.8)
    """
    split_name = "Training" if split_type == "train" else "Test"
    return f"""# Phase 3 Config: {split_name} Set for {dataset_name}
# This config points to the {split_type} split created by Phase 2

# Input: {split_name} split from Phase 2 (must match Phase 2 output naming)
input_data: "../../data/{dataset_name}_{split_type}_{min_frac_sites_retained}"

# Output directory for training data
output_data: "../../data"

# Dataset name for output files
dataset_name: "{dataset_name}_{split_type}_{min_frac_sites_retained}"

# Larch settings (should match unified config)
larch_command: "larch"
edge_distribution: "constant"
num_cores: 8
remove_duplicate_site_patterns: False

# Tree extraction parameters (optional - defaults shown)
# max_trees: 200                         # Max trees to extract per alignment
# max_spr_moves: 100                     # Max SPR moves per tree
# spr_move_divisor: 10                   # Divisor for constant SPR distribution
# subtree_max_attempts: 100              # Max attempts for subtree replacement
# subtree_target_non_mp_proportion: 0.167  # Target non-MP edge proportion (~1/6)
"""


def generate_train_config(dataset_name, min_frac_sites_retained=0.8):
    """Generate Phase 3 train config."""
    return generate_phase3_config(dataset_name, "train", min_frac_sites_retained)


def generate_test_config(dataset_name, min_frac_sites_retained=0.8):
    """Generate Phase 3 test config."""
    return generate_phase3_config(dataset_name, "test", min_frac_sites_retained)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_data = sys.argv[1]
    dataset_name = sys.argv[2]
    larch_path = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "configs"

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Generate configs
    prepare_config = generate_prepare_config(input_data, dataset_name)
    train_config = generate_train_config(dataset_name)
    test_config = generate_test_config(dataset_name)

    # Write files
    prepare_path = os.path.join(output_dir, f"{dataset_name}_prepare.yaml")
    train_path = os.path.join(output_dir, f"{dataset_name}_train.yaml")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.yaml")

    with open(prepare_path, "w") as f:
        f.write(prepare_config)
    with open(train_path, "w") as f:
        f.write(train_config)
    with open(test_path, "w") as f:
        f.write(test_config)

    print(f"Generated config files:")
    print(f"  {prepare_path}")
    print(f"  {train_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
