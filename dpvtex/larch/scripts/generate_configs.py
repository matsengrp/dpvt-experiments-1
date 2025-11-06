#!/usr/bin/env python3
"""
Generate config files for the three-phase DPVT pipeline.

Usage:
    python scripts/generate_configs.py <input_data_path> <dataset_name> [output_dir]

Example:
    python scripts/generate_configs.py ../../../data/simulated_alignments/alisim_alignment_15_seq_20_sites_100_algnmnts simulated_15seq_20sites_100algnmnts configs/
"""

import os
import sys
from pathlib import Path


def generate_prepare_config(input_data, dataset_name,
                            max_ambiguous_site_frac_per_seq=0.2,
                            min_frac_sites_retained=0.8,
                            test_fraction=0.2):
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
remove_duplicate_site_patterns: false    # Whether to deduplicate columns

# ============================================================================
# Phase 2: Dataset Preparation Settings (alignment-level filtering)
# ============================================================================
output_datasets: "../../shared_data"
min_frac_sites_retained: {min_frac_sites_retained}             # Exclude alignments that lost >{int((1-min_frac_sites_retained)*100)}% of sites

# Train/test splitting
create_train_test_split: true
test_fraction: {test_fraction}

# ============================================================================
# Phase 3: Training Data Generation Settings
# ============================================================================
larch_build: "../../../larch/build"
larch_output: "../../shared_data"
edge_distribution: "constant"            # Options: constant, uniform, treesearch_mimic, random_subtree
num_cores: 8
"""


def generate_train_config(dataset_name, min_frac_sites_retained=0.8):
    """Generate Phase 3 train config."""
    return f"""# Phase 3 Config: Training Set for {dataset_name}
# This config points to the train split created by Phase 2

# Input: Train split from Phase 2 (must match Phase 2 output naming)
input_data: "../../shared_data/{dataset_name}_train_{min_frac_sites_retained}"

# Output directory for training data
output_data: "../../shared_data"

# Dataset name for output files
dataset_name: "{dataset_name}_train_{min_frac_sites_retained}"

# Larch settings (should match unified config)
larch_build: "../../../larch/build"
edge_distribution: "constant"
num_cores: 8
remove_duplicate_site_patterns: false
"""


def generate_test_config(dataset_name, min_frac_sites_retained=0.8):
    """Generate Phase 3 test config."""
    return f"""# Phase 3 Config: Test Set for {dataset_name}
# This config points to the test split created by Phase 2

# Input: Test split from Phase 2 (must match Phase 2 output naming)
input_data: "../../shared_data/{dataset_name}_test_{min_frac_sites_retained}"

# Output directory for training data
output_data: "../../shared_data"

# Dataset name for output files
dataset_name: "{dataset_name}_test_{min_frac_sites_retained}"

# Larch settings (should match unified config)
larch_build: "../../../larch/build"
edge_distribution: "constant"
num_cores: 8
remove_duplicate_site_patterns: false
"""


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_data = sys.argv[1]
    dataset_name = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "configs"

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

    with open(prepare_path, 'w') as f:
        f.write(prepare_config)
    with open(train_path, 'w') as f:
        f.write(train_config)
    with open(test_path, 'w') as f:
        f.write(test_config)

    print(f"Generated config files:")
    print(f"  {prepare_path}")
    print(f"  {train_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
