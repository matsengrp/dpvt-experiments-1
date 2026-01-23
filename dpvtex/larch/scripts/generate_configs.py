#!/usr/bin/env python3
"""
Generate config files for the DPVT pipeline.

Usage:
    python scripts/generate_configs.py -i <input_data> -d <dataset_name> -l <larch_command> [options]

Arguments:
    -i, --input-data: Path to the input alignment directory
    -d, --dataset-name: Name for the dataset (used as prefix for config filenames)
    -l, --larch-command: Command to run larch (e.g., "larch", "larch-phylo", or full path)

Options:
    -o, --output-dir: Directory where config files will be created (default: "configs")
    --no-split: Skip train/test splitting (for simulated data)
    -e, --edge-distributions: Methods for introducing non-MP edges (can specify multiple)

Examples:
    # For empirical data (with train/test split):
    python scripts/generate_configs.py -i ../../data/empirical -d my_dataset -l larch

    # For simulated data with multiple edge distributions:
    python scripts/generate_configs.py -i ../../data/simulated -d sim_15seq -l larch \\
        -e constant -e random_subtree --no-split

    This will generate:
        With split (default):
            - configs/my_dataset_prepare.yaml
            - configs/my_dataset_train.yaml
            - configs/my_dataset_test.yaml
        With --no-split:
            - configs/sim_15seq_prepare.yaml
            - configs/sim_15seq_generate.yaml
"""

import argparse
import os

# =============================================================================
# Default values for generated configs
# =============================================================================

# Phase 1: Preprocessing
DEFAULT_MAX_AMBIGUOUS_SITE_FRAC_PER_SEQ = (
    0.2  # Remove sequences with >20% gaps/ambiguous
)

# Phase 2: Dataset preparation
DEFAULT_MIN_FRAC_SITES_RETAINED = 0.8  # Exclude alignments that lost >20% of sites
DEFAULT_TEST_FRACTION = 0.2  # Fraction of alignments for test set

# Phase 3: Training data generation
DEFAULT_NUM_CORES = 8
DEFAULT_MAX_TREES = 200  # Max trees to extract per alignment

# SPR parameters
DEFAULT_SPR_RADIUS = 2  # Max topological distance for SPR regraft (None = unlimited)
DEFAULT_SPR_TARGET_NON_MP_PROPORTION = 0.1  # Target non-MP edge proportion (10%)
DEFAULT_MAX_SPR_ATTEMPTS = 100  # Max SPR attempts before stopping

# Subtree replacement parameters
DEFAULT_SUBTREE_MAX_ATTEMPTS = 100  # Max attempts for subtree replacement
DEFAULT_SUBTREE_TARGET_NON_MP_PROPORTION = 0.1  # Target non-MP edge proportion
DEFAULT_SUBTREE_DEPTH = 3  # Subtree depth for replacement (None = tree_depth // 2)


def format_edge_distributions(edge_distributions):
    """Format edge distributions as YAML list.

    Args:
        edge_distributions: List of edge distribution method names.

    Returns:
        YAML-formatted string for the edge_distributions config key.
    """
    if len(edge_distributions) == 1:
        return f'edge_distributions: ["{edge_distributions[0]}"]'
    lines = ["edge_distributions:"]
    for ed in edge_distributions:
        lines.append(f'  - "{ed}"')
    return "\n".join(lines)


def generate_prepare_config(
    input_data,
    dataset_name,
    larch_command,
    create_train_test_split=True,
    max_ambiguous_site_frac_per_seq=DEFAULT_MAX_AMBIGUOUS_SITE_FRAC_PER_SEQ,
    min_frac_sites_retained=DEFAULT_MIN_FRAC_SITES_RETAINED,
    test_fraction=DEFAULT_TEST_FRACTION,
    edge_distributions=None,
):
    """Generate Phase 1 & 2 config (prepare)."""
    if edge_distributions is None:
        edge_distributions = ["constant"]

    split_section = f"""# Train/test splitting
create_train_test_split: {str(create_train_test_split)}
test_fraction: {test_fraction}"""

    if not create_train_test_split:
        split_section = """# Train/test splitting disabled (use _generate.yaml for Phase 3)
create_train_test_split: False"""

    edge_dist_yaml = format_edge_distributions(edge_distributions)

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

{split_section}

# ============================================================================
# Phase 3: Training Data Generation Settings
# ============================================================================
larch_command: {larch_command}                   # Command to run larch (can be "larch", "larch-phylo", or a full path)
larch_output: "../../data"
{edge_dist_yaml}            # Options: constant, uniform, treesearch_mimic, random_subtree
num_cores: {DEFAULT_NUM_CORES}

# Tree extraction parameters
max_trees: {DEFAULT_MAX_TREES}                           # Max trees to extract per alignment

# SPR parameters (for constant/uniform edge distributions)
spr_radius: {DEFAULT_SPR_RADIUS}                          # Max topological distance for SPR regraft (null = unlimited)
spr_target_non_mp_proportion: {DEFAULT_SPR_TARGET_NON_MP_PROPORTION}   # Target non-MP edge proportion
max_spr_attempts: {DEFAULT_MAX_SPR_ATTEMPTS}                   # Max SPR attempts before stopping

# Subtree replacement parameters
subtree_max_attempts: {DEFAULT_SUBTREE_MAX_ATTEMPTS}                # Max attempts for subtree replacement
subtree_target_non_mp_proportion: {DEFAULT_SUBTREE_TARGET_NON_MP_PROPORTION}  # Target non-MP edge proportion
subtree_depth: {DEFAULT_SUBTREE_DEPTH}                        # Subtree depth for replacement (null = tree_depth // 2)
"""


def generate_phase3_config(
    dataset_name,
    split_type,
    larch_command,
    min_frac_sites_retained=DEFAULT_MIN_FRAC_SITES_RETAINED,
    edge_distributions=None,
):
    """Generate Phase 3 config for train, test, or filtered (no-split) data.

    Args:
        dataset_name: Name of the dataset
        split_type: Either "train", "test", or "filtered"
        larch_command: Command to run larch
        min_frac_sites_retained: Minimum fraction of sites retained (default: 0.8)
        edge_distributions: List of methods for introducing non-MP edges
    """
    if edge_distributions is None:
        edge_distributions = ["constant"]

    if split_type == "filtered":
        split_name = "All Data (no train/test split)"
        comment = "filtered dataset created by Phase 2 (no train/test split)"
    else:
        split_name = "Training" if split_type == "train" else "Test"
        comment = f"{split_type} split created by Phase 2"

    edge_dist_yaml = format_edge_distributions(edge_distributions)

    return f"""# Phase 3 Config: {split_name} for {dataset_name}
# This config points to the {comment}

# Input: {split_name} from Phase 2 (must match Phase 2 output naming)
input_data: "../../data/{dataset_name}_{split_type}_{min_frac_sites_retained}"

# Output directory for training data
output_data: "../../data"

# Dataset name for output files
dataset_name: "{dataset_name}_{split_type}_{min_frac_sites_retained}"

# Larch settings (should match unified config)
larch_command: "{larch_command}"
{edge_dist_yaml}
num_cores: {DEFAULT_NUM_CORES}
remove_duplicate_site_patterns: False

# Tree extraction parameters
max_trees: {DEFAULT_MAX_TREES}                           # Max trees to extract per alignment

# SPR parameters (for constant/uniform edge distributions)
spr_radius: {DEFAULT_SPR_RADIUS}                          # Max topological distance for SPR regraft (null = unlimited)
spr_target_non_mp_proportion: {DEFAULT_SPR_TARGET_NON_MP_PROPORTION}   # Target non-MP edge proportion
max_spr_attempts: {DEFAULT_MAX_SPR_ATTEMPTS}                   # Max SPR attempts before stopping

# Subtree replacement parameters
subtree_max_attempts: {DEFAULT_SUBTREE_MAX_ATTEMPTS}                # Max attempts for subtree replacement
subtree_target_non_mp_proportion: {DEFAULT_SUBTREE_TARGET_NON_MP_PROPORTION}  # Target non-MP edge proportion
subtree_depth: {DEFAULT_SUBTREE_DEPTH}                        # Subtree depth for replacement (null = tree_depth // 2)
"""


def generate_train_config(
    dataset_name,
    larch_command,
    min_frac_sites_retained=DEFAULT_MIN_FRAC_SITES_RETAINED,
    edge_distributions=None,
):
    """Generate Phase 3 train config."""
    return generate_phase3_config(
        dataset_name,
        "train",
        larch_command,
        min_frac_sites_retained,
        edge_distributions,
    )


def generate_test_config(
    dataset_name,
    larch_command,
    min_frac_sites_retained=DEFAULT_MIN_FRAC_SITES_RETAINED,
    edge_distributions=None,
):
    """Generate Phase 3 test config."""
    return generate_phase3_config(
        dataset_name, "test", larch_command, min_frac_sites_retained, edge_distributions
    )


def generate_no_split_config(
    dataset_name,
    larch_command,
    min_frac_sites_retained=DEFAULT_MIN_FRAC_SITES_RETAINED,
    edge_distributions=None,
):
    """Generate Phase 3 config for filtered data (no train/test split)."""
    return generate_phase3_config(
        dataset_name,
        "filtered",
        larch_command,
        min_frac_sites_retained,
        edge_distributions,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate config files for the DPVT pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # For empirical data (with train/test split):
    python scripts/generate_configs.py -i ../../data/empirical -d my_dataset -l larch

    # For simulated data with multiple edge distributions:
    python scripts/generate_configs.py -i ../../data/simulated -d sim_data -l larch \\
        -e constant -e random_subtree --no-split
        """,
    )
    parser.add_argument(
        "-i",
        "--input-data",
        required=True,
        help="Path to the input alignment directory",
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        required=True,
        help="Name for the dataset (used as prefix for config filenames)",
    )
    parser.add_argument(
        "-l",
        "--larch-command",
        required=True,
        help='Command to run larch (e.g., "larch", "larch-phylo", or full path)',
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="configs",
        help="Directory where config files will be created (default: configs)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip train/test splitting (for simulated data)",
    )
    parser.add_argument(
        "-e",
        "--edge-distributions",
        action="append",
        choices=["constant", "uniform", "treesearch_mimic", "random_subtree"],
        help="Method(s) for introducing non-MP edges (can specify multiple, default: constant)",
    )

    args = parser.parse_args()

    # Default to ["constant"] if no edge distributions specified
    edge_distributions = args.edge_distributions or ["constant"]

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate prepare config
    prepare_config = generate_prepare_config(
        args.input_data,
        args.dataset_name,
        args.larch_command,
        create_train_test_split=not args.no_split,
        edge_distributions=edge_distributions,
    )

    # Write prepare config
    prepare_path = os.path.join(args.output_dir, f"{args.dataset_name}_prepare.yaml")
    with open(prepare_path, "w") as f:
        f.write(prepare_config)

    print("Generated config files:")
    print(f"  {prepare_path}")

    if args.no_split:
        # Generate single generate config for filtered data
        generate_config = generate_no_split_config(
            args.dataset_name,
            args.larch_command,
            edge_distributions=edge_distributions,
        )
        generate_path = os.path.join(
            args.output_dir, f"{args.dataset_name}_generate.yaml"
        )
        with open(generate_path, "w") as f:
            f.write(generate_config)
        print(f"  {generate_path}")
    else:
        # Generate train and test configs
        train_config = generate_train_config(
            args.dataset_name,
            args.larch_command,
            edge_distributions=edge_distributions,
        )
        test_config = generate_test_config(
            args.dataset_name,
            args.larch_command,
            edge_distributions=edge_distributions,
        )

        train_path = os.path.join(args.output_dir, f"{args.dataset_name}_train.yaml")
        test_path = os.path.join(args.output_dir, f"{args.dataset_name}_test.yaml")

        with open(train_path, "w") as f:
            f.write(train_config)
        with open(test_path, "w") as f:
            f.write(test_config)

        print(f"  {train_path}")
        print(f"  {test_path}")


if __name__ == "__main__":
    main()
