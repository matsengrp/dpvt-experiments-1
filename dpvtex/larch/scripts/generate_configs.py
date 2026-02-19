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
    -dp, --dataset-path: Directory where datasets will be created (default: "../../data")
    --no-split: Skip train/test splitting (for simulated data)
    -e, --edge-distributions: Methods for introducing non-MP edges (can specify multiple)
    -p, --proportions: Target non-MP edge proportions (can specify multiple, generates
                       one Phase 3 config per proportion)
    --p3-input-dir: Override the Phase 3 input directory path (for pre-existing data
                    that doesn't follow the standard naming convention)

Examples:
    # For empirical data (with train/test split):
    python scripts/generate_configs.py -i ../../data/empirical -d my_dataset -l larch

    # For simulated data with multiple edge distributions:
    python scripts/generate_configs.py -i ../../data/simulated -d sim_15seq -l larch \\
        -e constant -e random_subtree --no-split

    # Generate multiple configs with different non-MP proportions:
    python scripts/generate_configs.py -i ../../data/empirical -d my_dataset -l larch \\
        -p 0.1 -p 0.2 -p 0.3 -p 0.5

    # For pre-existing data that doesn't follow naming convention:
    python scripts/generate_configs.py -i ../../data/original -d pandit_test -l larch \\
        --no-split -p 0.1 --p3-input-dir "../../data/pandit_test_0.8"

    This will generate:
        With split (default):
            - configs/my_dataset_prepare.yaml
            - configs/my_dataset_train.yaml
            - configs/my_dataset_test.yaml
        With --no-split:
            - configs/sim_15seq_prepare.yaml
            - configs/sim_15seq_generate.yaml
        With --proportions (one config set per proportion):
            - configs/my_dataset_prepare_t0.1.yaml
            - configs/my_dataset_prepare_t0.2.yaml
            - configs/my_dataset_train_t0.1.yaml
            - configs/my_dataset_train_t0.2.yaml
            - configs/my_dataset_test_t0.1.yaml
            - configs/my_dataset_test_t0.2.yaml
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
    dataset_path,
    create_train_test_split=True,
    max_ambiguous_site_frac_per_seq=DEFAULT_MAX_AMBIGUOUS_SITE_FRAC_PER_SEQ,
    min_frac_sites_retained=DEFAULT_MIN_FRAC_SITES_RETAINED,
    test_fraction=DEFAULT_TEST_FRACTION,
    edge_distributions=None,
    spr_target_non_mp_proportion=None,
    subtree_target_non_mp_proportion=None,
    max_trees=DEFAULT_MAX_TREES,
):
    """Generate Phase 1 & 2 config (prepare).

    Args:
        input_data: Path to input alignment directory.
        dataset_name: Name for the dataset.
        larch_command: Command to run larch.
        dataset_path: Directory where datasets will be created.
        create_train_test_split: Whether to create train/test split.
        max_ambiguous_site_frac_per_seq: Max fraction of ambiguous sites per sequence.
        min_frac_sites_retained: Min fraction of sites retained after cleaning.
        test_fraction: Fraction of alignments for test set.
        edge_distributions: List of edge distribution methods.
        spr_target_non_mp_proportion: Target non-MP edge proportion for SPR methods.
        subtree_target_non_mp_proportion: Target non-MP edge proportion for subtree methods.
    """
    if edge_distributions is None:
        edge_distributions = ["constant"]

    # Use defaults if not specified
    spr_proportion = (
        spr_target_non_mp_proportion
        if spr_target_non_mp_proportion is not None
        else DEFAULT_SPR_TARGET_NON_MP_PROPORTION
    )
    subtree_proportion = (
        subtree_target_non_mp_proportion
        if subtree_target_non_mp_proportion is not None
        else DEFAULT_SUBTREE_TARGET_NON_MP_PROPORTION
    )

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
output_datasets: "{dataset_path}"
min_frac_sites_retained: {min_frac_sites_retained}             # Exclude alignments that lost >{int((1-min_frac_sites_retained)*100)}% of sites

{split_section}

# ============================================================================
# Phase 3: Training Data Generation Settings
# ============================================================================
larch_command: {larch_command}                   # Command to run larch (can be "larch", "larch-phylo", or a full path)
larch_output: "{dataset_path}"
{edge_dist_yaml}            # Options: constant, uniform, treesearch_mimic, random_subtree
num_cores: {DEFAULT_NUM_CORES}

# Tree extraction parameters
max_trees: {max_trees}                           # Max trees to extract per alignment

# SPR parameters (for constant/uniform edge distributions)
spr_radius: {DEFAULT_SPR_RADIUS}                          # Max topological distance for SPR regraft (null = unlimited)
spr_target_non_mp_proportion: {spr_proportion}   # Target non-MP edge proportion
max_spr_attempts: {DEFAULT_MAX_SPR_ATTEMPTS}                   # Max SPR attempts before stopping

# Subtree replacement parameters
subtree_max_attempts: {DEFAULT_SUBTREE_MAX_ATTEMPTS}                # Max attempts for subtree replacement
subtree_target_non_mp_proportion: {subtree_proportion}  # Target non-MP edge proportion
subtree_depth: {DEFAULT_SUBTREE_DEPTH}                        # Subtree depth for replacement (null = tree_depth // 2)
"""


def generate_phase3_config(
    dataset_name,
    split_type,
    larch_command,
    dataset_path,
    min_frac_sites_retained=DEFAULT_MIN_FRAC_SITES_RETAINED,
    edge_distributions=None,
    spr_target_non_mp_proportion=None,
    subtree_target_non_mp_proportion=None,
    input_dir_override=None,
    proportion_suffix="",
    max_trees=DEFAULT_MAX_TREES,
):
    """Generate Phase 3 config for train, test, or filtered (no-split) data.

    Args:
        dataset_name: Name of the dataset
        split_type: Either "train", "test", or "filtered"
        larch_command: Command to run larch
        min_frac_sites_retained: Minimum fraction of sites retained (default: 0.8)
        edge_distributions: List of methods for introducing non-MP edges
        spr_target_non_mp_proportion: Target proportion of non-MP edges for SPR methods.
            If None, uses DEFAULT_SPR_TARGET_NON_MP_PROPORTION.
        subtree_target_non_mp_proportion: Target proportion of non-MP edges for subtree
            replacement. If None, uses DEFAULT_SUBTREE_TARGET_NON_MP_PROPORTION.
        input_dir_override: If provided, use this path for input_data instead of
            constructing it from dataset_name and split_type.
        proportion_suffix: Suffix to append to the output dataset name (e.g., "_t0.05")
            to distinguish outputs generated with different target proportions.
    """
    if edge_distributions is None:
        edge_distributions = ["constant"]

    # Use defaults if not specified
    spr_proportion = (
        spr_target_non_mp_proportion
        if spr_target_non_mp_proportion is not None
        else DEFAULT_SPR_TARGET_NON_MP_PROPORTION
    )
    subtree_proportion = (
        subtree_target_non_mp_proportion
        if subtree_target_non_mp_proportion is not None
        else DEFAULT_SUBTREE_TARGET_NON_MP_PROPORTION
    )

    if split_type == "filtered":
        split_name = "All Data (no train/test split)"
        comment = "filtered dataset created by Phase 2 (no train/test split)"
    else:
        split_name = "Training" if split_type == "train" else "Test"
        comment = f"{split_type} split created by Phase 2"

    edge_dist_yaml = format_edge_distributions(edge_distributions)

    # Determine input path, output path, and dataset name
    if input_dir_override is not None:
        input_data_path = input_dir_override
        # Output goes to parent directory of input
        output_data_path = os.path.dirname(input_dir_override.rstrip("/"))
        # Extract dataset name from the override path for output naming
        output_dataset_name = (
            os.path.basename(input_dir_override.rstrip("/")) + proportion_suffix
        )
    else:
        input_data_path = (
            f"{dataset_path}/{dataset_name}_{split_type}_{min_frac_sites_retained}"
        )
        output_data_path = dataset_path
        output_dataset_name = (
            f"{dataset_name}_{split_type}_{min_frac_sites_retained}{proportion_suffix}"
        )

    return f"""# Phase 3 Config: {split_name} for {dataset_name}
# This config points to the {comment}

# Input: {split_name} from Phase 2 (must match Phase 2 output naming)
input_data: "{input_data_path}"

# Output directory for training data
output_data: "{output_data_path}"

# Dataset name for output files
dataset_name: "{output_dataset_name}"

# Larch settings (should match unified config)
larch_command: "{larch_command}"
{edge_dist_yaml}
num_cores: {DEFAULT_NUM_CORES}
remove_duplicate_site_patterns: False

# Tree extraction parameters
max_trees: {max_trees}                           # Max trees to extract per alignment

# SPR parameters (for constant/uniform edge distributions)
spr_radius: {DEFAULT_SPR_RADIUS}                          # Max topological distance for SPR regraft (null = unlimited)
spr_target_non_mp_proportion: {spr_proportion}   # Target non-MP edge proportion
max_spr_attempts: {DEFAULT_MAX_SPR_ATTEMPTS}                   # Max SPR attempts before stopping

# Subtree replacement parameters
subtree_max_attempts: {DEFAULT_SUBTREE_MAX_ATTEMPTS}                # Max attempts for subtree replacement
subtree_target_non_mp_proportion: {subtree_proportion}  # Target non-MP edge proportion
subtree_depth: {DEFAULT_SUBTREE_DEPTH}                        # Subtree depth for replacement (null = tree_depth // 2)
"""


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

    # Generate multiple configs with different non-MP proportions:
    python scripts/generate_configs.py -i ../../data/empirical -d my_dataset -l larch \\
        -p 0.1 -p 0.2 -p 0.3
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
        "-dp",
        "--dataset-path",
        default="../../data",
        help="Directory where datasets will be created (default: ../../data)",
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
    parser.add_argument(
        "-p",
        "--proportions",
        action="append",
        type=float,
        help="Target non-MP edge proportion(s) (can specify multiple, e.g., -p 0.1 -p 0.2)",
    )
    parser.add_argument(
        "--p3-input-dir",
        help="Override the Phase 3 input directory path (for pre-existing data that doesn't follow naming convention)",
    )
    parser.add_argument(
        "--max-trees",
        type=int,
        default=DEFAULT_MAX_TREES,
        help=f"Max trees to extract per alignment (default: {DEFAULT_MAX_TREES})",
    )

    args = parser.parse_args()

    # Default to ["constant"] if no edge distributions specified
    edge_distributions = args.edge_distributions or ["constant"]

    # Default to [None] if no proportions specified (uses defaults in config)
    proportions = args.proportions or [None]

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which split types to generate configs for
    split_types = ["filtered"] if args.no_split else ["train", "test"]

    print("Generated config files:")

    # Generate configs for each proportion
    for proportion in proportions:
        # Build suffix for filenames
        suffix = f"_t{proportion}" if proportion is not None else ""

        # Generate prepare config (for running full pipeline with run_all_on_simulated)
        prepare_config = generate_prepare_config(
            args.input_data,
            args.dataset_name,
            args.larch_command,
            args.dataset_path,
            create_train_test_split=not args.no_split,
            edge_distributions=edge_distributions,
            spr_target_non_mp_proportion=proportion,
            subtree_target_non_mp_proportion=proportion,
            max_trees=args.max_trees,
        )

        prepare_path = os.path.join(
            args.output_dir, f"{args.dataset_name}_prepare{suffix}.yaml"
        )
        with open(prepare_path, "w") as f:
            f.write(prepare_config)
        print(f"  {prepare_path}")

        # Generate Phase 3 configs for each split type
        for split_type in split_types:
            config = generate_phase3_config(
                args.dataset_name,
                split_type,
                args.larch_command,
                args.dataset_path,
                edge_distributions=edge_distributions,
                spr_target_non_mp_proportion=proportion,
                subtree_target_non_mp_proportion=proportion,
                input_dir_override=(
                    args.p3_input_dir if split_type == "filtered" else None
                ),
                proportion_suffix=suffix,
                max_trees=args.max_trees,
            )

            # Build filename with appropriate config type name
            config_type = "generate" if split_type == "filtered" else split_type
            filename = f"{args.dataset_name}_{config_type}{suffix}.yaml"

            config_path = os.path.join(args.output_dir, filename)
            with open(config_path, "w") as f:
                f.write(config)
            print(f"  {config_path}")


if __name__ == "__main__":
    main()
