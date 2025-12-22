import json
import os
import argparse


def generate_sim_config_files(
    num_sequences,
    num_sites,
    num_algnmnts,
    edge_distribution="constant",
    remove_duplicate_site_patterns=False,
    max_trees=200,
    max_spr_moves=100,
    spr_move_divisor=10,
    subtree_max_attempts=100,
    subtree_target_non_mp_proportion=1 / 6,
):
    """Generate config JSON files for running the larch pipeline.

    Creates configuration files for generating DPVT training/testing data from
    alignments simulated by alisim (via create_alisim_alignments.sh).

    Args:
        num_sequences: Number of sequences in each alignment.
        num_sites: Number of sites per alignment.
        num_algnmnts: Number of alignments in the dataset.
        edge_distribution: Method for introducing non-MP edges. Options:
            "constant", "uniform", "treesearch_mimic", "random_subtree".
        remove_duplicate_site_patterns: If True, remove duplicate site patterns.
        max_trees: Maximum trees to extract per alignment.
        max_spr_moves: Maximum SPR moves per tree.
        spr_move_divisor: Divisor for constant SPR distribution.
        subtree_max_attempts: Max attempts for subtree replacement.
        subtree_target_non_mp_proportion: Target non-MP edge proportion.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create output directory if it doesn't exist
    config_dir = os.path.join(script_dir, "..", "configs")
    os.makedirs(config_dir, exist_ok=True)

    # Create the dataset name
    dataset_name = f"alisim_alignment_{num_sequences}_seq_{num_sites}_sites_{num_algnmnts}_algnmnts"

    # Create the config dictionary
    config = {
        "input_data": os.path.join(
            script_dir,
            "..",
            "..",
            "..",
            "data",
            "simulated_alignments",
            dataset_name,
        ),
        "larch_command": os.path.join(
            script_dir, "..", "..", "..", "..", "larch", "build", "bin", "larch-usher"
        ),
        "output_data": os.path.join(script_dir, "..", "..", "..", "data"),
        "dataset_name": dataset_name,
        "num_cores": 2,
        "edge_distribution": edge_distribution,
        "remove_duplicate_site_patterns": remove_duplicate_site_patterns,
        # Tree extraction parameters
        "max_trees": max_trees,
        "max_spr_moves": max_spr_moves,
        "spr_move_divisor": spr_move_divisor,
        "subtree_max_attempts": subtree_max_attempts,
        "subtree_target_non_mp_proportion": subtree_target_non_mp_proportion,
    }

    # Create filename for the config
    filename = f"config_{num_sequences}seq_{num_sites}sites_{num_algnmnts}alignments_{edge_distribution}.json"
    if remove_duplicate_site_patterns.lower() == "true" or (
        type(remove_duplicate_site_patterns) == type(True)
        and remove_duplicate_site_patterns
    ):
        print("Removing duplicate site patterns from alignments.")
        filename = filename.replace(".json", "_no_dup_sites.json")
    filepath = os.path.join(config_dir, filename)

    # Write the JSON file
    if os.path.exists(filepath):
        print(f"Config file {filename} exists already.")
    else:
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Generated {filename}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate config files for larch pipeline."
    )
    parser.add_argument("num_sequences", type=int, help="Number of sequences.")
    parser.add_argument("num_sites", type=int, help="Number of sites")
    parser.add_argument(
        "num_algnmnts", type=int, help="Number of alignments in dataset"
    )
    parser.add_argument(
        "--edge_distribution",
        type=str,
        default="constant",
        choices=["constant", "uniform", "treesearch_mimic", "random_subtree"],
        help="Edge distribution method (default: constant)",
    )
    parser.add_argument(
        "--remove_duplicate_site_patterns",
        type=str,
        default="False",
        help="Remove duplicate site patterns from alignments (default: False)",
    )
    # Tree extraction parameters
    parser.add_argument(
        "--max_trees",
        type=int,
        default=200,
        help="Maximum trees to extract per alignment (default: 200)",
    )
    parser.add_argument(
        "--max_spr_moves",
        type=int,
        default=100,
        help="Maximum SPR moves per tree (default: 100)",
    )
    parser.add_argument(
        "--spr_move_divisor",
        type=int,
        default=10,
        help="Divisor for constant SPR distribution (default: 10)",
    )
    parser.add_argument(
        "--subtree_max_attempts",
        type=int,
        default=100,
        help="Maximum attempts for subtree replacement (default: 100)",
    )
    parser.add_argument(
        "--subtree_target_non_mp_proportion",
        type=float,
        default=1 / 6,
        help="Target non-MP edge proportion for subtree replacement (default: 1/6)",
    )
    args = parser.parse_args()
    generate_sim_config_files(
        args.num_sequences,
        args.num_sites,
        args.num_algnmnts,
        args.edge_distribution,
        args.remove_duplicate_site_patterns,
        args.max_trees,
        args.max_spr_moves,
        args.spr_move_divisor,
        args.subtree_max_attempts,
        args.subtree_target_non_mp_proportion,
    )


if __name__ == "__main__":
    main()
