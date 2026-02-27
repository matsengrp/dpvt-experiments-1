#!/usr/bin/env python
"""
Combine per-proportion pickle files into a single training set.

The larch pipeline generates separate pickle files for each target non-MP edge
proportion (e.g., 5%, 10%, ..., 100%). This script loads those per-proportion
pickles, subsamples each to a target number of trees, and combines them into a
single training pickle with uniform coverage across proportion levels.

Usage:
    python scripts/combine_proportion_pickles.py \
        --data-root ../../shared_data/orthomam_varied_prop_max1 \
        --output ../../shared_data/orthomam_varied_proportions.p \
        --trees-per-level 50

    # Dry run to inspect tree counts without loading large files:
    python scripts/combine_proportion_pickles.py \
        --data-root ../../shared_data/orthomam_varied_prop_max1 --dry-run
"""

import argparse
import os
import pickle
import random
import re
from glob import glob

DEFAULT_PATTERN = "orthomam_train_filtered_0.5_t*_spr_r2_t*.p"
DEFAULT_DATA_ROOT = "../../shared_data/orthomam_varied_prop_max1"
DEFAULT_TREES_PER_LEVEL = 50
DEFAULT_MAX_FILE_SIZE_GB = 10


def find_proportion_pickles(data_root, pattern):
    """Find per-proportion pickle files and extract proportion from filename.

    Returns:
        List of (proportion, file_path) tuples sorted by proportion.
    """
    full_pattern = os.path.join(data_root, pattern)
    paths = sorted(glob(full_pattern))

    results = []
    for path in paths:
        basename = os.path.basename(path)
        # Extract proportion from filename like "..._t0.05_spr_r2_t0.05.p"
        # or "..._t0.1_spr_r2_t0.1_m1.p" (with max_trees suffix)
        # The proportion appears after the last "_t" before ".p" or "_m{N}.p"
        name, _ = os.path.splitext(basename)
        # Strip optional _m{N} suffix before parsing proportion
        name = re.sub(r"_m\d+$", "", name)
        parts = name.rsplit("_t", 1)
        if len(parts) == 2:
            try:
                proportion = float(parts[1])
                results.append((proportion, path))
            except ValueError:
                print(
                    f"  WARNING: Could not parse proportion from {basename}, skipping"
                )
        else:
            print(f"  WARNING: Unexpected filename format: {basename}, skipping")

    return sorted(results, key=lambda x: x[0])


def get_file_size_gb(path):
    """Get file size in GB."""
    return os.path.getsize(path) / (1024**3)


def load_and_subsample(pickle_path, trees_per_level, rng):
    """Load a pickle and subsample to trees_per_level trees.

    Args:
        pickle_path: Path to pickle file (dict of tree -> labels).
        trees_per_level: Target number of trees to keep.
        rng: random.Random instance for reproducibility.

    Returns:
        Tuple of (subsampled_dict, original_count, sampled_count).
    """
    with open(pickle_path, "rb") as f:
        tree_dict = pickle.load(f)

    original_count = len(tree_dict)

    if original_count <= trees_per_level:
        return tree_dict, original_count, original_count

    items = list(tree_dict.items())
    sampled_items = rng.sample(items, trees_per_level)
    return dict(sampled_items), original_count, trees_per_level


def report_nonmp_stats(combined_dict):
    """Report non-MP edge fraction statistics for the combined dataset."""
    fractions = []
    for tree, labels in combined_dict.items():
        num_leaves = len(tree)
        # Internal edges excluding masked root + first child
        num_internal = len(labels) - 2 - num_leaves
        num_nonmp = sum(
            labels[2:]
        )  # leaves are always 0, so sum is just non-MP internals
        if num_internal > 0:
            frac = num_nonmp / num_internal
            fractions.append(frac)

    if not fractions:
        return

    fractions.sort()
    print(f"\nNon-MP edge fraction statistics ({len(fractions)} trees):")
    print(f"  Min:    {fractions[0]:.3f}")
    print(f"  25th:   {fractions[len(fractions)//4]:.3f}")
    print(f"  Median: {fractions[len(fractions)//2]:.3f}")
    print(f"  75th:   {fractions[3*len(fractions)//4]:.3f}")
    print(f"  Max:    {fractions[-1]:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-proportion pickle files into a single training set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Directory containing the per-proportion pickle files (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern for per-proportion pickles (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path for the combined pickle (default: <data_root>/orthomam_varied_proportions.p)",
    )
    parser.add_argument(
        "--trees-per-level",
        type=int,
        default=DEFAULT_TREES_PER_LEVEL,
        help=f"Number of trees to sample per proportion level (default: {DEFAULT_TREES_PER_LEVEL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-file-size-gb",
        type=float,
        default=DEFAULT_MAX_FILE_SIZE_GB,
        help=f"Skip files larger than this size in GB (default: {DEFAULT_MAX_FILE_SIZE_GB})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report file sizes and skip loading/combining",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        args.data_root, "orthomam_varied_proportions.p"
    )

    # Discover files
    print(f"Searching for files matching: {args.pattern}")
    print(f"  in: {args.data_root}")
    proportion_files = find_proportion_pickles(args.data_root, args.pattern)

    if not proportion_files:
        print("ERROR: No matching pickle files found.")
        return 1

    print(f"\nFound {len(proportion_files)} proportion levels:\n")

    # Report file sizes
    print(f"{'Proportion':>12}  {'Size':>10}  {'Path'}")
    print("-" * 80)
    skipped = []
    to_process = []
    for proportion, path in proportion_files:
        size_gb = get_file_size_gb(path)
        size_str = f"{size_gb:.1f} GB" if size_gb >= 1 else f"{size_gb*1024:.0f} MB"
        skip = size_gb > args.max_file_size_gb
        marker = " ** SKIP (too large)" if skip else ""
        print(f"{proportion:>12.2f}  {size_str:>10}  {os.path.basename(path)}{marker}")
        if skip:
            skipped.append((proportion, path, size_gb))
        else:
            to_process.append((proportion, path))

    if skipped:
        print(
            f"\nWARNING: {len(skipped)} file(s) exceed {args.max_file_size_gb} GB and will be skipped."
        )
        print("  These may be from earlier pipeline runs without max_trees=1.")
        print(
            "  Use --max-file-size-gb to adjust the threshold, or re-run the pipeline for these proportions."
        )

    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to combine files.")
        return 0

    # Load, subsample, and combine
    rng = random.Random(args.seed)
    combined = {}
    print(f"\nSubsampling to {args.trees_per_level} trees per level:\n")
    print(f"{'Proportion':>12}  {'Trees found':>12}  {'Sampled':>8}")
    print("-" * 40)

    for proportion, path in to_process:
        subsampled, original, sampled = load_and_subsample(
            path, args.trees_per_level, rng
        )
        combined.update(subsampled)
        marker = " (all)" if sampled == original else ""
        print(f"{proportion:>12.2f}  {original:>12}  {sampled:>8}{marker}")

    print(f"\nTotal trees in combined dataset: {len(combined)}")

    # Report non-MP statistics
    report_nonmp_stats(combined)

    # Save
    print(f"\nSaving combined dataset to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(combined, f)

    size_gb = get_file_size_gb(output_path)
    size_str = f"{size_gb:.1f} GB" if size_gb >= 1 else f"{size_gb*1024:.0f} MB"
    print(f"Output size: {size_str}")
    print("Done.")
    return 0


if __name__ == "__main__":
    exit(main())
