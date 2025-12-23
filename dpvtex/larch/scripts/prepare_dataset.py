"""
Prepare filtered datasets with symlinks for training data generation.

This module creates filtered output directories with symlinks to original
preprocessed alignments based on quality thresholds, and optionally splits
them into train/test sets.
"""

import os
import pandas as pd
import numpy as np


def _create_symlinks_for_split(alignments, source_dir, target_dir):
    """Create symlinks from source to target directory for given alignments.

    Returns the count of symlinks created.
    """
    os.makedirs(target_dir, exist_ok=True)
    created = 0
    for name in alignments:
        source_path = os.path.abspath(os.path.join(source_dir, name))
        target_path = os.path.join(target_dir, name)
        if os.path.exists(source_path) and not os.path.exists(target_path):
            os.symlink(source_path, target_path, target_is_directory=True)
            created += 1
    return created


def _write_split_manifest(
    manifest_path, split_type, source_manifest, alignments, test_fraction
):
    """Write a manifest file for a train/test split."""
    with open(manifest_path, "w") as f:
        f.write(f"# {split_type} set\n")
        f.write(f"# Source manifest: {source_manifest}\n")
        f.write(f"# Total alignments: {len(alignments)}\n")
        f.write(f"# Test fraction: {test_fraction}\n")
        f.write(f"# Date created: {pd.Timestamp.now()}\n\n")
        for name in sorted(alignments):
            f.write(f"{name}\n")


def create_filtered_dataset(
    source_dir, stats_file, output_dir, min_frac_sites_retained=0.8
):
    """
    Create output directory with symlinks to alignments passing quality filters.

    Parameters:
    -----------
    source_dir : str
        Source directory containing preprocessed alignment subdirectories
    stats_file : str
        Path to alignment_size_stats.csv file
    output_dir : str
        Output directory where symlinks will be created
    min_frac_sites_retained : float
        Minimum fraction of sites retained (cleaned_sites / original_sites)

    Returns:
    --------
    list
        List of alignment names that were included in the filtered dataset
    """
    # Read stats
    df = pd.read_csv(stats_file)

    # Filter by site ratio
    filtered_df = df[df["site_ratio"] >= min_frac_sites_retained]

    print(f"\nFiltering alignments:")
    print(f"  Total alignments: {len(df)}")
    print(f"  Passed site_ratio >= {min_frac_sites_retained}: {len(filtered_df)}")
    print(f"  Excluded: {len(df) - len(filtered_df)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create symlinks
    created = []
    skipped = []

    for _, row in filtered_df.iterrows():
        alignment_name = row["alignment_name"]
        source_path = os.path.abspath(os.path.join(source_dir, alignment_name))
        target_path = os.path.join(output_dir, alignment_name)

        if not os.path.exists(source_path):
            skipped.append(alignment_name)
            continue

        if not os.path.exists(target_path):
            os.symlink(source_path, target_path, target_is_directory=True)
            created.append(alignment_name)
        else:
            # Symlink already exists
            created.append(alignment_name)

    if skipped:
        print(f"\n  Warning: {len(skipped)} alignments not found in source directory")

    # Write manifest
    manifest_file = os.path.join(output_dir, "manifest.txt")
    with open(manifest_file, "w") as f:
        f.write(f"# Filtered dataset\n")
        f.write(f"# Source: {source_dir}\n")
        f.write(f"# Filter: site_ratio >= {min_frac_sites_retained}\n")
        f.write(f"# Total alignments: {len(created)}\n")
        f.write(f"# Date created: {pd.Timestamp.now()}\n\n")
        for name in sorted(created):
            f.write(f"{name}\n")

    print(f"\nCreated filtered dataset:")
    print(f"  Output directory: {output_dir}")
    print(f"  Alignments: {len(created)}")
    print(f"  Manifest: {manifest_file}")

    return created


def create_train_test_split(
    source_dir, filtered_manifest, train_dir, test_dir, test_fraction=0.2
):
    """
    Split filtered dataset into train/test sets with symlinks.

    Uses system randomness for splitting (not reproducible - different split each time).

    Parameters:
    -----------
    source_dir : str
        Source directory containing the original alignment subdirectories
    filtered_manifest : str
        Path to manifest file from filtered dataset
    train_dir : str
        Output directory for training set symlinks
    test_dir : str
        Output directory for test set symlinks
    test_fraction : float
        Fraction of alignments for test set (default: 0.2)

    Returns:
    --------
    tuple
        (train_alignments, test_alignments) - lists of alignment names
    """
    # Read manifest
    alignments = []
    with open(filtered_manifest, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                alignments.append(line)

    if len(alignments) == 0:
        raise ValueError(f"No alignments found in manifest: {filtered_manifest}")

    # Random split (uses system randomness, not reproducible)
    shuffled = np.random.permutation(alignments)
    n_test = int(len(shuffled) * test_fraction)
    n_train = len(shuffled) - n_test

    test_alignments = shuffled[:n_test].tolist()
    train_alignments = shuffled[n_test:].tolist()

    print(f"\nSplitting dataset:")
    print(f"  Total alignments: {len(alignments)}")
    print(f"  Train set: {n_train} ({(1-test_fraction)*100:.0f}%)")
    print(f"  Test set: {n_test} ({test_fraction*100:.0f}%)")

    # Create directories with symlinks
    train_created = _create_symlinks_for_split(train_alignments, source_dir, train_dir)
    test_created = _create_symlinks_for_split(test_alignments, source_dir, test_dir)

    # Write manifests
    train_manifest = os.path.join(train_dir, "manifest.txt")
    test_manifest = os.path.join(test_dir, "manifest.txt")
    _write_split_manifest(
        train_manifest, "Train", filtered_manifest, train_alignments, test_fraction
    )
    _write_split_manifest(
        test_manifest, "Test", filtered_manifest, test_alignments, test_fraction
    )

    print(f"\nCreated train/test split:")
    print(f"  Train directory: {train_dir}")
    print(f"    Symlinks created: {train_created}")
    print(f"    Manifest: {train_manifest}")
    print(f"  Test directory: {test_dir}")
    print(f"    Symlinks created: {test_created}")
    print(f"    Manifest: {test_manifest}")

    return train_alignments, test_alignments
