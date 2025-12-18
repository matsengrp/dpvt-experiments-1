import os
import pickle
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from collections import Counter


def get_dict(file_path):
    """Load a dictionary from a pickle file.

    Args:
        file_path: Path to the .p pickle file to load

    Returns:
        dict: The dictionary stored in the pickle file, or None if the file
            does not contain a dictionary
    """
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        if isinstance(data, dict):
            return data
        else:
            return None


def _count_trees_per_alignment(pickle_files):
    """Count the number of trees in each alignment pickle file.

    Args:
        pickle_files: List of (file_path, dataset_name) tuples.

    Returns:
        dict: Mapping of dataset_name to tree count.
    """
    alignment_tree_counts = {}
    for file_path, dataset_name in pickle_files:
        this_alignment_dict = _get_dict(file_path)
        # Skip None or empty dictionaries (from larch timeouts/failures)
        if this_alignment_dict is not None and len(this_alignment_dict) > 0:
            alignment_tree_counts[dataset_name] = len(this_alignment_dict)
    return alignment_tree_counts


def _compute_data_properties(alignment_dict):
    """Compute properties for a single alignment's tree dictionary.

    Args:
        alignment_dict: Dictionary mapping trees to edge label lists.

    Returns:
        list: [num_trees, num_leaves, num_MP_edges, num_non_MP_edges]
    """
    num_trees = len(alignment_dict)
    num_leaves = len(list(alignment_dict.keys())[0])
    num_MP_edges = sum(
        edge_labels.count(0) - (len(edge_labels)) / 2
        for edge_labels in alignment_dict.values()
    )
    num_non_MP_edges = sum(
        edge_labels.count(1) for edge_labels in alignment_dict.values()
    )
    return [num_trees, num_leaves, num_MP_edges, num_non_MP_edges]


def _load_and_balance_trees(
    pickle_files, median_trees, balance_by_median_num_MP_trees, logger
):
    """Load trees from pickle files and apply balancing if enabled.

    Alignments with more than the median number of trees are subsampled
    to the median count when balancing is enabled.

    Args:
        pickle_files: List of (file_path, dataset_name) tuples.
        median_trees: Median tree count across all alignments.
        balance_by_median_num_MP_trees: Whether to apply balancing.
        logger: PipelineLogger for progress tracking.

    Returns:
        tuple: (all_trees_dict, data_props, subsampled_alignments, trees_removed)
    """
    all_trees_dict = {}
    data_props = {}
    subsampled_alignments = []
    trees_removed = 0

    for file_path, dataset_name in pickle_files:
        this_alignment_dict = _get_dict(file_path)
        # Skip None or empty dictionaries (from larch timeouts/failures)
        if this_alignment_dict is None or len(this_alignment_dict) == 0:
            continue

        original_count = len(this_alignment_dict)

        # Apply balancing if enabled
        if balance_by_median_num_MP_trees and original_count > median_trees:
            target_count = ceil(median_trees)
            sampled_items = random.sample(
                list(this_alignment_dict.items()), target_count
            )
            this_alignment_dict = dict(sampled_items)

            trees_removed += original_count - target_count
            subsampled_alignments.append(
                {
                    "alignment": dataset_name,
                    "original": original_count,
                    "sampled": target_count,
                    "removed": original_count - target_count,
                }
            )

            logger.log(
                "AGGREGATION",
                f"  {dataset_name}: {original_count} → {target_count} trees (removed {original_count - target_count})",
            )

        # Compute data properties (using final tree count after balancing)
        data_props[dataset_name] = _compute_data_properties(this_alignment_dict)

        # Add to final dictionary
        all_trees_dict.update(this_alignment_dict)

    return all_trees_dict, data_props, subsampled_alignments, trees_removed


def _log_balancing_summary(
    logger, subsampled_alignments, trees_removed, total_trees_after
):
    """Log the summary of balancing operations.

    Args:
        logger: PipelineLogger instance for output.
        subsampled_alignments: List of dicts with subsampling info.
        trees_removed: Total number of trees removed by balancing.
        total_trees_after: Total tree count after balancing.
    """
    logger.log_section("AGGREGATION", "Balancing Summary")
    logger.log("AGGREGATION", f"Alignments subsampled: {len(subsampled_alignments)}")
    logger.log("AGGREGATION", f"Total trees removed: {trees_removed}")
    logger.log("AGGREGATION", f"Total trees after balancing: {total_trees_after}")

    logger.log("AGGREGATION", "\nSubsampled alignments:")
    for info in subsampled_alignments:
        logger.log(
            "AGGREGATION",
            f"  {info['alignment']}: {info['original']} → {info['sampled']} trees",
        )


def extract_trees_and_labels(
    data_dir,
    edge_distribution="constant",
    balance_by_median_num_MP_trees=True,
    logger=None,
):
    """Extracts trees and labels from .p files in the given directory.

    Args:
        data_dir (str): Directory containing .p files.
        edge_distribution (str): Type of edge distribution ("constant", "uniform", "treesearch_mimic", "random_subtree")
        balance_by_median_num_MP_trees (bool): If True, subsample alignments with more than median trees to balance dataset.
        logger (PipelineLogger): Logger for tracking operations.

    Returns:
        tuple: A tuple containing:
            - trees (list): List of trees.
            - labels (list): List of labels.
            - all_trees_dict (dict): Dictionary containing all trees.
            - data_props (dict): Dictionary containing properties of datasets.
    """
    # Collect all pickle files
    expected_suffix = _get_expected_suffix(edge_distribution)
    pickle_files = _collect_pickle_files(data_dir, expected_suffix)

    logger.log("AGGREGATION", f"Found {len(pickle_files)} pickle files to process")
    logger.log_section("AGGREGATION", "Pass 1: Counting trees per alignment")

    # PASS 1: Count trees per alignment
    alignment_tree_counts = _count_trees_per_alignment(pickle_files)
    median_trees = median(alignment_tree_counts.values())
    total_trees_before = sum(alignment_tree_counts.values())

    logger.log("AGGREGATION", f"Total alignments: {len(alignment_tree_counts)}")
    logger.log("AGGREGATION", f"Median trees per alignment: {median_trees}")
    logger.log("AGGREGATION", f"Total trees before balancing: {total_trees_before}")
    if balance_by_median_num_MP_trees:
        logger.log(
            "AGGREGATION",
            "Balancing enabled - will subsample alignments with > median trees",
        )
    else:
        logger.log("AGGREGATION", "Balancing disabled - using all trees")

    # PASS 2: Load and aggregate trees (with optional balancing)
    logger.log_section("AGGREGATION", "Pass 2: Loading and aggregating trees")

    all_trees_dict, data_props, subsampled_alignments, trees_removed = (
        _load_and_balance_trees(
            pickle_files, median_trees, balance_by_median_num_MP_trees, logger
        )
    )

    # Log balancing summary
    if balance_by_median_num_MP_trees:
        _log_balancing_summary(
            logger, subsampled_alignments, trees_removed, len(all_trees_dict)
        )

    trees = list(all_trees_dict.keys())
    labels = list(all_trees_dict.values())
    return trees, labels, all_trees_dict, data_props


def pickle_and_save_data(
    dpvt_train_data, dpvt_test_data, all_trees_dict, trees, labels
):
    """
    Pickle and save the training and testing data.
    If test data is not provided, save all data as training data.
    Args:
        dpvt_train_data (str): Path to save the training data.
        dpvt_test_data (str): Path to save the testing data.
        all_trees_dict (dict): Dictionary containing all trees.
        trees (list): List of trees.
        labels (list): List of labels.
    """
    if dpvt_test_data is None:
        with open(dpvt_train_data, "wb") as f:
            pickle.dump(all_trees_dict, f)
    else:
        # Commented out section could be used to split train and testing data
        # For Stratifying
        sum_of_ones = [sum(label) for label in labels]
        counter = Counter(sum_of_ones)

        # Convert sums to a categorical variable for balancing number of non-MP edges in
        # train/test/val
        categories = pd.qcut(
            sum_of_ones, q=min(len(counter), 4), labels=False, duplicates="drop"
        )

        try:
            # Attempt to split the data with stratification
            train_trees, test_trees, train_labels, test_labels = train_test_split(
                trees, labels, train_size=0.8, stratify=categories
            )
        except ValueError as e:
            # If a ValueError occurs (e.g., due to insufficient data for
            # stratification), print a custom message
            print(f"Error during train-test split: {e}")
            print(
                "The dataset is not large enough to split in training and testing data. Increase number of trees extracted from hDAG or even better the number of alignments used."
            )
            sys.exit(
                "Dataset too small, will not generate training/testing data split."
            )

        train_dict = {i: j for (i, j) in zip(train_trees, train_labels)}
        test_dict = {i: j for (i, j) in zip(test_trees, test_labels)}

        with open(dpvt_train_data, "wb") as f:
            pickle.dump(train_dict, f)

        with open(dpvt_test_data, "wb") as f:
            pickle.dump(test_dict, f)


def save_data_properties(data_props, data_props_file, data_dir):
    """
    Save data properties to a CSV file.
    Args:
        data_props (dict): Dictionary containing properties of datasets.
        data_props_file (str): Path to save the data properties file.
        data_dir (str): Directory containing subdirs with .p pickle files.
    """
    for root, dirs, files in os.walk(data_dir):
        alignment_length_file = [f for f in files if "cleaned_alignment_length" in f]
        if alignment_length_file:
            alignment_length_file = alignment_length_file[0]
            subdir = os.path.relpath(root, data_dir)
            data_subdir = root
            if os.path.isdir(data_subdir):
                alignment_length_file = os.path.join(root, alignment_length_file)
                dataset_name = alignment_length_file.split("/")[-2]
                if "_no_dup_sites" in data_props_file:
                    dataset_name += "_no_dup_sites"
                # only take those datasets for which we actually have pickled
                # tree dictionaries
                with open(alignment_length_file, "r") as f:
                    data_props[dataset_name].append(int(f.read().split(",")[0].strip()))
    data_props_df = pd.DataFrame.from_dict(
        data_props,
        columns=[
            "num_trees",
            "num_leaves",
            "MP edges",
            "non MP edges",
            "alignment_length",
        ],
        orient="index",
    )
    data_props_df.to_csv(data_props_file)
    print(f"Data properties saved to '{data_props_file}'")


def aggregate_data(
    data_dir,
    data_props_file,
    dpvt_train_data,
    edge_distribution="constant",
    dpvt_test_data=None,
):
    """
    Aggregate data from the specified directory and save it to a pickle file.
    Args:
        data_dir (str): Directory containing subdirs with .p pickle files.
        data_props_file (str): Path to save the data properties file.
        dpvt_train_data (str): Path to save the training data.
        edge_distribution (str): Type of edge distribution ("constant", "uniform", "treesearch_mimic", "random_subtree")
        dpvt_test_data (str): Path to save the testing data.
    """
    trees, labels, all_trees_dict, data_props = extract_trees_and_labels(
        data_dir, edge_distribution
    )
    pickle_and_save_data(dpvt_train_data, dpvt_test_data, all_trees_dict, trees, labels)
    save_data_properties(data_props, data_props_file, data_dir)
