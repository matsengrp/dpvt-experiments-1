import pickle
from sklearn.model_selection import train_test_split
from dpvt.wrapper import TreeDataset, TraversalDataset
from pathlib import Path
import os
import pandas as pd
from collections import Counter

# Get the absolute path to the directory where the current script is located
script_directory = Path(__file__).resolve().parent

dataset_dict = {
    "FourLeafFourSite": script_directory.parent / "data/4leaf4site.p",
    "FourLeaf": script_directory.parent / "data/4leaf.p",
    "FourLeafFourSiteTest": script_directory.parent / "data/4leaf4site_test.p",
    "FourLeafTest": script_directory.parent / "data/4leaf_test.p",
    "FourLeafFourSiteTrain": script_directory.parent / "data/4leaf4site_train.p",
    "FourLeafTest": script_directory.parent / "data/4leaf_test.p",
    "TenLeafTest": script_directory.parent
    / "data/10leaf_perfect_distinct_trees_test.p",
    "TenLeafTrain": script_directory.parent
    / "data/10leaf_perfect_distinct_trees_train.p",
    "TenLeafTrainSmall": script_directory.parent
    / "data/10leaf_perfect_distinct_trees_train_small.p",
    "ThirtyLeaf": script_directory.parent / "data/30leaf_perfect.p",
    "ThirtyLeafTest": script_directory.parent
    / "data/30leaf_perfect_distinct_trees_test.p",
    "ThirtyLeafTrain": script_directory.parent
    / "data/30leaf_perfect_distinct_trees_train.p",
    "ThirtyLeafDistinct": script_directory.parent
    / "data/30leaf_perfect_distinct_trees.p",
    "HarringtonSmallTest": script_directory.parent
    / "data/harrington-small_2024-06-10_test.p",
    "HarringtonSmallTrain": script_directory.parent
    / "data/harrington-small_2024-06-10_train.p",
    "HarringtonTinyTest": script_directory.parent / "data/harrington_tiny_test.p",
    "HarringtonBelow50": script_directory.parent / "data/harrington-small_0_to_50_taxa_subset.p",
    "HarringtonShortSequences": script_directory.parent / "data/harrington-small_0_to_30_sites_subset.p",
    "Alisim10leaf_10sites_500alignments_train": script_directory.parent / "data/larch_alisim_10_seq_10_site_500_alignments_2024-10-01_train.p",
    "Alisim10leaf_10sites_500alignments_test": script_directory.parent / "data/larch_alisim_10_seq_10_site_500_alignments_2024-10-01_test.p",
    "Alisim20leaf_10sites_500alignments_train": script_directory.parent / "data/larch_alisim_20_seq_10_site_500_alignments_2024-10-30_train.p",
    "Alisim20leaf_10sites_500alignments_test": script_directory.parent / "data/larch_alisim_20_seq_10_site_500_alignments_2024-10-30_test.p",
    "Alisim10leaf_20sites_500alignments_train": script_directory.parent / "data/larch_alisim_10_seq_20_site_500_alignments_2024-10-09_train.p",
    "Alisim10leaf_20sites_500alignments_test": script_directory.parent / "data/larch_alisim_10_seq_20_site_500_alignments_2024-10-09_test.p",
    "Alisim10leaf_50sites_500alignments_train": script_directory.parent / "data/larch_alisim_10_seq_50_site_500_alignments_2024-10-09_train.p",
    "Alisim10leaf_50sites_500alignments_test": script_directory.parent / "data/larch_alisim_10_seq_50_site_500_alignments_2024-10-09_test.p",
    "Alisim20leaf_100sites_500alignments_train": script_directory.parent / "data/larch_alisim_20_seq_100_site_500_alignments_2024-10-28_train.p",
    "Alisim20leaf_100sites_500alignments_test": script_directory.parent / "data/larch_alisim_20_seq_100_site_500_alignments_2024-10-28_test.p",
}

def data_of_nicknames(data_name, device):
    """
    Takes a dataset nickname string, which is a key in `dataset_dict`, and returns the
    corresponding data as a `TreeDataset` object.
    """
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    if device == "cpu-tree-dataset":
        tree_data = TreeDataset(trees, labels)
    else:
        tree_data = TraversalDataset(trees, labels, device)
    return tree_data


def train_val_data_of_nicknames(data_name, device):
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    # Split into balanced training, validation, and test data using sklearn
    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    sum_of_ones = [sum(label) for label in labels]
    counter = Counter(sum_of_ones)

    # Convert sums to a categorical variable for balancing number of non-MP edges in train/test/val
    categories = pd.qcut(
        sum_of_ones, q=min(len(counter), 4), labels=False, duplicates="drop"
    )

    train_data, val_data, train_labels, val_labels = train_test_split(
        trees,
        labels,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )

    if device == "cpu-tree-dataset":
        # no need to convert to traversal data structure, as this would add
        # one more traversal, hence increase runtime
        train_data = TreeDataset(train_data, train_labels)
        val_data = TreeDataset(val_data, val_labels)
    else:
        train_data = TraversalDataset(train_data, train_labels, device)
        val_data = TraversalDataset(val_data, val_labels, device)

    return train_data, val_data
