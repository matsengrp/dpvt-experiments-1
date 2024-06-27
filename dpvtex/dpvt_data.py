import pickle
from sklearn.model_selection import train_test_split
from dpvt.wrapper import TreeDataset, TraversalDataset
import numpy as np
from pathlib import Path
import os


# Get the absolute path to the directory where the current script is located
script_directory = Path(__file__).resolve().parent

dataset_dict = {
    "FourLeafFourSite": script_directory.parent / "data/4leaf4site.p",
    "FourLeaf": script_directory.parent / "data/4leaf.p",
    "FourLeafFourSiteTest": script_directory.parent / "data/4leaf4site_test.p",
    "FourLeafTest": script_directory.parent / "data/4leaf_test.p",
    "TenLeaf": script_directory.parent / "data/10leaf_perfect.p",
    "TenLeafTest": script_directory.parent / "data/10leaf_test.p",
    "ThirtyLeaf": script_directory.parent / "data/30leaf_perfect.p",
    "ThirtyLeafDistinct": script_directory.parent / "data/30leaf_perfect_distinct_trees.p",
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

    tree_data = TreeDataset(trees, labels, device)
    return tree_data



def train_val_data_of_nicknames(data_name, device):
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    # Split into balanced training, validation, and test data using sklearn
    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    # use number of bad edges to stratify dataset
    n_bad_edges = np.array([sum(label) for label in labels])

    train_data, val_data, train_labels, val_labels = (
        train_test_split(
            trees,
            labels,
            train_size=0.8,
            test_size=0.2,
            stratify=n_bad_edges,
            random_state=42,
        )
    )

    # train_data = TreeDataset(train_data, train_labels)
    # test_data = TreeDataset(test_data, test_labels)
    # val_data = TreeDataset(val_data, val_labels)
    train_data = TraversalDataset(train_data, train_labels, device)
    val_data = TraversalDataset(val_data, val_labels, device)

    return train_data, val_data
