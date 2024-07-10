import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    Dataset,
)
import numpy as np
from pathlib import Path
import os

# Get the absolute path to the directory where the current script is located
script_directory = Path(__file__).resolve().parent

dataset_dict = {
    "FourLeafFourSite": script_directory.parent / "data/4leaf4site.p",
    "FourLeaf": script_directory.parent / "data/4leaf.p",
    "TenLeaf": script_directory.parent / "data/10leaf_perfect.p",
    "TenLeafTest": script_directory.parent / "data/10leaf_test.p",
    "ThirtyLeaf": script_directory.parent / "data/30leaf_perfect.p",
}


class TreeDataset(Dataset):
    def __init__(self, data, labels, mask):
        self.data = data
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.mask[idx]

    def __str__(self):
        n_leaves = 1 + len(self.data[0])  # number of leaves in `self.data[0]`
        n_unmasked = sum(self.mask[0])
        return (
            f"TreeDataset\n"
            f"Number of samples: {len(self.data)}\n"
            f"Leaves per tree: {n_leaves}\n"
            f"Unmasked edges per tree: {n_unmasked}"
        )


def data_of_nicknames(data_name):
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

    masks = []
    for tree in trees:
        # mask leaves, root (which is leaf) and root (which contains data for edge
        # leading to root leaf)
        mask_list = [
            not (node.is_leaf() or node.is_root() or node.up.is_root())
            for node in tree.traverse("preorder")
        ]
        masks.append(mask_list)

    tree_data = TreeDataset(trees, labels, masks)
    return tree_data


def train_val_data_of_nicknames(data_name):
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    # split into 60% training, 20% validation, 20% testing
    train_size = int(0.6 * len(data_dict))
    val_size = int(0.2 * len(data_dict))
    test_size = len(data_dict) - train_size - val_size

    # Split into balanced training, validation, and test data using sklearn
    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    masks = []
    for tree in trees:
        # mask leaves, root (which is leaf) and root (which contains data for edge
        # leading to root leaf)
        mask_list = [
            not (node.is_leaf() or node.is_root() or node.up.is_root())
            for node in tree.traverse("preorder")
        ]
        masks.append(mask_list)

    # use number of bad edges to stratify dataset
    n_bad_edges = np.array([sum(label) for label in labels])

    (
        train_val_data,
        test_data,
        train_val_labels,
        test_labels,
        train_val_mask,
        test_mask,
        sum_train_val,
        _,
    ) = train_test_split(
        trees,
        labels,
        masks,
        n_bad_edges,
        test_size=test_size,
        stratify=n_bad_edges,
        random_state=42,
    )

    train_data, val_data, train_labels, val_labels, train_mask, val_mask = (
        train_test_split(
            train_val_data,
            train_val_labels,
            train_val_mask,
            test_size=val_size / (train_size + val_size),
            stratify=sum_train_val,
            random_state=42,
        )
    )

    train_data = TreeDataset(train_data, train_labels, train_mask)
    val_data = TreeDataset(val_data, val_labels, val_mask)
    test_data = TreeDataset(test_data, test_labels, test_mask)
    return train_data, val_data, test_data
