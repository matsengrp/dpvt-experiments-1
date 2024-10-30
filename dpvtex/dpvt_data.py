import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    Dataset,
)
from pathlib import Path
import os
import pandas as pd
from collections import Counter
import json


with open("data_nicknames.json", "r") as f:
    dataset_dict = json.load(f)

data_dir = dataset_dict.pop("data_dir")

dataset_dict = {key: data_dir + "/" + dataset_dict[key] for key in dataset_dict}

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
    print(data_name)
    print(dataset_dict)
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

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

    sum_of_ones = [sum(label) for label in labels]
    counter = Counter(sum_of_ones)

    # Convert sums to a categorical variable for balancing number of non-MP edges in train/test/val
    categories = pd.qcut(
        sum_of_ones, q=min(len(counter), 4), labels=False, duplicates="drop"
    )

    train_data, val_data, train_labels, val_labels, train_mask, val_mask = (
        train_test_split(
            trees,
            labels,
            masks,
            train_size=0.8,
            test_size=0.2,
            stratify=categories,
        )
    )

    train_data = TreeDataset(train_data, train_labels, train_mask)
    val_data = TreeDataset(val_data, val_labels, val_mask)
    return train_data, val_data
