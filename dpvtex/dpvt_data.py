import pickle
from sklearn.model_selection import train_test_split
from dpvt.wrapper import TreeDataset, TraversalDataset
from pathlib import Path
import os
import pandas as pd
from collections import Counter
import json


def load_nicknames_dict(data_nicknames_path):
    """Load dataset nicknames from a JSON configuration file.

    Args:
        data_nicknames_path: Path to the JSON file containing dataset nicknames.
            The JSON should have a "data_dir" key and nickname-to-filename mappings.

    Returns:
        dict: Dictionary mapping dataset nicknames to their full file paths.
    """
    with open(data_nicknames_path, "r") as f:
        dataset_dict = json.load(f)
    data_dir = dataset_dict.pop("data_dir")
    dataset_dict = {key: f"{data_dir}/{dataset_dict[key]}" for key in dataset_dict}
    return dataset_dict


def data_of_nicknames(
    data_name, device, data_nicknames_path, data_struct="TraversalDataset"
):
    """
    Takes a dataset nickname string, which is a key in `dataset_dict`, and returns the
    corresponding data as a `TreeDataset` object.
    """
    dataset_dict = load_nicknames_dict(data_nicknames_path)
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    if device == "cpu-tree-dataset" or data_struct == "TreeDataset":
        tree_data = TreeDataset(trees, labels)
    else:
        tree_data = TraversalDataset(trees, labels, device)
    return tree_data


def train_val_data_of_nicknames(data_name, device, data_nicknames_path):
    """Load a dataset by nickname and split into balanced training and validation sets.

    Performs stratified train/validation split (80/20) based on the distribution of
    non-MP (maximum parsimony) edges in each tree. Categories are dynamically adjusted
    to ensure each has at least 20% of the total trees.

    Args:
        data_name: Nickname of the dataset to load (key in the nicknames JSON).
        device: Device to use for the dataset ('cpu', 'cuda', or 'cpu-tree-dataset').
            If 'cpu-tree-dataset', returns TreeDataset; otherwise TraversalDataset.
        data_nicknames_path: Path to the JSON file containing dataset nicknames.

    Returns:
        tuple: (train_data, val_data) where each is either a TreeDataset or
            TraversalDataset depending on the device parameter.
    """
    dataset_dict = load_nicknames_dict(data_nicknames_path)
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    print(f"Reading pickle file: {file_path}", flush=True)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)
    print(f"Loaded {len(data_dict)} samples from pickle file", flush=True)

    # Split into balanced training, validation, and test data using sklearn
    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    sum_of_ones = [sum(label) for label in labels]

    # Convert sums to a categorical variable for balancing number of non-MP edges in train/test/val
    num_categories = 4  # Aim for 4 categories
    categories = pd.qcut(sum_of_ones, q=num_categories, labels=False, duplicates="drop")
    cat_counter = Counter(categories)

    while any(count < 0.2 * len(trees) for count in cat_counter.values()):
        # require minimum size of 20% of dataset for each category to ensure train/val split can be performed correctly
        print("decrease number of categories")
        num_categories -= 1
        categories = pd.qcut(
            sum_of_ones, q=num_categories, labels=False, duplicates="drop"
        )
        cat_counter = Counter(categories)

    train_data, val_data, train_labels, val_labels = train_test_split(
        trees,
        labels,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )
    print(
        f"Split into {len(train_data)} training and {len(val_data)} validation samples",
        flush=True,
    )

    if device == "cpu-tree-dataset":
        # no need to convert to traversal data structure, as this would add
        # one more traversal, hence increase runtime
        print("Creating TreeDataset for CPU", flush=True)
        train_data = TreeDataset(train_data, train_labels)
        val_data = TreeDataset(val_data, val_labels)
    else:
        print(f"Creating TraversalDataset for device: {device}", flush=True)
        print(
            f"Converting {len(train_data)} training trees to traversals (this may take a while)...",
            flush=True,
        )
        train_data = TraversalDataset(train_data, train_labels, device)
        print(
            f"Converting {len(val_data)} validation trees to traversals...", flush=True
        )
        val_data = TraversalDataset(val_data, val_labels, device)
        print("Data loading and conversion complete!", flush=True)

    return train_data, val_data
