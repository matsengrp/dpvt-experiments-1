import pickle
from sklearn.model_selection import train_test_split
from dpvt.wrapper import TreeDataset, TraversalDataset
import numpy as np
from pathlib import Path
import os
import pandas as pd
from collections import Counter
import json


with open("data_nicknames.json", "r") as f:
    dataset_dict = json.load(f)

data_dir = dataset_dict.pop("data_dir")

dataset_dict = {key: data_dir + "/" + dataset_dict[key] for key in dataset_dict}


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

    sum_of_ones = [sum(label) for label in labels]
    counter = Counter(sum_of_ones)

    # Convert sums to a categorical variable for balancing number of non-MP edges in train/test/val
    categories = pd.qcut(
        sum_of_ones, q=min(len(counter), 4), labels=False, duplicates="drop"
    )

    train_data, val_data, train_labels, val_labels = (
        train_test_split(
            trees,
            labels,
            train_size=0.8,
            test_size=0.2,
            stratify=categories,
        )
    )

    # train_data = TreeDataset(train_data, train_labels)
    # test_data = TreeDataset(test_data, test_labels)
    # val_data = TreeDataset(val_data, val_labels)
    train_data = TraversalDataset(train_data, train_labels, device)
    val_data = TraversalDataset(val_data, val_labels, device)

    return train_data, val_data
