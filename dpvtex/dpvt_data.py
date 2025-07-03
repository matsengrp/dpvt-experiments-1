import pickle
from sklearn.model_selection import train_test_split
from dpvt.wrapper import TreeDataset, TraversalDataset
from pathlib import Path
import os
import pandas as pd
from collections import Counter
import json


def load_nicknames_dict(data_nicknames_path):
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
    dataset_dict = load_nicknames_dict(data_nicknames_path)
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    # Split into balanced training, validation, and test data using sklearn
    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    sum_of_ones = [sum(label) for label in labels]

    # Convert sums to a categorical variable for balancing number of non-MP edges in train/test/val
    num_categories = 4  # Aim for 4 categories
    categories = pd.qcut(sum_of_ones, q=num_categories, labels=False, duplicates="drop")
    cat_counter = Counter(categories)

    while any(count < 0.2 * len(labels) for count in cat_counter.values()):
        print("decrease number of categories")
        num_categories -= 1
        categories = pd.qcut(
            sum_of_ones, q=num_categories, labels=False, duplicates="drop"
        )
        cat_counter = Counter(categories)

    train_trees, val_trees, train_labels, val_labels = train_test_split(
        trees,
        labels,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )

    if device == "cpu-tree-dataset":
        train_data = TreeDataset(train_trees, train_labels)
        val_data = TreeDataset(val_trees, val_labels)
    else:
        train_data = TraversalDataset(train_trees, train_labels, device)
        val_data = TraversalDataset(val_trees, val_labels, device)

    return train_data, val_data


def get_traversal_data_path(data_name, data_nicknames_path):
    """
    Get the path where preprocessed TraversalDataset data should be stored.

    Args:
        data_name: Dataset nickname
        data_nicknames_path: Path to data nicknames JSON file

    Returns:
        Path: Full path to the preprocessed data file
    """
    # Load the data nicknames to get the data_dir
    with open(data_nicknames_path, "r") as f:
        dataset_dict_raw = json.load(f)
    data_dir = dataset_dict_raw["data_dir"]

    # Get the absolute data directory path
    config_dir = os.path.dirname(os.path.abspath(data_nicknames_path))
    data_dir_abs = os.path.abspath(os.path.join(config_dir, data_dir))

    # Create TraversalDataset subdirectory if it doesn't exist
    traversal_dir = os.path.join(data_dir_abs, "TraversalDataset")
    os.makedirs(traversal_dir, exist_ok=True)

    # Generate filename for preprocessed data
    filename = f"{data_name}_traversal.p"
    preprocessed_path = os.path.join(traversal_dir, filename)

    return preprocessed_path


def save_preprocessed_traversal_data(dataset, save_path):
    """
    Utility function to save preprocessed TraversalDataset data to disk.

    Args:
        dataset: TraversalDataset to save
        save_path: Path where to save the data
    """
    print(f"Saving preprocessed data to {save_path}")
    dataset.save_preprocessed_data(save_path)
    print("Preprocessed data saved successfully!")


def load_preprocessed_traversal_data(load_path, device="cpu"):
    """
    Utility function to load preprocessed TraversalDataset data from disk.

    Args:
        load_path: Path to the preprocessed data file
        device: Device to use

    Returns:
        TraversalDataset: The loaded dataset
    """
    print(f"Loading preprocessed data from {load_path}")
    dataset = TraversalDataset(preprocessed_path=load_path, device=device)
    print(f"Loaded dataset with {len(dataset)} samples")
    return dataset


def preprocess_and_save_traversal_data(data_name, device, data_nicknames_path):
    """
    Load raw data, convert to TraversalDataset format, and save to disk.
    This preprocessing only needs to be done once per dataset.

    Args:
        data_name: Dataset nickname
        device: Device to use (typically "cpu" for preprocessing)
        data_nicknames_path: Path to data nicknames JSON file
    """
    # Get the save path
    save_path = get_traversal_data_path(data_name, data_nicknames_path)

    if os.path.exists(save_path):
        print(
            f"Preprocessed data already exists at {save_path}. Skipping preprocessing."
        )
        return save_path

    # Load the raw data
    print(f"Loading raw data for {data_name}...")
    dataset_dict = load_nicknames_dict(data_nicknames_path)
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)

    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    # Create TraversalDataset
    print(f"Creating TraversalDataset with {len(trees)} trees...")
    dataset = TraversalDataset(
        trees, labels, device="cpu"
    )  # Always use CPU for preprocessing

    # Save to disk
    save_preprocessed_traversal_data(dataset, save_path)

    return save_path


def train_val_data_of_nicknames_semipreprocessed(
    data_name, device, data_nicknames_path
):
    """
    Load raw data and split into train/val, then convert to TraversalDataset format.
    This differs from the fully preprocessed version by doing the train/val split
    before the expensive tensor conversion.

    Args:
        data_name: Dataset nickname
        device: Device to use
        data_nicknames_path: Path to data nicknames JSON file

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    dataset_dict = load_nicknames_dict(data_nicknames_path)
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)

    print(f"Loading raw data from {file_path}...")
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    # Split into balanced training, validation, and test data using sklearn
    labels = list(data_dict.values())
    trees = list(data_dict.keys())

    sum_of_ones = [sum(label) for label in labels]

    # Convert sums to a categorical variable for balancing number of non-MP edges in train/test/val
    num_categories = 4  # Aim for 4 categories
    categories = pd.qcut(sum_of_ones, q=num_categories, labels=False, duplicates="drop")
    cat_counter = Counter(categories)

    while any(count < 0.2 * len(labels) for count in cat_counter.values()):
        print("decrease number of categories")
        num_categories -= 1
        categories = pd.qcut(
            sum_of_ones, q=num_categories, labels=False, duplicates="drop"
        )
        cat_counter = Counter(categories)

    print("Splitting data into train and validation sets...")
    train_trees, val_trees, train_labels, val_labels = train_test_split(
        trees,
        labels,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )

    if device == "cpu-tree-dataset":
        train_dataset = TreeDataset(train_trees, train_labels)
        val_dataset = TreeDataset(val_trees, val_labels)
    else:
        # Check if preprocessed versions exist
        # Load the data nicknames to get the data_dir
        with open(data_nicknames_path, "r") as f:
            dataset_dict_raw = json.load(f)
        data_dir = dataset_dict_raw["data_dir"]

        # Get the absolute data directory path
        config_dir = os.path.dirname(os.path.abspath(data_nicknames_path))
        data_dir_abs = os.path.abspath(os.path.join(config_dir, data_dir))

        # Create TraversalDataset subdirectory if it doesn't exist
        traversal_dir = os.path.join(data_dir_abs, "TraversalDataset")
        os.makedirs(traversal_dir, exist_ok=True)

        train_preprocessed_path = os.path.join(
            traversal_dir, f"{data_name}_train_traversal.p"
        )
        val_preprocessed_path = os.path.join(
            traversal_dir, f"{data_name}_val_traversal.p"
        )

        if os.path.exists(train_preprocessed_path) and os.path.exists(
            val_preprocessed_path
        ):
            print("Loading preprocessed train/val datasets...")
            train_dataset = TraversalDataset(
                preprocessed_path=train_preprocessed_path, device=device
            )
            val_dataset = TraversalDataset(
                preprocessed_path=val_preprocessed_path, device=device
            )
        else:
            print(
                f"Creating TraversalDataset for training set ({len(train_trees)} trees)..."
            )
            train_dataset = TraversalDataset(
                trees=train_trees, labels=train_labels, device=device
            )

            print(
                f"Creating TraversalDataset for validation set ({len(val_trees)} trees)..."
            )
            val_dataset = TraversalDataset(
                trees=val_trees, labels=val_labels, device=device
            )

            # Save preprocessed versions for future use
            print("Saving preprocessed train/val datasets for future use...")
            train_dataset.save_preprocessed_data(train_preprocessed_path)
            val_dataset.save_preprocessed_data(val_preprocessed_path)

        return train_dataset, val_dataset


def train_val_data_from_preprocessed(data_name, device, data_nicknames_path):
    """
    Memory-efficient version using the semipreprocessed approach.
    This avoids loading the full 13GB dataset by preprocessing train/val separately.
    """
    return train_val_data_of_nicknames_semipreprocessed(
        data_name, device, data_nicknames_path
    )


def train_val_data_from_preprocessed_old(data_name, device, data_nicknames_path):
    """
    Load preprocessed TraversalDataset data and split into train/val sets.
    This avoids reprocessing data that has already been preprocessed.

    Args:
        data_name: Dataset nickname
        device: Device to use
        data_nicknames_path: Path to data nicknames JSON file

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    if device == "cpu-tree-dataset":
        # For cpu-tree-dataset, fall back to the original function since preprocessing isn't used
        return train_val_data_of_nicknames(data_name, device, data_nicknames_path)

    # Load the preprocessed data
    preprocessed_path = get_traversal_data_path(data_name, data_nicknames_path)

    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {preprocessed_path}. "
            "Make sure to run preprocessing first."
        )

    print(f"Loading preprocessed data from {preprocessed_path}")
    full_dataset = load_preprocessed_traversal_data(preprocessed_path, device)

    # We need to get the stratification categories from the original data
    dataset_dict = load_nicknames_dict(data_nicknames_path)
    file_path = dataset_dict[data_name]
    file_path = os.path.realpath(file_path)
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)

    labels = list(data_dict.values())
    sum_of_ones = [sum(label) for label in labels]

    # Convert sums to a categorical variable for balancing number of non-MP edges in train/test/val
    num_categories = 4  # Aim for 4 categories
    categories = pd.qcut(sum_of_ones, q=num_categories, labels=False, duplicates="drop")
    cat_counter = Counter(categories)

    while any(count < 0.2 * len(labels) for count in cat_counter.values()):
        print("decrease number of categories")
        num_categories -= 1
        categories = pd.qcut(
            sum_of_ones, q=num_categories, labels=False, duplicates="drop"
        )
        cat_counter = Counter(categories)

    # Split the tensor components directly
    print("Splitting preprocessed data into train and validation sets")
    train_traversal, val_traversal = train_test_split(
        full_dataset.traversal,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )
    train_mutations, val_mutations = train_test_split(
        full_dataset.mutations,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )
    train_labels, val_labels = train_test_split(
        full_dataset.labels,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )
    train_mask, val_mask = train_test_split(
        full_dataset.mask,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )

    # Create new TraversalDatasets with the split tensors
    train_data = TraversalDataset(
        traversal=train_traversal,
        mutations=train_mutations,
        traversal_labels=train_labels,
        mask=train_mask,
    )

    val_data = TraversalDataset(
        traversal=val_traversal,
        mutations=val_mutations,
        traversal_labels=val_labels,
        mask=val_mask,
    )

    return train_data, val_data
