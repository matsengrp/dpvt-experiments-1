import pickle
from sklearn.model_selection import train_test_split
from dpvt.wrapper import TreeDataset, TraversalDataset
from pathlib import Path
import os
import pandas as pd
from collections import Counter
import json
import psutil
import torch
from torch.utils.data import Dataset


def print_memory_usage(stage=""):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    
    # GPU memory if available
    gpu_mem = ""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
        gpu_mem = f" | GPU: {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved"
    
    print(f"[{stage}] Memory usage: {memory_gb:.2f}GB RAM{gpu_mem}")


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


def train_val_data_from_preprocessed(data_name, device, data_nicknames_path):
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
    print_memory_usage("Function start")
    
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
    print_memory_usage("After loading preprocessed data")

    # Get stratification categories more efficiently using the loaded dataset's labels
    print("Computing stratification categories from loaded dataset...")
    n_samples = len(full_dataset.labels)
    sum_of_ones = []
    
    # Process labels in chunks to avoid memory issues
    chunk_size = 1000
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunk_labels = full_dataset.labels[i:end_idx]
        for j in range(len(chunk_labels)):
            sum_of_ones.append(float(chunk_labels[j].sum()))
    
    print_memory_usage("After processing labels for stratification")

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

    print_memory_usage("Before creating indices split")
    
    # Split indices instead of data to avoid loading memory-mapped arrays
    print("Creating train/val indices for memory-efficient splitting")
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        train_size=0.8,
        test_size=0.2,
        stratify=categories,
        random_state=42,
    )
    print_memory_usage("After creating indices split")

    # Create indexed datasets that keep references to the original memory-mapped arrays
    print("Creating indexed train dataset...")
    train_data = IndexedDataset(full_dataset, train_indices)
    print_memory_usage("After creating train dataset")

    print("Creating indexed val dataset...")
    val_data = IndexedDataset(full_dataset, val_indices)
    print_memory_usage("After creating val dataset")

    print(f"Train dataset: {len(train_data)} samples")
    print(f"Val dataset: {len(val_data)} samples")

    return train_data, val_data


class IndexedDataset(Dataset):
    """Dataset that provides indexed access to a base dataset without copying data."""
    
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices
        self.device = "cpu"
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        print_memory_usage(f"Getting item {idx}")
        # Map to the actual index in the base dataset
        real_idx = self.indices[idx]
        result = self.base_dataset[real_idx]
        print_memory_usage(f"Got item {idx}")
        return result
