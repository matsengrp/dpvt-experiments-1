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


def data_of_nicknames(data_name, device, data_nicknames_path, data_struct="TraversalDataset"):
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
    num_categories = 4 # Aim for 4 categories
    categories = pd.qcut(
        sum_of_ones, q=num_categories, labels=False, duplicates="drop"
    )
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

    if device == "cpu-tree-dataset":
        # no need to convert to traversal data structure, as this would add
        # one more traversal, hence increase runtime
        train_data = TreeDataset(train_data, train_labels)
        val_data = TreeDataset(val_data, val_labels)
    else:
        train_data = TraversalDataset(train_data, train_labels, device)
        val_data = TraversalDataset(val_data, val_labels, device)

    return train_data, val_data


def preprocess_and_save_traversal_data(trees, labels, save_path, device="cpu"):
    """
    Utility function to preprocess trees and labels into TraversalDataset format
    and save to disk for later loading.

    Args:
        trees: List of ete3.Tree objects
        labels: List of labels corresponding to the trees
        save_path: Path where to save the preprocessed data
        device: Device to use for processing

    Returns:
        TraversalDataset: The created dataset (also saved to disk)
    """
    print(f"Preprocessing {len(trees)} trees and saving to {save_path}")

    # Create the dataset (this will do the heavy preprocessing)
    dataset = TraversalDataset(trees=trees, labels=labels, device=device)

    # Save the preprocessed data
    dataset.save_preprocessed_data(save_path)

    print(f"Preprocessing complete. Data saved to {save_path}")
    return dataset


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


def get_traversal_data_path(data_name, data_nicknames_path):
    """
    Get the path for preprocessed TraversalDataset data based on the dataset nickname.
    
    Args:
        data_name: Dataset nickname
        data_nicknames_path: Path to the data nicknames JSON file
        
    Returns:
        str: Path where the preprocessed data should be stored
    """
    dataset_dict = load_nicknames_dict(data_nicknames_path)
    data_dir = Path(dataset_dict[data_name]).parent
    traversal_dir = data_dir / "TraversalDataset"
    traversal_dir.mkdir(exist_ok=True)
    
    # Create filename based on original data filename
    original_filename = Path(dataset_dict[data_name]).stem
    traversal_path = traversal_dir / f"{original_filename}_traversal.p"
    
    return str(traversal_path)


def data_of_nicknames_with_preprocessing(data_name, device, data_nicknames_path, data_struct="TraversalDataset", force_reprocess=False):
    """
    Enhanced version of data_of_nicknames that can use preprocessed TraversalDataset data
    for faster loading, or preprocess and save data if not already available.
    
    Args:
        data_name: Dataset nickname
        device: Device to use
        data_nicknames_path: Path to data nicknames JSON file
        data_struct: Type of dataset to return ("TraversalDataset" or "TreeDataset")
        force_reprocess: If True, reprocess even if preprocessed data exists
        
    Returns:
        Dataset object (TraversalDataset or TreeDataset)
    """
    if data_struct == "TraversalDataset" and device != "cpu-tree-dataset":
        # Check if preprocessed data exists
        traversal_path = get_traversal_data_path(data_name, data_nicknames_path)
        
        if os.path.exists(traversal_path) and not force_reprocess:
            # Load preprocessed data
            return load_preprocessed_traversal_data(traversal_path, device)
        else:
            # Load original data and preprocess
            dataset_dict = load_nicknames_dict(data_nicknames_path)
            file_path = dataset_dict[data_name]
            file_path = os.path.realpath(file_path)
            with open(file_path, "rb") as f:
                data_dict = pickle.load(f)

            labels = list(data_dict.values())
            trees = list(data_dict.keys())
            
            # Preprocess and save
            return preprocess_and_save_traversal_data(trees, labels, traversal_path, device)
    else:
        # Use original function for TreeDataset or cpu-tree-dataset
        return data_of_nicknames(data_name, device, data_nicknames_path, data_struct)


def train_val_data_of_nicknames_with_preprocessing(data_name, device, data_nicknames_path, force_reprocess=False):
    """
    Enhanced version of train_val_data_of_nicknames that uses preprocessed TraversalDataset data
    for faster loading, then splits into train/val in memory.
    
    Args:
        data_name: Dataset nickname
        device: Device to use
        data_nicknames_path: Path to data nicknames JSON file
        force_reprocess: If True, reprocess even if preprocessed data exists
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    if device != "cpu-tree-dataset":
        # Load the full preprocessed dataset
        full_dataset = data_of_nicknames_with_preprocessing(
            data_name, device, data_nicknames_path, "TraversalDataset", force_reprocess
        )
        
        # Extract trees and labels from the dataset
        trees = full_dataset.trees
        labels = full_dataset.labels
        
        # Perform the same splitting logic as in train_val_data_of_nicknames
        sum_of_ones = [sum(label) for label in labels]

        # Convert sums to a categorical variable for balancing number of non-MP edges in train/test/val
        num_categories = 4  # Aim for 4 categories
        categories = pd.qcut(
            sum_of_ones, q=num_categories, labels=False, duplicates="drop"
        )
        cat_counter = Counter(categories)

        while any(count < 0.2 * len(trees) for count in cat_counter.values()):
            # require minimum size of 20% of dataset for each category to ensure train/val split can be performed correctly
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
        
        # Create separate datasets for train and val
        train_dataset = TraversalDataset(trees=train_trees, labels=train_labels, device=device)
        val_dataset = TraversalDataset(trees=val_trees, labels=val_labels, device=device)
        
        return train_dataset, val_dataset
    else:
        # Use original function for cpu-tree-dataset
        return train_val_data_of_nicknames(data_name, device, data_nicknames_path)


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
    if device == "cpu-tree-dataset":
        # For cpu-tree-dataset, fall back to the original function since preprocessing isn't used
        return train_val_data_of_nicknames(data_name, device, data_nicknames_path)
    
    # Load the preprocessed data
    preprocessed_path = get_traversal_data_path(data_name, data_nicknames_path)
    
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"Preprocessed data not found at {preprocessed_path}. "
                               "Make sure to run preprocessing first.")
    
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
    categories = pd.qcut(
        sum_of_ones, q=num_categories, labels=False, duplicates="drop"
    )
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
        full_dataset.traversal, train_size=0.8, test_size=0.2, 
        stratify=categories, random_state=42
    )
    train_mutations, val_mutations = train_test_split(
        full_dataset.mutations, train_size=0.8, test_size=0.2, 
        stratify=categories, random_state=42
    )
    train_labels, val_labels = train_test_split(
        full_dataset.labels, train_size=0.8, test_size=0.2, 
        stratify=categories, random_state=42
    )
    train_mask, val_mask = train_test_split(
        full_dataset.mask, train_size=0.8, test_size=0.2, 
        stratify=categories, random_state=42
    )
    
    # Create new TraversalDatasets with the split tensors
    train_data = TraversalDataset(traversal=train_traversal,
                                  mutations=train_mutations,
                                  traversal_labels=train_labels,
                                  mask=train_mask)
    
    val_data = TraversalDataset(traversal=val_traversal,
                                  mutations=val_mutations,
                                  traversal_labels=val_labels,
                                  mask=val_mask)

    return train_data, val_data





