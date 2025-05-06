import os
import pickle
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse


# Function to read a .p file and return the length of the dictionary it contains
def get_dict(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        if isinstance(data, dict):
            return data
        else:
            return None


def extract_trees_and_labels(data_dir):
    """
    Extracts trees and labels from .p files in the given directory.
    Args:
        data_dir (str): Directory containing .p files.
    Returns:
        tuple: A tuple containing:
            - trees (list): List of trees.
            - labels (list): List of labels.
            - all_trees_dict (dict): Dictionary containing all trees.
            - data_props (dict): Dictionary containing properties of datasets.
    """
    all_trees_dict = {}  # store all trees in one dict
    data_props = {}  # save properties of datasets in a dict

    # Traverse all subdirectories and process .p files
    for root, dirs, files in os.walk(data_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if (
                file_name.endswith(".p")
                and os.path.relpath(file_path, data_dir) != file_name
            ):
                file_path = os.path.join(root, file_name)
                dataset_name = file_name[:-2]
                dataset_name = dataset_name.split("_spr")[0] # remove suffix if it exists
                this_alignment_dict = get_dict(file_path)
                if this_alignment_dict is not None:
                    data_props[dataset_name] = [len(this_alignment_dict)]  # num_trees
                    data_props[dataset_name].append(
                        len(list(this_alignment_dict.keys())[0])
                    )  # num_leaves
                    data_props[dataset_name].append(
                        sum(
                            lst.count(0) - (len(lst)) / 2
                            for lst in this_alignment_dict.values()
                        )
                    )  # num MP edges (excluding pendant edges and root edge)
                    data_props[dataset_name].append(
                        sum(lst.count(1) for lst in this_alignment_dict.values())
                    )  # num non-MP edges

                    all_trees_dict.update(this_alignment_dict)

    # split data into training/validation and testing set. We split 80/20
    trees = list(all_trees_dict.keys())
    labels = list(all_trees_dict.values())
    return trees, labels, all_trees_dict, data_props


def pickle_and_save_data(dpvt_train_data, dpvt_test_data, all_trees_dict, trees, labels):
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
    if dpvt_test_data == None:
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
            print("The dataset is not large enough to split in training and testing data. Increase number of trees extracted from hDAG or even better the number of alignments used.")
            # Optionally, re-raise the error or handle it further
            sys.exit("Dataset too small, will not generate training/testing data split.")
            raise

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
        if "cleaned_alignment_length.txt" in files:
            subdir = os.path.relpath(root, data_dir)
            data_subdir = root
            if os.path.isdir(data_subdir):
                alignment_length_file = os.path.join(root, "cleaned_alignment_length.txt")
                dataset_name = alignment_length_file.split("/")[-2]
                if dataset_name in data_props:
                    # only take those datasets for which we actually have pickled
                    # tree dictionaries
                    with open(alignment_length_file, "r") as f:
                        data_props[dataset_name].append(int(f.read().split(",")[0].strip()))

    data_props_df = pd.DataFrame.from_dict(
        data_props,
        columns=["num_trees", "num_leaves", "MP edges", "non MP edges", "alignment_length"],
        orient="index",
    )
    data_props_df.to_csv(data_props_file)
    print(f"Data properties saved to '{data_props_file}'")


def aggregate_data(data_dir, data_props_file, dpvt_train_data, dpvt_test_data = None):
    args = parser.parse_args()

    data_dir = args.data_dir
    dpvt_train_data = args.output_train_data
    dpvt_test_data = args.output_test_data
    data_props_file = args.data_props_file

    trees, labels, all_trees_dict, data_props = extract_trees_and_labels(data_dir)
    pickle_and_save_data(dpvt_train_data, dpvt_test_data, all_trees_dict, trees, labels)
    save_data_properties(data_props, data_props_file, data_dir)