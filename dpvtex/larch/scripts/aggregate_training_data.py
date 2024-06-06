import os
import pickle
import pandas as pd
import sys
from ete3 import Tree


data_dir = sys.argv[1]  # directory containing subdirs with .p pickle files
dpvt_data = sys.argv[2]  # output pickle file that will contain aggregated data
data_props_file = sys.argv[
    3
]  # output csv that contains for each dataset the number of MP trees extracted,
# the number of leaves in those trees, and the length of the corresponding alignment
larch_data_dir = sys.argv[4]


# Function to read a .p file and return the length of the dictionary it contains
def get_dict(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        if isinstance(data, dict):
            return data
        else:
            return None


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
            print(dataset_name)
            this_alignment_dict = get_dict(file_path)
            if this_alignment_dict is not None:
                data_props[dataset_name] = [len(this_alignment_dict)]  # num_trees
                data_props[dataset_name].append(
                    len(list(this_alignment_dict.keys())[0])
                )  # num_leaves
                data_props[dataset_name].append(
                    sum(lst.count(0) - (len(lst))/2 for lst in this_alignment_dict.values())
                ) # num MP edges (excluding pendant edges and root edge)
                data_props[dataset_name].append(
                    sum(lst.count(1) for lst in this_alignment_dict.values())
                ) # num non-MP edges

                all_trees_dict.update(this_alignment_dict)


# Write data to file
with open(dpvt_data, "wb") as f:
    pickle.dump(all_trees_dict, f)


for root, dirs, files in os.walk(larch_data_dir):
    if "cleaned_alignment_length.txt" in files:
        subdir = os.path.relpath(root, larch_data_dir)
        data_subdir = os.path.join(data_dir, subdir)
        if os.path.isdir(data_subdir):
            alignment_length_file = os.path.join(root, "cleaned_alignment_length.txt")
            dataset_name = alignment_length_file.split("/")[-2]
            print(dataset_name)
            with open(alignment_length_file, "r") as f:
                data_props[dataset_name].append(int(f.read().strip()))

data_props_df = pd.DataFrame.from_dict(
    data_props,
    columns=["num_trees", "num_leaves", "MP edges", "non MP edges", "alignment_length"],
    orient="index",
)
data_props_df.to_csv(data_props_file)


print(f"Data properties saved to '{data_props_file}'")
