import os
import pickle
import pandas as pd
import sys


data_dir = sys.argv[1] # directory containing subdirs with .p pickle files
dpvt_data = sys.argv[2] # output pickle file that will contain aggregated data
length_file = sys.argv[3] # output csv that contains for each dataset the number of MP trees extracted

# Function to read a .p file and return the length of the dictionary it contains
def get_dict_length(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        if isinstance(data, dict):
            return data
        else:
            return None  # Not a dictionary


# Dictionary to store the lengths
length_dict = {}
all_trees_dict = {} # store all trees in one dict
# total number of trees
num_trees = 0

# Traverse all subdirectories and process .p files
for root, dirs, files in os.walk(data_dir):
    for file_name in files:
        if file_name.endswith('.p'):
            print(f"Count number of trees in {file_name}")
            file_path = os.path.join(root, file_name)
            this_dict = get_dict_length(file_path)
            num_trees += len(this_dict)
            if this_dict is not None:
                length_dict[file_path] = len(this_dict)
                all_trees_dict.update(this_dict)

length_df = pd.DataFrame.from_dict(length_dict, orient = 'index')
length_df.to_csv(length_file)

# Write data to file
with open(dpvt_data, "wb") as f:
    pickle.dump(all_trees_dict, f)

print(f"Dictionary lengths saved to '{length_file}'")
print(f"Total number of trees produced: {num_trees}")
