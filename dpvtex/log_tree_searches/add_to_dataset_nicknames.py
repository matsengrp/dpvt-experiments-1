#!/usr/bin/env python3
import os
import json
import sys


def main():
    """
    Add files with .p extension in provided directory to data nicknames json
    file for dpvt-experiments-1.
    """
    if len(sys.argv) < 2:
        print(
            "Error: Please provide (1) path to directory containing pickled datasets, (2) path to data nicknames json file"
        )
        sys.exit(1)
    else:
        data_dir = sys.argv[1]
        json_file = sys.argv[2]


    # Load existing JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Find all .p files and add them to the dictionary
    for filename in os.listdir(data_dir):
        if filename.endswith('.p'):
            # Get basename without extension
            if data_dir.split("/")[-2] == "data":
                basename = os.path.splitext(filename)[0]
            else:
                basename =  data_dir.split("/")[-2] + "_" + os.path.splitext(filename)[0]
            # we assume that all data is saved in the data_dir.
            # with [1:] we remove the leading "/"
            rel_data_dir = data_dir.split("data")[1][1:]
            dataset_name = data_dir.split("/")[-1]
            
            # Use basename as the key, and the relative path as the value
            relative_path = os.path.join(rel_data_dir, filename)
            
            # Add to the data dictionary
            print(basename, relative_path)
            data[basename] = relative_path

    # Write updated JSON back to file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated {json_file} with .p files from {data_dir}")


if __name__ == "__main__":
    main()
