#!/usr/bin/env python3
"""Add treesearch data files to the data nicknames JSON file."""
import json
import os
import sys


def main():
    """
    Add specific treesearch pickle files to data nicknames json file.

    Usage:
        python add_to_dataset_nicknames.py <nicknames_json> <file1.p> [file2.p ...]

    The files should be paths relative to the data_dir specified in the nicknames JSON.
    """
    if len(sys.argv) < 3:
        print(
            "Usage: python add_to_dataset_nicknames.py <nicknames_json> <file1.p> [file2.p ...]"
        )
        sys.exit(1)

    json_file = sys.argv[1]
    files_to_add = sys.argv[2:]

    # Load existing JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    data_dir = data.get("data_dir", ".")

    # Add each file with its nickname
    for filepath in files_to_add:
        # Get the path relative to data_dir
        abs_filepath = os.path.abspath(filepath)
        abs_data_dir = os.path.abspath(data_dir)

        if abs_filepath.startswith(abs_data_dir):
            rel_path = os.path.relpath(abs_filepath, abs_data_dir)
        else:
            # If not under data_dir, use the filename with treesearch/ prefix
            rel_path = os.path.join("treesearch", os.path.basename(filepath))

        # Create nickname from filename (without .p extension)
        nickname = os.path.basename(filepath).replace(".p", "")

        data[nickname] = rel_path
        print(f"  Added: {nickname} -> {rel_path}")

    # Write updated JSON back to file
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {json_file} with {len(files_to_add)} treesearch files")


if __name__ == "__main__":
    main()
