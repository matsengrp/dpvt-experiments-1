#!/usr/bin/env python3
"""Add treesearch data files to the data nicknames JSON file."""
import json
import os
import sys


VALID_START_TREE_TYPES = ("random", "nj")
STARTING_DIR_SUFFIX = "_starting"


def main():
    """
    Add specific treesearch pickle files to data nicknames json file.

    Usage:
        python add_to_dataset_nicknames.py <nicknames_json> <start_tree_type> <file1.p> [file2.p ...]

    Args:
        nicknames_json: Path to the JSON file containing dataset nicknames.
        start_tree_type: Type of starting tree used ("random" or "nj").
        file1.p, file2.p, ...: Pickle files to add to the nicknames.

    The files should be paths relative to the data_dir specified in the nicknames JSON.

    For files in the treesearch directory structure with starting tree type
    subdirectories (random_starting/ or nj_starting/), the nickname will include
    the starting tree type prefix (e.g., 'random_PF05036-dna_rep1_tree_search').
    """
    if len(sys.argv) < 4:
        print(
            "Usage: python add_to_dataset_nicknames.py <nicknames_json> <start_tree_type> <file1.p> [file2.p ...]"
        )
        print(f"  start_tree_type must be one of: {VALID_START_TREE_TYPES}")
        sys.exit(1)

    json_file = sys.argv[1]
    start_tree_type = sys.argv[2]
    files_to_add = sys.argv[3:]

    # Load existing JSON data or create new file with default structure
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        print(f"Creating new nicknames file: {json_file}")
        data = {"data_dir": "../data"}

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
            filename = os.path.basename(filepath)
            rel_path = os.path.join(
                "treesearch",
                start_tree_type + STARTING_DIR_SUFFIX,
                filename.split("_rep")[0],
                filename,
            )

        # Create nickname from filename (without .p extension)
        base_nickname = os.path.basename(filepath).replace(".p", "")

        # Add starting tree type prefix if present in path
        for tree_type in VALID_START_TREE_TYPES:
            if f"{tree_type}{STARTING_DIR_SUFFIX}" in filepath:
                nickname = f"{tree_type}_{base_nickname}"
                break
        else:
            nickname = base_nickname

        data[nickname] = rel_path
        print(f"  Added: {nickname} -> {rel_path}")

    # Write updated JSON back to file
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {json_file} with {len(files_to_add)} treesearch files")


if __name__ == "__main__":
    main()
