#!/usr/bin/env python3
"""Add treesearch data files to the data nicknames JSON file using glob patterns."""
import json
import os
import sys


VALID_START_TREE_TYPES = ("random", "nj")
STARTING_DIR_SUFFIX = "_starting"


def main():
    """
    Add glob patterns for treesearch pickle files to data nicknames json file.

    Usage:
        python add_to_dataset_nicknames.py <nicknames_json> <start_tree_type> <file1.p> [file2.p ...]

    Args:
        nicknames_json: Path to the JSON file containing dataset nicknames.
        start_tree_type: Type of starting tree used ("random" or "nj").
        file1.p, file2.p, ...: Pickle files used to determine the glob patterns.

    Creates one glob pattern entry per alignment directory, so each alignment
    can be referenced separately in configs while replicates are auto-discovered.
    """
    if len(sys.argv) < 4:
        print(
            "Usage: python add_to_dataset_nicknames.py <nicknames_json> <start_tree_type> <file1.p> [file2.p ...]"
        )
        print(f"  start_tree_type must be one of: {VALID_START_TREE_TYPES}")
        sys.exit(1)

    json_file = sys.argv[1]
    start_tree_type = sys.argv[2]
    files = sys.argv[3:]

    if start_tree_type not in VALID_START_TREE_TYPES:
        print(f"Error: start_tree_type must be one of: {VALID_START_TREE_TYPES}")
        sys.exit(1)

    # Load existing JSON data or create new file with default structure
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
    else:
        print(f"Creating new nicknames file: {json_file}")
        data = {"data_dir": "../data"}

    data_dir = data.get("data_dir", ".")
    # Resolve data_dir relative to the JSON file location, not cwd
    json_dir = os.path.dirname(os.path.abspath(json_file))
    abs_data_dir = os.path.normpath(os.path.join(json_dir, data_dir))

    starting_dir = f"{start_tree_type}{STARTING_DIR_SUFFIX}"

    # Group files by alignment directory
    alignment_dirs = {}  # alignment_name -> relative path to alignment dir

    for filepath in files:
        abs_filepath = os.path.abspath(filepath)
        if abs_filepath.startswith(abs_data_dir):
            rel_path = os.path.relpath(abs_filepath, abs_data_dir)
        else:
            rel_path = filepath

        # Extract alignment directory (parent of the file)
        alignment_dir = os.path.dirname(rel_path)
        alignment_name = os.path.basename(alignment_dir)

        if alignment_name not in alignment_dirs:
            alignment_dirs[alignment_name] = alignment_dir

    if not alignment_dirs:
        print("Error: no valid alignment directories found")
        sys.exit(1)

    # Extract dataset name from path structure
    sample_path = next(iter(alignment_dirs.values()))
    path_parts = sample_path.replace("\\", "/").split("/")
    starting_dir_idx = next(
        (i for i, p in enumerate(path_parts) if p == starting_dir), None
    )
    if starting_dir_idx is not None and starting_dir_idx > 0:
        dataset_name = path_parts[starting_dir_idx - 1]
        base_nickname = f"{start_tree_type}_treesearch_{dataset_name}_"
    else:
        base_nickname = f"{start_tree_type}_"

    # Add one glob pattern per alignment
    for alignment_name, alignment_dir in sorted(alignment_dirs.items()):
        glob_pattern = f"{alignment_dir}/*_tree_search.p"
        nickname = f"{base_nickname}{alignment_name}"
        data[nickname] = glob_pattern
        print(f"  Added: {nickname} -> {glob_pattern}")

    # Write updated JSON back to file
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(
        f"Updated {json_file} with {len(alignment_dirs)} glob patterns for {start_tree_type} treesearch files"
    )


if __name__ == "__main__":
    main()
