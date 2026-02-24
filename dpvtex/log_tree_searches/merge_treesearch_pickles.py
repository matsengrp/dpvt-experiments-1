"""Merge per-alignment treesearch pickles into a single training dataset.

Each per-alignment pickle contains a {tree: labels} dict. This script
merges all such dicts into one and saves the result.
"""

import argparse
import pickle
from glob import glob
import os


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-alignment treesearch pickles into one training pickle."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing per-alignment treesearch pickles (searched recursively for *_tree_search.p)",
    )
    parser.add_argument(
        "output_pickle",
        help="Path to write the merged pickle",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "**", "*_tree_search.p")
    pickle_files = sorted(glob(pattern, recursive=True))

    if not pickle_files:
        print(f"No *_tree_search.p files found in {args.input_dir}")
        return

    merged = {}
    for path in pickle_files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        overlap = set(merged.keys()) & set(data.keys())
        if overlap:
            print(f"WARNING: {len(overlap)} duplicate tree keys from {path}")
        merged.update(data)
        print(f"Loaded {len(data)} trees from {os.path.basename(path)}")

    os.makedirs(os.path.dirname(args.output_pickle), exist_ok=True)
    with open(args.output_pickle, "wb") as f:
        pickle.dump(merged, f)

    print(f"\nMerged {len(merged)} total trees from {len(pickle_files)} files")
    print(f"Saved to {args.output_pickle}")


if __name__ == "__main__":
    main()
