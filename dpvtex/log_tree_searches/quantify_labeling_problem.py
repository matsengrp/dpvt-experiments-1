"""Quantify labeling problems in tree search evaluation data.

Compares phangorn's best parsimony scores against larch's DAG optimum
to identify trees where edge labels may be incorrect.
"""

import argparse
import glob
import math
import os
import pickle
import sys

import pandas as pd

from dpvtex.larch.scripts.extract_data_from_hdag import (
    _load_dag_from_file,
    _prepare_dag_for_extraction,
)
from dpvtex.treesearch_plots import get_parsimony_scores


def get_dag_mp_score(dag_path, fasta_path):
    """Load a larch DAG, extract one MP tree, and compute its parsimony score.

    Args:
        dag_path: Path to larch protobuf DAG file (.pb or .pb.gz).
        fasta_path: Path to FASTA alignment file.

    Returns:
        int: The parsimony score of an optimal tree from the DAG.
    """
    dag = _load_dag_from_file(dag_path)
    seq_dag = _prepare_dag_for_extraction(dag)

    tree = next(seq_dag.get_histories()).to_ete(
        name_func=lambda n: n.label.node_id, features=["sequence"]
    )
    scores = get_parsimony_scores([tree], fasta_path)
    return scores[0]


def analyze_replicate(pickle_path, fasta_path, mp_score):
    """Analyze the last tree of a single replicate for labeling problems.

    All metrics are computed on the last tree in the pickle (the final output
    of phangorn's tree search).  Delegates to :func:`analyze_replicate_all_trees`
    and returns only the last-tree row.

    Args:
        pickle_path: Path to tree search pickle (dict of ete3.Tree -> list[int]).
        fasta_path: Path to FASTA alignment file.
        mp_score: The larch DAG's MP parsimony score.

    Returns:
        dict: Metrics for this replicate (all based on the last tree):
            - score_gap: last tree score - larch MP (negative = phangorn beats larch)
            - num_non_dag_edges: edges not supported by DAG in the last tree
            - frac_non_dag_edges: non-DAG edges / total edges in the last tree
    """
    rows = analyze_replicate_all_trees(pickle_path, fasta_path, mp_score)
    if rows is None:
        return None
    last = rows[-1]
    return {
        "score_gap": last["score_gap"],
        "num_non_dag_edges": last["num_non_dag_edges"],
        "frac_non_dag_edges": last["frac_non_dag_edges"],
    }


def analyze_replicate_all_trees(pickle_path, fasta_path, mp_score):
    """Analyze all trees in a single replicate for labeling problems.

    Returns one row per tree. ``frac_non_dag_edges`` is computed cheaply from
    labels for every tree.  ``score_gap`` is only computed for the **last**
    tree (parsimony scoring is expensive); other trees get NaN.

    Args:
        pickle_path: Path to tree search pickle (dict of ete3.Tree -> list[int]).
        fasta_path: Path to FASTA alignment file.
        mp_score: The larch DAG's MP parsimony score.

    Returns:
        list[dict] | None: One dict per tree with keys tree_index,
            normalized_tree_index, score_gap, num_non_dag_edges,
            frac_non_dag_edges.
    """
    with open(pickle_path, "rb") as f:
        data_dict = pickle.load(f)

    trees = list(data_dict.keys())
    labels_list = list(data_dict.values())
    num_trees = len(trees)

    if num_trees == 0:
        return None

    # Compute score_gap only for the last tree
    last_tree = trees[-1]
    last_score = get_parsimony_scores([last_tree], fasta_path)[0]
    last_score_gap = last_score - mp_score

    rows = []
    for i, (tree, labels) in enumerate(zip(trees, labels_list)):
        num_non_dag = sum(labels)
        total_edges = len(tree) - 2
        frac_non_dag = num_non_dag / total_edges if total_edges > 0 else 0.0

        rows.append(
            {
                "tree_index": i,
                "normalized_tree_index": i / (num_trees - 1) if num_trees > 1 else 1.0,
                "score_gap": last_score_gap if i == num_trees - 1 else math.nan,
                "num_non_dag_edges": num_non_dag,
                "frac_non_dag_edges": frac_non_dag,
            }
        )

    return rows


def _filter_last_tree_per_replicate(df):
    """Keep only the last tree (max tree_index) per replicate."""
    mask = (
        df.groupby(["dataset", "start_type", "replicate"])["tree_index"].transform(
            "max"
        )
        == df["tree_index"]
    )
    return df[mask]


def discover_replicates(data_root, datasets, start_types):
    """Discover all replicate pickle files.

    Args:
        data_root: Root data directory.
        datasets: List of dataset names.
        start_types: List of start type names.

    Yields:
        tuple: (dataset, start_type, pickle_path)
    """
    for dataset in datasets:
        for start_type in start_types:
            pattern = os.path.join(
                data_root,
                "treesearch",
                f"{start_type}_starting",
                dataset,
                f"{dataset}_rep*_tree_search.p",
            )
            matches = sorted(glob.glob(pattern))
            if not matches:
                print(f"  Warning: no pickle files found matching: {pattern}")
            for pickle_path in matches:
                yield dataset, start_type, pickle_path


def main():
    parser = argparse.ArgumentParser(
        description="Quantify labeling problems in tree search evaluation data."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root data directory (should contain treesearch/ and viral/treesearch/ subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the output CSV",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Datasets to analyze",
    )
    parser.add_argument(
        "--start-types",
        nargs="+",
        required=True,
        help="Start types to analyze",
    )
    parser.add_argument(
        "--all-trees",
        action="store_true",
        help="Analyze all intermediate trees (not just the last one). "
        "Output has one row per tree per replicate with tree_index and "
        "normalized_tree_index columns.",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    datasets = args.datasets
    start_types = args.start_types

    # Validate expected directory structure
    treesearch_dir = os.path.join(data_root, "treesearch")
    viral_treesearch_dir = os.path.join(data_root, "viral", "treesearch")
    errors = []
    if not os.path.isdir(treesearch_dir):
        errors.append(f"  Pickle directory not found: {treesearch_dir}")
    if not os.path.isdir(viral_treesearch_dir):
        errors.append(f"  FASTA/DAG directory not found: {viral_treesearch_dir}")
    if errors:
        print(f"Error: --data-root '{data_root}' does not have the expected layout.")
        print("Expected subdirectories:")
        print(
            f"  {{data-root}}/treesearch/{{start_type}}_starting/{{dataset}}/  (pickle files)"
        )
        print(f"  {{data-root}}/viral/treesearch/{{dataset}}/           (FASTA + DAG)")
        print("Missing:")
        for e in errors:
            print(e)
        sys.exit(1)

    print(f"Datasets: {datasets}")
    print(f"Start types: {start_types}")

    # Cache DAG MP score per dataset
    mp_score_cache = {}
    results = []

    for dataset, start_type, pickle_path in discover_replicates(
        data_root, datasets, start_types
    ):
        rep_name = os.path.basename(pickle_path)
        print(f"\nProcessing {dataset} / {start_type} / {rep_name} ...")

        fasta_path = os.path.join(
            data_root, "viral", "treesearch", dataset, "input.fasta"
        )
        if not os.path.exists(fasta_path):
            print(f"  Warning: FASTA not found: {fasta_path}, skipping")
            continue

        # Get or compute DAG MP score
        if dataset not in mp_score_cache:
            dag_path = os.path.join(
                data_root, "viral", "treesearch", dataset, "larch-output.pb"
            )
            if not os.path.exists(dag_path):
                print(f"  Warning: DAG not found: {dag_path}, skipping dataset")
                mp_score_cache[dataset] = None
                continue
            print(f"  Computing DAG MP score for {dataset} ...")
            mp_score_cache[dataset] = get_dag_mp_score(dag_path, fasta_path)
            print(f"  DAG MP score: {mp_score_cache[dataset]}")

        mp_score = mp_score_cache[dataset]
        if mp_score is None:
            continue

        if args.all_trees:
            rows = analyze_replicate_all_trees(pickle_path, fasta_path, mp_score)
            if rows is None:
                print("  Warning: empty pickle, skipping")
                continue
            for row in rows:
                row["dataset"] = dataset
                row["start_type"] = start_type
                row["replicate"] = rep_name
                row["mp_score"] = mp_score
                results.append(row)
            last = rows[-1]
            print(
                f"  {len(rows)} trees, "
                f"last: score_gap={last['score_gap']}, "
                f"frac_non_dag_edges={last['frac_non_dag_edges']:.4f}"
            )
        else:
            metrics = analyze_replicate(pickle_path, fasta_path, mp_score)
            if metrics is None:
                print("  Warning: empty pickle, skipping")
                continue
            metrics["dataset"] = dataset
            metrics["start_type"] = start_type
            metrics["replicate"] = rep_name
            metrics["mp_score"] = mp_score
            results.append(metrics)
            print(
                f"  score_gap={metrics['score_gap']}, "
                f"frac_non_dag_edges={metrics['frac_non_dag_edges']:.4f}"
            )

    if not results:
        print("\nNo results collected.")
        sys.exit(1)

    # Build DataFrame and save
    df = pd.DataFrame(results)
    base_cols = [
        "dataset",
        "start_type",
        "replicate",
        "mp_score",
        "score_gap",
        "num_non_dag_edges",
        "frac_non_dag_edges",
    ]
    if args.all_trees:
        column_order = (
            base_cols[:3] + ["tree_index", "normalized_tree_index"] + base_cols[3:]
        )
    else:
        column_order = base_cols
    df = df[column_order]
    dataset_tag = "_".join(datasets)
    suffix = "_all_trees" if args.all_trees else ""
    output_csv = os.path.join(
        args.output_dir, f"labeling_problem_{dataset_tag}{suffix}.csv"
    )
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Print summary (use last tree per replicate for all-trees mode)
    if args.all_trees:
        summary_df = _filter_last_tree_per_replicate(df)
    else:
        summary_df = df
    print("\n=== Summary by dataset (last tree per replicate) ===")
    for dataset in summary_df["dataset"].unique():
        ddf = summary_df[summary_df["dataset"] == dataset]
        print(f"\n{dataset}:")
        print(f"  MP score (larch):            {ddf['mp_score'].iloc[0]}")
        print(f"  Avg score gap:               {ddf['score_gap'].mean():.1f}")
        print(
            f"  Replicates with gap <= 0:    {(ddf['score_gap'] <= 0).sum()} / {len(ddf)}"
        )
        print(f"  Avg frac non-DAG edges:      {ddf['frac_non_dag_edges'].mean():.4f}")


if __name__ == "__main__":
    main()
