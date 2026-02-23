"""Quantify labeling problems in tree search evaluation data.

Compares phangorn's best parsimony scores against larch's DAG optimum
to identify trees where edge labels may be incorrect. See
docs/quantify_labeling_plan.md for full context.
"""

import argparse
import glob
import os
import pickle
import sys

import historydag as hdag
import pandas as pd

from dpvtex.treesearch_plots import get_parsimony_scores


def get_dag_mp_score(dag_path, fasta_path):
    """Load a larch DAG, extract one MP tree, and compute its parsimony score.

    Args:
        dag_path: Path to larch protobuf DAG file (.pb or .pb.gz).
        fasta_path: Path to FASTA alignment file.

    Returns:
        int: The parsimony score of an optimal tree from the DAG.
    """
    dag = hdag.mutation_annotated_dag.load_MAD_protobuf_file(
        dag_path, compact_genomes=True
    )
    dag.trim_optimal_weight()
    seq_dag = hdag.sequence_dag.SequenceHistoryDag.from_history_dag(dag)
    seq_dag.unlabel()

    # Extract one tree and compute its parsimony score
    tree = next(seq_dag.get_histories()).to_ete(
        name_func=lambda n: n.label.node_id, features=["sequence"]
    )
    scores = get_parsimony_scores([tree], fasta_path)
    return scores[0]


def analyze_replicate(pickle_path, fasta_path, mp_score):
    """Analyze a single replicate for labeling problems.

    Args:
        pickle_path: Path to tree search pickle (dict of ete3.Tree -> list[int]).
        fasta_path: Path to FASTA alignment file.
        mp_score: The larch DAG's MP parsimony score.

    Returns:
        dict: Metrics for this replicate:
            - score_gap: phangorn best - larch MP (negative = phangorn beats larch)
            - num_at_or_below_mp: trees with parsimony score <= larch MP
            - num_below_mp: trees that strictly beat larch
            - frac_late_search_at_mp: fraction of last 20% of trees at/below MP
            - total_nonmp_labels_in_mp_trees: suspect non-MP labels in MP-score trees
            - frac_suspect_labels: suspect labels / total edges in those trees
    """
    with open(pickle_path, "rb") as f:
        data_dict = pickle.load(f)

    trees = list(data_dict.keys())
    labels_list = list(data_dict.values())
    num_trees = len(trees)

    if num_trees == 0:
        return None

    # Compute parsimony scores for all trees
    pscores = get_parsimony_scores(trees, fasta_path)
    best_score = min(pscores)
    score_gap = best_score - mp_score

    # Count trees at or below MP
    num_at_or_below_mp = sum(1 for s in pscores if s <= mp_score)
    num_below_mp = sum(1 for s in pscores if s < mp_score)

    # Late-search statistics (last 20% of trees)
    late_start = int(num_trees * 0.8)
    late_scores = pscores[late_start:]
    frac_late_search_at_mp = (
        sum(1 for s in late_scores if s <= mp_score) / len(late_scores)
        if late_scores
        else 0.0
    )

    # Count suspect labels: non-MP labels in trees with score <= MP
    # Positions 0-1 are masked root/first-child (always labeled 0), so skip them
    total_suspect_labels = 0
    total_edges_in_mp_trees = 0
    for i, score in enumerate(pscores):
        if score <= mp_score:
            labels = labels_list[i]
            # Count non-MP labels at positions 2+ (skip masked root/first-child)
            suspect = sum(1 for label in labels[2:] if label == 1)
            total_suspect_labels += suspect
            total_edges_in_mp_trees += len(labels) - 2  # exclude masked positions

    frac_suspect_labels = (
        total_suspect_labels / total_edges_in_mp_trees
        if total_edges_in_mp_trees > 0
        else 0.0
    )

    return {
        "score_gap": score_gap,
        "num_at_or_below_mp": num_at_or_below_mp,
        "num_below_mp": num_below_mp,
        "frac_late_search_at_mp": frac_late_search_at_mp,
        "total_nonmp_labels_in_mp_trees": total_suspect_labels,
        "frac_suspect_labels": frac_suspect_labels,
    }


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
        print(f"  {{data-root}}/treesearch/{{start_type}}_starting/{{dataset}}/  (pickle files)")
        print(f"  {{data-root}}/viral/treesearch/{{dataset}}/           (FASTA + DAG)")
        print("Missing:")
        for e in errors:
            print(e)
        sys.exit(1)

    print(f"Datasets: {datasets}")
    print(f"Start types: {start_types}")

    # Cache DAG MP scores per dataset
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
            f"at_or_below_mp={metrics['num_at_or_below_mp']}, "
            f"frac_suspect={metrics['frac_suspect_labels']:.4f}"
        )

    if not results:
        print("\nNo results collected.")
        sys.exit(1)

    # Build DataFrame and save
    df = pd.DataFrame(results)
    column_order = [
        "dataset",
        "start_type",
        "replicate",
        "mp_score",
        "score_gap",
        "num_at_or_below_mp",
        "num_below_mp",
        "frac_late_search_at_mp",
        "total_nonmp_labels_in_mp_trees",
        "frac_suspect_labels",
    ]
    df = df[column_order]
    dataset_tag = "_".join(datasets)
    output_csv = os.path.join(args.output_dir, f"labeling_problem_{dataset_tag}.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Print summary
    print("\n=== Summary by dataset ===")
    for dataset in df["dataset"].unique():
        ddf = df[df["dataset"] == dataset]
        print(f"\n{dataset}:")
        print(f"  MP score (larch):            {ddf['mp_score'].iloc[0]}")
        print(f"  Avg score gap:               {ddf['score_gap'].mean():.1f}")
        print(f"  Replicates with gap <= 0:    {(ddf['score_gap'] <= 0).sum()} / {len(ddf)}")
        print(f"  Avg trees at/below MP:       {ddf['num_at_or_below_mp'].mean():.1f}")
        print(f"  Avg frac late search at MP:  {ddf['frac_late_search_at_mp'].mean():.3f}")
        print(f"  Avg frac suspect labels:     {ddf['frac_suspect_labels'].mean():.4f}")


if __name__ == "__main__":
    main()
