"""Visualize labeling problem metrics from quantify_labeling_problem.py CSVs.

Produces a multi-panel summary figure with strip plots for score gap
and fraction of non-DAG edges.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

from dpvtex.plotting import DATASET_NAMES, DPI, FONT_LARGE, FONT_MED, FONT_SMALL

PALETTE = "Dark2"


def _filter_last_tree_per_replicate(df):
    """Keep only the last tree (max tree_index) per replicate."""
    mask = (
        df.groupby(["dataset", "start_type", "replicate"])["tree_index"].transform(
            "max"
        )
        == df["tree_index"]
    )
    return df[mask]


def _get_display_name(dataset):
    """Map raw dataset name to a human-readable display name."""
    return DATASET_NAMES.get(dataset, dataset)


def _make_strip_panel(ax, df, y_col, title, ylabel):
    """Draw a strip plot with per-dataset jittered points and a reference line at 0.

    Args:
        ax: Matplotlib axes to draw on.
        df: DataFrame with 'dataset_display' and y_col columns.
        y_col: Column name for the y-axis values.
        title: Panel title.
        ylabel: Y-axis label.
    """
    sns.stripplot(
        data=df,
        x="dataset_display",
        y=y_col,
        hue="start_type",
        dodge=True,
        jitter=0.2,
        alpha=0.7,
        size=5,
        palette=PALETTE,
        ax=ax,
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=FONT_LARGE)
    ax.set_ylabel(ylabel, fontsize=FONT_MED)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=FONT_SMALL, rotation=30)
    ax.tick_params(axis="y", labelsize=FONT_SMALL)
    ax.legend(
        title="Start type",
        fontsize=FONT_SMALL - 2,
        title_fontsize=FONT_SMALL - 1,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )


def plot_labeling_problem(df, output_dir):
    """Create a multi-panel summary figure from labeling problem metrics.

    Panels are:
    1. Score gap by dataset (always shown)
    2. Fraction of non-DAG edges by dataset (if column present)

    Args:
        df: DataFrame loaded from one or more quantify_labeling_problem CSVs.
        output_dir: Directory to save the output figure.
    """
    df = df.copy()
    df["dataset_display"] = df["dataset"].map(_get_display_name)

    # Handle old CSVs that used "frac_suspect_labels" instead of "frac_non_dag_edges"
    if "frac_non_dag_edges" not in df.columns and "frac_suspect_labels" in df.columns:
        df["frac_non_dag_edges"] = df["frac_suspect_labels"]

    # Determine which optional panels to include
    has_non_dag = (
        "frac_non_dag_edges" in df.columns and df["frac_non_dag_edges"].notna().any()
    )

    panels = [
        ("score_gap", "Score gap (phangorn best − larch MP)", "Score gap"),
    ]
    if has_non_dag:
        panels.append(
            ("frac_non_dag_edges", "Fraction of edges not supported by DAG", "Fraction")
        )

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), squeeze=False)
    axes = axes.ravel()

    for ax, (col, title, ylabel) in zip(axes, panels):
        _make_strip_panel(ax, df, col, title, ylabel)

    fig.tight_layout(h_pad=3.0)
    output_path = os.path.join(output_dir, "labeling_problem_summary.pdf")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def plot_labeling_problem_all_trees(df, output_dir):
    """Scatter plot of frac_non_dag_edges for every intermediate tree.

    Layout: one row per dataset, one column per start_type.
    x-axis = replicate, y-axis = frac_non_dag_edges.
    Color = normalized_tree_index (viridis: dark=start, yellow=end).

    Args:
        df: DataFrame with tree_index and normalized_tree_index columns.
        output_dir: Directory to save the output figure.
    """
    df = df.copy()
    df["dataset_display"] = df["dataset"].map(_get_display_name)

    datasets = df["dataset_display"].unique()
    start_types = df["start_type"].unique()
    n_rows = len(datasets)
    n_cols = len(start_types)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols + 1, 4 * n_rows),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=1)

    for row_idx, dataset_disp in enumerate(datasets):
        for col_idx, st in enumerate(start_types):
            ax = axes[row_idx, col_idx]
            sub = df[(df["dataset_display"] == dataset_disp) & (df["start_type"] == st)]

            if sub.empty:
                ax.set_visible(False)
                continue

            # Assign integer x-position per replicate (sorted)
            reps = sorted(sub["replicate"].unique())
            rep_to_x = {r: i for i, r in enumerate(reps)}
            x = sub["replicate"].map(rep_to_x)

            sc = ax.scatter(
                x + np.random.default_rng(42).uniform(-0.2, 0.2, size=len(x)),
                sub["frac_non_dag_edges"],
                c=sub["normalized_tree_index"],
                cmap=cmap,
                norm=norm,
                s=12,
                alpha=0.7,
                edgecolors="none",
            )

            ax.set_xticks(range(len(reps)))
            ax.set_xticklabels(
                [r.replace("_tree_search.p", "") for r in reps],
                fontsize=FONT_SMALL - 2,
                rotation=45,
                ha="right",
            )
            ax.tick_params(axis="y", labelsize=FONT_SMALL)

            if row_idx == 0:
                ax.set_title(st, fontsize=FONT_LARGE)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{dataset_disp}\nFrac non-DAG edges",
                    fontsize=FONT_MED,
                )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.6,
        pad=0.04,
    )
    cbar.set_label("Search progress (0=start, 1=end)", fontsize=FONT_MED)
    cbar.ax.tick_params(labelsize=FONT_SMALL)
    output_path = os.path.join(output_dir, "labeling_problem_all_trees.pdf")
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot labeling problem metrics from quantify_labeling_problem CSVs."
    )
    parser.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="One or more CSV files produced by quantify_labeling_problem.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save output figure(s)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dfs = []
    for csv_path in args.csv_files:
        print(f"Loading {csv_path}")
        dfs.append(pd.read_csv(csv_path))
    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(df)} rows across {df['dataset'].nunique()} datasets")

    if "tree_index" in df.columns:
        # All-trees CSV: produce scatter plot and summary strip plot
        plot_labeling_problem_all_trees(df, args.output_dir)
        # Filter to last tree per replicate for the summary strip plot
        plot_labeling_problem(_filter_last_tree_per_replicate(df), args.output_dir)
    else:
        plot_labeling_problem(df, args.output_dir)


if __name__ == "__main__":
    main()
