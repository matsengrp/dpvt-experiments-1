"""Visualize labeling problem metrics from quantify_labeling_problem.py CSVs.

Produces a multi-panel summary figure with strip plots for score gap
and fraction of non-DAG edges.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dpvtex.plotting import DATASET_NAMES, DPI, FONT_LARGE, FONT_MED, FONT_SMALL

PALETTE = "Dark2"


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
    ax.legend(title="Start type", fontsize=FONT_SMALL - 2, title_fontsize=FONT_SMALL - 1)


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
    has_non_dag = "frac_non_dag_edges" in df.columns and df["frac_non_dag_edges"].notna().any()

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
    plot_labeling_problem(df, args.output_dir)


if __name__ == "__main__":
    main()
