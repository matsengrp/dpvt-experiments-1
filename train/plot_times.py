"""
Runtime visualization for DPVT training and testing.

This module provides functions to visualize training and testing times
for different models and datasets. It can be used standalone or called
from Snakemake with configuration parameters.

Usage:
    Standalone: python plot_times.py
    From Snakemake: Call generate_benchmark_plots() with config parameters
"""

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =============================================================================
# Constants
# =============================================================================

FONT_LARGE, FONT_MED, FONT_SMALL = 16, 14, 12
MARKER_SCALE, MIN_MARKER = 300, 50
DPI, PALETTE = 300, "Dark2"

MODEL_NAMES = {
    "TraverseAvgPooling": "Average pooling",
    "TraverseMaxPooling": "Maximum pooling",
    "TraverseNN": "Transformer encoder",
    "BaselineReversion": "Baseline reversion",
}

DEFAULT_TRAIN = {
    "simulated_25_seq_100_sites_500_algnmnts_few_spr_filtered_0.8_spr": "sim 25",
    "simulated_50_seq_100_sites_500_algnmnts_few_spr_filtered_0.8_spr": "sim 50",
    "orthomam_train_0.5_1000_samples_spr": "OrthoMaM",
}

DEFAULT_TEST = {
    "orthomam_test_0.5_spr": "OrthoMaM Test",
    "influenzaC_fluC_M_spr": "flu C M",
    "influenzaC_fluC_NS_spr": "flu C NS",
    "influenzaC_fluC_PB2_spr": "flu C PB2",
    "rotavirusA_H_H2_spr": "rotavirus A H H2",
    "simulated_25_seq_100_sites_200_algnmnts_few_spr_filtered_0.8_spr": "sim 25",
    "simulated_50_seq_100_sites_200_algnmnts_few_spr_filtered_0.8_spr": "sim 50",
    "pandit_full_0.8_spr": "PANDIT",
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PlotConfig:
    """Configuration for benchmark plotting.

    Attributes:
        nicknames_dict: Dictionary mapping dataset nicknames to file paths
        benchmark_prefix: Path prefix for benchmark log files
        train_nicknames: Maps training dataset names to display labels
        test_nicknames: Maps test dataset names to display labels
        models: Maps model names to display labels
    """

    nicknames_dict: dict
    benchmark_prefix: str
    train_nicknames: dict = field(default_factory=dict)
    test_nicknames: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)

    @property
    def data_dir(self) -> str:
        """Get the data directory from nicknames dict."""
        return self.nicknames_dict.get("data_dir", ".")

    @classmethod
    def from_args(
        cls,
        models: list[str],
        train_names: list[str],
        test_names: list[str],
        benchmark_dir: str,
        nicknames_path: str,
    ) -> "PlotConfig":
        """Create a PlotConfig from command-line or Snakemake arguments.

        Args:
            models: List of model names
            train_names: List of training dataset nicknames
            test_names: List of test dataset nicknames
            benchmark_dir: Path to benchmark_logs directory
            nicknames_path: Path to data_nicknames.json

        Returns:
            Configured PlotConfig instance
        """
        with open(nicknames_path) as f:
            nicknames = json.load(f)
        prefix = benchmark_dir if benchmark_dir.endswith("/") else f"{benchmark_dir}/"
        return cls(
            nicknames_dict=nicknames,
            benchmark_prefix=prefix,
            train_nicknames={n: n for n in train_names},
            test_nicknames={n: n for n in test_names},
            models={n: MODEL_NAMES.get(n, n) for n in models},
        )

    @classmethod
    def standalone(cls, nicknames_path: str = "my_data_nicknames.json") -> "PlotConfig":
        """Create config for standalone script execution with default values.

        Args:
            nicknames_path: Path to the nicknames JSON file

        Returns:
            Configured PlotConfig instance with default datasets
        """
        with open(nicknames_path) as f:
            nicknames = json.load(f)
        return cls(
            nicknames_dict=nicknames,
            benchmark_prefix="_output/run.final/benchmark_logs/",
            train_nicknames=DEFAULT_TRAIN.copy(),
            test_nicknames=DEFAULT_TEST.copy(),
            models={k: v for k, v in MODEL_NAMES.items() if k != "BaselineReversion"},
        )


# =============================================================================
# Data Loading
# =============================================================================


def get_benchmark_paths(cfg: PlotConfig, is_test: bool = False) -> list[str]:
    """Get all benchmark file paths for train or test benchmarks.

    Args:
        cfg: Plot configuration
        is_test: If True, get test benchmark paths; otherwise training paths

    Returns:
        List of benchmark TSV file paths
    """
    paths = []
    btype = "test_model" if is_test else "train_model"
    for model in cfg.models:
        for train in cfg.train_nicknames:
            if is_test:
                for test in cfg.test_nicknames:
                    fname = f"{model}-{train}-ON-{test}-Param0.tsv"
                    paths.append(f"{cfg.benchmark_prefix}{btype}/{fname}")
            else:
                fname = f"{model}-{train}-Param0.tsv"
                paths.append(f"{cfg.benchmark_prefix}{btype}/{fname}")
    return paths


def load_dataset_metadata(cfg: PlotConfig, nicknames: dict) -> tuple[dict, dict, dict]:
    """Load tree counts, average leaves, and average sites for datasets.

    Args:
        cfg: Plot configuration
        nicknames: Dictionary of dataset nicknames to load metadata for

    Returns:
        Tuple of three dictionaries:
        - counts: Maps nickname to number of trees
        - avg_leaves: Maps nickname to average leaves per tree
        - avg_sites: Maps nickname to average sites per tree
    """
    counts, leaves, sites = {}, {}, {}

    for name in nicknames:
        if name not in cfg.nicknames_dict:
            print(f"Warning: {name} not in nicknames_dict")
            continue

        fpath = f"{cfg.data_dir}/{cfg.nicknames_dict[name]}"
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)

            counts[name] = len(data)
            leaves[name] = (
                sum(len(t) + 1 for t in data.keys()) / len(data) if data else 0
            )

            total_sites, n_trees = 0, 0
            for tree in data.keys():
                tree_leaves = tree.get_leaves()
                if tree_leaves:
                    total_sites += len(tree_leaves[0].sequence)
                    n_trees += 1
            sites[name] = total_sites / n_trees if n_trees else 0

            print(
                f"{name}: {counts[name]} trees, "
                f"{leaves[name]:.1f} leaves, {sites[name]:.1f} sites"
            )
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return counts, leaves, sites


def load_benchmark_times(
    cfg: PlotConfig, paths: list[str], tree_counts: dict | None = None
) -> pd.DataFrame:
    """Load timing data from benchmark TSV files.

    Args:
        cfg: Plot configuration
        paths: List of paths to benchmark TSV files
        tree_counts: Optional dict mapping dataset nickname to tree count

    Returns:
        DataFrame with columns: model, model_label, train_data, train_label,
        time_seconds, and optionally test_data, test_label, num_trees
    """
    data = []

    for path in paths:
        if not Path(path).exists():
            print(f"Warning: {path} not found")
            continue

        df = pd.read_csv(path, sep="\t")
        time_str = df.iloc[0, 1]
        h, m, s = time_str.split(":")
        seconds = int(h) * 3600 + int(m) * 60 + float(s)

        parts = Path(path).stem.split("-")
        model = parts[0]

        entry = {
            "model": model,
            "model_label": cfg.models.get(model, model),
            "train_data": parts[1],
            "train_label": cfg.train_nicknames.get(parts[1], parts[1]),
            "time_seconds": seconds,
        }

        if "ON" in Path(path).stem:
            entry["test_data"] = parts[3]
            entry["test_label"] = cfg.test_nicknames.get(parts[3], parts[3])
            if tree_counts and parts[3] in tree_counts:
                entry["num_trees"] = tree_counts[parts[3]]

        data.append(entry)

    return pd.DataFrame(data)


# =============================================================================
# Data Preparation
# =============================================================================


def prepare_data_with_metadata(
    df: pd.DataFrame,
    data_col: str,
    label_col: str,
    avg_leaves: dict,
    avg_sites: dict,
    multiline: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Add metadata columns and compute sorted label order.

    Args:
        df: DataFrame to enrich
        data_col: Column containing dataset nicknames
        label_col: Column containing display labels
        avg_leaves: Dict mapping nicknames to average leaves
        avg_sites: Dict mapping nicknames to average sites
        multiline: If True, use multiline format for labels

    Returns:
        Tuple of (enriched DataFrame, sorted label order)
    """
    df = df.copy()
    df["avg_leaves"] = df[data_col].map(avg_leaves)
    df["avg_sites"] = df[data_col].map(avg_sites)

    sep = ",\n" if multiline else ", "
    df[f"{label_col}_sized"] = df.apply(
        lambda r: f"{r[label_col]}\n(n={int(r['avg_leaves'])}{sep}N={int(r['avg_sites'])})",
        axis=1,
    )

    order = (
        df.groupby(f"{label_col}_sized")["avg_leaves"]
        .first()
        .sort_values()
        .index.tolist()
    )
    return df, order


def prepare_training_data(cfg: PlotConfig) -> tuple[pd.DataFrame, list[str]] | None:
    """Load and prepare training timing data.

    Args:
        cfg: Plot configuration

    Returns:
        Tuple of (prepared DataFrame, sorted label order) or None if no data
    """
    _, leaves, sites = load_dataset_metadata(cfg, cfg.train_nicknames)
    df = load_benchmark_times(cfg, get_benchmark_paths(cfg, is_test=False))

    if df.empty:
        print("No training data found")
        return None

    df["time_minutes"] = df["time_seconds"] / 60
    return prepare_data_with_metadata(df, "train_data", "train_label", leaves, sites)


def prepare_testing_data(cfg: PlotConfig) -> tuple[pd.DataFrame, list[str]] | None:
    """Load and prepare testing timing data with per-tree calculations.

    Args:
        cfg: Plot configuration

    Returns:
        Tuple of (prepared DataFrame, sorted label order) or None if no data
    """
    counts, leaves, sites = load_dataset_metadata(cfg, cfg.test_nicknames)
    df = load_benchmark_times(cfg, get_benchmark_paths(cfg, is_test=True), counts)

    if df.empty:
        print("No testing data found")
        return None

    if "num_trees" not in df.columns:
        print("Warning: num_trees not found")
        return None

    df["time_per_tree"] = df["time_seconds"] / df["num_trees"]
    return prepare_data_with_metadata(
        df, "test_data", "test_label", leaves, sites, multiline=True
    )


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_training_times(
    cfg: PlotConfig, output_path: str = "training_times.pdf"
) -> None:
    """Create a bar plot of training times for all models and datasets.

    Args:
        cfg: Plot configuration
        output_path: Path to save the plot
    """
    result = prepare_training_data(cfg)
    if not result:
        return
    df, order = result

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="train_label_sized",
        y="time_minutes",
        hue="model_label",
        order=order,
        palette=PALETTE,
        ax=ax,
    )

    ax.set_xlabel(
        "Training dataset (n=leaves, N=sites)", fontsize=FONT_LARGE, labelpad=15
    )
    ax.set_ylabel("Time (minutes)", fontsize=FONT_LARGE)
    ax.set_title("Training times by model and dataset", fontsize=FONT_LARGE)
    ax.tick_params(labelsize=FONT_MED)
    ax.legend(title="Model", fontsize=FONT_MED, title_fontsize=FONT_LARGE)
    plt.xticks(ha="center", fontsize=FONT_LARGE)
    plt.tight_layout()

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_testing_times_bar(
    cfg: PlotConfig, output_prefix: str = "testing_times_per_tree"
) -> None:
    """Create bar plots of testing times per tree for each training dataset.

    Creates one PDF file per training dataset.

    Args:
        cfg: Plot configuration
        output_prefix: Prefix for output filenames (will append train dataset name)
    """
    result = prepare_testing_data(cfg)
    if not result:
        return
    df, order = result

    for train_label in df["train_label"].unique():
        subset = df[df["train_label"] == train_label]
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            data=subset,
            x="test_label_sized",
            y="time_per_tree",
            hue="model_label",
            order=order,
            palette=PALETTE,
            ax=ax,
        )

        ax.set_xlabel(
            "Test dataset (n=leaves, N=sites)", fontsize=FONT_LARGE, labelpad=15
        )
        ax.set_ylabel("Time per tree (seconds)", fontsize=FONT_LARGE)
        ax.set_title(
            f"Testing time per tree - trained on {train_label}", fontsize=FONT_LARGE
        )
        ax.tick_params(labelsize=FONT_MED)
        ax.legend(
            title="Model",
            fontsize=FONT_MED,
            title_fontsize=FONT_LARGE,
            loc="upper left",
        )
        plt.tight_layout()

        safe_name = train_label.replace(" ", "_").replace("=", "").replace("\n", "_")
        path = f"{output_prefix}_{safe_name}.pdf"
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)


def plot_testing_times_scatter(
    cfg: PlotConfig, output_prefix: str = "testing_times_scatter"
) -> None:
    """Create scatter plots of testing times with tree size on x-axis.

    Creates one PDF file per training dataset. Marker size indicates
    the number of sites in each dataset.

    Args:
        cfg: Plot configuration
        output_prefix: Prefix for output filenames (will append train dataset name)
    """
    result = prepare_testing_data(cfg)
    if not result:
        return
    df, _ = result

    # Normalize marker sizes
    min_s, max_s = df["avg_sites"].min(), df["avg_sites"].max()
    if max_s > min_s:
        df["marker_size"] = (
            (df["avg_sites"] - min_s) / (max_s - min_s) * MARKER_SCALE
        ) + MIN_MARKER
    else:
        df["marker_size"] = MIN_MARKER + MARKER_SCALE / 2

    colors = dict(
        zip(
            df["model"].unique(),
            sns.color_palette(PALETTE, n_colors=len(df["model"].unique())),
        )
    )

    for train_label in df["train_label"].unique():
        subset = df[df["train_label"] == train_label]
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot each model
        for idx, model in enumerate(subset["model"].unique()):
            mdata = subset[subset["model"] == model]
            ax.scatter(
                mdata["avg_leaves"],
                mdata["time_per_tree"],
                s=mdata["marker_size"],
                c=[colors[model]],
                alpha=0.6,
                edgecolors="black",
                linewidth=1.5,
                label=cfg.models.get(model, model),
            )
            # Add labels for non-MaxPooling models
            if "MaxPooling" not in model:
                y_off = -15 if idx % 2 == 0 else 15
                for _, row in mdata.iterrows():
                    ax.annotate(
                        row["test_label"],
                        (row["avg_leaves"], row["time_per_tree"]),
                        xytext=(5, y_off),
                        textcoords="offset points",
                        fontsize=FONT_SMALL,
                        alpha=0.8,
                    )

        ax.set_xlabel("Average Number of Leaves", fontsize=FONT_LARGE, labelpad=20)
        ax.set_ylabel("Time per Tree (seconds)", fontsize=FONT_LARGE)
        ax.set_title(
            f"Testing Time vs Tree Size - trained on {train_label}", fontsize=FONT_LARGE
        )
        ax.tick_params(labelsize=FONT_LARGE)
        ax.grid(True, alpha=0.3)

        # Model legend
        model_leg = ax.legend(
            title="Model",
            fontsize=FONT_MED,
            title_fontsize=FONT_MED,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        # Size legend
        size_handles = []
        for val in [min_s, (min_s + max_s) / 2, max_s]:
            ms = (
                ((val - min_s) / (max_s - min_s) * MARKER_SCALE + MIN_MARKER)
                if max_s > min_s
                else MIN_MARKER
            )
            size_handles.append(
                plt.scatter(
                    [],
                    [],
                    s=ms,
                    c="gray",
                    alpha=0.6,
                    edgecolors="black",
                    linewidth=1.5,
                    label=f"{int(val)} sites",
                )
            )
        size_leg = ax.legend(
            handles=size_handles,
            title="Sites",
            fontsize=FONT_SMALL,
            title_fontsize=FONT_SMALL,
            bbox_to_anchor=(1.02, 0.5),
            loc="upper left",
        )
        ax.add_artist(model_leg)

        plt.subplots_adjust(right=0.75)
        safe_name = train_label.replace(" ", "_").replace("=", "").replace("\n", "_")
        path = f"{output_prefix}_{safe_name}.pdf"
        plt.savefig(
            path, dpi=DPI, bbox_extra_artists=[model_leg, size_leg], bbox_inches="tight"
        )
        print(f"Saved: {path}")
        plt.close(fig)


# =============================================================================
# Public API
# =============================================================================


def generate_benchmark_plots(
    models: list[str],
    train_data_names: list[str],
    test_data_names: list[str],
    benchmark_dir: str,
    output_dir: str,
    nicknames_path: str,
) -> None:
    """Generate all benchmark plots using provided configuration.

    This is the main entry point for Snakemake integration.

    Args:
        models: List of model names (e.g., ["TraverseAvgPooling", "TraverseNN"])
        train_data_names: List of training data nicknames
        test_data_names: List of test data nicknames
        benchmark_dir: Path to benchmark_logs directory
        output_dir: Directory to save output plots
        nicknames_path: Path to data_nicknames.json
    """
    cfg = PlotConfig.from_args(
        models, train_data_names, test_data_names, benchmark_dir, nicknames_path
    )
    os.makedirs(output_dir, exist_ok=True)

    plot_training_times(cfg, os.path.join(output_dir, "training_times.pdf"))
    plot_testing_times_bar(cfg, os.path.join(output_dir, "testing_times_per_tree"))
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    config = PlotConfig.standalone()
    plot_training_times(config)
    plot_testing_times_bar(config)
