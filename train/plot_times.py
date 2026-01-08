"""Runtime visualization for DPVT training and testing."""

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

FONT_LARGE, FONT_MED, FONT_SMALL = 16, 14, 12
DPI, PALETTE = 300, "Dark2"

MODEL_NAMES = {
    "TraverseAvgPooling": "Average pooling",
    "TraverseMaxPooling": "Maximum pooling",
    "TraverseNN": "Transformer encoder",
    "BaselineReversion": "Baseline reversion",
}

DATASET_NAMES = {
    "orthomam": "OrthoMaM",
    "pandit": "PANDIT",
    "fluC_NS": "flu C NS",
    "fluC_M": "flu C M",
    "fluC_PB2": "flu C PB2",
    "rotavirus": "rotavirus",
}

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PlotConfig:
    """Configuration for benchmark plotting."""

    nicknames_dict: dict
    benchmark_prefix: str
    train_nicknames: dict = field(default_factory=dict)
    test_nicknames: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    profiler_prefix: str | None = None

    @property
    def data_dir(self) -> str:
        """Get the data directory from nicknames dict."""
        return self.nicknames_dict.get("data_dir", ".")

    @classmethod
    def from_args(cls, models, train_names, test_names, benchmark_dir, nicknames_path):
        """Create a PlotConfig from Snakemake arguments."""
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


# =============================================================================
# Data Loading
# =============================================================================


def get_benchmark_paths(config: PlotConfig, for_testing: bool = False) -> list[str]:
    """Get all benchmark file paths for train or test benchmarks."""
    paths = []
    btype = "test_model" if for_testing else "train_model"
    for model in config.models:
        for train in config.train_nicknames:
            if for_testing:
                for test in config.test_nicknames:
                    fname = f"{model}-{train}-ON-{test}-Param0.tsv"
                    paths.append(f"{config.benchmark_prefix}{btype}/{fname}")
            else:
                fname = f"{model}-{train}-Param0.tsv"
                paths.append(f"{config.benchmark_prefix}{btype}/{fname}")
    return paths


def load_dataset_metadata(
    config: PlotConfig, nicknames: dict
) -> tuple[dict, dict, dict]:
    """Load tree counts, average leaves, and average sites for datasets."""
    counts, leaves, sites = {}, {}, {}

    for name in nicknames:
        file_path = f"{config.data_dir}/{config.nicknames_dict[name]}"
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        counts[name] = len(data)
        leaves[name] = sum(len(t) + 1 for t in data.keys()) / len(data) if data else 0

        total_sites, n_trees = 0, 0
        for tree in data.keys():
            tree_leaves = tree.get_leaves()
            if tree_leaves:
                total_sites += len(tree_leaves[0].sequence)
                n_trees += 1
        sites[name] = total_sites / n_trees if n_trees else 0

        logger.info(
            f"{name}: {counts[name]} trees, {leaves[name]:.1f} leaves, {sites[name]:.1f} sites"
        )

    return counts, leaves, sites


def load_benchmark_times(
    config: PlotConfig, paths: list[str], tree_counts: dict | None = None
) -> pd.DataFrame:
    """Load timing data from benchmark TSV files."""
    data = []

    for path in paths:
        if not Path(path).exists():
            logger.warning(f"{path} not found")
            continue

        df = pd.read_csv(path, sep="\t")
        time_str = df.iloc[0, 1]
        h, m, s = time_str.split(":")
        seconds = int(h) * 3600 + int(m) * 60 + float(s)

        parts = Path(path).stem.split("-")
        model = parts[0]

        entry = {
            "model": model,
            "model_label": config.models.get(model, model),
            "train_data": parts[1],
            "train_label": config.train_nicknames.get(parts[1], parts[1]),
            "time_seconds": seconds,
        }

        if "ON" in Path(path).stem:
            entry["test_data"] = parts[3]
            entry["test_label"] = config.test_nicknames.get(parts[3], parts[3])
            if tree_counts and parts[3] in tree_counts:
                entry["num_trees"] = tree_counts[parts[3]]

        data.append(entry)

    return pd.DataFrame(data)


def load_preprocessing_times(config: PlotConfig) -> dict[tuple[str, str, str], float]:
    """Load all preprocessing times from profiler output files."""
    if not config.profiler_prefix:
        return {}

    times = {}
    for model in config.models:
        for train in config.train_nicknames:
            for test in config.test_nicknames:
                fname = f"preprocessing_test-{model}-{train}-ON-{test}-Param0.txt"
                path = Path(config.profiler_prefix) / fname
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    if line.startswith("Total"):
                        times[(model, train, test)] = float(line.split()[1])
                        break
    return times


def _is_simulated(name: str) -> bool:
    """Check if dataset name indicates simulated data."""
    return "simulated" in name.lower() or "alisim" in name.lower()


def _get_dataset_display_name(data_name: str, label: str) -> str:
    """Get display name for a dataset, checking mappings."""
    if _is_simulated(data_name):
        return "simulated"
    for key, display_name in DATASET_NAMES.items():
        if key.lower() in data_name.lower():
            return display_name
    return label


def _make_dataset_label(
    data_name: str,
    label: str,
    avg_leaves: float,
    avg_sites: float,
    tree_count: int | None = None,
) -> str:
    """Create a formatted dataset label for plot axes."""
    name = _get_dataset_display_name(data_name, label)
    if tree_count is not None:
        return f"{name}\n(n={int(avg_leaves)}, N={int(avg_sites)}, T={tree_count})"
    return f"{name}\n(n={int(avg_leaves)}, N={int(avg_sites)})"


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_training_times(
    config: PlotConfig, output_path: str = "training_times.pdf"
) -> None:
    """Create a bar plot of training times for all models and datasets."""
    _, leaves, sites = load_dataset_metadata(config, config.train_nicknames)
    df = load_benchmark_times(config, get_benchmark_paths(config, for_testing=False))
    if df.empty:
        logger.warning("No training data found")
        return

    df["avg_leaves"] = df["train_data"].map(leaves)
    df["time_minutes"] = df["time_seconds"] / 60
    df["label"] = df.apply(
        lambda r: _make_dataset_label(
            r["train_data"], r["train_label"], r["avg_leaves"], sites[r["train_data"]]
        ),
        axis=1,
    )
    order = df.groupby("label")["avg_leaves"].first().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="label",
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
    ax.legend(
        title="Model",
        fontsize=FONT_MED,
        title_fontsize=FONT_LARGE,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.xticks(ha="center", fontsize=FONT_LARGE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_testing_times(
    config: PlotConfig, output_prefix: str = "testing_times_per_tree"
) -> None:
    """Create bar plots showing preprocessing vs inference time per tree."""
    if not config.profiler_prefix:
        logger.warning("profiler_prefix not set, cannot create stacked plots")
        return

    counts, leaves, sites = load_dataset_metadata(config, config.test_nicknames)
    df = load_benchmark_times(
        config, get_benchmark_paths(config, for_testing=True), counts
    )
    if df.empty or "num_trees" not in df.columns:
        logger.warning("No testing data found")
        return

    preproc_times = load_preprocessing_times(config)
    if not preproc_times:
        logger.warning("No preprocessing times found")
        return

    df["avg_leaves"] = df["test_data"].map(leaves)
    df["time_per_tree"] = df["time_seconds"] / df["num_trees"]
    df["label"] = df.apply(
        lambda r: _make_dataset_label(
            r["test_data"],
            r["test_label"],
            r["avg_leaves"],
            sites[r["test_data"]],
            counts[r["test_data"]],
        ),
        axis=1,
    )
    order = df.groupby("label")["avg_leaves"].first().sort_values().index.tolist()

    # Add preprocessing/inference breakdown
    df["preproc_per_tree"] = df.apply(
        lambda r: preproc_times.get((r["model"], r["train_data"], r["test_data"]), 0)
        / r["num_trees"],
        axis=1,
    )
    df["inference_per_tree"] = (df["time_per_tree"] - df["preproc_per_tree"]).clip(
        lower=0
    )

    # Create one plot per training dataset
    for train_label in df["train_label"].unique():
        subset = df[df["train_label"] == train_label]
        train_data = subset["train_data"].iloc[0]
        train_display_name = _get_dataset_display_name(train_data, train_label)
        test_labels = [t for t in order if t in subset["label"].values]
        models = subset["model_label"].unique()
        if not test_labels:
            continue

        x = np.arange(len(test_labels))
        width = 0.8 / len(models)
        colors = sns.color_palette(PALETTE, n_colors=len(models))
        fig, ax = plt.subplots(figsize=(14, 7))

        for i, model in enumerate(models):
            model_data = subset[subset["model_label"] == model]
            preproc_vals, inference_vals = [], []
            for tl in test_labels:
                row = model_data[model_data["label"] == tl]
                preproc_vals.append(
                    row["preproc_per_tree"].values[0] if len(row) else 0
                )
                inference_vals.append(
                    row["inference_per_tree"].values[0] if len(row) else 0
                )

            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                preproc_vals,
                width,
                label=f"{model} (preprocess)",
                color=colors[i],
                alpha=0.5,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar(
                x + offset,
                inference_vals,
                width,
                bottom=preproc_vals,
                label=f"{model} (inference)",
                color=colors[i],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xlabel(
            "Test dataset (n=leaves, N=sites, T=trees)",
            fontsize=FONT_LARGE,
            labelpad=15,
        )
        ax.set_ylabel("Time per tree (seconds)", fontsize=FONT_LARGE)
        ax.set_title(
            f"Testing time breakdown - trained on {train_display_name}",
            fontsize=FONT_LARGE,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(test_labels, rotation=45, ha="right", fontsize=FONT_SMALL)
        ax.tick_params(labelsize=FONT_MED)
        ax.legend(
            title="Model",
            fontsize=FONT_SMALL,
            title_fontsize=FONT_MED,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        plt.tight_layout()

        safe_name = train_label.replace(" ", "_").replace("=", "").replace("\n", "_")
        plt.savefig(f"{output_prefix}_{safe_name}.pdf", dpi=DPI, bbox_inches="tight")
        logger.info(f"Saved: {output_prefix}_{safe_name}.pdf")
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
    profiler_dir: str | None = None,
) -> None:
    """Generate all benchmark plots (main entry point for Snakemake)."""
    config = PlotConfig.from_args(
        models, train_data_names, test_data_names, benchmark_dir, nicknames_path
    )
    # Add profiler prefix if provided
    if profiler_dir:
        config.profiler_prefix = (
            profiler_dir if profiler_dir.endswith("/") else f"{profiler_dir}/"
        )

    os.makedirs(output_dir, exist_ok=True)

    plot_training_times(config, os.path.join(output_dir, "training_times.pdf"))
    plot_testing_times(config, os.path.join(output_dir, "testing_times_per_tree"))

    logger.info(f"All plots saved to {output_dir}")
