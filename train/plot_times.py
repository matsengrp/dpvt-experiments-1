"""Runtime visualization for DPVT training and testing."""

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def get_benchmark_paths(cfg: PlotConfig, is_test: bool = False) -> list[str]:
    """Get all benchmark file paths for train or test benchmarks."""
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
    """Load tree counts, average leaves, and average sites for datasets."""
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
    """Load timing data from benchmark TSV files."""
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


def load_preprocessing_times(cfg: PlotConfig) -> dict[tuple[str, str, str], float]:
    """Load all preprocessing times from profiler output files."""
    if not cfg.profiler_prefix:
        return {}

    times = {}
    for model in cfg.models:
        for train in cfg.train_nicknames:
            for test in cfg.test_nicknames:
                fname = f"preprocessing_test-{model}-{train}-ON-{test}-Param0.txt"
                path = Path(cfg.profiler_prefix) / fname
                if not path.exists():
                    continue
                try:
                    for line in path.read_text().splitlines():
                        if line.startswith("Total"):
                            times[(model, train, test)] = float(line.split()[1])
                            break
                except (ValueError, IndexError):
                    pass
    return times


def _is_simulated(name: str) -> bool:
    """Check if dataset name indicates simulated data."""
    return "simulated" in name.lower() or "alisim" in name.lower()


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_training_times(cfg: PlotConfig, output_path: str = "training_times.pdf") -> None:
    """Create a bar plot of training times for all models and datasets."""
    # Load data
    _, leaves, sites = load_dataset_metadata(cfg, cfg.train_nicknames)
    df = load_benchmark_times(cfg, get_benchmark_paths(cfg, is_test=False))
    if df.empty:
        print("No training data found")
        return

    # Build labels with metadata
    df["avg_leaves"] = df["train_data"].map(leaves)
    df["time_minutes"] = df["time_seconds"] / 60

    def make_label(row):
        name = "simulated" if _is_simulated(row["train_data"]) else row["train_label"]
        return f"{name}\n(n={int(row['avg_leaves'])}, N={int(sites[row['train_data']])})"

    df["label"] = df.apply(make_label, axis=1)
    order = df.groupby("label")["avg_leaves"].first().sort_values().index.tolist()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="label", y="time_minutes", hue="model_label",
                order=order, palette=PALETTE, ax=ax)
    ax.set_xlabel("Training dataset (n=leaves, N=sites)", fontsize=FONT_LARGE, labelpad=15)
    ax.set_ylabel("Time (minutes)", fontsize=FONT_LARGE)
    ax.set_title("Training times by model and dataset", fontsize=FONT_LARGE)
    ax.tick_params(labelsize=FONT_MED)
    ax.legend(title="Model", fontsize=FONT_MED, title_fontsize=FONT_LARGE,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(ha="center", fontsize=FONT_LARGE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_testing_times(
    cfg: PlotConfig, output_prefix: str = "testing_times_per_tree"
) -> None:
    """Create bar plots showing preprocessing vs inference time per tree."""
    if not cfg.profiler_prefix:
        print("Warning: profiler_prefix not set, cannot create stacked plots")
        return

    # Load data
    counts, leaves, sites = load_dataset_metadata(cfg, cfg.test_nicknames)
    df = load_benchmark_times(cfg, get_benchmark_paths(cfg, is_test=True), counts)
    if df.empty or "num_trees" not in df.columns:
        print("No testing data found")
        return

    preproc_times = load_preprocessing_times(cfg)
    if not preproc_times:
        print("Warning: No preprocessing times found")
        return

    # Build labels and compute times
    df["avg_leaves"] = df["test_data"].map(leaves)
    df["time_per_tree"] = df["time_seconds"] / df["num_trees"]

    def make_label(row):
        name = "simulated" if _is_simulated(row["test_data"]) else row["test_label"]
        return f"{name}\n(n={int(row['avg_leaves'])},\nN={int(sites[row['test_data']])},\nT={int(counts[row['test_data']])})"

    df["label"] = df.apply(make_label, axis=1)
    order = df.groupby("label")["avg_leaves"].first().sort_values().index.tolist()

    # Add preprocessing/inference breakdown
    df["preproc_per_tree"] = df.apply(
        lambda r: preproc_times.get((r["model"], r["train_data"], r["test_data"]), 0) / r["num_trees"],
        axis=1
    )
    df["inference_per_tree"] = (df["time_per_tree"] - df["preproc_per_tree"]).clip(lower=0)

    # Create one plot per training dataset
    for train_label in df["train_label"].unique():
        subset = df[df["train_label"] == train_label]
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
                preproc_vals.append(row["preproc_per_tree"].values[0] if len(row) else 0)
                inference_vals.append(row["inference_per_tree"].values[0] if len(row) else 0)

            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, preproc_vals, width, label=f"{model} (preprocess)",
                   color=colors[i], alpha=0.5, edgecolor="black", linewidth=0.5)
            ax.bar(x + offset, inference_vals, width, bottom=preproc_vals,
                   label=f"{model} (inference)", color=colors[i], edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Test dataset (n=leaves, N=sites, T=trees)", fontsize=FONT_LARGE, labelpad=15)
        ax.set_ylabel("Time per tree (seconds)", fontsize=FONT_LARGE)
        ax.set_title(f"Testing time breakdown - trained on {train_label}\n"
                     "(lighter = preprocessing, darker = inference)", fontsize=FONT_LARGE)
        ax.set_xticks(x)
        ax.set_xticklabels(test_labels, rotation=45, ha="right", fontsize=FONT_SMALL)
        ax.tick_params(labelsize=FONT_MED)
        ax.legend(title="Model", fontsize=FONT_SMALL, title_fontsize=FONT_MED,
                  bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        safe_name = train_label.replace(" ", "_").replace("=", "").replace("\n", "_")
        plt.savefig(f"{output_prefix}_{safe_name}.pdf", dpi=DPI, bbox_inches="tight")
        print(f"Saved: {output_prefix}_{safe_name}.pdf")
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
    cfg = PlotConfig.from_args(
        models, train_data_names, test_data_names, benchmark_dir, nicknames_path
    )
    # Add profiler prefix if provided
    if profiler_dir:
        cfg.profiler_prefix = (
            profiler_dir if profiler_dir.endswith("/") else f"{profiler_dir}/"
        )

    os.makedirs(output_dir, exist_ok=True)

    plot_training_times(cfg, os.path.join(output_dir, "training_times.pdf"))
    plot_testing_times(
        cfg, os.path.join(output_dir, "testing_times_per_tree")
    )

    print(f"All plots saved to {output_dir}")
