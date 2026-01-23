"""Runtime visualization for DPVT training and testing."""

import json
import logging
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
DPI = 300
PALETTE = "Dark2"
BAR_GROUP_WIDTH = 0.8
BAR_CENTERING_OFFSET = 0.5

# Model display names - order determines bar order in plots (rightmost last)
MODEL_NAMES = {
    "TraverseAvgPooling": "Average pooling",
    "TraverseMaxPooling": "Maximum pooling",
    "TraverseNN": "Transformer encoder",
}

# Fixed color and order for consistent plots
_PALETTE_COLORS = sns.color_palette(PALETTE, n_colors=len(MODEL_NAMES))
MODEL_ORDER = list(MODEL_NAMES.values())
MODEL_COLORS = dict(zip(MODEL_ORDER, _PALETTE_COLORS))

DATASET_NAMES = {
    "orthomam": "OrthoMaM",
    "pandit": "PANDIT",
    "fluC_NS": "flu C NS",
    "fluC_M": "flu C M",
    "fluC_PB2": "flu C PB2",
    "rotavirus": "rota A H H2",
}

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PlotConfig:
    """Configuration for benchmark plotting."""

    nicknames_dict: dict
    benchmark_prefix: Path
    train_nicknames: dict = field(default_factory=dict)
    test_nicknames: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    profiler_prefix: Path | None = None

    @property
    def data_dir(self) -> Path:
        """Get the data directory from nicknames dict."""
        return Path(self.nicknames_dict.get("data_dir", "."))

    @classmethod
    def from_args(cls, models, train_names, test_names, benchmark_dir, nicknames_path):
        """Create a PlotConfig from Snakemake arguments."""
        if not models or not train_names or not test_names:
            raise ValueError(
                "models, train_names, and test_names must all be non-empty"
            )

        nicknames_file = Path(nicknames_path)
        if not nicknames_file.exists():
            raise FileNotFoundError(f"Nicknames file not found: {nicknames_path}")

        with open(nicknames_file) as f:
            nicknames = json.load(f)

        return cls(
            nicknames_dict=nicknames,
            benchmark_prefix=Path(benchmark_dir),
            train_nicknames={n: n for n in train_names},
            test_nicknames={n: n for n in test_names},
            models={n: MODEL_NAMES.get(n, n) for n in models},
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_time_to_seconds(time_str: str) -> float:
    """Parse time string to seconds. Supports 'HH:MM:SS' and 'X day(s), HH:MM:SS' formats."""
    try:
        days = 0
        time_part = time_str

        # Handle "X day(s), HH:MM:SS" format
        if "day" in time_str:
            day_part, time_part = time_str.split(", ", 1)
            days = int(day_part.split()[0])

        h, m, s = time_part.split(":")
        return days * 86400 + int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Expected time format 'HH:MM:SS' or 'X day(s), HH:MM:SS', got '{time_str}'") from e


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
        return f"{name}\nn={int(avg_leaves)}\nN={int(avg_sites)}\nT={tree_count}"
    return f"{name}\nn={int(avg_leaves)}, N={int(avg_sites)}"


def _add_labels_to_dataframe(
    df: pd.DataFrame,
    data_column: str,
    label_column: str,
    avg_leaves_by_dataset: dict,
    avg_sites_by_dataset: dict,
    tree_counts_by_dataset: dict | None = None,
) -> pd.DataFrame:
    """Add avg_leaves and formatted label columns to DataFrame."""
    df = df.copy()
    df["avg_leaves"] = df[data_column].map(avg_leaves_by_dataset)
    df["label"] = df.apply(
        lambda r: _make_dataset_label(
            r[data_column],
            r[label_column],
            r["avg_leaves"],
            avg_sites_by_dataset[r[data_column]],
            tree_counts_by_dataset[r[data_column]] if tree_counts_by_dataset else None,
        ),
        axis=1,
    )
    return df


def _compute_label_order(df: pd.DataFrame) -> list[str]:
    """Compute dataset label ordering by average leaves (ascending)."""
    return df.groupby("label")["avg_leaves"].first().sort_values().index.tolist()


# =============================================================================
# Data Loading
# =============================================================================


def get_benchmark_paths(config: PlotConfig, for_testing: bool = False) -> list[Path]:
    """Get all benchmark file paths for train or test benchmarks."""
    paths = []
    btype = "test_model" if for_testing else "train_model"
    for model in config.models:
        for train in config.train_nicknames:
            if for_testing:
                for test in config.test_nicknames:
                    fname = f"{model}-{train}-ON-{test}-Param0.tsv"
                    paths.append(config.benchmark_prefix / btype / fname)
            else:
                fname = f"{model}-{train}-Param0.tsv"
                paths.append(config.benchmark_prefix / btype / fname)
    return paths


def load_dataset_metadata(
    config: PlotConfig, nicknames: dict
) -> tuple[dict, dict, dict]:
    """Load tree counts, average leaves, and average sites for datasets."""
    tree_counts_by_dataset, avg_leaves_by_dataset, avg_sites_by_dataset = {}, {}, {}

    for name in nicknames:
        file_path = config.data_dir / config.nicknames_dict[name]
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        tree_counts_by_dataset[name] = len(data)
        avg_leaves_by_dataset[name] = (
            sum(len(t) + 1 for t in data.keys()) / len(data) if data else 0
        )

        total_sites, n_trees = 0, 0
        for tree in data.keys():
            tree_leaves = tree.get_leaves()
            if tree_leaves:
                total_sites += len(tree_leaves[0].sequence)
                n_trees += 1
        avg_sites_by_dataset[name] = total_sites / n_trees if n_trees else 0

        logger.info(
            f"{name}: {tree_counts_by_dataset[name]} trees, "
            f"{avg_leaves_by_dataset[name]:.1f} leaves, "
            f"{avg_sites_by_dataset[name]:.1f} sites"
        )

    return tree_counts_by_dataset, avg_leaves_by_dataset, avg_sites_by_dataset


def load_benchmark_times(
    config: PlotConfig,
    paths: list[Path],
    tree_counts_by_dataset: dict | None = None,
) -> pd.DataFrame:
    """Load timing data from benchmark TSV files."""
    if not paths:
        raise ValueError("No benchmark paths provided")

    existing_paths = [p for p in paths if p.exists()]
    if not existing_paths:
        raise FileNotFoundError(
            f"No benchmark files found. Searched {len(paths)} paths. "
            f"First missing: {paths[0]}"
        )

    missing_count = len(paths) - len(existing_paths)
    if missing_count:
        logger.warning(f"{missing_count}/{len(paths)} benchmark files not found")

    data = []
    for path in existing_paths:
        df = pd.read_csv(path, sep="\t")
        if df.empty or df.shape[1] < 2:
            raise ValueError(f"Benchmark file has insufficient data: {path}")

        seconds = _parse_time_to_seconds(df.iloc[0, 1])
        parts = path.stem.split("-")

        entry = {
            "model": parts[0],
            "model_label": config.models.get(parts[0], parts[0]),
            "train_data": parts[1],
            "train_label": config.train_nicknames.get(parts[1], parts[1]),
            "time_seconds": seconds,
        }

        if "ON" in path.stem:
            entry["test_data"] = parts[3]
            entry["test_label"] = config.test_nicknames.get(parts[3], parts[3])
            if tree_counts_by_dataset and parts[3] in tree_counts_by_dataset:
                entry["num_trees"] = tree_counts_by_dataset[parts[3]]

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
                path = config.profiler_prefix / fname
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    if line.startswith("Total"):
                        times[(model, train, test)] = float(line.split()[1])
                        break
    return times


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_training_times(
    config: PlotConfig, output_path: str = "training_times.pdf"
) -> None:
    """Create a bar plot of training times for all models and datasets."""
    _, avg_leaves_by_dataset, avg_sites_by_dataset = load_dataset_metadata(
        config, config.train_nicknames
    )
    df = load_benchmark_times(config, get_benchmark_paths(config, for_testing=False))
    if df.empty:
        logger.warning("No training data found")
        return

    df = _add_labels_to_dataframe(
        df, "train_data", "train_label", avg_leaves_by_dataset, avg_sites_by_dataset
    )
    df["time_minutes"] = df["time_seconds"] / 60
    order = _compute_label_order(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="label",
        y="time_minutes",
        hue="model_label",
        order=order,
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.set_xlabel(
        "Training dataset (avg number of leaves, avg number of sites, number of trees)", fontsize=FONT_LARGE, labelpad=15
    )
    ax.set_ylabel("Time (minutes)", fontsize=FONT_LARGE)
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


def _prepare_testing_data(
    config: PlotConfig,
) -> tuple[pd.DataFrame, list[str]] | None:
    """Load and prepare testing data with preprocessing times."""
    if not config.profiler_prefix:
        logger.warning("profiler_prefix not set, cannot create stacked plots")
        return None

    tree_counts_by_dataset, avg_leaves_by_dataset, avg_sites_by_dataset = (
        load_dataset_metadata(config, config.test_nicknames)
    )
    df = load_benchmark_times(
        config, get_benchmark_paths(config, for_testing=True), tree_counts_by_dataset
    )
    if df.empty or "num_trees" not in df.columns:
        logger.warning("No testing data found")
        return None

    preproc_times = load_preprocessing_times(config)
    if not preproc_times:
        logger.warning("No preprocessing times found")
        return None

    df = _add_labels_to_dataframe(
        df,
        "test_data",
        "test_label",
        avg_leaves_by_dataset,
        avg_sites_by_dataset,
        tree_counts_by_dataset,
    )
    df["time_per_tree"] = df["time_seconds"] / df["num_trees"]
    df["preproc_per_tree"] = df.apply(
        lambda r: preproc_times.get((r["model"], r["train_data"], r["test_data"]), 0)
        / r["num_trees"],
        axis=1,
    )
    df["inference_per_tree"] = (df["time_per_tree"] - df["preproc_per_tree"]).clip(
        lower=0
    )

    return df, _compute_label_order(df)


def _plot_single_test_breakdown(
    subset: pd.DataFrame,
    train_label: str,
    order: list[str],
    output_prefix: str,
) -> None:
    """Create a single stacked bar plot for one training dataset."""
    test_labels = [t for t in order if t in subset["label"].values]
    available_models = set(subset["model_label"].unique())
    models = [m for m in MODEL_ORDER if m in available_models]

    if not test_labels:
        return

    x = np.arange(len(test_labels))
    width = BAR_GROUP_WIDTH / len(models)
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, model in enumerate(models):
        model_data = subset[subset["model_label"] == model]
        preproc_vals, inference_vals = [], []
        for tl in test_labels:
            row = model_data[model_data["label"] == tl]
            preproc_vals.append(row["preproc_per_tree"].values[0] if len(row) else 0)
            inference_vals.append(
                row["inference_per_tree"].values[0] if len(row) else 0
            )

        color = MODEL_COLORS.get(model, _PALETTE_COLORS[i % len(_PALETTE_COLORS)])
        offset = (i - len(models) / 2 + BAR_CENTERING_OFFSET) * width
        ax.bar(
            x + offset,
            preproc_vals,
            width,
            label=f"{model} (preprocess)",
            color=color,
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
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel(
        "Test dataset (avg number of leaves, avg number of sites, number of trees)",
        fontsize=FONT_LARGE,
        labelpad=15,
    )
    ax.set_ylabel("Time per tree (seconds)", fontsize=FONT_LARGE)
    ax.set_xticks(x)
    ax.set_xticklabels(test_labels, ha="center", fontsize=FONT_SMALL)
    ax.tick_params(labelsize=FONT_MED)
    ax.legend(
        title="Model",
        fontsize=FONT_SMALL,
        title_fontsize=FONT_MED,
        loc="upper center",
    )
    plt.tight_layout()

    safe_name = train_label.replace(" ", "_").replace("=", "").replace("\n", "_")
    output_path = f"{output_prefix}_{safe_name}.pdf"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_testing_times(
    config: PlotConfig, output_prefix: str = "testing_times_per_tree"
) -> None:
    """Create bar plots showing preprocessing vs inference time per tree."""
    result = _prepare_testing_data(config)
    if result is None:
        return

    df, order = result
    for train_label in df["train_label"].unique():
        subset = df[df["train_label"] == train_label]
        _plot_single_test_breakdown(subset, train_label, order, output_prefix)


# =============================================================================
# Public API
# =============================================================================


def plot_available_training_times(
    run_dir: str,
    nicknames_path: str,
    train_nicknames: list[str],
    output_path: str = "training_times.pdf",
) -> None:
    """Plot training times for all available model/dataset combinations.

    This function discovers which benchmark files exist in the run directory
    and plots training times for all available combinations, allowing for
    incomplete model/dataset matrices.

    Args:
        run_dir: Directory containing benchmark files (with train_model subdir).
        nicknames_path: Path to the JSON file mapping nicknames to data files.
        train_nicknames: List of dataset nicknames to include in the plot.
        output_path: Output path for the PDF plot.
    """
    benchmark_dir = Path(run_dir) / "benchmark_logs" / "train_model"
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    with open(nicknames_path) as f:
        nicknames_dict = json.load(f)

    # Discover available benchmark files
    available_files = list(benchmark_dir.glob("*-*-Param0.tsv"))
    if not available_files:
        raise FileNotFoundError(f"No benchmark files found in {benchmark_dir}")

    # Parse filenames and filter by requested datasets
    data = []
    for path in available_files:
        parts = path.stem.replace("-Param0", "").split("-")
        if len(parts) != 2:
            continue
        model, train_data = parts
        if train_data not in train_nicknames:
            continue

        df = pd.read_csv(path, sep="\t")
        if df.empty or df.shape[1] < 2:
            continue

        seconds = _parse_time_to_seconds(df.iloc[0, 1])
        data.append({
            "model": model,
            "model_label": MODEL_NAMES.get(model, model),
            "train_data": train_data,
            "train_label": train_data,
            "time_seconds": seconds,
        })

    if not data:
        raise ValueError("No matching benchmark data found for specified datasets")

    df = pd.DataFrame(data)

    # Load dataset metadata
    data_dir = Path(nicknames_dict.get("data_dir", "."))
    tree_counts_by_dataset, avg_leaves_by_dataset, avg_sites_by_dataset = {}, {}, {}

    for name in df["train_data"].unique():
        if name not in nicknames_dict:
            continue
        file_path = data_dir / nicknames_dict[name]
        if not file_path.exists():
            continue
        with open(file_path, "rb") as f:
            pkl_data = pickle.load(f)

        tree_counts_by_dataset[name] = len(pkl_data)
        avg_leaves_by_dataset[name] = (
            sum(len(t) + 1 for t in pkl_data.keys()) / len(pkl_data) if pkl_data else 0
        )

        total_sites, n_trees = 0, 0
        for tree in pkl_data.keys():
            tree_leaves = tree.get_leaves()
            if tree_leaves:
                total_sites += len(tree_leaves[0].sequence)
                n_trees += 1
        avg_sites_by_dataset[name] = total_sites / n_trees if n_trees else 0

    # Add labels to dataframe
    df = _add_labels_to_dataframe(
        df,
        "train_data",
        "train_label",
        avg_leaves_by_dataset,
        avg_sites_by_dataset,
        tree_counts_by_dataset,
    )
    df["time_minutes"] = df["time_seconds"] / 60
    order = _compute_label_order(df)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="label",
        y="time_minutes",
        hue="model_label",
        order=order,
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.set_xlabel(
        "Training dataset (avg number of leaves, avg number of sites, number of trees)",
        fontsize=FONT_LARGE,
        labelpad=15,
    )
    ax.set_ylabel("Time (minutes)", fontsize=FONT_LARGE)
    ax.tick_params(labelsize=FONT_MED)
    ax.legend(
        title="Model",
        fontsize=FONT_MED,
        title_fontsize=FONT_LARGE,
        # bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.xticks(ha="center", fontsize=FONT_LARGE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    logger.info(f"Saved: {output_path}")
    plt.close(fig)


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
    if profiler_dir:
        config.profiler_prefix = Path(profiler_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_training_times(config, str(output_path / "training_times.pdf"))
    plot_testing_times(config, str(output_path / "testing_times_per_tree"))

    logger.info(f"All plots saved to {output_dir}")
