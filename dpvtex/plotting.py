"""Plotting utilities for DPVT training summaries and visualizations.

This module provides functions for generating summary plots from training results,
including performance heatmaps, hyperparameter summaries, and runtime visualizations.
"""

import json
import logging
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import tbparse
import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

FONT_LARGE, FONT_MED, FONT_SMALL = 16, 14, 12
DPI = 300
PALETTE = "Dark2"

# Heatmap sizing (inches)
HEATMAP_COL_WIDTH = 1
HEATMAP_BASE_WIDTH = 10
HEATMAP_ROW_HEIGHT = 0.5
HEATMAP_ROW_HEIGHT_PER_LABEL = 0.2
HEATMAP_BASE_HEIGHT = 2

# Label positioning (figure fraction coordinates)
YLABEL_BASE_OFFSET = -0.055
YLABEL_MIXED_SOURCE_ADJUSTMENT = 0.07

# Model display names for plots
MODEL_NAMES = {
    "TraverseMaxPooling": "Maximum",
    "TraverseAvgPooling": "Average",
    "TraverseNN": "Transformer\nEncoder",
    "BaselineReversion": "Baseline",
}

# Standard model ordering for consistent plots
MODEL_ORDER = [
    "TraverseMaxPooling",
    "TraverseAvgPooling",
    "TraverseNN",
    "BaselineReversion",
]

# Dataset display names for plots
DATASET_NAMES = {
    "orthomam": "OrthoMaM",
    "pandit": "PANDIT",
    "fluC_NS": "flu C NS",
    "fluC_M": "flu C M",
    "fluC_PB2": "flu C PB2",
    "fluC": "flu C",
    "rotavirus": "rota A H H2",
    "alisim": "sim",
    "simulated": "sim",
}

# Metric display labels
METRIC_LABELS = {
    "test_auroc": "AUROC",
    "test_accuracy": "Accuracy",
    "test_loss": "Loss",
}

# Label mappings for heatmap axes
COLUMN_LABELS = {
    "model": "Model",
    "train_data": "Dataset",
    "test_data": "Dataset",
    "train_num_leaves": "Number of Leaves",
    "train_num_sites": "Number of Sites",
    "train_num_trees": "Number of Trees",
    "test_num_leaves": "Number of Leaves",
    "test_num_sites": "Number of Sites",
    "test_num_trees": "Number of Trees",
    "train_perturbation": "Perturbation method",
    "test_perturbation": "Perturbation method",
}


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class LabelConfig:
    """Configuration for which labels to show on heatmap axes.

    By default (all fields None), labels are auto-detected: only shown if
    values vary across datasets. Set to True/False to manually override.

    Attributes:
        show_num_leaves: Show number of leaves (n) in labels.
        show_num_sites: Show number of sites (N) in labels.
        show_num_trees: Show number of trees (T) in labels.
        show_nonmp_fraction: Show non-MP edge fraction in labels.
        show_perturbation: Show perturbation method in labels.
    """

    show_num_leaves: bool | None = None
    show_num_sites: bool | None = None
    show_num_trees: bool | None = None
    show_nonmp_fraction: bool | None = None
    show_perturbation: bool | None = None


def _should_show_label(values, override: bool | None) -> bool:
    """Determine if a label should be shown.

    Args:
        values: List of values to check for variation.
        override: If True/False, use that value. If None, auto-detect.

    Returns:
        True if label should be shown, False otherwise.
    """
    if override is not None:
        return override
    unique_values = set(v for v in values if v is not None and pd.notna(v))
    return len(unique_values) > 1


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_data(file_path, file_type=None):
    """Load data from JSON, YAML, or pickle files.

    Args:
        file_path: Path to the file to load.
        file_type: Optional file type override ('json', 'yaml', 'yml', 'pickle', 'pkl', 'p').
            If None, inferred from file extension.

    Returns:
        Loaded data object, or None if file type not recognized.
    """
    file_path = str(file_path)
    # Handle cpu_/gpu_ prefixes in paths
    if "cpu_" in file_path or "gpu_" in file_path:
        file_path = file_path.replace("cpu_", "").replace("gpu_", "")

    if file_type is None:
        file_type = file_path.split(".")[-1]

    if file_type in ["json"]:
        with open(file_path, "r") as file:
            return json.load(file)
    elif file_type in ["yaml", "yml"]:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    elif file_type in ["pickle", "pkl", "p"]:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    else:
        raise ValueError(
            f'Unrecognized file type "{file_type}". '
            f"Supported types: json, yaml, yml, pickle, pkl, p"
        )


def get_stats_from_data(data_path, take_first=True):
    """Extract the number of trees, leaves, and sites from a dataset.

    Args:
        data_path: Path to a pickle file containing tree data.
        take_first: If True, takes the first tree as representative. Otherwise,
            creates min-max ranges over all trees.

    Returns:
        Dictionary with keys: num_trees, num_leaves, num_sites, num_leaves_range, num_sites_range.
    """
    if data_path is None:
        return {
            "num_trees": None,
            "num_leaves": None,
            "num_sites": None,
            "num_leaves_range": [np.inf, -np.inf],
            "num_sites_range": [np.inf, -np.inf],
        }

    data_stats = {
        "num_trees": None,
        "num_leaves": [],
        "num_sites": [],
        "num_leaves_range": [np.inf, -np.inf],
        "num_sites_range": [np.inf, -np.inf],
    }

    logger.info(f"Loading data stats from: {data_path}")
    data_dict = load_data(data_path)

    for i, (tree, vec) in enumerate(data_dict.items()):
        leaves = tree.get_leaves()
        num_leaves = [len(leaves) + 1]  # add one for root leaf
        num_sites = [len(leaves[0].sequence)]

        data_stats["num_leaves"] += num_leaves
        data_stats["num_sites"] += num_sites
        data_stats["num_leaves_range"][0] = min(
            data_stats["num_leaves_range"][0], np.min(num_leaves)
        )
        data_stats["num_leaves_range"][1] = max(
            data_stats["num_leaves_range"][1], np.max(num_leaves)
        )
        data_stats["num_sites_range"][0] = min(
            data_stats["num_sites_range"][0], np.min(num_sites)
        )
        data_stats["num_sites_range"][1] = max(
            data_stats["num_sites_range"][1], np.max(num_sites)
        )
        if take_first:
            break

    data_stats["num_trees"] = len(data_dict.items())
    data_stats["num_leaves"] = int(np.mean(data_stats["num_leaves"]))
    data_stats["num_sites"] = int(np.mean(data_stats["num_sites"]))
    return data_stats


def build_data_stats_dict(df, nicknames_dict, working_dir=".", take_first=True):
    """Build dict of number of trees, leaves, and sites for each dataset in DataFrame.

    Args:
        df: DataFrame with train_data and test_data columns.
        nicknames_dict: Dictionary mapping dataset nicknames to file paths.
        working_dir: Base directory for resolving relative paths.
        take_first: If True, use only first tree for stats (faster).

    Returns:
        Dictionary mapping dataset names to their stats dictionaries.
    """
    data_stats = {}
    data_names = list(set(df["train_data"].tolist() + df["test_data"].tolist()))

    for i, data_name in enumerate(data_names):
        if (i + 1) % 5 == 0:
            logger.info(f"Loading data stats: {i + 1}/{len(data_names)}")

        if "baseline" in data_name:
            data_path = None
        elif data_name in nicknames_dict:
            data_path = f"{working_dir}/{nicknames_dict[data_name]}"
        else:
            logger.warning(f"Dataset {data_name} not found in nicknames dict")
            data_path = None

        data_stats[data_name] = get_stats_from_data(data_path, take_first)

    return data_stats


def get_df_from_log(log_path):
    """Load a DataFrame from a TensorBoard log directory.

    Args:
        log_path: Path to the TensorBoard log directory.

    Returns:
        DataFrame with scalar values from the log.
    """
    if "cpu_" in log_path or "gpu_" in log_path:
        log_path = log_path.replace("cpu_", "").replace("gpu_", "")
    reader = tbparse.SummaryReader(log_path)
    return reader.scalars


# =============================================================================
# Dataset Name Parsing Functions
# =============================================================================


def extract_num_leaves(data_name):
    """Extract number of leaves from a dataset name, or None if not found."""
    match = re.search(r"simulated_(\d+)_seq", data_name)
    return match.group(1) if match else None


def extract_num_sites(data_name):
    """Extract number of sites from a dataset name, or None if not found."""
    match = re.search(r"_(\d+)_sites", data_name)
    return match.group(1) if match else None


def extract_num_trees(data_name):
    """Extract number of trees (alignments) from a dataset name, or None if not found."""
    match = re.search(r"_(\d+)_algnmnts", data_name)
    return match.group(1) if match else None


def extract_nonmp_fraction(data_name):
    """Extract non-MP edge fraction (e.g. '_t0.1') from a dataset name, or None if not found."""
    # Match pattern like _t0.1.p, _t0.1, or _t0.1_suffix
    match = re.search(r"_t(\d+\.?\d*)(?:\.p|_|$)", data_name)
    if match:
        return float(match.group(1))
    return None


def get_dataset_display_name(
    data_name, num_leaves=None, num_sites=None, num_trees=None, nonmp_fraction=None
):
    """Get a human-readable display name for a dataset.

    Parses dataset names and extracts metadata for display. Handles simulated,
    rotavirus, flu, orthomam, and pandit datasets.

    Args:
        data_name: Original dataset name or identifier.
        num_leaves: Optional number of leaves to include in label.
        num_sites: Optional number of sites to include in label.
        num_trees: Optional number of trees to include in label.
        nonmp_fraction: Optional non-MP edge fraction to include in label.

    Returns:
        Human-readable dataset label suitable for plotting.
    """
    # Match the longest key in DATASET_NAMES that appears in data_name
    name = data_name
    for key in sorted(DATASET_NAMES, key=len, reverse=True):
        if key in data_name:
            name = DATASET_NAMES[key]
            break

    # Add stats if provided
    parts = [name]
    if num_leaves is not None:
        parts.append(f"n={num_leaves}")
    if num_sites is not None:
        parts.append(f"N={num_sites}")
    if num_trees is not None:
        parts.append(f"T={num_trees}")
    if nonmp_fraction is not None and nonmp_fraction != "default":
        parts.append(f"t={nonmp_fraction}")

    return "\n".join(parts)


# =============================================================================
# Formatting Utilities
# =============================================================================


def plt_subplots(*args, **kwargs):
    """Create subplots ensuring axes is always iterable."""
    fig, axs = plt.subplots(*args, **kwargs)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    return fig, axs


def truncate_to_significant_digits(x, sig_digits):
    """Truncate a number x to the specified number of significant digits sig_digits."""
    if x == 0:
        return 0
    magnitude = np.floor(np.log10(abs(x)))
    factor = 10 ** (sig_digits - magnitude - 1)
    return np.floor(x * factor) / factor


def format_number(x, sig_digits=4, sci_range=(1e-6, 1e6)):
    """Format a number x with optional scientific notation."""
    x = truncate_to_significant_digits(x, sig_digits)
    if (x != 0) and (x < sci_range[0] or x > sci_range[1]):
        return f"{x:.{sig_digits}e}"
    return x


# =============================================================================
# Heatmap Helper Functions
# =============================================================================


def _determine_label_visibility(df_sorted, label_config):
    """Determine which labels should be shown on the heatmap.

    Auto-detects label visibility based on data variation, unless overridden
    by label_config settings. May modify df_sorted to fill non-MP fraction NaNs.

    Args:
        df_sorted: Sorted DataFrame with training/testing data columns.
        label_config: LabelConfig with optional overrides.

    Returns:
        Tuple of (df_sorted, flags) where flags is a dict of display booleans.
    """
    flags = {
        "train_leaves": _should_show_label(
            df_sorted["train_num_leaves"].tolist(), label_config.show_num_leaves
        ),
        "train_sites": _should_show_label(
            df_sorted["train_num_sites"].tolist(), label_config.show_num_sites
        ),
        "train_trees": _should_show_label(
            df_sorted["train_num_trees"].tolist(), label_config.show_num_trees
        ),
        "test_sites": _should_show_label(
            df_sorted["test_num_sites"].tolist(), label_config.show_num_sites
        ),
        "train_nonmp": False,
        "test_nonmp": False,
    }

    # Auto-detect non-MP fraction display.
    # Fill None with "default" so datasets without an explicit fraction
    # are distinguishable from those with one (e.g. _t0.1).
    if "train_nonmp_fraction" in df_sorted.columns:
        df_sorted = df_sorted.copy()
        df_sorted["train_nonmp_fraction"] = df_sorted["train_nonmp_fraction"].fillna(
            "default"
        )
        df_sorted["test_nonmp_fraction"] = df_sorted["test_nonmp_fraction"].fillna(
            "default"
        )
        flags["train_nonmp"] = _should_show_label(
            df_sorted["train_nonmp_fraction"].tolist(),
            label_config.show_nonmp_fraction,
        )
        flags["test_nonmp"] = _should_show_label(
            df_sorted["test_nonmp_fraction"].tolist(),
            label_config.show_nonmp_fraction,
        )

    # Check for multiple dataset sources in training data
    flags["mixed_train_sources"] = (
        sum(
            df_sorted["train_data"].str.contains(key, na=False).any()
            for key in DATASET_NAMES
        )
        >= 2
    )

    # Check for mixed perturbation methods in training data
    train_methods = [
        df_sorted["train_data"].str.contains("spr", na=False).any(),
        df_sorted["train_data"].str.contains("subtree", na=False).any(),
        df_sorted["train_data"].str.contains("uniform", na=False).any(),
    ]
    flags["mixed_training"] = _should_show_label(
        train_methods, label_config.show_perturbation
    )

    # Check for mixed perturbation methods in testing data
    test_methods = [
        df_sorted["test_data"].str.contains("spr", na=False).any(),
        df_sorted["test_data"].str.contains("subtree", na=False).any(),
        df_sorted["test_data"].str.contains("uniform", na=False).any(),
    ]
    flags["mixed_testing"] = _should_show_label(
        test_methods, label_config.show_perturbation
    )

    return df_sorted, flags


def _extract_perturbation_method(data_series):
    """Extract perturbation method labels from a dataset name series."""
    return np.select(
        [
            data_series.str.contains("spr_subtree", na=False),
            data_series.str.contains("spr", na=False),
            data_series.str.contains("subtree", na=False),
            data_series.str.contains("uniform", na=False),
            data_series.str.contains("treesearch", na=False),
        ],
        ["SPR+Subtree", "SPR", "Subtree", "uniform", "treesearch_mimic"],
        default="unknown",
    )


def _build_heatmap_columns(flags):
    """Build column lists for heatmap pivot table based on display flags.

    Args:
        flags: Dictionary of display flags from _determine_label_visibility.

    Returns:
        Tuple of (indices, test_cols) column name lists.
    """
    indices = ["model"]
    extra_cols = []
    if flags["mixed_train_sources"]:
        extra_cols.append("train_data_name")
    if flags["train_leaves"]:
        extra_cols.append("train_num_leaves")
    if flags["train_sites"]:
        extra_cols.append("train_num_sites")
    if flags["train_trees"]:
        extra_cols.append("train_num_trees")
    if flags["train_nonmp"]:
        extra_cols.append("train_nonmp_fraction")

    if flags["mixed_training"]:
        extra_cols = ["train_perturbation"] + extra_cols
    indices.extend(extra_cols)

    test_cols = []
    if flags["mixed_testing"]:
        test_cols.append("test_perturbation")
    test_cols.append("test_num_leaves")
    if flags["test_sites"]:
        test_cols.append("test_num_sites")
    if flags["test_nonmp"]:
        test_cols.append("test_nonmp_fraction")

    return indices, test_cols


def _create_heatmap_pivot(df_sorted, indices, test_cols, value_column):
    """Create pivot table for heatmap visualization.

    Handles type coercion, multi-source dataset display names, and NaN cleanup.

    Args:
        df_sorted: Sorted DataFrame with data.
        indices: Row index columns for the pivot table.
        test_cols: Column index columns for the pivot table.
        value_column: Column name for the cell values.

    Returns:
        Pivot table DataFrame ready for heatmap rendering.
    """
    df_for_pivot = df_sorted.copy()
    for col in indices + test_cols:
        if isinstance(df_for_pivot[col].dtype, pd.CategoricalDtype):
            df_for_pivot[col] = df_for_pivot[col].astype(str)
        if (
            pd.api.types.is_integer_dtype(df_for_pivot[col])
            and pd.isna(df_for_pivot[col]).any()
        ):
            df_for_pivot[col] = df_for_pivot[col].astype(str)
        df_for_pivot[col] = df_for_pivot[col].fillna("N/A")

    # Use display names when multiple data sources are present
    matched_sources = sum(
        df_for_pivot["test_data"].str.contains(key, na=False).any()
        for key in DATASET_NAMES
    )
    if matched_sources >= 2:
        df_for_pivot["test_data_name"] = df_for_pivot.apply(
            lambda row: get_dataset_display_name(
                row["test_data"],
                row["test_num_leaves"],
                row["test_num_sites"],
                row["test_num_trees"],
                row.get("test_nonmp_fraction"),
            ),
            axis=1,
        )
        test_cols = ["test_data_name"]

    heatmap_data = df_for_pivot.pivot_table(
        index=indices,
        columns=test_cols,
        values=value_column,
        dropna=False,
    )

    # Drop rows/columns with all NaN
    heatmap_data = heatmap_data[~heatmap_data.isna().all(axis=1)]
    heatmap_data = heatmap_data.dropna(axis=1, how="all")
    return heatmap_data


def _build_labels_from_tuples(items, prefixes, start_offset=0):
    """Build formatted labels from a sequence of tuples or scalars.

    Args:
        items: Sequence of tuples or scalars to label.
        prefixes: List of string prefixes per position. Empty string means
                  show the value as-is (no prefix).
        start_offset: Starting index into each tuple.

    Returns:
        List of newline-joined label strings.
    """
    labels = []
    for item in items:
        tup = item if isinstance(item, tuple) else (item,)
        parts = []
        for i, prefix in enumerate(prefixes):
            val = tup[start_offset + i] if start_offset + i < len(tup) else ""
            parts.append(f"{prefix}{val}" if prefix else str(val))
        labels.append("\n".join(parts))
    return labels


def _build_x_labels(heatmap_data, flags):
    """Build formatted x-axis labels for heatmap columns.

    Returns columns as-is when they are already display names (multi-source case).
    Otherwise builds prefixes for perturbation method, n=, N=, and t= columns
    based on active flags.

    Args:
        heatmap_data: Pivot table with potentially multi-level columns.
        flags: Display flags dictionary.

    Returns:
        List of formatted label strings for each column.
    """
    if (
        not isinstance(heatmap_data.columns, pd.MultiIndex)
        and heatmap_data.columns.name == "test_data_name"
    ):
        return list(heatmap_data.columns)

    prefixes = []
    if flags["mixed_testing"]:
        prefixes.append("")
    prefixes.append("n=")
    if flags["test_sites"]:
        prefixes.append("N=")
    if flags["test_nonmp"]:
        prefixes.append("t=")

    return _build_labels_from_tuples(heatmap_data.columns, prefixes)


def _build_secondary_y_labels(heatmap_data, flags):
    """Build secondary y-axis labels for multiindex heatmap rows.

    Each row label shows the training data attributes (leaves, sites, trees, etc.)
    that vary across the datasets.

    Args:
        heatmap_data: Pivot table with potentially multi-level index.
        flags: Display flags dictionary.

    Returns:
        List of label strings for each row, or True for default labels.
    """
    if not isinstance(heatmap_data.index, pd.MultiIndex):
        return True

    prefixes = []
    if flags["mixed_training"]:
        prefixes.append("")
    if flags["mixed_train_sources"]:
        prefixes.append("")
    if flags["train_leaves"]:
        prefixes.append("n=")
    if flags["train_sites"]:
        prefixes.append("N=")
    if flags["train_trees"]:
        prefixes.append("T=")
    if flags["train_nonmp"]:
        prefixes.append("t=")

    return _build_labels_from_tuples(heatmap_data.index, prefixes, start_offset=1)


def _render_heatmap_layout(fig, ax, heatmap_data, flags, title):
    """Position model labels and format heatmap axes for multiindex data.

    Adds model name labels on the left, horizontal dividers between models,
    baseline label handling, and descriptive axis labels.

    Args:
        fig: Matplotlib figure.
        ax: Matplotlib axes with the rendered heatmap.
        heatmap_data: Pivot table used for the heatmap.
        flags: Display flags dictionary.
        title: Title for the heatmap.
    """
    if not isinstance(heatmap_data.index, pd.MultiIndex):
        return

    # Find row spans for each model and draw dividers
    models = heatmap_data.index.get_level_values(0).unique()
    model_rows = {}
    current_model = None
    start_idx = 0

    for i, idx in enumerate(heatmap_data.index):
        model = idx[0]
        if current_model != model:
            if current_model is not None:
                model_rows[current_model] = (start_idx, i - 1)
                ax.axhline(y=i, color="black", linewidth=1)
            current_model = model
            start_idx = i

    if current_model is not None:
        model_rows[current_model] = (start_idx, len(heatmap_data.index) - 1)

    # Position model name labels
    bbox = ax.get_position()
    axis_left = bbox.x0
    axis_height = bbox.height
    axis_bottom = bbox.y0

    baseline_offset = -0.5 if "BaselineReversion" in model_rows else 0
    ylabel_shift = YLABEL_BASE_OFFSET
    if flags["mixed_training"]:
        ylabel_shift += YLABEL_MIXED_SOURCE_ADJUSTMENT

    max_end_idx = max(end for start, end in model_rows.values())

    for model, (start, end) in model_rows.items():
        if "Baseline" in model:
            continue

        display_name = MODEL_NAMES.get(model, model)
        center_row = max_end_idx - ((start + end) / 2 + baseline_offset)
        fig_y_pos = axis_bottom + ((center_row / len(heatmap_data.index)) * axis_height)

        fig.text(
            axis_left - ylabel_shift + 0.14,
            fig_y_pos,
            display_name,
            va="center",
            ha="center",
            fontsize=FONT_LARGE,
        )

    # Add "Trained model" y-axis label
    num_displayed = sum(
        [flags["train_leaves"], flags["train_sites"], flags["train_trees"]]
    )
    if num_displayed <= 1:
        ylabel_shift -= 0.05
    ylabel = "Trained model\n"
    ylabel_add = []
    if flags["train_leaves"]:
        ylabel_add.append("n: number of leaves")
    if flags["train_sites"]:
        ylabel_add.append("N: number of sites")
    if flags["train_trees"]:
        ylabel_add.append("T: number of trees")
    ylabel = ylabel + ','.join(ylabel_add)

    fig.text(
        axis_left - ylabel_shift + 0.01,
        axis_bottom + (axis_height / 2),
        ylabel,
        va="center",
        ha="center",
        rotation=90,
        fontsize=FONT_LARGE,
    )

    # Handle baseline model label
    if any("Baseline" in model for model in models):
        yticks = plt.yticks()
        positions = yticks[0]
        labels = [label.get_text() for label in plt.gca().get_yticklabels()]
        baseline_label = ax.get_yticklabels()[0]
        baseline_label.set_fontsize(FONT_LARGE)
        for model, (start, _) in model_rows.items():
            if "Baseline" in model and start < len(labels):
                labels[start] = "BaselineReversion"
        plt.yticks(positions, labels)

    plt.yticks(rotation=0, fontsize=FONT_LARGE)
    plt.xticks(rotation=0, fontsize=FONT_LARGE)
    plt.subplots_adjust(left=0.3)

    # Build descriptive x-axis label
    xlabel = "Testing data\n n: number of leaves"
    if flags["test_sites"]:
        xlabel = "Testing data\n n: avg number of leaves, N: avg number of sites, T: number of trees"
    if flags["mixed_testing"]:
        if flags["test_sites"]:
            xlabel = "Testing data: perturbation method - number of leaves - number of sites - number of trees"
        else:
            xlabel = "Testing data: perturbation method - number of leaves"

    fig.text(
        axis_left + (bbox.width / 2) + 0.15,
        axis_bottom - 0.3,
        xlabel,
        va="center",
        ha="center",
        fontsize=FONT_LARGE,
    )
    ax.set_xlabel("")
    ax.set_title(title)
    plt.title("")


# =============================================================================
# Heatmap Plotting
# =============================================================================


def build_performance_heatmap(
    df,
    value_column,
    output_path,
    title="",
    v_range=(0.0, 1.0),
    label_config: LabelConfig | None = None,
):
    """Build a performance heatmap from summary DataFrame.

    Creates a heatmap showing model performance (AUROC, accuracy, or loss)
    across different training and testing dataset configurations.

    Args:
        df: DataFrame with columns: model, train_data, test_data, train_num_leaves,
            train_num_sites, train_num_trees, test_num_leaves, test_num_sites,
            test_num_trees, and the value_column.
        value_column: Column name for the values to display (e.g., "test_auroc").
        output_path: Path to save the output PDF.
        title: Optional title for the plot.
        v_range: Tuple (min, max) for the color scale.
        label_config: Configuration for which labels to show. If None, uses
            auto-detection (only shows labels that vary across datasets).
    """
    if label_config is None:
        label_config = LabelConfig()

    df_sorted = df.sort_values(by=["model", "train_num_leaves", "train_num_sites"])
    df_sorted, flags = _determine_label_visibility(df_sorted, label_config)

    # Add display name column for training data sources
    if flags["mixed_train_sources"]:
        df_sorted["train_data_name"] = df_sorted["train_data"].apply(
            lambda name: get_dataset_display_name(name)
        )

    # Add perturbation columns
    if flags["mixed_testing"]:
        df_sorted["test_perturbation"] = _extract_perturbation_method(
            df_sorted["test_data"]
        )
    else:
        df_sorted["test_perturbation"] = ""

    if flags["mixed_training"]:
        df_sorted["train_perturbation"] = _extract_perturbation_method(
            df_sorted["train_data"]
        )

    indices, test_cols = _build_heatmap_columns(flags)
    heatmap_data = _create_heatmap_pivot(df_sorted, indices, test_cols, value_column)
    secondary_labels = _build_secondary_y_labels(heatmap_data, flags)
    x_labels = _build_x_labels(heatmap_data, flags)

    # Render heatmap
    num_y_labels = sum(
        [
            flags["train_leaves"],
            flags["train_sites"],
            flags["train_trees"],
            flags["train_nonmp"],
            flags["mixed_training"],
            flags["mixed_train_sources"],
        ]
    )
    row_height = HEATMAP_ROW_HEIGHT + num_y_labels * HEATMAP_ROW_HEIGHT_PER_LABEL
    fig_width = (len(heatmap_data.columns) * HEATMAP_COL_WIDTH) + HEATMAP_BASE_WIDTH
    fig_height = (len(heatmap_data) * row_height) + HEATMAP_BASE_HEIGHT
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = sns.color_palette("Purples", as_cmap=True)
    sns_heatmap = sns.heatmap(
        data=heatmap_data,
        yticklabels=secondary_labels,
        xticklabels=x_labels,
        annot=True,
        annot_kws={"fontsize": FONT_LARGE},
        cbar_kws={"label": value_column, "shrink": 0.8},
        vmin=v_range[0],
        vmax=v_range[1],
        cmap=cmap,
        fmt=".2f",
        ax=ax,
    )
    ax.set_ylabel("")
    plt.xticks(rotation=0, fontsize=FONT_MED)
    cbar = sns_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONT_MED)
    cbar.set_label(METRIC_LABELS.get(value_column, value_column), fontsize=FONT_MED)

    plt.tight_layout()
    _render_heatmap_layout(fig, ax, heatmap_data, flags, title)

    logger.info(f"Saving heatmap to: {output_path}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Bar and Line Plot Functions
# =============================================================================


def plot_hyperparameters_summary(df, output_path, hyperparams=None):
    """Create a bar plot summarizing hyperparameters across models.

    Args:
        df: DataFrame with model_and_train_data column and hyperparameter columns.
        output_path: Path to save the output PDF.
        hyperparams: List of hyperparameter column names. Defaults to common ones.
    """
    if hyperparams is None:
        hyperparams = [
            "learning_rate",
            "batch_size",
            "accum_grad_batches",
            "max_epochs",
            "feature_length",
            "dim_mlp_layers",
        ]

    if len(df["model_and_train_data"].unique()) >= 10:
        logger.info("Too many trained models, skipping hyperparameter summary plot")
        return

    num_params = len(hyperparams)
    train_df = df.groupby("model_and_train_data")[hyperparams].first().reset_index()
    melt_df = pd.melt(
        train_df, id_vars="model_and_train_data", var_name="Metric", value_name="Value"
    )
    melt_df = melt_df[melt_df.Metric.isin(hyperparams)]

    plt.figure(figsize=(3 * num_params, 8))
    sns.barplot(
        x="Metric", y="Value", hue="model_and_train_data", data=melt_df, palette="deep"
    )
    plt.xticks(rotation=45)
    plt.title("Hyper Parameters by Model")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved hyperparameter summary to: {output_path}")


def plot_training_runtimes_by_leaves(df, output_dir):
    """Create bar plots of training runtimes grouped by number of leaves.

    Args:
        df: DataFrame with train_data, train_num_leaves, train_walltime, model columns.
        output_dir: Directory to save output PDFs.
    """
    for num_leaves in df["train_num_leaves"].unique():
        fig, ax = plt.subplots()

        leaf_pattern = f"{num_leaves}_seqs|{num_leaves}_leaves|{num_leaves}leaves|{num_leaves}seqs|{num_leaves}leaf|{num_leaves}seq"
        this_df = df[df["train_data"].str.contains(leaf_pattern, na=False)].copy()
        this_df.sort_values(by="train_num_sites", inplace=True)

        if len(this_df) > 0:
            sns.barplot(
                y="train_data",
                x="train_walltime",
                data=this_df,
                palette="deep",
                hue="model",
                errorbar=None,
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_ylabel("Train Data")
            ax.set_xlabel("Training Runtime (in secs)")

            plt.tight_layout()
            output_path = f"{output_dir}/train_walltime_{num_leaves}seq.barplot.pdf"
            plt.savefig(output_path)
            logger.info(f"Saved runtime plot to: {output_path}")
            plt.close(fig)


def plot_metric_by_model(df, metric_column, output_path, model_list):
    """Create bar plots of a metric split by model.

    Args:
        df: DataFrame with model, train_data, and metric columns.
        metric_column: Name of the metric column to plot.
        output_path: Path to save the output PDF.
        model_list: List of model names to include.
    """
    fig, axes = plt_subplots(
        nrows=1, ncols=len(model_list), figsize=(15, 6), sharey=True
    )

    x_pad_factor = 0.05
    x_pad = abs(df[metric_column].min() - df[metric_column].max()) * x_pad_factor
    x_min = df[metric_column].min() - x_pad
    x_max = df[metric_column].max() + x_pad

    for ax, model in zip(axes, model_list):
        this_df = df[df["model"] == model]
        sns.barplot(
            y="train_data",
            x=metric_column,
            data=this_df,
            palette="deep",
            hue="train_data",
            ax=ax,
        )
        ax.set_ylabel("Train Data")
        ax.set_xlabel(metric_column)
        ax.set_title(model)
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format_number(x))
        )
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved metric plot to: {output_path}")


def plot_training_dynamics(custom_dfs, output_dir, model_list, measures=None):
    """Create line plots of training dynamics over epochs/batches.

    Args:
        custom_dfs: Dictionary mapping (model, train_data, test_data, param_id) to DataFrames.
        output_dir: Directory to save output PDFs.
        model_list: List of model names to include.
        measures: Dictionary of {title: column_name} for metrics to plot.
    """
    if measures is None:
        measures = {
            "Runtime over Batches": "walltime_per_batch",
            "Loss over Batches": "loss_per_batch",
            "Runtime over Epochs": "walltime_per_epoch",
            "Loss over Epochs": "avgloss_per_epoch",
        }

    df_list = []
    for key, df in custom_dfs.items():
        this_df = df.copy()
        this_df["model_and_train_data"] = key[0] + "-" + key[1]
        this_df["model"] = key[0]
        df_list.append(this_df)

    runtime_df = pd.concat(df_list)

    for title, col in measures.items():
        fig, axes = plt_subplots(
            nrows=1, ncols=len(model_list), figsize=(15, 6), sharey=True
        )
        this_tag_df = runtime_df[runtime_df.tag == col]
        x_max = this_tag_df["step"].max() + 2

        handles, labels = None, None
        for ax, model in zip(axes, model_list):
            this_df = this_tag_df[this_tag_df["model"] == model]
            plot = sns.lineplot(
                y="value",
                x="step",
                data=this_df,
                hue="model_and_train_data",
                palette="deep",
                ax=ax,
                legend=True,
            )
            ax.set_xlabel("Training Step")
            ax.set_ylabel(title)
            ax.set_title(model)
            ax.set_xlim(0, x_max)
            plt.tight_layout()
            handles, labels = plot.get_legend_handles_labels()
            ax.get_legend().remove()

        if handles and labels:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        output_path = f"{output_dir}/{col}.barplot.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved training dynamics plot to: {output_path}")


def plot_loss_over_time(custom_dfs, output_path, model_list):
    """Create line plots of loss over walltime.

    Args:
        custom_dfs: Dictionary mapping (model, train_data, test_data, param_id) to DataFrames.
        output_path: Path to save the output PDF.
        model_list: List of model names to include.
    """
    fig, axes = plt_subplots(
        nrows=1, ncols=len(model_list), figsize=(15, 6), sharey=True
    )
    model_list = list(model_list)
    label_set = set()
    handles, labels = None, None

    for key, df in custom_dfs.items():
        label = key[0] + "-" + key[1]
        if label in label_set:
            continue
        label_set.add(label)

        df_time = df[df.tag == "walltime_per_batch"]
        df_loss = df[df.tag == "loss_per_batch"]
        ax = axes[model_list.index(key[0])]
        ax.plot(df_time.value, df_loss.value, label=key[1])
        ax.set_title(key[0])

    fig.supylabel("Training loss", fontsize=FONT_SMALL)
    fig.supxlabel("Walltime (in seconds)", fontsize=FONT_SMALL)

    plt.xticks(rotation=45)

    if handles and labels:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved loss over time plot to: {output_path}")


_MISSING_PATH_SENTINEL = "none"


def _find_pr_curve_pdf(test_llog_path):
    """Find the pr_curve.pdf saved by the model in the TensorBoard log directory.

    Args:
        test_llog_path: Path to the test TensorBoard log directory. A value of
            None or the sentinel string "none" (used for baseline rows) returns None.

    Returns:
        Path to pr_curve.pdf, or None if not found.
    """
    if test_llog_path is None or str(test_llog_path) == _MISSING_PATH_SENTINEL:
        return None
    pdf_files = sorted(Path(str(test_llog_path)).glob("*/pr_curve.pdf"))
    # Use the last sorted match: Lightning auto-increments version_N dirs on each run.
    if not pdf_files:
        logger.debug(f"No pr_curve.pdf found under {test_llog_path}.")
        return None
    return pdf_files[-1]


def _find_pr_curve_csv(test_llog_path):
    """Find the pr_curve.csv saved by the model in the TensorBoard log directory.

    Args:
        test_llog_path: Path to the test TensorBoard log directory. A value of
            None or the sentinel string "none" (used for baseline rows) returns None.

    Returns:
        Path to pr_curve.csv, or None if not found.
    """
    if test_llog_path is None or str(test_llog_path) == _MISSING_PATH_SENTINEL:
        return None
    csv_files = sorted(Path(str(test_llog_path)).glob("*/pr_curve.csv"))
    # Use the last sorted match: Lightning auto-increments version_N dirs on each run.
    if not csv_files:
        logger.debug(f"No pr_curve.csv found under {test_llog_path}.")
        return None
    return csv_files[-1]


def plot_precision_recall_curves(
    grid, row_labels, col_labels, output_path, title="Precision-Recall Curves"
):
    """Plot a grid of precision-recall curves using seaborn.

    Lays out PR curves so that each column corresponds to a model and each row
    to a training configuration (e.g. number of leaves). Missing combinations
    are left blank.

    Args:
        grid: Dictionary mapping (row_label, col_label) -> DataFrame with
            "recall" and "precision" columns.
        row_labels: Ordered list of row labels shown on the left of each row.
        col_labels: Ordered list of column labels shown above each column.
        output_path: Path to save the output PDF.
        title: Title for the overall figure.
    """
    nrows = len(row_labels)
    ncols = len(col_labels)
    if nrows == 0 or ncols == 0:
        logger.warning("No PR curve data found, skipping plot.")
        return
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )

    for r, row_label in enumerate(row_labels):
        for c, col_label in enumerate(col_labels):
            ax = axes[r][c]
            df = grid.get((row_label, col_label))
            if df is not None and not df.empty and "avg_precision" in df.columns:
                ap = df["avg_precision"].iloc[0]
                sns.lineplot(
                    data=df, x="recall", y="precision", ax=ax, label=f"AP={ap:.2f}"
                )
                ax.legend(loc="lower left", fontsize=FONT_LARGE)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(col_label, fontsize=FONT_LARGE)
            ax.set_ylabel(
                f"{row_label}\nPrecision" if c == 0 else "", fontsize=FONT_LARGE
            )
            ax.set_xlabel("Recall" if r == nrows - 1 else "", fontsize=FONT_LARGE)

    fig.suptitle(title, fontsize=FONT_LARGE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved precision-recall curves to: {output_path}")


# =============================================================================
# Public API
# =============================================================================


def _add_data_stats_columns(
    summary_df, get_data_stats_from_name, nicknames_dict, working_dir
):
    """Add num_leaves, num_sites, num_trees, and non-MP fraction columns.

    Augments the summary DataFrame with dataset statistics either extracted
    from dataset names or loaded from pickle files.

    Args:
        summary_df: Summary DataFrame to augment (modified in place).
        get_data_stats_from_name: If True, parse stats from dataset name strings.
            Otherwise, load from pickle files via nicknames_dict.
        nicknames_dict: Dictionary mapping dataset nicknames to file paths.
        working_dir: Base directory for resolving relative paths.
    """
    extractors = {
        "num_leaves": extract_num_leaves,
        "num_sites": extract_num_sites,
        "num_trees": extract_num_trees,
    }

    if get_data_stats_from_name:
        for prefix in ("train", "test"):
            for stat, extractor in extractors.items():
                summary_df[f"{prefix}_{stat}"] = pd.to_numeric(
                    summary_df[f"{prefix}_data"].apply(extractor), errors="coerce"
                )
    else:
        data_stats = build_data_stats_dict(
            summary_df, nicknames_dict, working_dir, take_first=False
        )
        for prefix in ("train", "test"):
            for stat in extractors:
                summary_df[f"{prefix}_{stat}"] = [
                    data_stats[x][stat] for x in summary_df[f"{prefix}_data"]
                ]

    # Extract non-MP fraction from dataset names (always from name, not pickle)
    for prefix in ("train", "test"):
        summary_df[f"{prefix}_nonmp_fraction"] = summary_df[f"{prefix}_data"].apply(
            extract_nonmp_fraction
        )

    int_columns = [
        f"{prefix}_{stat}"
        for prefix in ("train", "test")
        for stat in ("num_leaves", "num_sites", "num_trees")
    ]
    summary_df[int_columns] = summary_df[int_columns].astype("Int64")


def generate_summary_plots(
    summary_csv_path,
    results_dir,
    config_path=None,
    data_nicknames_path=None,
    baseline_csv_path=None,
    working_dir=".",
    get_data_stats_from_name=False,
    plot_details=False,
    label_config: LabelConfig | None = None,
):
    """Generate all summary plots from a training run.

    Main entry point for Snakemake workflow to generate plots from summary CSV.

    Args:
        summary_csv_path: Path to the summary CSV file.
        results_dir: Directory to save output plots.
        config_path: Optional path to config YAML file.
        data_nicknames_path: Optional path to data nicknames JSON file.
        baseline_csv_path: Optional path to baseline summary CSV.
        working_dir: Base directory for resolving relative paths.
        get_data_stats_from_name: If True, extract stats from dataset names.
        plot_details: If True, generate additional detailed plots.
        label_config: Configuration for which labels to show on heatmaps.
            If None, uses auto-detection (only shows labels that vary).

    Returns:
        Dictionary with paths to generated plots.
    """
    from dpvtex.dpvt_data import load_nicknames_dict

    # Create output directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load summary data
    summary_df = pd.read_csv(summary_csv_path)

    # Load baseline if available
    if baseline_csv_path and Path(baseline_csv_path).exists():
        baseline_summary_df = pd.read_csv(baseline_csv_path)
        summary_df = pd.concat([summary_df, baseline_summary_df], ignore_index=True)

    # Load config and nicknames
    config_data = None
    if config_path and Path(config_path).exists():
        config_data = load_data(config_path)

    nicknames_dict = {}
    if data_nicknames_path and Path(data_nicknames_path).exists():
        nicknames_dict = load_nicknames_dict(data_nicknames_path)

    # Add summary columns
    summary_df["label"] = summary_df[
        ["model", "train_data", "test_data", "param_id"]
    ].apply(lambda x: "\n".join(x.astype(str)), axis=1)
    summary_df["percent_epochs"] = summary_df["train_epochs"] / summary_df["max_epochs"]
    summary_df["model_and_train_data"] = (
        summary_df["model"].astype(str) + "-" + summary_df["train_data"].astype(str)
    )

    # Add dataset statistics and non-MP fraction columns
    _add_data_stats_columns(
        summary_df, get_data_stats_from_name, nicknames_dict, working_dir
    )

    # Set model order
    summary_df["model"] = pd.Categorical(
        summary_df["model"], categories=MODEL_ORDER, ordered=True
    )
    model_list = summary_df["model"].unique()

    generated_plots = {}

    # Generate heatmaps
    heatmap_settings = {
        "test_auroc": {"title": "Test AUROC", "v_range": (0.0, 1.0)},
        "test_loss": {"title": "Test Loss", "v_range": (0.0, None)},
        "test_accuracy": {"title": "Test accuracy", "v_range": (0.0, 1.0)},
    }

    for value_name, settings in heatmap_settings.items():
        if value_name not in summary_df.columns:
            continue
        output_path = f"{results_dir}/{value_name}_heatmap.pdf"
        build_performance_heatmap(
            df=summary_df,
            value_column=value_name,
            output_path=output_path,
            title=settings["title"],
            v_range=settings["v_range"],
            label_config=label_config,
        )
        generated_plots[value_name] = output_path

    # Generate hyperparameter summary
    hyperparam_path = f"{results_dir}/hyperparameter_summary.barplot.pdf"
    plot_hyperparameters_summary(summary_df, hyperparam_path)
    generated_plots["hyperparameters"] = hyperparam_path

    # Generate runtime plots by leaves
    plot_training_runtimes_by_leaves(summary_df, results_dir)

    # Generate detailed plots if requested
    if plot_details:
        # Load log data
        custom_dfs = {}
        for _, row in summary_df.iterrows():
            if "baseline" in row.train_data:
                continue
            label = (row.model, row.train_data, row.test_data, row.param_id)
            if hasattr(row, "train_clog_path") and row.train_clog_path:
                custom_dfs[label] = get_df_from_log(row.train_clog_path)

        if custom_dfs:
            plot_training_dynamics(custom_dfs, results_dir, model_list)
            loss_path = f"{results_dir}/loss_per_walltime.lineplot.pdf"
            plot_loss_over_time(custom_dfs, loss_path, model_list)
            generated_plots["loss_over_time"] = loss_path

        # Generate metric bar plots
        x_measures = {
            "Training Runtime (in secs)": "train_walltime",
            "Learning Rate": "learning_rate",
            "Batch Size": "batch_size",
            "Gradient Accumulation": "accum_grad_batches",
            "Max Epochs": "max_epochs",
            "Train Epochs": "train_epochs",
            "Train Steps": "train_steps",
            "Percent of Max Epochs Used": "percent_epochs",
        }

        for x_title, x_col in x_measures.items():
            if x_col not in summary_df.columns:
                continue
            output_path = f"{results_dir}/{x_col}.barplot.pdf"
            plot_metric_by_model(summary_df, x_col, output_path, model_list)
            generated_plots[x_col] = output_path

    # Generate PR curve plots (one per test dataset)
    if "test_llog_path" in summary_df.columns:
        for test_data_name in summary_df["test_data"].unique():
            safe_test = re.sub(r"[^\w.-]", "_", str(test_data_name))
            test_df = summary_df[summary_df["test_data"] == test_data_name]
            # Grid keyed by (leaf-count row label, model column label) so the same
            # model lands in the same column and the same leaf count in the same row.
            grid = {}
            models_seen = {}  # model_name -> column label (display name)
            rows_seen = {}  # leaf count (None last) -> row label
            for _, row in test_df.iterrows():
                model_name = str(row["model"])
                train_name = str(row["train_data"])
                # Copy individual PDF to results_dir if available
                pdf_path = _find_pr_curve_pdf(row.get("test_llog_path"))
                if pdf_path is not None:
                    dest = (
                        Path(results_dir)
                        / f"pr_curve_{model_name}_{train_name}_ON_{safe_test}.pdf"
                    )
                    try:
                        shutil.copy2(pdf_path, dest)
                        generated_plots[dest.stem] = str(dest)
                    except OSError as e:
                        logger.warning(
                            f"Failed to copy PR curve PDF {pdf_path} to {dest}: {e}"
                        )
                # Load CSV for combined seaborn grid
                csv_path = _find_pr_curve_csv(row.get("test_llog_path"))
                if csv_path is None:
                    continue
                try:
                    pr_df = pd.read_csv(csv_path)
                except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
                    logger.warning(f"Failed to read PR curve CSV {csv_path}: {e}")
                    continue
                if pr_df.empty:
                    logger.warning(f"PR curve CSV has no data rows: {csv_path}")
                    continue
                model_display = MODEL_NAMES.get(model_name, model_name)
                train_leaves = row.get("train_num_leaves")
                leaves_val = None if pd.isna(train_leaves) else int(train_leaves)
                dataset_display = get_dataset_display_name(train_name)
                row_label = (
                    f"{dataset_display}\nn={leaves_val}"
                    if leaves_val is not None
                    else dataset_display
                )
                models_seen[model_name] = model_display
                rows_seen[(dataset_display, leaves_val)] = row_label
                grid[(row_label, model_display)] = pr_df
            if grid:
                # Columns ordered by MODEL_ORDER (unknown models appended),
                # rows by ascending leaf count (unknown/None last).
                ordered_models = [m for m in MODEL_ORDER if m in models_seen]
                ordered_models += [m for m in models_seen if m not in MODEL_ORDER]
                col_labels = [models_seen[m] for m in ordered_models]
                row_labels = [
                    rows_seen[k]
                    for k in sorted(rows_seen, key=lambda v: (v[0], v[1] is None, v[1]))
                ]
                pr_output_path = f"{results_dir}/pr_curves_{safe_test}.pdf"
                plot_precision_recall_curves(
                    grid,
                    row_labels,
                    col_labels,
                    pr_output_path,
                    title=f"Precision-Recall: {get_dataset_display_name(str(test_data_name))}",
                )
                generated_plots[f"pr_curves_{safe_test}"] = pr_output_path

    logger.info(f"Generated {len(generated_plots)} plots in {results_dir}")
    return generated_plots
