"""Plotting functions for treesearch evaluation results."""

import os
import pickle

import historydag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO

from dpvtex.dpvt_data import load_nicknames_dict
from dpvtex.evaluate_individual_trees import extract_alignment_base
from dpvtex.plotting import extract_nonmp_fraction


# =============================================================================
# Statistical / Interpolation Helpers
# =============================================================================
def interpolate_to_common_grid(values_list, max_length):
    """
    Interpolate multiple value arrays to a common normalized grid.

    Args:
        values_list: List of arrays, each containing values from one replicate.
        max_length: Length of the common grid to interpolate onto.

    Returns:
        tuple: (common_x, interpolated_array) where common_x is the normalized
               x-axis [0, 1] and interpolated_array is a 2D numpy array with
               each row being one interpolated replicate.
    """
    common_x = np.linspace(0, 1, max_length)
    interpolated = []

    for values in values_list:
        if len(values) == 0:
            continue
        dataset_x = np.linspace(0, 1, len(values))
        interp_values = np.interp(
            common_x, dataset_x, values, left=np.nan, right=np.nan
        )
        interpolated.append(interp_values)

    if not interpolated:
        return common_x, np.array([])

    return common_x, np.array(interpolated)


def calculate_percentile_bands(values_array, percentiles):
    """
    Calculate median and percentile bands from an array of replicate values.

    Args:
        values_array: 2D numpy array with replicates as rows.
        percentiles: List of [lower, upper] percentile values (e.g., [2.5, 97.5]).

    Returns:
        tuple: (median, lower_band, upper_band) arrays.
    """
    median = np.nanmedian(values_array, axis=0)
    lower = np.nanpercentile(values_array, percentiles[0], axis=0)
    upper = np.nanpercentile(values_array, percentiles[1], axis=0)
    return median, lower, upper


def get_parsimony_scores(tree_list, fasta_path):
    """Calculate parsimony scores for a list of trees using sequence data."""
    pscore_list = []
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    for tree in tree_list:
        for node in tree.get_leaves():
            node.add_feature("sequence", sequences[node.name])
        historydag.parsimony.disambiguate(tree)
        pscore = historydag.parsimony.parsimony_score(tree)
        pscore_list.append(pscore)
    return pscore_list


# =============================================================================
# Plotting Constants
# =============================================================================
FIGURE_WIDTH = 12
FIGURE_HEIGHT_PER_ROW = 5
FONT_SIZE_LABEL = 14
FONT_SIZE_TITLE = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 14
LINE_WIDTH_MAIN = 2.0
LINE_WIDTH_SECONDARY = 1.5
LINE_ALPHA = 0.9
FILL_ALPHA = 0.2
GRID_ALPHA = 0.7
MARKER_SIZE = 3

METRIC_LABELS = {
    "auroc": "AUROC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "tp": "True Positives",
    "fp": "False Positives",
    "tn": "True Negatives",
    "fn": "False Negatives",
}

NONMP_LINESTYLES = ["-", "--", ":", "-."]


def _nonmp_fraction_key(name):
    """Return a string key for the non-MP fraction in a dataset name."""
    frac = extract_nonmp_fraction(name)
    return str(frac) if frac is not None else "default"


def _collect_parsimony_and_nonmp_data(test_data_names, dataset_dict, fasta_dir):
    """Collect parsimony scores and fraction of non-MP edges for all test datasets.

    Args:
        test_data_names: List of test dataset nicknames.
        dataset_dict: Dictionary mapping nicknames to file paths.
        fasta_dir: Directory containing FASTA files.

    Returns:
        tuple: (all_parsimony_scores, all_frac_non_mp_edges, max_length)
    """
    all_parsimony_scores = []
    all_frac_non_mp_edges = []
    max_length = 0

    for test_data in test_data_names:
        test_data_path = dataset_dict[test_data]
        with open(test_data_path, "rb") as f:
            data_dict = pickle.load(f)

        # Extract alignment name from path
        filename = os.path.basename(dataset_dict[test_data])
        data_basename = extract_alignment_base(filename)
        fasta_path = os.path.join(fasta_dir, data_basename, "input.fasta")

        parsimony_scores = get_parsimony_scores(list(data_dict.keys()), fasta_path)
        num_int_edges = len(list(data_dict.keys())[0]) - 2
        frac_non_mp_edges = [
            sum(labels) / num_int_edges for labels in data_dict.values()
        ]

        max_length = max(max_length, len(parsimony_scores))
        all_parsimony_scores.append(parsimony_scores)
        all_frac_non_mp_edges.append(frac_non_mp_edges)

    return all_parsimony_scores, all_frac_non_mp_edges, max_length


def _filter_comparison_data(
    csv_file,
    test_data_names,
    compare_by,
    fixed_model,
    fixed_training_data,
    include_baseline,
):
    """Load and filter the evaluation data based on comparison type.

    Args:
        csv_file: Path to CSV file with evaluation data.
        test_data_names: List of test dataset nicknames.
        compare_by: "model" or "training_data".
        fixed_model: Model to fix when compare_by="training_data".
        fixed_training_data: Training data to fix when compare_by="model".
        include_baseline: Whether to include baseline models.

    Returns:
        tuple: (filtered_df, comparison_column, is_baseline_model)

    Raises:
        ValueError: If required parameters are missing or compare_by is invalid.
    """
    df = pd.read_csv(csv_file)

    # Filter for test datasets that match our test_data_names
    base_patterns = [extract_alignment_base(name) for name in test_data_names]
    mask = df["test_data_name"].apply(
        lambda x: any(base in x for base in base_patterns)
    )
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No matching test datasets for {base_patterns}. "
            f"Available: {df['test_data_name'].unique().tolist()}"
        )

    # Add base_test_name column for grouping replicates
    filtered_df["base_test_name"] = filtered_df["test_data_name"].apply(
        extract_alignment_base
    )

    # Add non-MP fraction column for distinguishing datasets by line style
    filtered_df["nonmp_fraction"] = filtered_df["base_test_name"].apply(
        _nonmp_fraction_key
    )

    is_baseline_model = False

    if compare_by == "model":
        if fixed_training_data is None:
            raise ValueError("Must specify fixed_training_data when compare_by='model'")

        if include_baseline:
            combined_mask = (filtered_df["train_data_name"] == fixed_training_data) | (
                filtered_df["train_data_name"] == "baseline"
            )
            filtered_df = filtered_df[combined_mask]
        else:
            filtered_df = filtered_df[
                filtered_df["train_data_name"] == fixed_training_data
            ]

        comparison_column = "model_name"

    elif compare_by == "training_data":
        if fixed_model is None:
            raise ValueError("Must specify fixed_model when compare_by='training_data'")
        if "Baseline" in fixed_model:
            is_baseline_model = True
        filtered_df = filtered_df[filtered_df["model_name"] == fixed_model]
        comparison_column = "train_data_name"

    return filtered_df, comparison_column, is_baseline_model


def _plot_metric_panel(
    ax,
    metric,
    filtered_df,
    comparison_values,
    comparison_column,
    color_dict,
    linestyle_dict,
    percentiles,
    is_baseline_model,
    is_first_metric,
):
    """Plot a single metric panel with all comparison values.

    Args:
        ax: Matplotlib axes object.
        metric: Name of the metric to plot.
        filtered_df: Filtered DataFrame with evaluation data.
        comparison_values: Unique values to compare (models or training datasets).
        comparison_column: Column name for comparison ("model_name" or "train_data_name").
        color_dict: Dictionary mapping comparison values to colors.
        linestyle_dict: Dictionary mapping non-MP fraction keys to line styles.
        percentiles: List of [lower, upper] percentile values.
        is_baseline_model: Whether plotting a baseline model.
        is_first_metric: Whether this is the first metric (for legend collection).

    Returns:
        tuple: (model_handles, model_labels, style_handles, style_labels)
    """
    metric_label = METRIC_LABELS.get(metric, metric)
    model_handles = []
    model_labels = []
    style_handles = []
    style_labels = []
    seen_values = set()
    seen_fractions = set()

    for value in comparison_values:
        value_df = filtered_df[filtered_df[comparison_column] == value]
        base_test_groups = value_df.groupby("base_test_name")

        # Collect color legend entry once per comparison value (solid proxy)
        if is_first_metric and not is_baseline_model and value not in seen_values:
            seen_values.add(value)
            proxy = plt.Line2D(
                [0],
                [0],
                color=color_dict[value],
                linestyle="-",
                linewidth=LINE_WIDTH_MAIN,
            )
            model_handles.append(proxy)
            model_labels.append(value)

        for _, group_df in base_test_groups:
            test_datasets = group_df["test_data_name"].unique()
            frac_key = group_df["nonmp_fraction"].iloc[0]
            linestyle = linestyle_dict.get(frac_key, "-")

            # Collect style legend entry once per unique fraction
            if is_first_metric and frac_key not in seen_fractions:
                seen_fractions.add(frac_key)
                style_handle = plt.Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle=linestyle,
                    linewidth=LINE_WIDTH_MAIN,
                )
                frac_label = f"t={frac_key}" if frac_key != "default" else "default"
                style_handles.append(style_handle)
                style_labels.append(frac_label)

            if len(test_datasets) > 1:  # We have replicates
                replicate_values = []
                for test_dataset in test_datasets:
                    test_df = group_df[group_df["test_data_name"] == test_dataset]
                    test_df = test_df.sort_values("tree_idx")
                    if len(test_df) > 0:
                        replicate_values.append(test_df[metric].values)

                if not replicate_values:
                    continue

                rep_max_len = max(len(v) for v in replicate_values)
                common_x, all_values = interpolate_to_common_grid(
                    replicate_values, rep_max_len
                )
                if len(all_values) == 0:
                    continue

                median, lower, upper = calculate_percentile_bands(
                    all_values, percentiles
                )

                ax.plot(
                    common_x,
                    median,
                    color=color_dict[value],
                    linestyle=linestyle,
                    label=f"{value}",
                    linewidth=LINE_WIDTH_MAIN,
                    alpha=LINE_ALPHA,
                )
                ax.fill_between(
                    common_x, lower, upper, color=color_dict[value], alpha=FILL_ALPHA
                )

            else:  # Single dataset (no replicates)
                test_dataset = test_datasets[0]
                test_df = group_df[group_df["test_data_name"] == test_dataset]
                test_df = test_df.sort_values("tree_idx")

                normalized_x = np.linspace(0, 1, len(test_df))
                ax.plot(
                    normalized_x,
                    test_df[metric].values,
                    color=color_dict[value],
                    linestyle=linestyle,
                    label=value,
                    alpha=LINE_ALPHA,
                    markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH_SECONDARY,
                )

    ax.set_ylabel(metric_label, fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"{metric_label}", fontsize=FONT_SIZE_TITLE)
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)

    return model_handles, model_labels, style_handles, style_labels


def _plot_bottom_panel(
    ax,
    all_parsimony_scores,
    all_frac_non_mp_edges,
    max_length,
    percentiles,
    parsimony_color,
    nonmp_color,
):
    """Plot the bottom panel with parsimony scores and non-MP edge fractions.

    Args:
        ax: Matplotlib axes object.
        all_parsimony_scores: List of parsimony score arrays.
        all_frac_non_mp_edges: List of non-MP edge fraction arrays.
        max_length: Maximum length for interpolation grid.
        percentiles: List of [lower, upper] percentile values.
        parsimony_color: Color for parsimony scores.
        nonmp_color: Color for non-MP edges.

    Returns:
        tuple: (all_handles, all_labels) for legend creation.
    """
    all_handles = []
    all_labels = []
    bottom_max_length = max_length if max_length > 0 else 100

    # Plot parsimony scores
    valid_parsimony = [s for s in all_parsimony_scores if len(s) > 0]
    if valid_parsimony:
        common_x, parsimony_array = interpolate_to_common_grid(
            valid_parsimony, bottom_max_length
        )

        if len(parsimony_array) > 0:
            parsimony_median, parsimony_lower, parsimony_upper = (
                calculate_percentile_bands(parsimony_array, percentiles)
            )

            (line1,) = ax.plot(
                common_x,
                parsimony_median,
                color=parsimony_color,
                linewidth=LINE_WIDTH_MAIN,
                label="Parsimony Score (median)",
            )

            if len(valid_parsimony) > 1:
                ax.fill_between(
                    common_x,
                    parsimony_lower,
                    parsimony_upper,
                    color=parsimony_color,
                    alpha=FILL_ALPHA,
                )
                all_handles.append(line1)
                all_labels.append("Parsimony Score (median)")
            else:
                all_handles.append(line1)
                all_labels.append("Parsimony Score")
    else:
        common_x = np.linspace(0, 1, bottom_max_length)
        ax.text(
            0.5,
            0.5,
            "No parsimony scores available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Plot non-MP edges on twin axis
    valid_fracs = [f for f in all_frac_non_mp_edges if len(f) > 0]
    if valid_fracs:
        ax_twin = ax.twinx()
        _, fracs_array = interpolate_to_common_grid(valid_fracs, bottom_max_length)

        if len(fracs_array) > 0:
            fracs_median, fracs_lower, fracs_upper = calculate_percentile_bands(
                fracs_array, percentiles
            )

            (line2,) = ax_twin.plot(
                common_x,
                fracs_median,
                color=nonmp_color,
                linewidth=LINE_WIDTH_MAIN,
                label="Fraction of non-MP Edges (median)",
            )

            if len(valid_fracs) > 1:
                ax_twin.fill_between(
                    common_x,
                    fracs_lower,
                    fracs_upper,
                    color=nonmp_color,
                    alpha=FILL_ALPHA,
                )

            all_handles.append(line2)
            all_labels.append("Fraction of non-MP Edges")

        ax_twin.set_ylabel(
            "Fraction of non-MP Edges", color=nonmp_color, fontsize=FONT_SIZE_LABEL
        )
        ax_twin.tick_params(axis="y", labelcolor=nonmp_color, labelsize=FONT_SIZE_TICK)

    # Style the bottom panel
    ax.set_xlabel("Normalized Tree Index (0-1)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Parsimony Score", color=parsimony_color, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="y", labelcolor=parsimony_color, labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
    ax.set_title("Parsimony Scores and Non-MP Edges", fontsize=FONT_SIZE_TITLE)

    return all_handles, all_labels


def _add_legends_and_save(
    fig,
    output_file,
    model_handles,
    model_labels,
    all_handles,
    all_labels,
    comparison_column,
    is_baseline_model,
    style_handles=None,
    style_labels=None,
):
    """Add legends to the figure and save to file.

    Args:
        fig: Matplotlib figure object.
        output_file: Path to save the figure.
        model_handles: Legend handles for models/training data.
        model_labels: Legend labels for models/training data.
        all_handles: Legend handles for parsimony/non-MP.
        all_labels: Legend labels for parsimony/non-MP.
        comparison_column: Column name for comparison.
        is_baseline_model: Whether plotting a baseline model.
        style_handles: Legend handles for non-MP fraction line styles.
        style_labels: Legend labels for non-MP fraction line styles.
    """
    # Add model/training data legend above plots
    if model_handles and model_labels and not is_baseline_model:
        unique_labels = []
        unique_handles = []
        seen_labels = set()
        for handle, label in zip(model_handles, model_labels):
            if label not in seen_labels:
                seen_labels.add(label)
                unique_labels.append(label)
                unique_handles.append(handle)

        # If there are multiple non-MP fractions, append style entries
        if style_handles and style_labels and len(style_handles) > 1:
            unique_handles.extend(style_handles)
            unique_labels.extend(style_labels)

        first_legend = fig.legend(
            unique_handles,
            unique_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            title=comparison_column.replace("_", " ").title(),
            title_fontsize=FONT_SIZE_LABEL,
            fontsize=FONT_SIZE_LEGEND,
            ncol=min(3, len(unique_labels)),
        )
        fig.add_artist(first_legend)

    # Add parsimony/non-MP legend below plots
    if all_handles:
        bottom_legend = fig.legend(
            all_handles,
            all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            fontsize=FONT_SIZE_LEGEND,
            ncol=2,
        )
        fig.add_artist(bottom_legend)

    plt.rcParams.update({"font.size": FONT_SIZE_LABEL})
    fig.subplots_adjust(top=0.90, bottom=0.08)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def plot_treesearch_evaluation(
    csv_file,
    data_nicknames_file,
    test_data_name,
    output_file,
    fasta_dir,
    metrics=["auroc", "accuracy", "precision", "recall", "tp", "fp", "tn", "fn"],
    compare_by="model",
    fixed_model=None,
    fixed_training_data=None,
    percentiles=[2.5, 97.5],
    include_baseline=False,
):
    """Plot all performance metrics for multiple models or training datasets in a
    single PDF. Plots each replicate as a separate line for clear visualization.

    Args:
        csv_file (str): Path to the CSV file containing evaluation data.
        data_nicknames_file (str): Path to the dataset nicknames JSON file.
        test_data_name (str or list): Nickname of the testing dataset(s).
        output_file (str): Path for saving the output PDF file.
        fasta_dir (str): Path to the directory containing the FASTA files.
        metrics (list): Metrics to plot (default: all standard metrics).
        compare_by (str): What to compare - "model" or "training_data".
        fixed_model (str): When compare_by="training_data", the model to fix.
        fixed_training_data (str): When compare_by="model", the training data to fix.
        percentiles (list): Percentile values for confidence bands (default: [2.5, 97.5]).
        include_baseline (bool): Whether to include baseline models in the comparison.
    """
    # Normalize test_data_name to list
    test_data_names = (
        [test_data_name] if isinstance(test_data_name, str) else test_data_name
    )

    # Load data nicknames
    dataset_dict = load_nicknames_dict(data_nicknames_file)

    # Collect parsimony and non-MP data
    all_parsimony_scores, all_frac_non_mp_edges, max_length = (
        _collect_parsimony_and_nonmp_data(test_data_names, dataset_dict, fasta_dir)
    )

    # Filter comparison data
    filtered_df, comparison_column, is_baseline_model = _filter_comparison_data(
        csv_file,
        test_data_names,
        compare_by,
        fixed_model,
        fixed_training_data,
        include_baseline,
    )

    # Setup colors
    comparison_values = filtered_df[comparison_column].unique()
    palette = sns.color_palette("Dark2", len(comparison_values) + 2)
    color_dict = dict(zip(comparison_values, palette[: len(comparison_values)]))
    parsimony_color = palette[-2]
    nonmp_color = palette[-1]

    # Setup line styles for non-MP fractions (e.g. t0.1, t0.2, default)
    unique_fractions = sorted(
        filtered_df["nonmp_fraction"].unique(),
        key=lambda x: (x != "default", x),
    )
    linestyle_dict = {
        frac: NONMP_LINESTYLES[i % len(NONMP_LINESTYLES)]
        for i, frac in enumerate(unique_fractions)
    }

    # Create figure
    n_metrics = len(metrics)
    n_rows = n_metrics + 1
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT_PER_ROW * n_rows))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.4)

    # Plot metrics panels
    all_model_handles = []
    all_model_labels = []
    all_style_handles = []
    all_style_labels = []
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[i, 0])
        handles, labels, style_handles, style_labels = _plot_metric_panel(
            ax,
            metric,
            filtered_df,
            comparison_values,
            comparison_column,
            color_dict,
            linestyle_dict,
            percentiles,
            is_baseline_model,
            is_first_metric=(i == 0),
        )
        if i == 0:
            all_model_handles = handles
            all_model_labels = labels
            all_style_handles = style_handles
            all_style_labels = style_labels

        # Set x-axis label only on last metric panel
        if i == n_metrics - 1:
            ax.set_xlabel("Normalized Tree Index (0-1)", fontsize=FONT_SIZE_LABEL)
        else:
            ax.set_xlabel("")

    # Plot bottom panel
    ax_bottom = fig.add_subplot(gs[n_rows - 1, 0])
    all_handles, all_labels = _plot_bottom_panel(
        ax_bottom,
        all_parsimony_scores,
        all_frac_non_mp_edges,
        max_length,
        percentiles,
        parsimony_color,
        nonmp_color,
    )

    # Add legends and save
    _add_legends_and_save(
        fig,
        output_file,
        all_model_handles,
        all_model_labels,
        all_handles,
        all_labels,
        comparison_column,
        is_baseline_model,
        style_handles=all_style_handles,
        style_labels=all_style_labels,
    )
