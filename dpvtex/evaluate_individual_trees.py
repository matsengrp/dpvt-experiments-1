import torch
import json
import pickle
import os
import re
import historydag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from torchmetrics import AUROC
from torchmetrics.classification import BinaryROC
from dpvtex.dpvt_data import data_of_nicknames, load_nicknames_dict
from dpvtex.dpvt_zoo import build_model, prepend_dir_to_path, get_model_str
from dpvt import models

torch.set_num_threads(1)

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


# =============================================================================
# Interpolation Helper (DRY fix)
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


def load_model(
    model_name, hyperparameter_path=None, trained_model_ckpt=None, device="cpu"
):
    """
    Load a model based on model name and optional hyperparameters and checkpoint.

    Args:
        model_name: Name of the model to load
        hyperparameter_path: Path to hyperparameters JSON file (not needed for BaselineReversion)
        trained_model_ckpt: Path to trained model checkpoint (not needed for BaselineReversion)
        device: Device to run the model on (cpu, cuda, etc.)

    Returns:
        tuple: (model, device)
    """
    model_class = build_model(model_name)

    # For BaselineReversion, we don't need to load from checkpoint
    if model_name == "BaselineReversion":
        model = model_class()
        # BaselineReversion must run on CPU
        device = "cpu"
    else:
        # For trained models, load hyperparameters and checkpoint
        with open(hyperparameter_path, "r") as f:
            hparams = json.load(f)

        # Load the trained model
        model = model_class.load_from_checkpoint(
            trained_model_ckpt,
            learning_rate=hparams["learning_rate"],
            feature_length=hparams["feature_length"],
            dim_mlp_layers=hparams["dim_mlp_layers"],
        )

    # Set model to evaluation mode
    model.eval()

    # Move model to the appropriate device if needed
    if device != "cpu-tree-dataset":
        model = model.to(device)

    return model, device


def get_predictions(model, traversal=None, mutations=None, tree=None):
    """
    Generate model predictions from either traversal/mutations or tree input.

    Args:
        model: The model to use for predictions
        traversal: Tree traversal tensor (for standard models)
        mutations: Mutations tensor (for standard models)
        tree: Tree object (for BaselineReversion)

    Returns:
        tuple: (logits, probabilities)
    """
    with torch.no_grad():
        if tree is not None:  # For BaselineReversion
            logits = model.get_reversion_labels_from_tree(tree)
            probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
        else:  # For standard models
            # Forward pass for a single tree
            logits = torch.stack(
                [
                    model.forward_on_traversal(t, m)
                    for (t, m) in zip(traversal, mutations)
                ]
            ).squeeze(0)
            probs = torch.sigmoid(logits)

    return logits, probs


def calculate_metrics(masked_probs, masked_labels):
    """
    Calculate evaluation metrics for model predictions.

    Args:
        masked_probs: Predicted probabilities after masking
        masked_labels: True labels after masking

    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Calculate AUROC
    auroc = AUROC(task="binary")
    auroc_value = (
        auroc(masked_probs, masked_labels.int())
        if len(masked_labels) > 0
        else float("nan")
    )

    # Calculate accuracy
    predictions = (masked_probs > 0.5).float()
    accuracy = (
        (predictions == masked_labels).float().mean().item()
        if len(masked_labels) > 0
        else float("nan")
    )

    # Count true positives, false positives, etc.
    tp = ((predictions == 1) & (masked_labels == 1)).sum().item()
    fp = ((predictions == 1) & (masked_labels == 0)).sum().item()
    tn = ((predictions == 0) & (masked_labels == 0)).sum().item()
    fn = ((predictions == 0) & (masked_labels == 1)).sum().item()

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else float("nan")
    )

    # Calculate average predicted probability for positive and negative examples
    avg_prob_pos = (
        masked_probs[masked_labels == 1].mean().item()
        if (masked_labels == 1).sum() > 0
        else float("nan")
    )
    avg_prob_neg = (
        masked_probs[masked_labels == 0].mean().item()
        if (masked_labels == 0).sum() > 0
        else float("nan")
    )

    # Return all metrics as a dictionary
    return {
        "auroc": float(auroc_value),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "avg_prob_pos": float(avg_prob_pos),
        "avg_prob_neg": float(avg_prob_neg),
    }


def evaluate_individual_trees(
    model_name,
    train_data_name,
    trained_model_ckpt,
    test_data_name,
    device,
    hyperparameter_path,
    output_file,
    data_nicknames_path="data_nicknames.json",
):
    """
    Evaluates model performance on individual trees from the test dataset,
    specifically designed for TraversalDataset.

    Args:
        model_name (str): Name of the model to evaluate.
        train_data_name (str): Name of the training dataset.
        trained_model_ckpt (str): Path to the trained model checkpoint.
        test_data_name (str): Name of the test dataset.
        device (str): Device to run the model on.
        hyperparameter_path (str): Path to the hyperparameters file.
        output_file (str): Path to save the evaluation results.
        data_nicknames_path (str): Path to the dataset nicknames file.

    Returns:
        pandas.DataFrame: DataFrame containing evaluation metrics for each tree.
    """
    # Load the test data
    test_data = data_of_nicknames(test_data_name, device, data_nicknames_path)

    # Load model and move to appropriate device
    model, device = load_model(
        model_name, hyperparameter_path, trained_model_ckpt, device
    )

    # Initialize metrics for each tree
    results = []

    # Process each tree individually
    for i in range(len(test_data)):
        # For TraversalDataset, extract the data for a single tree
        traversal = test_data.traversal[
            i : i + 1
        ]  # Keep batch dimension but select single tree
        mutations = test_data.mutations[
            i : i + 1
        ]  # Keep batch dimension but select single tree
        labels = test_data.labels[i]
        mask = test_data.mask[i]

        # Ensure data is on the right device
        if device != "cpu-tree-dataset" and device != "cpu":
            traversal = traversal.to(device)
            mutations = mutations.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

        # Get model predictions
        logits, probs = get_predictions(model, traversal, mutations)

        # Apply mask to only evaluate on valid edges
        mask = mask.unsqueeze(-1)
        labels = labels.unsqueeze(-1)
        masked_logits = logits[mask]
        masked_labels = labels[mask]
        masked_probs = probs[mask]

        # Calculate metrics for this tree
        metrics = calculate_metrics(masked_probs, masked_labels)

        # Store the results
        results.append(
            {
                "tree_idx": i,
                "num_edges": len(masked_labels),
                "num_pos": int((masked_labels == 1).sum().item()),
                "num_neg": int((masked_labels == 0).sum().item()),
                **metrics,
            }
        )

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    return results_df


def evaluate_baseline_reversion_on_trees(
    test_data_name,
    output_file,
    data_nicknames_path="data_nicknames.json",
):
    """
    Evaluates the BaselineReversion model on individual trees from the test dataset.
    Since BaselineReversion doesn't require training, this function is simplified
    compared to evaluate_individual_trees.

    Args:
        test_data_name (str): Name of the test dataset.
        output_file (str): Path to save the evaluation results.
        data_nicknames_path (str): Path to the dataset nicknames file.

    Returns:
        pandas.DataFrame: DataFrame containing evaluation metrics for each tree.
    """

    # Load the test data (must use CPU for BaselineReversion)
    device = "cpu"
    test_data = data_of_nicknames(
        test_data_name, device, data_nicknames_path, data_struct="TreeDataset"
    )

    # Initialize the BaselineReversion model
    model, _ = load_model("BaselineReversion")

    # Initialize metrics for each tree
    results = []

    # Process each tree individually
    for i in range(len(test_data)):
        # Get the actual tree, labels and mask for BaselineReversion
        tree = test_data.data[i]
        labels = test_data.labels[i]
        mask = test_data.mask[i]

        # BaselineReversion directly generates predictions from the tree
        logits, probs = get_predictions(model, tree=tree)

        # Apply mask to only evaluate on valid edges
        masked_logits = logits[mask]
        masked_labels = labels[mask]
        masked_probs = probs[mask]

        # Calculate metrics for this tree
        metrics = calculate_metrics(masked_probs, masked_labels)

        # Store the results
        results.append(
            {
                "tree_idx": i,
                "num_edges": len(masked_labels),
                "num_pos": int((masked_labels == 1).sum().item()),
                "num_neg": int((masked_labels == 0).sum().item()),
                **metrics,
            }
        )

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    return results_df


def concatenate_tree_eval_files(
    eval_paths, model_names, train_data_names, test_data_names, param_ids, summary_file
):
    """
    Concatenate multiple tree evaluation CSV files and add metadata columns.
    Args:
        eval_paths (list): List of paths to CSV files to concatenate model_names
        (list): List of model names to match in filenames train_data_names
        (list): List of training dataset names to match in filenames
        test_data_names (list): List of test dataset names to match in filenames
        param_ids (list): List of parameter IDs to match in filenames
    Returns:
        pandas.DataFrame: Concatenated dataframe with metadata columns
    """
    # Compile patterns for identifying replicate datasets
    base_test_patterns = {}
    for test_data in test_data_names:
        base_name = test_data.split("_rep")[0]
        base_test_patterns[base_name] = re.compile(f"{base_name}(_rep\\d+)?$")

    dfs = []
    for csv_file in eval_paths:
        # Read the CSV file first
        df = pd.read_csv(csv_file)

        # Parse model, train_data, test_data from the file path
        path_parts = str(csv_file).split("/")
        file_name = path_parts[-1]

        # Extract information from filename
        model_name = None
        for model in model_names:
            if model in file_name:
                model_name = model
                break

        train_data_name = None
        for train_data in train_data_names:
            if train_data in file_name:
                train_data_name = train_data
                break

        # For test data, handle replicates by using regex patterns
        test_data_name = None

        # First, try exact matches with the test data names
        for test_data in test_data_names:
            if test_data in file_name:
                test_data_name = test_data
                break

        # If no exact match found, check for replicate pattern matches
        if test_data_name is None:
            for base_name, pattern in base_test_patterns.items():
                # Look for replicate suffix in filename
                rep_match = re.search(r"_rep(\d+)", file_name)
                if rep_match and base_name in file_name:
                    rep_num = rep_match.group(1)
                    test_data_name = f"{base_name}_rep{rep_num}"
                    break

        param_id = None
        for param in param_ids:
            if param in file_name:
                param_id = param
                break

        # Add metadata columns
        df["model_name"] = model_name
        df["train_data_name"] = train_data_name
        df["test_data_name"] = test_data_name
        df["param_id"] = param_id

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure column order is consistent - metadata columns first
    metadata_cols = ["model_name", "train_data_name", "test_data_name", "param_id"]
    other_cols = [col for col in combined_df.columns if col not in metadata_cols]
    combined_df = combined_df[metadata_cols + other_cols]

    combined_df.to_csv(summary_file, index=False)
    return combined_df


# =============================================================================
# Plotting Helper Functions
# =============================================================================


def _collect_parsimony_and_nonmp_data(test_data_names, dataset_dict, fasta_dir):
    """
    Collect parsimony scores and fraction of non-MP edges for all test datasets.

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
        data_basename = filename.split("_tree_search")[0]
        if "_rep" in data_basename:
            data_basename = data_basename.split("_rep")[0]
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
    """
    Load and filter the evaluation data based on comparison type.

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
    base_patterns = [name.split("_rep")[0] for name in test_data_names]
    mask = df["test_data_name"].apply(
        lambda x: any(base in x for base in base_patterns)
    )
    filtered_df = df[mask].copy()

    # Add base_test_name column for grouping replicates
    filtered_df["base_test_name"] = filtered_df["test_data_name"].apply(
        lambda x: x.split("_rep")[0] if "_rep" in x else x
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
    percentiles,
    is_baseline_model,
    is_first_metric,
):
    """
    Plot a single metric panel with all comparison values.

    Args:
        ax: Matplotlib axes object.
        metric: Name of the metric to plot.
        filtered_df: Filtered DataFrame with evaluation data.
        comparison_values: Unique values to compare (models or training datasets).
        comparison_column: Column name for comparison ("model_name" or "train_data_name").
        color_dict: Dictionary mapping comparison values to colors.
        percentiles: List of [lower, upper] percentile values.
        is_baseline_model: Whether plotting a baseline model.
        is_first_metric: Whether this is the first metric (for legend collection).

    Returns:
        tuple: (model_handles, model_labels) for legend creation.
    """
    metric_label = METRIC_LABELS.get(metric, metric)
    model_handles = []
    model_labels = []

    for value in comparison_values:
        value_df = filtered_df[filtered_df[comparison_column] == value]
        base_test_groups = value_df.groupby("base_test_name")

        for _, group_df in base_test_groups:
            test_datasets = group_df["test_data_name"].unique()

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

                line = ax.plot(
                    common_x,
                    median,
                    color=color_dict[value],
                    label=f"{value}",
                    linewidth=LINE_WIDTH_MAIN,
                    alpha=LINE_ALPHA,
                )
                ax.fill_between(
                    common_x, lower, upper, color=color_dict[value], alpha=FILL_ALPHA
                )

                if is_first_metric and not is_baseline_model:
                    model_handles.append(line[0])
                    model_labels.append(f"{value}")

            else:  # Single dataset (no replicates)
                test_dataset = test_datasets[0]
                test_df = group_df[group_df["test_data_name"] == test_dataset]
                test_df = test_df.sort_values("tree_idx")

                normalized_x = np.linspace(0, 1, len(test_df))
                line = ax.plot(
                    normalized_x,
                    test_df[metric].values,
                    color=color_dict[value],
                    label=value,
                    alpha=LINE_ALPHA,
                    markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH_SECONDARY,
                )

                if is_first_metric and not is_baseline_model:
                    model_handles.append(line[0])
                    model_labels.append(value)

    ax.set_ylabel(metric_label, fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"{metric_label}", fontsize=FONT_SIZE_TITLE)
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK)

    return model_handles, model_labels


def _plot_bottom_panel(
    ax,
    all_parsimony_scores,
    all_frac_non_mp_edges,
    max_length,
    percentiles,
    parsimony_color,
    nonmp_color,
):
    """
    Plot the bottom panel with parsimony scores and non-MP edge fractions.

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
):
    """
    Add legends to the figure and save to file.

    Args:
        fig: Matplotlib figure object.
        output_file: Path to save the figure.
        model_handles: Legend handles for models/training data.
        model_labels: Legend labels for models/training data.
        all_handles: Legend handles for parsimony/non-MP.
        all_labels: Legend labels for parsimony/non-MP.
        comparison_column: Column name for comparison.
        is_baseline_model: Whether plotting a baseline model.
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

        first_legend = fig.legend(
            unique_handles,
            unique_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            title=comparison_column.replace("_", " ").title(),
            title_fontsize=FONT_SIZE_LABEL,
            fontsize=FONT_SIZE_LEGEND,
            ncol=min(2, len(unique_labels)),
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


# =============================================================================
# Main Plotting Function
# =============================================================================


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
    """
    Plots all performance metrics for multiple models or training datasets in a
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

    # Create figure
    n_metrics = len(metrics)
    n_rows = n_metrics + 1
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT_PER_ROW * n_rows))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.4)

    # Plot metrics panels
    all_model_handles = []
    all_model_labels = []
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[i, 0])
        handles, labels = _plot_metric_panel(
            ax,
            metric,
            filtered_df,
            comparison_values,
            comparison_column,
            color_dict,
            percentiles,
            is_baseline_model,
            is_first_metric=(i == 0),
        )
        if i == 0:
            all_model_handles = handles
            all_model_labels = labels

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
    )
