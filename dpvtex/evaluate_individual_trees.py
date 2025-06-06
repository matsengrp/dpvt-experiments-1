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
from dpvtex.dpvt_data import data_of_nicknames
from dpvtex.dpvt_zoo import build_model, prepend_dir_to_path, get_model_str


def get_rep_tested_model_str(
    model_name, train_data_name, test_data_name, param_id, rep_id
):
    # get model strings for replicates
    model = f"{model_name}-{train_data_name}-ON-{test_data_name}-{param_id}"
    return model


def get_model_str(model_name, train_data_name, test_data_name=None, param_id=None):
    if test_data_name:
        path = get_tested_model_str(
            model_name, train_data_name, test_data_name, param_id
        )
    else:
        path = get_trained_model_str(model_name, train_data_name, param_id)
    return path


def build_replicates_log_paths(
    model_name,
    train_data_name,
    test_data_name,
    param_id,
    device,
    timestamp,
    log_name,
    step_name,
    output_dir=".",
):
    paths = []
    while True:
        model_str = get_model_str(model_name, train_data_name, test_data_name, param_id)
        path = f"run.{timestamp}/{log_name}/{step_name}/{model_str}"
        path = prepend_dir_to_path(path, output_dir)
        path = f"{os.getcwd()}/{path}"
    return paths


def get_parsimony_scores(tree_list, fasta_path):
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


def plot_auroc_over_time(
    csv_file,
    data_nicknames_file,
    test_data_name,
    output_file_basename,
    fasta_dir,
    metrics=["auroc"],
):
    """
    Plots AUROC over time for each tree in the dataset whose AUROCs are provided
    in df in order and saves the plot to a file. Also adds line for number of
    non-MP edges in the dataset. Args:
        csv_file (str): Path to the CSV file containing AUROC data.
        data_nicknames_file (str): Path to the dataset nicknames JSON file.
        test_data_name (str): Nickname of the testing dataset.
        output_file_basename (str): Path to basename for file saving plot.
            Gets extended by "_metric.pdf" for each metric.
        fasta_dir (str): Path to the directory containing the FASTA files.
        metric (list): Metrics to plot (default: ["auroc"]).
    """
    with open(data_nicknames_file, "r") as f:
        dataset_dict = json.load(f)
    test_data_path = os.path.join(
        dataset_dict["data_dir"], dataset_dict[test_data_name]
    )
    with open(test_data_path, "rb") as f:
        data_dict = pickle.load(f)

    # We require the test_data_name to be in the format
    # "{fasta_basename}_tree_search_test.fasta" This is how they are generated
    # in our pipeline, so this should generally be fine
    data_basename = dataset_dict[test_data_name].split("_rep")[0]
    fasta_path = fasta_dir + "/" + data_basename + "/" + data_basename + ".fasta"

    parsimony_scores = get_parsimony_scores(list(data_dict.keys()), fasta_path)
    num_int_edges = len(list(data_dict.keys())[0]) - 2
    frac_non_mp_edges = [sum(l) / num_int_edges for l in data_dict.values()]

    df = pd.read_csv(csv_file)

    metric_labels = {
        "auroc": "AUROC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
    }
    for metric in metrics:
        metric_label = metric_labels[metric]
        # Create figure with two subplots (panels)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, 1]}
        )

        # Top panel: Plot just the metric Create normalized x-axis for the
        # metric values as well
        normalized_x_metric = np.linspace(0, 1, len(df))
        ax1.plot(
            normalized_x_metric,
            df[metric].values,
            color="blue",
            label=metric_label,
            marker="o",
            linestyle="-",
            markersize=4,
        )
        ax1.set_xlabel("Normalized Tree Index (0-1)")  # Add normalized x-axis label
        ax1.set_ylabel(metric_label, color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_title(f"{metric_label} by Normalized Tree Index")
        ax1.legend(loc="best")
        ax1.grid(True, linestyle="--", alpha=0.7)  # Add grid for better readability

        # Bottom panel: Plot parsimony scores AND non-MP edges Primary y-axis
        # for parsimony scores
        if parsimony_scores and not all(pd.isna(score) for score in parsimony_scores):
            # Normalize x-axis for parsimony scores
            normalized_x_scores = np.linspace(0, 1, len(parsimony_scores))
            ax2.plot(
                normalized_x_scores,
                parsimony_scores,
                color="green",
                label="Parsimony Score",
            )
            ax2.set_xlabel("Normalized Tree Index (0-1)", fontsize=14)
            ax2.set_ylabel("Parsimony Score", color="green", fontsize=14)
            ax2.tick_params(axis="y", labelcolor="green", labelsize=14)
            ax2.tick_params(axis="x", labelsize=14)
        else:
            ax2.set_title("No parsimony scores available", fontsize=14)

        # Secondary y-axis for non-MP edges - also with normalized x-axis
        ax2_twin = ax2.twinx()
        normalized_x_edges = np.linspace(0, 1, len(frac_non_mp_edges))
        ax2_twin.plot(
            normalized_x_edges,
            frac_non_mp_edges,
            color="red",
            label="Fraction of non-MP Edges",
        )
        ax2_twin.set_ylabel("Fraction of non-MP Edges", color="red", fontsize=14)
        ax2_twin.tick_params(axis="y", labelcolor="red", labelsize=14)

        # Explicitly set x-axis limits to ensure normalization is visible
        ax2.set_xlim(0, 1)

        # Add combined legend for bottom panel
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="center left",
            bbox_to_anchor=(1.1, 0.9),
            fontsize=12,
        )

        ax2.set_title(
            "Parsimony Scores and Non-MP Edges by Normalized Tree Index", fontsize=14
        )
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Add a suptitle to the entire figure If we have multiple replicates,
        # indicate that in the title
        replicate_text = (
            f" ({len(test_data_names)} Replicates)" if len(test_data_names) > 1 else ""
        )
        fig.suptitle(
            f"Performance Comparison - {base_test_name}{replicate_text}", fontsize=16
        )
        plt.rcParams.update({"font.size": 14})  # Set default font size globally
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle

        # Save figure
        output_file = f"{output_file_basename}_{metric}.pdf"
        plt.savefig(output_file)
        plt.clf()


def evaluate_individual_trees(
    model_name,
    train_data_name,
    trained_model_ckpt,
    test_data_name,
    device,
    hyperparameter_path,
    output_dir=".",
    data_nicknames_path="data_nicknames.json",
    output_file=None,
):
    """
    Evaluates model performance on individual trees from the test dataset,
    specifically designed for TraversalDataset.

    Args:
        model_name (str): Name of the model to evaluate. train_data_name (str):
        Name of the training dataset. trained_model_ckpt (str): Path to the
        trained model checkpoint. test_data_name (str): Name of the test
        dataset. device (str): Device to run the model on. hyperparameter_path
        (str): Path to the hyperparameters file. output_dir (str): Output
        directory. data_nicknames_path (str): Path to the dataset nicknames
        file. output_file (str): Optional path to save the evaluation results.

    Returns:
        pandas.DataFrame: DataFrame containing evaluation metrics for each tree.
    """

    # Load the test data
    test_data = data_of_nicknames(test_data_name, device, data_nicknames_path)

    # Load model and hyperparameters
    model_class = build_model(model_name)
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

    # Initialize metrics for each tree
    results = []

    # Move model to the appropriate device
    if device != "cpu-tree-dataset" and device != "cpu":
        model = model.to(device)

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

        with torch.no_grad():
            # Forward pass for a single tree
            logits = torch.stack(
                [
                    model.forward_on_traversal(t, m)
                    for (t, m) in zip(traversal, mutations)
                ]
            ).squeeze(0)
            probs = torch.sigmoid(logits)

        # Apply mask to only evaluate on valid edges
        mask = mask.unsqueeze(-1)
        labels = labels.unsqueeze(-1)
        masked_logits = logits[mask]
        masked_labels = labels[mask]
        masked_probs = probs[mask]

        # Calculate metrics for this tree
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

        # Calculate average predicted probability for positive and negative
        # examples
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

        # Store the results
        results.append(
            {
                "tree_idx": i,
                "num_edges": len(masked_labels),
                "num_pos": int((masked_labels == 1).sum().item()),
                "num_neg": int((masked_labels == 0).sum().item()),
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
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


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
    print(f"Processing {len(eval_paths)} evaluation files...")

    # Compile patterns for identifying replicate datasets
    base_test_patterns = {}
    for test_data in test_data_names:
        base_name = test_data.split("_rep")[0]
        base_test_patterns[base_name] = re.compile(f"{base_name}(_rep\\d+)?$")

    dfs = []
    for csv_file in eval_paths:
        try:
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

            # Skip if we couldn't identify all the metadata
            if None in [model_name, train_data_name, test_data_name, param_id]:
                print(
                    f"Warning: Could not identify all metadata for {file_name}, skipping."
                )
                print(
                    f"Found: model={model_name}, train={train_data_name}, test={test_data_name}, param={param_id}"
                )
                continue

            # Add metadata columns
            df["model_name"] = model_name
            df["train_data_name"] = train_data_name
            df["test_data_name"] = test_data_name
            df["param_id"] = param_id

            dfs.append(df)

        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

    # Combine all dataframes
    if not dfs:
        print("Warning: No evaluation files could be processed!")
        return pd.DataFrame()

    print(f"Successfully processed {len(dfs)} evaluation files.")
    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure column order is consistent - metadata columns first
    metadata_cols = ["model_name", "train_data_name", "test_data_name", "param_id"]
    other_cols = [col for col in combined_df.columns if col not in metadata_cols]
    combined_df = combined_df[metadata_cols + other_cols]

    # Print summary of test_data_name values
    test_datasets = combined_df["test_data_name"].unique()
    print(f"Found {len(test_datasets)} unique test datasets in the combined data:")
    for test_dataset in test_datasets:
        print(f"  - {test_dataset}")

    combined_df.to_csv(summary_file, index=False)
    print(f"Wrote combined dataset with {len(combined_df)} rows to {summary_file}")
    return combined_df


def plot_model_comparison(
    csv_file,
    data_nicknames_file,
    test_data_name,
    output_file_basename,
    fasta_dir,
    metrics=["auroc"],
    compare_by="model",
    fixed_model=None,
    fixed_training_data=None,
):
    """
    Plots performance metrics for multiple models or training datasets.

    Args:
        csv_file (str): Path to the CSV file containing evaluation data for
        multiple models/datasets. data_nicknames_file (str): Path to the dataset
        nicknames JSON file. test_data_name (str or list): Nickname of the
        testing dataset or a list of test datasets
                                      (e.g., for replicates).
        output_file_basename (str): Path to basename for file saving plot.
        fasta_dir (str): Path to the directory containing the FASTA files.
        metrics (list): Metrics to plot (default: ["auroc"]). compare_by (str):
        What to compare - "model" or "training_data". fixed_model (str): When
        compare_by="training_data", the model to fix. fixed_training_data (str):
        When compare_by="model", the training data to fix.
    """
    # Convert test_data_name to list if it's a string
    if isinstance(test_data_name, str):
        test_data_names = [test_data_name]
    else:
        test_data_names = test_data_name

    # Load data nicknames
    with open(data_nicknames_file, "r") as f:
        dataset_dict = json.load(f)

    # Get base test name (without replicate info) for plot title
    if len(test_data_names) > 0:
        base_test_name = test_data_names[0].split("_rep")[0]
    else:
        base_test_name = "unknown"

    # Collect parsimony scores and fraction of non-MP edges for all replicates
    all_parsimony_scores = []
    all_frac_non_mp_edges = []
    max_length = 0

    # Process each test dataset to get parsimony scores and non-MP edges
    for test_data in test_data_names:
        try:
            test_data_path = os.path.join(
                dataset_dict["data_dir"], dataset_dict[test_data]
            )
            with open(test_data_path, "rb") as f:
                data_dict = pickle.load(f)

            # Calculate parsimony scores and non-MP edges for this replicate
            data_basename = dataset_dict[test_data].split("_tree_search")[0]
            if "_rep" in data_basename:
                data_basename = data_basename.split("_rep")[0]
            fasta_path = (
                fasta_dir + "/" + data_basename + "/" + data_basename + ".fasta"
            )

            parsimony_scores = get_parsimony_scores(list(data_dict.keys()), fasta_path)
            num_int_edges = len(list(data_dict.keys())[0]) - 2
            frac_non_mp_edges = [sum(l) / num_int_edges for l in data_dict.values()]

            # Track maximum length for normalization later
            max_length = max(max_length, len(parsimony_scores))

            # Store the values
            all_parsimony_scores.append(parsimony_scores)
            all_frac_non_mp_edges.append(frac_non_mp_edges)

            print(f"Processed {test_data}: {len(parsimony_scores)} trees")
        except Exception as e:
            print(f"Error processing {test_data}: {e}")

    # Read the comparison data
    df = pd.read_csv(csv_file)
    print(f"Original dataframe has {len(df)} rows")

    # Filter for test datasets that match our test_data_names
    base_patterns = [name.split("_rep")[0] for name in test_data_names]
    mask = df["test_data_name"].apply(
        lambda x: any(base in x for base in base_patterns)
    )
    filtered_df = df[mask]

    # Add base_test_name column for grouping replicates
    filtered_df["base_test_name"] = filtered_df["test_data_name"].apply(
        lambda x: x.split("_rep")[0] if "_rep" in x else x
    )

    # Print debug info
    unique_test_names = filtered_df["test_data_name"].unique()
    print(f"Found {len(unique_test_names)} matching test datasets:")
    for name in unique_test_names:
        print(f"  - {name}")

    if len(filtered_df) == 0:
        print("No data found that matches the test dataset patterns!")
        print(f"Available test datasets: {df['test_data_name'].unique()}")
        return

    # Filter data based on comparison type
    if compare_by == "model":
        if fixed_training_data is None:
            raise ValueError("Must specify fixed_training_data when compare_by='model'")
        filtered_df = filtered_df[filtered_df["train_data_name"] == fixed_training_data]
        comparison_column = "model_name"
        title_prefix = (
            f"Models trained on {fixed_training_data}, tested on {base_test_name}"
        )
    elif compare_by == "training_data":
        if fixed_model is None:
            raise ValueError("Must specify fixed_model when compare_by='training_data'")
        filtered_df = filtered_df[filtered_df["model_name"] == fixed_model]
        comparison_column = "train_data_name"
        title_prefix = f"Model {fixed_model} trained on different datasets, tested on {base_test_name}"
    else:
        raise ValueError("compare_by must be 'model' or 'training_data'")

    # Get unique values for the comparison
    comparison_values = filtered_df[comparison_column].unique()
    metric_labels = {
        "auroc": "AUROC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
    }

    # Create color palette for models/training datasets
    palette = sns.color_palette(
        "Dark2", len(comparison_values) + 2
    )  # +2 for parsimony and non-MP
    color_dict = dict(zip(comparison_values, palette[: len(comparison_values)]))

    # Reserve the last two colors for parsimony and non-MP edges
    parsimony_color = palette[-2]  # Second-to-last color for parsimony
    nonmp_color = palette[-1]  # Last color for non-MP edges

    # Create plots for each metric
    for metric in metrics:
        metric_label = metric_labels.get(metric, metric)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1]}
        )

        # For each comparison value, collect data from all replicates
        for value in comparison_values:
            value_df = filtered_df[filtered_df[comparison_column] == value]

            # Group by replicates - now each replicate will have its own line
            for test_name in test_data_names:
                test_df = value_df[value_df["test_data_name"] == test_name]
                test_df = test_df.sort_values("tree_idx")

                # Use different alpha values for better visibility
                alpha_val = 0.7 if len(test_data_names) == 1 else 0.8

                # Create a label that includes replicate information
                if "_rep" in test_name:
                    rep_num = test_name.split("_rep")[1].split("_")[0]
                    rep_label = f"{value} (Rep {rep_num})"
                else:
                    rep_label = value

                # Plot the actual metric values using the normalized x-axis
                # Create a normalized x-axis (0 to 1) to make datasets with
                # different lengths comparable
                total_trees = len(test_df)
                normalized_x = np.linspace(0, 1, total_trees)

                ax1.plot(
                    normalized_x,
                    test_df[
                        metric
                    ].values,  # Use the actual metric values from the dataframe
                    color=color_dict[value],  # Same color for same model/training data
                    label=rep_label,
                    alpha=alpha_val,
                    markersize=4,
                    linewidth=2 if len(test_data_names) > 1 else 1.5,
                )

        ax1.set_xlabel("", fontsize=14)  # No x-label on top panel
        ax1.set_ylabel(metric_label, fontsize=14)
        ax1.set_title(f"{metric_label} by Tree Index", fontsize=14)
        ax1.legend(
            loc="center left",
            bbox_to_anchor=(1.1, 0.9),
            title=comparison_column.replace("_", " ").title(),
            fontsize=12,
        )
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.tick_params(axis="both", which="major", labelsize=14)

        # Bottom panel: Plot parsimony scores AND non-MP edges with percentile
        # bands
        ax2.set_title(
            "Parsimony Scores and Non-MP Edges by Normalized Tree Index", fontsize=14
        )
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Create common x-axis grid for interpolation
        common_x = (
            np.linspace(0, 1, max_length) if max_length > 0 else np.linspace(0, 1, 100)
        )

        # Process parsimony scores with percentile bands
        if all_parsimony_scores and not all(
            len(scores) == 0 for scores in all_parsimony_scores
        ):
            # Interpolate all parsimony scores to common grid
            interpolated_pscores = []
            for scores in all_parsimony_scores:
                if len(scores) > 0:
                    dataset_x = np.linspace(0, 1, len(scores))
                    interp_values = np.interp(
                        common_x, dataset_x, scores, left=np.nan, right=np.nan
                    )
                    interpolated_pscores.append(interp_values)

            # Calculate median and percentiles for parsimony scores
            if interpolated_pscores:
                parsimony_array = np.array(interpolated_pscores)
                parsimony_median = np.nanmedian(parsimony_array, axis=0)
                parsimony_lower = np.nanpercentile(
                    parsimony_array, percentiles[0], axis=0
                )
                parsimony_upper = np.nanpercentile(
                    parsimony_array, percentiles[1], axis=0
                )

                # Plot parsimony scores with median and percentile band
                (line1,) = ax2.plot(
                    common_x,
                    parsimony_median,
                    color=parsimony_color,  # Use reserved color from Dark2 palette
                    linewidth=2.0,
                    label=f"Parsimony Score (median)",
                )

                # Only add percentile band if we have multiple replicates
                if len(all_parsimony_scores) > 1:
                    ax2.fill_between(
                        common_x,
                        parsimony_lower,
                        parsimony_upper,
                        color=parsimony_color,
                        alpha=0.2,
                        label=f"Parsimony Score ({percentiles[0]}-{percentiles[1]}%)",
                    )

                    # Add rectangle for legend
                    pscore_band = plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        color=parsimony_color,
                        alpha=0.2,
                        label=f"Parsimony Score ({percentiles[0]}-{percentiles[1]}%)",
                    )
                    all_handles.extend([line1, pscore_band])
                    all_labels.extend(
                        [
                            f"Parsimony Score (median)",
                            f"Parsimony Score ({percentiles[0]}-{percentiles[1]}%)",
                        ]
                    )
                else:
                    all_handles.append(line1)
                    all_labels.append(f"Parsimony Score")
        else:
            ax2.text(
                0.5,
                0.5,
                "No parsimony scores available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        # Process non-MP edges with percentile bands
        if all_frac_non_mp_edges and not all(
            len(fracs) == 0 for fracs in all_frac_non_mp_edges
        ):
            # Secondary y-axis for non-MP edges
            ax2_twin = ax2.twinx()

            # Interpolate all non-MP edge fractions to common grid
            interpolated_fracs = []
            for fracs in all_frac_non_mp_edges:
                if len(fracs) > 0:
                    dataset_x = np.linspace(0, 1, len(fracs))
                    interp_values = np.interp(
                        common_x, dataset_x, fracs, left=np.nan, right=np.nan
                    )
                    interpolated_fracs.append(interp_values)

            # Calculate median and percentiles for non-MP edges
            if interpolated_fracs:
                fracs_array = np.array(interpolated_fracs)
                fracs_median = np.nanmedian(fracs_array, axis=0)
                fracs_lower = np.nanpercentile(fracs_array, percentiles[0], axis=0)
                fracs_upper = np.nanpercentile(fracs_array, percentiles[1], axis=0)

                # Plot non-MP edges with median and percentile band
                (line2,) = ax2_twin.plot(
                    common_x,
                    fracs_median,
                    color=nonmp_color,  # Use reserved color from Dark2 palette
                    linewidth=2.0,
                    label=f"Fraction of non-MP Edges (median)",
                )

                # Only add percentile band if we have multiple replicates
                if len(all_frac_non_mp_edges) > 1:
                    ax2_twin.fill_between(
                        common_x,
                        fracs_lower,
                        fracs_upper,
                        color=nonmp_color,
                        alpha=0.2,
                        label=f"Non-MP Edges ({percentiles[0]}-{percentiles[1]}%)",
                    )

                    # Add rectangle for legend
                    nonmp_band = plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        color=nonmp_color,
                        alpha=0.2,
                        label=f"Non-MP Edges ({percentiles[0]}-{percentiles[1]}%)",
                    )
                    all_handles.extend([line2, nonmp_band])
                    all_labels.extend(
                        [
                            f"Fraction of non-MP Edges (median)",
                            f"Non-MP Edges ({percentiles[0]}-{percentiles[1]}%)",
                        ]
                    )
                else:
                    all_handles.append(line2)
                    all_labels.append(f"Fraction of non-MP Edges")

            ax2_twin.set_ylabel(
                "Fraction of non-MP Edges", color=nonmp_color, fontsize=14
            )
            ax2_twin.tick_params(axis="y", labelcolor=nonmp_color, labelsize=14)

        # Explicitly set x-axis limits to ensure normalization is visible
        ax2.set_xlim(0, 1)

        # Add a suptitle to the entire figure If we have multiple replicates,
        # indicate that in the title
        replicate_text = (
            f" ({len(test_data_names)} Replicates)" if len(test_data_names) > 1 else ""
        )
        fig.suptitle(
            f"Performance Comparison - {base_test_name}{replicate_text}", fontsize=16
        )
        plt.rcParams.update({"font.size": 14})  # Set default font size globally
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle

        # Save figure
        output_file = f"{output_file_basename}-{metric}.pdf"
        plt.savefig(output_file)
        plt.close(fig)


def plot_all_metrics_comparison(
    csv_file,
    data_nicknames_file,
    test_data_name,
    output_file,
    fasta_dir,
    metrics=["auroc", "accuracy", "precision", "recall"],
    compare_by="model",
    fixed_model=None,
    fixed_training_data=None,
    percentiles=[25, 75],
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
    """
    # Convert test_data_name to list if it's a string
    if isinstance(test_data_name, str):
        test_data_names = [test_data_name]
    else:
        test_data_names = test_data_name

    # Load data nicknames
    with open(data_nicknames_file, "r") as f:
        dataset_dict = json.load(f)

    # Get base test name (without replicate info) for plot title
    if len(test_data_names) > 0:
        base_test_name = test_data_names[0].split("_rep")[0]
    else:
        base_test_name = "unknown"

    # Collect parsimony scores and fraction of non-MP edges for all replicates
    all_parsimony_scores = []
    all_frac_non_mp_edges = []
    max_length = 0

    # Process each test dataset to get parsimony scores and non-MP edges
    for test_data in test_data_names:
        try:
            test_data_path = os.path.join(
                dataset_dict["data_dir"], dataset_dict[test_data]
            )
            with open(test_data_path, "rb") as f:
                data_dict = pickle.load(f)

            # Calculate parsimony scores and non-MP edges for this replicate
            data_basename = dataset_dict[test_data].split("_tree_search")[0]
            if "_rep" in data_basename:
                data_basename = data_basename.split("_rep")[0]
            fasta_path = (
                fasta_dir + "/" + data_basename + "/" + data_basename + ".fasta"
            )

            parsimony_scores = get_parsimony_scores(list(data_dict.keys()), fasta_path)
            num_int_edges = len(list(data_dict.keys())[0]) - 2
            frac_non_mp_edges = [sum(l) / num_int_edges for l in data_dict.values()]

            # Track maximum length for normalization later
            max_length = max(max_length, len(parsimony_scores))

            # Store the values
            all_parsimony_scores.append(parsimony_scores)
            all_frac_non_mp_edges.append(frac_non_mp_edges)

            print(f"Processed {test_data}: {len(parsimony_scores)} trees")
        except Exception as e:
            print(f"Error processing {test_data}: {e}")

    # Read the comparison data
    df = pd.read_csv(csv_file)
    print(f"Original dataframe has {len(df)} rows")

    # Filter for test datasets that match our test_data_names
    base_patterns = [name.split("_rep")[0] for name in test_data_names]
    mask = df["test_data_name"].apply(
        lambda x: any(base in x for base in base_patterns)
    )
    filtered_df = df[mask]

    # Add base_test_name column for grouping replicates
    filtered_df["base_test_name"] = filtered_df["test_data_name"].apply(
        lambda x: x.split("_rep")[0] if "_rep" in x else x
    )

    # Print debug info
    unique_test_names = filtered_df["test_data_name"].unique()
    print(f"Found {len(unique_test_names)} matching test datasets:")
    for name in unique_test_names:
        print(f"  - {name}")

    if len(filtered_df) == 0:
        print("No data found that matches the test dataset patterns!")
        print(f"Available test datasets: {df['test_data_name'].unique()}")
        return

    # Filter data based on comparison type
    if compare_by == "model":
        if fixed_training_data is None:
            raise ValueError("Must specify fixed_training_data when compare_by='model'")
        filtered_df = filtered_df[filtered_df["train_data_name"] == fixed_training_data]
        comparison_column = "model_name"
        title_prefix = (
            f"Models trained on {fixed_training_data}, tested on {base_test_name}"
        )
    elif compare_by == "training_data":
        if fixed_model is None:
            raise ValueError("Must specify fixed_model when compare_by='training_data'")
        filtered_df = filtered_df[filtered_df["model_name"] == fixed_model]
        comparison_column = "train_data_name"
        title_prefix = f"Model {fixed_model} trained on different datasets, tested on {base_test_name}"
    else:
        raise ValueError("compare_by must be 'model' or 'training_data'")

    # Get unique values for the comparison
    comparison_values = filtered_df[comparison_column].unique()

    if len(comparison_values) == 0:
        print(f"No data found for {comparison_column}!")
        return

    # Create color palette for the different models/training datasets
    metric_labels = {
        "auroc": "AUROC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
    }

    # Create color palette for models/training datasets
    palette = sns.color_palette(
        "Dark2", len(comparison_values) + 2
    )  # +2 for parsimony and non-MP
    color_dict = dict(zip(comparison_values, palette[: len(comparison_values)]))

    # Reserve the last two colors for parsimony and non-MP edges
    parsimony_color = palette[-2]  # Second-to-last color for parsimony
    nonmp_color = palette[-1]  # Last color for non-MP edges

    # Calculate grid dimensions for the metrics plots
    n_metrics = len(metrics)
    n_rows = n_metrics + 1  # +1 for the parsimony/non-MP plot

    # Create figure with a grid of subplots
    fig = plt.figure(figsize=(12, 5 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.4)

    # Create a list to store all legend handles and labels
    all_handles = []
    all_labels = []
    model_handles = []
    model_labels = []

    # Create subplot for each metric
    for i, metric in enumerate(metrics):
        metric_label = metric_labels.get(metric, metric)
        ax = fig.add_subplot(gs[i, 0])

        # Plot for each comparison value (model/training dataset)
        for value_idx, value in enumerate(comparison_values):
            # Filter for this comparison value
            value_df = filtered_df[filtered_df[comparison_column] == value]

            # Group by base test name to combine replicates
            base_test_groups = value_df.groupby("base_test_name")

            for base_test_name, group_df in base_test_groups:
                # Get all replicate datasets for this base test name
                test_datasets = group_df["test_data_name"].unique()

                if len(test_datasets) > 1:  # We have replicates
                    # First determine the max length among replicates to set up interpolation grid
                    max_length = 0
                    for test_dataset in test_datasets:
                        test_df = group_df[group_df["test_data_name"] == test_dataset]
                        max_length = max(max_length, len(test_df))

                    # Create a common x-axis grid for interpolation
                    common_x = np.linspace(0, 1, max_length)

                    # Collect interpolated values for all replicates
                    all_values = []
                    for test_dataset in test_datasets:
                        test_df = group_df[group_df["test_data_name"] == test_dataset]
                        test_df = test_df.sort_values("tree_idx")

                        if len(test_df) == 0:
                            continue

                        # Create normalized x-axis for this dataset
                        dataset_x = np.linspace(0, 1, len(test_df))
                        # Interpolate values onto common grid
                        interpolated_values = np.interp(
                            common_x,
                            dataset_x,
                            test_df[metric].values,
                            left=np.nan,
                            right=np.nan,
                        )
                        all_values.append(interpolated_values)

                    if not all_values:
                        continue

                    # Convert to numpy array for percentile calculation
                    all_values = np.array(all_values)

                    # Calculate median and percentiles
                    median = np.nanmedian(all_values, axis=0)
                    lower = np.nanpercentile(all_values, percentiles[0], axis=0)
                    upper = np.nanpercentile(all_values, percentiles[1], axis=0)

                    # Plot median line
                    line = ax.plot(
                        common_x,
                        median,
                        color=color_dict[value],
                        label=f"{value} (median)",
                        linewidth=2.0,
                        alpha=0.9,
                    )

                    # Plot percentile band
                    ax.fill_between(
                        common_x,
                        lower,
                        upper,
                        color=color_dict[value],
                        alpha=0.2,
                        label=f"{value} ({percentiles[0]}-{percentiles[1]} percentile)",
                    )

                    # Only add to legend for the first metric plot
                    if i == 0:
                        model_handles.append(line[0])
                        model_labels.append(f"{value} (median)")

                        # Add fill_between to legend
                        model_handles.append(
                            plt.Rectangle(
                                (0, 0),
                                1,
                                1,
                                color=color_dict[value],
                                alpha=0.2,
                            )
                        )
                        model_labels.append(
                            f"{value} ({percentiles[0]}-{percentiles[1]} percentile)"
                        )

                else:  # Single dataset (no replicates)
                    # Process just the one dataset
                    test_dataset = test_datasets[0]
                    test_df = group_df[group_df["test_data_name"] == test_dataset]
                    test_df = test_df.sort_values("tree_idx")

                    if len(test_df) == 0:
                        print(f"No data found for {test_dataset} with {value}")
                        continue

                    # Create a normalized x-axis
                    normalized_x = np.linspace(0, 1, len(test_df))

                    # Plot the actual metric values
                    line = ax.plot(
                        normalized_x,
                        test_df[metric].values,
                        color=color_dict[value],
                        label=value,
                        alpha=0.9,
                        markersize=3,
                        linewidth=1.5,
                    )

                    # Only add to legend for the first metric plot
                    if i == 0:
                        model_handles.append(line[0])
                        model_labels.append(value)

        # Set axis labels and styling
        ax.set_ylabel(metric_label, fontsize=14)
        if i < n_metrics - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Normalized Tree Index (0-1)", fontsize=14)
        ax.set_title(f"{metric_label}", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=14)

    # Bottom panel: Plot parsimony scores AND non-MP edges with percentile bands
    ax_bottom = fig.add_subplot(gs[n_rows - 1, 0])

    # Create common x-axis grid for interpolation
    common_x = (
        np.linspace(0, 1, max_length) if max_length > 0 else np.linspace(0, 1, 100)
    )

    # Process parsimony scores with percentile bands
    if all_parsimony_scores and not all(
        len(scores) == 0 for scores in all_parsimony_scores
    ):
        # Interpolate all parsimony scores to common grid
        interpolated_pscores = []
        for scores in all_parsimony_scores:
            if len(scores) > 0:
                dataset_x = np.linspace(0, 1, len(scores))
                interp_values = np.interp(
                    common_x, dataset_x, scores, left=np.nan, right=np.nan
                )
                interpolated_pscores.append(interp_values)

        # Calculate median and percentiles for parsimony scores
        if interpolated_pscores:
            parsimony_array = np.array(interpolated_pscores)
            parsimony_median = np.nanmedian(parsimony_array, axis=0)
            parsimony_lower = np.nanpercentile(parsimony_array, percentiles[0], axis=0)
            parsimony_upper = np.nanpercentile(parsimony_array, percentiles[1], axis=0)

            # Plot parsimony scores with median and percentile band
            (line1,) = ax_bottom.plot(
                common_x,
                parsimony_median,
                color=parsimony_color,  # Use reserved color from Dark2 palette
                linewidth=2.0,
                label=f"Parsimony Score (median)",
            )

            # Only add percentile band if we have multiple replicates
            if len(all_parsimony_scores) > 1:
                ax_bottom.fill_between(
                    common_x,
                    parsimony_lower,
                    parsimony_upper,
                    color=parsimony_color,
                    alpha=0.2,
                    label=f"Parsimony Score ({percentiles[0]}-{percentiles[1]}%)",
                )

                # Add rectangle for legend
                pscore_band = plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    color=parsimony_color,
                    alpha=0.2,
                    label=f"Parsimony Score ({percentiles[0]}-{percentiles[1]}%)",
                )
                all_handles.extend([line1, pscore_band])
                all_labels.extend(
                    [
                        f"Parsimony Score (median)",
                        f"Parsimony Score ({percentiles[0]}-{percentiles[1]}%)",
                    ]
                )
            else:
                all_handles.append(line1)
                all_labels.append(f"Parsimony Score")
        else:
            ax_bottom.text(
                0.5,
                0.5,
                "No parsimony scores available",
                ha="center",
                va="center",
                transform=ax_bottom.transAxes,
            )

    # Process non-MP edges with percentile bands
    if all_frac_non_mp_edges and not all(
        len(fracs) == 0 for fracs in all_frac_non_mp_edges
    ):
        # Secondary y-axis for non-MP edges
        ax_bottom_twin = ax_bottom.twinx()

        # Interpolate all non-MP edge fractions to common grid
        interpolated_fracs = []
        for fracs in all_frac_non_mp_edges:
            if len(fracs) > 0:
                dataset_x = np.linspace(0, 1, len(fracs))
                interp_values = np.interp(
                    common_x, dataset_x, fracs, left=np.nan, right=np.nan
                )
                interpolated_fracs.append(interp_values)

        # Calculate median and percentiles for non-MP edges
        if interpolated_fracs:
            fracs_array = np.array(interpolated_fracs)
            fracs_median = np.nanmedian(fracs_array, axis=0)
            fracs_lower = np.nanpercentile(fracs_array, percentiles[0], axis=0)
            fracs_upper = np.nanpercentile(fracs_array, percentiles[1], axis=0)

            # Plot non-MP edges with median and percentile band
            (line2,) = ax_bottom_twin.plot(
                common_x,
                fracs_median,
                color=nonmp_color,  # Use reserved color from Dark2 palette
                linewidth=2.0,
                label=f"Fraction of non-MP Edges (median)",
            )

            # Only add percentile band if we have multiple replicates
            if len(all_frac_non_mp_edges) > 1:
                ax_bottom_twin.fill_between(
                    common_x,
                    fracs_lower,
                    fracs_upper,
                    color=nonmp_color,
                    alpha=0.2,
                    label=f"Non-MP Edges ({percentiles[0]}-{percentiles[1]}%)",
                )

                # Add rectangle for legend
                nonmp_band = plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    color=nonmp_color,
                    alpha=0.2,
                    label=f"Non-MP Edges ({percentiles[0]}-{percentiles[1]}%)",
                )
                all_handles.extend([line2, nonmp_band])
                all_labels.extend(
                    [
                        f"Fraction of non-MP Edges (median)",
                        f"Non-MP Edges ({percentiles[0]}-{percentiles[1]}%)",
                    ]
                )
            else:
                all_handles.append(line2)
                all_labels.append(f"Fraction of non-MP Edges")

        ax_bottom_twin.set_ylabel(
            "Fraction of non-MP Edges", color=nonmp_color, fontsize=14
        )
        ax_bottom_twin.tick_params(axis="y", labelcolor=nonmp_color, labelsize=14)

    # Styling for the bottom plot
    ax_bottom.set_xlabel("Normalized Tree Index (0-1)", fontsize=14)
    ax_bottom.set_ylabel("Parsimony Score", color=parsimony_color, fontsize=14)
    ax_bottom.tick_params(axis="y", labelcolor=parsimony_color, labelsize=14)
    ax_bottom.tick_params(axis="x", labelsize=14)
    ax_bottom.set_xlim(0, 1)
    ax_bottom.grid(True, linestyle="--", alpha=0.7)
    ax_bottom.set_title("Parsimony Scores and Non-MP Edges", fontsize=14)

    # Add a suptitle to the entire figure
    replicate_text = (
        f" ({len(unique_test_names)} Datasets)" if len(unique_test_names) > 1 else ""
    )
    fig.suptitle(
        f"{title_prefix}{replicate_text}\nPerformance Metrics Comparison", fontsize=16
    )

    # Position legend for model/training data comparisons ABOVE all plots
    if model_handles and model_labels:
        unique_labels = []
        unique_handles = []
        seen_labels = set()
        for handle, label in zip(model_handles, model_labels):
            if label not in seen_labels:
                seen_labels.add(label)
                unique_labels.append(label)
                unique_handles.append(handle)

        # Position legend at the top of the figure
        first_legend = fig.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),  # Position at top center
            title=comparison_column.replace("_", " ").title(),
            title_fontsize=14,
            fontsize=12,
            ncol=min(3, len(unique_labels)),  # Use multiple columns for better spacing
        )
        # Ensure the first legend is drawn
        fig.add_artist(first_legend)

    # Position legend for parsimony/non-MP edges BELOW all plots
    if all_handles:
        bottom_legend = fig.legend(
            all_handles,
            all_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),  # Position at bottom center
            fontsize=12,
            ncol=2,  # Use two columns for better spacing
        )
        fig.add_artist(bottom_legend)

    # Remove individual legends from the parsimony/non-MP plot
    # (Comment out or remove the ax_bottom.legend() call)

    plt.rcParams.update({"font.size": 14})  # Set default font size globally

    # Adjust figure to make room for legends on top and bottom
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.12)  # Adjust top and bottom margins

    # Save figure
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
