import torch
import json
import pandas as pd
import pickle
import os
import historydag
from Bio import SeqIO
from torchmetrics import AUROC
from torchmetrics.classification import BinaryROC
import matplotlib.pyplot as plt
import seaborn as sns
from dpvtex.dpvt_data import data_of_nicknames
from dpvtex.dpvt_zoo import build_model


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
    Plots AUROC over time for each tree in the
    dataset whose AUROCs are provided in df in order and saves the plot to a file.
    Also adds line for number of non-MP edges in the dataset.
    Args:
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
    data_basename = dataset_dict[test_data_name].split("_tree_search")[0]
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

        # Top panel: Plot just the metric
        sns.scatterplot(
            data=df, x="tree_idx", y=metric, ax=ax1, color="blue", label=metric_label
        )
        ax1.set_xlabel("")  # No x-label on top panel
        ax1.set_ylabel(metric_label, color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_title(f"{metric_label} by Tree Index")
        ax1.legend(loc="best")

        # Bottom panel: Plot parsimony scores AND non-MP edges Primary y-axis
        # for parsimony scores
        if parsimony_scores and not all(pd.isna(score) for score in parsimony_scores):
            x_range_scores = range(len(parsimony_scores))
            ax2.plot(
                x_range_scores, parsimony_scores, color="green", label="Parsimony Score"
            )
            ax2.set_xlabel("Tree Index")
            ax2.set_ylabel("Parsimony Score", color="green")
            ax2.tick_params(axis="y", labelcolor="green")
        else:
            ax2.set_title("No parsimony scores available")
        # Secondary y-axis for non-MP edges
        ax2_twin = ax2.twinx()
        x_range = range(len(frac_non_mp_edges))
        ax2_twin.plot(
            x_range, frac_non_mp_edges, color="red", label="Fraction of non-MP Edges"
        )
        ax2_twin.set_ylabel("Fraction of non-MP Edges", color="red")
        ax2_twin.tick_params(axis="y", labelcolor="red")

        # Add combined legend for bottom panel
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

        ax2.set_title("Parsimony Scores and Non-MP Edges by Tree Index")

        fig.tight_layout()
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
        model_name (str): Name of the model to evaluate.
        train_data_name (str): Name of the training dataset.
        trained_model_ckpt (str): Path to the trained model checkpoint.
        test_data_name (str): Name of the test dataset.
        device (str): Device to run the model on.
        hyperparameter_path (str): Path to the hyperparameters file.
        output_dir (str): Output directory.
        data_nicknames_path (str): Path to the dataset nicknames file.
        output_file (str): Optional path to save the evaluation results.

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
        eval_paths (list): List of paths to CSV files to concatenate
        model_names (list): List of model names to match in filenames
        train_data_names (list): List of training dataset names to match in filenames
        test_data_names (list): List of test dataset names to match in filenames
        param_ids (list): List of parameter IDs to match in filenames
    Returns:
        pandas.DataFrame: Concatenated dataframe with metadata columns
    """
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
                test_data_name = None
        for test_data in test_data_names:
            if test_data in file_name:
                test_data_name = test_data
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

    # Combine all dataframes
    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure column order is consistent - metadata columns first
    metadata_cols = ["model_name", "train_data_name", "test_data_name", "param_id"]
    other_cols = [col for col in combined_df.columns if col not in metadata_cols]
    combined_df = combined_df[metadata_cols + other_cols]
    combined_df.to_csv(summary_file, index=False)


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
        multiple models/datasets.
        data_nicknames_file (str): Path to the dataset nicknames JSON file.
        test_data_name (str): Nickname of the testing dataset.
        output_file_basename (str): Path to basename for file saving plot.
        fasta_dir (str): Path to the directory containing the FASTA files.
        metrics (list): Metrics to plot (default: ["auroc"]).
        compare_by (str): What to compare - "model" or "training_data".
        fixed_model (str): When compare_by="training_data", the model to fix.
        fixed_training_data (str): When compare_by="model", the training data to fix.
    """
    # Load data
    with open(data_nicknames_file, "r") as f:
        dataset_dict = json.load(f)
    test_data_path = os.path.join(
        dataset_dict["data_dir"], dataset_dict[test_data_name]
    )
    with open(test_data_path, "rb") as f:
        data_dict = pickle.load(f)

    # Calculate parsimony scores and non-MP edges
    data_basename = dataset_dict[test_data_name].split("_tree_search")[0]
    fasta_path = fasta_dir + "/" + data_basename + "/" + data_basename + ".fasta"

    parsimony_scores = get_parsimony_scores(list(data_dict.keys()), fasta_path)
    num_int_edges = len(list(data_dict.keys())[0]) - 2
    frac_non_mp_edges = [sum(l) / num_int_edges for l in data_dict.values()]

    # Read the comparison data
    df = pd.read_csv(csv_file)

    print("compare_by", compare_by)
    print("fixed_model", fixed_model)
    print("fixed_training_data", fixed_training_data)

    # # Filter data based on comparison type
    if compare_by == "model":
        if fixed_training_data is None:
            raise ValueError("Must specify fixed_training_data when compare_by='model'")
        filtered_df = df[df["train_data_name"] == fixed_training_data]
        comparison_column = "model_name"
        title_prefix = (
            f"Models trained on {fixed_training_data}, tested on {test_data_name}"
        )
    elif compare_by == "training_data":
        if fixed_model is None:
            raise ValueError("Must specify fixed_model when compare_by='training_data'")
        filtered_df = df[df["model_name"] == fixed_model]
        comparison_column = "train_data_name"
        title_prefix = f"Model {fixed_model} trained on different datasets, tested on {test_data_name}"
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

    palette = sns.color_palette("Dark2", len(comparison_values))
    color_dict = dict(zip(comparison_values, palette))

    # Create plots for each metric
    for metric in metrics:
        metric_label = metric_labels.get(metric, metric)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1]}
        )

        # Top panel: Plot metrics for each comparison value
        for value in comparison_values:
            value_df = filtered_df[filtered_df[comparison_column] == value]
            value_df = value_df.sort_values("tree_idx")
            ax1.plot(
                value_df["tree_idx"],
                value_df[metric],
                "o-",
                color=color_dict[value],
                label=value,
                alpha=0.7,
                markersize=4,
            )

        ax1.set_xlabel("")  # No x-label on top panel
        ax1.set_ylabel(metric_label)
        ax1.set_title(f"{title_prefix} - {metric_label} by Tree Index")
        ax1.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title=comparison_column.replace("_", " ").title(),
        )
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Bottom panel: Plot parsimony scores AND non-MP edges (same as original
        # function) Primary y-axis for parsimony scores
        if parsimony_scores and not all(pd.isna(score) for score in parsimony_scores):
            x_range_scores = range(len(parsimony_scores))
            ax2.plot(
                x_range_scores, parsimony_scores, color="green", label="Parsimony Score"
            )
            ax2.set_xlabel("Tree Index")
            ax2.set_ylabel("Parsimony Score", color="green")
            ax2.tick_params(axis="y", labelcolor="green")
        else:
            ax2.set_title("No parsimony scores available")

        # Secondary y-axis for non-MP edges
        ax2_twin = ax2.twinx()
        x_range = range(len(frac_non_mp_edges))
        ax2_twin.plot(
            x_range, frac_non_mp_edges, color="red", label="Fraction of non-MP Edges"
        )
        ax2_twin.set_ylabel("Fraction of non-MP Edges", color="red")
        ax2_twin.tick_params(axis="y", labelcolor="red")

        # Add combined legend for bottom panel
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )

        ax2.set_title("Parsimony Scores and Non-MP Edges by Tree Index")
        ax2.grid(True, linestyle="--", alpha=0.7)

        # Add a suptitle to the entire figure
        fig.suptitle(f"Performance Comparison - {test_data_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle

        # Save figure
        output_file = f"{output_file_basename}-{metric}.pdf"
        print(output_file)
        plt.savefig(output_file)
        plt.close(fig)
