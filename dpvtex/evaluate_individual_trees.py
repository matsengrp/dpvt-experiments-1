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


def plot_auroc_over_time(csv_file, data_nicknames_file, test_data_name, output_file_basename, fasta_dir, metrics=["auroc"]):
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
    test_data_path = os.path.join(dataset_dict["data_dir"], dataset_dict[test_data_name])
    with open(test_data_path, "rb") as f:
        data_dict = pickle.load(f)
    
    # We require the test_data_name to be in the format "{fasta_basename}_tree_search_test.fasta"
    # This is how they are generated in our pipeline, so this should generally be fine
    data_basename = dataset_dict[test_data_name].split("_tree_search")[0]
    fasta_path = fasta_dir + "/" + data_basename + "/" + data_basename + ".fasta"
    
    parsimony_scores = get_parsimony_scores(list(data_dict.keys()), fasta_path)
    num_int_edges = len(list(data_dict.keys())[0]) - 2
    frac_non_mp_edges = [sum(l)/num_int_edges for l in data_dict.values()]

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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1]})

        # Top panel: Plot just the metric
        sns.scatterplot(data=df, x="tree_idx", y=metric, ax=ax1, color='blue', label=metric_label)
        ax1.set_xlabel("")  # No x-label on top panel
        ax1.set_ylabel(metric_label, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f"{metric_label} by Tree Index")
        ax1.legend(loc='best')

        # Bottom panel: Plot parsimony scores AND non-MP edges
        # Primary y-axis for parsimony scores
        if parsimony_scores and not all(pd.isna(score) for score in parsimony_scores):
            x_range_scores = range(len(parsimony_scores))
            ax2.plot(x_range_scores, parsimony_scores, color='green', label='Parsimony Score')
            ax2.set_xlabel("Tree Index")
            ax2.set_ylabel("Parsimony Score", color='green')
            ax2.tick_params(axis='y', labelcolor='green')
        else:
            ax2.set_title("No parsimony scores available")
        # Secondary y-axis for non-MP edges
        ax2_twin = ax2.twinx()
        x_range = range(len(frac_non_mp_edges))
        ax2_twin.plot(x_range, frac_non_mp_edges, color='red', label='Fraction of non-MP Edges')
        ax2_twin.set_ylabel("Fraction of non-MP Edges", color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')

        # Add combined legend for bottom panel
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

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
