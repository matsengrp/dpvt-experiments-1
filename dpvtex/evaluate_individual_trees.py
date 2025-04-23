import torch
import json
import pandas as pd
import pickle
import os
from torchmetrics import AUROC
from torchmetrics.classification import BinaryROC
import matplotlib.pyplot as plt
import seaborn as sns
from dpvtex.dpvt_data import data_of_nicknames
from dpvtex.dpvt_zoo import build_model


def plot_auroc_over_time(csv_file, data_nicknames_file, train_data_name, output_file):
    """
    Plots AUROC over time for each tree in the
    dataset whose AUROCs are provided in df in order and saves the plot to a file.
    Also adds line for number of non-MP edges in the dataset.
    Args:
        csv_file (str): Path to the CSV file containing AUROC data.
        data_nicknames_file (str): Path to the dataset nicknames JSON file.
        train_data_name (str): Nickname of the training dataset.
        output_file (str): Path to save the plot.
    """
    with open(data_nicknames_file, "r") as f:
        dataset_dict = json.load(f)
    train_data_path = os.path.join(dataset_dict["data_dir"], dataset_dict[train_data_name])
    with open(train_data_path, "rb") as f:
        data_dict = pickle.load(f)
    num_mp_edges = [sum(l) for l in data_dict.values()]

    df = pd.read_csv(csv_file)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot AUROC/accuracy on the primary y-axis
    sns.scatterplot(data=df, x="tree_idx", y="accuracy", ax=ax1, color='blue', label='Accuracy')
    ax1.set_xlabel("Tree Index")
    ax1.set_ylabel("Accuracy", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for MP edges
    ax2 = ax1.twinx()

    # Plot MP edges as a line on the secondary y-axis
    # Make sure the x range matches the data range in the CSV
    x_range = range(min(len(num_mp_edges), len(df)))
    ax2.plot(x_range, num_mp_edges[:len(x_range)], color='red', label='MP Edges')
    ax2.set_ylabel("Number of MP Edges", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a title and legend
    plt.title("Accuracy and MP Edges by Tree Index")

    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Adjust layout and save
    fig.tight_layout()
    plt.savefig(output_file)
    plt.close()


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
