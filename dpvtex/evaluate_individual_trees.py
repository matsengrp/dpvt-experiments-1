import json
import os
import re

import pandas as pd
import torch
from torchmetrics import AUROC

from dpvtex.dpvt_data import data_of_nicknames
from dpvtex.dpvt_zoo import build_model

torch.set_num_threads(1)

TREE_SEARCH_SUFFIX = "_tree_search"
REPLICATE_SEPARATOR = "_rep"


def extract_alignment_base(name: str) -> str:
    """Extract base alignment name, e.g. "alignment_rep1_tree_search" -> "alignment"."""
    return name.replace(TREE_SEARCH_SUFFIX, "").split(REPLICATE_SEPARATOR)[0]


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
    # Load the test data and model
    test_data = data_of_nicknames(test_data_name, device, data_nicknames_path)
    model, device = load_model(
        model_name, hyperparameter_path, trained_model_ckpt, device
    )

    # Initialize metrics for each tree
    results = []

    # Process each tree individually
    for i in range(len(test_data)):
        # For TraversalDataset, extract the data for a single tree
        # Keep batch dimension but select single tree
        traversal = test_data.traversal[i : i + 1]
        mutations = test_data.mutations[i : i + 1]
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


def _compile_replicate_patterns(test_data_names):
    """Build {base_name: compiled_regex} for matching replicates."""
    patterns = {}
    for test_data in test_data_names:
        base_name = extract_alignment_base(test_data)
        patterns[base_name] = re.compile(f"{base_name}(_rep\\d+)?$")
    return patterns


def _parse_metadata_from_filename(
    file_name,
    model_names,
    train_data_names,
    test_data_names,
    param_ids,
    base_test_patterns,
):
    """Extract metadata (model, train data, test data, param ID) from a filename.

    Candidates are tried longest-first so that more specific names
    (e.g. ``..._spr_r2_t0.1``) match before less specific prefixes.

    Args:
        file_name: Basename of the evaluation CSV file.
        model_names: List of model name candidates.
        train_data_names: List of training dataset name candidates.
        test_data_names: List of test dataset name candidates.
        param_ids: List of parameter ID candidates.
        base_test_patterns: Dict from :func:`_compile_replicate_patterns`.

    Returns:
        dict: ``{model_name, train_data_name, test_data_name, param_id}``.
    """
    model_name = None
    for model in sorted(model_names, key=len, reverse=True):
        if model in file_name:
            model_name = model
            break

    train_data_name = None
    for train_data in sorted(train_data_names, key=len, reverse=True):
        if train_data in file_name:
            train_data_name = train_data
            break

    # For test data, handle replicates by using regex patterns
    test_data_name = None

    # First, try exact matches with the test data names (longest first)
    for test_data in sorted(test_data_names, key=len, reverse=True):
        if test_data in file_name:
            test_data_name = test_data
            break

    # If no exact match found, check for replicate pattern matches
    if test_data_name is None:
        for base_name, pattern in base_test_patterns.items():
            rep_match = re.search(r"_rep(\d+)", file_name)
            if rep_match and base_name in file_name:
                rep_num = rep_match.group(1)
                test_data_name = f"{base_name}_rep{rep_num}"
                break

    param_id = None
    for param in sorted(param_ids, key=len, reverse=True):
        if param in file_name:
            param_id = param
            break

    return {
        "model_name": model_name,
        "train_data_name": train_data_name,
        "test_data_name": test_data_name,
        "param_id": param_id,
    }


def concatenate_tree_eval_files(
    eval_paths, model_names, train_data_names, test_data_names, param_ids, summary_file
):
    """Concatenate multiple tree evaluation CSV files and add metadata columns.

    Args:
        eval_paths (list): List of paths to CSV files to concatenate.
        model_names (list): List of model names to match in filenames.
        train_data_names (list): List of training dataset names to match in filenames.
        test_data_names (list): List of test dataset names to match in filenames.
        param_ids (list): List of parameter IDs to match in filenames.
        summary_file (str): Path to write the concatenated CSV.

    Returns:
        pandas.DataFrame: Concatenated dataframe with metadata columns.
    """
    base_test_patterns = _compile_replicate_patterns(test_data_names)

    dfs = []
    for csv_file in eval_paths:
        df = pd.read_csv(csv_file)
        file_name = str(csv_file).split("/")[-1]

        metadata = _parse_metadata_from_filename(
            file_name,
            model_names,
            train_data_names,
            test_data_names,
            param_ids,
            base_test_patterns,
        )
        for key, value in metadata.items():
            df[key] = value

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure column order is consistent - metadata columns first
    metadata_cols = ["model_name", "train_data_name", "test_data_name", "param_id"]
    other_cols = [col for col in combined_df.columns if col not in metadata_cols]
    combined_df = combined_df[metadata_cols + other_cols]

    combined_df.to_csv(summary_file, index=False)
    return combined_df
