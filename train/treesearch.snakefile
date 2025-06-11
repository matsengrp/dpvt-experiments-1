import sys, os, shutil
from datetime import datetime
import json, yaml
from pathlib import Path
import pandas as pd
import re

from dpvtex.dpvt_zoo import (
    build_log_path,
    get_trained_model_path,
    get_model_params_path,
    get_baseline_result_path,
)

from dpvtex.evaluate_individual_trees import (
    evaluate_individual_trees, 
    evaluate_baseline_reversion_on_trees,
    concatenate_tree_eval_files,
    plot_treesearch_evaluation
)

# Import config from main Snakefile
configfile: "config.yaml"

# Config settings
working_dir = os.getcwd(),
output_dir = config["output_dir"]
fasta_dir = config["fasta_dir"]
data_nicknames_path = config["data_nicknames_path"]
model_names = config["models"]
train_data_names = config["train_data"]
test_data_names = config["test_data"]
device = config["device"]
timestamp = config["timestamp"]
metrics = config["metrics"]
num_replicates = config["replicates"]

# Automatically identify baseline models based on name (any model containing "Baseline")
baseline_models = [model for model in model_names if "Baseline" in model]
regular_models = [model for model in model_names if "Baseline" not in model]

# Find all test data sets with replicates
with open(data_nicknames_path, "r") as f:
    dataset_dict = json.load(f)

test_data_names_with_reps = []
for test_data_name in test_data_names:
    # Check if we have replicate datasets (test_data_name_rep1, test_data_name_rep2, etc.)
    base_name = test_data_name.split("_rep")[0]

    # Find all matching replicate datasets
    all_test_datasets = [name for name in dataset_dict.keys() 
                        if name == test_data_name or 
                        (name.startswith(base_name) and "_rep" in name)]
    test_data_names_with_reps += all_test_datasets


if bool(config["use_hyperparameter_optimize"]):
    param_ids = ["ParamOpt"]
else:
    param_ids = ["Param0"]

# Helper functions
def get_trained_model_ckpt(model, train_data, param_id, device, timestamp, output_dir):
    path = get_trained_model_path(
        model, train_data, param_id, device, timestamp, output_dir
    )
    return f"{path}.ckpt"

def get_model_params_json(model, train_data, param_id, device, timestamp, output_dir):
    path = get_model_params_path(
        model, train_data, param_id, device, timestamp, output_dir
    )
    return f"{path}.json"

def get_individual_tree_eval_path(
    model_name,
    train_data_name,
    test_data_name,
    param_id,
    device,
    timestamp,
    output_dir,
):
    path = build_log_path(
        model_name=model_name,
        train_data_name=train_data_name,
        test_data_name=test_data_name,
        param_id=param_id,
        device=device,
        timestamp=timestamp,
        log_name="tree_eval_logs",
        step_name="individual_tree_eval",
        output_dir=output_dir,
    )
    return f"{path}.csv"


def generate_data_pairs(train_data_names, test_data_names):
    data_pairs = []
    for train_data in train_data_names:
        for test_data in test_data_names:
            data_pairs.append((train_data, test_data))
    return data_pairs


def generate_individual_tree_eval_paths(
    model_names,
    data_pairs,
    param_ids,
    device,
    timestamp,
    output_dir,
):
    eval_paths = [
        get_individual_tree_eval_path(
            model_name=model_name,
            train_data_name=train_data_name,
            test_data_name=test_data_name,
            param_id=param_id,
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        )
        for model_name in model_names
        for train_data_name, test_data_name in data_pairs
        for param_id in param_ids
    ]
    return eval_paths


def generate_individual_tree_eval_plot_paths(
    model_names,
    data_pairs,
    param_ids,
    device,
    timestamp,
    output_dir,
    metrics = ["auroc"],
):
    eval_paths = [
        get_individual_tree_eval_path(
            model_name=model_name,
            train_data_name=train_data_name,
            test_data_name=test_data_name,
            param_id=param_id,
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        )[:-4] + "_" + metric + ".pdf"
        for model_name in model_names
        for train_data_name, test_data_name in data_pairs
        for param_id in param_ids
        for metric in metrics
    ]
    return eval_paths


def generate_model_comparison_plot_paths(
    test_data_names,
    model_names,
    train_data_names,
    metrics=["auroc"],
    output_dir=".",
    timestamp="latest",
):
    """Generate paths for model comparison plots."""
    plot_paths = []

    # Generate paths for comparing different models with fixed training data
    for test_data in test_data_names:
        for train_data in train_data_names:
            for metric in metrics:
                plot_paths.append(
                    f"{output_dir}/run.{timestamp}/tree_eval_logs/model_comparison_{test_data}-model-{train_data}-{metric}.pdf"
                )

    # Generate paths for comparing different training datasets with fixed model
    for test_data in test_data_names:
        for model in model_names:
            for metric in metrics:
                plot_paths.append(
                    f"{output_dir}/run.{timestamp}/tree_eval_logs/model_comparison_{test_data}-training_data-{model}-{metric}.pdf"
                )
    return plot_paths


def generate_comparison_plot_paths(
    test_data_names,
    model_names,
    train_data_names,
    output_dir=".",
    timestamp="latest",
):
    """Generate paths for model comparison plots."""
    plot_paths = []

    # Generate paths for comparing different models with fixed training data
    for test_data in test_data_names:
        for train_data in train_data_names:
            plot_paths.append(
                f"{output_dir}/run.{timestamp}/tree_eval_logs/model_comparison-{test_data}-model-{train_data}.pdf"
            )

    # Generate paths for comparing different training datasets with fixed model
    for test_data in test_data_names:
        for model in model_names:
            plot_paths.append(
                f"{output_dir}/run.{timestamp}/tree_eval_logs/model_comparison-{test_data}-training_data-{model}.pdf"
            )
    return plot_paths


def generate_baseline_eval_paths(
    baseline_models,
    test_data_names,
    timestamp,
    output_dir,
):
    """Generate evaluation paths for baseline models (which don't require training)."""
    eval_paths = [
        get_individual_tree_eval_path(
            model_name=model_name,
            train_data_name="baseline",  # Use "baseline" instead of a real training dataset
            test_data_name=test_data_name,
            param_id="baseline",  # Use "baseline" instead of a real parameter ID
            device="cpu",  # Baseline models only run on CPU
            timestamp=timestamp,
            output_dir=output_dir,
        )
        for model_name in baseline_models
        for test_data_name in test_data_names
    ]
    return eval_paths


# Generate paths
data_pairs = generate_data_pairs(train_data_names, test_data_names)
rep_data_pairs = generate_data_pairs(train_data_names, test_data_names_with_reps)

# Generate paths for regular models (requiring training)
regular_eval_paths = generate_individual_tree_eval_paths(
    model_names=regular_models,
    data_pairs=rep_data_pairs,
    param_ids=param_ids,
    device=device,
    timestamp=timestamp,
    output_dir=output_dir,
)

# Generate paths for baseline models (no training required)
baseline_eval_paths = generate_baseline_eval_paths(
    baseline_models=baseline_models,
    test_data_names=test_data_names_with_reps,
    timestamp=timestamp,
    output_dir=output_dir,
)

# Combine all evaluation paths
eval_paths = regular_eval_paths + baseline_eval_paths

comparison_plot_paths = generate_comparison_plot_paths(
    test_data_names=test_data_names,
    model_names=model_names,
    train_data_names=train_data_names,
    output_dir=output_dir,
    timestamp=timestamp,
)


# Specify ruleorder - baseline models should be handled by the evaluate_baseline_models rule
ruleorder: evaluate_baseline_model > evaluate_individual_trees

# Rules
rule all:
    input:
        comparison_plot_paths


rule optimize_hyperparameters_step:
    output:
        hyperparameter_path=get_model_params_json(
            model="{model_name}",
            train_data="{train_data_name}",
            param_id="{param_id}",
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        ),
    run:
        if use_hyperparameter_optimize:
            optimize_hyperparameters(
                model_name=wildcards.model_name,
                data_name=wildcards.train_data_name,
                best_model_hparams_filepath=output.hyperparameter_path,
                device=device,
                profiling=False,
                n_trials=n_hyperparameter_trials,
                timestamp=timestamp,
                param_id=wildcards.param_id,
                output_dir=output_dir,
                data_nicknames_path=data_nicknames_path,
            )
        else:
            with open(output.hyperparameter_path, "w") as file:
                file.write(json.dumps(param_dicts[wildcards.param_id]))


rule train_model_step:
    input:
        hyperparameter_path=get_model_params_json(
            model="{model_name}",
            train_data="{train_data_name}",
            param_id="{param_id}",
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        ),
    output:
        trained_model=get_trained_model_ckpt(
            model="{model_name}",
            train_data="{train_data_name}",
            param_id="{param_id}",
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        ),
    run:
        train_model(
            model_name=wildcards.model_name,
            data_name=wildcards.train_data_name,
            train_checkpoint=output.trained_model,
            device=device,
            hyperparameter_path=input.hyperparameter_path,
            profiling=False,
            timestamp=timestamp,
            param_id=wildcards.param_id,
            output_dir=output_dir,
            data_nicknames_path=data_nicknames_path,
        )


rule evaluate_individual_trees:
    input:
        trained_model=get_trained_model_ckpt(
            model="{model_name}",
            train_data="{train_data_name}",
            param_id="{param_id}",
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        ),
        hyperparameter_path=get_model_params_json(
            model="{model_name}",
            train_data="{train_data_name}",
            param_id="{param_id}",
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        ),
    output:
        eval_path=get_individual_tree_eval_path(
            model_name="{model_name}",
            train_data_name="{train_data_name}",
            test_data_name="{test_data_name}",
            param_id="{param_id}",
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        ),
    wildcard_constraints:
        # Exclude baseline models from this rule
        model_name="(?!.*Baseline).*"
    run:
        evaluate_individual_trees(
            model_name=wildcards.model_name,
            train_data_name=wildcards.train_data_name,
            trained_model_ckpt=input.trained_model,
            test_data_name=wildcards.test_data_name,
            device=device,
            hyperparameter_path=input.hyperparameter_path,
            output_dir=output_dir,
            data_nicknames_path=data_nicknames_path,
            output_file=output.eval_path,
        )


rule evaluate_baseline_model:
    output:
        eval_path=get_individual_tree_eval_path(
            model_name="{baseline_model}",
            train_data_name="baseline",
            test_data_name="{test_data_name}",
            param_id="baseline",
            device="cpu",
            timestamp=timestamp,
            output_dir=output_dir,
        ),
    wildcard_constraints:
        # only use rule for baseline model
        baseline_model="|".join(baseline_models)
    run:
        evaluate_baseline_reversion_on_trees(
            test_data_name=wildcards.test_data_name,
            output_dir=output_dir,
            data_nicknames_path=data_nicknames_path,
            output_file=output.eval_path,
            timestamp=timestamp,
        )


rule concat_tree_eval:
    """
    Take all csvs for individual tree evaluation and concatenate them into one
    """
    input:
        eval_paths=eval_paths,
    output:
        summary_csv="{output_dir}/run.{timestamp}/tree_eval_logs/tree_eval_summary.csv",
    run:
        # Add "baseline" to the list of training data names and param IDs for concatenation
        all_train_data = train_data_names + ["baseline"]
        all_param_ids = param_ids + ["baseline"]
        
        concatenate_tree_eval_files(
            input.eval_paths,
            model_names,
            all_train_data,
            test_data_names_with_reps,
            all_param_ids,
            output.summary_csv,
        )


rule plot_treesearch_evaluation:
    input:
        summary_csv="{output_dir}/run.{timestamp}/tree_eval_logs/tree_eval_summary.csv",
    output:
        plot_model_path="{output_dir}/run.{timestamp}/tree_eval_logs/model_comparison-{test_data_name}-{compare_by}-{fixed_value}.pdf",
    run:
        # Determine what we're fixing based on compare_by
        if wildcards.compare_by == "model":
            fixed_model = None
            fixed_training_data = wildcards.fixed_value
            
            # Find matching test datasets (including replicates)
            matching_test_data = [name for name in test_data_names_with_reps 
                                 if wildcards.test_data_name in name]
            
            if len(matching_test_data) == 0:
                matching_test_data = [wildcards.test_data_name]
                
            print(f"Test datasets for all metrics comparison: {matching_test_data}")
            
        else:  # compare_by == "training_data"
            fixed_model = wildcards.fixed_value
            fixed_training_data = None
            
            # Find matching test datasets
            matching_test_data = [name for name in test_data_names_with_reps 
                                 if wildcards.test_data_name in name]
            
            if len(matching_test_data) == 0:
                matching_test_data = [wildcards.test_data_name]

        plot_treesearch_evaluation(
            csv_file=input.summary_csv,
            data_nicknames_file=data_nicknames_path,
            test_data_name=matching_test_data,
            output_file=f"{wildcards.output_dir}/run.{wildcards.timestamp}/tree_eval_logs/model_comparison-{wildcards.test_data_name}-{wildcards.compare_by}-{wildcards.fixed_value}.pdf",
            fasta_dir=fasta_dir,
            compare_by=wildcards.compare_by,
            fixed_model=fixed_model,
            fixed_training_data=fixed_training_data,
            include_baseline=(wildcards.compare_by == "model")  # Include baseline models when comparing models
        )