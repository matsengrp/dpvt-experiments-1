import sys, os, shutil
from datetime import datetime
import json, yaml
from pathlib import Path

from dpvtex.dpvt_zoo import (
    build_log_path,
    get_trained_model_path,
    get_model_params_path,
)

from dpvtex.evaluate_individual_trees import (
    evaluate_individual_trees, 
    plot_auroc_over_time
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
metric = config["metric"]

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
    metric = "auroc",
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
    ]
    return eval_paths

# Generate paths
data_pairs = generate_data_pairs(train_data_names, test_data_names)
eval_paths = generate_individual_tree_eval_paths(
    model_names=model_names,
    data_pairs=data_pairs,
    param_ids=param_ids,
    device=device,
    timestamp=timestamp,
    output_dir=output_dir,
)
plot_paths = generate_individual_tree_eval_plot_paths(
    model_names=model_names,
    data_pairs=data_pairs,
    param_ids=param_ids,
    device=device,
    timestamp=timestamp,
    output_dir=output_dir,
    metric = metric,
)

# Rules
rule all:
    input: plot_paths

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

rule plot_individual_tree_eval:
    input:
        eval_path=get_individual_tree_eval_path(
            model_name="{model_name}",
            train_data_name="{train_data_name}",
            test_data_name="{test_data_name}",
            param_id="{param_id}",
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        ),
    output:
        plot_path=get_individual_tree_eval_path(
            model_name="{model_name}",
            train_data_name="{train_data_name}",
            test_data_name="{test_data_name}",
            param_id="{param_id}",
            device=device,
            timestamp=timestamp,
            output_dir=output_dir,
        )[:-4]+ "_" + metric + ".pdf",
    run:
        plot_auroc_over_time(
            input.eval_path,
            data_nicknames_path,
            wildcards.test_data_name,
            output.plot_path,
            fasta_dir,
            metric,
        )