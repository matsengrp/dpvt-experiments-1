import os
import json
import time
import itertools

from dpvtex.dpvt_zoo import (
    train_model,
    optimize_hyperparameters,
    build_log_path,
    get_trained_model_path,
    get_model_params_path,
)

from dpvtex.dpvt_data import load_nicknames_dict

from dpvtex.evaluate_individual_trees import (
    evaluate_individual_trees,
    evaluate_baseline_reversion_on_trees,
    concatenate_tree_eval_files,
    extract_alignment_base,
)
from dpvtex.treesearch_plots import plot_treesearch_evaluation

# Set threading environment variables to prevent parallel execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# Import config from main Snakefile
configfile: "treesearch_config.yaml"


# Config settings
output_dir = config["output_dir"]
fasta_dir = config["fasta_dir"]
data_nicknames_path = config["data_nicknames_path"]
model_names = config["models"]
train_data_names = config["train_data"]
test_data_names = config["test_data"]
device = config["device"]
timestamp = config["timestamp"]
metrics = config["metrics"]
use_hyperparameter_optimize = bool(config["use_hyperparameter_optimize"])
n_hyperparameter_trials = config.get("n_hyperparameter_trials", 100)
hyperparameters = config["hyperparameters"]
write_benchmarks = config.get("write_benchmarks", True)

# Get expanded dataset dict (with glob patterns resolved)
dataset_dict = load_nicknames_dict(data_nicknames_path)

# Automatically identify baseline models based on name (any model containing "Baseline")
baseline_models = [model for model in model_names if "Baseline" in model]
regular_models = [model for model in model_names if "Baseline" not in model]

# Find all test data sets with replicates
test_data_names_with_reps = []
for test_data_name in test_data_names:
    alignment_base = extract_alignment_base(test_data_name)

    # Find all matching replicate datasets
    # Matches: exact name, or {alignment_base}_rep{i}_tree_search pattern
    all_test_datasets = [
        name
        for name in dataset_dict.keys()
        if name == test_data_name
        or (
            name.startswith(alignment_base)
            and "_rep" in name
            and "_tree_search" in name
        )
    ]
    test_data_names_with_reps += all_test_datasets


def generate_hyperparameter_dicts(hyperparameters, use_hyperparameter_optimize):
    if use_hyperparameter_optimize:
        return ["ParamOpt"], {}

    param_dicts = {}
    keys = hyperparameters.keys()
    values = list(hyperparameters.values())
    for id, combination in enumerate(itertools.product(*values)):
        param_id = f"Param{id}"
        result_dict = dict(zip(keys, combination))
        param_dicts[param_id] = result_dict
    return list(param_dicts.keys()), param_dicts

param_ids, param_dicts = generate_hyperparameter_dicts(
    hyperparameters=hyperparameters,
    use_hyperparameter_optimize=use_hyperparameter_optimize,
)

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


def get_benchmark_path(step_name, model_name, train_data_name, test_data_name, param_id, device, timestamp, output_dir):
    path = build_log_path(
        model_name=model_name,
        train_data_name=train_data_name,
        test_data_name=test_data_name,
        param_id=param_id,
        device=device,
        timestamp=timestamp,
        log_name="benchmarks",
        step_name=step_name,
        output_dir=output_dir,
    )
    return f"{path}.tsv"


def write_benchmark_file(benchmark_path, elapsed_seconds):
    """Write a Snakemake-compatible benchmark TSV with elapsed wall clock time."""
    os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
    h = int(elapsed_seconds // 3600)
    m = int((elapsed_seconds % 3600) // 60)
    s = elapsed_seconds % 60
    with open(benchmark_path, "w") as f:
        f.write("s\th:m:s\tmax_rss\tmax_vms\tmax_uss\tmax_pss\tio_in\tio_out\tmean_load\tcpu_time\n")
        f.write(f"{elapsed_seconds:.4f}\t{h}:{m:02d}:{s:05.2f}\t-\t-\t-\t-\t-\t-\t-\t-\n")


# Generate paths
rep_data_pairs = list(itertools.product(train_data_names, test_data_names_with_reps))

# Generate paths for regular models (requiring training)
regular_eval_paths = [
    get_individual_tree_eval_path(
        model_name=model_name,
        train_data_name=train_data_name,
        test_data_name=test_data_name,
        param_id=param_id,
        device=device,
        timestamp=timestamp,
        output_dir=output_dir,
    )
    for model_name in regular_models
    for train_data_name, test_data_name in rep_data_pairs
    for param_id in param_ids
]

# Generate paths for baseline models (no training required)
baseline_eval_paths = [
    get_individual_tree_eval_path(
        model_name=model_name,
        train_data_name="baseline",
        test_data_name=test_data_name,
        param_id="baseline",
        device="cpu",
        timestamp=timestamp,
        output_dir=output_dir,
    )
    for model_name in baseline_models
    for test_data_name in test_data_names_with_reps
]

# Combine all evaluation paths
eval_paths = regular_eval_paths + baseline_eval_paths

comparison_plot_paths = [
    f"{output_dir}/run.{timestamp}/tree_eval_logs/model_comparison-{test_data}-{compare_by}-{fixed_value}.pdf"
    for test_data in test_data_names
    for compare_by, fixed_value in (
        [("model", td) for td in train_data_names]
        + [("training_data", m) for m in model_names]
    )
]




# Rules
rule all:
    input:
        comparison_plot_paths,


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
        start = time.time()
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
        if write_benchmarks:
            write_benchmark_file(
                get_benchmark_path("train_model", wildcards.model_name, wildcards.train_data_name, "none", wildcards.param_id, device, timestamp, output_dir),
                time.time() - start,
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
        model_name="(?!.*Baseline).*",
    run:
        start = time.time()
        evaluate_individual_trees(
            model_name=wildcards.model_name,
            train_data_name=wildcards.train_data_name,
            trained_model_ckpt=input.trained_model,
            test_data_name=wildcards.test_data_name,
            device=device,
            hyperparameter_path=input.hyperparameter_path,
            output_file=output.eval_path,
            data_nicknames_path=data_nicknames_path,
        )
        if write_benchmarks:
            write_benchmark_file(
                get_benchmark_path("evaluate_individual_trees", wildcards.model_name, wildcards.train_data_name, wildcards.test_data_name, wildcards.param_id, device, timestamp, output_dir),
                time.time() - start,
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
        baseline_model="|".join(baseline_models),
    run:
        start = time.time()
        evaluate_baseline_reversion_on_trees(
            test_data_name=wildcards.test_data_name,
            output_file=output.eval_path,
            data_nicknames_path=data_nicknames_path,
        )
        if write_benchmarks:
            write_benchmark_file(
                get_benchmark_path("evaluate_baseline", wildcards.baseline_model, "baseline", wildcards.test_data_name, "baseline", "cpu", timestamp, output_dir),
                time.time() - start,
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
        alignment_base = extract_alignment_base(wildcards.test_data_name)

        # Find matching test datasets (including replicates)
        matching_test_data = [
            name
            for name in test_data_names_with_reps
            if name.startswith(alignment_base) and "_tree_search" in name
        ]

        if len(matching_test_data) == 0:
            matching_test_data = [wildcards.test_data_name]

            # Determine what we're fixing based on compare_by
        if wildcards.compare_by == "model":
            fixed_model = None
            fixed_training_data = wildcards.fixed_value
        else:  # compare_by == "training_data"
            fixed_model = wildcards.fixed_value
            fixed_training_data = None

        plot_treesearch_evaluation(
            csv_file=input.summary_csv,
            data_nicknames_file=data_nicknames_path,
            test_data_name=matching_test_data,
            output_file=output.plot_model_path,
            fasta_dir=fasta_dir,
            metrics=metrics,
            compare_by=wildcards.compare_by,
            fixed_model=fixed_model,
            fixed_training_data=fixed_training_data,
            include_baseline=(
                wildcards.compare_by == "model"
            ),  # Include baseline models when comparing models
        )
