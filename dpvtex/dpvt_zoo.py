from dpvt import models
from dpvt.wrapper import Wrap, HyperWrap, Wraplet
from dpvtex.dpvt_data import load_nicknames_dict,data_of_nicknames,train_val_data_of_nicknames
import json
import torch
import os
import tbparse
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import numpy as np


from lightning.pytorch.callbacks import Callback
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(1)
from pytorch_lightning import seed_everything

# seed_everything(42, workers=True)
torch.set_default_dtype(torch.float64)  # Set default to float64 for higher precision


todays_date = datetime.now().strftime("%Y-%m-%d")


def build_model(model_name):
    if model_name == "TraverseNN":
        model = models.TraverseNN
    elif model_name == "TraverseMaxPooling":
        model = models.TraverseMaxPooling
    elif model_name == "TraverseAvgPooling":
        model = models.TraverseAvgPooling
    elif model_name == "BaselineReversion":
        model = models.BaselineReversion
    return model


def get_trained_model_str(model_name, train_data_name, param_id):
    model = f"{model_name}-{train_data_name}"
    if param_id != None:
        model = f"{model_name}-{train_data_name}-{param_id}"
    if train_data_name == None:
        model = f"{model_name}"
    return model


def get_tested_model_str(model_name, train_data_name, test_data_name, param_id):
    model = f"{model_name}-{train_data_name}-ON-{test_data_name}"
    if param_id != None:
        model = f"{model_name}-{train_data_name}-ON-{test_data_name}-{param_id}"
    if train_data_name == None:
        model = f"{model_name}-ON-{test_data_name}"
    return model


def get_model_str(model_name, train_data_name, test_data_name=None, param_id=None):
    if test_data_name:
        path = get_tested_model_str(
            model_name, train_data_name, test_data_name, param_id
        )
    else:
        path = get_trained_model_str(model_name, train_data_name, param_id)
    return path


def prepend_dir_to_path(path, output_dir=None):
    if output_dir is not None:
        path = str(Path(output_dir) / Path(path))
    return path


def append_dir_to_path(path, sub_dir=None):
    if sub_dir is not None:
        path = str(Path(path) / Path(sub_dir))
    return path


def build_log_path(
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
    model_str = get_model_str(model_name, train_data_name, test_data_name, param_id)
    path = f"run.{timestamp}/{log_name}/{step_name}/{model_str}"
    path = prepend_dir_to_path(path, output_dir)
    path = f"{os.getcwd()}/{path}"
    return path


def get_trained_model_path(
    model_name, train_data_name, param_id, device, timestamp, output_dir="."
):
    path = build_log_path(
        model_name=model_name,
        train_data_name=train_data_name,
        test_data_name=None,
        param_id=param_id,
        device=device,
        timestamp=timestamp,
        log_name="checkpoint_logs",
        step_name="train_model",
        output_dir=output_dir,
    )
    return path


def get_tested_model_path(
    model_name,
    train_data_name,
    test_data_name,
    param_id,
    device,
    timestamp,
    output_dir=".",
):
    path = build_log_path(
        model_name=model_name,
        train_data_name=train_data_name,
        test_data_name=test_data_name,
        param_id=param_id,
        device=device,
        timestamp=timestamp,
        log_name="checkpoint_logs",
        step_name="test_model",
        output_dir=output_dir,
    )
    return path


def get_model_params_path(
    model_name,
    train_data_name,
    param_id,
    device,
    timestamp,
    output_dir=".",
):
    path = build_log_path(
        model_name=model_name,
        train_data_name=train_data_name,
        test_data_name=None,
        param_id=param_id,
        device=device,
        timestamp=timestamp,
        log_name="checkpoint_logs",
        step_name="optimize_hyperparameters",
        output_dir=output_dir,
    )
    return path


def get_baseline_result_path(
    model_name,
    test_data_name,
    timestamp,
    output_dir=".",
):
    """
    Generate a path for baseline model test results. This is independent from
    other path functions, as we don't require training for baseline models.
    
    Args:
        model_name: Name of the baseline model
        test_data_name: Name of the test dataset
        timestamp: Timestamp for the run
        output_dir: Base output directory
        
    Returns:
        Path to the baseline result JSON file
    """
    from pathlib import Path
    
    # Create a cleaner path structure without param_id
    path = f"run.{timestamp}/baseline_results/{model_name}-ON-{test_data_name}"
    
    # Add output directory if provided
    if output_dir is not None:
        path = str(Path(output_dir) / Path(path))
        
    # Convert to absolute path
    path = f"{os.getcwd()}/{path}"
    
    return f"{path}.json"



def build_paths_dict(
    model_name,
    train_data_name,
    test_data_name,
    device,
    param_id,
    timestamp,
    output_dir=".",
):
    def _get_path(test_data_name, log_name, step_name):
        path = build_log_path(
            model_name=model_name,
            train_data_name=train_data_name,
            test_data_name=test_data_name,
            device=device,
            param_id=param_id,
            timestamp=timestamp,
            log_name=log_name,
            step_name=step_name,
            output_dir=output_dir,
        )
        return path

    dirs = {
        "hyper_json": _get_path(None, "checkpoint_logs", "optimize_hyperparameters"),
        "hyper_llog": _get_path(None, "lightning_logs", "optimize_hyperparameters"),
        "hyper_benchmark": _get_path(None, "benchmark_logs", "optimize_hyperparameters"),
        "hyper_checkpoint": _get_path(
            None, "checkpoint_logs", "optimize_hyperparameters"
        ),
        "train_llog": _get_path(None, "lightning_logs", "train_model"),
        "train_benchmark": _get_path(None, "benchmark_logs", "train_model"),
        "train_clog": _get_path(None, "custom_logs", "train_model"),
        "train_checkpoint": _get_path(None, "checkpoint_logs", "train_model"),
        "test_llog": _get_path(test_data_name, "lightning_logs", "test_model"),
        "test_checkpoint": _get_path(test_data_name, "checkpoint_logs", "test_model"),
    }
    paths = {
        "hyper_json": f"{dirs['hyper_json']}.json",
        "hyper_llog": f"{dirs['hyper_llog']}",
        "hyper_benchmark": f"{dirs['hyper_benchmark']}.tsv",
        "hyper_checkpoint": f"{dirs['hyper_checkpoint']}.ckpt",
        "train_llog": f"{dirs['train_llog']}",
        "train_benchmark": f"{dirs['train_benchmark']}.tsv",
        "train_clog": f"{dirs['train_clog']}",
        "train_checkpoint": f"{dirs['train_checkpoint']}.ckpt",
        "test_llog": f"{dirs['test_llog']}",
        "test_checkpoint": f"{dirs['test_checkpoint']}.ckpt",
    }
    return dirs, paths


def optimize_hyperparameters(
    model_name,
    data_name,
    best_model_hparams_filepath,
    device,
    profiling=False,
    n_trials=100,
    timestamp=str(todays_date),
    param_id=None,
    output_dir=".",
    data_nicknames_path="data_nicknames.json",
):
    dir_dict, path_dict = build_paths_dict(
        model_name=model_name,
        train_data_name=data_name,
        test_data_name=None,
        device=device,
        param_id=param_id,
        timestamp=timestamp,
        output_dir=output_dir
    )
    train_data, val_data = train_val_data_of_nicknames(data_name, device, data_nicknames_path)
    model = build_model(model_name)
    log_path = path_dict["hyper_llog"]
    checkpoint_dir = dir_dict["hyper_checkpoint"]

    hyper_wrap = HyperWrap(
        model=model,
        train_data=train_data,
        val_data=val_data,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        profiling=profiling,
        device=device,
        n_trials=n_trials,
        added_callbacks=[],
    )
    hyper_wrap.optuna_optimize(best_model_hparams_filepath)
    return model


def train_model(
    model_name,
    data_name,
    train_checkpoint,
    device,
    hyperparameter_path,
    profiling=False,
    timestamp=str(todays_date),
    param_id=None,
    output_dir=".",
    data_nicknames_path="data_nicknames.json",
    **wrap_kwargs,
):
    """
    Creates a model in class `model_name` and trains it on data `data_name`.
    """
    dir_dict, path_dict = build_paths_dict(
        model_name=model_name,
        train_data_name=data_name,
        test_data_name=None,
        device=device,
        param_id=param_id,
        timestamp=timestamp,
        output_dir=output_dir
    )
    # set final and test checkpoint strings
    if train_checkpoint is None:
        train_checkpoint = path_dict["train_checkpoint"]
    # Update default parameters with any provided keyword arguments
    wrap_params = {**wrap_kwargs}
    train_data, val_data = train_val_data_of_nicknames(data_name, device, data_nicknames_path)
    model = build_model(model_name)
    log_path = path_dict["train_llog"]
    custom_log_path = path_dict["train_clog"]

    wrap = Wrap(
        train_data=train_data,
        val_data=val_data,
        test_data=None,
        model=model,
        log_path=log_path,
        profiling=profiling,
        device=device,
        hyperparameter_path=hyperparameter_path,
        added_callbacks=[CustomCallback(log_path=custom_log_path)],
        timestamp=timestamp,
        **wrap_params,
    )
    wrap.train(train_checkpoint)
    return model


def continue_train_model(
    model_name,
    data_name,
    device,
    hyperparameter_path,
    epochs=200,
    accum_grad_batches=1,
    feature_length=32,
    dim_mlp_layers=32,
    train_checkpoint=None,
    timestamp=str(todays_date),
    param_id=None,
    output_dir=".",
    data_nicknames_path="data_nicknames.json",
    **wrap_kwargs,
):
    """
    Loads a model in class `model_name` that was previously trained on `data_name`, and
    continue training it.
    """
    dir_dict, path_dict = build_paths_dict(
        model_name=model_name,
        train_data_name=data_name,
        test_data_name=None,
        device=device,
        param_id=param_id,
        timestamp=timestamp,
        output_dir=output_dir
    )
    # set final and test checkpoint strings
    if train_checkpoint is None:
        train_checkpoint = path_dict["train_checkpoint"]
    # load trained model
    try:
        model = build_model(model_name).load_from_checkpoint(train_checkpoint)
    except FileNotFoundError as e:
        raise ValueError(
            f"Model {model_name} trained on data {data_name} does not have saved checkpoint."
        ) from e

    # Update default parameters with any provided keyword arguments
    wrap_params = {**wrap_kwargs}
    # load dataset
    train_data, val_data = train_val_data_of_nicknames(data_name, device, data_nicknames_path)
    log_path = path_dict["train_llog"]
    custom_log_path = path_dict["train_clog"]

    wrap = Wrap(
        train_data=train_data,
        val_data=val_data,
        test_data=None,
        model=model,
        log_path=log_path,
        device=device,
        epochs=epochs,
        accum_grad_batches=accum_grad_batches,
        feature_length=feature_length,
        dim_mlp_layers=dim_mlp_layers,
        hyperparameter_path=hyperparameter_path,
        added_callbacks=[CustomCallback(log_path=custom_log_path)],
        **wrap_params,
    )
    wrap.train(train_checkpoint)
    return model


def test_model(
    trained_model_name,
    train_data_name,
    trained_model_ckpt,
    test_data_name,
    test_checkpoint,
    device,
    hyperparameter_path,
    accum_grad_batches=1,
    timestamp=str(todays_date),
    param_id=None,
    output_dir=".",
    data_nicknames_path="data_nicknames.json",
    **wrap_kwargs,
):
    """
    Loads a trained model, specified by `model_name` and `trained_data_name`, loads
    it from checkpoint `trained_model_ckpt` and tests it on `test_data_name` dataset
    and saves trained model to checkpoint `test_checkpoint`.
    """
    dir_dict, path_dict = build_paths_dict(
        model_name=trained_model_name,
        train_data_name=train_data_name,
        test_data_name=test_data_name,
        device=device,
        param_id=param_id,
        timestamp=timestamp,
        output_dir=output_dir
    )
    # Update default parameters with any provided keyword arguments
    wrap_params = {**wrap_kwargs}
    model = build_model(trained_model_name)
    with open(hyperparameter_path, "r") as f:
        hparams = json.load(f)
    model.load_from_checkpoint(
        trained_model_ckpt,
        learning_rate=hparams["learning_rate"],
        feature_length=hparams["feature_length"],
        dim_mlp_layers=hparams["dim_mlp_layers"],
    )
    # load dataset
    test_data = data_of_nicknames(test_data_name, device, data_nicknames_path)
    log_path = path_dict["test_llog"]

    test_wrap = Wrap(
        train_data=None,
        val_data=None,
        test_data=test_data,
        model=model,
        log_path=log_path,
        device=device,
        accum_grad_batches=accum_grad_batches,
        hyperparameter_path=hyperparameter_path,
        added_callbacks=[],
        timestamp=timestamp,
        **wrap_params,
    )

    # evaluate model
    test_wrap.test(trained_model_ckpt, test_checkpoint)


def test_baseline_model(
    model_name,
    test_data_name,
    result_path,
    timestamp=str(todays_date),
    output_dir=".",
    data_nicknames_path="data_nicknames.json",
    **wrap_kwargs
):
    """
    Tests a baseline model on a dataset without requiring training.
    
    Args:
        model_name: Name of the baseline model to test
        test_data_name: Nickname of the test dataset
        result_path: Path to save test results
        timestamp: Timestamp for logging
        output_dir: Output directory
        data_nicknames_path: Path to data nicknames JSON file
        **wrap_kwargs: Additional parameters for the Wraplet
    
    Returns:
        Test results
    """
    # Build model based on name
    model = build_model(model_name)
    device = "cpu"  # Always use CPU for baseline models
    # Load test data
    test_data = data_of_nicknames(test_data_name, device, data_nicknames_path, data_struct="TreeDataset")
    
    # Create lightweight wrapper for testing
    test_wrap = Wraplet(
        test_data=test_data,
        model=model,
        device="cpu",
        **wrap_kwargs
    )
    
    # Test the model and get results
    results = test_wrap.test()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # Save results to JSON
    result_data = {
        'model': model_name,
        'dataset': test_data_name,
        'device': "cpu",
        'timestamp': timestamp,
        'results': results[0] if results else {}
    }
    
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Baseline model {model_name} tested on {test_data_name}, results saved to {result_path}")
    return results


def get_df_from_log(log_path):
    reader = tbparse.SummaryReader(log_path)
    return reader.scalars


def aggregate_data_to_csv(
    model_name,
    train_data_name,
    test_data_name,
    device,
    timestamp,
    param_id,
    n_hyperparameter_trials,
    hyperparameter_path,
    trained_model_ckpt,
    tested_model_ckpt,
    csv_output_path,
    output_dir=".",
):
    """
    Aggregate result data in a CSV entry.
    """

    # fetch hyperparams
    with open(hyperparameter_path) as f:
        opt_hyperparams = json.load(f)

    # fetch logs
    dir_dict, path_dict = build_paths_dict(
        model_name=model_name,
        train_data_name=train_data_name,
        test_data_name=test_data_name,
        device=device,
        timestamp=timestamp,
        param_id=param_id,
        output_dir=output_dir,
    )

    # fetch training stats
    df = get_df_from_log(path_dict["train_llog"])
    train_walltime = df[df.tag == "train_final_walltime"].value.iloc[0]
    train_epochs = df[df.tag == "train_final_epoch"].value.iloc[0]
    train_steps = df[df.tag == "train_final_step"].value.iloc[0]
    train_stopped_early = df[df.tag == "train_stopped_early"].value.iloc[0]

    # fetch testing stats
    df = get_df_from_log(path_dict["test_llog"])
    test_auroc = df[df.tag == "test_auroc"].value.iloc[0]
    test_accuracy = df[df.tag == "test_accuracy"].value.iloc[0]
    test_loss = df[df.tag == "test_loss"].value.iloc[0]

    df_row = pd.DataFrame(
        {
            # config settings
            "model": [model_name],
            "train_data": [train_data_name],
            "test_data": [test_data_name],
            "device": [device],
            "timestamp": [timestamp],
            "param_id": [param_id],
            "n_hyperparam_trials": [n_hyperparameter_trials],
            # hyperparameters
            "learning_rate": [opt_hyperparams["learning_rate"]],
            "batch_size": [opt_hyperparams["batch_size"]],
            "accum_grad_batches": [opt_hyperparams["accum_grad_batches"]],
            "max_epochs": [opt_hyperparams["epochs"]],
            "feature_length": [opt_hyperparams["feature_length"]],
            "dim_mlp_layers": [opt_hyperparams["dim_mlp_layers"]],
            # number of training steps, epochs
            "train_steps": [train_steps],
            "train_epochs": [train_epochs],
            "train_stopped_early": [train_stopped_early],
            "test_auroc": [test_auroc],
            "test_loss": [test_loss],
            "test_accuracy": [test_accuracy],
            # runtime
            "train_walltime": [train_walltime],
            # data paths
            "trained_model_checkpoint": [trained_model_ckpt],
            "tested_model_checkpoint": [tested_model_ckpt],
        }
    )
    for key, path in path_dict.items():
        df_row[f"{key}_path"] = path

    df_row.to_csv(csv_output_path)
    return


def aggregate_baseline_data_to_csv(
    model_name,
    test_data_name,
    timestamp,
    result_path,
    csv_output_path,
    output_dir=".",
):
    """
    Aggregate baseline model result data in a CSV entry.
    
    Args:
        model_name: Name of the baseline model
        test_data_name: Name of the test dataset
        timestamp: Timestamp for the run
        result_path: Path to the baseline result JSON file
        csv_output_path: Path to save the CSV output
        output_dir: Base output directory
    """
    # Read results from JSON file
    with open(result_path, 'r') as f:
        result_data = json.load(f)
    
    # Extract AUROC from results
    test_auroc = np.nan
    if 'results' in result_data and 'test_auroc' in result_data['results']:
        test_auroc = result_data['results']['test_auroc']
    device="cpu" #always cpu for baseline
    # Create DataFrame with results.
    # Columns match those from more complex models and are filled with NaN where
    # appropriate
    df_row = pd.DataFrame(
        {
            # config settings
            "model": [model_name],
            "train_data": ["baseline"],
            "test_data": [test_data_name],
            "device": [device],
            "timestamp": [timestamp],
            "param_id": ["baseline"],  # Use "baseline" as a placeholder
            "n_hyperparam_trials": [np.nan],
            # hyperparameters - not applicable for baseline
            "learning_rate": [np.nan],
            "batch_size": [np.nan],
            "accum_grad_batches": [np.nan],
            "max_epochs": [np.nan],
            "feature_length": [np.nan],
            "dim_mlp_layers": [np.nan],
            # number of training steps, epochs - not applicable
            "train_steps": [np.nan],
            "train_epochs": [np.nan],
            "train_stopped_early": [np.nan],
            # test auroc
            "test_auroc": [test_auroc],
            "test_loss": [np.nan],
            # runtime
            "train_walltime": [np.nan],
            # data paths
            "trained_model_checkpoint": ["none"],
            "tested_model_checkpoint": [result_path],
        }
    )
    
    # Add paths if necessary to match standard CSV format
    path_dict = {
        "hyper_json": "none",
        "hyper_llog": "none",
        "hyper_benchmark": "none",
        "hyper_checkpoint": "none",
        "train_llog": "none",
        "train_benchmark": "none",
        "train_clog": "none",
        "train_checkpoint": "none",
        "test_llog": "none",
        "test_checkpoint": result_path,
    }
    
    for key, path in path_dict.items():
        df_row[f"{key}_path"] = path
    
    # Save to CSV
    df_row.to_csv(csv_output_path)
    print(f"Baseline results written to {csv_output_path}")
    return



def concatenate_csvs(
    input_csv_paths,
    output_csv_path,
):
    """
    Concatenates multiple CSV files into one CSV file.
    """

    dfs = []
    for csv_path in input_csv_paths:
        df = pd.read_csv(csv_path)
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)
    result_df.to_csv(output_csv_path, index=False)


class CustomCallback(Callback):
    """
    Callback for logging hyperparameters, total_epochs, number_of_steps, auroc, runtimes.
    """

    def __init__(self, log_path):
        self.log_path = log_path
        self.writer = SummaryWriter(f"{log_path}")
        self.start_time = {}

    def log_start(self, trainer, pl_module, prefix=""):
        self.total_steps = 0
        self.start_time[prefix] = time.time()
        self.total_epoch_loss = 0
        self.total_epoch_measures = 0

    def log_end(self, trainer, pl_module, prefix=""):
        total_time = time.time() - self.start_time[prefix]
        trainer.logger.log_metrics({f"{prefix}_final_walltime": total_time})
        trainer.logger.log_metrics({f"{prefix}_final_epoch": trainer.current_epoch})
        trainer.logger.log_metrics({f"{prefix}_final_step": self.total_steps})
        stopped_early = trainer.current_epoch + 1 < trainer.max_epochs
        trainer.logger.log_metrics({f"{prefix}_stopped_early": stopped_early})

    def log_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, prefix=""):
        self.total_steps += 1
        loss = outputs["loss"] if "loss" in outputs else None
        if loss is not None:
            self.total_epoch_measures += 1
            self.total_epoch_loss += loss
            self.writer.add_scalar(
                "loss_per_batch",
                loss,
                self.total_steps,
                walltime=(time.time() - self.start_time[prefix]),
            )
        self.writer.add_scalar(
            "walltime_per_batch",
            (time.time() - self.start_time[prefix]),
            self.total_steps,
            walltime=(time.time() - self.start_time[prefix]),
        )

    def log_epoch_end(self, trainer, pl_module, prefix=""):
        avg_loss = self.total_epoch_loss / self.total_epoch_measures
        self.writer.add_scalar(
            "avgloss_per_epoch",
            avg_loss,
            trainer.current_epoch,
            walltime=(time.time() - self.start_time[prefix]),
        )
        self.writer.add_scalar(
            "walltime_per_epoch",
            (time.time() - self.start_time[prefix]),
            trainer.current_epoch,
            walltime=(time.time() - self.start_time[prefix]),
        )
        self.total_epoch_steps = 0
        self.total_epoch_loss = 0
        pass

    def close(self):
        self.writer.close()

    # hooks

    def on_train_start(self, trainer, pl_module):
        self.log_start(trainer, pl_module, "train")

    def on_train_end(self, trainer, pl_module):
        self.log_end(trainer, pl_module, "train")
        self.close()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_batch_end(trainer, pl_module, outputs, batch, batch_idx, "train")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_epoch_end(trainer, pl_module, "train")
