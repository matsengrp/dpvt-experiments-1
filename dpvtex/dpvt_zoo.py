from dpvt import models
from dpvt.wrapper import Wrap, HyperWrap
from dpvtex.dpvt_data import (
    data_of_nicknames,
    train_val_data_of_nicknames,
)
import json
import torch
import os
import tbparse
import pandas as pd
import time
from datetime import datetime
from pathlib import Path


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
    return model


def trained_model_str(model_name, train_data_name, param_id=None):
    if param_id is None:
        param_id = "ParamNull"
    model = f"{model_name}-{train_data_name}-{param_id}"
    return model


def tested_model_str(model_name, train_data_name, test_data_name, param_id=None):
    if param_id is None:
        param_id = "ParamNull"
    model = f"{model_name}-{train_data_name}-ON-{test_data_name}-{param_id}"
    return model


def model_str(model_name, train_data_name, test_data_name=None, param_id=None):
    if test_data_name:
        path = tested_model_str(model_name, train_data_name, test_data_name, param_id)
    else:
        path = trained_model_str(model_name, train_data_name, param_id)
    return path


def prepend_dir_to_path(path, root_dir=None):
    if root_dir is not None:
        path = str(Path(root_dir) / Path(path))
    return path


def append_dir_to_path(path, sub_dir=None):
    if sub_dir is not None:
        path = str(Path(path) / Path(sub_dir))
    return path


def trained_model_path(model_name, train_data_name, param_id):
    path = f"trained_models/{trained_model_str(model_name, train_data_name, param_id)}"
    return path


def tested_model_path(model_name, train_data_name, test_data_name, param_id):
    path = f"tested_models/{tested_model_str(model_name, train_data_name, test_data_name, param_id)}"
    return path


def model_params_path(model_name, train_data_name, param_id):
    path = f"hyper_checkpoints/{trained_model_str(model_name, train_data_name, param_id)}"
    return path


def train_model(
    model_name,
    data_name,
    train_checkpoint,
    device,
    hyperparameter_path,
    accum_grad_batches=1,
    feature_length=32,
    dim_mlp_layers=32,
    profiling=False,
    timestamp=str(todays_date),
    param_id=None,
    **wrap_kwargs,
):
    """
    Creates a model in class `model_name` and trains it on data `data_name`.
    """
    # set final and test checkpoint strings
    if train_checkpoint is None:
        train_checkpoint = f"{trained_model_path(model_name, data_name, param_id)}.ckpt"
    # Update default parameters with any provided keyword arguments
    wrap_params = {**wrap_kwargs}
    train_data, val_data = train_val_data_of_nicknames(data_name, device)
    model = build_model(model_name)
    model_str = trained_model_str(model_name, data_name, param_id)
    wrap = Wrap(
        train_data=train_data,
        val_data=val_data,
        test_data=None,
        model=model,
        log_path=model_str,
        profiling=profiling,
        device=device,
        accum_grad_batches=accum_grad_batches,
        feature_length=feature_length,
        dim_mlp_layers=dim_mlp_layers,
        hyperparameter_path=hyperparameter_path,
        added_callbacks=[CustomCallback(name=model_str, log_dir=f"custom_logs/train")],
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
    **wrap_kwargs,
):
    """
    Loads a model in class `model_name` that was previously trained on `data_name`, and
    continue training it.
    """
    # set final and test checkpoint strings
    if train_checkpoint is None:
        train_checkpoint = f"{trained_model_path(model_name, data_name, param_id)}.ckpt"
    # load trained model
    try:
        model = build_model(model_name).load_from_checkpoint(train_checkpoint)
    except FileNotFoundError as e:
        raise ValueError(
            f"Model {model_name} trained on data {data_name} does not have saved checkpoint."
        ) from e

    model_str = trained_model_str(model_name, data_name, param_id)

    # Update default parameters with any provided keyword arguments
    wrap_params = {**wrap_kwargs}
    # load dataset
    train_data, val_data = train_val_data_of_nicknames(data_name, device)
    wrap = Wrap(
        train_data=train_data,
        val_data=val_data,
        test_data=None,
        model=model,
        log_path=model_str,
        device=device,
        epochs=epochs,
        accum_grad_batches=accum_grad_batches,
        feature_length=feature_length,
        dim_mlp_layers=dim_mlp_layers,
        hyperparameter_path=hyperparameter_path,
        added_callbacks=[
            # CustomCallback(name=model_str, log_dir=f"custom_logs/continue/")
        ],
        **wrap_params,
    )
    wrap.train(train_checkpoint)
    return model


def optimize_hyperparameters(
    model_name,
    data_name,
    best_model_hparams_filepath,
    device,
    profiling=False,
    n_trials=100,
    timestamp=str(todays_date),
    param_id=None,
):
    train_data, val_data = train_val_data_of_nicknames(data_name, device)
    model = build_model(model_name)
    model_str = trained_model_str(model_name, data_name, param_id)
    hyper_wrap = HyperWrap(
        model=model,
        train_data=train_data,
        val_data=val_data,
        log_path=model_str,
        checkpoint_dir="hyper_checkpoints/",
        profiling=profiling,
        device=device,
        n_trials=n_trials,
        added_callbacks=[
            # CustomCallback(name=model_str, log_dir=f"custom_logs/optimize/")
        ],
    )
    hyper_wrap.optuna_optimize(best_model_hparams_filepath)
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
    **wrap_kwargs,
):
    """
    Loads a trained model, specified by `model_name` and `trained_data_name`, loads
    it from checkpoint `trained_model_ckpt` and tests it on `test_data_name` dataset
    and saves trained model to checkpoint `test_checkpoint`.
    """
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
    model_str = tested_model_str(
        trained_model_name, train_data_name, test_data_name, param_id
    )
    # load dataset
    test_data = data_of_nicknames(test_data_name, device)
    test_wrap = Wrap(
        train_data=None,
        val_data=None,
        test_data=test_data,
        model=model,
        log_path=model_str,
        device=device,
        accum_grad_batches=accum_grad_batches,
        hyperparameter_path=hyperparameter_path,
        added_callbacks=[
            # CustomCallback(name=model_str, log_dir=f"custom_logs/test")
        ],
        timestamp=timestamp,
        **wrap_params,
    )

    # evaluate model
    test_wrap.test(trained_model_ckpt, test_checkpoint)


def lightning_log_path(
    model_name,
    train_data_name,
    test_data_name,
    device,
    param_id,
    timestamp,
    root_dir=".",
    sub_dir=None,
):
    path = f"lightning_logs/{device}_{timestamp}/{model_str(model_name, train_data_name, test_data_name, param_id)}"
    path = append_dir_to_path(path, sub_dir)
    path = prepend_dir_to_path(path, root_dir)
    return path


def hyperparameter_log_path(
    model_name,
    train_data_name,
    test_data_name,
    device,
    param_id,
    timestamp,
    root_dir=".",
    sub_dir=None,
):
    path = f"hyper_checkpoints/{model_str(model_name, train_data_name, test_data_name, param_id)}"
    path = append_dir_to_path(path, sub_dir)
    path = prepend_dir_to_path(path, root_dir)
    return path


def custom_log_path(
    model_name,
    train_data_name,
    test_data_name,
    device,
    param_id,
    timestamp,
    root_dir=".",
    sub_dir=None,
):
    path = f"custom_logs/train/{model_str(model_name, train_data_name, test_data_name, param_id=param_id)}"
    path = prepend_dir_to_path(path, root_dir)
    return path


def hyperparameter_json_path(
    model_name,
    train_data_name,
    test_data_name,
    device,
    param_id,
    timestamp,
    root_dir=".",
    sub_dir=None,
):
    path = f"hyper_checkpoints/{model_str(model_name, train_data_name, test_data_name, param_id=param_id)}.json"
    path = append_dir_to_path(path, sub_dir)
    path = prepend_dir_to_path(path, root_dir)
    return path


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
    sub_dir = "version_0"
    path_dict = {
        "hyperparameter_json": hyperparameter_json_path(
            model_name=model_name,
            train_data_name=train_data_name,
            test_data_name=None,
            device=device,
            param_id=param_id,
            timestamp=timestamp,
            root_dir=output_dir,
            sub_dir=sub_dir,
        ),
        "hyperparameter_llog": hyperparameter_log_path(
            model_name=model_name,
            train_data_name=train_data_name,
            test_data_name=None,
            device=device,
            param_id=param_id,
            timestamp=timestamp,
            root_dir=output_dir,
            sub_dir=None,
        ),
        "train_llog": lightning_log_path(
            model_name=model_name,
            train_data_name=train_data_name,
            test_data_name=None,
            device=device,
            param_id=param_id,
            timestamp=timestamp,
            root_dir=output_dir,
            sub_dir=sub_dir,
        ),
        "test_llog": lightning_log_path(
            model_name=model_name,
            train_data_name=train_data_name,
            test_data_name=test_data_name,
            device=device,
            param_id=param_id,
            timestamp=timestamp,
            root_dir=output_dir,
            sub_dir=sub_dir,
        ),
        "train_clog": custom_log_path(
            model_name=model_name,
            train_data_name=train_data_name,
            test_data_name=None,
            device=device,
            param_id=param_id,
            timestamp=timestamp,
            root_dir=output_dir,
            sub_dir=sub_dir,
        ),
    }

    # fetch training stats
    df = get_df_from_log(path_dict["train_llog"])
    train_walltime = df[df.tag == "train_final_walltime"].value.iloc[0]
    train_epochs = df[df.tag == "train_final_epoch"].value.iloc[0]
    train_steps = df[df.tag == "train_final_step"].value.iloc[0]
    train_stopped_early = df[df.tag == "train_stopped_early"].value.iloc[0]

    # fetch testing stats
    df = get_df_from_log(path_dict["test_llog"])
    test_auroc = df[df.tag == "test_auroc"].value.iloc[0]
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
            # test auroc
            "test_auroc": [test_auroc],
            "test_loss": [test_loss],
            # runtime
            "train_walltime": [train_walltime],
            # data paths
            "hyperparam_json_path": [path_dict["hyperparameter_json"]],
            "hyperparam_llog_path": [path_dict["hyperparameter_llog"]],
            "train_llog_path": [path_dict["train_llog"]],
            "test_llog_path": [path_dict["test_llog"]],
            "train_clog_path": [path_dict["train_clog"]],
            "trained_model_ckpt_path": [trained_model_ckpt],
            "tested_model_ckpt_path": [tested_model_ckpt],
        }
    )

    df_row.to_csv(csv_output_path)
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
    Callback for logging hyperparameters, total_epochs, number_of_steps, auroc, runtimes
    """

    def __init__(
        self, name="model-traindata-ON-testdata-ParamId", log_dir="custom_logs"
    ):
        self.name = name
        self.log_dir = log_dir
        self.writer = SummaryWriter(f"{log_dir}/{name}")
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
