from dpvt import models
from dpvt.wrapper import Wrap, HyperWrap
from dpvtex.dpvt_data import (
    data_of_nicknames,
    train_val_data_of_nicknames,
)
import json
import torch

torch.set_num_threads(1)
from pytorch_lightning import seed_everything

# seed_everything(42, workers=True)
torch.set_default_dtype(torch.float64)  # Set default to float64 for higher precision


def get_model(model_name):
    if model_name == "TraverseNN":
        model = models.TraverseNN
    elif model_name == "TraverseMaxPooling":
        model = models.TraverseMaxPooling
    elif model_name == "TraverseAvgPooling":
        model = models.TraverseAvgPooling
    return model


def trained_model_str(model_name, data_name):
    return f"{model_name}-{data_name}"


def tested_model_str(model_name, train_data_name, test_data_name):
    return f"{model_name}-{train_data_name}-ON-{test_data_name}"


def trained_model_path(model_name, data_name):
    return f"trained_models/{trained_model_str(model_name, data_name)}"


def tested_model_path(model_name, train_data_name, test_data_name):
    return (
        f"tested_models/{tested_model_str(model_name, train_data_name, test_data_name)}"
    )


def best_model_params_path(model_name, data_name):
    return f"hyper_checkpoints/{trained_model_str(model_name, data_name)}"


def lightning_log_path(device, model_name, data_name, date="TIMESTAMP"):
    return f"lightning_logs/{device}_{date}/{trained_model_str(model_name, data_name)}"


def csv_log_path(model_name, train_data_name, test_data_name=None, device=None, date="TIMESTAMP"):
    dir_path = f"csvs/{device}_{date}"
    if test_data_name:
        return f"{dir_path}/{tested_model_str(model_name, train_data_name, test_data_name)}"
    return f"{dir_path}/{trained_model_str(model_name, train_data_name)}.csv"


def generate_csv_log_paths(model_names, data_pairs, device=None, date="TIMESTAMP"):
    csv_log_paths = []
    for model_name in model_names:
        for train_data_name,test_data_name in data_pairs:
            # csv_log_paths.append(csv_log_path(model_name, train_data_name, device=device, date=date))
            csv_log_paths.append(csv_log_path(model_name, train_data_name, test_data_name=test_data_name, device=device, date=date))
    print(f'CSV_LOG_PATHS:\n{csv_log_paths}')
    return csv_log_paths


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
    **wrap_kwargs,
):
    """
    Creates a model in class `model_name` and trains it on data `data_name`.
    """
    # set final and test checkpoint strings
    if train_checkpoint is None:
        train_checkpoint = trained_model_path(model_name, data_name) + ".ckpt"
    # Update default parameters with any provided keyword arguments
    wrap_params = {**wrap_kwargs}
    train_data, val_data = train_val_data_of_nicknames(data_name, device)
    model = get_model(model_name)
    model_str = trained_model_str(model_name, data_name)
    wrap = Wrap(
        train_data,
        val_data,
        test_data=None,
        model=model,
        log_path=model_str,
        profiling=profiling,
        device=device,
        accum_grad_batches=accum_grad_batches,
        feature_length=feature_length,
        dim_mlp_layers=dim_mlp_layers,
        hyperparameter_path=hyperparameter_path,
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
    **wrap_kwargs,
):
    """
    Loads a model in class `model_name` that was previously trained on `data_name`, and
    continue training it.
    """
    # set final and test checkpoint strings
    if train_checkpoint is None:
        train_checkpoint = trained_model_path(model_name, data_name) + ".ckpt"
    # load trained model
    try:
        model = get_model(model_name).load_from_checkpoint(train_checkpoint)
    except FileNotFoundError as e:
        raise ValueError(
            f"Model {model_name} trained on data {data_name} does not have saved checkpoint."
        ) from e

    model_str = trained_model_str(model_name, data_name)
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
    n_trials=100
):
    train_data, val_data = train_val_data_of_nicknames(data_name, device)
    model = get_model(model_name)
    model_str = trained_model_str(model_name, data_name)
    hyper_wrap = HyperWrap(
        model,
        train_data,
        val_data,
        model_str,
        profiling=profiling,
        device=device,
        n_trials=n_trials,
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
    **wrap_kwargs,
):
    """
    Loads a trained model, specified by `model_name` and `trained_data_name`, loads
    it from checkpoint `trained_model_ckpt` and tests it on `test_data_name` dataset
    and saves trained model to checkpoint `test_checkpoint`.
    """
    # Update default parameters with any provided keyword arguments
    wrap_params = {**wrap_kwargs}
    model = get_model(trained_model_name)
    with open(hyperparameter_path, "r") as f:
        hparams = json.load(f)
    model.load_from_checkpoint(trained_model_ckpt, learning_rate = hparams["learning_rate"], feature_length = hparams["feature_length"], dim_mlp_layers = hparams["dim_mlp_layers"])
    model_str = tested_model_str(trained_model_name, train_data_name, test_data_name)
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
        **wrap_params,
    )

    # evaluate model
    test_wrap.test(trained_model_ckpt, test_checkpoint)

def aggregate_data_to_csv(
    trained_model_name,
    train_data_name,
    trained_model_ckpt,
    test_data_name,
    test_checkpoint,
    device,
    hyperparameter_path,
    lightning_log_path,
    csv_outpath
):
  """
  Aggregate result data in a CSV entry.
  """

  import numpy as np
  import pandas as pd
  import tbparse

  print("aggregate data [BEGIN]")

  with open(hyperparameter_path) as f:
      best_hyperparams = json.load(f)
  print(f"best_hyperparams:\n{best_hyperparams}")
  print(f"csv_outpath: {csv_outpath}")

  df = pd.DataFrame({
    'model': [trained_model_name],
    'train_data': [train_data_name],
    'test_data': [test_data_name],
    'learning_rate': [best_hyperparams['learning_rate']],
    'batch_size': [best_hyperparams['batch_size']],
    'accum_grad_batches': [best_hyperparams['accum_grad_batches']]
  })

  print("aggregate data [END]")
  return
