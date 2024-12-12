from dpvt import models
from dpvt.wrapper import Wrap, HyperWrap
from dpvtex.dpvt_data import (
    data_of_nicknames,
    train_val_data_of_nicknames,
)
import json
import torch

from lightning.pytorch.callbacks import Callback

torch.set_num_threads(1)
from pytorch_lightning import seed_everything

# seed_everything(42, workers=True)
torch.set_default_dtype(torch.float64)  # Set default to float64 for higher precision

from datetime import datetime

todays_date = datetime.now().strftime("%Y-%m-%d")


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


def model_str(model_name, train_data_name, test_data_name=None):
    if test_data_name:
        return tested_model_str(model_name, train_data_name, test_data_name)
    return trained_model_str(model_name, train_data_name)


def trained_model_path(model_name, data_name):
    return f"trained_models/{trained_model_str(model_name, data_name)}"


def tested_model_path(model_name, train_data_name, test_data_name):
    return (
        f"tested_models/{tested_model_str(model_name, train_data_name, test_data_name)}"
    )


def best_model_params_path(model_name, data_name):
    return f"hyper_checkpoints/{trained_model_str(model_name, data_name)}"


def csv_log_path(model_name, train_data_name, test_data_name=None, device=None, date=str(todays_date)):
    path = f"result_csvs/{device}_{date}"
    return f"{path}/{model_str(model_name, train_data_name, test_data_name)}.csv"


def generate_csv_log_paths(model_names, data_pairs, device=None, date=str(todays_date)):
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
    timestamp=str(todays_date),
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
        added_callbacks=[TimerCallback()],
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
        added_callbacks=[],
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

import os
import tbparse
import pandas as pd

# get lightning logs from training and testing
def get_lightning_log_path(model_name, train_data_name, device, timestamp, root_dir, test_data_name=None, version=None):
    path = f'{root_dir}/lightning_logs/{device}_{timestamp}/{model_str(model_name, train_data_name, test_data_name)}'
    path = append_version_to_path(path, version)
    return path

# get lightning logs from hyperparameter optimization
def get_hyperparam_log_path(model_name, train_data_name, device, timestamp, root_dir, test_data_name=None, version=None):
    path = f'{root_dir}/hyper_checkpoints/{model_str(model_name, train_data_name)}'
    path = append_version_to_path(path, version)
    return path

# get json of hyperparameters
def get_hyperparam_json_path(model_name, train_data_name, device, timestamp, root_dir, test_data_name=None, version=None):
    path = f'{root_dir}/hyper_checkpoints/{model_str(model_name, train_data_name)}.json'
    path = append_version_to_path(path, version)
    return path

# get dataframe from lightning log
def get_df_from_log(log_path):
    reader = tbparse.SummaryReader(log_path)
    return reader.scalars

# append version folder
def append_version_to_path(path, version=None):
    if version:
        return f'{path}/{version}'
    return path


def aggregate_data_to_csv(
    model_name,
    train_data_name,
    test_data_name,
    device,
    timestamp,
    n_hyperparameter_trials,
    hyperparameter_path,
    trained_model_ckpt,
    tested_model_ckpt,
    csv_outpath,
):
  """
  Aggregate result data in a CSV entry.
  """

  # fetch hyperparams
  with open(hyperparameter_path) as f:
      opt_hyperparams = json.load(f)
  # print(f"best_hyperparams:\n{opt_hyperparams}")

  # fetch logs
  root_dir = '.'
  version = 'version_0'
  hyperparam_json_path = get_hyperparam_json_path(model_name, train_data_name, device, timestamp, root_dir=root_dir, test_data_name=None, version=version)
  hyperparam_log_path = get_hyperparam_log_path(model_name, train_data_name, device, timestamp, root_dir=root_dir, test_data_name=None, version=version)
  train_log_path = get_lightning_log_path(model_name, train_data_name, device, timestamp, root_dir=root_dir, test_data_name=None, version=version)
  test_log_path = get_lightning_log_path(model_name, train_data_name, device, timestamp, root_dir=root_dir, test_data_name=test_data_name, version=version)

  log_dfs = {}
  log_dfs['log_hyperparam'] = get_df_from_log(f'{hyperparam_log_path}')
  log_dfs['log_train'] = get_df_from_log(f'{train_log_path}')
  log_dfs['log_test'] = get_df_from_log(f'{test_log_path}')

  # for key,df in log_dfs.items():
  #   print('KEY:', key)
  #   print(set(df.tag))
  #   for tag in set(df.tag):
  #       tag_df = df[df.tag == tag]
  #       print(f'{tag}: {len(tag_df)}')
  #       print(f'max_step: {max(set(tag_df.step))}')
  #       print(f'max_value: {max(set(tag_df.value))}')

  # fetch training stats
  df = log_dfs['log_train']
  train_walltime = df[df.tag == 'train_wall_time'].value.iloc[0]
  train_epochs = df[df.tag == 'train_final_epoch'].value.iloc[0]
  # train_steps = df[df.tag == 'train_final_step'].value.iloc[0]

  # fetch testing stats
  df = log_dfs['log_test']
  test_auroc = df[df.tag == 'test_auroc'].value.iloc[0]
  test_loss = df[df.tag == 'test_loss'].value.iloc[0]

  df_row = pd.DataFrame({
    # config settings
    'model': [model_name],
    'train_data': [train_data_name],
    'test_data': [test_data_name],
    'device': [device],
    'timestamp': [timestamp],
    'n_hyperparam_trials': [n_hyperparameter_trials],
    # hyperparameters
    'learning_rate': [opt_hyperparams['learning_rate']],
    'batch_size': [opt_hyperparams['batch_size']],
    'accum_grad_batches': [opt_hyperparams['accum_grad_batches']],
    'max_epochs': [opt_hyperparams['epochs']],
    'feature_length': [opt_hyperparams['feature_length']],
    'dim_mlp_layers': [opt_hyperparams['dim_mlp_layers']],
    # number of training steps, epochs
    'train_steps': [-1],
    'train_epochs': [train_epochs],
    # test auroc
    'test_auroc': [test_auroc],
    'test_loss': [test_loss],
    # runtime
    'train_walltime': [train_walltime],
    # paths
    'hyperparam_json_path': [hyperparam_json_path],
    'hyperparam_log_path': [hyperparam_log_path],
    'train_log_path': [train_log_path],
    'test_log_path': [test_log_path],
    'trained_model_ckpt_path': [trained_model_ckpt],
    'tested_model_ckpt_path': [tested_model_ckpt],
  })

  # output single row to file
  print(f"csv_outpath: {csv_outpath}")
  df_row.to_csv(csv_outpath, index=False)

  # append row to final file
  csv_final_outpath = 'result_csvs/FINAL.csv'
  print(f"csv_final_outpath: {csv_final_outpath}")
  if os.path.exists(csv_final_outpath):
      final_df = pd.read_csv(csv_final_outpath)
      final_df = pd.concat([final_df, df_row], ignore_index=True)
      final_df.to_csv(csv_final_outpath, index=False)
  else:
      df_row.to_csv(csv_final_outpath, index=False)
  return

import time

class TimerCallback(Callback):
    """
    Callback for logging hyperparameters, total_epochs, number_of_steps, auroc, runtimes
    """
    def __init__(self):
        self.start_time = {}
        self.total_steps = {}

    def log_start(self, trainer, pl_module, prefix=""):
        self.start_time[prefix] = time.time()
        self.total_steps[prefix] = 0

    def log_end(self, trainer, pl_module, prefix=""):
        total_time = time.time() - self.start_time[prefix]
        trainer.logger.log_metrics({f"{prefix}_wall_time": total_time})
        trainer.logger.log_metrics({f"{prefix}_final_epoch": trainer.current_epoch})
        stopped_early = trainer.current_epoch + 1 < trainer.max_epochs
        trainer.logger.log_metrics({f"{prefix}_stopped_early": stopped_early})
        trainer.logger.log_metrics({f"{prefix}_final_step": self.total_steps[prefix]})

    def log_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, prefix=""):
        self.total_steps[prefix] += 1

    # hooks

    def on_train_start(self, trainer, pl_module):
        self.log_start(trainer, pl_module, "train")

    def on_train_end(self, trainer, pl_module):
        self.log_end(trainer, pl_module, "train")
