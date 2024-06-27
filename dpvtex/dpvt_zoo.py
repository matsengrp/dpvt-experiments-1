from dpvt import models
from dpvt.wrapper import Wrap, HyperWrap
from dpvtex.dpvt_data import (
    data_of_nicknames,
    train_val_data_of_nicknames,
)

import torch
torch.set_num_threads(1)

torch.set_default_dtype(torch.float64)  # Set default to float64 for higher precision

def get_model(model_name):
    if model_name == "TraverseNN":
        model = models.TraverseNN
    elif model_name == "TraverseMaxPooling":
        model = models.TraverseMaxPooling
    elif model_name == "TraverseAvgPooling":
        model = models.TraverseAvgPooling
    elif model_name == "TransformerEncoderTraversal":
        model = models.TransformerEncoderTraversal
    return model


def trained_model_str(model_name, data_name):
    return f"{model_name}-{data_name}"

def tested_model_str(model_name, train_data_name, test_data_name):
    return f"{model_name}-{train_data_name}-ON-{test_data_name}"

def trained_model_path(model_name, data_name):
    return f"trained_models/{trained_model_str(model_name, data_name)}"

def tested_model_path(model_name, train_data_name, test_data_name):
    return f"tested_models/{tested_model_str(model_name, train_data_name, test_data_name)}"

def best_model_params_path(model_name, data_name):
    return f"hyper_checkpoints/{trained_model_str(model_name, data_name)}"


def train_model(model_name, data_name, device, train_checkpoint=None, **wrap_kwargs):
    """
    Creates a model in class `model_name` and trains it on data `data_name`.
    """
    # set final and test checkpoint strings
    if train_checkpoint is None:
        train_checkpoint = trained_model_path(model_name, data_name) + ".ckpt"
    # hyperparameters (only used if no hyperparameter testing done)
    default_params = {
        "learning_rate": 0.01,
        "batch_size": 1024,
        "epochs": 200,
    }
    # Update default parameters with any provided keyword arguments
    wrap_params = {**default_params, **wrap_kwargs}
    train_data, val_data = train_val_data_of_nicknames(data_name, device)
    model = get_model(model_name)
    train_data, val_data = train_val_data_of_nicknames(data_name, device)
    model = get_model(model_name)
    model_str = trained_model_str(model_name, data_name)
    wrap = Wrap(
        train_data,
        val_data,
        test_data=None,
        model=model,
        log_path=model_str,
        device=device,
        **wrap_params,
    )
    wrap.train(train_checkpoint)
    return model


def continue_train_model(model_name, data_name, device, train_checkpoint=None, **wrap_kwargs):
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
    # hyperparameters (only used if no hyperparameter testing done)
    default_params = {
        "learning_rate": 0.01,
        "batch_size": 1024,
        "epochs": 100,
    }
    # Update default parameters with any provided keyword arguments
    wrap_params = {**default_params, **wrap_kwargs}
    # load dataset
    train_data, val_data = train_val_data_of_nicknames(data_name)
    wrap = Wrap(
        train_data=train_data,
        val_data=val_data,
        test_data=None,
        model=model,
        log_path=model_str,
        device=device,
        **wrap_params,
    )
    wrap.train(train_checkpoint)
    return model


def optimize_hyperparameters(model_name, data_name, best_model_hparams_filepath, device):
    train_data, val_data = train_val_data_of_nicknames(data_name, device)
    model = get_model(model_name)
    model_str = trained_model_str(model_name, data_name)
    hyper_wrap = HyperWrap(
        model, train_data, val_data, model_str, device = device, epochs=100, n_trials=1
    )  # n_trials chosen small for testing
    hyper_wrap.optuna_optimize(best_model_hparams_filepath)
    return model


def test_model(
    trained_model_name, train_data_name, trained_model_ckpt, test_data_name, test_checkpoint, **wrap_kwargs
):
    """
    Loads a trained model, specified by `model_name` and `trained_data_name`, loads
    it from checkpoint `trained_model_ckpt` and tests it on `test_data_name` dataset
    and saves trained model to checkpoint `test_checkpoint`.
    """
    # Update default parameters with any provided keyword arguments
    wrap_params = {**wrap_kwargs}
    model = get_model(trained_model_name).load_from_checkpoint(trained_model_ckpt)
    model_str = tested_model_str(trained_model_name, train_data_name, test_data_name)
    # load dataset
    test_data = data_of_nicknames(test_data_name)
    test_wrap = Wrap(
        train_data=None,
        val_data=None,
        test_data=test_data,
        model=model,
        log_path=model_str,
        device=device,
        **wrap_params,
    )

    # evaluate model
    result = test_wrap.test(test_checkpoint)

    test_auroc = result[0]["test_auroc"]
    return test_auroc
