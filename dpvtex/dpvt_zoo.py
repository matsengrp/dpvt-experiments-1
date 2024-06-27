import pandas as pd
import os
from pathlib import Path

from dpvt import models
from dpvt.wrapper import Wrap, HyperWrap
from dpvtex.dpvt_data import train_val_data_of_nicknames


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


def trained_model_path(model_name, data_name):
    return f"trained_models/{trained_model_str(model_name, data_name)}"


def best_model_params_path(model_name, data_name):
    return f"hyper_checkpoints/{trained_model_str(model_name, data_name)}"


def train_model(
    model_name, data_name, final_checkpoint=None, test_checkpoint=None, **wrap_kwargs
):
    """
    Creates a model in class `model_name` and trains it on data `data_name`.
    """
    # set final and test checkpoint strings
    if final_checkpoint is None:
        final_checkpoint = trained_model_path(model_name, data_name) + ".ckpt"
    if test_checkpoint is None:
        test_checkpoint = trained_model_path(model_name, data_name) + "_test.ckpt"
    # hyperparameters (only used if no hyperparameter testing done)
    default_params = {
        "learning_rate": 0.01,
        "batch_size": 1024,
        "epochs": 2,
    }
    # Update default parameters with any provided keyword arguments
    wrap_params = {**default_params, **wrap_kwargs}
    train_data, val_data, test_data = train_val_data_of_nicknames(data_name)
    model = get_model(model_name)
    model_str = trained_model_str(model_name, data_name)
    wrap = Wrap(
        train_data, val_data, test_data, model, log_path=model_str, **wrap_params
    )
    wrap.train(final_checkpoint)
    wrap.test(test_checkpoint)
    return model


def continue_train_model(
    model_name, data_name, final_checkpoint=None, test_checkpoint=None, **wrap_kwargs
):
    """
    Loads a model in class `model_name` that was previously trained on `data_name`, and
    continue training it.
    """
    # set final and test checkpoint strings
    if final_checkpoint is None:
        final_checkpoint = trained_model_path(model_name, data_name) + ".ckpt"
    if test_checkpoint is None:
        test_checkpoint = trained_model_path(model_name, data_name) + "_test.ckpt"
    # hyperparameters (only used if no hyperparameter testing done)
    default_params = {
        "learning_rate": 0.01,
        "batch_size": 1024,
        "epochs": 2,
    }
    # Update default parameters with any provided keyword arguments
    wrap_params = {**default_params, **wrap_kwargs}
    # load trained model
    script_directory = Path(__file__)
    path = script_directory.parent.parent / "train"
    path = path / (trained_model_path(model_name, data_name) + ".ckpt")
    try:
        model = get_model(model_name).load_from_checkpoint(path)
    except FileNotFoundError as e:
        # print(f"Model {model_name} trained on {data_name} does not have saved checkpoint.")
        raise ValueError(
            f"Model {model_name} trained on data {data_name} does not have saved checkpoint."
        ) from e
    model_str = trained_model_str(model_name, data_name)
    # load dataset
    train_data, val_data, test_data = train_val_data_of_nicknames(data_name)
    wrap = Wrap(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model=model,
        log_path=model_str,
        **wrap_params,
    )
    wrap.train(final_checkpoint)
    results = wrap.test(test_checkpoint)
    return results


def optimize_hyperparameters(model_name, data_name, best_model_hparams_filepath):
    train_data, val_data, _ = train_val_data_of_nicknames(data_name)
    model = get_model(model_name)
    model_str = trained_model_str(model_name, data_name)
    hyper_wrap = HyperWrap(
        model, train_data, val_data, model_str, n_trials=1
    )  # n_trials chosen small for testing
    hyper_wrap.optuna_optimize(best_model_hparams_filepath)
    return model


def validate_model(
    model_name, trained_data_name, val_data_name, directory=".", **wrap_kwargs
):
    """
    Loads a trained model, specified by `model_name` and `trained_data_name`, and
    validates the model on the dataset `val_data_name`
    """
    # hyperparameters (only used if no hyperparameter testing done)
    default_params = {
        # "learning_rate": 0.01,
        # "batch_size": 1024,
        # "epochs": 2,
    }
    # Update default parameters with any provided keyword arguments
    wrap_params = {**default_params, **wrap_kwargs}
    # load trained model
    path = "../train/" + trained_model_path(model_name, trained_data_name) + ".ckpt"
    model = get_model(model_name).load_from_checkpoint(path)
    # load dataset
    train_data, val_data, test_data = train_val_data_of_nicknames(val_data_name)
    # create string for logging
    model_str = trained_model_str(model_name, trained_data_name)
    model_str = model_str + "-ON-" + val_data_name
    val_wrap = Wrap(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        model=model,
        log_path=model_str,
        **wrap_params,
    )

    # evaluate model
    results = val_wrap.trainer.validate(val_wrap.model, val_wrap.val_loader)
    val_loss = results[0]["val_loss"]

    return val_loss

    bce_loss = val_wrap.evaluate()
    crepe_basename = os.path.basename(model_name)
    df = pd.DataFrame(
        {
            "crepe_prefix": [model_name],
            "crepe_basename": [crepe_basename],
            "dataset_name": [data_name],
            "bce_loss": [bce_loss],
        }
    )
    df.to_csv(
        f"{directory}/{crepe_basename}-ON-{data_name}.csv",
        index=False,
    )
