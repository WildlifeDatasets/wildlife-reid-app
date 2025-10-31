import argparse
import json
import logging
import os
import io
from typing import Callable, Tuple, Union, Optional
from collections import OrderedDict
import pandas as pd
import torch.nn as nn
import yaml
import warnings
import timm
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


def get_model_target_size(model: nn.Module) -> Optional[int]:
    """Get target size (number of output classes) of a `timm` model.

    Parameters
    ----------
    model
        PyTorch model from `timm` library.

    Returns
    -------
    target_size
        Output feature size of a prediction head.
    """
    target_size = None
    if isinstance(model, nn.DataParallel):
        model = model.module  # unwrap model from data parallel wrapper for multi-gpu training
    elif hasattr(model, "model"):
        model = model.model
    cls_name = model.default_cfg["classifier"]

    # iterate through nested modules
    parts = cls_name.split(".")
    module = model
    for i, part_name in enumerate(parts):
        module = getattr(module, part_name)
    # set target size of the last nested module if it is a linear layer
    # other layers like nn.Identity are ignored
    if hasattr(module, "out_features"):
        target_size = module.out_features

    if target_size is None:
        warnings.warn(
            "Could not find target size (number of classes) "
            f"of the model {model.__class__.__name__}."
        )

    return target_size


def get_model(
    architecture_name: str,
    target_size: int = None,
    pretrained: bool = False,
    *,
    checkpoint_path: Union[str, io.BytesIO] = None,
    strict: bool = True,
) -> nn.Module:
    """Get a `timm` model.

    Parameters
    ----------
    architecture_name
        Name of the network architecture from `timm` library.
    target_size
        Output feature size of the new prediction head.
    pretrained
        If true load pretrained weights from `timm` library.
    checkpoint_path
        Path (or IO Buffer) with checkpoint weights to load after the model is initialized.
    strict
        Whether to strictly enforce the keys in state_dict to match
        between the model and checkpoint weights from file.
        Used when argument checkpoint_path is specified.

    Returns
    -------
    model
        PyTorch model from `timm` library.
    """
    pretrained = pretrained and checkpoint_path is None
    model = timm.create_model(architecture_name, pretrained=pretrained)

    # load model with classification head if missing
    # models like ViT trained with DINO do not have a classification head by default
    # classification head is missing to get `in_features` value in the method `set_prediction_head`
    if model.default_cfg["num_classes"] == 0 and target_size is not None:
        model = timm.create_model(architecture_name, pretrained=pretrained, num_classes=1000)

    # load custom weights
    if checkpoint_path is not None:
        logger.debug("Loading pre-trained checkpoint.")
        weights = torch.load(checkpoint_path, map_location="cpu")

        # remove prefix "module." created by nn.DataParallel wrapper
        if all([k.startswith("module.") for k in weights.keys()]):
            weights = OrderedDict({k[7:]: v for k, v in weights.items()})

        # identify target size in the weights
        cls_name = model.default_cfg["classifier"]
        if f"{cls_name}.bias" in weights and f"{cls_name}.weight" in weights:
            weights_target_size = weights[f"{cls_name}.bias"].shape[0]
            model_target_size = model.default_cfg["num_classes"]
            if weights_target_size != model_target_size:
                # set different target size based on the checkpoint weights
                in_features = weights[f"{cls_name}.weight"].shape[1]
                model = set_prediction_head(model, weights_target_size, in_features=in_features)

        # load checkpoint weights
        model.load_state_dict(weights, strict=strict)

    # set classification head
    if target_size is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_target_size = get_model_target_size(model)
        if target_size != model_target_size:
            logger.debug("Setting new prediction head with random initial weights.")
            model = set_prediction_head(model, target_size)

    return model


def set_prediction_head(model: nn.Module, target_size: int, *, in_features: int = None):
    """Replace prediction head of a `timm` model.

    Parameters
    ----------
    model
        PyTorch model from `timm` library.
    target_size
        Output feature size of the new prediction head.
    in_features
        Number of input features for the prediction head.
        The parameter is needed in special cases,
        e.g., when the current prediction head is `nn.Identity`.

    Returns
    -------
    model
        The input `timm` model with new prediction head.
    """
    assert hasattr(model, "default_cfg")
    cls_name = model.default_cfg["classifier"]
    # iterate through nested modules
    parts = cls_name.split(".")
    module = model
    for i, part_name in enumerate(parts):
        if i == len(parts) - 1:
            last_layer = getattr(module, part_name)
            in_features = in_features or last_layer.in_features
            setattr(module, part_name, nn.Linear(in_features, target_size))
        else:
            module = getattr(module, part_name)
    return model


def load_model(
    config: dict, checkpoint_path: str = None, strict: bool = True
) -> Tuple[nn.Module, tuple, tuple]:
    """Load model with pre-trained checkpoint.

    Options from YAML configuration file:
     - `architecture` - any architecture name from timm library.
     - `number_of_classes` - integer value.
     - (optional) `pretrained_checkpoint` - options:
         - "timm" (default) - pre-trained checkpoint from timm.
         - "none" - randomly initialized weights.
         - <path> - path to a custom checkpoint.
     - (optional) `multigpu` - if true, use `nn.DataParallel` model wrapper.

    Pre-trained checkpoint can be set using `config` dictionary or `checkpoint_path` argument.

    Priority:
     - Use `checkpoint_path` path to a custom checkpoint when `checkpoint_path` is specified.
     - Otherwise use configuration from `config`.

    Parameters
    ----------
    config
        A dictionary with experiment configuration.
        It should contain `architecture`, `number_of_classes`, and optionally `multigpu`.
    checkpoint_path
        Path to the pre-trained model checkpoint.
        The argument overrides `pretrained_checkpoint` setting in `config` dictionary.
    strict
        Whether to strictly enforce the keys in state_dict to match
        between the model and checkpoint weights from file.
        Used when argument checkpoint_path is specified.

    Returns
    -------
    model
        PyTorch model.
    model_mean
        Tuple with mean used to normalize images during training.
    model_std
        Tuple with standard deviation used to normalize images during training.
    """
    assert "architecture" in config
    assert "number_of_classes" in config
    pretrained_checkpoint = config.get("pretrained_checkpoint", "timm")
    pretrained = False
    if checkpoint_path is None:
        # validate pretrained_checkpoint parameter and set variables pretrained and checkpoint_path
        if pretrained_checkpoint.lower() == "timm":
            pretrained = True
        elif pretrained_checkpoint.lower() == "none":
            pretrained = False
        elif os.path.isfile(pretrained_checkpoint):
            pretrained = False
            checkpoint_path = pretrained_checkpoint
        else:
            raise ValueError(
                "Invalid value in config parameter 'pretrained_checkpoint'. "
                "Use one of the options: 'timm' | 'none' | <path>."
            )

    model = get_model(
        config["architecture"],
        config["number_of_classes"],
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        strict=strict,
    )
    model_mean = tuple(model.default_cfg["mean"])
    model_std = tuple(model.default_cfg["std"])
    if config.get("multigpu", False):  # multi gpu model
        model = nn.DataParallel(model)
        logger.info("Using nn.DataParallel for multiple GPU support.")
    return model, model_mean, model_std
