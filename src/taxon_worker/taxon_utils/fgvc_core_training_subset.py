from typing import Type
import os
import random
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import NamedTuple, Optional, Union

import warnings
from typing import Tuple

import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader

import numpy as np
from tqdm.auto import tqdm

import warnings
from typing import Union

from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader

from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader


from .fgvc_subset.utils.wandb import log_progress

SchedulerType = Union[ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR]

from typing import Callable

# from .base_trainer import BaseTrainer
# from .classification_trainer import ClassificationTrainer
# from .training_outputs import PredictOutput
import logging

logger = logging.getLogger(__name__)


class BatchOutput(NamedTuple):
    """Tuple returned from `train_batch` and `predict_batch` trainer methods."""

    preds: np.ndarray
    targs: np.ndarray
    loss: Union[float, dict]


class TrainEpochOutput(NamedTuple):
    """Tuple returned from `train_epoch` trainer method."""

    avg_loss: float
    avg_scores: Optional[dict] = {}
    max_grad_norm: Optional[float] = None
    other_avg_losses: Optional[dict] = {}


class PredictOutput(NamedTuple):
    """Tuple returned from `predict` trainer method."""

    preds: Optional[np.ndarray] = None
    targs: Optional[np.ndarray] = None
    avg_loss: Optional[float] = np.nan
    avg_scores: Optional[dict] = {}


import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# from .training_outputs import BatchOutput, PredictOutput, TrainEpochOutput
# from .training_utils import to_device, to_numpy
from typing import Iterable, List, Optional, Union

import numpy as np
import torch


def to_device(
    *tensors: List[Union[torch.Tensor, dict]], device: torch.device
) -> List[Union[torch.Tensor, dict]]:
    """Convert pytorch tensors to device.

    Parameters
    ----------
    tensors
        (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.
    device
        Device to use (CPU,CUDA,CUDA:0,...).

    Returns
    -------
    (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.
    """
    out = []
    for tensor in tensors:
        if isinstance(tensor, dict):
            tensor = {k: v.to(device) for k, v in tensor.items()}
        else:
            tensor = tensor.to(device)
        out.append(tensor)
    return out if len(out) > 1 else out[0]


def to_numpy(*tensors: List[Union[torch.Tensor, dict]]) -> List[Union[np.ndarray, dict]]:
    """Convert pytorch tensors to numpy arrays.

    Parameters
    ----------
    tensors
        (One or multiple items) Pytorch tensor or dictionary of pytorch tensors.

    Returns
    -------
    (One or multiple items) Numpy array or dictionary of numpy arrays.
    """
    out = []
    for tensor in tensors:
        if isinstance(tensor, dict):
            tensor = {k: v.detach().cpu().numpy() for k, v in tensor.items()}
        else:
            tensor = tensor.detach().cpu().numpy()
        out.append(tensor)
    return out if len(out) > 1 else out[0]


def concat_arrays(
    *lists: List[List[Union[np.ndarray, dict]]]
) -> List[Optional[List[Union[np.ndarray, dict]]]]:
    """Concatenate lists of numpy arrays with predictions and targets to numpy arrays.

    Parameters
    ----------
    lists
        (One or multiple items) List of numpy arrays or dictionary of lists.

    Returns
    -------
    (One or multiple items) Numpy array or dictionary of numpy arrays.
    """

    def _safer_concat(array_list):
        num_elems = sum([len(x) for x in array_list])
        out_array = np.zeros((num_elems, *array_list[0].shape[1:]), dtype=array_list[0].dtype)
        np.concatenate(array_list, axis=0, out=out_array)
        return out_array

    out = []
    for array_list in lists:
        concatenated = None
        if len(array_list) > 0:
            if isinstance(array_list[0], dict):
                # concatenate list of dicts of numpy arrays to a dict of numpy arrays
                concatenated = {}
                for k in array_list[0].keys():
                    concatenated[k] = _safer_concat([x[k] for x in array_list])
            else:
                # concatenate list of numpy arrays to a numpy array
                concatenated = _safer_concat(array_list)
        out.append(concatenated)
    return out if len(out) > 1 else out[0]


def get_gradient_norm(
    model_params: Union[torch.Tensor, Iterable[torch.Tensor]], norm_type: float = 2.0
) -> float:
    """Compute norm of model parameter gradients.

    Parameters
    ----------
    model_params
        Model parameters.
    norm_type
        The order of norm.

    Returns
    -------
    Norm of model parameter gradients.
    """
    grads = [p.grad for p in model_params if p.grad is not None]
    if len(grads) == 0:
        total_norm = 0.0
    else:
        norms = torch.stack([torch.norm(g.detach(), norm_type) for g in grads])
        total_norm = torch.norm(norms, norm_type).item()
    return total_norm


class BaseTrainer:
    """Class to perform training of a neural network and/or run inference.

    Parameters
    ----------
    model
        Pytorch neural network.
    trainloader
        Pytorch dataloader with training data.
    criterion
        Loss function.
    optimizer
        Optimizer algorithm.
    validloader
        Pytorch dataloader with validation data.
    accumulation_steps
        Number of iterations to accumulate gradients before performing optimizer step.
    clip_grad
        Max norm of the gradients for the gradient clipping.
    device
        Device to use (cpu,0,1,2,...).
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader = None,
        criterion: nn.Module = None,
        optimizer: Optimizer = None,
        *,
        validloader: DataLoader = None,
        accumulation_steps: int = 1,
        clip_grad: float = None,
        device: torch.device = None,
        **kwargs,
    ):
        super().__init__()
        # model and loss arguments
        self.model = model
        self.criterion = criterion

        # data arguments
        self.trainloader = trainloader
        self.validloader = validloader

        # optimization arguments
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.clip_grad = clip_grad

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def train_batch(self, batch: tuple) -> BatchOutput:
        """Run a training iteration on one batch.

        Parameters
        ----------
        batch
            Tuple of arbitrary size with image and target pytorch tensors
            and optionally additional items depending on the dataloaders.

        Returns
        -------
        BatchOutput tuple with predictions, ground-truth targets, and average loss.
        """
        assert len(batch) >= 2
        imgs, targs = batch[0], batch[1]
        imgs, targs = to_device(imgs, targs, device=self.device)
        # apply Mixup or Cutmix if MixupMixin is used in the final class
        targs_ = targs  # keep original targets to return by the method
        if hasattr(self, "apply_mixup") and len(imgs) % 2 == 0:  # batch size should be even
            imgs, targs = self.apply_mixup(imgs, targs)

        preds = self.model(imgs)
        loss = self.criterion(preds, targs)
        _loss = loss.item()

        # scale the loss to the mean of the accumulated batch size
        loss = loss / self.accumulation_steps
        loss.backward()

        # convert to numpy
        preds, targs = to_numpy(preds, targs_)
        return BatchOutput(preds, targs, _loss)

    def predict_batch(self, batch: tuple, *, model: nn.Module = None) -> BatchOutput:
        """Run a prediction iteration on one batch.

        Parameters
        ----------
        batch
            Tuple of arbitrary size with image and target pytorch tensors
            and optionally additional items depending on the dataloaders.
        model
            Alternative PyTorch model to use for prediction like EMA model.

        Returns
        -------
        BatchOutput tuple with predictions, ground-truth targets, and average loss.
        """
        model = model or self.model
        assert len(batch) >= 2
        imgs, targs = batch[0], batch[1]
        imgs = to_device(imgs, device=self.device)

        # run inference and compute loss
        with torch.no_grad():
            preds = model(imgs)
        loss = 0.0
        if self.criterion is not None:
            targs = to_device(targs, device=self.device)
            loss = self.criterion(preds, targs).item()

        # convert to numpy
        preds, targs = to_numpy(preds, targs)
        return BatchOutput(preds, targs, loss)

    def train_epoch(self, *args, **kwargs) -> TrainEpochOutput:
        """Train one epoch."""
        raise NotImplementedError()

    def predict(self, *args, **kwargs) -> PredictOutput:
        """Run inference."""
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        """Train neural network."""
        raise NotImplementedError()


# ImageFile.LOAD_TRUNCATED_IMAGES = True


# from ..models import get_model_target_size


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


class MixupMixin:
    """Mixin class that adds LR scheduler functionality to the trainer class.

    The SchedulerMixin supports PyTorch and timm schedulers.

    Parameters
    ----------
    model
        Pytorch neural network.
        MixupMixin uses it to get number of classes.
    trainloader
        Pytorch dataloader with training data.
        MixupMixin uses it to get number of classes.
    mixup
        Mixup alpha value, mixup is active if > 0.
    cutmix
        Cutmix alpha value, cutmix is active if > 0.
    mixup_prob
        Probability of applying mixup or cutmix per batch.
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader = None,
        *args,
        mixup: float = 0.0,
        cutmix: float = 0.0,
        mixup_prob: float = 1.0,
        **kwargs,
    ):
        # set default values (in case of script sets them as None)
        mixup = mixup or 0.0
        cutmix = cutmix or 0.0
        mixup_prob = mixup_prob or 1.0

        # create mixup class
        if mixup > 0.0 or cutmix > 0.0:
            # get number of classes from model and trainset for Mixup method
            num_classes = get_model_target_size(model)
            if num_classes is not None:
                self.mixup_fn = Mixup(
                    mixup_alpha=mixup,
                    cutmix_alpha=cutmix,
                    cutmix_minmax=None,
                    prob=mixup_prob,
                    switch_prob=0.5,
                    mode="batch",
                    correct_lam=True,
                    label_smoothing=0.1,
                    num_classes=num_classes,
                )
            else:
                warnings.warn("Could not identify number of classes from model. Not using MixUp.")
        else:
            self.mixup_fn = None

        # call parent class to initialize trainer
        super().__init__(*args, model=model, trainloader=trainloader, **kwargs)

    def apply_mixup(
        self, imgs: torch.Tensor, targs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup or cutmix method if arguments `mixup` or `cutmix` were used in Trainer."""
        if self.mixup_fn is not None:
            imgs, targs = self.mixup_fn(imgs, targs)
        return imgs, targs


class EMAMixin:
    """Mixin class that adds model weight averaging functionality to the trainer class.

    The EMAMixin supports Exponential Moving Average strategy.

    Parameters
    ----------
    model
        Pytorch neural network.
        EMAMixin uses it to create `AveragedModel` for EMA.
    trainloader
        Pytorch dataloader with training data.
        EMAMixin uses it to update BN parameters at the end of training.
    device
        Device to use (cpu,0,1,2,...).
        EMAMixin uses it for setting destination of ema_model.
    apply_ema
        Apply EMA model weight averaging if true.
    ema_start_epoch
        Epoch number when to start model averaging.
    ema_decay
        Model weight decay.
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader = None,
        device: torch.device = None,
        *args,
        apply_ema: bool = False,
        ema_start_epoch: int = 0,
        ema_decay: float = 0.9999,
        **kwargs,
    ):
        # set default values (in case of script sets them as None)
        self.apply_ema = apply_ema or False
        self.ema_start_epoch = ema_start_epoch or 0
        self.ema_decay = ema_decay or 0.9999

        self.model = model
        self.trainloader = trainloader
        self.device = device
        self.ema_model = None

        if isinstance(model, nn.DataParallel):
            warnings.warn(
                "EMAMixin does not support training on multiple GPUs. "
                "Batch Norm statistics will be inaccurate in the averaged model."
            )

        # call parent class to initialize trainer
        super().__init__(*args, model=model, trainloader=trainloader, device=device, **kwargs)

    def create_ema_model(self):
        """Initialize EMA averaged model."""
        self.ema_model = ModelEmaV2(self.model, decay=self.ema_decay, device=self.device)

    def get_ema_model(self):
        """Get EMA averaged model."""
        return self.ema_model and self.ema_model.module

    def make_ema_update(self, epoch: int):
        """Update weights of the EMA averaged model."""
        if self.apply_ema and epoch >= self.ema_start_epoch:
            if self.ema_model is None:
                self.create_ema_model()
            self.ema_model.update(self.model)


class SchedulerMixin:
    """Mixin class that adds LR scheduler functionality to the trainer class.

    The SchedulerMixin supports PyTorch and timm schedulers.

    Parameters
    ----------
    scheduler
        LR scheduler algorithm.
    validloader
        Pytorch dataloader with validation data.
        SchedulerMixin uses it to validate it is not None when `scheduler=ReduceLROnPlateau`.
    """

    def __init__(
        self,
        *args,
        scheduler: SchedulerType = None,
        validloader: DataLoader = None,
        **kwargs,
    ):
        # validate scheduler
        if scheduler is not None:
            assert isinstance(scheduler, (ReduceLROnPlateau, CosineLRScheduler, CosineAnnealingLR))
            if isinstance(scheduler, ReduceLROnPlateau):
                assert (
                    validloader is not None
                ), "Scheduler ReduceLROnPlateau requires validation set to update learning rate."
        self.scheduler = scheduler
        self.validloader = validloader

        # call parent class to initialize trainer
        super().__init__(*args, validloader=validloader, **kwargs)

    def make_timm_scheduler_update(self, num_updates: int):
        """Make scheduler step update after training one iteration.

        This is specific to `timm` schedulers.

        Parameters
        ----------
        num_updates
            Iteration number.
        """
        if self.scheduler is not None and isinstance(self.scheduler, CosineLRScheduler):
            self.scheduler.step_update(num_updates=num_updates)

    def make_scheduler_step(self, epoch: int = None, *, valid_loss: float = None):
        """Make scheduler step after training one epoch.

        The method uses different arguments depending on the scheduler type.

        Parameters
        ----------
        epoch
            Current epoch number. The method expects start index 1 (instead of 0).
        valid_loss
            Average validation loss to use for `ReduceLROnPlateau` scheduler.
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if valid_loss is not None:
                    self.scheduler.step(valid_loss)  # pytorch implementation
                else:
                    warnings.warn(
                        "Scheduler ReduceLROnPlateau requires validation set "
                        "to update learning rate."
                    )
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()  # pytorch implementation
            elif isinstance(self.scheduler, CosineLRScheduler):
                if epoch is not None:
                    self.scheduler.step(epoch)  # timm implementation
                else:
                    warnings.warn(
                        "Scheduler CosineLRScheduler requires epoch number to update learning rate."
                    )
            else:
                raise ValueError(f"Unsupported scheduler type: {self.scheduler}")


from typing import Callable, Union

import numpy as np


class ScoresMonitor:
    """Helper class for monitoring scores during training.

    Parameters
    ----------
    scores_fn
        Callable function for evaluating training scores.
        The function should accept `preds` and `targs` and return a dictionary with scores.
    num_samples
        Number of samples in the dataset.
    eval_batches
        If true the method evaluates scores on each mini-batch during training.
        Otherwise, it stores predictions and targets (`preds`, `targs`)
        and evaluates scores on full dataset.
        Set `eval_batches=False` in cases where all data points are needed to compute a score,
        e.g. F1 score in classification.
    store_preds_targs
        If true the method stores predictions and targets (`preds`, `targs`) for later use.
    """

    def __init__(
        self,
        scores_fn: Callable,
        num_samples: int,
        *,
        eval_batches: bool = True,
        store_preds_targs: bool = False,
    ):
        self.metrics_fc = scores_fn
        self.num_samples = num_samples
        self.eval_batches = eval_batches
        self.store_preds_targs = store_preds_targs

        # initialize score variables used for eager evaluation
        self._avg_scores = None

        # initialize (preds, targs) variables used for lazy evaluation
        self._i = 0
        self._bs = None
        self._preds_all = None
        self._targs_all = None

    def reset(self):
        """Reset internal variables including average scores and stored predictions and targets."""
        self._avg_scores = None
        self._i = 0
        self._bs = None
        self._preds_all = None
        self._targs_all = None

    def _update_scores(self, preds: Union[np.ndarray, dict], targs: Union[np.ndarray, dict]):
        batch_scores = self.metrics_fc(preds, targs)
        batch_scores = {k: v / self.num_samples for k, v in batch_scores.items()}
        if self._avg_scores is None:
            self._avg_scores = batch_scores
        else:
            for k in self._avg_scores.keys():
                self._avg_scores[k] += batch_scores[k]

    def _store_preds_targs(self, preds: Union[np.ndarray, dict], targs: Union[np.ndarray, dict]):
        if self._preds_all is None and self._targs_all is None:

            def init_array(x):
                return np.zeros((self.num_samples, *x.shape[1:]), dtype=x.dtype)

            # initialize empty array
            if isinstance(preds, dict):
                self._preds_all = {k: init_array(v) for k, v in preds.items()}
                self._bs = preds[list(preds.keys())[0]].shape[0]
            else:
                self._preds_all = init_array(preds)
                self._bs = preds.shape[0]
            if isinstance(targs, dict):
                self._targs_all = {k: init_array(v) for k, v in targs.items()}
            else:
                self._targs_all = init_array(targs)
            self._i = 0

        start_index = self._i * self._bs
        end_index = (self._i + 1) * self._bs
        if isinstance(preds, dict):
            for k, v in preds.items():
                self._preds_all[k][start_index:end_index] = v
        else:
            self._preds_all[start_index:end_index] = preds
        if isinstance(targs, dict):
            for k, v in targs.items():
                self._targs_all[k][start_index:end_index] = v
        else:
            self._targs_all[start_index:end_index] = targs
        self._i += 1

    def update(self, preds: Union[np.ndarray, dict], targs: Union[np.ndarray, dict]):
        """Evaluate scores based on the given predictions and targets and update average scores.

        Parameters
        ----------
        preds
            Numpy array or dictionary of numpy arrays with predictions.
        targs
            Numpy array or dictionary of numpy arrays with ground-truth targets.
        """
        if self.eval_batches:
            self._update_scores(preds, targs)

        if not self.eval_batches or self.store_preds_targs:
            self._store_preds_targs(preds, targs)

    @property
    def avg_scores(self) -> dict:
        """Get average scores."""
        if self.eval_batches:
            scores = self._avg_scores
        else:
            scores = self.metrics_fc(self._preds_all, self._targs_all)
        return scores

    @property
    def preds_all(self) -> np.ndarray:
        """Get stored predictions from the full dataset."""
        return self._preds_all

    @property
    def targs_all(self) -> np.ndarray:
        """Get stored predictions from the full dataset."""
        return self._targs_all


class LossMonitor:
    """Helper class for monitoring loss(es) during training.

    Parameters
    ----------
    num_batches
        Number of batches in dataloader.
    """

    def __init__(self, num_batches: int):
        self.num_batches = num_batches
        self._avg_loss = None

    def reset(self):
        """Reset internal variable average loss."""
        self._avg_loss = None

    def update(self, loss: Union[float, dict]):
        """Update average loss."""
        if isinstance(loss, float):
            if self._avg_loss is None:
                self._avg_loss = 0.0  # initialize average loss
            assert isinstance(self._avg_loss, float)
            self._avg_loss += loss / self.num_batches
        elif isinstance(loss, dict):
            if self._avg_loss is None:
                self._avg_loss = {k: 0.0 for k in loss.keys()}  # initialize average loss
            assert isinstance(self._avg_loss, dict)
            for k, v in loss.items():
                self._avg_loss[k] += v / self.num_batches
        else:
            raise ValueError()

    @property
    def avg_loss(self) -> float:
        """Get average loss."""
        if isinstance(self._avg_loss, float):
            avg_loss = self._avg_loss
        elif isinstance(self._avg_loss, dict):
            assert "loss" in self._avg_loss
            avg_loss = self._avg_loss["loss"]
        else:
            raise ValueError()
        return avg_loss

    @property
    def other_avg_losses(self) -> dict:
        """Get other average losses."""
        if isinstance(self._avg_loss, float):
            other_avg_losses = {}
        elif isinstance(self._avg_loss, dict):
            other_avg_losses = {k: v for k, v in self._avg_loss.items() if k != "loss"}
        else:
            raise ValueError()
        return other_avg_losses


import logging
import logging.config
import os
from typing import Optional

import yaml

_module_dir = os.path.abspath(os.path.dirname(__file__))
LOGGER_CONFIG = os.path.join(_module_dir, "../config/logging.yaml")
TRAINING_LOGGER_CONFIG = os.path.join(_module_dir, "../config/training_logging.yaml")


def setup_logging():
    """Setup logging configuration from a file."""
    with open(LOGGER_CONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def setup_training_logger(training_log_file: Optional[str]) -> logging.Logger:
    """
    Setup logging configuration from a file.

    Parameters
    ----------
    training_log_file
        Name of the log file to write training logs.
    """
    # load logging config
    with open(TRAINING_LOGGER_CONFIG, "r") as f:
        config = yaml.safe_load(f.read())
    assert "handlers" in config, "Logging configuration file should contain handlers."
    training_handler_name = "training_file_handler"
    assert (
        training_handler_name in config["handlers"]
    ), f"Logging configuration file is missing field '{training_handler_name}'."
    assert (
        len(config["loggers"]) == 1
    ), "Logging configuration file should contain only one training logger."

    # update configuration
    config["handlers"][training_handler_name]["filename"] = training_log_file

    # set logging
    logging.config.dictConfig(config)

    # get logger instance
    logger_name = list(config["loggers"].keys())[0]
    logger = logging.getLogger(logger_name)
    return logger


class TrainingState:
    """Class to log scores, track best scores, and save checkpoints with best scores.

    Parameters
    ----------
    model
        Pytorch neural network.
    path
        Experiment path for saving training outputs like checkpoints or logs.
    optimizer
        Optimizer instance for saving training state in case of interruption and need to resume.
    scheduler
        Scheduler instance for saving training state in case of interruption and need to resume.
    resume
        If True resumes run from a checkpoint with optimizer and scheduler state.
    device
        Device to use (cpu,0,1,2,...).
    """

    STATE_VARIABLES = (
        "last_epoch",
        "_elapsed_training_time",
        "best_loss",
        "best_scores_loss",
        "best_metrics",
        "best_scores_metrics",
    )

    def __init__(
        self,
        model: nn.Module,
        path: str = ".",
        *,
        ema_model: nn.Module = None,
        optimizer: Optimizer,
        scheduler: SchedulerType = None,
        resume: bool = False,
        device: torch.device = None,
    ):
        if resume:
            assert optimizer is not None
        self.model = model
        self.ema_model = ema_model
        self.path = path or "."
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        os.makedirs(self.path, exist_ok=True)

        # setup training logger
        self.t_logger = setup_training_logger(
            training_log_file=os.path.join(self.path, "training.log")
        )

        if resume:
            self.resume_training()
            self.t_logger.info(f"Resuming training after epoch {self.last_epoch}.")
        else:
            # create training state variables
            self.last_epoch = 0
            self._elapsed_training_time = 0.0

            self.best_loss = np.inf
            self.best_scores_loss = None

            self.best_metrics = {}  # best other metrics like accuracy or f1 score
            self.best_scores_metrics = {}  # string with all scores for each best metric

            self.t_logger.info("Training started.")
        self.start_training_time = time.time()

    def resume_training(self):
        """Resume training state from checkpoint.pth.tar file stored in the experiment directory."""
        # load training checkpoint to the memory
        checkpoint_path = os.path.join(self.path, "checkpoint.pth.tar")
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"Training checkpoint '{checkpoint_path}' not found.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # restore state variables of this class' instance (TrainingState)
        for variable in self.STATE_VARIABLES:
            if variable not in checkpoint["training_state"]:
                raise ValueError(
                    f"Training checkpoint '{checkpoint_path} is missing variable '{variable}'."
                )
        for k, v in checkpoint["training_state"].items():
            setattr(self, k, v)

        # load model, optimizer, and scheduler checkpoints
        self.model.load_state_dict(checkpoint["model"])
        if self.device is not None:
            self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler:
            if "scheduler" not in checkpoint:
                raise ValueError(f"Training checkpoint '{checkpoint_path}' is missing scheduler.")
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        # restore random state
        random_state = checkpoint["random_state"]
        random.setstate(random_state["python_random_state"])
        np.random.set_state(random_state["np_random_state"])
        torch.set_rng_state(random_state["torch_random_state"])
        if torch.cuda.is_available() and random_state["torch_cuda_random_state"] is not None:
            torch.cuda.set_rng_state(random_state["torch_cuda_random_state"])

    def _save_training_state(self, epoch: int):
        if self.optimizer is not None:
            # save state variables of this class' instance (TrainingState)
            training_state = {}
            for variable in self.STATE_VARIABLES:
                training_state[variable] = getattr(self, variable)

            # save random state variables
            random_state = dict(
                python_random_state=random.getstate(),
                np_random_state=np.random.get_state(),
                torch_random_state=torch.get_rng_state(),
                torch_cuda_random_state=(
                    torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                ),
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler and self.scheduler.state_dict(),
                    "training_state": training_state,
                    "random_state": random_state,
                },
                os.path.join(self.path, "checkpoint.pth.tar"),
            )

    def _save_checkpoint(self, epoch: int, metric_name: str, metric_value: float):
        """Save checkpoint to .pth file and log score.

        Parameters
        ----------
        epoch
            Epoch number.
        metric_name
            Name of metric (e.g. loss) based on which checkpoint is saved.
        metric_value
            Value of metric based on which checkpoint is saved.
        """
        metric_name = metric_name.lower()
        self.t_logger.info(
            f"Epoch {epoch} - "
            f"Save checkpoint with best validation {metric_name}: {metric_value:.6f}"
        )
        torch.save(
            self.model.state_dict(),
            os.path.join(self.path, f"best_{metric_name}.pth"),
        )

    def step(self, epoch: int, scores_str: str, valid_loss: float, valid_metrics: dict = None):
        """Log scores and save the best loss and metrics.

        Save checkpoints if the new best loss and metrics were achieved.
        Save training state for resuming the training if optimizer and scheduler are passed.

        The method should be called after training and validation of one epoch.

        Parameters
        ----------
        epoch
            Epoch number.
        scores_str
            Validation scores to log.
        valid_loss
            Validation loss based on which checkpoint is saved.
        valid_metrics
            Other validation metrics based on which checkpoint is saved.
        """
        self.last_epoch = epoch
        self.t_logger.info(f"Epoch {epoch} - {scores_str}")

        # save model checkpoint based on validation loss
        if valid_loss is not None and valid_loss is not np.nan and valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_scores_loss = scores_str
            self._save_checkpoint(epoch, "loss", self.best_loss)

        # save model checkpoint based on other metrics
        if valid_metrics is not None:
            if len(self.best_metrics) == 0:
                # set first values for self.best_metrics
                self.best_metrics = valid_metrics.copy()
                self.best_scores_metrics = {k: scores_str for k in self.best_metrics.keys()}
                for metric_name, metric_value in valid_metrics.items():
                    self._save_checkpoint(epoch, metric_name, metric_value)
            else:
                for metric_name, metric_value in valid_metrics.items():
                    if metric_value > self.best_metrics[metric_name]:
                        self.best_metrics[metric_name] = metric_value
                        self.best_scores_metrics[metric_name] = scores_str
                        self._save_checkpoint(epoch, metric_name, metric_value)

        # save training state for resuming the training
        self._save_training_state(epoch)

    def finish(self):
        """Log best scores achieved during training and save checkpoint of last epoch.

        The method should be called after training of all epochs is done.
        """
        # save checkpoint of the last epoch
        self.t_logger.info("Save checkpoint of the last epoch")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.path, f"epoch_{self.last_epoch}.pth"),
        )
        if self.ema_model is not None:
            self.t_logger.info("Save checkpoint of the EMA model")
            torch.save(
                self.ema_model.state_dict(),
                os.path.join(self.path, "EMA.pth"),
            )

        # remove training state
        os.remove(os.path.join(self.path, "checkpoint.pth.tar"))

        # make final training logs
        self.t_logger.info(f"Best scores (validation loss): {self.best_scores_loss}")
        for metric_name, best_scores_metric in self.best_scores_metrics.items():
            self.t_logger.info(f"Best scores (validation {metric_name}): {best_scores_metric}")
        elapsed_training_time = time.time() - self.start_training_time + self._elapsed_training_time
        self.t_logger.info(f"Training done in {elapsed_training_time}s.")


def set_random_seed(seed=777):
    """Set random seed.

    The method ensures multiple runs of the same experiment yield the same result.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClassificationTrainer(SchedulerMixin, MixupMixin, EMAMixin, BaseTrainer):
    """Class to perform training of a classification neural network and/or run inference.

    Parameters
    ----------
    model
        Pytorch neural network.
    trainloader
        Pytorch dataloader with training data.
    criterion
        Loss function.
    optimizer
        Optimizer algorithm.
    validloader
        Pytorch dataloader with validation data.
    scheduler
        Scheduler algorithm.
    accumulation_steps
        Number of iterations to accumulate gradients before performing optimizer step.
    clip_grad
        Max norm of the gradients for the gradient clipping.
    device
        Device to use (cpu,0,1,2,...).
    train_scores_fn
        Function for evaluating scores on the training data.
    valid_scores_fn
        Function for evaluating scores on the validation data.
    wandb_train_prefix
        Prefix string to include in the name of training scores logged to W&B.
    wandb_valid_prefix
        Prefix string to include in the name of validations scores logged to W&B.
    mixup
        Mixup alpha value, mixup is active if > 0.
    cutmix
        Cutmix alpha value, cutmix is active if > 0.
    mixup_prob
        Probability of applying mixup or cutmix per batch.
    apply_ema
        Apply EMA model weight averaging if true.
    ema_start_epoch
        Epoch number when to start model averaging.
    ema_decay
        Model weight decay.
    """

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        *,
        validloader: DataLoader = None,
        scheduler: SchedulerType = None,
        accumulation_steps: int = 1,
        clip_grad: float = None,
        device: torch.device = None,
        train_scores_fn: Callable = None,
        valid_scores_fn: Callable = None,
        wandb_train_prefix: str = "Train. ",
        wandb_valid_prefix: str = "Val. ",
        # mixup parameters
        mixup: float = None,
        cutmix: float = None,
        mixup_prob: float = None,
        # ema parameters
        apply_ema: bool = False,
        ema_start_epoch: int = 0,
        ema_decay: float = 0.9999,
        **kwargs,
    ):
        if train_scores_fn is None:

            def _train_scores_fn(preds, targs):
                return Exception("Not implemented in fgvc subset")
                return classification_scores(preds, targs, top_k=None, return_dict=True)

            train_scores_fn = _train_scores_fn
        if valid_scores_fn is None:

            def _valid_scores_fn(preds, targs):
                return Exception("Not implemented in fgvc subset")
                # return classification_scores(preds, targs, top_k=3, return_dict=True)

            valid_scores_fn = _valid_scores_fn
        assert hasattr(train_scores_fn, "__call__")
        assert hasattr(valid_scores_fn, "__call__")
        self.train_scores_fn = train_scores_fn
        self.valid_scores_fn = valid_scores_fn

        self.wandb_train_prefix = wandb_train_prefix
        self.wandb_valid_prefix = wandb_valid_prefix

        super().__init__(
            model=model,
            trainloader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            validloader=validloader,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            clip_grad=clip_grad,
            device=device,
            mixup=mixup,
            cutmix=cutmix,
            mixup_prob=mixup_prob,
            apply_ema=apply_ema,
            ema_start_epoch=ema_start_epoch,
            ema_decay=ema_decay,
        )
        if len(kwargs) > 0:
            warnings.warn(f"Class {self.__class__.__name__} got unused key arguments: {kwargs}")

    def train_epoch(self, epoch: int, dataloader: DataLoader) -> TrainEpochOutput:
        """Train one epoch.

        Parameters
        ----------
        epoch
            Epoch number.
        dataloader
            PyTorch dataloader with training data.

        Returns
        -------
        TrainEpochOutput tuple with average loss and average scores.
        """
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        num_updates = epoch * len(dataloader)
        max_grad_norm = 0.0
        loss_monitor = LossMonitor(num_batches=len(dataloader))
        scores_monitor = ScoresMonitor(
            scores_fn=self.train_scores_fn, num_samples=len(dataloader.dataset), eval_batches=False
        )
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.train_batch(batch)
            loss_monitor.update(loss)
            scores_monitor.update(preds, targs)

            # make optimizer step
            if (i - 1) % self.accumulation_steps == 0:
                grad_norm = get_gradient_norm(self.model.parameters(), norm_type=2)
                max_grad_norm = max(max_grad_norm, grad_norm)  # store maximum gradient norm
                if self.clip_grad is not None:  # apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # update average model
                self.make_ema_update(epoch + 1)

                # update lr scheduler from timm library
                num_updates += 1
                self.make_timm_scheduler_update(num_updates)

        return TrainEpochOutput(
            loss_monitor.avg_loss,
            scores_monitor.avg_scores,
            max_grad_norm,
            loss_monitor.other_avg_losses,
        )

    def predict(
        self, dataloader: DataLoader, return_preds: bool = True, *, model: nn.Module = None
    ) -> PredictOutput:
        """Run inference.

        Parameters
        ----------
        dataloader
            PyTorch dataloader with validation/test data.
        return_preds
            If True, the method returns predictions and ground-truth targets.
        model
            Alternative PyTorch model to use for prediction like EMA model.

        Returns
        -------
        PredictOutput tuple with predictions, ground-truth targets,
        average loss, and average scores.
        """
        model = model or self.model
        model.to(self.device)
        model.eval()

        # logger.debug("LossMonitor and ScoresMonitor are not defined in fgvc subset")
        loss_monitor = LossMonitor(num_batches=len(dataloader))
        scores_monitor = ScoresMonitor(
            scores_fn=self.valid_scores_fn,
            num_samples=len(dataloader.dataset),
            eval_batches=False,
            store_preds_targs=return_preds,
        )
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            preds, targs, loss = self.predict_batch(batch, model=model)
            loss_monitor.update(loss)
            scores_monitor.update(preds, targs)
        return PredictOutput(
            # preds,
            # targs,
            # None,
            # None
            scores_monitor.preds_all,
            scores_monitor.targs_all,
            loss_monitor.avg_loss,
            scores_monitor.avg_scores,
        )

    def train(
        self,
        num_epochs: int = 1,
        seed: int = 777,
        path: str = None,
        resume: bool = False,
    ):
        """Train neural network.

        Parameters
        ----------
        num_epochs
            Number of epochs to train.
        seed
            Random seed to set.
        path
            Experiment path for saving training outputs like checkpoints or logs.
        resume
            If True resumes run from a checkpoint with optimizer and scheduler state.
        """
        # create training state
        training_state = TrainingState(
            self.model,
            path=path,
            ema_model=self.get_ema_model(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            resume=resume,
            device=self.device,
        )

        # run training loop
        if not resume:
            # set random seed when training from the start
            # otherwise, when resuming training use state from the checkpoint
            set_random_seed(seed)
        for epoch in range(training_state.last_epoch, num_epochs):
            # apply training and validation on one epoch
            start_epoch_time = time.time()
            train_output = self.train_epoch(epoch, self.trainloader)
            predict_output = PredictOutput()
            ema_predict_output = None
            if self.validloader is not None:
                predict_output = self.predict(self.validloader, return_preds=False)
                if getattr(self, "ema_model") is not None:
                    ema_predict_output = self.predict(
                        self.validloader, return_preds=False, model=self.get_ema_model()
                    )
            elapsed_epoch_time = time.time() - start_epoch_time

            # make a scheduler step
            lr = self.optimizer.param_groups[0]["lr"]
            self.make_scheduler_step(epoch + 1, valid_loss=predict_output.avg_loss)

            # log scores to W&B
            ema_scores = ema_predict_output.avg_scores if ema_predict_output is not None else {}
            ema_scores = {f"{k} (EMA)": v for k, v in ema_scores.items()}
            log_progress(
                epoch + 1,
                train_loss=train_output.avg_loss,
                valid_loss=predict_output.avg_loss,
                train_scores={**train_output.avg_scores, **train_output.other_avg_losses},
                valid_scores={**predict_output.avg_scores, **ema_scores},
                lr=lr,
                max_grad_norm=train_output.max_grad_norm,
                train_prefix=self.wandb_train_prefix,
                valid_prefix=self.wandb_valid_prefix,
            )

            # log scores to file and save model checkpoints
            _scores = {
                "avg_train_loss": f"{train_output.avg_loss:.4f}",
                "avg_val_loss": f"{predict_output.avg_loss:.4f}",
                **{
                    s: f"{predict_output.avg_scores.get(s, np.nan):.2%}"
                    for s in ["F1", "Accuracy", "Recall@3"]
                },
                "time": f"{elapsed_epoch_time:.0f}s",
            }
            training_state.step(
                epoch + 1,
                scores_str="\t".join([f"{k}: {v}" for k, v in _scores.items()]),
                valid_loss=predict_output.avg_loss,
                valid_metrics=predict_output.avg_scores,
            )

        # save last checkpoint, log best scores and total training time
        training_state.finish()


def predict(
    model: nn.Module,
    testloader: DataLoader,
    *,
    criterion: nn.Module = None,
    device: torch.device = None,
    trainer_cls: Type[BaseTrainer] = ClassificationTrainer,
    trainer_kws: dict = None,
    predict_kws: dict = None,
    **kwargs,
) -> PredictOutput:
    """Run inference.

    Parameters
    ----------
    model
        PyTorch neural network.
    testloader
        PyTorch dataloader with test data.
    criterion
        Loss function.
    device
        Device to use (CPU,CUDA,CUDA:0,...).
    trainer_cls
        Trainer class that implements `train`, `train_epoch`, and `predict` functions
        and inherits from `BaseTrainer` PyTorch class.
    trainer_kws
        Additional keyword arguments for the trainer class.

    Returns
    -------
    preds
        Numpy array with predictions.
    targs
        Numpy array with ground-truth targets.
    avg_loss
        Average loss.
    avg_scores
        Average scores.
    """
    trainer_kws = trainer_kws or {}
    predict_kws = predict_kws or {}
    trainer = trainer_cls(
        model=model,
        trainloader=None,
        criterion=criterion,
        optimizer=None,
        device=device,
        **trainer_kws,
        **kwargs,
    )
    return trainer.predict(
        dataloader=testloader,
        return_preds=True,
        **predict_kws,
    )
