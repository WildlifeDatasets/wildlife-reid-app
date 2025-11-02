import logging
import os
from itertools import chain

import numpy as np
import pandas as pd
import timm
import torch
from torch.optim import SGD, AdamW
from train.callbacks import AccuracyCallback, EpochCheckpoint, FileEpochLog
from train.data_split import split_data
from train.tools import load_data, save_data
from train.trainer import CarnivoreIDTrainer
from wildlife_tools import realize
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.train import ArcFaceLoss
from wildlife_tools.train.callbacks import EpochCallbacks

from utils import config
from utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")

logger.debug(f"{config.RABBITMQ_URL=}")
logger.debug(f"{config.REDIS_URL=}")
logger.debug(f"{config.POSTGRES_URL=}")


def get_transforms(image_size):
    """Get training and validation transforms."""
    config = {
        "method": "TransformTimm",
        "input_size": image_size,
        "is_training": True,
        "auto_augment": "rand-m10-n2-mstd1",
    }
    transform = realize(config)
    resize_t, flip_t, rand_t, tensor_t, normalize_t = transform.transforms
    for t in rand_t.ops:
        t.kwargs["fillcolor"] = (0, 0, 0)
    resize_t.scale = (0.7, 1)
    transform.transforms = [resize_t, rand_t, tensor_t, normalize_t]

    config = {
        "method": "TransformTimm",
        "input_size": image_size,
        "is_training": False,
        "auto_augment": "rand-m10-n2-mstd1",
    }
    transform_val = realize(config)

    return transform, transform_val


def get_io_size(model):
    """Get input image size and embedding size from model."""
    if not hasattr(model, "default_cfg") or not hasattr(model, "feature_info"):
        return None

    image_size = np.max(model.default_cfg["input_size"])
    embedding_size = model.feature_info[-1]["num_chs"]

    return image_size, embedding_size


def load_model(model_name, model_checkpoint=""):
    """Load model from timm with optional checkpoint."""
    # load model checkpoint
    model = timm.create_model(model_name, num_classes=0, pretrained=True)

    if model_checkpoint:
        model_ckpt = torch.load(model_checkpoint)["model"]
        model.load_state_dict(model_ckpt)

    return model


def train_identification_model(
    input_metadata_file: str,
    organization_id: int,
    identification_model: dict,
    **kwargs,
):
    """Train identification model based on provided metadata file and configuration."""
    config = {
        "lr": 0.0001,
        "device": "cuda",
        "epochs": 100,
        "num_workers": 8,
        "batch_size": 16,
        "accumulation_steps": 4,
        "optimizer": "adamw",
        "scheduler": "cos",
        "image_size": 256,
        "embedding_size": 768,
        "identification_model": identification_model,
        "organization_id": organization_id,
    }

    # logger.debug(f"{input_metadata_file=}")
    # logger.debug(f"{identification_model=}")

    # Check if to start or continue training
    output_model_path = identification_model["path"]
    output_folder = os.path.dirname(output_model_path)
    resume = False
    if os.path.exists(os.path.join(output_folder, "config.json")) and os.path.exists(
        os.path.join(output_folder, "status.json")
    ):
        resume = True

    logger.debug(f"{output_folder=}")
    logger.debug(f"{resume=}")

    # generate output folder path
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # output_folder = f"training_{organization_id}_{timestamp}"
    # os.makedirs(output_folder, exist_ok=True)

    # Create status and config files
    if not resume:
        logger.info("Initializing training")
        status = {"epochs_trained": 0, "last_checkpoint_path": "", "stage": "init"}
        save_data(os.path.join(output_folder, "status.json"), status)
        save_data(os.path.join(output_folder, "config.json"), config)
    else:
        status: dict = load_data(os.path.join(output_folder, "status.json"))
        config: dict = load_data(os.path.join(output_folder, "config.json"))
        if status["stage"] == "finished":
            logger.info("Training already finished. Exiting...")
            return
        logger.info(
            f"Continuing training from folder: {output_folder}. "
            f"Starting from epoch: {status['epochs_trained']}, target epochs: {config['epochs']}"
        )

    # Load model
    model = load_model(identification_model["init_path"], model_checkpoint=status["last_checkpoint_path"])
    io_size = get_io_size(model)
    if io_size is not None:
        image_size, embedding_size = io_size
    else:
        image_size, embedding_size = config["image_size"], config["embedding_size"]

    # Load necessary metadata
    metadata = pd.read_csv(input_metadata_file)
    logger.debug(f"{metadata.sequence_number=}")
    metadata["path"] = metadata["image_path"].apply(lambda p: p.replace("images", "masked_images"))
    metadata["identity"] = metadata["label"]

    # logger.debug(f"{metadata.columns=}")
    # TODO: check if metadata contain observation_id
    assert "path" in metadata
    assert "identity" in metadata

    metadata = metadata.loc[:, ["path", "identity"]]

    if "observation_id" not in metadata.columns:
        metadata["observation_id"] = list(range(len(metadata)))

    # Data filtration
    logger.info("Starting training metadata filtration")
    logger.info(
        f"    Initialization\n"
        f"    {'    '*15}-> file counts: {len(metadata)}, identity counts: {len(set(metadata['identity']))}"
    )

    # Remove identities that occur only once
    metadata = metadata[metadata["identity"].duplicated(keep=False)].reset_index(drop=True)
    logger.info(
        f"    Single file identities filtration\n"
        f"    {'    '*15}-> file counts: {len(metadata)}, identity counts: {len(set(metadata['identity']))}"
    )

    # Check if the files exist
    exists_mask = metadata["path"].map(os.path.exists)
    # failed_paths = metadata.loc[~exists_mask, "path"].tolist()
    metadata = metadata.loc[exists_mask].reset_index(drop=True)
    logger.info(
        f"    Missing files filtration\n"
        f"    {'    '*15}-> file counts: {len(metadata)}, identity counts: {len(set(metadata['identity']))}"
    )

    if len(metadata) <= 1:
        logger.info("Cancelling training, not enough representativ images.")
        return {"status": "ERROR", "error": "Cancelling training, not enough representativ images."}

    # Split data
    # remove_tail = 5
    # if len(metadata) < 100:
    #     remove_tail = 0
    # metadata_train, metadata_val = split_data(metadata, 0.8, remove_tail=remove_tail, remove_both_tails=True)
    metadata_train, metadata_val = split_data(metadata, 0.8, remove_tail=0, remove_both_tails=False)

    # Create datasets
    train_transforms, val_transforms = get_transforms(image_size)
    img_load = "full"  # crop_black - if not cropped
    train_dataset = WildlifeDataset(metadata=metadata_train, transform=train_transforms, img_load=img_load)
    val_dataset = WildlifeDataset(metadata=metadata_val, transform=val_transforms, img_load=img_load)

    # Create loss function
    objective = ArcFaceLoss(
        num_classes=np.max([val_dataset.num_classes, train_dataset.num_classes]),
        embedding_size=embedding_size,
        margin=0.25,
        scale=16,
    )

    # Create optimizer and scheduler
    params = chain(model.parameters(), objective.parameters())
    if config["optimizer"].lower() == "sgd":
        optimizer = SGD(params=params, lr=config["lr"], momentum=0.9)
    elif config["optimizer"].lower() == "adamw":
        optimizer = AdamW(params=params, lr=config["lr"])
    else:
        raise ValueError("No optimizer name was provided.")

    scheduler = None
    if config["scheduler"] == "cos":
        lr_min = config["lr"] * 1e-3
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=lr_min)
    elif config["scheduler"] == "wu_cos":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            pct_start=0.1,
            div_factor=100,
            steps_per_epoch=1,
            epochs=config["epochs"],
        )

    # Load checkpoint if resume is True and checkpoint exists
    if resume and os.path.exists(status["last_checkpoint_path"]):
        checkpoint = torch.load(status["last_checkpoint_path"])
        logger.info(
            f"Resuming training from checkpoint: {status['last_checkpoint_path']}\n"
            f"Starting from epoch: {checkpoint['epoch']}/{config['epochs']}"
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

    # Define callbacks
    callbacks = [
        AccuracyCallback(train_dataset, val_dataset, log_period=10),
        EpochCheckpoint(
            status_path=os.path.join(output_folder, "status.json"),
            checkpoint_path=output_model_path,
        ),
        FileEpochLog(status_path=os.path.join(output_folder, "status.json")),
    ]
    epoch_callback = EpochCallbacks(callbacks)

    status["stage"] = "training"
    save_data(os.path.join(output_folder, "status.json"), status)

    logger.info(f"Starting training with {len(metadata)} images and {len(set(metadata['identity']))} identities.")
    # define trainer and start training
    trainer = CarnivoreIDTrainer(
        val_dataset=val_dataset,
        dataset=train_dataset,
        model=model,
        objective=objective,
        optimizer=optimizer,
        epoch_callback=epoch_callback,
        scheduler=scheduler,
        device=config["device"],
        epochs=config["epochs"],
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        accumulation_steps=config["accumulation_steps"],
        start_epoch=status["epochs_trained"],
    )
    trainer.train()

    status = load_data(os.path.join(output_folder, "status.json"))
    status["stage"] = "finished"
    save_data(os.path.join(output_folder, "status.json"), status)

    logger.info("Training finished!!!")
