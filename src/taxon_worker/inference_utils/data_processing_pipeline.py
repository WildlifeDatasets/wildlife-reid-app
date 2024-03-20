import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import wandb
import yaml
from scipy.special import softmax

from fgvc.core.training import predict
from fgvc.datasets import get_dataloaders
from fgvc.utils.experiment import load_model

from .config import RESOURCES_DIR, WANDB_API_KEY, WANDB_ARTIFACT_PATH, WANDB_ARTIFACT_PATH_CROPPED
from .dataset_tools import data_preprocessing
from .prediction_dataset import PredictionDataset

logger = logging.getLogger("app")
MEDIA_DIR_PATH = Path("/shared_data/media")


def get_model_config(is_cropped: bool = False) -> Tuple[dict, str, dict]:
    """Load model configuration from W&B including training config and fine-tuned checkpoint."""
    # load model_meta.json and check if the artifact path is the same as the previous one
    model_meta_path = Path(RESOURCES_DIR) / "model_meta.json"
    reset_model = True
    if model_meta_path.is_file():
        with open(model_meta_path) as f:
            model_meta = json.load(f)
        if model_meta["WANDB_ARTIFACT_PATH"] == WANDB_ARTIFACT_PATH:
            reset_model = False
        else:
            logger.debug(
                f"New model={WANDB_ARTIFACT_PATH}. "
                + f"Old model={model_meta.get('artifact_path', 'None')}."
            )
    if reset_model:
        logger.debug("Resetting model.")
        shutil.rmtree(RESOURCES_DIR, ignore_errors=True)
        model_meta = {"WANDB_ARTIFACT_PATH": WANDB_ARTIFACT_PATH}
        model_meta_path.parent.mkdir(exist_ok=True, parents=True)
        with open(model_meta_path, "w") as f:
            json.dump(model_meta, f)

    # get artifact and run from W&B
    if is_cropped:
        wandb_artifact_path = WANDB_ARTIFACT_PATH_CROPPED
        model_config_path = Path(RESOURCES_DIR) / "model_config_cropped.json"
    else:
        wandb_artifact_path = WANDB_ARTIFACT_PATH
        model_config_path = Path(RESOURCES_DIR) / "model_config.json"
    try:
        api = wandb.Api(api_key=WANDB_API_KEY)
        artifact = api.artifact(wandb_artifact_path)
        run = artifact.logged_by()
        config = run.config
        # save model config locally for later use without internet
        model_config_path.parent.mkdir(exist_ok=True, parents=True)
        with open(model_config_path, "w") as f:
            json.dump(config, f)

        logger.debug(f"Downloading artifact {wandb_artifact_path}.")
        artifact_files = [x.name for x in artifact.files()]
        logger.debug(f"Artifact files: {artifact_files}")

        # check if all artifact files are downloaded and optionally download artifact files
        all_files_downloaded = all([(Path(RESOURCES_DIR) / x).is_file() for x in artifact_files])
        if not all_files_downloaded:
            logger.debug("Downloading artifact files.")
            artifact.download(root=RESOURCES_DIR)
    except (wandb.CommError, ConnectionError):
        logger.error("Connection Error. Cannot reach W&B server. Trying previous configuration.")
        artifact_files = [fn.relative_to(RESOURCES_DIR) for fn in Path(RESOURCES_DIR).glob("*")]
        with open(model_config_path) as f:
            config = json.load(f)

    # get artifact contents
    assert (
        sum([Path(x).suffix.lower() == ".pth" for x in artifact_files]) == 1
    ), "Only one '.pth' file expected in the W&B artifact."
    _suffix2name = {Path(x).suffix.lower(): x for x in artifact_files}
    checkpoint_path = Path(RESOURCES_DIR) / _suffix2name[".pth"]
    artifact_config_path = Path(RESOURCES_DIR) / "config.yaml"
    with open(artifact_config_path) as f:
        artifact_config = yaml.safe_load(f)

    return config, checkpoint_path, artifact_config


def load_model_and_predict(image_paths: list) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """Load model, create dataloaders, and run inference."""
    # from .data_preprocessing import detect_animal, pad_image, detect_animals
    # is_detected = detect_animals(image_paths)
    config, checkpoint_path, artifact_config = get_model_config()

    logger.info("Creating model and loading fine-tuned checkpoint.")
    model, model_mean, model_std = load_model(config, checkpoint_path)
    logger.debug(
        f"model_mean={model_mean}, model_std={model_std}, "
        + f"checkpoint_path={checkpoint_path}, config={config}"
    )
    logger.debug(f"{image_paths[0]=}")

    logger.info("Creating DataLoaders.")
    _, testloader, _, _ = get_dataloaders(
        None,
        image_paths,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=0,
        dataset_cls=PredictionDataset,
    )
    # do_confidence_thresholding: Do not label uncertain data.
    if "do_confidence_thresholding" in artifact_config:
        do_confidence_thresholding = artifact_config["do_confidence_thresholding"]
    else:
        do_confidence_thresholding = True

    logger.info("Running inference.")
    predict_output = predict(model, testloader)
    logits = predict_output.preds
    if "temperature" in artifact_config:
        logits = logits / artifact_config["temperature"]
    else:
        logger.warning("Artifact config from W&B is missing 'temperature' parameter.")
    probs = softmax(logits, 1)

    id2thresholds = artifact_config.get("id2th")

    assert list(id2thresholds.keys()) == sorted(
        list(id2thresholds.keys())
    ), "Artifact id2th does not contain sorted list of thresholds for every class."

    id2label = artifact_config.get("id2label")
    assert np.max(list(id2label.keys())) == (
        len(id2label) - 1
    ), "Some of the labels is missing in id2label."

    # add inference results to the metadata dataframe
    if do_confidence_thresholding:
        class_ids, probs_top = do_thresholding_on_probs(probs, id2thresholds)
        # extend labels and add there new label "Not Classified"
        id2label[len(id2label)] = "Not Classified"
    else:
        class_ids = np.argmax(probs, 1)
        probs_top = np.max(probs, 1)

    return class_ids, probs_top, id2label


def do_thresholding_on_probs(probs: np.array, id2threshold: dict) -> Tuple[np.array, np.array]:
    """Use the thresholds to do the classification.

    The images with all softmax values under the classification threshold are marked
    as not-classified.

    Args:
        probs (numpy.array): softmax of logits with shape = [n_samples, n_classes].
        id2threshold (dict): Thresholds in dictionary. Key is the class id, value is the threshold.

    Returns:
        class_ids: Ids of classes. The highest class id represent the not classified category.
        top_probs: Softmaxed probability. The not-classified is calculated as 1 - top_prob.

    """
    assert probs.shape[1] == len(
        id2threshold
    ), "There should be the same number of columns as the number of classes."

    top_probs = np.max(probs, 1)
    class_ids = np.argmax(probs, 1)
    thresholds = np.array(list(id2threshold.values()))
    thresholds_per_sample = thresholds[class_ids]

    is_classified = top_probs > thresholds_per_sample
    # add new label with id = maximum_id + 1
    class_ids[~is_classified] = len(id2threshold)
    top_probs[~is_classified] = 1 - top_probs[~is_classified]

    return class_ids, top_probs


def data_processing(
    zip_path: Path,
    media_dir_path: Path,
    csv_path: Path,
    *,
    num_cores: Optional[int] = None,
    contains_identities: bool = False,
) -> pd.DataFrame:
    """Preprocessing and prediction on data in ZIP file.

    Files are renamed according to the hash based on input path.

    Parameters
    ----------
    zip_path
        Path to input zip file.
    media_dir_path
        Path to content of zip. The file names are hashed.
    csv_path
        Path to output CSV file.
    """
    logger.debug(f"{media_dir_path=}, {zip_path=}, {csv_path=}")
    # create metadata dataframe
    metadata, _ = data_preprocessing(
        zip_path,
        media_dir_path,
        num_cores=num_cores,
        contains_identities=contains_identities,
    )
    metadata = keep_correctly_loaded_images(metadata)

    run_inference(metadata)

    # move_files_from_temp(media_dir_path, metadata)

    # save metadata file
    metadata.to_csv(csv_path, encoding="utf-8-sig")
    return metadata


# def move_files_from_temp(media_dir_path, metadata):
#     """ Move files from temporary directory to media directory. """
#     new_image_paths = []
#     for i, row in metadata.iterrows():
#         image_path = Path(media_dir_path) / row["image_path"]
#         target_dir = Path(media_dir_path)
#         target_dir.mkdir(parents=True, exist_ok=True)
#         target_image_path = target_dir / row["image_path"]
#         shutil.move(image_path, target_image_path)
#         new_image_paths.append(str(row["image_path"]))
#     metadata["image_path"] = new_image_paths


def run_inference(metadata):
    """Use full_image_path for taxon prediction."""
    # run inference
    # image_path = metadata["image_path"].apply(lambda x: os.path.join(MEDIA_DIR_PATH, x))
    image_path = metadata["full_image_path"]
    logger.debug(f"{image_path=}")
    class_ids, probs_top, id2label = load_model_and_predict(image_path)
    # add inference results to the metadata dataframe
    metadata["predicted_class_id"] = class_ids
    metadata["predicted_prob"] = probs_top
    metadata["predicted_category"] = np.nan
    if id2label is not None:
        metadata["predicted_category"] = metadata["predicted_class_id"].apply(
            lambda x: id2label.get(x, np.nan)
        )


def keep_correctly_loaded_images(metadata):
    """Remove file from list if there is the error message."""
    logger.debug(f"len(metadata)={len(metadata)}")
    # metadata = metadata[metadata["media_type"] == "image"].reset_index(drop=True)
    # keep media_type==image and media_type == video
    metadata = metadata[
        (metadata["media_type"] == "image") | (metadata["media_type"] == "video")
    ].reset_index(drop=True)
    # drop media_type== "unknown"
    # mediadata = metadata[metadata["media_type"] != "unknown"].reset_index(drop=True)
    logger.debug(f"len(metadata)={len(metadata)}")
    metadata = metadata[metadata["read_error"] == ""].reset_index(drop=True)
    logger.debug(f"len(metadata)={len(metadata)}")
    return metadata
