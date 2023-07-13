import logging
import os
import shutil
import uuid
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

from .config import RESOURCES_DIR, WANDB_API_KEY, WANDB_ARTIFACT_PATH
from .dataset_tools import (
    SumavaInitialProcessing,
    extend_df_with_sequence_id,
    extract_information_from_dir_structure,
    get_lynx_id_in_sumava,
    make_dataset,
)
from .inout import extract_archive
from .prediction_dataset import PredictionDataset

logger = logging.getLogger("app")


def analyze_dataset_directory(dataset_dir_path: Path, num_cores: Optional[int] = None):
    """Get species, locality, datetime and sequence_id from directory with media files.

    Parameters
    ----------
    dataset_dir_path
        Input directory.

    Returns
    -------
    metadata: DataFrame
        Image and video metadata.
    duplicates: DataFrame
        List of duplicit files.
    """
    init_processing = SumavaInitialProcessing(dataset_dir_path, num_cores=num_cores)
    df0 = init_processing.make_paths_and_exifs_parallel(
        mask="**/*.*", make_exifs=True, make_csv=False
    )

    df = extract_information_from_dir_structure(df0)

    df["datetime"] = pd.to_datetime(df0.datetime, errors="coerce")
    df["read_error"] = list(df0["read_error"])

    df.loc[:, "sequence_number"] = None

    # Get ID of lynx from directories in basedir beside "TRIDENA" and "NETRIDENA"
    df["unique_name"] = df["vanilla_path"].apply(get_lynx_id_in_sumava)

    df = extend_df_with_sequence_id(df, time_limit="120s")

    # Create list of duplicates based on the same EXIF time
    duplicates = df[df.delta_datetime != pd.Timedelta("0s")]
    duplicates = duplicates.copy().reset_index(drop=True)
    # duplicates.to_csv(
    #     "../../../resources/Sumava/list_of_duplicities.csv"
    # )

    # Remove duplicities
    # does not work if the images with unique name are also in TRIDENA or NETRIDENA
    df = df[df.delta_datetime != pd.Timedelta("0s")].reset_index(drop=True)

    # Turn NaN int None
    metadata = df.where(pd.notnull(df), None)

    return metadata, duplicates


def data_preprocessing(
    zip_path: Path, media_dir_path: Path, num_cores: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocessing of data in zip file.

    If the Sumava data dir structure is present, the additional information is extracted.
    Sumava data dir structure: "TRIDENA/SEASON/LOCATION/DATE/SPECIES"

    Parameters
    ----------
    zip_path: file with zipped images
    media_dir_path: output dir for media files with hashed names
    csv_path: Path to csv file

    Returns
    -------
    metadata: DataFrame - Image and video metadata

    duplicates: DataFrame - List of duplicate files
    """
    # create temporary directory
    tmp_dir = Path(f"/tmp/{str(uuid.uuid4())}")
    tmp_dir.mkdir(exist_ok=False, parents=True)

    # extract files to the temporary directory
    extract_archive(zip_path, output_dir=tmp_dir)

    # create metadata directory
    df, duplicates = analyze_dataset_directory(tmp_dir, num_cores=num_cores)
    # df["vanilla_path"].map(lambda fn: dataset_tools.make_hash(fn, prefix="media_data"))
    df = make_dataset(
        df=df,
        dataset_name=None,
        dataset_base_dir=tmp_dir,
        output_path=media_dir_path,
        hash_filename=True,
        make_tar=False,
        move_files=True,
        create_csv=False,
    )

    return df, duplicates


def get_model_config() -> Tuple[dict, str, dict]:
    """Load model configuration from W&B including training config and fine-tuned checkpoint."""
    # get artifact and run from W&B
    api = wandb.Api(api_key=WANDB_API_KEY)
    artifact = api.artifact(WANDB_ARTIFACT_PATH)
    run = artifact.logged_by()
    config = run.config
    artifact_files = [x.name for x in artifact.files()]

    # check if all artifact files are downloaded and optionally download artifact files
    all_files_downloaded = all([(Path(RESOURCES_DIR) / x).is_file() for x in artifact_files])
    if not all_files_downloaded:
        artifact.download(root=RESOURCES_DIR)

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


def load_model_and_predict(image_paths: list) -> Tuple[np.ndarray, Optional[dict]]:
    """Load model, create dataloaders, and run inference."""
    config, checkpoint_path, artifact_config = get_model_config()

    logger.info("Creating model and loading fine-tuned checkpoint.")
    model, model_mean, model_std = load_model(config, checkpoint_path)

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
    thresholds_extended = (
        np.array(list(id2threshold.values())).reshape(1, -1).repeat(repeats=probs.shape[0], axis=0)
    )

    probs_cp = probs.copy()
    probs_cp[probs <= thresholds_extended] = 0

    top_probs = np.max(probs_cp, 1)
    class_ids = np.argmax(probs_cp, 1)
    is_there_some_class = np.max(probs > thresholds_extended, axis=1)
    # add new label with id = maximum_id + 1
    class_ids[~is_there_some_class] = len(id2threshold)
    top_probs[~is_there_some_class] = 1 - top_probs[~is_there_some_class]

    return class_ids, top_probs


def data_processing(
    zip_path: Path,
    media_dir_path: Path,
    csv_path: Path,
    *,
    num_cores: Optional[int] = None,
):
    """Preprocessing and prediction on data in ZIP file.

    Files are renamed according to the hash based on input path.

    Args:
        zip_path: Path to input zip file.
        media_dir_path: Path to content of zip. The file names are hashed.
        csv_path: Path to output CSV file.

    """
    # create metadata dataframe
    metadata, _ = data_preprocessing(zip_path, media_dir_path, num_cores=num_cores)
    logger.debug(f"len(metadata)={len(metadata)}")
    metadata = metadata[metadata["media_type"] == "image"].reset_index(drop=True)
    logger.debug(f"len(metadata)={len(metadata)}")
    metadata = metadata[metadata["read_error"] == ""].reset_index(drop=True)
    logger.debug(f"len(metadata)={len(metadata)}")

    # run inference
    image_path = metadata["image_path"].apply(lambda x: os.path.join(media_dir_path, x))
    class_ids, _, id2label = load_model_and_predict(image_path)

    # add inference results to the metadata dataframe
    metadata["predicted_class_id"] = class_ids
    metadata["predicted_category"] = np.nan
    if id2label is not None:
        metadata["predicted_category"] = metadata["predicted_class_id"].apply(
            lambda x: id2label.get(x, np.nan)
        )

    # create category subdirectories and move images based on prediction
    new_image_paths = []
    for i, row in metadata.iterrows():
        if pd.notnull(row["predicted_category"]):
            predicted_category = row["predicted_category"]
        else:
            predicted_category = f"class_{row['predicted_class_id']}"

        image_path = Path(media_dir_path) / row["image_path"]
        target_dir = Path(media_dir_path, predicted_category)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image_path = target_dir / row["image_path"]
        shutil.move(image_path, target_image_path)
        new_image_paths.append(os.path.join(predicted_category, row["image_path"]))
    metadata["image_path"] = new_image_paths

    # save metadata file
    metadata.to_csv(csv_path)
