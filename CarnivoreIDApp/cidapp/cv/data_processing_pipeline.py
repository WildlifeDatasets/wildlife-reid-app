import logging
import shutil
import tarfile
from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import wandb
from scipy.special import softmax
from tqdm import tqdm

from fgvc.utils.experiment import load_model
from . import dataset_tools

logger = logging.getLogger(__file__)


def extract_tarfile(tarfile_path: Path, output_dir_path: Path):
    """Extract tar file."""
    with tarfile.open(tarfile_path, "r") as tf:
        tf.extractall(path=output_dir_path)


def extract_zipfile(zipfile_path: Path, output_dir_path: Path):
    """Extract content of zip file."""
    # loading the temp.zip and creating a zip object
    with ZipFile(zipfile_path) as zObject:
        # Extracting all the members of the zip
        # into a specific location.
        zObject.extractall(path=output_dir_path)


def analyze_dataset_directory(dataset_dir_path: Path):
    """Get species, locality, datetime and sequence_id from directory with media files.

    Parameters
    ----------
    dataset_dir_path: Input directory

    Returns
    -------
    metadata: DataFrame - Image and video metadata

    duplicates: DataFrame - List of duplicit files

    """
    init_processing = dataset_tools.SumavaInitialProcessing(dataset_dir_path)
    df0 = init_processing.make_paths_and_exifs_parallel(
        mask="**/*.*", make_exifs=True, make_csv=False
    )

    df = dataset_tools.extract_information_from_dir_structure(df0)
    # df["mediatype"] = df.vanilla_path.progress_map(
    #     lambda path:
    #     "video"
    #     if Path(path).suffix.lower() in (".avi", ".m4v")
    #     else "image"
    #     if Path(path).suffix.lower() in (".jpg", "png")
    #     else "unknown"
    # )

    df["datetime"] = pd.to_datetime(df0.datetime, errors="coerce")

    df.loc[:, "sequence_number"] = None

    # Get ID of lynx from directories in basedir beside "TRIDENA" and "NETRIDENA"
    tqdm.pandas(desc="unique_name    ")
    df.unique_name = df.vanilla_path.progress_map(dataset_tools.get_lynx_id_in_sumava)

    df = dataset_tools.extend_df_with_sequence_id(df, time_limit="120s")

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


def data_preprocessing(zip_path: Path, media_dir_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    duplicates: DataFrame - List of duplicit files
    """
    temp_dir = Path("./temp")
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(exist_ok=True, parents=True)

    if zip_path.suffix.lower() in (".tar", ".tar.gz"):
        extract_tarfile(zip_path, temp_dir)
    elif zip_path.suffix.lower() in (".zip"):
        extract_zipfile(zip_path, temp_dir)

    df, duplicates = analyze_dataset_directory(temp_dir)
    # df["image_path"] = \
    # df["vanilla_path"].map(lambda fn: dataset_tools.make_hash(fn, prefix="media_data"))
    df = dataset_tools.make_dataset(
        dataframe=df,
        dataset_name=None,
        dataset_base_dir=temp_dir,
        output_path=media_dir_path,
        hash_filename=True,
        make_tar=False,
        copy_files=True,
    )

    return df, duplicates


def get_prediction_parameters():
    """Prepare config, model weights, names of classes and device."""
    # Download Artifact v2
    api = wandb.Api()
    resources_path = Path("./resources")

    run = api.run("zcu_cv/CarnivoreID-Classification/runs/wucd6qgr")

    pth_files = list(resources_path.glob(f'{run.config["run_name"]}*.pth'))
    if len(pth_files) > 0:
        weights_path = pth_files[0]
    else:
        run.logged_artifacts()[0].download(resources_path)
        pth_files = list(resources_path.glob(f'{run.config["run_name"]}*.pth'))
        weights_path = pth_files[0]

    config = run.config

    classid_category_map = {i: str(i) for i in range(config["number_of_classes"])}

    device = "gpu" if torch.cuda.is_available() else "cpu"

    return config, weights_path, classid_category_map, device


def prediction(metadata: pd.DataFrame):
    """Do the prediction of files listed in dataframe."""
    from fgvc.core.training import predict
    from fgvc.datasets import get_dataloaders

    config, weights_path, classid_category_map, device = get_prediction_parameters()

    model, model_mean, model_std = load_model(config, weights_path)

    logger.info("Creating DataLoaders.")
    _, testloader, _, _ = get_dataloaders(
        None,
        metadata,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        # architecture=config["architecture"]
    )
    logits, targs, _, scores = predict(model, testloader, device=device)
    return logits, targs, scores


def data_processing(zip_path: Path, media_dir_path: Path, csv_path: Path):
    """Preprocessing and prediction on data in ZIP file.

    Files are renamed according to the hash based on input path.
    Parameters
    ----------
    zip_path: path to input zip file.
    media_dir_path: Path to content of zip. The file names are hashed.
    csv_path: Path to output CSV file.
    """
    metadata, _ = data_preprocessing(zip_path, media_dir_path)

    metadata = metadata[metadata.media_type == "image"].reset_index(drop=True)

    if "class_id" not in metadata:
        # TODO create loader with class_id not required
        logger.warning("Create less restrictive loader")
        metadata["class_id"] = 0

    metadata["image_path"] = metadata["image_path"].apply(
        lambda path: str(Path(media_dir_path) / path)
        # path.split("/", 5)[-1]
    )
    metadata.class_id = metadata.class_id.astype(int)
    logits, targs, scores = prediction(metadata)

    probs = softmax(logits, 1)
    targs = metadata["class_id"]
    preds = np.argmax(probs, 1)
    targs_probs = probs[np.arange(len(probs)), targs]
    preds_probs = probs[np.arange(len(probs)), preds]

    # metadata["class_id"]
    metadata["targ_prob"] = targs_probs
    metadata["pred"] = preds
    metadata["pred_prob"] = preds_probs

    metadata.to_csv(csv_path)
