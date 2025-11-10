import json
import logging
import shutil
import traceback
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import skimage.io
import wandb
import yaml
from PIL import Image
from scipy.special import softmax
from tqdm import tqdm

from .config import RESOURCES_DIR, WANDB_API_KEY, WANDB_ARTIFACT_PATH, WANDB_ARTIFACT_PATH_CROPPED
from .dataset_tools import data_preprocessing

# from fgvc.core.training import predict
# from fgvc.datasets import get_dataloaders
# from fgvc.utils.experiment import load_model
from .fgvc_core_training_subset import predict
from .fgvc_datasets_subset import get_dataloaders
from .fgvc_utils_experiment_subset import load_model
from .prediction_dataset import PredictionDataset

# from taxon_worker.infrastructure_utils import mem
try:
    from ..infrastructure_utils import log_tools, mem
except ImportError:
    from infrastructure_utils import log_tools, mem


logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)
MEDIA_DIR_PATH = Path("/shared_data/media")
TAXON_CLASSIFICATION_MODEL_DICT = None


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
            logger.debug(f"New model={WANDB_ARTIFACT_PATH}. " + f"Old model={model_meta.get('artifact_path', 'None')}.")
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


def load_model_and_predict_and_add_not_classified(
    image_paths: list,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """Load model, create dataloaders, and run inference.

    Function returns thresholded predictions and raw predictions
    """
    # from .data_preprocessing import detect_animal, pad_image, detect_animals
    # is_detected = detect_animals(image_paths)

    taxon_classification_model_dict = get_taxon_classification_model()
    artifact_config = taxon_classification_model_dict["artifact_config"]
    config = taxon_classification_model_dict["config"]
    model = taxon_classification_model_dict["model"]
    model_mean = taxon_classification_model_dict["model_mean"]
    model_std = taxon_classification_model_dict["model_std"]

    # artifact_config, config, model, model_mean, model_std

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

    logger.info("Running inference. Taxon classification.")
    predict_output = predict(model, testloader)
    logger.info("Inference done.")
    release_taxon_classification_model()
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
    # # print all items in id2label
    # logger.debug("id2label")
    # for k, v in id2label.items():
    #     logger.debug(f"{k=}, {v=}")

    assert np.max(list(id2label.keys())) == (len(id2label) - 1), "Some of the labels is missing in id2label."

    # Get values with no thresholding
    class_ids_raw, probs_top_raw = get_top_predictions(probs)

    # add inference results to the metadata dataframe
    if do_confidence_thresholding:
        class_ids, probs_top = do_thresholding_on_probs(probs, id2thresholds)
        # extend labels and add there new label "Not Classified"
        id2label[len(id2label)] = "Not Classified"
    else:
        class_ids = np.argmax(probs, 1)
        probs_top = np.max(probs, 1)

    return class_ids, probs_top, id2label, class_ids_raw, probs_top_raw


def get_taxon_classification_model():
    """Load model and return the model and its configuration."""
    logger.debug(f"Before taxon classification model: {mem.get_vram()}     {mem.get_ram()}")
    global TAXON_CLASSIFICATION_MODEL_DICT

    if TAXON_CLASSIFICATION_MODEL_DICT is None:
        mem.wait_for_gpu_memory(0.5)
        config, checkpoint_path, artifact_config = get_model_config(
            # is_cropped=False
        )
        logger.info("Creating model and loading fine-tuned checkpoint.")

        model, model_mean, model_std = load_model(config, checkpoint_path)
        TAXON_CLASSIFICATION_MODEL_DICT = {
            "model": model,
            "model_mean": model_mean,
            "model_std": model_std,
            # "checkpoint_path": checkpoint_path,
            "artifact_config": artifact_config,
            "config": config,
        }
        logger.debug(
            f"model_mean={model_mean}, model_std={model_std}, " + f"checkpoint_path={checkpoint_path}, config={config}"
        )
    logger.debug(f"After taxon classification model load: {mem.get_vram()}     {mem.get_ram()}")
    return TAXON_CLASSIFICATION_MODEL_DICT


def release_taxon_classification_model():
    """Release the taxon classification model."""
    global TAXON_CLASSIFICATION_MODEL_DICT
    TAXON_CLASSIFICATION_MODEL_DICT = None


def get_top_predictions(probs: np.array) -> Tuple[np.array, np.array]:
    """Get the top predictions from the softmaxed logits."""
    top_probs = np.max(probs, 1)
    class_ids = np.argmax(probs, 1)
    return class_ids, top_probs


def do_thresholding_on_probs(probs: np.array, id2threshold: dict) -> Tuple[np.array, np.array]:
    """Use the thresholds to do the classification and add class "Not Classified".

    The images with all softmax values under the classification threshold are marked
    as not-classified.

    Args:
        probs (numpy.array): softmax of logits with shape = [n_samples, n_classes].
        id2threshold (dict): Thresholds in dictionary. Key is the class id, value is the threshold.

    Returns:
        class_ids: Ids of classes. The highest class id represent the not classified category.
        top_probs: Softmaxed probability. The not-classified is calculated as 1 - top_prob.

    """
    assert probs.shape[1] == len(id2threshold), "There should be the same number of columns as the number of classes."

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
    sequence_time_limit_s: Optional[int] = None,
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
    if sequence_time_limit_s is None:
        sequence_time_limit_s = 120
    logger.debug(f"{media_dir_path=}, {zip_path=}, {csv_path=}")
    # create metadata dataframe
    metadata, _ = data_preprocessing(
        zip_path,
        media_dir_path,
        num_cores=num_cores,
        contains_identities=contains_identities,
        sequence_time_limit_s=sequence_time_limit_s,
    )
    metadata, df_failing = keep_correctly_loaded_images(metadata)
    df_failing.to_csv(csv_path.with_suffix(".failed.csv"), encoding="utf-8-sig")

    run_taxon_classification_inference(metadata)

    # move_files_from_temp(media_dir_path, metadata)

    # save metadata file
    metadata.to_csv(csv_path, encoding="utf-8-sig")
    return metadata


def run_taxon_classification_inference(metadata):
    """Use full_image_path for taxon prediction."""
    # run inference
    # image_path = metadata["image_path"].apply(lambda x: os.path.join(MEDIA_DIR_PATH, x))
    image_path = metadata["full_image_path"]
    logger.debug(f"image path    {image_path=}")
    (
        class_ids,
        probs_top,
        id2label,
        class_ids_raw,
        probs_top_raw,
    ) = load_model_and_predict_and_add_not_classified(image_path)

    # Add class Animalia
    id2label[len(id2label)] = "Animalia"

    use_detector_class_if_classification_fails(id2label, metadata, class_ids, probs_top)

    # add inference results to the metadata dataframe
    metadata["predicted_class_id"] = class_ids
    metadata["predicted_prob"] = probs_top
    metadata["predicted_class_id_raw"] = class_ids_raw
    metadata["predicted_prob_raw"] = probs_top_raw
    metadata["predicted_category"] = np.nan
    metadata["predicted_category_raw"] = np.nan
    if id2label is not None:
        metadata["predicted_category"] = metadata["predicted_class_id"].apply(lambda x: id2label.get(x, np.nan))
        metadata["predicted_category_raw"] = metadata["predicted_class_id_raw"].apply(lambda x: id2label.get(x, np.nan))


def use_detector_class_if_classification_fails(
    id2label: dict, metadata: pd.DataFrame, class_ids: np.ndarray, probs_top: np.ndarray
) -> Tuple[pd.DataFrame, np.array, np.array]:
    """Use the detection results if the classification fails.

    If the record is Not Classified then we use the output class of the detector.
    """
    # invert dict id2label
    label2id = {v: k for k, v in id2label.items()}
    id_not_classified = label2id["Not Classified"]
    for i, predicted_class_id in enumerate(class_ids):
        if predicted_class_id == id_not_classified:
            detection_results = metadata["detection_results"][i]
            if (detection_results is not None) and (len(detection_results) > 0):
                detection_result = detection_results[0]
                # logger.debug(f"{detection_result['class']=}")
                # logger.debug(f"{detection_result['confidence']=}")
                if detection_result["class"] == "person":
                    class_ids[i] = label2id["Homo sapiens"]
                    probs_top[i] = detection_result["confidence"]

                if detection_result["class"] == "vehicle":
                    class_ids[i] = label2id["Vehicle"]
                    probs_top[i] = detection_result["confidence"]

                if detection_result["class"] == "animal":
                    class_ids[i] = label2id["Animalia"]
                    probs_top[i] = detection_result["confidence"]

    return metadata, class_ids, probs_top


def keep_correctly_loaded_images(metadata) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove file from list if there is the error message."""
    # metadata = metadata[metadata["media_type"] == "image"].reset_index(drop=True)
    # keep media_type==image and media_type == video
    metadata = metadata[(metadata["media_type"] == "image") | (metadata["media_type"] == "video")].reset_index(
        drop=True
    )
    # drop media_type== "unknown"
    # mediadata = metadata[metadata["media_type"] != "unknown"].reset_index(drop=True)

    df_failing = metadata[metadata["read_error"] == ""].copy().reset_index(drop=True)
    metadata = metadata[metadata["read_error"] == ""].reset_index(drop=True)
    return metadata, df_failing


# new function


# TODO make preview on taxon worker
def make_previews(metadata, output_dir, preview_width=1200, force: bool = False):
    """Create preview image for video."""
    output_dir = Path(output_dir)
    for i, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Creating previews"):
        mediafile_path = Path(row["absolute_media_path"])
        # output_dir = Path(settings.MEDIA_ROOT) / mediafile.parent.outputdir
        # abs_pth = output_dir / "thumbnails" / Path(mediafile.mediafile.name).name
        preview_abs_pth = output_dir / "previews" / Path(mediafile_path).name

        if row["media_type"] == "image":
            # preview_rel_pth = os.path.relpath(preview_abs_pth, settings.MEDIA_ROOT)
            # logger.debug(f"Creating preview for {mediafile_path}")
            make_thumbnail_from_file(mediafile_path, preview_abs_pth, width=preview_width)
        elif row["media_type"] == "video":
            # logger.debug(f"Creating preview for {mediafile_path}")
            convert_to_mp4(mediafile_path, preview_abs_pth, force=force)

    return metadata


class TempLogContext:
    def __init__(self, logger_names: list, levels: list):

        self.logger_names = logger_names
        self.levels = levels
        self.loggers = [logging.getLogger(logger_name) for logger_name in logger_names]

    def __enter__(self):
        self.old_levels = [logger.level for logger in self.loggers]
        for logger, level in zip(self.loggers, self.levels):
            logger.setLevel(level)

    def __exit__(self, exc_type, exc_value, traceback):
        for logger, level in zip(self.loggers, self.old_levels):
            logger.setLevel(level)


def make_thumbnail_from_file(image_path: Path, thumbnail_path: Path, width: int = 800) -> bool:
    """Create a smaller thumbnail image from the input image.

    Returns:
        True if the processing succeeded, False otherwise.
    """
    try:
        with log_tools.TempLogContext(
            ["skimage.io", "PIL", "tifffile"], [logging.WARNING, logging.WARNING, logging.WARNING]
        ):
            image = skimage.io.imread(image_path)

        if image is None:
            raise ValueError("Image could not be read.")

        # Rescale
        scale = float(width) / image.shape[1]
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image_rescaled = cv2.resize(image, new_size)

        # Convert to PIL Image for saving (better format support)
        if image_rescaled.dtype != np.uint8:
            image_rescaled = (image_rescaled * 255).astype(np.uint8)
        if image_rescaled.ndim == 2:  # grayscale
            pil_image = Image.fromarray(image_rescaled, mode="L")
        else:
            pil_image = Image.fromarray(image_rescaled)

        thumbnail_path.parent.mkdir(exist_ok=True, parents=True)

        # Choose quality
        suffix = thumbnail_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            quality = 85
            save_kwargs = {"quality": quality, "optimize": True}
        elif suffix == ".webp":
            save_kwargs = {"quality": 85, "method": 5}
        else:
            save_kwargs = {}

        pil_image.save(thumbnail_path, **save_kwargs)
        return True

    except Exception:
        logger.warning(f"Cannot create thumbnail from file '{image_path}'. Exception: {traceback.format_exc()}")
        return False


# def make_thumbnail_from_file(image_path: Path, thumbnail_path: Path, width: int = 800) -> bool:
#     """Create small thumbnail image from input image.
#
#     Returns:
#         True if the processing is ok.
#
#     """
#     try:
#         with log_tools.TempLogContext(
#             ["skimage.io", "PIL", "tifffile"], [logging.WARNING, logging.WARNING, logging.WARNING]
#         ):
#             image = skimage.io.imread(image_path)
#         scale = float(width) / image.shape[1]
#         scale = [scale, scale, 1]
#         image_rescaled = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1])
#         # image_rescaled = skimage.transform.rescale(image, scale=scale, anti_aliasing=True)
#         # image_rescaled = (image_rescaled * 255).astype(np.uint8)
#         # logger.info(f"{image_rescaled.shape=}, {image_rescaled.dtype=}")
#         thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
#         if thumbnail_path.suffix.lower() in (".jpg", ".jpeg"):
#             quality = 85
#         else:
#             quality = None
#         skimage.io.imsave(thumbnail_path, image_rescaled, quality=quality)
#         return True
#     except Exception:
#         logger.warning(f"Cannot create thumbnail from file '{image_path}'. Exception: {traceback.format_exc()}")
#         return False


def convert_to_mp4(input_video_path: Union[str, Path], output_video_path: Union[str, Path], force: bool = False):
    """Convert video to MP4 format using H.264 video codec and AAC audio codec."""
    import os.path
    import subprocess

    input_video_path = str(input_video_path)
    output_video_path = str(output_video_path)

    # Check if input file exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"The input file '{input_video_path}' does not exist.")
    if os.path.exists(output_video_path):
        logger.warning(f"The output file '{output_video_path}' already exists. Force overwrite={force}.")
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg command to convert video to MP4 (H.264 + AAC)
    command = [
        "ffmpeg",
        # "-y",  # Overwrite output file if it exists
        "-i",
        str(input_video_path),  # Input video file
        "-c:v",
        "libx264",  # Set the video codec to H.264
        "-c:a",
        "aac",  # Set the audio codec to AAC
        "-b:a",
        "192k",  # Audio bitrate (you can adjust this)
        "-strict",
        "experimental",  # For using AAC
        str(output_video_path),  # Output video file
    ]
    if force:
        # Overwrite output file if it exists
        command.insert(1, "-y")

    try:
        # Run the ffmpeg command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # check=True
        )
        if result.returncode != 0:
            logger.error(f"Error during conversion: {result.stderr}\nCommand: {' '.join(command)}")
            raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
        # logger.debug(f"Conversion successful! Output saved at '{output_video_path}'")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during conversion of {str(input_video_path)} to {str(output_video_path)}: {e}")
        logger.debug(traceback.format_exc())
