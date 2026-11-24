import hashlib
import json
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import requests.exceptions
import skimage.io
import wandb
import yaml
from PIL import Image
from scipy.special import softmax
from tqdm import tqdm
from wildlife_model_cache import WEIGHTS_DIR, log_model_operation

from .config import WANDB_API_KEY, WANDB_ARTIFACT_PATH, WANDB_ARTIFACT_PATH_CROPPED
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


def get_model_config(is_cropped: bool = False) -> Tuple[dict, Path, dict]:
    """Load the W&B artifact and its configuration from a complete cached revision."""
    artifact_ref = WANDB_ARTIFACT_PATH_CROPPED if is_cropped else WANDB_ARTIFACT_PATH
    ref_key = hashlib.sha256(artifact_ref.encode()).hexdigest()
    cache_dir = WEIGHTS_DIR / "wandb" / ref_key
    try:
        with log_model_operation("W&B artifact lookup", artifact_ref):
            artifact = wandb.Api(api_key=WANDB_API_KEY).artifact(artifact_ref)
        digest = hashlib.sha256(artifact.digest.encode()).hexdigest()
        model_dir = cache_dir / digest
        if not model_dir.is_dir():
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache_dir) as staging:
                download_dir = Path(staging) / "artifact"
                with log_model_operation("W&B artifact download", artifact_ref):
                    artifact.download(root=str(download_dir))
                with (download_dir / "config.yaml").open() as config_file:
                    artifact_config = yaml.safe_load(config_file)
                try:
                    with log_model_operation("W&B run configuration lookup", artifact_ref):
                        config = artifact.logged_by().config
                except Exception as exc:
                    logger.warning("Failed to get run config from W&B artifact: %s", exc)
                    config = {
                        "number_of_classes": len(artifact_config["id2label"]),
                        "architecture": "swin_small_patch4_window7_224",
                        "augmentations": "vit_heavy",
                        "image_size": [224, 224],
                        "batch_size": 64,
                    }
                checkpoints = [entry.name for entry in artifact.files() if Path(entry.name).suffix.lower() == ".pth"]
                if len(checkpoints) != 1:
                    raise ValueError("Expected exactly one .pth file in the W&B artifact")
                if not (download_dir / checkpoints[0]).is_file():
                    raise FileNotFoundError(download_dir / checkpoints[0])
                with (download_dir / "model_config.json").open("w") as config_file:
                    json.dump({"config": config, "checkpoint": checkpoints[0]}, config_file)
                try:
                    download_dir.rename(model_dir)
                except OSError:
                    if not model_dir.is_dir():
                        raise
    except (wandb.CommError, requests.exceptions.ConnectionError):
        cached = [path.parent for path in cache_dir.glob("*/model_config.json")]
        if not cached:
            raise
        model_dir = max(cached, key=lambda path: (path / "model_config.json").stat().st_mtime)
        logger.warning("W&B is unavailable. Using a cached model for %s.", artifact_ref)

    with (model_dir / "model_config.json").open() as config_file:
        model_config = json.load(config_file)
    with (model_dir / "config.yaml").open() as config_file:
        artifact_config = yaml.safe_load(config_file)
    return model_config["config"], model_dir / model_config["checkpoint"], artifact_config


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

    logger.info("Classification stage: creating dataloaders for %s images.", len(image_paths))
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

    logger.info("Classification stage: running inference for %s images.", len(image_paths))
    predict_output = predict(model, testloader)
    logger.info("Classification stage: inference done.")
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
        try:
            config, checkpoint_path, artifact_config = get_model_config(
                # is_cropped=False
            )
        except Exception as e:
            logger.error("Error loading model configuration from W&B.")
            logger.debug(f"{WANDB_API_KEY[:4]}...")  # log only part of the API key
            logger.debug("W&B model cache: %s", WEIGHTS_DIR / "wandb")
            raise e

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
    image_path = metadata["full_image_path"]
    logger.info("Classification stage: starting for %s media files.", len(metadata))
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
    logger.info("Classification stage: finished for %s media files.", len(metadata))


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
    """Split readable media from failed rows, including missing converted files."""
    metadata = metadata.copy()
    supported_media = metadata["media_type"].isin(("image", "video"))
    metadata.loc[~supported_media, "read_error"] = "Unsupported media type"

    if "read_error" not in metadata:
        metadata["read_error"] = ""
    metadata["read_error"] = metadata["read_error"].fillna("").astype(str)

    if "full_image_path" in metadata:
        missing_file = ~metadata["full_image_path"].apply(lambda path: Path(path).is_file())
        metadata.loc[missing_file & (metadata["read_error"] == ""), "read_error"] = metadata.loc[
            missing_file & (metadata["read_error"] == ""), "full_image_path"
        ].apply(lambda path: f"Prepared media file does not exist: {path}")

    failing = metadata[metadata["read_error"] != ""].copy().reset_index(drop=True)
    correct = metadata[metadata["read_error"] == ""].copy().reset_index(drop=True)
    return correct, failing


# new function


def _variant_rel_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _media_variant_paths(row, output_dir: Path) -> dict:
    mediafile_path = Path(row["absolute_media_path"])
    mediafile_name = mediafile_path.name

    preview_abs_pth = output_dir / "previews" / mediafile_name
    if row["media_type"] == "video":
        preview_abs_pth = preview_abs_pth.with_suffix(".mp4")
    else:
        preview_abs_pth = preview_abs_pth.with_suffix(".webp")

    thumbnail_abs_pth = (output_dir / "thumbnails" / mediafile_name).with_suffix(".webp")
    static_thumbnail_abs_pth = (output_dir / "static_thumbnails" / mediafile_name).with_suffix(".webp")
    return {
        "preview_path": preview_abs_pth,
        "thumbnail_path": thumbnail_abs_pth,
        "static_thumbnail_path": static_thumbnail_abs_pth,
    }


def _optional_path_from_row(row, column_name: str, fallback: Path) -> Path:
    value = row.get(column_name)
    if value is None or pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
        return fallback
    return Path(value)


def make_gif_from_video_file(video_path: Path, thumbnail_path: Path, width: int = 400, num_frames: int = 30) -> bool:
    """Create an animated thumbnail from a video file."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            logger.warning("Cannot create animated thumbnail from video without frames: %s", video_path)
            return False

        frames = []
        for i in range(num_frames):
            frame_idx = int(i * frame_count / num_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            scale = float(width) / frame.shape[1]
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not frames:
            logger.warning("Cannot create animated thumbnail from unreadable video: %s", video_path)
            return False

        thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
        frame_one = Image.fromarray(frames[0])
        remaining_frames = [Image.fromarray(frame) for frame in frames[1:]]
        save_kwargs = {
            "append_images": remaining_frames,
            "save_all": True,
            "duration": 100,
            "loop": 0,
        }
        if thumbnail_path.suffix.lower() == ".webp":
            save_kwargs.update({"format": "WEBP", "quality": 85, "method": 5})
        frame_one.save(thumbnail_path, **save_kwargs)
        return True
    except Exception:
        logger.warning(
            "Cannot create animated thumbnail from video '%s'. Exception: %s",
            video_path,
            traceback.format_exc(),
        )
        return False


def make_previews(
    metadata,
    output_dir,
    preview_width=1200,
    thumbnail_width=400,
    force: bool = False,
    progress_callback=None,
):
    """Create preview, animated thumbnail and static thumbnail files for each media file."""
    output_dir = Path(output_dir)
    logger.info("Preview stage: creating media variants for %s media rows.", len(metadata))
    generated_by_media_path = {}
    for position, (_, row) in enumerate(tqdm(metadata.iterrows(), total=len(metadata), desc="Creating previews")):
        mediafile_path = Path(row["absolute_media_path"])
        media_key = str(mediafile_path)
        if media_key not in generated_by_media_path:
            variant_paths = _media_variant_paths(row, output_dir)
            preview_abs_pth = variant_paths["preview_path"]
            thumbnail_abs_pth = variant_paths["thumbnail_path"]
            static_thumbnail_abs_pth = variant_paths["static_thumbnail_path"]
            static_source_path = _optional_path_from_row(row, "full_image_path", mediafile_path)

            if row["media_type"] == "image":
                if force or not preview_abs_pth.exists():
                    make_thumbnail_from_file(mediafile_path, preview_abs_pth, width=preview_width)
                if force or not thumbnail_abs_pth.exists():
                    make_thumbnail_from_file(mediafile_path, thumbnail_abs_pth, width=thumbnail_width)
            elif row["media_type"] == "video":
                if force or not preview_abs_pth.exists():
                    convert_to_mp4(mediafile_path, preview_abs_pth, force=force)
                if force or not thumbnail_abs_pth.exists():
                    make_gif_from_video_file(mediafile_path, thumbnail_abs_pth, width=thumbnail_width)

            if force or not static_thumbnail_abs_pth.exists():
                make_thumbnail_from_file(static_source_path, static_thumbnail_abs_pth, width=thumbnail_width)

            generated_by_media_path[media_key] = {
                column_name: _variant_rel_path(path, output_dir)
                for column_name, path in variant_paths.items()
                if path.exists()
            }

        for column_name, rel_path in generated_by_media_path[media_key].items():
            metadata.loc[row.name, column_name] = rel_path
        if progress_callback is not None:
            progress_callback(position + 1, len(metadata))

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
