import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# from fgvc.taxon_utils.taxon_utils import set_cuda_device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("app")
logger.info(f"Using device: {DEVICE}")

logger.info("Initializing MegaDetector model and loading pre-trained checkpoint.")

MEDIA_DIR = Path("/shared_data/media")
DETECTION_MODEL = None


def download_file(url: str, output_file: str):
    """Download file from url."""
    import requests

    # r = requests.get(url, allow_redirects=True)
    # with open(output_file, "wb") as f:
    #     f.write(r.content)
    # download file from url with tqdm progressbar
    # https://stackoverflow.com/a/37573701/4419811
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(output_file, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        logger.error("ERROR, something went wrong")


def download_file_if_does_not_exists(url: str, output_file: str):
    """Download file from url."""
    logger.debug("Checking if file does not exists.")
    if not os.path.exists(output_file):
        logger.debug("File does not exists. Downloading.")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        download_file(url, output_file)


def pad_image(image: np.ndarray, bbox: Union[list, np.ndarray], border: float = 0.25) -> np.ndarray:
    """Crop the image, pad to square and add a border."""
    # get bbox and image
    x0, y0, x1, y1 = np.round(bbox).astype(int)
    w, h = x1 - x0, y1 - y0
    cropped_image = image[y0:y1, x0:x1]

    # add padding
    dif = np.abs(w - h)
    pad_value_0 = np.floor(dif / 2).astype(int)
    pad_value_1 = dif - pad_value_0
    pad_w = 0
    pad_h = 0

    if w > h:
        y0 -= pad_value_0
        y1 += pad_value_1
        pad_h += pad_value_0
    else:
        x0 -= pad_value_0
        x1 += pad_value_1
        pad_w += pad_value_0

    border = np.round((np.max([w, h]) * (border / 2)) / 2).astype(int)
    pad_w += border
    pad_h += border

    padded_image = np.pad(cropped_image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    return padded_image


def get_detection_model(force_reload: bool = False):
    """Load the detection model if not loaded before."""
    global DETECTION_MODEL
    if DETECTION_MODEL is None:
        model_url = r"https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
        model_file = Path("/root/resources/md_v5a.0.0.pt")
        download_file_if_does_not_exists(model_url, model_file)

        logger.debug(f"Loading model from file: {model_file}. {model_file.exists()=}")

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        DETECTION_MODEL = torch.hub.load(
            "ultralytics/yolov5",  # repo_or_dir
            "custom",  # model
            str(model_file.expanduser()),  # args for callable model
            # force_reload=True,
            force_reload=force_reload,
            device=DEVICE,
        )
    return DETECTION_MODEL


def del_detection_model():
    """Release the detection model."""
    global DETECTION_MODEL
    DETECTION_MODEL = None

DETECTION_MODEL = get_detection_model(force_reload=True)
del_detection_model()

def detect_animals_in_one_image(image_rgb: np.ndarray) -> Optional[List[Dict[str, Any]]]:
    """Detect an animal in a given image.

    Expected classes are: {0: 'animal', 1: 'person', 2: 'vehicle'}
    """
    detection_model = get_detection_model()
    results = detection_model(image_rgb)
    id2label = results.names

    batch_idx = 0
    results = results.xyxy[batch_idx].cpu().numpy()

    if len(results) == 0:
        return None

    # results_list = [None] * len(results)
    # for i in range(len(results)):
    results_list = [
        {
            "bbox": list(int(_) for _ in results[i][:4].tolist()),
            "confidence": results[i][4],
            "class": id2label[results[i][5]],
            "size": image_rgb.shape[:2],
        }
        for i in range(len(results))
    ]
    # results_list[i] = {
    #     "bbox": list(int(_) for _ in results[i][:4].tolist()),
    #     "confidence": results[i][4],
    #     "class": id2label[results[i][5]],
    #     "size": image_rgb.shape[:2]
    # }

    return results_list


def detect_animal_on_metadata(metadata: pd.DataFrame, border=0.0) -> pd.DataFrame:
    """Do the detection and segmentation on images in metadata.

    Returns:
        pd.DataFrame: metadata with added detection results.
    """
    assert "full_image_path" in metadata
    logger.info("Running detection inference.")
    for row_idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_abs_path = row["full_image_path"]
        try:
            if (
                row["media_type"] == "video"
                and row["full_image_path"] == row["absolute_media_path"]
            ):
                # there are no detected animals in video
                continue

            image = cv2.imread(str(image_abs_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detect_animals_in_one_image(image_rgb=image)

            if results is None:
                # there are no detected animals in image
                logger.debug(f"No detection in image: {image_abs_path}")
                row["detection_results"] = []
                metadata.loc[row_idx] = row
                continue

            row["detection_results"] = results
            for ii, result in enumerate(results):
                # if result["class"] == "animal":
                base_path = Path(image_abs_path).parent.parent / "detection_images"
                save_path = base_path / (
                    Path(image_abs_path).stem + f".{ii}" + Path(image_abs_path).suffix
                )
                base_path.mkdir(exist_ok=True, parents=True)

                padded_image = pad_image(image, result["bbox"], border=border)
                Image.fromarray(padded_image).convert("RGB").save(save_path)
                if ii == 0:
                    # if there is at least one detection save the very first one as the main image
                    save_path = base_path / (Path(image_abs_path).name)
                    Image.fromarray(padded_image).convert("RGB").save(save_path)

            metadata.loc[row_idx] = row
        except Exception:
            logger.warning(
                f"Cannot process image '{image_abs_path}'. Exception: {traceback.format_exc()}"
            )
    del_detection_model()
    return metadata
