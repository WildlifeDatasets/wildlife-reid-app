import logging
import os.path
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("app")

DEVICE = "0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")


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
        output_file.parent.mkdir(parents=True, exist_ok=True)
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


model_url = r"https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
model_file = Path("~/resources/md_v5a.0.0.pt")
download_file_if_does_not_exists(model_url, model_file)

DETECTION_MODEL = None


def get_detection_model():
    global DETECTION_MODEL
    if DETECTION_MODEL is None:
        DETECTION_MODEL = torch.hub.load(
            "ultralytics/yolov5",  # repo_or_dir
            "custom",  # model
            str(Path("~/resources/md_v5a.0.0.pt").expanduser()),  # args for callable model
            force_reload=True,
            device=DEVICE,
        )
        return DETECTION_MODEL
    else:
        return DETECTION_MODEL


def release_detection_model():
    global DETECTION_MODEL
    DETECTION_MODEL = None


# logger.info("Initializing SAM model and loading pre-trained checkpoint.")
# _checkpoint_path = Path("~/resources/sam_vit_b_01ec64.pth").expanduser()
# SAM = sam_model_registry["vit_b"](checkpoint=str(_checkpoint_path))
# SAM.to(device=DEVICE)
# SAM_PREDICTOR = SamPredictor(SAM)
# SAM = None


def detect_animal(image_path: list) -> dict[str, Union[np.ndarray, Any]]:
    """Detect an animal in a given image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    logger.info("Running detection inference.")
    model = get_detection_model()
    results = model(image)
    id2label = results.names
    results = results.xyxy[0].cpu().numpy()
    if len(results) == 0:
        return {
            "bbox": np.array([0, 0, 0, 0]),
            "confidence": 0,
            "class": "nothing",
        }
    else:
        return {
            "bbox": np.array(list(int(_) for _ in results[0][:4].tolist())),
            "confidence": results[0][4],
            "class": id2label[results[0][5]],
        }


def detect_animals(image_paths: list[Path]) -> list[bool]:
    """Detect animals in a list of images."""

    detected_animals = [False] * len(image_paths)
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detect_animal(image)
        if result["class"] == "animal":
            cropped_animal = pad_image(image, result["bbox"], border=0.25)
            base_path = Path(image_path).parent.parent / "cropped_images"
            save_path = base_path / Path(image_path).name
            base_path.mkdir(exist_ok=True, parents=True)
            Image.fromarray(cropped_animal).convert("RGB").save(save_path)
            detected_animals[i] = True
