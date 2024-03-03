import logging
import os
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch
from numpy import ndarray
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

# from fgvc.utils.utils import set_cuda_device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("app")
logger.info(f"Using device: {DEVICE}")

logger.info("Initializing MegaDetector model and loading pre-trained checkpoint.")


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


model_url = r"https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
model_file = Path("/detection_worker/resources/md_v5a.0.0.pt")
download_file_if_does_not_exists(model_url, model_file)

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
DETECTION_MODEL = torch.hub.load(
    "ultralytics/yolov5",  # repo_or_dir
    "custom",  # model
    str(Path("/detection_worker/resources/md_v5a.0.0.pt").expanduser()),  # args for callable model
    force_reload=True,
    device="cpu",
)


download_file_if_does_not_exists(
    "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/sam_vit_h_4b8939.pth",
    "/detection_worker/resources/sam_vit_h_4b8939.pth",
)

logger.info("Initializing SAM model and loading pre-trained checkpoint.")
_checkpoint_path = Path("/detection_worker/resources/sam_vit_h_4b8939.pth").expanduser()
SAM = sam_model_registry["vit_h"](checkpoint=str(_checkpoint_path))
SAM.to(device=DEVICE)
SAM_PREDICTOR = SamPredictor(SAM)
# SAM = None


def detect_animal(image_path: list) -> dict[str, Union[ndarray, Any]]:
    """Detect an animal in a given image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    logger.info("Running detection inference.")
    results = DETECTION_MODEL(image)
    id2label = results.names
    results = results.xyxy[0].cpu().numpy()

    return {
        "bbox": np.array(list(int(_) for _ in results[0][:4].tolist())),
        "confidence": results[0][4],
        "class": id2label[results[0][5]],
    }


def segment_animal(image_path: str, bbox: list, cropped=True) -> np.ndarray:
    """Segment an animal in a given image using SAM model."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    logger.debug("Running segmentation inference.")
    SAM_PREDICTOR.set_image(image)
    sam_input_box = np.array([int(point) for point in bbox])

    logger.debug(f"{sam_input_box=}")

    masks, _, _ = SAM_PREDICTOR.predict(
        point_coords=None,
        point_labels=None,
        box=sam_input_box[None, :],
        multimask_output=False,
    )
    logger.debug(f"{masks=}")

    foregroud_image = image.copy()
    foregroud_image[masks[0] == False] = 0  # noqa

    return pad_image(foregroud_image, bbox, border=0.25)
