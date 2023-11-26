import logging
from typing import Any, Union

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from numpy import ndarray
from segment_anything import SamPredictor, sam_model_registry

# from fgvc.utils.utils import set_cuda_device

logger = logging.getLogger("app")
# device = set_cuda_device("0" if torch.cuda.is_available() else "cpu")
device = "0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

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
    if not os.path.exists(output_file):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, output_file)


model_url = r"https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
model_file = Path("resources/md_v5a.0.0.pt")
download_file_if_does_not_exists(model_url, model_file)

DETECTION_MODEL = torch.hub.load(
    "ultralytics/yolov5",  # repo_or_dir
    "custom",  # model
    "/detection_worker/resources/md_v5a.0.0.pt",  # args for callable model
    force_reload=True,
    device=device,
)

logger.info("Initializing SAM model and loading pre-trained checkpoint.")
_checkpoint = "/detection_worker/resources/sam_vit_h_4b8939.pth"
SAM = sam_model_registry["vit_h"](checkpoint=_checkpoint)
SAM.to(device=0)
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


def segment_animal(image_path: list, bbox: list, cropped=True) -> np.ndarray:
    """Segment an animal in a given image using SAM model."""

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    SAM_PREDICTOR.set_image(image)
    sam_input_box = np.array([int(point) for point in bbox])

    masks, _, _ = SAM_PREDICTOR.predict(
        point_coords=None,
        point_labels=None,
        box=sam_input_box[None, :],
        multimask_output=False,
    )

    foregroud_image = image.copy()
    foregroud_image[masks[0] == False] = 0

    return foregroud_image[int(bbox[1]) - 5: int(bbox[3]) + 5, int(bbox[0]) - 5: int(bbox[2]) + 5]
