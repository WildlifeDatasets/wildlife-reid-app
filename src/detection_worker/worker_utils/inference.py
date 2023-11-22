import logging
from typing import Dict, Tuple, Union, Any
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray

from segment_anything import sam_model_registry, SamPredictor

from fgvc.core.models import get_model
from fgvc.core.training import predict
from fgvc.datasets import PredictionDataset, get_dataloaders
from fgvc.utils.utils import set_cuda_device


logger = logging.getLogger("app")
device = set_cuda_device("0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

logger.info("Initializing MegaDetector model and loading pre-trained checkpoint.")
DETECTION_MODEL = torch.hub.load(
    "ultralytics/yolov5",  # repo_or_dir
    "custom",  # model
    "resources/md_v5a.0.0.pt",  # args for callable model
    force_reload=True,
    device=device,
)

logger.info("Initializing SAM model and loading pre-trained checkpoint.")
SAM = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
SAM.to(device=0)
SAM_PREDICTOR = SamPredictor(SAM)


def detect_animal(image_path: list) -> dict[str, Union[ndarray, Any]]:
    """Detect an animal in a given image."""

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    logger.info("Running detection inference.")
    results = DETECTION_MODEL(image)
    id2label = results.names
    results = results.xyxy[0].cpu().numpy()

    return {"bbox": np.array(list(int(_) for _ in results[0][:4].tolist())),
            "confidence": results[0][4],
            "class": id2label[results[0][5]]}


def segment_animal(image_path: list, bbox: list) -> np.ndarray:
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

    return foregroud_image[int(bbox[1]) - 5:int(bbox[3]) + 5, int(bbox[0]) - 5:int(bbox[2]) + 5]






