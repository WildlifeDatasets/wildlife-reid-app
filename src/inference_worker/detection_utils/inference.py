import logging
import os
from pathlib import Path
from typing import Any, Union, List, Dict

import cv2
import numpy as np
import torch
from numpy import ndarray
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from PIL import Image
import traceback

# from fgvc.inference_utils.inference_utils import set_cuda_device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("app")
logger.info(f"Using device: {DEVICE}")

logger.info("Initializing MegaDetector model and loading pre-trained checkpoint.")

MEDIA_DIR = Path("/shared_data/media")


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


DETECTION_MODEL = None
SAM = None
SAM_PREDICTOR = None

def get_detection_model():
    global DETECTION_MODEL
    if DETECTION_MODEL is None:
        model_url = r"https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
        model_file = Path("/inference_worker/resources/md_v5a.0.0.pt")
        download_file_if_does_not_exists(model_url, model_file)

        logger.debug(f"Loading model from file: {model_file}. {model_file.exists()=}")

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        DETECTION_MODEL = torch.hub.load(
            "ultralytics/yolov5",  # repo_or_dir
            "custom",  # model
            str(model_file.expanduser()),  # args for callable model
            force_reload=True,
            device="cpu",
        )
    return DETECTION_MODEL

def del_detection_model():
    global DETECTION_MODEL
    DETECTION_MODEL = None

def get_sam_model():
    global SAM
    global SAM_PREDICTOR
    if SAM is None:

        download_file_if_does_not_exists(
            "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/sam_vit_h_4b8939.pth",
            "/inference_worker/resources/sam_vit_h_4b8939.pth",
        )

        logger.info("Initializing SAM model and loading pre-trained checkpoint.")
        _checkpoint_path = Path("/inference_worker/resources/sam_vit_h_4b8939.pth").expanduser()
        SAM = sam_model_registry["vit_h"](checkpoint=str(_checkpoint_path))
        SAM.to(device=DEVICE)
        SAM_PREDICTOR = SamPredictor(SAM)

    return SAM_PREDICTOR

def del_sam_model():
    global SAM
    global SAM_PREDICTOR
    SAM = None
    SAM_PREDICTOR = None

def detect_animals(image_rgb:np.ndarray) -> List[Dict[str, Union[ndarray, Any]]]:
    """Detect an animal in a given image."""

    logger.info("Running detection inference.")
    detection_model = get_detection_model()
    results = detection_model(image_rgb)
    id2label = results.names
    results_list = [None] * len(results.xyxy)
    for i in range(len(results.xyxy)):
        result = results.xyxy[0].cpu().numpy()

        results_list[i] = {
            "bbox": np.array(list(int(_) for _ in result[0][:4].tolist())),
            "confidence": result[0][4],
            "class": id2label[result[0][5]],
        }

    del_detection_model()
    return results_list


def segment_animal(image_path: str, bbox: list, cropped=True) -> np.ndarray:
    """Segment an animal in a given image using SAM model."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    logger.debug("Running segmentation inference.")
    sam_predictor = get_sam_model()
    sam_predictor.set_image(image)
    sam_input_box = np.array([int(point) for point in bbox])

    logger.debug(f"{sam_input_box=}")

    masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=sam_input_box[None, :],
        multimask_output=False,
    )
    logger.debug(f"{masks=}")

    foregroud_image = image.copy()
    foregroud_image[masks[0] == False] = 0  # noqa

    return pad_image(foregroud_image, bbox, border=0.25)


def detect_and_segment_animal_on_metadata(metadata, border=0.25, do_segmentation:bool=True):
    assert "image_path" in metadata
    masked_images = []
    detection_results = []
    for image_path in tqdm(metadata["image_path"]):
        # TODO image path should be relative to local storage (/shared_data/media/)
        image_abs_path = MEDIA_DIR / image_path
        try:

            image = cv2.imread(str(image_abs_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detect_animals(image_rgb=image)
            detection_results.append(results)
            # logger.debug(f"results={results}")
            for ii, result in enumerate(results):

                # if result["class"] == "animal":
                base_path = Path(image_abs_path).parent.parent / "detection_images"
                save_path = base_path / (Path(image_abs_path).stem + f".{ii}" + Path(image_abs_path).suffix)
                base_path.mkdir(exist_ok=True, parents=True)

                padded_image = pad_image(image, result["bbox"], border=border)
                Image.fromarray(padded_image).convert("RGB").save(save_path)
                if ii == 0:
                    # if there is at least one detection save the very first one as the main image
                    save_path = base_path / (Path(image_abs_path).name)
                    Image.fromarray(padded_image).convert("RGB").save(save_path)


                # save_path = None
                # if result["class"] == "animal":
                #     cropper_animal = segment_animal(image_path, result["bbox"])
                #
                #     base_path = Path(image_path).parent.parent / "masked_images"
                #     save_path = base_path / Path(image_path).name
                #     base_path.mkdir(exist_ok=True, parents=True)
                #     Image.fromarray(cropper_animal).convert("RGB").save(save_path)
            # masked_images.append(str(save_path))
        except Exception:
            logger.warning(
                f"Cannot process image '{image_abs_path}'. Exception: {traceback.format_exc()}"
            )
            # masked_images.append("")
            detection_results.append(None)
    # metadata["masked_image_path"] = masked_images
    metadata["detection_results"] = masked_images
