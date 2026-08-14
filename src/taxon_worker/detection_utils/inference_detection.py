import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

try:
    from ..infrastructure_utils import mem
except ImportError:
    from infrastructure_utils import mem

# import infrastructure_utils from local directory

# from fgvc.taxon_utils.taxon_utils import set_cuda_device
# DEVICE = torch.device(0 if torch.cuda.is_available() else "cpu")
DEVICE = mem.get_torch_cuda_device_if_available(0)

logger = logging.getLogger("app")
logger.info(f"Using device: {DEVICE}")

logger.info("Initializing MegaDetector model and loading pre-trained checkpoint.")

MEDIA_DIR = Path("/shared_data/media")
DETECTION_MODEL = None
DETECTION_MODEL_WARMED_UP = False
ORIENTATION_MODEL = None

SAM3 = None
SAM3_PREDICTOR: Any | None = None
SAM3_CHECKPOINT_PATH = "/root/resources/sam3/sam3.pt"
SAM3_HF_REPO_ID = "facebook/sam3"
SAM3_HF_FILENAME = "sam3.pt"

CLS_TO_ORIENTATION = {0: "back", 1: "front", 2: "left", 3: "right"}
KEEP_DETECTION_MODEL_LOADED = os.getenv("TAXON_KEEP_DETECTION_MODEL_LOADED", "").lower() in ("1", "true", "yes")
WARM_UP_DETECTION_MODEL_ON_START = os.getenv("TAXON_WARM_UP_DETECTION_MODEL_ON_START", "true").lower() in (
    "1",
    "true",
    "yes",
)
SAM3_FALLBACK_CONF = float(os.getenv("TAXON_SAM3_FALLBACK_CONF", "0.5"))


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
    logger.debug(f"Checking if file ({output_file}) does not exists.")
    if not os.path.exists(output_file):
        logger.debug(f"File does not exists. Downloading from url: {url} to {output_file}.")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        download_file(url, output_file)


def download_sam3_checkpoint_if_missing(
    checkpoint_path: str = SAM3_CHECKPOINT_PATH,
) -> str:
    """Download SAM3 weights from Hugging Face if not present locally."""
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    from huggingface_hub import hf_hub_download

    logger.info(f"SAM3 checkpoint not found at {checkpoint_path}, downloading from Hugging Face.")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=SAM3_HF_REPO_ID, filename=SAM3_HF_FILENAME)
    shutil.copy(downloaded, checkpoint_path)
    return checkpoint_path


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
    logger.debug("Before detection model.")
    logger.debug(f"{mem.get_vram(DEVICE)}     {mem.get_ram()}")
    if DETECTION_MODEL is None:
        model_url = r"https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
        filename = model_url.split("/")[-1]
        model_file = Path("/root/resources/") / filename
        download_file_if_does_not_exists(model_url, model_file)

        logger.debug(f"Loading model from file: {model_file}. {model_file.exists()=}")

        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        DETECTION_MODEL = torch.hub.load(
            "ultralytics/yolov5:915bbf2",  # repo_or_dir tag v7.0
            "custom",  # model
            str(model_file.expanduser()),  # args for callable model
            # force_reload=True,
            force_reload=force_reload,
            trust_repo=True,
            device=DEVICE,
        )
        DETECTION_MODEL.conf = 0.05
        warm_up_detection_model(DETECTION_MODEL)

    logger.debug("After detection model.")
    logger.debug(f"{mem.get_vram(DEVICE)}     {mem.get_ram()}")
    return DETECTION_MODEL


def warm_up_detection_model(model):
    """Run one tiny inference so CUDA kernels are initialized before the first real image."""
    global DETECTION_MODEL_WARMED_UP

    if DETECTION_MODEL_WARMED_UP:
        return

    logger.debug("Warming up detection model.")
    try:
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.inference_mode():
            model(dummy_image)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        DETECTION_MODEL_WARMED_UP = True
        logger.debug("Detection model warm-up finished.")
    except Exception:
        logger.warning(f"Detection model warm-up failed: {traceback.format_exc()}")


def get_orientation_model(model_name="resnet10t", model_checkpoint=""):
    """Load the orientation classification model."""
    # create model
    model = timm.create_model(model_name, num_classes=4, pretrained=True)

    # load model checkpoint
    if model_checkpoint:
        model_ckpt = torch.load(model_checkpoint)["model"]
        model.load_state_dict(model_ckpt)

    model = model.to(DEVICE).eval()
    return model


def del_detection_model():
    """Release the detection model."""
    global DETECTION_MODEL, DETECTION_MODEL_WARMED_UP
    DETECTION_MODEL = None
    DETECTION_MODEL_WARMED_UP = False
    torch.cuda.empty_cache()


def _load_sam3_model() -> Any | None:
    """Load SAM3 once and move the cached CPU model to the inference device."""
    global SAM3
    global SAM3_PREDICTOR

    if SAM3_PREDICTOR is not None:
        return SAM3_PREDICTOR

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        logger.warning("SAM3 package unavailable: %s", exc)
        return None

    logger.debug(f"Before SAM3 model: {mem.get_vram(DEVICE)}     {mem.get_ram()}")
    if SAM3 is None:
        try:
            download_sam3_checkpoint_if_missing()
        except Exception as exc:
            logger.warning("SAM3 weights unavailable: %s", exc)
            return None

        try:
            logger.info("Initializing SAM3 model and loading pre-trained checkpoint.")
            _checkpoint_path = Path(SAM3_CHECKPOINT_PATH).expanduser()
            SAM3 = build_sam3_image_model(
                checkpoint_path=str(_checkpoint_path),
                load_from_HF=False,
                device="cpu",
            )
        except Exception as exc:
            logger.warning("Failed to load SAM3 model: %s", exc)
            SAM3 = None
            torch.cuda.empty_cache()
            return None
    else:
        logger.info("Reusing cached SAM3 model from CPU memory.")

    mem.wait_for_gpu_memory(0.5)
    SAM3.to(device=DEVICE)
    SAM3_PREDICTOR = Sam3Processor(SAM3, device=str(DEVICE))
    logger.debug(f"After SAM3 model: {mem.get_vram(DEVICE)}     {mem.get_ram()}")
    return SAM3_PREDICTOR


def del_sam3_model():
    """Release SAM3 GPU memory while retaining weights in CPU memory."""
    global SAM3
    global SAM3_PREDICTOR

    SAM3_PREDICTOR = None
    if SAM3 is not None:
        SAM3.to(device="cpu")
    torch.cuda.empty_cache()


def detect_animals_with_megadetector(image_rgb: np.ndarray) -> Optional[List[Dict[str, Any]]]:
    """Detect animals/person/vehicle with MegaDetector. Returns detection_results dicts or None."""
    global DETECTION_MODEL

    if DETECTION_MODEL is None:
        logger.debug("Detection model is not loaded. Loading the model.")
        DETECTION_MODEL = get_detection_model()
        results = DETECTION_MODEL(image_rgb)
        logger.debug("Model loaded for the first time.")
    else:
        results = DETECTION_MODEL(image_rgb)
    id2label = results.names

    batch_idx = 0
    results = results.xyxy[batch_idx].cpu().numpy()

    if len(results) == 0:
        return None

    return [
        {
            "bbox": list(int(_) for _ in results[i][:4].tolist()),
            "confidence": float(results[i][4]),
            "class": id2label[results[i][5]],
            "size": image_rgb.shape[:2],
        }
        for i in range(len(results))
    ]


def detect_animals_with_sam3(image_rgb: np.ndarray) -> Optional[List[Dict[str, Any]]]:
    """Detect animals with SAM3 text prompt. Returns MegaDetector-compatible detection dicts."""
    predictor = _load_sam3_model()
    if predictor is None:
        return None

    height, width = image_rgb.shape[:2]
    image = Image.fromarray(image_rgb)

    try:
        with torch.autocast(dtype=torch.bfloat16, device_type=DEVICE.type):
            inference_state = predictor.set_image(image)
            output = predictor.set_text_prompt(state=inference_state, prompt="animal")
            boxes, scores = output["boxes"], output["scores"]

            if len(scores) == 0:
                logger.debug("SAM3 text prompt returned no detections.")
                return None

            boxes = boxes.detach().cpu().numpy()
            scores = scores.float().detach().cpu().numpy()

            results_list: List[Dict[str, Any]] = []
            for i in range(len(scores)):
                bbox = boxes[i]
                x0 = max(0, min(int(bbox[0]), width - 1))
                y0 = max(0, min(int(bbox[1]), height - 1))
                x1 = max(0, min(int(bbox[2]), width))
                y1 = max(0, min(int(bbox[3]), height))
                if x1 <= x0 or y1 <= y0:
                    continue
                results_list.append(
                    {
                        "bbox": [x0, y0, x1, y1],
                        "confidence": float(scores[i]),
                        "class": "animal",
                        "size": image_rgb.shape[:2],
                    }
                )

            if not results_list:
                return None

            results_list.sort(key=lambda det: det["confidence"], reverse=True)
            return results_list
    except Exception:
        logger.warning(f"SAM3 detection failed: {traceback.format_exc()}")
        return None


def warm_up_detection_model_on_start():
    """Warm up MegaDetector on worker start and release it again unless configured otherwise."""
    global DETECTION_MODEL

    if not WARM_UP_DETECTION_MODEL_ON_START:
        return

    DETECTION_MODEL = get_detection_model()
    if not KEEP_DETECTION_MODEL_LOADED:
        del_detection_model()


warm_up_detection_model_on_start()


def detect_animals_in_one_image(image_rgb: np.ndarray) -> Optional[List[Dict[str, Any]]]:
    """Detect an animal in a given image.

    Expected classes are: {0: 'animal', 1: 'person', 2: 'vehicle'}
    Falls back to SAM3 text-prompt detection when MegaDetector is empty or low-confidence.
    """
    megadetector_results = detect_animals_with_megadetector(image_rgb)

    # SAM3 fallback when MegaDetector finds nothing or only low-confidence boxes
    max_conf = (max(float(det["confidence"]) for det in megadetector_results) if megadetector_results else None)
    if megadetector_results is None or max_conf < SAM3_FALLBACK_CONF:
        logger.info(
            "MegaDetector empty or low-confidence (max_conf=%s, threshold=%s); trying SAM3 fallback.",
            max_conf,
            SAM3_FALLBACK_CONF,
        )
        sam3_results = detect_animals_with_sam3(image_rgb)
        if sam3_results:
            logger.info("SAM3 fallback returned %s detection(s).", len(sam3_results))
            return sam3_results
        logger.debug("SAM3 fallback unavailable or empty; keeping MegaDetector results.")

    return megadetector_results


def detect_animals_in_images(
    images_rgb: np.ndarray,
    batch_size: int = 1,
    pbar: Optional[tqdm] = None,
) -> List[Optional[List[Dict[str, Any]]]]:
    """Detect animals in a list of images."""
    global DETECTION_MODEL

    if DETECTION_MODEL is None:
        DETECTION_MODEL = get_detection_model()

    all_detections = []

    # split images into batches
    for i in range(0, len(images_rgb), batch_size):
        batch = list(images_rgb[i : i + batch_size])
        # logger.debug(f"{len(batch)=}, {len(images_rgb)=}")
        # logger.debug(f"{batch.shape=}")

        # here is the problem, because
        results = DETECTION_MODEL(batch)
        id2label = results.names
        if pbar is not None:
            pbar.update(float(len(batch)) / len(images_rgb))

        # results.xyxy is list of tensors, each tensor contains detections for one image in batch.
        for idx, single_result in enumerate(results.xyxy):
            # frame_id is the index of the frame in the original list of images
            frame_id = i + idx

            detections_np = single_result.cpu().numpy()

            if len(detections_np) == 0:
                # Pokud jsme nic nedetekovali, uložíme None
                all_detections.append(None)
                continue

            current_image_detections = []
            for det in detections_np:
                # det je ve formátu [x1, y1, x2, y2, confidence, class_id]
                bbox = list(map(int, det[:4].tolist()))
                conf = float(det[4])
                class_name = id2label[int(det[5])]

                detection_dict = {
                    "bbox": bbox,
                    "confidence": conf,
                    "class": class_name,
                    "size": batch[idx].shape[:2],  # (height, width)
                    "frame": frame_id,  # přidáváme pořadí snímku
                }
                current_image_detections.append(detection_dict)

            all_detections.append(current_image_detections)

    return all_detections


def human_annonymization(rgb_image: np.ndarray, bboxes: List[List[int]]) -> np.ndarray:
    """Annonymize humans in the image."""
    # get bbox and image
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        cropped_image = rgb_image[y0:y1, x0:x1]

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

        border = np.round((np.max([w, h]) * (0.25 / 2)) / 2).astype(int)
        pad_w += border
        pad_h += border

        padded_image = np.pad(cropped_image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
        rgb_image[y0:y1, x0:x1] = padded_image
    return rgb_image


def detect_animal_orientation(image_rgb: np.array, image_size: int = 176):
    """Detect animal orientation in cropped images."""
    global ORIENTATION_MODEL

    if ORIENTATION_MODEL is None:
        ORIENTATION_MODEL = get_orientation_model(
            "hf-hub:strakajk/Lynx-Orientation-ResNet10t-176"
            # "resnet10t", "resources/resnet10_02-b-13-02_19-08-16_orientation.pth"
        )

    transforms = T.Compose(
        [
            T.Resize(size=(image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image_rgb = Image.fromarray(image_rgb).convert("RGB")
    image = transforms(image_rgb)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    prediction = ORIENTATION_MODEL(image)
    cls_idx = torch.argmax(prediction)
    cls_idx = cls_idx.item()
    prediction = F.softmax(prediction, 1)
    score = prediction[0][cls_idx].item()

    return CLS_TO_ORIENTATION[cls_idx], score


def detect_animal_on_metadata(metadata: pd.DataFrame, border=0.0, progress_callback=None) -> pd.DataFrame:
    """Do the detection and segmentation on images in metadata.

    Returns:
        pd.DataFrame: metadata with added detection results.
    """
    assert "full_image_path" in metadata
    logger.info("Detection stage: starting for %s media files.", len(metadata))
    for position, (row_idx, row) in enumerate(
        tqdm(metadata.iterrows(), total=len(metadata), desc="Animal detection")
    ):
        if progress_callback is not None:
            progress_callback(position, len(metadata))
        image_abs_path = row["full_image_path"]
        try:
            if row["media_type"] == "video" and row["full_image_path"] == row["absolute_media_path"]:
                # there are no detected animals in video
                continue

            image = cv2.imread(str(image_abs_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # logger.debug(f"{image.shape=}")
            results = detect_animals_in_one_image(image_rgb=image)

            # "bbox": list(int(_) for _ in results[i][:4].tolist()),

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
                save_path = base_path / (Path(image_abs_path).stem + f".{ii}" + Path(image_abs_path).suffix)
                base_path.mkdir(exist_ok=True, parents=True)

                padded_image = pad_image(image, result["bbox"], border=border)
                Image.fromarray(padded_image).convert("RGB").save(save_path)

                # predict the orientation
                try:
                    orientation, score = detect_animal_orientation(image_rgb=padded_image)
                    row["detection_results"][ii]["orientation"] = orientation
                    row["detection_results"][ii]["orientation_score"] = score
                except Exception:
                    row["detection_results"][ii]["orientation"] = "unknown"
                    row["detection_results"][ii]["orientation_score"] = -1.0

                if ii == 0:
                    # if there is at least one detection save the very first one as the main image
                    save_path = base_path / (Path(image_abs_path).name)
                    Image.fromarray(padded_image).convert("RGB").save(save_path)

            metadata.loc[row_idx] = row
        except Exception:
            logger.warning(f"Cannot process image '{image_abs_path}'. Exception: {traceback.format_exc()}")
    if progress_callback is not None:
        progress_callback(len(metadata), len(metadata))
    if not KEEP_DETECTION_MODEL_LOADED:
        del_detection_model()
    del_sam3_model()
    logger.info("Detection stage: finished for %s media files.", len(metadata))
    return metadata
