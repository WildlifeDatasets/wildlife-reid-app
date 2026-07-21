import ast
import logging
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
from tqdm import tqdm
from wildlife_tools import realize
from wildlife_tools.data import FeatureDataset, WildlifeDataset
from wildlife_tools.features import AlikedExtractor, DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.pairwise.collectors import CollectAll
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue

from .postprocessing import _sequence_max_conf, _sequence_voting, _sequence_weighted_voting
from .wildfusion_utils import SimilarityPipelineExtended, WildFusionExtended

try:
    from ..infrastructure_utils import mem
except ImportError:
    from infrastructure_utils import mem

logger = logging.getLogger("app")
# DEVICE = set_cuda_device("1") if torch.cuda.is_available() else "cpu"
DEVICE = mem.get_torch_cuda_device_if_available(0)  # TODO set device to 1
logger.setLevel(logging.DEBUG)
logger.info(f"Using device: {DEVICE}")

# IDENTIFICATION_MODELS = None
# SAM: Optional[Sam] = None
# SAM_PREDICTOR: Optional[SamPredictor] = None
SAM: Sam | None = None
SAM_PREDICTOR: SamPredictor | None = None
IDENTIFICATION_MODELS: dict[str, SimilarityPipelineExtended] | None = None
SAM3 = None
SAM3_PREDICTOR: Any | None = None
SEGMENTATION_BACKEND: Literal["sam3", "sam"] | None = None
SAM3_CHECKPOINT_PATH = "/root/resources/sam3/sam3.pt"
SAM3_HF_REPO_ID = "facebook/sam3"
SAM3_HF_FILENAME = "sam3.pt"
BBOX_RELATIVE_COLUMNS = ("bbox_cx", "bbox_cy", "bbox_w", "bbox_h")


def _normalize_bbox_cxcywh(bbox_cxcywh: torch.Tensor, width: int, height: int) -> list[float]:
    """Normalize cxcywh bbox coordinates to [0, 1] by image size."""
    bbox = bbox_cxcywh.clone()
    bbox[..., 0] /= width
    bbox[..., 1] /= height
    bbox[..., 2] /= width
    bbox[..., 3] /= height
    return bbox.flatten().tolist()

@dataclass
class Prediction:
    name: str
    db_idx: int
    score: float
    path: str


class CarnivoreDataset(WildlifeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image_paths(self):
        """Return the image paths."""
        return self.metadata["path"].astype(str).values


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
        logger.debug(f"File does not exists. Downloading. {output_file=}")
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


def get_identification_model(model_name, model_checkpoint=""):
    """Load the model from the given model name and checkpoint."""
    # no need of 'global' if only reading the variable
    global IDENTIFICATION_MODELS

    if IDENTIFICATION_MODELS is not None:
        return
    IDENTIFICATION_MODELS = None

    logger.debug("Before identification model.")
    logger.debug(f"{mem.get_vram(DEVICE)}     {mem.get_ram()}")
    logger.debug(f"{model_name=}")
    mem.wait_for_gpu_memory(0.5)

    logger.info("Initializing identification model.")
    # load model checkpoint
    model = timm.create_model(model_name, num_classes=0, pretrained=True)
    if model_checkpoint:
        model_ckpt = torch.load(model_checkpoint, map_location=torch.device("cpu"))["model"]
        model.load_state_dict(model_ckpt)

    identification_model = model.to(DEVICE).eval()

    config = {
        "method": "TransformTimm",
        "input_size": np.max(identification_model.default_cfg["input_size"][1:]),
        "is_training": False,
        "auto_augment": "rand-m10-n2-mstd1",
    }

    matcher_aliked = SimilarityPipelineExtended(
        matcher=MatchLightGlue(features="aliked"),
        extractor=AlikedExtractor(),
        transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]),
        calibration=IsotonicCalibration(),
    )

    matcher_mega = SimilarityPipelineExtended(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(identification_model, batch_size=4, num_workers=1, device=DEVICE),
        transform=realize(config),
        calibration=IsotonicCalibration(),
    )

    IDENTIFICATION_MODELS = {"mega": matcher_mega, "aliked": matcher_aliked}

    logger.debug("After identification model.")
    logger.debug(f"{mem.get_vram(DEVICE)}     {mem.get_ram()}")
    return IDENTIFICATION_MODELS


def _load_sam_model() -> SamPredictor:
    """Load SAM once and move the cached CPU model to the inference device."""
    global SAM
    global SAM_PREDICTOR

    if SAM_PREDICTOR is not None:
        return SAM_PREDICTOR

    logger.debug(f"Before segmentation model: {mem.get_vram(DEVICE)}     {mem.get_ram()}")
    model_zoo = {
        "vit_b": "sam_vit_b_01ec64",
        "vit_l": "sam_vit_l_0b3195",
        "vit_h": "sam_vit_h_4b8939",
    }
    model_version = os.environ["SAM_MODEL_VERSION"]
    if SAM is None:
        download_file_if_does_not_exists(
            f"https://dl.fbaipublicfiles.com/segment_anything/{model_zoo[model_version]}.pth",
            # f"http://ptak.felk.cvut.cz/plants/DanishFungiDataset/{model_zoo[model_version]}.pth",
            f"/root/resources/{model_zoo[model_version]}.pth",
        )

        logger.info(f"Initializing SAM model ({model_version}) and loading pre-trained checkpoint.")
        _checkpoint_path = Path(f"/root/resources/{model_zoo[model_version]}.pth").expanduser()
        SAM = sam_model_registry[model_version](checkpoint=str(_checkpoint_path))
    else:
        logger.info(f"Reusing cached SAM model ({model_version}) from CPU memory.")

    mem.wait_for_gpu_memory(0.5)
    SAM.to(device=DEVICE)
    SAM_PREDICTOR = SamPredictor(SAM)
    logger.debug(f"After segmentation model: {mem.get_vram(DEVICE)}     {mem.get_ram()}")
    return SAM_PREDICTOR


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

    logger.debug(f"Before segmentation model: {mem.get_vram(DEVICE)}     {mem.get_ram()}")
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
    logger.debug(f"After segmentation model: {mem.get_vram(DEVICE)}     {mem.get_ram()}")
    return SAM3_PREDICTOR


def get_segmentation_model() -> Literal["sam3", "sam"]:
    """Load the segmentation model. Prefer SAM3, fall back to SAM."""
    global SEGMENTATION_BACKEND

    if SEGMENTATION_BACKEND is not None:
        return SEGMENTATION_BACKEND

    if _load_sam3_model() is not None:
        SEGMENTATION_BACKEND = "sam3"
        logger.info("Using SAM3 for segmentation")
        return SEGMENTATION_BACKEND

    logger.warning("SAM3 is not available, falling back to SAM for segmentation")
    _load_sam_model()
    SEGMENTATION_BACKEND = "sam"
    return SEGMENTATION_BACKEND

def del_identification_model():
    """Release the identification model."""
    global IDENTIFICATION_MODELS
    IDENTIFICATION_MODELS = None
    torch.cuda.empty_cache()


def del_sam_model():
    """Release segmentation GPU memory while retaining weights in CPU memory."""
    global SAM
    global SAM_PREDICTOR
    global SAM3
    global SAM3_PREDICTOR
    global SEGMENTATION_BACKEND

    SAM_PREDICTOR = None
    SAM3_PREDICTOR = None
    SEGMENTATION_BACKEND = None
    if SAM is not None:
        SAM.to(device="cpu")
    if SAM3 is not None:
        SAM3.to(device="cpu")
    torch.cuda.empty_cache()


def init_models(identification_model_path):
    """Initialize identification and segmentation models."""
    get_identification_model(identification_model_path)
    get_segmentation_model()


def del_models():
    """Remove identification and segmentation models from memory."""
    del_identification_model()
    del_sam_model()


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


def _segment_animal_sam(image_path: str, bbox: list, border: float = 0.25) -> np.ndarray:
    """Segment an animal in a given image using SAM model."""
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"OpenCV cannot read image '{image_path}'.")
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
    foregroud_image[masks[0] == False] = 0  # noqa

    return pad_image(foregroud_image, bbox, border=border)


def _segment_animal_sam3(image_path: str, bbox: list, border: float = 0.25) -> np.ndarray:
    """Segment an animal in a given image using SAM3 model."""
    from sam3.model.box_ops import box_xyxy_to_cxcywh

    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"OpenCV cannot read image '{image_path}'.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    width, height = image.size

    with torch.autocast(dtype=torch.bfloat16, device_type=DEVICE.type):
        inference_state = SAM3_PREDICTOR.set_image(image)

        # Use bbox prompt first
        bbox_xyxy = torch.tensor(bbox).view(-1, 4)
        bbox_cxcywh = box_xyxy_to_cxcywh(bbox_xyxy)
        norm_bbox_cxcywh = _normalize_bbox_cxcywh(bbox_cxcywh, width, height)

        # The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        output = SAM3_PREDICTOR.add_geometric_prompt(state=inference_state, box=norm_bbox_cxcywh, label=True)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        mask_source = "bbox_prompt"

        # Fallback to text prompt
        if len(scores) == 0:
            SAM3_PREDICTOR.reset_all_prompts(inference_state)
            output = SAM3_PREDICTOR.set_text_prompt(state=inference_state, prompt="animal")
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            mask_source = "text_prompt"

        # Process predictions
        if len(scores) == 0:
            mask = np.ones([height, width])
            bbox = [0, 0, width, height]
            score = 0
            logger.debug("No mask found, using fallback full image mask.")
            mask_source = "fallback_full"
        else:
            masks = masks.detach().cpu().numpy()
            boxes = boxes.detach().cpu().numpy()
            scores = scores.float().detach().cpu().numpy()
            idx = np.argmax(scores)

            mask = masks[idx][0]
            bbox = boxes[idx]
            score = scores[idx]

        logger.debug(f"Using mask and bbox from: {mask_source}, bbox: {bbox}, score: {score}")

    # Mask and crop the input
    foregroud_image = np.array(image).copy()
    foregroud_image[mask == False] = 0  # noqa

    # Clip bbox values to image boundaries
    bbox = [
        max(0, min(bbox[0], width - 1)),
        max(0, min(bbox[1], height - 1)),
        max(0, min(bbox[2], width)),
        max(0, min(bbox[3], height)),
    ]
    return pad_image(foregroud_image, bbox, border=border)


def segment_animal(image_path: str, bbox: list, border: float = 0.25) -> np.ndarray:
    """Segment an animal using the active segmentation model (SAM3 or SAM)."""
    backend = get_segmentation_model()
    if backend == "sam3":
        return _segment_animal_sam3(image_path, bbox, border=border)
    return _segment_animal_sam(image_path, bbox, border=border)


def _is_valid_xyxy_bbox(bbox) -> bool:
    """Return True if bbox is xyxy with positive width and height."""
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    return x1 > x0 and y1 > y0


def _get_masking_bbox(row: pd.Series, image_path: str, row_idx) -> list[int] | None:
    """Get the bbox for masking from the row."""
    has_relative_bbox = all(
        column in row.index and not pd.isna(row[column]) and row[column] != ""
        for column in BBOX_RELATIVE_COLUMNS
    )

    # try to build bbox from relative columns
    if has_relative_bbox:
        try:
            image = cv2.imread(image_path)
            if image is None or image.size == 0:
                raise ValueError(f"OpenCV cannot read image '{image_path}'.")
            height, width = image.shape[:2]
            cx, cy, bw, bh = (float(row[column]) for column in BBOX_RELATIVE_COLUMNS)
            x0, y0 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
            x1, y1 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
            bbox = [max(0, min(width, x0)), max(0, min(height, y0)), max(0, min(width, x1)), max(0, min(height, y1))]
            if _is_valid_xyxy_bbox(bbox):
                logger.debug(f"Masking bbox from relative columns: bbox={bbox}")
                return bbox
            logger.warning(
                f"Degenerate relative bbox for image {image_path} at metadata row {row_idx}: {bbox}; "
                "falling back to detection_results."
            )
        except (ValueError, TypeError) as exc:
            logger.warning(
                f"Could not build bbox from relative columns for image {image_path} at metadata row {row_idx}: {exc}; "
                "falling back to detection_results."
            )

    # try to build bbox from detection results
    if pd.isna(row.get("detection_results")):
        logger.debug(f"No detection results for image: {image_path}, row['detection_results'] is None.")
        return None

    try:
        detection_results = ast.literal_eval(row["detection_results"])
    except (SyntaxError, ValueError) as exc:
        logger.warning(f"Could not parse detection_results for image {image_path} at metadata row {row_idx}: {exc}")
        return None

    if not detection_results:
        logger.debug(f"No detection results for image: {image_path}")
        return None

    try:
        bbox = detection_results[0]["bbox"]
    except (IndexError, KeyError, TypeError) as exc:
        logger.warning(f"Detection result for image {image_path} at metadata row {row_idx} does not contain a usable bbox: {exc}")
        return None

    if not _is_valid_xyxy_bbox(bbox):
        logger.warning(
            f"Degenerate detection bbox for image {image_path} at metadata row {row_idx}: {bbox}"
        )
        return None

    logger.debug(f"Masking bbox from detection_results: bbox={bbox}")
    return bbox


def mask_images(metadata: pd.DataFrame, tqdm_desc="Masking images") -> pd.DataFrame:
    """Mask images using the segmentation model."""
    masked_paths = []
    segmentation_backend = get_segmentation_model()
    logger.info(f"Masking images with {segmentation_backend.upper()}")

    for row_idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc=tqdm_desc):
        image_path = row["image_path"]
        bbox = _get_masking_bbox(row, image_path, row_idx)
        if bbox is None:
            masked_paths.append(str(image_path))
            continue

        try:
            cropped_animal = segment_animal(image_path, bbox)
        except Exception as exc:
            logger.warning(f"Skipping mask for image {image_path} at metadata row {row_idx} with bbox {bbox}: {exc}")
            masked_paths.append(str(image_path))
            continue

        # save masked image
        base_path = Path(image_path).parent.parent / "masked_images"

        save_path = base_path / Path(image_path).name
        base_path.mkdir(exist_ok=True, parents=True)
        Image.fromarray(cropped_animal).convert("RGB").save(save_path)
        masked_paths.append(str(save_path))

    metadata["image_path"] = masked_paths
    del_sam_model()
    return metadata


def encode_images(metadata: pd.DataFrame, identification_model_path: str, tqdm_desc="") -> list:
    """Create feature vectors from given images."""
    # no need of 'global' if only reading the variable
    # global IDENTIFICATION_MODELS
    get_identification_model(identification_model_path)
    metadata = mask_images(metadata, tqdm_desc=f"Masking images: {tqdm_desc}")
    logger.info("Creating DataLoaders.")

    dataset = CarnivoreDataset(
        metadata=metadata,
        root="",
        img_load="full",
        col_path="image_path",
        col_label="label",
    )

    logger.info("Running inference.")
    # extract global and local features
    features_mega = IDENTIFICATION_MODELS["mega"].get_feature_dataset(dataset)
    features_aliked = IDENTIFICATION_MODELS["aliked"].get_feature_dataset(dataset)

    # postprocess global features
    _features_mega = []
    for _features in features_mega.features:
        _features_mega.append(_features.tolist())
    features_mega = _features_mega

    # postprocess local features - remove unnecessary feature keys
    keep_keys = ["keypoints", "descriptors", "image_size"]
    _features_aliked = []
    for fidx in range(len(features_aliked.features)):
        _features = {}
        for key in keep_keys:
            _features[key] = features_aliked.features[fidx][key].numpy().tolist()
        _features_aliked.append(_features)
    features_aliked = _features_aliked

    # gather features
    features = list(zip(features_mega, features_aliked))

    return features


def prepare_feature_types(features):
    """Prepare feature types for identification."""
    mega_features = []
    aliked_features = []
    for _mega_features, _aliked_features in features:
        mega_features.append(torch.tensor(_mega_features))
        aliked_features.append({k: torch.tensor(v) for k, v in _aliked_features.items()})
    mega_features = np.array(mega_features)
    return aliked_features, mega_features


def calibrate_models(calibrated_features: list, calibration_metadata: pd.DataFrame):
    """Calibrate identification models."""
    logger.debug(f"Calibrating identification models with {len(calibrated_features)} images.")
    # prepare feature datasets
    calibration_aliked_features, calibration_mega_features = prepare_feature_types(calibrated_features)
    calibration_mega_features = FeatureDataset(calibration_mega_features, calibration_metadata)
    calibration_aliked_features = FeatureDataset(calibration_aliked_features, calibration_metadata)

    # calibrate models before identification
    IDENTIFICATION_MODELS["mega"].fit_calibration(calibration_mega_features, calibration_mega_features)
    IDENTIFICATION_MODELS["aliked"].fit_calibration(calibration_aliked_features, calibration_aliked_features)


def compute_partial(
    query_features: list,
    database_features: list,
    query_metadata: pd.DataFrame,
    database_metadata: pd.DataFrame,
    identification_model_path: str,
    target: str,
    pairs: tuple = None,
):
    """Compare input feature vectors with the reference feature vectors and make predictions."""
    assert len(query_features) == len(query_metadata)
    assert len(database_features) == len(database_metadata)
    valid_targets = ["priority", "scores"]
    assert target in valid_targets, f"Invalid target: {target} for partial computation, valid targets: {valid_targets}"
    if target == "scores":
        assert pairs is not None, "Pairs must be provided for scores computation"
    logger.info(f"Starting identification of {len(query_metadata)} images.")

    get_identification_model(identification_model_path)

    # gather features
    database_aliked_features, database_mega_features = prepare_feature_types(database_features)
    query_aliked_features, query_mega_features = prepare_feature_types(query_features)

    # wrap features in feature dataset
    database_mega_features = FeatureDataset(database_mega_features, database_metadata)
    database_aliked_features = FeatureDataset(database_aliked_features, database_metadata)
    query_mega_features = FeatureDataset(query_mega_features, query_metadata)
    query_aliked_features = FeatureDataset(query_aliked_features, query_metadata)

    database_features = {
        DeepFeatures: database_mega_features,
        AlikedExtractor: database_aliked_features,
    }
    query_features = {DeepFeatures: query_mega_features, AlikedExtractor: query_aliked_features}

    # identify individuals
    wildfusion = WildFusionExtended(
        calibrated_matchers=[IDENTIFICATION_MODELS["aliked"], IDENTIFICATION_MODELS["mega"]],
        priority_matcher=IDENTIFICATION_MODELS["mega"],
    )

    if target == "priority":
        priority = wildfusion.get_partial_priority(query_features, database_features)
        return priority

    elif target == "scores":
        scores = wildfusion.get_partial_scores(query_features, database_features, pairs)
        return scores


def _get_top_predictions(similarity, database_metadata, top_k: int = 1):
    """
    Returns:
        List[List[Prediction]]
    """
    predictions_all = []
    database_labels = np.array(database_metadata["identity"])
    database_paths = np.array(database_metadata["path"])

    for row in similarity:
        sort_idx = np.argsort(row)[::-1]
        seen_names = set()
        image_predictions = []

        for idx in sort_idx:
            name = database_labels[idx]
            path = database_paths[idx]

            if name in seen_names:
                continue
            seen_names.add(name)

            pred = Prediction(name=name, db_idx=int(idx), score=float(row[idx]), path=path)

            image_predictions.append(pred)
            if len(image_predictions) == top_k:
                break

        predictions_all.append(image_predictions)
    return predictions_all


def _post_process_sequence(
    predictions: List[List[Prediction]],
    query_metadata: pd.DataFrame,
    top_k: int,
    method: Optional[str] = None,
    ignore_seq_ids: list = [],
):
    if method is None:
        return predictions

    if "sequence_number" not in query_metadata.columns:
        raise ValueError("query_metadata must contain 'sequence_number' column")

    # Group images by sequence
    sequence_to_indices = {}
    for idx, seq_id in enumerate(query_metadata["sequence_number"]):
        sequence_to_indices.setdefault(seq_id, []).append(idx)

    sequence_to_predictions = {}

    for seq_id, indices in sequence_to_indices.items():
        seq_preds = [predictions[i] for i in indices]

        if seq_id in ignore_seq_ids:
            for idx in indices:
                sequence_to_predictions[idx] = predictions[idx]
            continue

        if method == "voting":
            aggregated = _sequence_voting(seq_preds, top_k)
        elif method == "weighted_voting":
            aggregated = _sequence_weighted_voting(seq_preds, top_k)
        elif method == "max_conf":
            aggregated = _sequence_max_conf(seq_preds, top_k)
        else:
            raise ValueError(f"Unknown post-processing method: {method}")

        for idx in indices:
            sequence_to_predictions[idx] = aggregated

    # Map back
    final = []
    for idx in range(len(query_metadata)):
        final.append(sequence_to_predictions[idx])

    return final


def identify_from_similarity(
    similarity: np.ndarray,
    database_metadata: pd.DataFrame,
    query_metadata: pd.DataFrame,
    top_k: int,
    post_process: str = "",
):
    """Get top-k predictions from similarity matrix."""
    logger.info(f"Predicting top-{top_k}, with post-processing: {post_process}")

    predictions = _get_top_predictions(similarity, database_metadata, top_k=top_k)

    # Apply sequence post-processing
    if post_process and post_process is not None:
        predictions = _post_process_sequence(predictions, query_metadata, top_k, method=post_process, ignore_seq_ids=[])

    # reformat results
    pred_image_paths = []
    pred_class_ids = []
    scores = []
    result_idx = {}
    for qidx, row in enumerate(predictions):
        pred_class_ids.append([int(pred.name) for pred in row])
        pred_image_paths.append([pred.path for pred in row])
        scores.append(np.clip([pred.score for pred in row], 0, 1).tolist())
        result_idx[qidx] = [pred.db_idx for pred in row]
    print(scores)

    # return path to original image
    masked_image_paths = pred_image_paths
    _pred_image_paths = []
    for paths in pred_image_paths:
        _pred_image_paths.append([p.replace("/masked_images/", "/images/") for p in paths])
    pred_image_paths = _pred_image_paths

    output = {
        "pred_image_paths": pred_image_paths,
        "pred_masked_paths": masked_image_paths,
        "pred_class_ids": pred_class_ids,
        "scores": scores,
    }
    return output, result_idx


def get_keypoints(keypoint_matcher, query_features, database_features, max_kp=10):
    """Run matcher and return top matched keypoint pairs."""
    score_thr = 0.9
    skip_kp = 10
    keypoint_output = keypoint_matcher(query_features, database_features)
    _keypoints = []
    for _keypoint_output in keypoint_output:
        thr_mask = _keypoint_output["scores"] >= score_thr

        scores = _keypoint_output["scores"][thr_mask]
        kps0 = _keypoint_output["kpts0"][thr_mask]
        kps1 = _keypoint_output["kpts1"][thr_mask]
        if len(kps0) < skip_kp:
            _keypoints.append(([], []))
            continue
        try:
            sort_idx = np.argsort(scores)[::-1][:max_kp]
            kps0 = kps0[sort_idx].tolist()
            kps1 = kps1[sort_idx].tolist()
        except Exception as e:
            logger.debug(f"{traceback.format_exc()}")
            logger.debug(f"{scores=}")
            logger.debug(f"{max_kp=}")
            logger.warning(f"Error in get_keypoints: {e}")

            kps0 = []
            kps1 = []

        _keypoints.append((kps0, kps1))

    return _keypoints


def identify(
    query_features: list,
    database_features: list,
    query_metadata: pd.DataFrame,
    database_metadata: pd.DataFrame,
    identification_model_path,
    top_k: int = 3,
    cal_images: int = 50,
    image_budget: int = 100,
) -> dict:
    """Compare input feature vectors with the reference feature vectors and make predictions."""
    assert len(query_features) == len(query_metadata)
    assert len(database_features) == len(database_metadata)
    logger.info(f"Starting identification of {len(query_metadata)} images.")

    global IDENTIFICATION_MODELS
    get_identification_model(identification_model_path)

    # gather features
    database_aliked_features, database_mega_features = prepare_feature_types(database_features)
    query_aliked_features, query_mega_features = prepare_feature_types(query_features)

    # wrap features in feature dataset
    calibration_mega_features = FeatureDataset(database_mega_features[:cal_images], database_metadata[:cal_images])
    calibration_aliked_features = FeatureDataset(database_aliked_features[:cal_images], database_metadata[:cal_images])
    database_mega_features = FeatureDataset(database_mega_features, database_metadata)
    database_aliked_features = FeatureDataset(database_aliked_features, database_metadata)
    query_mega_features = FeatureDataset(query_mega_features, query_metadata)
    query_aliked_features = FeatureDataset(query_aliked_features, query_metadata)

    # calibrate models before identification
    IDENTIFICATION_MODELS["mega"].fit_calibration(calibration_mega_features, calibration_mega_features)
    IDENTIFICATION_MODELS["aliked"].fit_calibration(calibration_aliked_features, calibration_aliked_features)

    database_features = {
        DeepFeatures: database_mega_features,
        AlikedExtractor: database_aliked_features,
    }
    query_features = {DeepFeatures: query_mega_features, AlikedExtractor: query_aliked_features}

    # identify individuals
    wildfusion = WildFusionExtended(
        calibrated_matchers=[IDENTIFICATION_MODELS["aliked"], IDENTIFICATION_MODELS["mega"]],
        priority_matcher=IDENTIFICATION_MODELS["mega"],
    )
    similarity = wildfusion(query_features, database_features, B=image_budget)
    logger.debug(f"{similarity.shape=}")
    IDENTIFICATION_MODELS = None

    output, result_idx = identify_from_similarity(
        similarity, database_metadata, query_metadata, top_k, post_process=os.environ.get("POST_PROCESS", None)
    )

    # calculate keypoints
    max_kp = int(os.environ.get("VISUALIZATION_KEYPOINTS", 10))
    collector = CollectAll()
    keypoint_matcher = MatchLightGlue(features="aliked", collector=collector)

    keypoints = []
    logger.debug(result_idx)
    for qidx, didx in result_idx.items():
        qidx = [qidx]
        keypoint_query_features = FeatureDataset(
            np.array(query_aliked_features.features)[qidx],
            query_aliked_features.metadata.iloc[qidx],
        )
        keypoint_database_features = FeatureDataset(
            np.array(database_aliked_features.features)[didx],
            database_aliked_features.metadata.iloc[didx],
        )

        _keypoints = get_keypoints(keypoint_matcher, keypoint_query_features, keypoint_database_features, max_kp=max_kp)
        keypoints.append(_keypoints)

    output["keypoints"] = keypoints
    return output
