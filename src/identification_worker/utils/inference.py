import ast
import logging
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from wildlife_tools import realize
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity

from fgvc.utils.utils import set_cuda_device

from .inference_local import get_merged_predictions
from .postprocessing import feature_top

logger = logging.getLogger("app")
DEVICE = set_cuda_device("1" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

IDENTIFICATION_MODEL = None
SAM = None
SAM_PREDICTOR = None


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


def get_identification_model(model_name, model_checkpoint=""):
    """Load the model from the given model name and checkpoint."""
    global IDENTIFICATION_MODEL
    logger.debug(f"{torch.cuda.memory_snapshot()=}")
    if IDENTIFICATION_MODEL is not None:
        return

    logger.info("Initializing identification model.")
    # load model checkpoint
    model = timm.create_model(model_name, num_classes=0, pretrained=True)
    if model_checkpoint:
        model_ckpt = torch.load(model_checkpoint, map_location=torch.device("cpu"))["model"]
        model.load_state_dict(model_ckpt)
        logger.debug(f"{torch.cuda.memory_snapshot()=}")

    IDENTIFICATION_MODEL = model.to(DEVICE).eval()


def get_sam_model() -> SamPredictor:
    """Load the SAM model if not loaded before."""
    global SAM
    global SAM_PREDICTOR
    logger.debug(f"{torch.cuda.memory_snapshot()=}")
    model_zoo = {
        "vit_b": "sam_vit_b_01ec64",
        "vit_l": "sam_vit_l_0b3195",
        "vit_h": "sam_vit_h_4b8939"
    }
    model_version = os.environ['SAM_MODEL_VERSION']
    if SAM is None:
        download_file_if_does_not_exists(
            f"http://ptak.felk.cvut.cz/plants/DanishFungiDataset/{model_zoo[model_version]}.pth",
            f"/identification_worker/resources/{model_zoo[model_version]}.pth",
        )

        logger.info(f"Initializing SAM model ({model_version}) and loading pre-trained checkpoint.")
        _checkpoint_path = Path(
            f"/identification_worker/resources/{model_zoo[model_version]}.pth"
        ).expanduser()
        SAM = sam_model_registry[model_version](checkpoint=str(_checkpoint_path))
        SAM.to(device=DEVICE)
        SAM_PREDICTOR = SamPredictor(SAM)
        logger.debug(f"{torch.cuda.memory_snapshot()=}")
    return SAM_PREDICTOR


def del_identification_model():
    """Release the identification model."""
    global IDENTIFICATION_MODEL
    IDENTIFICATION_MODEL = None


def del_sam_model():
    """Release the SAM model."""
    global SAM
    global SAM_PREDICTOR
    SAM = None
    SAM_PREDICTOR = None


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


def segment_animal(image_path: str, bbox: list, border: float = 0.25) -> np.ndarray:
    """Segment an animal in a given image using SAM model."""
    global SAM_PREDICTOR

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

    foregroud_image = image.copy()
    foregroud_image[masks[0] == False] = 0  # noqa

    return pad_image(foregroud_image, bbox, border=border)


def mask_images(metadata: pd.DataFrame) -> pd.DataFrame:
    """Mask images using SAM model."""
    masked_paths = []
    get_sam_model()
    for row_idx, row in metadata.iterrows():
        image_path = row["image_path"]
        # detection_results = ast.literal_eval(
        #    ast.literal_eval(row["detection_results"])["detection_results"])
        detection_results = ast.literal_eval(row["detection_results"])
        # detection_results = json.loads(row["detection_results"])
        logger.debug(f"{detection_results=}")
        if len(detection_results) == 0:
            logger.debug(f"No detection results for image: {image_path}")
            masked_paths.append(str(image_path))
            continue

        bbox = detection_results[0]["bbox"]

        cropped_animal = segment_animal(image_path, bbox)

        base_path = Path(image_path).parent.parent / "masked_images"

        save_path = base_path / Path(image_path).name
        base_path.mkdir(exist_ok=True, parents=True)
        Image.fromarray(cropped_animal).convert("RGB").save(save_path)
        logger.debug(f"Saving masked file: {save_path}")

        masked_paths.append(str(save_path))

    metadata["image_path"] = masked_paths
    del_sam_model()
    return metadata


def encode_images(metadata: pd.DataFrame) -> np.ndarray:
    """Create feature vectors from given images."""
    global IDENTIFICATION_MODEL
    get_identification_model(os.environ["IDENTIFICATION_MODEL_VERSION"])
    metadata = mask_images(metadata)
    logger.info("Creating DataLoaders.")
    config = {
        "method": "TransformTimm",
        "input_size": np.max(IDENTIFICATION_MODEL.default_cfg["input_size"][1:]),
        "is_training": False,
        "auto_augment": "rand-m10-n2-mstd1",
    }
    transform = realize(config)

    dataset = CarnivoreDataset(
        metadata=metadata,
        root="",
        transform=transform,
        img_load="full",
        col_path="image_path",
        col_label="label",
    )

    logger.info("Running inference.")
    extractor = DeepFeatures(IDENTIFICATION_MODEL, batch_size=4, num_workers=1, device=DEVICE)
    features = extractor(dataset)

    del_identification_model()
    return features


def _get_top_predictions(similarity: np.ndarray, paths: list, identities: list, top_k: int = 1):
    """Get top-k predictions from similarity matrix."""
    top_results = []
    for row_idx, row in enumerate(similarity):
        top_names = []
        top_paths = []
        top_scores = []

        sort_idx = np.argsort(row)[::-1]
        names_sorted = identities[sort_idx]
        paths_sorted = paths[sort_idx]
        scores_sorted = row[sort_idx]

        for i, (s, n, p) in enumerate(zip(scores_sorted, names_sorted, paths_sorted)):
            if n in top_names:
                continue
            top_names.append(n)
            top_paths.append(p)
            top_scores.append(s)
            if len(top_names) == top_k:
                break

        if len(top_names) < top_k:
            diff = top_k - len(top_names)
            top_names.extend([top_names[-1]] * diff)
            top_paths.extend([top_paths[-1]] * diff)
            top_scores.extend([top_scores[-1]] * diff)
        top_results.append((top_names, top_paths, top_scores))

    return top_results


def identify(
    features: np.ndarray,
    reference_features: np.ndarray,
    reference_image_paths: list,
    reference_class_ids: list,
    metadata: pd.DataFrame,
    top_k: int = 3,
) -> dict:
    """Compare input feature vectors with the reference feature vectors and make predictions."""
    assert len(reference_features) == len(reference_image_paths)
    assert len(reference_features) == len(reference_class_ids)
    logger.info(f"Starting identification of {len(features)} images.")

    similarity_measure = CosineSimilarity()
    similarity = similarity_measure(
        features.astype(np.float32), reference_features.astype(np.float32)
    )["cosine"]
    logger.debug(f"{similarity.shape=}")

    # postprocessing
    idx = metadata.index[metadata["sequence_number"] >= 0].tolist()
    _similarity = similarity[idx, :]
    _features = features[idx]
    _metadata = metadata.iloc[idx]
    if len(_metadata) > 0:
        logger.info("Starting identification postprocessing.")
        mew_features = feature_top(_features, _metadata, _similarity, "top_score")
        features[idx] = mew_features
        similarity = similarity_measure(
            features.astype(np.float32), reference_features.astype(np.float32)
        )["cosine"]

    query_metadata = pd.DataFrame(
        {
            "path": metadata["image_path"],
            "identity": [-1] * len(metadata["image_path"]),
            "split": ["test"] * len(metadata["image_path"]),
        }
    )
    database_metadata = pd.DataFrame(
        {
            "path": reference_image_paths,
            "identity": reference_class_ids,
            "split": ["train"] * len(reference_class_ids),
        }
    )

    logger.info(f"Starting loftr prediction.")
    predicted_idx, keypoints = get_merged_predictions(
        query_metadata,
        database_metadata,
        similarity,
        top_k=top_k,
        k_range=int(os.environ["LOFTR_K_RANGE"]),
        thr_range=int(os.environ["LOFTR_THRESHOLD_RANGE"]),
        threshold=float(os.environ["LOFTR_CONFIDENCE_THRESHOLD"]),
        num_kp=10,
        identities=True,
    )

    pred_image_paths = []
    pred_class_ids = []
    scores = []
    for query_idx, reference_idxs in enumerate(predicted_idx):
        if len(reference_idxs) < top_k:
            diff = top_k - len(reference_idxs)
            reference_idxs.extend([reference_idxs[-1]] * diff)

        _pred_image_paths = [reference_image_paths[idx] for idx in reference_idxs]
        _pred_class_ids = [int(reference_class_ids[idx]) for idx in reference_idxs]
        _scores = np.clip(similarity[query_idx, reference_idxs], 0, 1).astype(float).tolist()

        pred_image_paths.append(_pred_image_paths)
        pred_class_ids.append(_pred_class_ids)
        scores.append(_scores)

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
        "keypoints": keypoints,
    }
    return output
