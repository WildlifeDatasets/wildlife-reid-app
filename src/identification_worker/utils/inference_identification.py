import ast
import logging
import os
import traceback
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm
from wildlife_tools import realize
from wildlife_tools.data import FeatureDataset, WildlifeDataset
from wildlife_tools.features import AlikedExtractor, DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.pairwise.collectors import CollectAll
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue

from .wildfusion_utils import SimilarityPipelineExtended, WildFusionExtended

try:
    from ..infrastructure_utils import mem
except ImportError:
    print(traceback.format_exc())
    from infrastructure_utils import mem

logger = logging.getLogger("app")
# DEVICE = set_cuda_device("1") if torch.cuda.is_available() else "cpu"
DEVICE = mem.get_torch_cuda_device_if_available(0)  # TODO set device to 1
logger.setLevel(logging.DEBUG)
logger.info(f"Using device: {DEVICE}")

IDENTIFICATION_MODELS = None
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
    global IDENTIFICATION_MODELS

    if IDENTIFICATION_MODELS is not None:
        return
    IDENTIFICATION_MODELS = None

    logger.debug("Before identification model.")
    logger.debug(f"{mem.get_vram(DEVICE)}     {mem.get_ram()}")
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


def get_sam_model() -> SamPredictor:
    """Load the SAM model if not loaded before."""
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

        mem.wait_for_gpu_memory(0.5)
        logger.info(f"Initializing SAM model ({model_version}) and loading pre-trained checkpoint.")
        _checkpoint_path = Path(f"/root/resources/{model_zoo[model_version]}.pth").expanduser()
        SAM = sam_model_registry[model_version](checkpoint=str(_checkpoint_path))
        SAM.to(device=DEVICE)
        SAM_PREDICTOR = SamPredictor(SAM)
    logger.debug(f"After segmentation model: {mem.get_vram(DEVICE)}     {mem.get_ram()}")
    return SAM_PREDICTOR


def del_identification_model():
    """Release the identification model."""
    global IDENTIFICATION_MODELS
    IDENTIFICATION_MODELS = None
    torch.cuda.empty_cache()


def del_sam_model():
    """Release the SAM model."""
    global SAM
    global SAM_PREDICTOR
    SAM = None
    SAM_PREDICTOR = None
    torch.cuda.empty_cache()


def init_models(identification_model_path):
    """Initialize identification and segmentation models."""
    # get_identification_model(os.environ["IDENTIFICATION_MODEL_VERSION"])
    get_identification_model(identification_model_path)
    get_sam_model()


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


def segment_animal(image_path: str, bbox: list, border: float = 0.25) -> np.ndarray:
    """Segment an animal in a given image using SAM model."""
    global SAM_PREDICTOR

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # logger.debug("Running segmentation inference.")
    SAM_PREDICTOR.set_image(image)
    sam_input_box = np.array([int(point) for point in bbox])
    # logger.debug(f"{sam_input_box=}")

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
    for row_idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Masking images"):
        image_path = row["image_path"]
        # detection_results = ast.literal_eval(
        #    ast.literal_eval(row["detection_results"])["detection_results"])
        if row["detection_results"] is None:
            logger.debug(
                f"No detection results for image: {image_path}, row['detection_results'] is None."
            )
            masked_paths.append(str(image_path))
            continue
        detection_results = ast.literal_eval(row["detection_results"])
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

        masked_paths.append(str(save_path))

    metadata["image_path"] = masked_paths
    del_sam_model()
    return metadata


def encode_images(metadata: pd.DataFrame, identification_model_path:str) -> list:
    """Create feature vectors from given images."""
    global IDENTIFICATION_MODELS
    get_identification_model(identification_model_path)
    metadata = mask_images(metadata)
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


def _get_top_predictions(similarity: np.ndarray, paths: list, identities: list, top_k: int = 1):
    """Get top-k predictions from similarity matrix."""
    top_results = []
    for row_idx, row in enumerate(similarity):
        top_names = []
        top_paths = []
        top_scores = []
        top_idx = []

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
            top_idx.append(sort_idx[i])
            if len(top_names) == top_k:
                break

        if len(top_names) < top_k:
            diff = top_k - len(top_names)
            top_names.extend([top_names[-1]] * diff)
            top_paths.extend([top_paths[-1]] * diff)
            top_scores.extend([top_scores[-1]] * diff)
            top_idx.extend([top_idx[-1]] * diff)
        top_results.append((top_names, top_paths, top_scores, top_idx))

    return top_results


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
    calibration_aliked_features, calibration_mega_features = prepare_feature_types(
        calibrated_features
    )
    calibration_mega_features = FeatureDataset(calibration_mega_features, calibration_metadata)
    calibration_aliked_features = FeatureDataset(calibration_aliked_features, calibration_metadata)

    # calibrate models before identification
    IDENTIFICATION_MODELS["mega"].fit_calibration(
        calibration_mega_features, calibration_mega_features
    )
    IDENTIFICATION_MODELS["aliked"].fit_calibration(
        calibration_aliked_features, calibration_aliked_features
    )


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
    assert (
        target in valid_targets
    ), f"Invalid target: {target} for partial computation, valid targets: {valid_targets}"
    if target == "scores":
        assert pairs is not None, "Pairs must be provided for scores computation"
    logger.info(f"Starting identification of {len(query_metadata)} images.")

    global IDENTIFICATION_MODELS
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


def identify_from_similarity(similarity, database_metadata, top_k):
    """Get top-k predictions from similarity matrix."""
    # get top k predictions
    top_predictions = _get_top_predictions(
        similarity, database_metadata["path"], database_metadata["identity"], top_k=top_k
    )

    # reformat results
    pred_image_paths = []
    pred_class_ids = []
    scores = []
    result_idx = {}
    for qidx, row in enumerate(top_predictions):
        pred_class_ids.append(row[0])
        pred_image_paths.append(row[1])
        scores.append(np.clip(row[2], 0, 1).tolist())
        result_idx[qidx] = row[3]

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
    calibration_mega_features = FeatureDataset(
        database_mega_features[:cal_images], database_metadata[:cal_images]
    )
    calibration_aliked_features = FeatureDataset(
        database_aliked_features[:cal_images], database_metadata[:cal_images]
    )
    database_mega_features = FeatureDataset(database_mega_features, database_metadata)
    database_aliked_features = FeatureDataset(database_aliked_features, database_metadata)
    query_mega_features = FeatureDataset(query_mega_features, query_metadata)
    query_aliked_features = FeatureDataset(query_aliked_features, query_metadata)

    # calibrate models before identification
    IDENTIFICATION_MODELS["mega"].fit_calibration(
        calibration_mega_features, calibration_mega_features
    )
    IDENTIFICATION_MODELS["aliked"].fit_calibration(
        calibration_aliked_features, calibration_aliked_features
    )

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
    del IDENTIFICATION_MODELS

    output, result_idx = identify_from_similarity(similarity, database_metadata, top_k)

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

        _keypoints = get_keypoints(
            keypoint_matcher, keypoint_query_features, keypoint_database_features, max_kp=max_kp
        )
        keypoints.append(_keypoints)

    output["keypoints"] = keypoints
    return output
