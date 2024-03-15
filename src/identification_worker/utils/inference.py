import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import timm
import torch
from wildlife_tools import realize
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity

from fgvc.utils.utils import set_cuda_device

from .postprocessing import feature_top

logger = logging.getLogger("app")
device = set_cuda_device("0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class CarnivoreDataset(WildlifeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image_paths(self):
        """Return the image paths."""
        return self.metadata["path"].astype(str).values


def load_model(model_name, model_checkpoint=""):
    """Load the model from the given model name and checkpoint."""
    # load model checkpoint
    model = timm.create_model(model_name, num_classes=0, pretrained=True)
    if model_checkpoint:
        if not torch.cuda.is_available():
            model_ckpt = torch.load(model_checkpoint, map_location=torch.device("cpu"))["model"]
        else:
            model_ckpt = torch.load(model_checkpoint)["model"]
        model.load_state_dict(model_ckpt)

    model = model.to(device).eval()
    return model


MODEL = load_model(
    # "hf-hub:BVRA/MegaDescriptor-T-224",
    "hf-hub:strakajk/Lynx-MegaDescriptor-T-224"
    # model_checkpoint="/identification_worker/resources/MegaDescriptor-T-224-c-15-01_18-00-02.pth"
)


def encode_images(metadata: pd.DataFrame) -> np.ndarray:
    """Create feature vectors from given images."""
    logger.info("Creating DataLoaders.")

    config = {
        "method": "TransformTimm",
        "input_size": np.max(MODEL.default_cfg["input_size"][1:]),
        "is_training": False,
        "auto_augment": "rand-m10-n2-mstd1",
    }
    transform = realize(config)

    logger.debug(f"encode_images-metadata: {metadata}")
    new_image_paths = []
    image_paths = metadata.image_path
    for image_path in image_paths:
        new_image_path = Path(image_path.replace("/images/", "/masked_images/"))
        if not new_image_path.is_file():
            new_image_path = image_path
        new_image_paths.append(str(new_image_path))
    metadata["image_path"] = new_image_paths
    # metadata['masked_image_path'] = np.where(
    #     metadata['masked_image_path'].isna(),
    #     metadata['image_path'], metadata['masked_image_path']
    # )

    dataset = CarnivoreDataset(
        metadata=metadata,
        root="",
        transform=transform,
        img_load="full",
        col_path="image_path",
        col_label="label",
    )

    logger.info("Running inference.")
    extractor = DeepFeatures(MODEL, batch_size=4, num_workers=1, device=device)
    features = extractor(dataset)

    return features


def _get_top_predictions(similarity: np.ndarray, paths: list, identities: list, top_k: int = 1):
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
    top_k: int = 1,
) -> Tuple[list, np.ndarray, np.ndarray]:
    """Compare input feature vectors with the reference feature vectors and make predictions."""
    assert len(reference_features) == len(reference_image_paths)
    assert len(reference_features) == len(reference_class_ids)

    similarity_measure = CosineSimilarity()
    similarity = similarity_measure(
        features.astype(np.float32), reference_features.astype(np.float32)
    )["cosine"]

    # postprocessing
    idx = metadata.index[metadata["sequence_number"] >= 0].tolist()
    _similarity = similarity[idx, :]
    _features = features[idx]
    _metadata = metadata.iloc[idx]
    if len(_metadata) > 0:
        mew_features = feature_top(_features, _metadata, _similarity, "top_score")
        features[idx] = mew_features
        similarity = similarity_measure(
            features.astype(np.float32), reference_features.astype(np.float32)
        )["cosine"]

    top_predictions = _get_top_predictions(
        similarity, reference_image_paths, reference_class_ids, top_k=top_k
    )

    pred_image_paths = []
    pred_class_ids = []
    scores = []

    for row in top_predictions:
        pred_class_ids.append(row[0])
        pred_image_paths.append(row[1])
        scores.append(row[2])

    # return path to original image
    _pred_image_paths = []
    for paths in pred_image_paths:
        _pred_image_paths.append([p.replace("/masked_images/", "/images/") for p in paths])
    pred_image_paths = _pred_image_paths

    return pred_image_paths, np.array(pred_class_ids), np.array(scores)
