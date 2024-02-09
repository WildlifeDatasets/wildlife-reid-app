import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from fgvc.core.models import get_model
from fgvc.core.training import predict
from fgvc.datasets import PredictionDataset, get_dataloaders
from fgvc.utils.utils import set_cuda_device

from wildlife_tools.data import WildlifeDataset
from wildlife_tools import realize
from wildlife_tools.data import SplitMetadata
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
import timm

logger = logging.getLogger("app")
device = set_cuda_device("0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class ModelWrapper(nn.Module):
    """A wrapper class for timm model that normalizes encoded features."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.encoder = model
        self.default_cfg = model.default_cfg

    def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run forward pass of encoder and generate feature vectors."""
        features = self.encoder(imgs)
        features = F.normalize(features, dim=1)
        return features

    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass."""
        features = self.forward_features(imgs)
        return {"features": features}


logger.info("Creating model and loading fine-tuned checkpoint.")
MODEL = get_model(
    "hf-hub:BVRA/wildlife-mega-L-384",
    pretrained=True,
    # checkpoint_path="/identification_worker/resources/wildlife-mega-L-384.pth",
)
MODEL = ModelWrapper(MODEL)
MODEL_MEAN = tuple(MODEL.default_cfg["mean"])
MODEL_STD = tuple(MODEL.default_cfg["std"])


def encode_images(image_paths: list) -> np.ndarray:
    """Create feature vectors from given images."""
    logger.info("Creating DataLoaders.")
    _, testloader, _, _ = get_dataloaders(
        None,
        image_paths,
        augmentations="vit_heavy",
        image_size=(384, 384),
        model_mean=MODEL_MEAN,
        model_std=MODEL_STD,
        batch_size=4,
        num_workers=1,
        dataset_cls=PredictionDataset,
    )

    logger.info("Running inference.")
    predict_output = predict(
        MODEL, testloader, device=device, valid_scores_fn=(lambda *args, **kwargs: {})
    )
    features = predict_output.preds["features"]

    return features


class CarnivoreDataset(WildlifeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image_paths(self):
        return self.metadata["path"].astype(str).values


def load_model(model_name, model_checkpoint=""):
    # load model checkpoint
    model = timm.create_model(model_name, num_classes=0, pretrained=True)
    if model_checkpoint:
        if not torch.cuda.is_available():
            model_ckpt = torch.load(model_checkpoint, map_location=torch.device('cpu'))["model"]
        else:
            model_ckpt = torch.load(model_checkpoint)["model"]
        model.load_state_dict(model_ckpt)

    model = model.to(device).eval()
    return model


MODEL = load_model(
    "hf-hub:BVRA/MegaDescriptor-T-224",
    "/identification_worker/resources/MegaDescriptor-T-224-c-15-01_18-00-02.pth"
)


def encode_images_wildlife(metadata: pd.DataFrame, split="") -> np.ndarray:
    """Create feature vectors from given images."""
    logger.info("Creating DataLoaders.")

    config = {
        'method': 'TransformTimm',
        'input_size': np.max(MODEL.default_cfg["input_size"][1:]),
        'is_training': False,
        'auto_augment': 'rand-m10-n2-mstd1',
    }
    transform = realize(config)

    dataset = CarnivoreDataset(
        metadata=metadata,
        root="",
        #split=SplitMetadata('split', split),
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
        top_results.append((top_names, top_paths, top_scores))

    return top_results

def identify_wildlife(
        features: np.ndarray,
        reference_features: np.ndarray,
        reference_image_paths: list,
        reference_class_ids: list,
        top_k: int = 1,
) -> Tuple[list, np.ndarray, np.ndarray]:
    """Compare input feature vectors with the reference feature vectors and make predictions."""
    assert len(reference_features) == len(reference_image_paths)
    assert len(reference_features) == len(reference_class_ids)

    similarity_measure = CosineSimilarity()
    similarity = similarity_measure(features.astype(np.float32), reference_features.astype(np.float32))['cosine']

    top_predictions = _get_top_predictions(similarity, reference_image_paths, reference_class_ids, top_k=top_k)

    pred_image_paths = []
    pred_class_ids = []
    scores = []

    for row in top_predictions:
        pred_class_ids.append(row[0])
        pred_image_paths.append(row[1])
        scores.append(row[2])

    return pred_image_paths, np.array(pred_class_ids), np.array(scores)


def identify(
        features: np.ndarray,
        reference_features: np.ndarray,
        reference_image_paths: list,
        reference_class_ids: list,
        top_k: int = 1,
) -> Tuple[list, np.ndarray, np.ndarray]:
    """Compare input feature vectors with the reference feature vectors and make predictions."""
    assert len(reference_features) == len(reference_image_paths)
    assert len(reference_features) == len(reference_class_ids)
    distance = 1 - torch.matmul(
        F.normalize(torch.tensor(features, dtype=torch.float32)),
        F.normalize(torch.tensor(reference_features, dtype=torch.float32)).T,
    )
    scores, idx = distance.topk(k=top_k, largest=False)
    scores, idx = scores.numpy(), idx.numpy()
    pred_image_paths = [[reference_image_paths[i] for i in row] for row in idx]
    pred_class_ids = np.array([[reference_class_ids[i] for i in row] for row in idx])

    return pred_image_paths, pred_class_ids, scores
