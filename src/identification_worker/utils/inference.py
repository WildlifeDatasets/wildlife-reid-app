import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fgvc.core.models import get_model
from fgvc.core.training import predict
from fgvc.datasets import PredictionDataset, get_dataloaders
from fgvc.utils.utils import set_cuda_device

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
