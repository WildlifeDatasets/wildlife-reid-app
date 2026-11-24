"""Persistent MegaDetector weights and YOLOv5 code for the taxon worker."""

import shutil

from wildlife_model_cache import (
    SHARED_DATA_DIR,
    WEIGHTS_DIR,
    cache_checkpoint,
    download_checkpoint_if_missing,
    has_checkpoint,
)

MEGADETECTOR_CHECKPOINT_PATH = WEIGHTS_DIR / "megadetector" / "md_v5a.0.0.pt"
TORCH_HUB_CACHE_DIR = SHARED_DATA_DIR / "cache" / "torch" / "hub"
MEGADETECTOR_URL = "https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"


def get_megadetector_checkpoint():
    """Reuse a previously cached MegaDetector checkpoint when initializing the new cache."""
    legacy_path = SHARED_DATA_DIR / "models" / "megadetector" / MEGADETECTOR_CHECKPOINT_PATH.name
    if has_checkpoint(legacy_path):
        return cache_checkpoint(
            MEGADETECTOR_CHECKPOINT_PATH, lambda path: shutil.copyfile(legacy_path, path), operation="Checkpoint copy"
        )
    return download_checkpoint_if_missing(MEGADETECTOR_URL, MEGADETECTOR_CHECKPOINT_PATH)
