"""Persistent MegaDetector weights and Torch Hub cache for the taxon worker."""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("app")

# Keep weights and YOLOv5 code in the shared volume to shorten image builds and
# preserve the download cache across image rebuilds and Docker/container restarts.
SHARED_DATA_DIR = Path(os.getenv("SHARED_DATA_PATH", "/shared_data"))
MEGADETECTOR_CHECKPOINT_PATH = SHARED_DATA_DIR / "models" / "megadetector" / "md_v5a.0.0.pt"
TORCH_HUB_CACHE_DIR = SHARED_DATA_DIR / "cache" / "torch" / "hub"
MEGADETECTOR_URL = "https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"


def download_checkpoint_if_missing(url: str, checkpoint_path: Path) -> Path:
    """Reuse a non-empty checkpoint or download and atomically publish a complete file."""
    import requests

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        logger.info("Downloading model checkpoint %s to %s", url, checkpoint_path)
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            expected_size = int(response.headers.get("content-length", 0))
            with tempfile.NamedTemporaryFile(
                dir=checkpoint_path.parent, prefix=checkpoint_path.name + ".", suffix=".part", delete=False
            ) as output:
                temporary_path = Path(output.name)
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
                    downloaded_size += len(chunk)
            if downloaded_size == 0 or (expected_size and downloaded_size != expected_size):
                raise ValueError(f"Incomplete model download: got {downloaded_size} bytes, expected {expected_size}")
        temporary_path.replace(checkpoint_path)
        return checkpoint_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
