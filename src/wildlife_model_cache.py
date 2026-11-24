"""Locate and publish runtime model weights on the shared volume."""

import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("app")

SHARED_DATA_DIR = Path(os.getenv("SHARED_DATA_PATH", "/shared_data"))
WEIGHTS_DIR = SHARED_DATA_DIR / "models" / "weights"
SAM3_CHECKPOINT_PATH = WEIGHTS_DIR / "sam3" / "sam3.pt"


@contextmanager
def log_model_operation(operation: str, model):
    """Log the phase, outcome, and elapsed time without exposing request details."""
    started = time.monotonic()
    logger.info("%s started: %s", operation, model)
    try:
        yield
    except Exception as exc:
        logger.warning("%s failed after %.1f s (%s): %s", operation, time.monotonic() - started, type(exc).__name__, model)
        raise
    else:
        logger.info("%s completed in %.1f s: %s", operation, time.monotonic() - started, model)


def has_checkpoint(path: Path) -> bool:
    """Return whether a published checkpoint is available."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except FileNotFoundError:
        return False


def publish_checkpoint(temporary_path: Path, checkpoint_path: Path) -> None:
    """Publish a finished download and accept a concurrent publisher on SSHFS."""
    if has_checkpoint(checkpoint_path):
        return
    try:
        temporary_path.replace(checkpoint_path)
    except OSError:
        if not has_checkpoint(checkpoint_path):
            raise


def cache_checkpoint(checkpoint_path, fetch, *, on_download=None, operation="Model checkpoint download") -> Path:
    """Stage each attempt separately so readers see only completed downloads."""
    checkpoint_path = Path(checkpoint_path)
    if has_checkpoint(checkpoint_path):
        return checkpoint_path

    with log_model_operation(operation, checkpoint_path):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if on_download is not None:
            on_download()
        with tempfile.TemporaryDirectory(dir=checkpoint_path.parent) as staging:
            temporary_path = Path(staging) / checkpoint_path.name
            fetch(temporary_path)
            if not has_checkpoint(temporary_path):
                raise ValueError(f"Incomplete model download: {checkpoint_path}")
            publish_checkpoint(temporary_path, checkpoint_path)
    return checkpoint_path


def download_checkpoint_if_missing(url: str, checkpoint_path: Path) -> Path:
    """Stream an HTTP checkpoint into the shared cache."""

    def fetch(temporary_path):
        import requests

        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            expected_size = int(response.headers.get("content-length", 0))
            with temporary_path.open("wb") as output:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
                    downloaded_size += len(chunk)
            if not downloaded_size or (expected_size and downloaded_size != expected_size):
                raise ValueError(f"Incomplete model download: got {downloaded_size} bytes, expected {expected_size}")

    return cache_checkpoint(checkpoint_path, fetch)


def download_sam3_checkpoint_if_missing(*, on_download=None) -> Path:
    """Share the Hugging Face SAM3 checkpoint across both workers."""

    def fetch(temporary_path):
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_dir=temporary_path.parent)

    return cache_checkpoint(SAM3_CHECKPOINT_PATH, fetch, on_download=on_download)
