"""Encode observation crops independently of the legacy reference-image database."""

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingRunStopped(Exception):
    """The API asked this feature worker to stop."""


def fingerprint(specification):
    """Use the same canonical JSON hash as the API's embedding service."""
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def compute_manifest(manifest_path, report_progress=None):
    """Encode missing global and ALIKED features in the shared job manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached = set(manifest["cached"].get(manifest.get("cached_profile"), []))
    rows = [
        row for row in manifest["observations"]
        if f"{row['observation_id']}:{row['input_fingerprint']}" not in cached
    ]
    marker = manifest_path.with_name("cancelled")
    last_label = None
    next_report = 0
    crop_count = 0

    def checkpoint(label, completed=None, total=None):
        nonlocal last_label, next_report, crop_count
        now = time.monotonic()
        if label == "Segmenting observations":
            crop_count = completed + 1
        if label == last_label and now < next_report:
            return
        if marker.is_file():
            raise EmbeddingRunStopped
        last_label = label
        next_report = now + 2
        if report_progress:
            report_progress({"label": label, "completed": completed, "total": total})

    try:
        return _encode_manifest(manifest_path, manifest, rows, checkpoint)
    except EmbeddingRunStopped:
        for row in rows[:crop_count]:
            try:
                Path(row["feature_image"]).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove crop for stopped observation %s", row["observation_id"], exc_info=True)
        raise


def _encode_manifest(manifest_path, manifest, rows, checkpoint):
    import pandas as pd

    from .inference_identification import del_models, encode_images, get_segmentation_model

    config = manifest["feature_config"]
    checkpoint("Checking source files", 0, len(rows))
    if config["weights_sha256"] and file_digest(manifest["weights_path"]) != config["weights_sha256"]:
        raise ValueError("The model checkpoint changed after this run was prepared.")
    for index, row in enumerate(rows):
        checkpoint("Checking source files", index, len(rows))
        if fingerprint(row["input_specification"]) != row["input_fingerprint"]:
            raise ValueError("Invalid observation input fingerprint.")
        if file_digest(row["image_path"]) != row["input_specification"]["source_sha256"]:
            raise ValueError("Source media changed after this run was prepared.")
    data = []
    if rows:
        try:
            if get_segmentation_model() != config["segmentation"]:
                raise ValueError("The worker segmentation backend does not match the configured feature profile.")
            metadata = pd.DataFrame(
                {
                    "image_path": row["image_path"],
                    "feature_image": row["feature_image"],
                    "observation_id": row["observation_id"],
                    "label": "",
                    **dict(zip(("bbox_cx", "bbox_cy", "bbox_w", "bbox_h"), row["input_specification"]["bbox_cxcywh"])),
                }
                for row in rows
            )
            features = encode_images(
                metadata, manifest["model_source"], manifest["weights_path"], checkpoint=checkpoint,
            )
            data = [
                {
                    "observation_id": row["observation_id"],
                    "input_fingerprint": row["input_fingerprint"],
                    "vector": vector,
                    "local_features": local,
                    "feature_image": str(path),
                }
                for row, (vector, local), path in zip(rows, features, metadata["image_path"], strict=True)
            ]
        finally:
            del_models()
    checkpoint("Writing features")
    dimension = len(data[0]["vector"]) if data else manifest["cached_dimension"]
    profile = {"model_source": manifest["model_source"], "feature_config": config, "dimension": dimension}
    output = manifest_path.with_name("output.json")
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"profile": profile, "embeddings": data}, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    return {"output_path": str(output)}


def file_digest(path):
    """Hash a file without loading it into memory."""
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
