"""Encode observation crops independently of the legacy reference-image database."""

import hashlib
import json
from pathlib import Path


def fingerprint(specification):
    """Use the same canonical JSON hash as the API's embedding service."""
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def compute_manifest(manifest_path):
    """Encode missing global and ALIKED features in the shared job manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import pandas as pd

    from .inference_identification import del_models, encode_images, get_segmentation_model

    config = manifest["feature_config"]
    cached = set(manifest["cached"].get(manifest.get("cached_profile"), []))
    rows = [
        row for row in manifest["observations"]
        if f"{row['observation_id']}:{row['input_fingerprint']}" not in cached
    ]
    if config["weights_sha256"] and file_digest(manifest["weights_path"]) != config["weights_sha256"]:
        raise ValueError("The model checkpoint changed after this run was prepared.")
    for row in rows:
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
            features = encode_images(metadata, manifest["model_source"], manifest["weights_path"])
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
