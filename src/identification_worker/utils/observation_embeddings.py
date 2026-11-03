"""Encode observation crops independently of the legacy reference-image database."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def fingerprint(specification):
    """Use the same canonical JSON hash as the API's embedding service."""
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class TimmObservationEncoder:
    """Create a versioned RGB bbox-crop encoder with evaluation transforms."""

    def __init__(self, model_source, weights_path):
        import timm
        import torch
        from timm.data import create_transform, resolve_model_data_config

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(model_source, num_classes=0, pretrained=not bool(weights_path))
        if weights_path:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(checkpoint["model"])
        state_hash = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            state_hash.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            state_hash.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)
        self.specification = {
            "schema": 1,
            "model_source": model_source,
            "model_state_sha256": state_hash.hexdigest(),
            "dimension": int(self.model.num_features),
            "vector_normalization": "none",
            "preprocessing": {
                "implementation": "rgb-bbox-timm-eval-v1",
                "data_config": data_config,
                "orientation_rotation": False,
                "segmentation": False,
            },
            "libraries": {name: version(name) for name in ("torch", "torchvision", "timm", "Pillow")},
        }
        # Normalize tuples to lists so the on-disk profile has exactly this hash.
        self.specification = json.loads(json.dumps(self.specification))
        self.model.eval()

    def encode(self, row):
        """Encode exactly one observation; absent boxes use the entire image."""
        from PIL import Image

        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        bbox = row["input_specification"]["bbox_cxcywh"]
        if all(value is not None for value in bbox):
            cx, cy, width, height = bbox
            image_width, image_height = image.size
            left = max(0, min(image_width, int((cx - width / 2) * image_width)))
            top = max(0, min(image_height, int((cy - height / 2) * image_height)))
            right = max(0, min(image_width, int((cx + width / 2) * image_width)))
            bottom = max(0, min(image_height, int((cy + height / 2) * image_height)))
            if right <= left or bottom <= top:
                raise ValueError("Observation bbox is empty after clipping to the image.")
            image = image.crop((left, top, right, bottom))
        with self.torch.inference_mode():
            self.model.to(self.device)
            vector = self.model(self.transform(image).unsqueeze(0).to(self.device))
        vector = vector.detach().cpu().reshape(-1).tolist()
        if len(vector) != self.specification["dimension"]:
            raise ValueError("Model output dimension differs from its profile.")
        return vector


def compute_manifest(manifest_path, encoder_factory=TimmObservationEncoder):
    """Load the actual encoder profile, skip cached inputs and write only new vectors."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoder = encoder_factory(manifest["model_source"], manifest["weights_path"])
    profile = encoder.specification
    cached = set(manifest["cached"].get(fingerprint(profile), []))
    embeddings = []
    skipped = 0
    for row in manifest["observations"]:
        if fingerprint(row["input_specification"]) != row["input_fingerprint"]:
            raise ValueError("Invalid observation input fingerprint.")
        digest = hashlib.sha256()
        with open(row["image_path"], "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["input_specification"]["source_sha256"]:
            raise ValueError("Source media changed after the embedding job was prepared; submit a new job.")
        key = f"{row['observation_id']}:{row['input_fingerprint']}"
        if key in cached:
            skipped += 1
            continue
        embeddings.append(
            {
                "observation_id": row["observation_id"],
                "input_fingerprint": row["input_fingerprint"],
                "vector": encoder.encode(row),
            }
        )
    output_path = manifest_path.with_name("output.json")
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"profile": profile, "embeddings": embeddings}, allow_nan=False), encoding="utf-8"
    )
    temporary_path.replace(output_path)
    return {"output_path": str(output_path), "computed": len(embeddings), "cached": skipped}
