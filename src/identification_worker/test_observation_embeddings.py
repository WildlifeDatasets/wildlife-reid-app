"""Test cache selection without loading a GPU or downloading model weights."""

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from utils.observation_embeddings import compute_manifest, fingerprint


class ObservationEncoderTests(unittest.TestCase):
    def setUp(self):
        """Prepare two observations and a deterministic encoder test double."""
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        image = root / "image.jpg"
        image.write_bytes(b"image-content")
        self.specification = {"source_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
        self.manifest = {
            "model_source": "test",
            "weights_path": "",
            "cached": {},
            "observations": [
                {
                    "observation_id": index,
                    "image_path": str(image),
                    "input_specification": self.specification,
                    "input_fingerprint": fingerprint(self.specification),
                }
                for index in (1, 2)
            ],
        }
        self.path = root / "input.json"

    def run_manifest(self):
        """Write the manifest and execute the production cache selection loop."""
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return compute_manifest(self.path)

    def test_shared_run_uses_wildfusion_encoder_only_for_missing_observations(self):
        """Use the existing global and ALIKED extractor with independent crop paths."""
        config = {"weights_sha256": "", "segmentation": "sam3", "revision": "test"}
        self.manifest["feature_config"] = config
        self.manifest["cached_profile"] = "shared-profile"
        for row in self.manifest["observations"]:
            row["input_specification"] = {**self.specification, "bbox_cxcywh": [0.5, 0.5, 0.3, 0.3]}
            row["input_fingerprint"] = fingerprint(row["input_specification"])
            row["feature_image"] = str(self.path.with_name(f"crop-{row['observation_id']}.png"))
        cached, missing = self.manifest["observations"]
        self.manifest["cached"] = {"shared-profile": [f"1:{cached['input_fingerprint']}"]}
        local = {"keypoints": [[1, 2]], "descriptors": [[0.2, 0.5]], "image_size": [10, 10]}
        inference = Mock()
        inference.get_segmentation_model.return_value = "sam3"

        def encode(metadata, model, weights):
            self.assertEqual(metadata["observation_id"].tolist(), [2])
            self.assertEqual(metadata["bbox_w"].tolist(), [0.3])
            metadata["image_path"] = metadata["feature_image"]
            return [([0.1, 0.9], local)]

        inference.encode_images.side_effect = encode
        with patch.dict(sys.modules, {"utils.inference_identification": inference}):
            result = self.run_manifest()
        response = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
        self.assertEqual(response["profile"]["feature_config"], config)
        self.assertEqual(response["embeddings"][0]["local_features"], local)
        self.assertEqual(response["embeddings"][0]["feature_image"], missing["feature_image"])


    def test_changed_source_is_rejected_before_encoding(self):
        """Keep features tied to the bytes submitted for the observation."""
        self.manifest["feature_config"] = {"weights_sha256": "", "segmentation": "sam3", "revision": "test"}
        Path(self.manifest["observations"][0]["image_path"]).write_bytes(b"replacement")
        inference = Mock()
        with patch.dict(sys.modules, {"utils.inference_identification": inference}):
            with self.assertRaisesRegex(ValueError, "Source media changed"):
                self.run_manifest()
        inference.encode_images.assert_not_called()


if __name__ == "__main__":
    unittest.main()
