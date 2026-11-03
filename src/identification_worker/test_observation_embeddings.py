"""Test cache selection without loading a GPU or downloading model weights."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

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
        self.encoder = Mock()
        self.encoder.specification = {"dimension": 2, "model_state_sha256": "model-v1"}
        self.encoder.encode.return_value = [0.1, 0.9]
        self.factory = Mock(return_value=self.encoder)
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
        return compute_manifest(self.path, self.factory)

    def test_cache_hit_skips_only_matching_observation(self):
        """Multiple observations on a shared media file still have separate cache keys."""
        self.manifest["cached"] = {fingerprint(self.encoder.specification): [f"1:{fingerprint(self.specification)}"]}
        result = self.run_manifest()
        self.assertEqual((result["computed"], result["cached"]), (1, 1))
        self.assertEqual(self.encoder.encode.call_args.args[0]["observation_id"], 2)

    def test_new_weights_or_input_require_recomputation(self):
        """Never reuse vectors from a different model state or input revision."""
        self.manifest["cached"] = {fingerprint(self.encoder.specification): ["1:old-input"]}
        self.assertEqual(self.run_manifest()["computed"], 2)
        self.encoder.specification = {**self.encoder.specification, "model_state_sha256": "model-v2"}
        self.assertEqual(self.run_manifest()["computed"], 2)

    def test_changed_source_fails_even_for_cached_input(self):
        """Fail if media changed between preparation and execution."""
        Path(self.manifest["observations"][0]["image_path"]).write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "Source media changed"):
            self.run_manifest()
        self.encoder.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
