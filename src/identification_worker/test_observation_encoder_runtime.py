"""Exercise the real timm crop encoder using local random weights, without downloads."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from utils.observation_embeddings import TimmObservationEncoder, fingerprint


class TimmObservationEncoderTests(unittest.TestCase):
    def test_local_weights_crop_and_profile_changes(self):
        """Encode two crops separately and identify a changed checkpoint at the same path."""
        import timm
        import torch
        from PIL import Image

        torch.set_num_threads(1)
        with TemporaryDirectory() as directory, patch.object(torch.cuda, "is_available", return_value=False):
            root = Path(directory)
            checkpoint = root / "weights.pth"
            model = timm.create_model("resnet18", pretrained=False, num_classes=0)
            torch.save({"model": model.state_dict()}, checkpoint)
            encoder = TimmObservationEncoder("resnet18", str(checkpoint))
            image_path = root / "source.png"
            image = Image.new("RGB", (128, 64), "white")
            image.paste(Image.new("RGB", (64, 64), "black"), (0, 0))
            image.save(image_path)
            vectors = [
                encoder.encode(
                    {"image_path": str(image_path), "input_specification": {"bbox_cxcywh": [cx, 0.5, 0.5, 1.0]}}
                )
                for cx in (0.25, 0.75)
            ]
            self.assertEqual(len(vectors[0]), 512)
            self.assertNotEqual(vectors[0], vectors[1])
            old_key = fingerprint(encoder.specification)
            same = TimmObservationEncoder("resnet18", str(checkpoint))
            self.assertEqual(fingerprint(same.specification), old_key)
            with torch.no_grad():
                model.conv1.weight.add_(0.01)
            torch.save({"model": model.state_dict()}, checkpoint)
            changed = TimmObservationEncoder("resnet18", str(checkpoint))
            self.assertNotEqual(fingerprint(changed.specification), old_key)


if __name__ == "__main__":
    unittest.main()
