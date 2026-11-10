"""Image-copy detection tests independent of Celery, Django, and classifier models."""

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from duplicates.engine import scan
from PIL import Image, PngImagePlugin


class DuplicateEngineTest(unittest.TestCase):
    def setUp(self):
        """Build deterministic textured images in an isolated directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(12)
        self.pixels = cv2.GaussianBlur(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8), (7, 7), 0)
        for _ in range(100):
            x, y = rng.integers([15, 15], [785, 585])
            cv2.circle(self.pixels, (int(x), int(y)), int(rng.integers(3, 12)), (20, 180, 240), -1)

    def item(self, key, pixels=None, name=None):
        """Write one image and return its manifest reference."""
        name = name or f"{key}.png"
        Image.fromarray(self.pixels if pixels is None else pixels).save(self.root / name)
        return {"id": key, "source": name, "name": name, "media_type": "image"}

    def test_pixel_equality_groups_triplicate_with_different_metadata(self):
        """A byte-identical subgroup must not duplicate the complete pixel group."""
        first = self.item(1)
        shutil.copyfile(self.root / "1.png", self.root / "2.png")
        second = {**first, "id": 2, "source": "2.png"}
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Description", "Different metadata")
        Image.fromarray(self.pixels).save(self.root / "3.png", pnginfo=metadata)
        third = {**first, "id": 3, "source": "3.png"}
        result = scan([first, second, third], self.root, include_crops=False)
        self.assertEqual(len(result["proposals"]), 1)
        self.assertEqual(result["proposals"][0]["members"], [1, 2, 3])
        self.assertEqual(result["proposals"][0]["kind"], "pixel_exact")

    def test_crop_localization(self):
        """Localize a real crop and preserve its direction and normalized coordinates."""
        source = self.item(1)
        crop = self.item(2, self.pixels[100:500, 150:650])
        result = scan([source, crop], self.root)
        self.assertEqual(len(result["proposals"]), 1)
        match = result["proposals"][0]
        self.assertEqual(match["members"], [2, 1])
        np.testing.assert_allclose(match["evidence"]["polygon"][0], [150 / 800, 100 / 600], atol=0.004)

    def test_matching_background_with_changed_object_is_rejected(self):
        """Matching SIFT points on the background must not hide a changed object."""
        source = self.item(1)
        altered = self.pixels[100:500, 150:650].copy()
        altered[170:230, 220:280] = [255, 0, 0]
        crop = self.item(2, altered)
        self.assertEqual(scan([source, crop], self.root)["proposals"], [])

    def test_video_byte_equality_without_decoding_preview(self):
        """Video detection uses actual media bytes, never a representative image."""
        (self.root / "a.mp4").write_bytes(b"identical video bytes")
        (self.root / "b.mp4").write_bytes(b"identical video bytes")
        items = [
            {"id": index, "source": name, "name": name, "media_type": "video"}
            for index, name in enumerate(["a.mp4", "b.mp4"], 1)
        ]
        result = scan(items, self.root)
        self.assertEqual(result["proposals"][0]["kind"], "byte_exact")

    def test_untextured_images_can_still_match_exactly(self):
        """Lack of SIFT features must not fail exact duplicate detection."""
        items = [self.item(key, np.zeros((100, 100, 3), dtype=np.uint8)) for key in (1, 2)]
        self.assertEqual(scan(items, self.root)["proposals"][0]["kind"], "pixel_exact")

    def test_paths_outside_media_root_are_rejected(self):
        """An invalid path must not become a duplicate candidate."""
        valid = self.item(1)
        result = scan([valid, {**valid, "id": 2, "source": "../outside.png"}], self.root)
        self.assertIn("error", result["items"][1])
        self.assertEqual(result["proposals"], [])

    def test_resized_crop(self):
        """A uniformly resized crop remains discoverable without a whole-image hash match."""
        source = self.item(1)
        pixels = cv2.resize(self.pixels[100:500, 150:650], (375, 300), interpolation=cv2.INTER_AREA)
        crop = self.item(2, pixels)
        matches = scan([source, crop], self.root)["proposals"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["kind"], "crop")

    def test_high_bit_depth_is_not_silently_reduced(self):
        """A 16-bit raster cannot be labeled pixel-exact after an implicit 8-bit conversion."""
        first = self.item(1, np.zeros((100, 100, 3), dtype=np.uint8))
        Image.fromarray(np.zeros((100, 100), dtype=np.uint16)).save(self.root / "16bit.png")
        second = {**first, "id": 2, "source": "16bit.png"}
        result = scan([first, second], self.root)
        self.assertIn("pixel_error", result["items"][1])
        self.assertEqual(result["proposals"], [])


if __name__ == "__main__":
    unittest.main()
