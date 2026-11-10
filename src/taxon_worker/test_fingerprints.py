"""Exact hashing is file-local, independent of crop matching and Django."""

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from duplicates.fingerprints import HASH_VERSION, hash_files
from PIL import Image, PngImagePlugin


class ExactFingerprintTest(unittest.TestCase):
    def setUp(self):
        """Prepare actual media sources in a private temporary directory."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        Image.fromarray(np.zeros((40, 80, 3), dtype=np.uint8)).save(self.root / "a.png")

    def item(self, key, source="a.png", media_type="image", reference=""):
        """Describe one selected worker input, without any group or database knowledge."""
        return {
            "id": key,
            "source": source,
            "reference": reference,
            "media_type": media_type,
            "revision": 3,
            "name": source,
        }

    def test_single_selected_image_needs_no_crop_search(self):
        """The worker accepts one image and never extracts feature descriptors."""
        with patch("duplicates.engine.features", side_effect=AssertionError("Crop matching must not run")):
            result = hash_files([self.item(1)], self.root)
        self.assertEqual(result["version"], HASH_VERSION)
        self.assertEqual(len(result["items"][0]["file_sha256"]), 64)
        self.assertEqual(len(result["items"][0]["pixel_sha256"]), 64)
        self.assertEqual(result["items"][0]["reference_sha256"], "")

    def test_metadata_differences_do_not_change_pixel_hash(self):
        """Exact pixel equality survives metadata-only changes, while byte hashes distinguish them."""
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("description", "a different header")
        Image.open(self.root / "a.png").save(self.root / "b.png", pnginfo=metadata)
        first, second = hash_files([self.item(1), self.item(2, "b.png")], self.root)["items"]
        self.assertNotEqual(first["file_sha256"], second["file_sha256"])
        self.assertEqual(first["pixel_sha256"], second["pixel_sha256"])

    def test_video_reference_is_recorded_separately(self):
        """Reference image equality never substitutes for whole-video byte equality."""
        (self.root / "a.mp4").write_bytes(b"video one")
        (self.root / "b.mp4").write_bytes(b"video two")
        first, second = hash_files(
            [
                self.item(1, "a.mp4", "video", "a.png"),
                self.item(2, "b.mp4", "video", "a.png"),
            ],
            self.root,
        )["items"]
        self.assertNotEqual(first["file_sha256"], second["file_sha256"])
        self.assertEqual(first["reference_sha256"], second["reference_sha256"])
        self.assertEqual(first["pixel_sha256"], "")

    def test_one_missing_source_does_not_fail_other_selected_images(self):
        """Each input reports its own result, allowing partial progress within a batch."""
        first, second = hash_files([self.item(1, "missing.png"), self.item(2)], self.root)["items"]
        self.assertIn("error", first)
        self.assertTrue(second["pixel_sha256"])


class ExactHashDeliveryTest(unittest.TestCase):
    def setUp(self):
        """Use the actual Celery task entrypoint without a broker or database."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(
            os.environ,
            {
                "RABBITMQ_URL": "memory://",
                "REDIS_URL": "cache+memory://",
                "DUPLICATE_SCAN_ROOT": str(self.root / "private"),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        import duplicate_worker

        self.worker = duplicate_worker
        media_root = patch.object(duplicate_worker, "ROOT", self.root)
        media_root.start()
        self.addCleanup(media_root.stop)
        self.batch_uuid = str(uuid.uuid4())
        self.directory = self.root / "private" / "hash_batches" / self.batch_uuid
        self.directory.mkdir(parents=True)
        Image.fromarray(np.zeros((40, 80, 3), dtype=np.uint8)).save(self.root / "a.png")
        item = {"id": 1, "name": "a.png", "source": "a.png", "reference": "", "media_type": "image", "revision": 0}
        (self.directory / "manifest.json").write_text(json.dumps({"items": [item]}), encoding="utf-8")

    def test_worker_reports_progress_and_replays_completed_result(self):
        """At-least-once delivery must replay a callback without hashing the same batch again."""
        first = self.worker.hash_media_files.run(self.batch_uuid)
        progress = json.loads((self.directory / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["completed"], 1)
        result_time = (self.directory / "result.json").stat().st_mtime_ns
        with patch.object(self.worker, "hash_files", side_effect=AssertionError("Must reuse the completed result")):
            second = self.worker.hash_media_files.run(self.batch_uuid)
        self.assertEqual(first, second)
        self.assertEqual((self.directory / "result.json").stat().st_mtime_ns, result_time)
        self.assertTrue(self.worker.hash_media_files.acks_late)
        self.assertTrue(self.worker.hash_media_files.reject_on_worker_lost)


if __name__ == "__main__":
    unittest.main()
