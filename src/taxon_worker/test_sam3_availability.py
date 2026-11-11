"""Check actionable SAM3 failures without downloading weights or running inference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from detection_utils import inference_detection as detection
from detection_utils.errors import DetectorUnavailableError


class Sam3AvailabilityTest(unittest.TestCase):
    def setUp(self):
        """Force an uncached model and replace checkpoint access with a failure."""
        for name in ("SAM3", "SAM3_PREDICTOR"):
            patcher = patch.object(detection, name, None)
            patcher.start()
            self.addCleanup(patcher.stop)
        download = patch.object(detection, "download_sam3_checkpoint_if_missing")
        self.download = download.start()
        self.addCleanup(download.stop)
        self.image = np.zeros((10, 10, 3), dtype=np.uint8)

    def test_gated_weights_explain_access_and_token_requirements(self):
        """Do not hide a gated-repository error behind a generic GPU warning."""
        for status in (401, 403):
            with self.subTest(status=status):
                error = RuntimeError("private response details")
                error.response = SimpleNamespace(status_code=status)
                self.download.side_effect = error
                with self.assertRaises(DetectorUnavailableError) as raised:
                    detection.detect_animals_with_sam3(self.image, strict=True)
                message = str(raised.exception)
                self.assertIn("facebook/sam3", message)
                self.assertIn("HF_TOKEN", message)
                self.assertNotIn("private response details", message)
                self.assertIs(raised.exception.__cause__, error)

    def test_download_failure_has_distinct_safe_message(self):
        """Keep network failures distinct from permission errors without leaking details."""
        self.download.side_effect = OSError("private network details")
        with self.assertRaisesRegex(DetectorUnavailableError, "network access") as raised:
            detection.detect_animals_with_sam3(self.image, strict=True)
        self.assertNotIn("private network details", str(raised.exception))

    def test_import_keeps_its_non_strict_fallback(self):
        """Existing imports may continue without SAM3 when weights are unavailable."""
        self.download.side_effect = OSError("offline")
        self.assertIsNone(detection.detect_animals_with_sam3(self.image))

    def test_bbox_task_returns_actionable_error_and_cleans_up(self):
        """Carry a safe detector error through the Celery result to the API callback."""
        import worker

        message = "SAM3 weights are unavailable: Hugging Face denied access. Configure HF_TOKEN."
        with (
            patch.object(worker, "detect_file", side_effect=DetectorUnavailableError(message)),
            patch.object(detection, "del_detection_model"),
            patch.object(detection, "del_sam3_model") as cleanup,
        ):
            result = worker.redetect_bboxes.run("example.jpg", {"detector": "auto"})
        self.assertEqual(result, {"status": "error", "message": message})
        cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
