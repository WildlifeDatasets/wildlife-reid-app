"""Test the detection-only contract without downloading weights or requiring a GPU."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from bbox_detection import detect_file


class DetectionContractTest(unittest.TestCase):
    def setUp(self):
        self.detector = SimpleNamespace(
            SAM3_FALLBACK_CONF=0.5,
            detect_animals_with_megadetector=Mock(return_value=None),
            detect_animals_with_sam3=Mock(return_value=None),
        )
        self.options = {"detector": "auto", "confidence": 0.5}
        modules = patch.dict(sys.modules, {"detection_utils": SimpleNamespace(inference_detection=self.detector)})
        modules.start()
        self.addCleanup(modules.stop)
        image = patch("bbox_detection.cv2.imread", return_value=np.zeros((100, 200, 3), dtype=np.uint8))
        image.start()
        self.addCleanup(image.stop)

    def test_empty_success_is_explicit(self):
        result = detect_file("example.jpg", self.options)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["detections"], [])
        self.detector.detect_animals_with_sam3.assert_called_once()
        self.assertTrue(self.detector.detect_animals_with_sam3.call_args.kwargs["strict"])

    def test_sam_failure_is_not_a_successful_empty_detection(self):
        self.detector.detect_animals_with_sam3.side_effect = RuntimeError("GPU unavailable")
        with self.assertRaises(RuntimeError):
            detect_file("example.jpg", self.options)

    def test_person_detection_does_not_suppress_animal_fallback(self):
        self.detector.detect_animals_with_megadetector.return_value = [
            {"class": "person", "confidence": 0.99, "bbox": [0, 0, 200, 100]}
        ]
        self.detector.detect_animals_with_sam3.return_value = [
            {"class": "animal", "confidence": 0.8, "bbox": [20, 10, 80, 90]}
        ]
        result = detect_file("example.jpg", self.options)
        self.assertEqual(result["detector"], "sam3")
        self.assertEqual(result["detections"][0]["bbox"], [0.1, 0.1, 0.4, 0.9])

    def test_forced_sam_does_not_run_megadetector(self):
        detect_file("example.jpg", {**self.options, "detector": "sam3"})
        self.detector.detect_animals_with_megadetector.assert_not_called()

    def test_megadetector_only_does_not_run_sam(self):
        detect_file("example.jpg", {**self.options, "detector": "megadetector"})
        self.detector.detect_animals_with_sam3.assert_not_called()

    def test_path_outside_shared_media_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_file("../../private.jpg", self.options)

    def test_unreadable_image_is_an_error(self):
        with patch("bbox_detection.cv2.imread", return_value=None), self.assertRaises(ValueError):
            detect_file("example.jpg", self.options)


if __name__ == "__main__":
    unittest.main()
