"""Pure metadata regressions; runnable without Django or a database."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "duplicate_metadata", Path(__file__).resolve().parents[1] / "duplicate_metadata.py"
)
metadata_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata_module)
comparable_metadata = metadata_module.comparable_metadata


class DuplicateMetadataTest(unittest.TestCase):
    def payload(self, upload):
        return {
            "full_image_path": f"/processing/{upload}/image.webp",
            "absolute_media_path": f"/processing/{upload}/image.webp",
            "preview_path": f"previews/{upload}/image.webp",
            "exif": repr({
                "SourceFile": f"/tmp/{upload}/image.jpg",
                "File:Directory": f"/tmp/{upload}",
                "File:FileModifyDate": upload,
                "File:FileAccessDate": upload,
                "File:FileInodeChangeDate": upload,
                "EXIF:Make": "Apple",
                "EXIF:DateTimeOriginal": "2026:09:13 21:56:16",
                "EXIF:GPSLatitude": 50.0,
            }),
            "detection_results": "same observations",
            "content_hash": "same-content",
        }

    def test_reimport_runtime_paths_and_file_times_are_equal(self):
        self.assertEqual(comparable_metadata(self.payload("first")), comparable_metadata(self.payload("second")))

    def test_real_exif_changes_remain_different(self):
        for field, replacement in [
            ("EXIF:Make", "Different camera"),
            ("EXIF:DateTimeOriginal", "2026:09:14 21:56:16"),
            ("EXIF:GPSLatitude", 51.0),
        ]:
            with self.subTest(field=field):
                a = self.payload("first")
                b = self.payload("second")
                exif = metadata_module.ast.literal_eval(b["exif"])
                exif[field] = replacement
                b["exif"] = exif
                self.assertNotEqual(comparable_metadata(a), comparable_metadata(b))

    def test_observation_and_unknown_metadata_are_preserved(self):
        for field in ["detection_results", "camera", "note"]:
            with self.subTest(field=field):
                a, b = self.payload("first"), self.payload("second")
                b[field] = "different meaningful information"
                self.assertNotEqual(comparable_metadata(a), comparable_metadata(b))

    def test_exif_string_encodings_compare_as_objects(self):
        a, b = self.payload("first"), self.payload("second")
        b["exif"] = json.dumps(metadata_module.ast.literal_eval(b["exif"]))
        self.assertEqual(comparable_metadata(a), comparable_metadata(b))

    def test_unparseable_exif_is_not_discarded(self):
        a, b = self.payload("first"), self.payload("second")
        a["exif"], b["exif"] = "not valid exif A", "not valid exif B"
        self.assertNotEqual(comparable_metadata(a), comparable_metadata(b))

    def test_comparison_does_not_modify_stored_metadata(self):
        value = self.payload("first")
        value["exif"] = metadata_module.ast.literal_eval(value["exif"])
        before = copy.deepcopy(value)
        comparable_metadata(value)
        self.assertEqual(value, before)


if __name__ == "__main__":
    unittest.main()
