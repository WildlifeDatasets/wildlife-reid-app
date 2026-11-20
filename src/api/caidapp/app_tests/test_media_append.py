from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from caidapp.app_tests.factories import UploadedArchiveFactory
from caidapp.services.media_append import append_image
from caidapp.tasks import apply_added_image_inference, run_species_prediction_async
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.test import TestCase
from PIL import Image


class MediaAppendTest(TestCase):
    def test_added_image_queues_only_its_id_and_reuses_placeholder_on_result(self):
        """Dispatch after registration and apply a duplicate delivery only once."""
        with TemporaryDirectory() as root, self.settings(MEDIA_ROOT=root):
            upload = UploadedArchiveFactory(outputdir=str(Path(root) / "output"))
            content = BytesIO()
            Image.new("RGB", (8, 8), "red").save(content, format="JPEG")
            image = ContentFile(content.getvalue(), name="camera.jpg")
            with patch("caidapp.tasks.signature") as send, self.captureOnCommitCallbacks(execute=True):
                media = append_image(upload.pk, image)

            self.assertEqual(send.call_args.args, ("predict_prepared_media",))
            self.assertEqual(send.call_args.kwargs["kwargs"]["files"], [{
                "media_id": media.pk, "media_type": "image", "image_path": media.image_file.name,
            }])
            placeholder_id = media.observations.get().pk
            result = [{
                "media_id": media.pk,
                "detection_results": [
                    {"bbox": [1, 1, 7, 7], "size": [8, 8], "orientation": "unknown", "orientation_score": 0},
                    {"bbox": [2, 2, 6, 6], "size": [8, 8], "orientation": "unknown", "orientation_score": 0},
                ],
                "predicted_category": "Canis lupus",
                "predicted_category_raw": "Canis lupus",
                "predicted_prob_raw": 0.95,
            }]
            apply_added_image_inference(result, media.pk)
            apply_added_image_inference(result, media.pk)
            self.assertEqual(media.observations.count(), 2)
            first = media.observations.get(pk=placeholder_id)
            self.assertFalse(first.is_no_detection_placeholder)
            self.assertEqual(first.predicted_taxon.name, "Canis lupus")

    def test_collision_and_retry_keep_distinct_images_and_reconstructable_names(self):
        """Retry on a directory collision and reuse an existing addition without overwriting either file."""
        with TemporaryDirectory() as root, self.settings(MEDIA_ROOT=root):
            upload = UploadedArchiveFactory(outputdir=str(Path(root) / "output"))
            original = Path(root) / "IMG_0001.JPG"
            Image.new("RGB", (8, 8), "red").save(original)
            original_bytes = original.read_bytes()
            service = "caidapp.services.media_append"
            with (
                patch(f"{service}.timezone.now", return_value=datetime(2026, 11, 19, 14, 32, 8, tzinfo=timezone.utc)),
                patch(f"{service}.random_string", side_effect=["k7m2qx", "k7m2qx", "y83zz6"]) as random_code,
            ):
                first = append_image(
                    upload.pk, ContentFile(original_bytes, name=original.name), addition_key="first"
                )
                second_bytes = BytesIO()
                Image.new("RGB", (8, 8), "blue").save(second_bytes, format="JPEG")
                image = ContentFile(second_bytes.getvalue(), name="IMG_0001.JPG")
                second = append_image(upload.pk, image, addition_key="second")
                retried = append_image(upload.pk, image, addition_key="first")
                self.assertEqual(random_code.call_count, 3)

            self.assertEqual(retried.pk, first.pk)
            self.assertEqual(first.original_filename, "added_2026-11-19T14-32-08Z_k7m2qx/IMG_0001.JPG")
            self.assertEqual(second.original_filename, "added_2026-11-19T14-32-08Z_y83zz6/IMG_0001.JPG")
            for media, content in ((first, original_bytes), (second, second_bytes.getvalue())):
                path = media.metadata_json["original_path"]
                self.assertEqual(Path(media.mediafile.name).name, sha256(path.encode()).hexdigest() + ".jpg")
                self.assertEqual(Path(media.mediafile.path).read_bytes(), content)
                self.assertTrue(media.observations.get().is_no_detection_placeholder)
            self.assertEqual(upload.mediafile_set.count(), 2)

            with self.assertRaises(ValidationError):
                run_species_prediction_async(upload, force_init=True)
            upload.refresh_from_db()
            self.assertEqual(upload.taxon_status, "TV")
            self.assertEqual(upload.mediafile_set.count(), 2)
