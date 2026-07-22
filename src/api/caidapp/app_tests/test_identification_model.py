from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import TestCase

from caidapp import admin, models
from .factories import WorkGroupFactory


class IdentificationModelSourceTest(TestCase):
    def test_build_trained_identification_model_name_replaces_existing_timestamp(self):
        created_at = datetime(2026, 7, 10, 9, 45, 30)

        name = models.build_trained_identification_model_name(
            "LynxV4-MegaDescriptor-v2-T-256-20260701-121314",
            "My Fancy WG",
            created_at,
        )

        self.assertEqual(name, "LynxV4-MegaDescriptor-v2-T-256-my-fancy-wg-20260710-094530")

    def test_hf_model_source_is_used_directly(self):
        identification_model = models.IdentificationModel(
            name="HF model",
            model_path="hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        )

        self.assertEqual(
            identification_model.get_runtime_model_source(),
            "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        )
        self.assertEqual(identification_model.get_runtime_checkpoint_path(), "")

    def test_local_checkpoint_uses_base_model_source(self):
        identification_model = models.IdentificationModel(
            name="Local checkpoint",
            model_path="file:/shared_data/media/models/lynx/model.pth",
            base_model_path="hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        )

        self.assertEqual(
            identification_model.get_runtime_model_source(),
            "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        )
        self.assertEqual(
            identification_model.get_runtime_checkpoint_path(),
            "/shared_data/media/models/lynx/model.pth",
        )

    def test_local_checkpoint_can_inherit_base_model_from_source_model(self):
        workgroup = WorkGroupFactory()
        source_model = models.IdentificationModel.objects.create(
            name="Base",
            model_path="hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
            workgroup=workgroup,
        )
        derived_model = models.IdentificationModel(
            name="Derived",
            model_path="/shared_data/media/models/derived/model.pth",
            workgroup=workgroup,
            source_identification_model=source_model,
        )

        self.assertEqual(
            derived_model.get_runtime_model_source(),
            "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        )

    def test_legacy_local_checkpoint_without_base_source_uses_legacy_fallback(self):
        identification_model = models.IdentificationModel(
            name="Legacy",
            model_path="/shared_data/media/models/legacy/model.pth",
        )

        self.assertEqual(
            identification_model.get_runtime_base_model_source(),
            models.LEGACY_TRAINING_BASE_MODEL_SOURCE,
        )

    def test_local_checkpoint_check_reports_existing_file(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            checkpoint.touch()
            identification_model = models.IdentificationModel(
                name="Existing checkpoint",
                model_path=f"file:{checkpoint}",
                base_model_path="hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
            )

            result = admin.check_identification_model_file(identification_model)

        self.assertEqual(result.status, "ok")

    def test_local_checkpoint_check_reports_missing_file(self):
        identification_model = models.IdentificationModel(
            name="Missing checkpoint",
            model_path="file:/missing/model.pth",
            base_model_path="hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        )

        result = admin.check_identification_model_file(identification_model)

        self.assertEqual(result.status, "invalid")
        self.assertIn("neexistuje", result.detail)

    def test_remote_model_check_does_not_require_local_checkpoint(self):
        identification_model = models.IdentificationModel(
            name="Hub model",
            model_path="hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256",
        )

        result = admin.check_identification_model_file(identification_model)

        self.assertEqual(result.status, "external")
