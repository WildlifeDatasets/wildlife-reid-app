"""Check automatic embedding debounce timers and input-change triggers."""

from datetime import timedelta
from unittest.mock import Mock, patch

from django.db import transaction
from django.test import TestCase
from django.utils import timezone

from caidapp.models import AnimalObservation, IdentificationModel, ObservationEmbeddingSchedule
from caidapp.observation_embeddings import prepare_scheduled_embeddings, schedule_embeddings

from .factories import MediaFileFactory


class EmbeddingSchedulingTests(TestCase):
    def setUp(self):
        """Prepare one eligible observation in a completed upload."""
        self.mediafile = MediaFileFactory()
        self.observation = self.mediafile.observations.first()
        self.observation.is_no_detection_placeholder = False
        self.observation.save()
        self.workgroup = self.mediafile.parent.owner.workgroup

    def test_changes_extend_one_pending_timer(self):
        """A burst of edits updates the deadline without publishing one task per edit."""
        timestamp = timezone.now()
        with patch("caidapp.observation_embeddings._publish_preparation") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                with patch("caidapp.observation_embeddings.timezone.now", return_value=timestamp):
                    first = schedule_embeddings(self.workgroup.pk)
                with patch(
                    "caidapp.observation_embeddings.timezone.now", return_value=timestamp + timedelta(minutes=3)
                ):
                    second = schedule_embeddings(self.workgroup.pk)
            self.assertEqual(first, second)
            publish.assert_called_once()
        schedule = ObservationEmbeddingSchedule.objects.get(workgroup=self.workgroup)
        self.assertEqual(schedule.eta, timestamp + timedelta(minutes=13))
        # Saving a stale WorkGroup settings instance must not erase its independent timer.
        self.workgroup.save()
        self.assertEqual(ObservationEmbeddingSchedule.objects.get(workgroup=self.workgroup).task_id, first)

    def test_timer_defers_then_dispatches_once_and_ignores_old_delivery(self):
        """An early timer follows the extended deadline and obsolete messages do nothing."""
        first = schedule_embeddings(self.workgroup.pk)
        with patch("caidapp.observation_embeddings._publish_preparation") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                result = prepare_scheduled_embeddings(self.workgroup.pk, first)
            self.assertEqual(result["status"], "deferred")
            publish.assert_called_once()
        self.assertEqual(prepare_scheduled_embeddings(self.workgroup.pk, first)["status"], "superseded")
        ObservationEmbeddingSchedule.objects.filter(workgroup=self.workgroup).update(eta=timezone.now())
        with patch("caidapp.observation_embeddings.enqueue_embeddings", return_value=Mock(id="gpu-task")) as enqueue:
            ready = prepare_scheduled_embeddings(self.workgroup.pk, result["task_id"])
            self.assertEqual(ready, {"status": "queued", "task_id": "gpu-task"})
            enqueue.assert_called_once_with(self.workgroup.pk, skip_invalid=True, completed_uploads_only=True)
            self.assertEqual(prepare_scheduled_embeddings(self.workgroup.pk, result["task_id"])["status"], "superseded")
        self.assertIsNone(ObservationEmbeddingSchedule.objects.get(workgroup=self.workgroup).task_id)

    def test_rollback_never_publishes(self):
        """Rolled-back edits cannot enqueue an embedding job."""
        with patch("caidapp.observation_embeddings._publish_preparation") as publish:
            with self.captureOnCommitCallbacks(execute=True):
                with transaction.atomic():
                    schedule_embeddings(self.workgroup.pk)
                    transaction.set_rollback(True)
            publish.assert_not_called()
        self.assertFalse(ObservationEmbeddingSchedule.objects.exists())

    def test_observation_creation_and_bbox_edits_trigger_after_commit(self):
        """New observations and changed crops are inputs even without identity labels."""
        with patch("caidapp.observation_embedding_signals.schedule_embeddings") as schedule:
            with self.captureOnCommitCallbacks(execute=True):
                AnimalObservation.objects.create(mediafile=self.mediafile)
            schedule.assert_called_once_with(self.workgroup.pk)
            schedule.reset_mock()
            with self.captureOnCommitCallbacks(execute=True):
                self.observation.bbox_x_center = self.observation.bbox_y_center = 0.5
                self.observation.bbox_width = self.observation.bbox_height = 0.2
                self.observation.save()
            schedule.assert_called_once_with(self.workgroup.pk)

    def test_identity_metadata_and_partial_unsaved_bbox_edits_do_not_trigger(self):
        """Only persisted input changes should move the timer."""
        with patch("caidapp.observation_embedding_signals.schedule_embeddings") as schedule:
            with self.captureOnCommitCallbacks(execute=True):
                self.observation.metadata_json = {"reviewed": True}
                self.observation.save()
                self.observation.bbox_width = 0.2
                self.observation.save(update_fields=["identity"])
            schedule.assert_not_called()

    def test_placeholder_creation_does_not_trigger(self):
        """No-detection placeholders are not embedding inputs."""
        with patch("caidapp.observation_embedding_signals.schedule_embeddings") as schedule:
            with self.captureOnCommitCallbacks(execute=True):
                mediafile = MediaFileFactory(parent=self.mediafile.parent)
            # The source save may request a scan, but a placeholder-only workgroup is filtered by the scheduler.
            schedule.reset_mock()
            with self.captureOnCommitCallbacks(execute=True):
                mediafile.observations.first().save()
            schedule.assert_not_called()

    def test_completed_import_and_model_changes_trigger(self):
        """Import completion and a new selected encoder both request current embeddings."""
        archive = self.mediafile.parent
        archive.import_finished = False
        archive.save(update_fields=["import_finished"])
        self.assertIsNone(schedule_embeddings(self.workgroup.pk))
        model = IdentificationModel.objects.create(name="New model", model_path="timm:resnet18")
        with patch("caidapp.observation_embedding_signals.schedule_embeddings") as schedule:
            with self.captureOnCommitCallbacks(execute=True):
                archive.import_finished = True
                archive.save(update_fields=["import_finished"])
            schedule.assert_called_once_with(self.workgroup.pk)
            schedule.reset_mock()
            with self.captureOnCommitCallbacks(execute=True):
                self.workgroup.identification_model = model
                self.workgroup.save(update_fields=["identification_model"])
            schedule.assert_called_once_with(self.workgroup.pk)
            schedule.reset_mock()
            with self.captureOnCommitCallbacks(execute=True):
                model.model_path = "timm:resnet34"
                model.save()
            schedule.assert_called_once_with(self.workgroup.pk)

    def test_source_changes_trigger(self):
        """Replacing the selected source file requests a fresh input version."""
        with patch("caidapp.observation_embedding_signals.schedule_embeddings") as schedule:
            with self.captureOnCommitCallbacks(execute=True):
                self.mediafile.image_file = "changed/source.jpg"
                self.mediafile.save(update_fields=["image_file"])
            schedule.assert_called_once_with(self.workgroup.pk)
