"""Verify durable completion and asynchronous submission of observation embeddings."""

import json
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from caidapp.embedding_runs import cleanup_embedding_runs, create_embedding_run, prepare_run, refresh_run
from caidapp.models import EmbeddingRunItem, ObservationEmbeddingSchedule, TaskRun
from caidapp.observation_embeddings import fingerprint, prepare_scheduled_embeddings, store_embeddings
from caidapp.tasks import store_observation_embeddings
from celery.exceptions import Retry
from django.core.exceptions import ValidationError
from django.db import OperationalError
from django.utils import timezone

from .test_observation_embeddings import ObservationEmbeddingTests


class EmbeddingRunServiceTests(ObservationEmbeddingTests):
    def test_submission_does_not_hash_and_double_click_is_idempotent(self):
        """Keep HTTP submission cheap and duplicate requests stable."""
        with patch("caidapp.embedding_runs.signature") as publish, patch(
            "caidapp.observation_embeddings.snapshot"
        ) as snap:
            with self.captureOnCommitCallbacks(execute=True):
                run = create_embedding_run(self.workgroup.pk, [self.first.pk], request_key="double-click")
                duplicate = create_embedding_run(self.workgroup.pk, [self.first.pk], request_key="double-click")
            self.assertEqual(run.pk, duplicate.pk)
            self.assertEqual(run.items.count(), 1)
            snap.assert_not_called()
            publish.assert_called_once()

    def test_cached_completion_validates_current_inputs_and_is_idempotent(self):
        """Verify cached inputs again and preserve stable outcome on redelivery."""
        output, path = self.result([self.first, self.second])
        store_embeddings(output, path)
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text())
        manifest["cached"] = {
            fingerprint(self.spec): [
                f"{row['observation_id']}:{row['input_fingerprint']}" for row in manifest["observations"]
            ]
        }
        manifest_path.write_text(json.dumps(manifest))
        result_path = Path(output["output_path"])
        result_path.write_text(json.dumps({"profile": self.spec, "embeddings": []}))
        run = TaskRun.objects.create(workgroup=self.workgroup)
        for obs in [self.first, self.second]:
            EmbeddingRunItem.objects.create(
                run=run, observation=obs, observation_id_snapshot=obs.pk, mediafile_id_snapshot=obs.mediafile_id
            )
        self.second.bbox_x_center = self.second.bbox_y_center = 0.5
        self.second.bbox_width = self.second.bbox_height = 0.2
        self.second.save()
        store_embeddings(output, path, run.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, "partial")
        self.assertEqual(run.result["cached"], 1)
        self.assertEqual(run.result["stale"], 1)
        original = run.result.copy()
        store_embeddings(output, path, run.pk)
        run.refresh_from_db()
        self.assertEqual(run.result, original)

    def test_compute_success_is_not_database_success(self):
        """Wait for committed storage after GPU success."""
        run = TaskRun.objects.create(
            workgroup=self.workgroup, status="queued", compute_task_id="gpu", store_task_id="db"
        )
        with patch("caidapp.embedding_runs.AsyncResult", side_effect=[Mock(state="SUCCESS"), Mock(state="PENDING")]):
            refresh_run(run)
        self.assertEqual(run.status, "finalizing")
        self.assertIsNone(run.finished_at)

    def test_duplicate_prepare_does_not_submit_twice(self):
        """Claim a preparation run only once across task redelivery."""
        run = TaskRun.objects.create(workgroup=self.workgroup)
        with patch("caidapp.observation_embeddings.enqueue_embeddings", return_value=Mock(id="gpu")) as enqueue:
            prepare_run(run.pk)
            prepare_run(run.pk)
        enqueue.assert_called_once()

    def test_cleanup_retains_active_and_outside_directories(self):
        """Delete only old terminal manifests under the dedicated directory."""
        root = Path(self.directory.name)
        paths = [
            root / "observation_embeddings" / str(uuid.uuid4()),
            root / "observation_embeddings" / str(uuid.uuid4()),
            root / str(uuid.uuid4()),
        ]
        for directory, status in zip(paths, ["succeeded", "queued", "failed"]):
            directory.mkdir(parents=True)
            (directory / "input.json").write_text("{}")
            TaskRun.objects.create(
                workgroup=self.workgroup,
                status=status,
                finished_at=timezone.now() - timedelta(days=8),
                manifest_path=str(directory / "input.json"),
            )
        self.assertEqual(cleanup_embedding_runs(), 1)
        self.assertFalse(paths[0].exists())
        self.assertTrue(paths[1].exists())
        self.assertTrue(paths[2].exists())

    def test_database_error_retries_only_before_deadline(self):
        """Retry transient database failure within the bounded deadline only."""
        with patch("caidapp.observation_embeddings.store_embeddings", side_effect=OperationalError("offline")):
            with self.assertRaises(Retry):
                store_observation_embeddings.apply(args=[{}, "unused"], throw=True)
            with self.assertRaises(OperationalError):
                store_observation_embeddings.apply(
                    args=[{}, "unused"],
                    kwargs={"retry_deadline": (timezone.now() - timedelta(seconds=1)).isoformat()},
                    throw=True,
                )

    def test_validation_failure_is_terminal_and_never_retried(self):
        """Keep invalid results terminal even if a later callback is delivered."""
        run = TaskRun.objects.create(workgroup=self.workgroup)
        EmbeddingRunItem.objects.create(
            run=run,
            observation=self.first,
            observation_id_snapshot=self.first.pk,
            mediafile_id_snapshot=self.first.mediafile_id,
        )
        with patch("caidapp.observation_embeddings.store_embeddings", side_effect=ValidationError("invalid")):
            with self.assertRaises(ValidationError):
                store_observation_embeddings.apply(args=[{}, "unused"], kwargs={"run_id": run.pk}, throw=True)
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.items.get().status, "failed")
        output, manifest = self.result([self.first])
        store_embeddings(output, manifest, run.pk)
        run.refresh_from_db()
        self.assertEqual(run.status, "failed")

    def test_failed_automatic_run_reservation_keeps_timer(self):
        """Roll back timer consumption when durable run creation fails."""
        ObservationEmbeddingSchedule.objects.update_or_create(
            workgroup=self.workgroup, defaults={"task_id": "timer", "eta": timezone.now() - timedelta(seconds=1)}
        )
        with patch("caidapp.embedding_runs.create_embedding_run", side_effect=RuntimeError("unexpected")):
            with self.assertRaises(RuntimeError):
                prepare_scheduled_embeddings(self.workgroup.pk, "timer")
        self.assertEqual(ObservationEmbeddingSchedule.objects.get(workgroup=self.workgroup).task_id, "timer")
