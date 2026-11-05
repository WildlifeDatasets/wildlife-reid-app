"""Exercise UUID backfill on pre-existing observation rows."""

from unittest.mock import patch

from caidapp.models import AnimalObservation
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

from .factories import MediaFileFactory


class ObservationUuidMigrationTests(TransactionTestCase):
    def test_existing_rows_receive_distinct_public_uuids(self):
        """Adding a unique UUID field must not reuse one migration-time default."""
        with patch("caidapp.observation_embedding_signals.schedule_embeddings"):
            mediafile = MediaFileFactory()
            first = mediafile.observations.first()
            first.is_no_detection_placeholder = False
            first.save()
            first_id = first.pk
            second_id = AnimalObservation.objects.create(mediafile=mediafile).pk
        executor = MigrationExecutor(connection)
        try:
            executor.migrate([("caidapp", "0179_observation_embedding_schedule")])
            executor = MigrationExecutor(connection)
            executor.migrate([("caidapp", "0180_animalobservation_uuid")])
            uuids = list(AnimalObservation.objects.filter(pk__in=[first_id, second_id]).values_list("uuid", flat=True))
            self.assertEqual(len(uuids), 2)
            self.assertTrue(all(value is not None for value in uuids))
            self.assertEqual(len(set(uuids)), 2)
        finally:
            executor = MigrationExecutor(connection)
            executor.migrate(executor.loader.graph.leaf_nodes())
