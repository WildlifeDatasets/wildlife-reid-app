"""Queue a versioned observation embedding backfill for a workgroup."""

from caidapp.embedding_runs import create_embedding_run
from caidapp.models import WorkGroup
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Immediately backfill missing current embeddings; normal input changes schedule this automatically."

    def add_arguments(self, parser):
        """Allow workgroup scoping and optional individual observation selection."""
        parser.add_argument("--workgroup", type=int, required=True)
        parser.add_argument("--observation-id", type=int, action="append")

    def handle(self, *args, **options):
        """Submit the job without requiring ground-truth identities."""
        try:
            task = create_embedding_run(options["workgroup"], options["observation_id"], trigger="cli")
        except (ValidationError, WorkGroup.DoesNotExist, OSError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(f"Queued embedding run: {task.uuid}")
