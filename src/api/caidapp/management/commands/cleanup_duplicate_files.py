"""Recover file deletions after a broker outage or filesystem error."""

from caidapp import duplicate_services, models
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Retry pending file cleanup after committed duplicate merges (no restore support)."

    def handle(self, *args, **options):
        """Retry durable deletion intents, leaving failures available for the next run."""
        ids = list(models.DuplicateFileCleanup.objects.filter(status="pending").values_list("pk", flat=True))
        duplicate_services.cleanup_files(ids)
        pending = models.DuplicateFileCleanup.objects.filter(status="pending").count()
        self.stdout.write(f"Processed {len(ids)} cleanup intents; {pending} still pending.")
