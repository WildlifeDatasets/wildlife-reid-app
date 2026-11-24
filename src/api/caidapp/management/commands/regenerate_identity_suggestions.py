"""Queue observation reviews after deploying the new review schema."""

from celery.result import AsyncResult
from django.core.management.base import BaseCommand

from caidapp.models import WorkGroup
from caidapp.tasks import get_uploaded_archives_pending_identification
from caidapp.views import run_identification_bulk


class Command(BaseCommand):
    help = "Regenerate reviewable identity suggestions for eligible unassigned observations."

    def add_arguments(self, parser):
        """Allow limiting or previewing the regeneration."""
        parser.add_argument("--workgroup", type=int, action="append")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, **options):
        """Queue the existing identification task once per eligible workgroup."""
        groups = WorkGroup.objects.order_by("pk")
        if options["workgroup"]:
            groups = groups.filter(pk__in=options["workgroup"])
        for workgroup in groups.iterator():
            if not workgroup.identification_model_is_initialized():
                self.stdout.write(f"{workgroup.pk}: initialize the selected identification model first.")
                continue
            task_id = workgroup.identification_scheduled_run_task_id
            if workgroup.identification_reid_status == "Processing" and (
                not task_id or AsyncResult(task_id).state not in {"SUCCESS", "FAILURE", "REVOKED"}
            ):
                self.stdout.write(f"{workgroup.pk}: an identification run is already in progress.")
                continue
            archives = list(get_uploaded_archives_pending_identification(workgroup))
            if not archives:
                self.stdout.write(f"{workgroup.pk}: no eligible identification uploads.")
            elif options["dry_run"]:
                self.stdout.write(f"{workgroup.pk}: would process {len(archives)} uploads.")
            elif run_identification_bulk(workgroup, uploaded_archives=archives):
                self.stdout.write(f"{workgroup.pk}: queued {len(archives)} uploads for review.")
            else:
                self.stdout.write(f"{workgroup.pk}: no usable input for identification.")
