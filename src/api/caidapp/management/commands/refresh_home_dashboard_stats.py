from django.core.management.base import BaseCommand

from caidapp.models import WorkGroup
from caidapp.services.home_dashboard import refresh_workgroup_home_dashboard_snapshot


class Command(BaseCommand):
    help = "Refresh persisted home dashboard statistics for workgroups."

    def add_arguments(self, parser):
        parser.add_argument("--workgroup-id", type=int, default=None)

    def handle(self, *args, **options):
        workgroups = WorkGroup.objects.all().order_by("id")
        if options["workgroup_id"] is not None:
            workgroups = workgroups.filter(id=options["workgroup_id"])

        if not workgroups.exists():
            self.stdout.write(self.style.WARNING("No workgroups found to refresh."))
            return

        refreshed = 0
        for workgroup in workgroups:
            snapshot = refresh_workgroup_home_dashboard_snapshot(workgroup)
            refreshed += 1
            total_mediafiles = snapshot.payload.get("summary", {}).get("total_mediafiles", 0)
            self.stdout.write(
                f"Refreshed workgroup {workgroup.id} ({workgroup.name}) with {total_mediafiles} media files."
            )

        self.stdout.write(self.style.SUCCESS(f"Refreshed {refreshed} home dashboard snapshot(s)."))
