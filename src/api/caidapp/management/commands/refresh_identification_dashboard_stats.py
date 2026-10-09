from django.core.management.base import BaseCommand

from caidapp import models
from caidapp.services.identification_dashboard import refresh_identification_dashboard_snapshot


class Command(BaseCommand):
    help = "Manually refresh identification dashboard snapshots for all or one workgroup."

    def add_arguments(self, parser):
        parser.add_argument("--workgroup-id", type=int)

    def handle(self, *args, **options):
        workgroups = models.WorkGroup.objects.all()
        if options["workgroup_id"] is not None:
            workgroups = workgroups.filter(pk=options["workgroup_id"])
        for workgroup in workgroups.iterator():
            refresh_identification_dashboard_snapshot(workgroup)
