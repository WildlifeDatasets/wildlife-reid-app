from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS, transaction
from django.db.models import Count, Q

from caidapp.models import AnimalObservation, MediaFile


OBJECT_DATA_QUERY = (
    Q(taxon__isnull=False)
    | Q(predicted_taxon__isnull=False)
    | Q(predicted_taxon_confidence__isnull=False)
    | Q(identity__isnull=False)
    | Q(bbox_x_center__isnull=False)
    | Q(bbox_y_center__isnull=False)
    | Q(bbox_width__isnull=False)
    | Q(bbox_height__isnull=False)
    | Q(identity_is_representative=True)
    | Q(taxon_verified=True)
    | Q(taxon_verified_at__isnull=False)
    | ~Q(orientation="N")
    | Q(metadata_json__isnull=False)
)


def invariant_violations(database=DEFAULT_DB_ALIAS):
    mediafiles = MediaFile.objects.using(database).annotate(
        observation_count=Count("observations"),
        placeholder_count=Count(
            "observations",
            filter=Q(observations__is_no_detection_placeholder=True),
        ),
    )
    return {
        "missing": mediafiles.filter(observation_count=0),
        "conflicting": mediafiles.filter(observation_count__gt=1, placeholder_count__gt=0),
        "malformed": AnimalObservation.objects.using(database).filter(is_no_detection_placeholder=True).filter(
            OBJECT_DATA_QUERY
        ),
    }


class Command(BaseCommand):
    help = "Audit the invariant that every media file has an observation and placeholders are unambiguous."

    def add_arguments(self, parser):
        parser.add_argument(
            "--repair",
            action="store_true",
            help="Repair violations. Without this flag the command is strictly read-only.",
        )
        parser.add_argument(
            "--database",
            default=DEFAULT_DB_ALIAS,
            help="Database alias to audit (default: default).",
        )

    def handle(self, *args, **options):
        database = options["database"]
        violations = invariant_violations(database)
        counts = {name: queryset.count() for name, queryset in violations.items()}
        self._write_report(counts)

        if not any(counts.values()):
            self.stdout.write(self.style.SUCCESS("Observation invariant is valid."))
            return

        if not options["repair"]:
            raise CommandError("Observation invariant violations found; no data was changed.")

        with transaction.atomic(using=database):
            missing_ids = list(violations["missing"].values_list("id", flat=True))
            AnimalObservation.objects.using(database).bulk_create(
                [
                    AnimalObservation(
                        mediafile_id=mediafile_id,
                        is_no_detection_placeholder=True,
                    )
                    for mediafile_id in missing_ids
                ],
                batch_size=1000,
            )

            # Preserve any object data first. A malformed placeholder may also
            # be part of a conflict and must not be deleted as if it were empty.
            violations["malformed"].update(is_no_detection_placeholder=False)

            conflicting_ids = list(
                invariant_violations(database)["conflicting"].values_list("id", flat=True)
            )
            AnimalObservation.objects.using(database).filter(
                mediafile_id__in=conflicting_ids,
                is_no_detection_placeholder=True,
            ).delete()

        remaining = invariant_violations(database)
        remaining_counts = {name: queryset.count() for name, queryset in remaining.items()}
        if any(remaining_counts.values()):
            self._write_report(remaining_counts, heading="Violations remaining after repair")
            raise CommandError("The repair did not restore the observation invariant.")

        self.stdout.write(self.style.SUCCESS("Observation invariant repaired successfully."))

    def _write_report(self, counts, heading="Observation invariant audit"):
        self.stdout.write(heading)
        self.stdout.write(f"  Media files without observations: {counts['missing']}")
        self.stdout.write(f"  Media files mixing placeholders and observations: {counts['conflicting']}")
        self.stdout.write(f"  Placeholders containing object data: {counts['malformed']}")
