import csv

from django.core.management.base import BaseCommand

from caidapp.models import MediaFile


LEGACY_FIELDS = (
    "taxon_id",
    "predicted_taxon_id",
    "predicted_taxon_confidence",
    "identity_id",
    "identity_is_representative",
    "orientation",
    "taxon_verified",
    "taxon_verified_at",
    "animal_number",
)


def _meaningful_legacy_values(mediafile):
    values = {
        "taxon_id": mediafile.taxon_id,
        "predicted_taxon_id": mediafile.predicted_taxon_id,
        "predicted_taxon_confidence": mediafile.predicted_taxon_confidence,
        "identity_id": mediafile.identity_id,
        "identity_is_representative": mediafile.identity_is_representative,
        "orientation": mediafile.orientation if mediafile.orientation not in {"", "N", None} else None,
        "taxon_verified": mediafile.taxon_verified,
        "taxon_verified_at": mediafile.taxon_verified_at,
        "animal_number": mediafile.animal_number,
    }
    return {field: value for field, value in values.items() if value not in {None, False, ""}}


def _single_observation_mismatches(mediafile, observation, legacy_values):
    mismatches = []
    observation_values = {
        "taxon_id": observation.taxon_id,
        "predicted_taxon_id": observation.predicted_taxon_id,
        "predicted_taxon_confidence": observation.predicted_taxon_confidence,
        "identity_id": observation.identity_id,
        "identity_is_representative": observation.identity_is_representative,
        "orientation": observation.orientation,
        "taxon_verified": observation.taxon_verified,
        "taxon_verified_at": observation.taxon_verified_at,
    }
    for field, legacy_value in legacy_values.items():
        if field == "animal_number":
            if legacy_value != 1:
                mismatches.append(field)
        elif observation_values.get(field) != legacy_value:
            mismatches.append(field)
    return mismatches


class Command(BaseCommand):
    help = "Audit animal-related legacy fields on MediaFile without changing data."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output",
            help="Optional CSV path for media files with meaningful legacy values.",
        )

    def handle(self, *args, **options):
        counters = {
            "total_mediafiles": 0,
            "with_meaningful_legacy_data": 0,
            "legacy_without_observation": 0,
            "legacy_with_one_observation": 0,
            "legacy_with_one_observation_mismatch": 0,
            "legacy_with_multiple_observations": 0,
        }
        rows = []
        queryset = MediaFile.objects.prefetch_related("observations").order_by("id")
        for mediafile in queryset.iterator(chunk_size=500):
            counters["total_mediafiles"] += 1
            legacy_values = _meaningful_legacy_values(mediafile)
            if not legacy_values:
                continue
            counters["with_meaningful_legacy_data"] += 1
            observations = list(mediafile.observations.all())
            mismatches = []
            if not observations:
                category = "no_observation"
                counters["legacy_without_observation"] += 1
            elif len(observations) == 1:
                category = "one_observation"
                counters["legacy_with_one_observation"] += 1
                mismatches = _single_observation_mismatches(mediafile, observations[0], legacy_values)
                if mismatches:
                    counters["legacy_with_one_observation_mismatch"] += 1
            else:
                category = "multiple_observations"
                counters["legacy_with_multiple_observations"] += 1
            rows.append(
                {
                    "mediafile_id": mediafile.id,
                    "observation_count": len(observations),
                    "observation_ids": ",".join(str(observation.id) for observation in observations),
                    "category": category,
                    "legacy_fields": ",".join(legacy_values),
                    "mismatched_fields": ",".join(mismatches),
                }
            )

        for key, value in counters.items():
            self.stdout.write(f"{key}: {value}")

        output_path = options.get("output")
        if output_path:
            with open(output_path, "w", newline="", encoding="utf-8-sig") as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=(
                        "mediafile_id",
                        "observation_count",
                        "observation_ids",
                        "category",
                        "legacy_fields",
                        "mismatched_fields",
                    ),
                )
                writer.writeheader()
                writer.writerows(rows)
            self.stdout.write(self.style.SUCCESS(f"CSV written to {output_path}"))
