"""Export a permission-scoped workgroup dataset for beta-stability."""

import json

from caidapp.beta_stability_export import BetaExportError, export_beta_stability
from caidapp.models import AnimalObservation, EmbeddingProfile, WorkGroup
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Export observation annotations, current embeddings and optional source images to a beta-stability ZIP."

    def add_arguments(self, parser):
        """Require an explicit workgroup, profile selection and destination."""
        parser.add_argument("--workgroup", type=int, required=True)
        parser.add_argument(
            "--profile", action="append", required=True, help="Embedding profile UUID; repeat for multiple profiles."
        )
        parser.add_argument("--output", required=True)
        parser.add_argument("--observation-id", action="append", type=int)
        parser.add_argument("--include-images", action="store_true")
        parser.add_argument("--allow-partial", action="store_true")

    def handle(self, *args, **options):
        """Run the same export service used by the web download."""
        try:
            WorkGroup.objects.get(pk=options["workgroup"])
            observations = AnimalObservation.objects.filter(mediafile__parent__owner__workgroup_id=options["workgroup"])
            if options["observation_id"] is not None:
                requested = set(options["observation_id"])
                observations = observations.filter(pk__in=requested)
                if set(observations.values_list("pk", flat=True)) != requested:
                    raise ValidationError("Some requested observations are outside the workgroup or no longer exist.")
            profiles = [EmbeddingProfile.objects.get(uuid=value) for value in options["profile"]]
            result = export_beta_stability(
                observations,
                profiles,
                options["output"],
                include_images=options["include_images"],
                allow_partial=options["allow_partial"],
            )
        except BetaExportError as exc:
            self.stderr.write(json.dumps(exc.report, indent=2))
            raise CommandError(str(exc)) from exc
        except (ValidationError, WorkGroup.DoesNotExist, EmbeddingProfile.DoesNotExist, OSError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(f"Exported {result['report']['exported']} observations to {options['output']}")
