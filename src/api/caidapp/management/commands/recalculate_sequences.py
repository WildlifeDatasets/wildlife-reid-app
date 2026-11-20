from caidapp.models import UploadedArchive
from caidapp.services.sequences import recalculate_upload_sequences
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Recalculate sequences for one upload using its current capture times and localities."

    def add_arguments(self, parser):
        """Accept the ID of one upload."""
        parser.add_argument("upload_id", type=int)

    def handle(self, *args, **options):
        """Run the sequence calculation and report its result."""
        upload_id = options["upload_id"]
        try:
            count = recalculate_upload_sequences(upload_id)
        except UploadedArchive.DoesNotExist as exc:
            raise CommandError(f"Upload {upload_id} does not exist.") from exc
        except ValidationError as exc:
            raise CommandError(exc.messages[0]) from exc
        self.stdout.write(f"Recalculated {count} sequences for upload {upload_id}.")
