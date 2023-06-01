import logging

import django
from celery import shared_task
from caidapp.models import UploadedArchive
from pathlib import Path
from django.conf import settings
import random
import skimage.io
import skimage.transform
import os.path
from .fs_data import make_thumbnail_from_file

logger = logging.getLogger("app")


"""
Celery tasks used by "API worker" inside of the API docker container.
The API worker is different from e.g. Inference worker with its own docker container
because it has access to the database and other django resources,
and functions as a queue for processing worker responses.
"""


@shared_task(bind=True)
def predict_on_success(
    self, output: dict, *args, uploaded_archive_id: int, zip_file: str, csv_file: str, **kwargs
):
    """Success callback invoked after running predict function in inference worker."""
    status = output.get("status", "unknown")
    logger.info(f"Inference task finished with status '{status}'. Updating database record.")
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    if "status" not in output:
        logger.critical(f"Unexpected error {output=} is missing 'status' field.")
        uploaded_archive.status = "Unknown"
    elif output["status"] == "DONE":
        uploaded_archive.status = "Finished"
        uploaded_archive.zip_file = zip_file
        uploaded_archive.csv_file = csv_file
        make_thumbnail_for_uploaded_archive(uploaded_archive)
    else:
        uploaded_archive.status = "Failed"
    uploaded_archive.finished_at = django.utils.timezone.now()
    uploaded_archive.save()


@shared_task
def predict_on_error(task_id: str, *args, uploaded_archive_id: int, **kwargs):
    """Error callback invoked after running predict function in inference worker."""
    logger.critical(f"Worker task with id '{task_id}' failed due to unexpected internal error.")
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    uploaded_archive.status = "Failed"
    uploaded_archive.finished_at = django.utils.timezone.now()
    uploaded_archive.save()


def make_thumbnail_for_uploaded_archive(uploaded_archive:UploadedArchive):
    """
    Make small image representing the upload.
    """
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    thumbnail_path = output_dir / "thumbnail.jpg"

    make_thumbnail_from_file(output_dir, thumbnail_path)

    uploaded_archive.thumbnail = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)

def get_files_from_upload(uploaded_archive:UploadedArchive):
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir




