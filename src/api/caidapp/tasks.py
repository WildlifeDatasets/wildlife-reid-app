import logging
import os.path
from pathlib import Path

import django
import pandas as pd
from celery import shared_task
from django.conf import settings

from .fs_data import make_thumbnail_from_file
from .models import MediaFile, UploadedArchive, get_location, get_taxon

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
        uploaded_archive.zip_file = zip_file
        uploaded_archive.csv_file = csv_file
        make_thumbnail_for_uploaded_archive(uploaded_archive)
        get_image_files_from_uploaded_archive(uploaded_archive)
        uploaded_archive.status = "Finished"
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


def make_thumbnail_for_uploaded_archive(uploaded_archive: UploadedArchive):
    """Make small image representing the upload."""
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    abs_thumbnail_path = output_dir / "thumbnail.jpg"
    csv_file = Path(settings.MEDIA_ROOT) / str(uploaded_archive.csv_file)
    df = pd.read_csv(csv_file)
    if len(df["image_path"]) > 0:
        image_path = list(df["image_path"].sample(1))[0]
        abs_pth = output_dir / "images" / image_path
        # make_thumbnail_from_directory(output_dir, thumbnail_path)
        make_thumbnail_from_file(abs_pth, abs_thumbnail_path, width=600)

        uploaded_archive.thumbnail = os.path.relpath(abs_thumbnail_path, settings.MEDIA_ROOT)


def get_image_files_from_uploaded_archive(
    uploaded_archive: UploadedArchive, thumbnail_width: int = 400
):
    """Extract filenames from uploaded archive CSV and create MediaFile objects."""
    logger.debug("getting images from uploaded archive")
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    csv_file = Path(settings.MEDIA_ROOT) / str(uploaded_archive.csv_file)
    logger.debug(f"{csv_file} {Path(csv_file).exists()}")

    df = pd.read_csv(csv_file)
    # for fn in df["image_path"]:
    for index, row in df.iterrows():
        abs_pth = output_dir / "images" / row["image_path"]
        rel_pth = os.path.relpath(abs_pth, settings.MEDIA_ROOT)
        logger.debug(f"{abs_pth}")
        logger.debug(f"{rel_pth}")
        taxon = get_taxon(row["predicted_category"])
        captured_at = row["datetime"]
        if captured_at == "":
            captured_at = None
        location = get_location(str(uploaded_archive.location_at_upload))

        abs_pth_thumbnail = output_dir / "thumbnails" / row["image_path"]
        rel_pth_thumbnail = os.path.relpath(abs_pth_thumbnail, settings.MEDIA_ROOT)

        if make_thumbnail_from_file(abs_pth, abs_pth_thumbnail, width=thumbnail_width):
            thumbnail = str(rel_pth_thumbnail)
        else:
            thumbnail = None

        mf = MediaFile(
            parent=uploaded_archive,
            mediafile=str(rel_pth),
            category=taxon,
            location=location,
            captured_at=captured_at,
            thumbnail=thumbnail,
        )
        mf.save()
        logger.debug(f"{mf}")
