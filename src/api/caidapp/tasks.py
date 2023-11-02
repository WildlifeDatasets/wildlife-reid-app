import logging
import os.path
from pathlib import Path
from typing import List

import django
import pandas as pd
from celery import shared_task
from django.conf import settings

from .fs_data import make_thumbnail_from_file
from .models import (
    IndividualIdentity,
    MediaFile,
    UploadedArchive,
    get_location,
    get_taxon,
    get_unique_name,
)

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
        sync_mediafiles_uploaded_archive_with_csv(uploaded_archive)
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
    logger.debug("making thumbnail for uploaded archive")
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


def sync_mediafiles_uploaded_archive_with_csv(
    uploaded_archive: UploadedArchive, thumbnail_width: int = 400,
    create_missing:bool = True
):
    """Extract filenames from uploaded archive CSV and create MediaFile objects.

    If the processing is repeated, the former Mediafiles are used and updated.
    """
    logger.debug("getting images from uploaded archive")
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    csv_file = Path(settings.MEDIA_ROOT) / str(uploaded_archive.csv_file)
    logger.debug(f"{csv_file} {Path(csv_file).exists()}")


    update_csv = False
    df = pd.read_csv(csv_file)
    # for fn in df["image_path"]:
    for index, row in df.iterrows():
        abs_pth = output_dir / "images" / row["image_path"]
        rel_pth = os.path.relpath(abs_pth, settings.MEDIA_ROOT)
        logger.debug(f"vanilla_path={row['vanilla_path']}")
        logger.debug(f"relative_pth={rel_pth}")
        taxon = get_taxon(row["predicted_category"])
        captured_at = row["datetime"]
        if captured_at == "":
            captured_at = None

        abs_pth_thumbnail = output_dir / "thumbnails" / row["image_path"]
        rel_pth_thumbnail = os.path.relpath(abs_pth_thumbnail, settings.MEDIA_ROOT)

        mediafile_set = uploaded_archive.mediafile_set.filter(mediafile=str(rel_pth))
        if len(mediafile_set) == 0:
            if create_missing:
                location = get_location(str(uploaded_archive.location_at_upload))
                logger.debug(f"Creating thumbnail for {rel_pth}")
                if make_thumbnail_from_file(abs_pth, abs_pth_thumbnail, width=thumbnail_width):
                    thumbnail = str(rel_pth_thumbnail)
                else:
                    thumbnail = None

                mf = MediaFile(
                    parent=uploaded_archive,
                    mediafile=str(rel_pth),
                    captured_at=captured_at,
                    thumbnail=thumbnail,
                    location=location,
                )
                mf.save()
            else:
                # row["error"] = "deleted"
                df.loc[index, "error"] = "deleted"
                continue
        else:
            mf = mediafile_set[0]
            logger.debug("Using Mediafile generated before")
            # generate thumbnail if necessary
            if (mf.thumbnail is None) or (not abs_pth_thumbnail.exists()):
                if make_thumbnail_from_file(abs_pth, abs_pth_thumbnail, width=thumbnail_width):
                    mf.thumbnail = str(rel_pth_thumbnail)
                    mf.save()
                else:
                    logger.warning(f"Cannot generate thumbnail for {abs_pth}")

        if mf.category is None:
            mf.category = taxon
        else:
            # row["predicted_category"] = mf.category.name
            df.loc[index, "predicted_category"] = mf.category.name
            update_csv = True
        if mf.identity is None:
            # update only if the identity is not set
            identity = get_unique_name(
                row["unique_name"], workgroup=uploaded_archive.owner.workgroup
            )
            logger.debug(f"identity={identity}")
            mf.identity = identity

        mf.save()

        if mf.identity is not None:
            # row["unique_name"] = mf.identity.name
            df.loc[index, "unique_name"] = mf.identity.name
            update_csv = True

        logger.debug(f"{mf}")
    if update_csv:
        df.to_csv(csv_file, index=False)
        logger.debug(f"CSV updated. path={csv_file}")
    uploaded_archive.output_updated_at = django.utils.timezone.now()
    uploaded_archive.save()


@shared_task
def init_identification_on_success(*args, **kwargs):
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug("init_identificaion done.")


@shared_task
def init_identification_on_error(*args, **kwargs):
    """Callback invoked after failing init_identification function in inference worker."""
    logger.error("init_identificaion done with error.")


# @shared_task
@shared_task(bind=True)
def identify_on_error(self, uuid, *args, **kwargs):
    """Callback invoked after failing init_identification function in inference worker."""
    logger.error("identify done with error.")
    result = self.AsyncResult(uuid)
    error_message = result.result if result.failed() else "No error message available"
    logger.error(f"identify done with error: {error_message}")

    # logger.debug(f"args={args}")
    # logger.debug(f"kwargs={kwargs}")
    # logger.debug(f"self={self}")
    # logger.debug(f"dir(self)={dir(self)}")


@shared_task(bind=True)
def identify_on_success(
    self, output: dict, *args, uploaded_archive_id: int, mediafile_ids: List[int], **kwargs
):
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug(f"identify_on_success with {len(mediafile_ids)}")
    logger.debug(f"self={self}")
    # logger.debug(f"uuid={uuid}")
    logger.debug(f"output={output}")
    logger.debug(f"args={args}")
    logger.debug(f"uploaded_archive_id={uploaded_archive_id}")
    logger.debug(f"mediafile_ids={mediafile_ids}")
    logger.debug(f"kwargs={kwargs}")
    data = output["data"]
    for i, mediafile_id in enumerate(mediafile_ids):
        mediafile = MediaFile.objects.get(id=mediafile_id)
        identity_id = data["pred_class_ids"][i]
        mediafile.identity = IndividualIdentity.objects.get(id=identity_id)
        if mediafile.identity.name != data["pred_labels"][i]:
            logger.warning(
                f"Identity name mismatch: {mediafile.identity.name} != {data['pred_labels'][i]}"
            )

        mediafile.save()

    logger.debug("identify done.")
