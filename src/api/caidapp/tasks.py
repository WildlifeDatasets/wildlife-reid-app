import json
import logging
import os.path
from pathlib import Path

import django
import pandas as pd
from celery import shared_task
from celery import chain, signature
from django.conf import settings

from .fs_data import make_thumbnail_from_file
from .models import (
    IndividualIdentity,
    MediaFile,
    MediafilesForIdentification,
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


def get_image_files_from_uploaded_archive(
    uploaded_archive: UploadedArchive, thumbnail_width: int = 400
):
    """Extract filenames from uploaded archive CSV and create MediaFile objects.

    If the processing is repeated, the former Mediafiles are used and updated.
    """
    logger.debug("getting images from uploaded archive")
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    csv_file = Path(settings.MEDIA_ROOT) / str(uploaded_archive.csv_file)
    logger.debug(f"{csv_file} {Path(csv_file).exists()}")

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

        mf.category = taxon
        if mf.identity is None:
            # update only if the identity is not set
            identity = get_unique_name(
                row["unique_name"], workgroup=uploaded_archive.owner.workgroup
            )
            logger.debug(f"identity={identity}")
            mf.identity = identity
        mf.save()
        logger.debug(f"{mf}")


@shared_task
def init_identification_on_success(*args, **kwargs):
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug("init_identificaion done.")


@shared_task
def init_identification_on_error(*args, **kwargs):
    """Callback invoked after failing init_identification function in inference worker."""
    logger.error("init_identificaion done with error.")


@shared_task(bind=True)
def on_error(self, uuid, *args, **kwargs):
    """Callback invoked after failing init_identification function in inference worker."""
    logger.error("Process finished with error.")
    result = self.AsyncResult(uuid)
    error_message = result.result if result.failed() else "No error message available"
    logger.error(f"Error message: {error_message}")

    logger.debug(f"self={self}")
    logger.debug(f"args={args}")
    logger.debug(f"kwargs={kwargs}")

# @shared_task
@shared_task(bind=True)
def on_error_in_upload_processing(self, uuid, *args, **kwargs):
    """Callback invoked after failing init_identification function in inference worker."""
    logger.error("Process finished with error.")
    result = self.AsyncResult(uuid)
    error_message = result.result if result.failed() else "No error message available"
    logger.error(f"Upload processing with error: {error_message}")

    logger.debug(f"self={self}")
    logger.debug(f"args={args}")
    logger.debug(f"kwargs={kwargs}")
    # logger.debug(f"dir(self)={dir(self)}")


@shared_task(bind=True)
def log_output(self, output: dict, *args, **kwargs):
    logger.debug("log_output")
    logger.debug(f"{output=}")
    logger.debug(f"{args=}")
    logger.debug(f"{kwargs=}")

@shared_task(bind=True)
def detection_on_success(self, output: dict, *args, **kwargs):
    logger.debug("detection on success")
    logger.debug(f"{output=}")
    logger.debug(f"{args=}")
    logger.debug(f"{kwargs=}")

    logger.debug("calling...")
    # identify_signature = signature(
    #     "detectionsimplelog",
    #     kwargs = {
    #         "pokus": 4,
    #     },
    # )
    identify_signature = signature(
        "identify",
        kwargs = kwargs,
    )
    identify_task = identify_signature.apply_async(
        link=identify_on_success.s(
            # output=output,
        ),
        link_error=on_error_in_upload_processing.s(),
    )
    logger.debug(f"{identify_task=}")



@shared_task(bind=True)
def identify_on_success(self, output: dict, *args, **kwargs):
    """Callback invoked after running init_identification function in inference worker."""
    status = output.get("status", "unknown")
    logger.info(f"Identification task finished with status '{status}'. Updating database record.")

    logger.debug(f"self={self}")
    logger.debug(f"output={output}")
    logger.debug(f"args={args}")
    logger.debug(f"kwargs={kwargs}")

    if "status" not in output:
        logger.critical(f"Unexpected error {output=} is missing 'status' field.")
        # TODO - should the app return some error response to the user?
    elif output["status"] == "DONE":
        # load output file
        output_json_file = output["output_json_file"]
        with open(output_json_file, "r") as f:
            data = json.load(f)
        logger.debug(f"Loaded output data: {data=}")
        assert "mediafile_ids" in data
        assert "pred_image_paths" in data
        assert "pred_class_ids" in data
        assert "pred_labels" in data
        assert "scores" in data

        media_root = Path(settings.MEDIA_ROOT)

        mediafile_ids = data["mediafile_ids"]
        for i, mediafile_id in enumerate(mediafile_ids):

            top_k_class_ids = data["pred_class_ids"][i]
            top_k_labels = data["pred_labels"][i]
            top_k_paths = data["pred_image_paths"][i]
            top_k_scores = data["scores"][i]

            mediafile = MediaFile.objects.get(id=mediafile_id)

            if (top_k_scores[0]) > settings.IDENTITY_MANUAL_CONFIRMATION_THRESHOLD:

                identity_id = top_k_class_ids[0]  # top-1
                mediafile.identity = IndividualIdentity.objects.get(id=identity_id)
                logger.debug(
                    f"{mediafile} is {mediafile.identity.name} with score={top_k_scores[0]}. "
                    + "No need of manual confirmation."
                )
                if mediafile.identity.name != top_k_labels[0]:  # top-1
                    logger.warning(
                        f"Identity name mismatch: {mediafile.identity.name} != {top_k_labels[0]}"
                    )

                mediafile.save()

            else:
                top1_abspath = Path(top_k_paths[0])
                top1_relpath = top1_abspath.relative_to(media_root)
                top1_mediafile = MediaFile.objects.get(mediafile=str(top1_relpath))
                # top1_identity_id = top_k_class_ids[0]  # top-1

                top2_abspath = Path(top_k_paths[1])
                top2_relpath = top2_abspath.relative_to(media_root)
                top2_mediafile = MediaFile.objects.get(mediafile=str(top2_relpath))
                # top2_identity_id = top_k_class_ids[1]  # top-1

                top3_abspath = Path(top_k_paths[2])
                top3_relpath = top3_abspath.relative_to(media_root)
                top3_mediafile = MediaFile.objects.get(mediafile=str(top3_relpath))
                # top3_identity_id = top_k_class_ids[2]  # top-1

                mfi, created = MediafilesForIdentification.objects.get_or_create(
                    mediafile=mediafile,
                )

                mfi.top1mediafile = top1_mediafile
                mfi.top1score = top_k_scores[0]
                mfi.top1name = top_k_labels[0]
                mfi.top2mediafile = top2_mediafile
                mfi.top2score = top_k_scores[1]
                mfi.top2name = top_k_labels[1]
                mfi.top3mediafile = top3_mediafile
                mfi.top3score = top_k_scores[2]
                mfi.top3name = top_k_labels[2]
                mfi.save()
                if top1_mediafile.identity.name != top_k_labels[0]:
                    logger.warning(
                        f"Identity mismatch: {top1_mediafile.identity.name} != {top_k_labels[0]}"
                    )
                if top2_mediafile.identity.name != top_k_labels[1]:
                    logger.warning(
                        f"Identity mismatch: {top2_mediafile.identity.name} != {top_k_labels[1]}"
                    )
                if top3_mediafile.identity.name != top_k_labels[2]:
                    logger.warning(
                        f"Identity mismatch: {top3_mediafile.identity.name} != {top_k_labels[2]}"
                    )

        logger.debug("identify done.")

        # simple_log_sig = signature("iworker_simple_log")
        # simple_log_task = simple_log_sig.apply_async()

    else:
        # identification failed
        logger.error("Identification failed.")
        # TODO - should the app return some error response to the user?
        pass

@shared_task(bind=True)
def simple_log(self, *args, **kwargs):
    """Simple log task."""
    logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
    return {"status": "DONE"}

def _find_mediafiles_for_identification(mediafile_paths: list) -> MediafilesForIdentification:
    """Find mediafiles for identification.

    :param mediafile_paths: List of paths of mediafiles to identify.
    :return: MediafilesForIdentification object.
    """
