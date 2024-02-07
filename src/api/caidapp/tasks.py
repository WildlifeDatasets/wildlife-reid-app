import json
import logging
import os.path
from pathlib import Path

import django
import numpy as np
import pandas as pd
from celery import shared_task, signature
from django.conf import settings

from .fs_data import make_thumbnail_from_file
from .models import (
    IndividualIdentity,
    MediaFile,
    MediafilesForIdentification,
    UploadedArchive,
    WorkGroup,
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
def predict_species_on_success(
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
        # update_metadata_csv_by_uploaded_archive(uploaded_archive)
        # create missing take effect only if the processing is done for the first time
        # in other cases the file should be removed from CSV before the processing is run
        update_uploaded_archive_by_metadata_csv(uploaded_archive, create_missing=True)
        uploaded_archive.status = "Finished"
    else:
        uploaded_archive.status = "Failed"
    uploaded_archive.finished_at = django.utils.timezone.now()
    uploaded_archive.save()


@shared_task
def predict_species_on_error(task_id: str, *args, uploaded_archive_id: int, **kwargs):
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


def run_species_prediction_async(uploaded_archive: UploadedArchive):
    """Run species prediction asynchronously."""
    # update record in the database
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    uploaded_archive.started_at = django.utils.timezone.now()
    output_archive_file = output_dir / "images.zip"
    output_metadata_file = output_dir / "metadata.csv"
    uploaded_archive.csv_file = str(Path(uploaded_archive.outputdir) / "metadata.csv")
    uploaded_archive.status = "Processing"
    uploaded_archive.save()
    logger.debug(f"updating uploaded archive, {uploaded_archive.csv_file=}")

    # if the metadata file exists, it is updated
    update_metadata_csv_by_uploaded_archive(uploaded_archive)

    # send celery message to the data worker
    logger.info("Sending request to inference worker.")
    sig = signature(
        "predict",
        kwargs={
            "input_archive_file": str(
                Path(settings.MEDIA_ROOT) / uploaded_archive.archivefile.name
            ),
            "output_dir": str(output_dir),
            "output_archive_file": str(output_archive_file),
            "output_metadata_file": str(output_metadata_file),
            "contains_identities": uploaded_archive.contains_identities,
        },
    )
    task = sig.apply_async(
        link=predict_species_on_success.s(
            uploaded_archive_id=uploaded_archive.id,
            zip_file=os.path.relpath(str(output_archive_file), settings.MEDIA_ROOT),
            csv_file=os.path.relpath(str(output_metadata_file), settings.MEDIA_ROOT),
        ),
        link_error=predict_species_on_error.s(uploaded_archive_id=uploaded_archive.id),
    )
    logger.info(f"Created worker task with id '{task.task_id}'.")




def make_thumbnail_for_mediafile_if_necessary(mediafile: MediaFile, thumbnail_width: int = 400):
    """Make small image representing the upload."""
    logger.debug("making thumbnail for mediafile")
    mediafile_path = Path(settings.MEDIA_ROOT) / mediafile.mediafile.name
    output_dir = Path(settings.MEDIA_ROOT) / mediafile.parent.outputdir
    abs_pth = output_dir / "thumbnails" / Path(mediafile.mediafile.name).name
    rel_pth = os.path.relpath(abs_pth, settings.MEDIA_ROOT)
    # make_thumbnail_from_directory(output_dir, thumbnail_path)
    if (mediafile.thumbnail is None) or (not abs_pth.exists()):
        logger.debug(f"Creating thumbnail for {rel_pth}")
        if make_thumbnail_from_file(mediafile_path, abs_pth, width=thumbnail_width):
            mediafile.thumbnail = str(rel_pth)
            mediafile.save()
        else:
            logger.warning(f"Cannot generate thumbnail for {abs_pth}")


def _get_rel_and_abs_paths_based_on_csv_row(row: dict, output_dir: Path):
    abs_pth = output_dir / "images" / row["image_path"]
    rel_pth = os.path.relpath(abs_pth, settings.MEDIA_ROOT)
    if "vanilla_Path" in row:
        logger.debug(f"vanilla_path={row['vanilla_path']}")
    logger.debug(f"relative_pth={rel_pth}")

    return rel_pth, abs_pth


def update_uploaded_archive_by_metadata_csv(
    uploaded_archive: UploadedArchive, thumbnail_width: int = 400, create_missing: bool = True
) -> None:
    """Extract filenames from uploaded archive CSV and create MediaFile objects.

    If the processing is repeated, the former Mediafiles are used and updated.
    If the MediaFile was updated by user, the update is skipped.
    """
    logger.debug("getting images from uploaded archive")
    csv_file = Path(settings.MEDIA_ROOT) / str(uploaded_archive.csv_file)
    logger.debug(f"{csv_file} {Path(csv_file).exists()}")

    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir

    df = pd.read_csv(csv_file, index_col=0)

    for index, row in df.iterrows():
        rel_pth, _ = _get_rel_and_abs_paths_based_on_csv_row(row, output_dir)
        captured_at = row["datetime"]
        logger.debug(f"{captured_at=}, {type(captured_at)}")
        if (captured_at == "") or (isinstance(captured_at, float) and np.isnan(captured_at)):
            captured_at = None

        try:
            mf = uploaded_archive.mediafile_set.get(mediafile=str(rel_pth))
            logger.debug("Using Mediafile generated before")
        except MediaFile.DoesNotExist:
            if create_missing:
                location = get_location(str(uploaded_archive.location_at_upload))

                mf = MediaFile(
                    parent=uploaded_archive,
                    mediafile=str(rel_pth),
                    captured_at=captured_at,
                    location=location,
                )
                mf.save()
                logger.debug(f"Created new Mediafile {mf}")
            else:
                df.loc[index, "deleted"] = True
                logger.debug(f"Mediafile {rel_pth} not found. Skipping.")
                continue
            # generate thumbnail if necessary
        make_thumbnail_for_mediafile_if_necessary(mf, thumbnail_width=thumbnail_width)

        # if the mediafile was updated by user, we believe into users input
        if mf.updated_by is None:
            mf.category = get_taxon(row["predicted_category"])
            mf.identity = get_unique_name(
                row["unique_name"], workgroup=uploaded_archive.owner.workgroup
            )
            mf.save()
            logger.debug(f"identity={mf.identity}")
        logger.debug(f"{mf}")
    uploaded_archive.save()


def update_metadata_csv_by_uploaded_archive(
    uploaded_archive: UploadedArchive,
    # thumbnail_width: int = 400, create_missing: bool = True
):
    """Update metadata CSV file by MediaFiles in UploadedArchive."""
    logger.debug("Updating metadata by uploaded archive...")
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    logger.debug(f"{uploaded_archive.csv_file=}")
    csv_file = Path(settings.MEDIA_ROOT) / str(uploaded_archive.csv_file)
    logger.debug(f"{csv_file=} {Path(csv_file).exists()}")
    if not Path(csv_file).exists():
        logger.warning(f"CSV file {csv_file} does not exist. Skipping.")
        return

    update_csv = False
    df = pd.read_csv(csv_file, index_col=0)
    df["deleted"] = True
    df["location name"] = ""
    df["location coordinates"] = ""
    # for fn in df["image_path"]:
    for index, row in df.iterrows():
        rel_pth, _ = _get_rel_and_abs_paths_based_on_csv_row(row, output_dir)

        try:
            mf = uploaded_archive.mediafile_set.get(mediafile=str(rel_pth))
            logger.debug("Using Mediafile generated before")
        except MediaFile.DoesNotExist:
            df.loc[index, "deleted"] = True
            update_csv = True
            logger.debug(f"Mediafile {rel_pth} not found. The row will be removed from CSV.")
            continue
        # generate thumbnail if necessary
        make_thumbnail_for_mediafile_if_necessary(mf)

        if mf.category:
            df.loc[index, "predicted_category"] = mf.category.name
            update_csv = True
        if mf.identity:
            df.loc[index, "unique_name"] = mf.identity.name
            update_csv = True
        if mf.location:
            df.loc[index, "location name"] = mf.location.name
            if mf.location.location:
                df.loc[index, "location coordinates"] = str(mf.location.location)

    # delete rows with missing mediafiles
    df = df[df["deleted"] == False]  # noqa: E712
    if update_csv:
        df.to_csv(csv_file, encoding="utf-8-sig")
        logger.debug(f"CSV updated. path={csv_file}")
    uploaded_archive.output_updated_at = django.utils.timezone.now()
    uploaded_archive.save()


@shared_task
def init_identification_on_success(*args, **kwargs):
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug(f"{args=}")
    logger.debug(f"{kwargs=}")
    workgroup_id = kwargs.pop("workgroup_id")
    workgroup = WorkGroup.objects.get(id=workgroup_id)
    now = django.utils.timezone.now()
    workgroup.identification_init_finished_at = now
    workgroup.save()
    logger.debug(f"{workgroup=}")
    logger.debug(f"{workgroup.identification_init_finished_at=}")

    logger.debug("init_identification done.")


@shared_task
def init_identification_on_error(*args, **kwargs):
    """Callback invoked after failing init_identification function in inference worker."""
    logger.error("init_identification done with error.")


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
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug("log_output")
    logger.debug(f"{output=}")
    logger.debug(f"{args=}")
    logger.debug(f"{kwargs=}")


@shared_task(bind=True)
def detection_on_success(self, output: dict, *args, **kwargs):
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug("detection on success")
    logger.debug(f"{output=}")
    logger.debug(f"{args=}")
    logger.debug(f"{kwargs=}")

    uploaded_archive_id: int = kwargs.pop("uploaded_archive_id")
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    uploaded_archive.status = "...detection done"
    uploaded_archive.save()
    identify_signature = signature(
        "identify",
        kwargs=kwargs,
    )
    identify_task = identify_signature.apply_async(
        link=identify_on_success.s(
            uploaded_archive_id=uploaded_archive_id,
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

    uploaded_archive_id: int = kwargs.pop("uploaded_archive_id")
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    uploaded_archive.status = "...identification done"
    uploaded_archive.save()

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

            _prepare_mediafile_for_identification(data, i, media_root, mediafile_id)

        uploaded_archive.status = "Identification finished"
        uploaded_archive.save()
        logger.debug("identify done.")

    else:
        # identification failed
        uploaded_archive.status = "Identification failed"
        uploaded_archive.save()
        logger.error("Identification failed.")
        # TODO - should the app return some error response to the user?


def _prepare_mediafile_for_identification(data, i, media_root, mediafile_id):
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

        top2_abspath = Path(top_k_paths[1])
        top2_relpath = top2_abspath.relative_to(media_root)
        top2_mediafile = MediaFile.objects.get(mediafile=str(top2_relpath))

        top3_abspath = Path(top_k_paths[2])
        top3_relpath = top3_abspath.relative_to(media_root)
        top3_mediafile = MediaFile.objects.get(mediafile=str(top3_relpath))

        mfi, _ = MediafilesForIdentification.objects.get_or_create(
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
        _identity_mismatch_waning(top1_mediafile, top2_mediafile, top3_mediafile, top_k_labels)


def _identity_mismatch_waning(top1_mediafile: MediaFile, top2_mediafile: MediaFile, top3_mediafile: MediaFile, top_k_labels:list) -> None:
    """Warn if the identity mismatch is detected."""
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
