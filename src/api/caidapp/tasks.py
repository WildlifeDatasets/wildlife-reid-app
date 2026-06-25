import ast
import copy
import datetime
import json
import logging
import os
import os.path
import shutil
import tempfile
import traceback
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Generator
from zoneinfo import ZoneInfo

import django
import numpy as np
import pandas as pd
import tqdm
from celery import current_app, shared_task, signature
from django.conf import settings
from django.utils.timezone import now

from . import fs_data, model_tools, models
from .fs_data import make_thumbnail_from_file
from .log_tools import StatusCounts
from .model_extra import compute_identity_suggestions
from .models import (
    CaIDUser,
    IndividualIdentity,
    Locality,
    MediaFile,
    MediafilesForIdentification,
    UploadedArchive,
    WorkGroup,
    get_locality,
    get_taxon,
    get_unique_code,
    get_unique_name,
    user_has_access_filter_params,
)

# from joblib import Parallel, delayed
# from tqdm import tqdm

logger = logging.getLogger("app")


def resolve_identification_selection(
    workgroup: WorkGroup,
    uploaded_archive: UploadedArchive | None = None,
    selection: dict | None = None,
):
    """Return the queryset and resolved taxon settings for one identification run."""
    selection = selection or {}

    selection_uploaded_archive_ids = selection.get("uploaded_archive_ids")
    if selection_uploaded_archive_ids is None and uploaded_archive is not None:
        selection_uploaded_archive_ids = [uploaded_archive.id]

    observation_taxon = selection.get("observation_taxon")
    if observation_taxon is None and uploaded_archive is not None:
        observation_taxon = uploaded_archive.taxon_for_identification
    if observation_taxon is None and workgroup.default_taxon_for_identification and workgroup.check_taxon_before_identification:
        observation_taxon = workgroup.default_taxon_for_identification

    require_observations = selection.get("require_observations")
    if require_observations is None:
        require_observations = observation_taxon is not None

    mediafiles = workgroup.mediafiles_for_identification(
        uploaded_archive_ids=selection_uploaded_archive_ids,
        sequence_ids=selection.get("sequence_ids"),
        mediafile_ids=selection.get("mediafile_ids"),
        require_import_finished=selection.get("require_import_finished", True),
        observation_taxon=observation_taxon,
        require_identity=selection.get("require_identity", False),
        require_observations=require_observations,
    )
    return mediafiles, observation_taxon, require_observations


def clear_identification_queue_for_uploaded_archive(uploaded_archive: UploadedArchive) -> tuple[int, int]:
    """Delete queued manual identification items and suggestions for one upload."""
    queue_qs = MediafilesForIdentification.objects.filter(mediafile__parent=uploaded_archive)
    deleted_queue_count = queue_qs.count()
    deleted_suggestion_count = models.MediafileIdentificationSuggestion.objects.filter(
        for_identification__in=queue_qs
    ).count()
    queue_qs.delete()
    return deleted_queue_count, deleted_suggestion_count


def get_uploaded_archives_pending_identification(workgroup: WorkGroup):
    """Return identification uploads that still have at least one eligible unidentified media file."""
    candidate_archives = UploadedArchive.objects.filter(
        owner__workgroup=workgroup,
        is_for_identification=True,
        import_finished=True,
    ).order_by("uploaded_at", "id")

    eligible_archive_ids = []
    for uploaded_archive in candidate_archives:
        mediafiles, _, _ = resolve_identification_selection(workgroup, uploaded_archive=uploaded_archive)
        if mediafiles.exists():
            eligible_archive_ids.append(uploaded_archive.id)

    return candidate_archives.filter(id__in=eligible_archive_ids)


def _task_log_context(task_name: str, task_id: str | None = None, extra: dict | None = None) -> str:
    """Build a compact task context string for debugging task handoffs."""
    parts = [
        f"task={task_name}",
        f"pid={os.getpid()}",
        f"thread={threading.get_ident()}",
    ]
    if task_id:
        parts.append(f"task_id={task_id}")
    if extra:
        parts.extend(f"{key}={value!r}" for key, value in extra.items())
    return " ".join(parts)


"""
Celery tasks used by "API worker" inside of the API docker container.
The API worker is different from e.g. Inference worker with its own docker container
because it has access to the database and other django resources,
and functions as a queue for processing worker responses.
"""


class DuplicateFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_log = None

    def filter(self, record):
        """Filter out duplicate log messages."""
        record.lineno
        current_log = (record.module, record.levelno, record.msg)
        if current_log != self.last_log:
            self.last_log = current_log
            return True
        return False


@shared_task(bind=True)
def on_success_predict_taxon(
    self,
    output: dict,
    *args,
    uploaded_archive_id: int,
    zip_file: str,
    csv_file: str,
    extract_identites: bool = False,
    **kwargs,
):
    """Import media files after running predict function in taxon worker.

    This function is called after the taxon classification is finished. After this function the detection is called.
    """
    status = output.get("status", "unknown")
    logger.info(
        "Entering on_success_predict_taxon: %s",
        _task_log_context(
            "on_success_predict_taxon",
            getattr(getattr(self, "request", None), "id", None),
            {
                "uploaded_archive_id": uploaded_archive_id,
                "status": status,
                "extract_identites": extract_identites,
                "zip_file": zip_file,
                "csv_file": csv_file,
            },
        ),
    )
    print(f"Taxon classification finished with status {status}")
    logger.info(f"Taxon classification finished with status '{status}'. Updating database record.")
    try:
        logger.debug(
            "Loading UploadedArchive for taxon callback: %s",
            _task_log_context(
                "on_success_predict_taxon",
                getattr(getattr(self, "request", None), "id", None),
                {"uploaded_archive_id": uploaded_archive_id},
            ),
        )
        uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
        logger.debug(
            "Loaded UploadedArchive for taxon callback: %s",
            _task_log_context(
                "on_success_predict_taxon",
                getattr(getattr(self, "request", None), "id", None),
                {
                    "uploaded_archive_id": uploaded_archive_id,
                    "current_taxon_status": uploaded_archive.taxon_status,
                    "current_identification_status": uploaded_archive.identification_status,
                },
            ),
        )

        if "status" not in output:
            logger.critical(f"Unexpected error {output=} is missing 'status' field.")
            uploaded_archive.taxon_status = "U"
            uploaded_archive.identification = "U"
        elif output["status"] == "DONE":
            logger.debug(
                "Taxon callback output is DONE: %s",
                _task_log_context(
                    "on_success_predict_taxon",
                    getattr(getattr(self, "request", None), "id", None),
                    {
                        "uploaded_archive_id": uploaded_archive_id,
                        "output_keys": sorted(output.keys()),
                    },
                ),
            )
            uploaded_archive.zip_file = zip_file
            uploaded_archive.csv_file = csv_file
            uploaded_archive.import_error_spreadsheet = str(Path(csv_file).with_suffix(".failed.csv"))
            logger.debug("Creating thumbnail for uploaded archive %s", uploaded_archive_id)
            make_thumbnail_for_uploaded_archive(uploaded_archive)
            # update_metadata_csv_by_uploaded_archive(uploaded_archive)
            # create missing take effect only if the processing is done for the first time
            # in other cases the file should be removed from CSV before the processing is run
            logger.debug(f"{uploaded_archive.contains_identities=}")
            logger.debug(
                "Updating UploadedArchive from metadata csv: %s",
                _task_log_context(
                    "on_success_predict_taxon",
                    getattr(getattr(self, "request", None), "id", None),
                    {
                        "uploaded_archive_id": uploaded_archive_id,
                        "create_missing": True,
                        "extract_identites": extract_identites,
                    },
                ),
            )
            update_uploaded_archive_by_metadata_csv(
                uploaded_archive, create_missing=True, extract_identites=extract_identites
            )
            if uploaded_archive.taxon_for_identification:
                logger.debug(
                    "Assigning unidentified media for identification: %s",
                    _task_log_context(
                        "on_success_predict_taxon",
                        getattr(getattr(self, "request", None), "id", None),
                        {"uploaded_archive_id": uploaded_archive_id},
                    ),
                )
                assign_unidentified_to_identification(caiduser=uploaded_archive.owner)
            uploaded_archive.import_finished = True
            uploaded_archive.is_for_identification = (
                (uploaded_archive.taxon_for_identification is not None) or
                uploaded_archive.contains_single_taxon or
                uploaded_archive.contains_identities or
                uploaded_archive.is_for_identification
            )
            uploaded_archive.taxon_status = "TAID"
            uploaded_archive.identification_status = "IR"  # Ready for identification
            uploaded_archive.status_message = "Taxon classification finished."
            uploaded_archive.finished_at = django.utils.timezone.now()
            logger.debug(
                "Saving UploadedArchive after taxon callback: %s",
                _task_log_context(
                    "on_success_predict_taxon",
                    getattr(getattr(self, "request", None), "id", None),
                    {"uploaded_archive_id": uploaded_archive_id},
                ),
            )
            uploaded_archive.save()
            schedule_init_identification_after_representative_upload(uploaded_archive)
            logger.debug("Updating capture time range for uploaded archive %s", uploaded_archive_id)
            uploaded_archive.update_earliest_and_latest_captured_at()
            logger.debug("Creating sequences for uploaded archive %s", uploaded_archive_id)
            uploaded_archive.make_sequences()
            logger.debug("Running async detection on success taxon classification")
            # run_detection_async(uploaded_archive)  # this is probably not necessary
        else:
            logger.warning(
                "Taxon callback received non-DONE status: %s",
                _task_log_context(
                    "on_success_predict_taxon",
                    getattr(getattr(self, "request", None), "id", None),
                    {"uploaded_archive_id": uploaded_archive_id, "status": output.get("status")},
                ),
            )
            uploaded_archive.taxon_status = "F"
            uploaded_archive.identification_status = "F"
            uploaded_archive.finished_at = django.utils.timezone.now()
            if "error" in output:
                logger.error(f"{output['error']=}")
                uploaded_archive.status_message = output["error"]

                nt_kwargs = dict(
                    message=f"Taxon classification failed: {output['error']}",
                    level=models.Notification.ERROR,
                )

                if uploaded_archive.owner.workgroup:
                    nt_kwargs["workgroups"] = [uploaded_archive.owner.workgroup]

                else:
                    nt_kwargs["users"] = [uploaded_archive.owner]
                models.Notification.create_for(**nt_kwargs)
            uploaded_archive.save()
    except Exception as e:
        logger.debug(str(traceback.format_exc()))
        logger.error(
            "Error during on_success_predict_taxon: %s | %s",
            e,
            _task_log_context(
                "on_success_predict_taxon",
                getattr(getattr(self, "request", None), "id", None),
                {"uploaded_archive_id": uploaded_archive_id, "status": status},
            ),
        )
        uploaded_archive.taxon_status = "F"
        uploaded_archive.finished_at = django.utils.timezone.now()
        uploaded_archive.status_message = str(traceback.format_exc())
        uploaded_archive.save()


def _prepare_dataframe_for_identification(mediafiles) -> dict:
    media_root = Path(settings.MEDIA_ROOT)
    csv_len = len(mediafiles)
    csv_data = {
        "image_path": [None] * csv_len,
        "mediafile_id": [None] * csv_len,
        "class_id": [None] * csv_len,
        "label": [None] * csv_len,
        "locality_id": [None] * csv_len,
        "locality_name": [None] * csv_len,
        "locality_coordinates": [None] * csv_len,
        "detection_results": [None] * csv_len,
        "sequence_number": [None] * csv_len,
    }
    logger.debug(f"number of records={len(mediafiles)}")
    for i, mediafile in enumerate(mediafiles):
        # if mediafile.identity is not None:
        identity = _get_identification_identity(mediafile)
        csv_data["image_path"][i] = _get_identification_source_image_path(mediafile, media_root)
        csv_data["mediafile_id"][i] = mediafile.id
        csv_data["class_id"][i] = int(identity.id) if identity else None
        csv_data["label"][i] = str(identity.name) if identity else None
        csv_data["locality_id"][i] = int(mediafile.locality.id) if mediafile.locality else None
        csv_data["locality_name"][i] = str(mediafile.locality.name) if mediafile.locality else ""
        csv_data["locality_coordinates"][i] = (
            str(mediafile.effective_location) if mediafile.effective_location else None
        )
        csv_data["sequence_number"][i] = mediafile.sequence.local_id if mediafile.sequence else None
        # logger.debug(f"{mediafile.metadata_json=}")
        if mediafile.metadata_json and "detection_results" in mediafile.metadata_json:
            detection_results = mediafile.metadata_json["detection_results"]
        else:
            detection_results = None
        csv_data["detection_results"][i] = detection_results

    return csv_data


def _get_identification_identity(mediafile: MediaFile) -> IndividualIdentity | None:
    """Return the identity that should label a mediafile in identification metadata."""
    if mediafile.identity:
        return mediafile.identity

    representative_observation = (
        mediafile.observations.filter(identity_is_representative=True, identity__isnull=False)
        .order_by("id")
        .first()
    )
    if representative_observation:
        return representative_observation.identity

    return mediafile.identity_from_observations


def _get_identification_source_image_path(mediafile: MediaFile, media_root: Path) -> str:
    """Return an image path suitable for identification embedding extraction."""
    if mediafile.media_type != "video":
        return str(media_root / mediafile.image_file.name)

    static_thumbnail_name = getattr(mediafile.static_thumbnail, "name", "")
    static_thumbnail_path = media_root / static_thumbnail_name if static_thumbnail_name else None
    if static_thumbnail_path and static_thumbnail_path.exists():
        return str(static_thumbnail_path)

    logger.info("Generating missing static thumbnail for video mediafile %s before identification init.", mediafile.id)
    mediafile.make_thumbnail_for_mediafile_if_necessary()
    mediafile.refresh_from_db(fields=["static_thumbnail"])

    static_thumbnail_name = getattr(mediafile.static_thumbnail, "name", "")
    static_thumbnail_path = media_root / static_thumbnail_name if static_thumbnail_name else None
    if static_thumbnail_path and static_thumbnail_path.exists():
        return str(static_thumbnail_path)

    raise FileNotFoundError(
        f"Video mediafile {mediafile.id} has no readable static thumbnail for identification."
    )


def count_identification_media_types(mediafiles) -> tuple[int, int]:
    """Return image and video counts for an identification queryset."""
    return mediafiles.filter(media_type="image").count(), mediafiles.filter(media_type="video").count()


def create_identification_run_statistic(
    workgroup: WorkGroup,
    operation: str,
    image_number: int,
    video_number: int,
) -> models.IdentificationRunStatistic:
    """Create a lightweight duration/progress audit row for one identification worker run."""
    return models.IdentificationRunStatistic.objects.create(
        workgroup=workgroup,
        operation=operation,
        image_number=image_number,
        video_number=video_number,
        status="started",
    )


def finish_identification_run_statistic(statistic_id: int | None, status: str, task_id: str = ""):
    """Finish an identification run statistic without letting audit failures affect callbacks."""
    if not statistic_id:
        return
    try:
        statistic = models.IdentificationRunStatistic.objects.get(id=statistic_id)
        finished_at = now()
        statistic.finished_at = finished_at
        statistic.duration_seconds = max((finished_at - statistic.created_at).total_seconds(), 0)
        statistic.status = status
        if task_id:
            statistic.task_id = task_id
        statistic.save(update_fields=["finished_at", "duration_seconds", "status", "task_id"])
    except Exception:
        logger.warning("Could not finish identification run statistic %s", statistic_id, exc_info=True)


# def run_taxon_classification_async(uploaded_archive: UploadedArchive, link=None, link_error=None):
#     """Run taxon classification asynchronously."""
#
#     if link_error is None:
#         link_error = on_error_with_uploaded_archive.s()
#
#
#
#     sig = signature(
#         "predict",
#         kwargs={
#             "input_archive_file": str(
#                 Path(settings.MEDIA_ROOT) / uploaded_archive.archivefile.name
#             ),
#             "output_dir": str(output_dir),
#             "output_archive_file": str(output_archive_file),
#             "output_metadata_file": str(output_metadata_file),
#             "contains_identities": uploaded_archive.contains_identities,
#         },
#     )


@shared_task(bind=True)
def do_cloud_import_for_user(
    self,
    caiduser_id: int,
    contains_single_taxon: bool = False,
    contains_identities: bool = False,
):
    """Import files from cloud storage."""
    from .models import CaIDUser

    # Retrieve the CaIDUser instance
    caiduser = CaIDUser.objects.get(id=caiduser_id)

    path = Path(caiduser.import_dir)
    imported_dir = path / "_trash_bin" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    imported_dir.mkdir(exist_ok=True, parents=True)
    dirs_to_be_deleted = []
    for yield_dict in _iterate_over_locality_checks(path, caiduser):

        if yield_dict.parent_dir_to_be_deleted:
            dirs_to_be_deleted.append(yield_dict)
            continue

        if yield_dict.is_already_processed:
            continue

        # make zip from dir
        uploaded_archive = UploadedArchive.objects.create(
            owner=caiduser,
            # archivefile=zip_name,
            contains_single_taxon=False,
            contains_identities=False,
            taxon_status="C",
            uploaded_at=django.utils.timezone.now(),
        )
        logger.debug(f"{yield_dict.path_of_locality_check=}, {yield_dict.path_of_locality_check.exists()=}")
        uploaded_archive.save()
        zip_path = model_tools.get_zip_path_in_unique_folder(uploaded_archive, yield_dict.zip_name)
        zip_path_absolute = Path(settings.MEDIA_ROOT) / zip_path
        logger.debug(f"{zip_path=}, {zip_path_absolute=}")
        if yield_dict.path_of_locality_check.is_dir():
            make_zipfile(zip_path_absolute, yield_dict.path_of_locality_check)
        else:
            # if it is a file, copy it
            zip_path_absolute.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(yield_dict.path_of_locality_check, zip_path_absolute)
        if yield_dict.locality and len(yield_dict.locality) > 0:
            locality = get_locality(caiduser, yield_dict.locality)
            uploaded_archive.locality_at_upload_object = locality
            uploaded_archive.locality_at_upload = yield_dict.locality
        uploaded_archive.contains_identities = contains_identities
        uploaded_archive.archivefile = zip_path
        uploaded_archive.save()
        logger.debug("Zip file created. Ready to start processing.")
        run_species_prediction_async(uploaded_archive, extract_identites=False)

        # move imported files to _imported directory with subdirectory with "now"
        relative_path = yield_dict.path_of_locality_check.relative_to(path)
        imported_path = imported_dir / relative_path
        # move directory
        shutil.move(yield_dict.path_of_locality_check, imported_path)
    for dir_to_be_deleted in dirs_to_be_deleted:
        # rmdir ignore errors
        shutil.rmtree(dir_to_be_deleted.path_of_locality_check, ignore_errors=True)
        # dir_to_be_deleted.path_of_locality_check.rmdir(ignore_errors=True)
    # move imported files to processed directory
    caiduser.dir_import_status = "Finished"
    caiduser.save()


# @shared_task(bind=True)
def do_cloud_import_for_user_async(
    caiduser: CaIDUser, contains_identities: bool = False, contains_single_taxon: bool = False
):
    """Run cloud import asynchronously."""
    # sig = do_cloud_import_for_user.s(caiduser=caiduser)
    # run async
    # sig.apply_async()
    caiduser.dir_import_status = "Processing"
    caiduser.save()
    sig = signature(
        "caidapp.tasks.do_cloud_import_for_user",
        kwargs={
            "caiduser_id": caiduser.id,
            "contains_single_taxon": contains_single_taxon,
            "contains_identities": contains_identities,
        },
    )
    sig.apply_async()


def _create_mediafiles_zip_file(mediafiles, abs_zip_path, metadata_records=None):
    """Create a zip file for media files in the background."""
    logger.debug(f"Creating zip file for {len(mediafiles)} media files.")
    abs_zip_path = Path(abs_zip_path)
    abs_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        mediafiles_dir = Path(tmpdirname) / "images"
        mediafiles_dir.mkdir()
        if metadata_records:
            metadata_df = pd.DataFrame.from_records(metadata_records)
            metadata_df = model_tools.convert_datetime_to_naive(metadata_df)
            metadata_df.to_csv(mediafiles_dir / "metadata.csv", index=False)
            with pd.ExcelWriter(mediafiles_dir / "metadata.xlsx", engine="openpyxl") as writer:
                metadata_df.to_excel(writer, index=False, sheet_name="Media files")
        for mediafile in tqdm.tqdm(mediafiles):
            src = Path(settings.MEDIA_ROOT) / mediafile["path"]
            dst = mediafiles_dir / mediafile["output_name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        # Assume `make_zipfile` is your custom function to create a zip
        logger.debug(f"Creating zip file {abs_zip_path}")
        make_zipfile(abs_zip_path, mediafiles_dir)
    logger.debug(f"Zip file created: {abs_zip_path}")
    return str(abs_zip_path)


@shared_task
def create_mediafiles_zip(user_hash, mediafiles, abs_zip_path):
    """Create a zip file for media files in the background."""
    return _create_mediafiles_zip_file(mediafiles, abs_zip_path)


@shared_task
def create_mediafiles_zip_with_metadata(user_hash, mediafiles, abs_zip_path, metadata_records):
    """Create a zip file with media files and metadata in the background."""
    return _create_mediafiles_zip_file(mediafiles, abs_zip_path, metadata_records)


@shared_task
def clean_old_mediafile_zips(dirpath: str, glob_pattern: str = "mediafiles_*.zip", max_age_days: int = 7):
    """Clean up zpped mediafiles older than a week files in the specified directory."""
    logger.debug(f"Cleaning up old mediafile zips in {dirpath}")
    dirpath = Path(dirpath)
    if not dirpath.exists():
        logger.warning(f"Directory {dirpath} does not exist. Skipping cleanup.")
        return

    now = datetime.datetime.now()
    for file in dirpath.glob(glob_pattern):
        file_age = now - datetime.datetime.fromtimestamp(file.stat().st_mtime)
        if file_age > datetime.timedelta(days=max_age_days):
            logger.debug(f"Removing old zip file: {file}")
            file.unlink()
        else:
            logger.debug(f"Keeping recent zip file: {file}")


def make_zipfile(output_filename: Path, source_dir: Path):
    """Make archive (zip, tar.gz) from a folder.

    Parameters
    ----------
    output_filename: Path of output file
    source_dir: Path to input directory
    """
    import shutil

    output_filename = Path(output_filename)
    source_dir = Path(source_dir)
    archive_type = "zip"

    shutil.make_archive(output_filename.parent / output_filename.stem, archive_type, root_dir=source_dir)


# remove this function. The 'detect' function does not exist anymore.
# def run_detection_async(uploaded_archive: UploadedArchive, link=None, link_error=None):
#     """Run detection and mask preparation on UploadedArchive."""
#     mediafiles = uploaded_archive.mediafile_set.all()
#     logger.debug(f"Running detection with {len(mediafiles)} records...")
#
#     csv_data = _prepare_dataframe_for_identification(mediafiles)
#     media_root = Path(settings.MEDIA_ROOT)
#     identity_metadata_file = media_root / uploaded_archive.outputdir / "detection_metadata.csv"
#     cropped_identity_metadata_file = (
#         media_root / uploaded_archive.outputdir / "detection_metadata.csv"
#     )
#     pd.DataFrame(csv_data).to_csv(identity_metadata_file, index=False)
#
#     # logger.debug("Calling run_detection and run_identification ...")
#     detect_sig = signature(
#         "detect",
#         kwargs={
#             "input_metadata_path": str(identity_metadata_file),
#             "output_metadata_path": str(cropped_identity_metadata_file),
#         },
#     )
#     tasks = chain(
#         detect_sig,
#         # simple_log_sig,
#         # identify_sig,
#     )
#     tasks.apply_async(
#         # link=identify_on_success.s(
#         link=detection_on_success_after_species_prediction.s(
#             # csv file should contain image_path, class_id, label
#             # input_metadata_file_path=str(identity_metadata_file),
#             # organization_id=uploaded_archive.owner.workgroup.id,
#             # output_json_file_path=str(output_json_file),
#             # top_k=3,
#             uploaded_archive_id=uploaded_archive.id,
#             # mediafiles=mediafiles,
#             # metadata_file=str(identity_metadata_file),
#             # mediafile_ids=mediafile_ids
#             # zip_file=os.path.relpath(str(output_archive_file), settings.MEDIA_ROOT),
#             # csv_file=os.path.relpath(str(output_metadata_file), settings.MEDIA_ROOT),
#         ),
#         link_error=on_error_with_uploaded_archive.s(
#             # uploaded_archive_id=uploaded_archive.id
#         ),
#     )
#


@shared_task(bind=True)
def on_error_with_uploaded_archive(self, task_id: str, *args, uploaded_archive_id: int, **kwargs):
    """Error callback invoked after running predict function in inference worker."""
    logger.critical(f"Worker task with id '{task_id}' failed due to unexpected internal error.")
    logger.debug(f"self={self}")
    logger.debug(f"args={args}")
    logger.debug(f"kwargs={kwargs}")
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    uploaded_archive.taxon_status = "F"
    uploaded_archive.finished_at = django.utils.timezone.now()
    uploaded_archive.save()
    result = self.AsyncResult(task_id)
    error_message = result.result if result.failed() else "No error message available"
    logger.error(f"Detection error message: {error_message}")


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


def run_species_prediction_async(
    uploaded_archive: UploadedArchive,
    link=None,
    link_error=None,
    extract_identites: bool = False,
    force_init: bool = False,
):
    """Run species prediction asynchronously."""
    try:
        logger.info(
            "Initializing taxon prediction: %s",
            _task_log_context(
                "run_species_prediction_async",
                extra={
                    "uploaded_archive_id": uploaded_archive.id,
                    "force_init": force_init,
                    "extract_identites": extract_identites,
                    "contains_identities": uploaded_archive.contains_identities,
                },
            ),
        )
        _run_taxon_classification_init_message(uploaded_archive, commit=False)
        output_archive_file, output_dir, output_metadata_file = _run_taxon_classification_init(
            uploaded_archive, commit=True
        )
    except Exception as e:
        logger.error(f"Error during init: {e}")
        import traceback

        uploaded_archive.taxon_status = "F"
        uploaded_archive.status_message = traceback.format_exc()
        uploaded_archive.save()
        return
    logger.debug(f"updating uploaded archive, {uploaded_archive.csv_file=}")
    # csv_file = Path(settings.MEDIA_ROOT) / str(uploaded_archive.csv_file)
    # logger.debug(f"{csv_file} {Path(csv_file).exists()}")

    # output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir

    logger.debug(f"{output_metadata_file=}, {output_metadata_file.exists()=}")
    if Path(output_metadata_file).exists():
        df = pd.read_csv(output_metadata_file, index_col=0)
        logger.debug(f"{len(df)=}")
    else:
        logger.warning(
            "Output metadata file does not exist before dispatch: %s",
            _task_log_context(
                "run_species_prediction_async",
                extra={
                    "uploaded_archive_id": uploaded_archive.id,
                    "output_metadata_file": str(output_metadata_file),
                },
            ),
        )

    if link is None:
        logger.debug("setting default link for run_species_prediction_async to ")
        link = (
            on_success_predict_taxon.s(
                uploaded_archive_id=uploaded_archive.id,
                zip_file=os.path.relpath(str(output_archive_file), settings.MEDIA_ROOT),
                csv_file=os.path.relpath(str(output_metadata_file), settings.MEDIA_ROOT),
                extract_identites=extract_identites,
            ),
        )
    if link_error is None:
        link_error = (on_error_with_uploaded_archive.s(uploaded_archive_id=uploaded_archive.id),)

    if force_init:
        # remove all mediafiles
        uploaded_archive.mediafile_set.all().delete()
    else:
        # if the metadata file exists, it is updated
        update_metadata_csv_by_uploaded_archive(uploaded_archive)

    if uploaded_archive.owner.workgroup is not None:
        sequence_time_limit_s = uploaded_archive.owner.workgroup.sequence_time_limit
        detection_model_path = uploaded_archive.owner.workgroup.detection_model_path
        detection_model_architecture = uploaded_archive.owner.workgroup.detection_model_architecture
    else:
        sequence_time_limit_s = 120
        detection_model_path = None
        detection_model_architecture = None

    # send celery message to the data worker
    logger.info("Sending request to inference worker.")
    logger.debug(f"{uploaded_archive.contains_identities=}")
    sig = signature(
        "predict",
        kwargs={
            "input_archive_file": str(Path(settings.MEDIA_ROOT) / uploaded_archive.archivefile.name),
            "output_dir": str(output_dir),
            "output_archive_file": str(output_archive_file),
            "output_metadata_file": str(output_metadata_file),
            "contains_identities": uploaded_archive.contains_identities,
            "force_init": force_init,
            "sequence_time_limit_s": sequence_time_limit_s,
            "detection_model_path": detection_model_path,
            "detection_model_architecture": detection_model_architecture,
            "path_structure_regex": uploaded_archive.path_structure_regex or None,
            "path_structure_mapping": uploaded_archive.import_mapping.get("directory_mapping") or None,
        },
    )

    logger.debug(f"{link=}, {link_error=}")
    task = sig.apply_async(
        link=link,
        link_error=link_error,
    )
    uploaded_archive.taxon_task_id = task.id
    uploaded_archive.save(update_fields=["taxon_task_id"])
    logger.info(
        "Created worker task: %s",
        _task_log_context(
            "run_species_prediction_async",
            task.task_id,
            {
                "uploaded_archive_id": uploaded_archive.id,
                "queue_task": "predict",
                "output_dir": str(output_dir),
            },
        ),
    )


def _run_taxon_classification_init_message(uploaded_archive: UploadedArchive, commit: bool = False):
    expected_time_message = timedelta_to_human_readable(
        _estimate_time_for_taxon_classification_of_uploaded_archive(uploaded_archive)
    )
    logger.debug(f"{expected_time_message=}")
    uploaded_archive.taxon_status = "TAIP"
    uploaded_archive.identification_status = "TAIP"
    uploaded_archive.status_message = "Processing will be done " + expected_time_message

    if commit:
        uploaded_archive.save()


def _run_taxon_classification_init(uploaded_archive, commit: bool = False):
    # update record in the database
    output_dir = Path(settings.MEDIA_ROOT) / uploaded_archive.outputdir
    uploaded_archive.started_at = django.utils.timezone.now()
    output_archive_file = output_dir / "images.zip"
    output_metadata_file = output_dir / "metadata.csv"
    uploaded_archive.csv_file = str(Path(uploaded_archive.outputdir) / "metadata.csv")
    if commit:
        uploaded_archive.save()
    return output_archive_file, output_dir, output_metadata_file


def _estimate_time_for_taxon_classification_of_uploaded_archive(
    uploaded_archive: UploadedArchive,
) -> datetime.timedelta:
    """Estimate time to process archive."""
    # count files in archive
    # file_count_dict = count_files_in_archive(uploaded_archive.archivefile.path)
    file_count_dict = uploaded_archive.number_of_media_files_in_archive()
    # file_count = file_count_dict["file_count"]
    image_count = file_count_dict["image_count"]
    video_count = file_count_dict["video_count"]
    # estimate time to process
    # it is time for taxon classification + detection + segmentation
    # on CPU 22s per image, 0.1s per image on GPU
    time_per_image = datetime.timedelta(seconds=0.5)  # detection 0.1, exif 0.25, taxon 0.01
    time_per_video = datetime.timedelta(seconds=60)

    time_to_process = datetime.timedelta(seconds=10) + ((time_per_image * image_count) + (time_per_video * video_count))
    logger.debug(f"{time_to_process=}, {file_count_dict=}")
    return time_to_process


def timedelta_to_human_readable(timedelta: datetime.timedelta) -> str:
    """Convert timedelta to human readable string."""
    # Convert time_to_process into a human-readable format
    total_seconds = timedelta.total_seconds()
    if total_seconds < 60:
        return "in a few seconds"
    elif total_seconds < 3600:
        minutes = int(total_seconds / 60)
        if minutes == 1:
            return "in a minute"
        else:
            return f"in {minutes} minutes"
    else:
        hours = int(total_seconds / 3600)
        if hours == 1:
            return "in an hour"
        else:
            return f"in {hours} hours"


def _get_rel_and_abs_paths_based_on_csv_row(row: dict, output_dir: Path):
    abs_pth = output_dir / "images" / row["image_path"]
    rel_pth = os.path.relpath(abs_pth, settings.MEDIA_ROOT)
    if "vanilla_Path" in row:
        logger.debug(f"original_path={row['original_path']}")
    logger.debug(f"relative_pth={rel_pth}")

    return rel_pth, abs_pth


MEDIAFILE_VARIANT_CSV_COLUMNS = {
    "preview_path": "preview",
    "thumbnail_path": "thumbnail",
    "static_thumbnail_path": "static_thumbnail",
}


def _is_missing_metadata_value(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return isinstance(value, str) and value.strip() == ""


def _resolve_variant_rel_path(value, output_dir: Path) -> str | None:
    if _is_missing_metadata_value(value):
        return None

    variant_path = Path(str(value))
    if variant_path.is_absolute():
        abs_path = variant_path
    else:
        upload_relative_path = output_dir / variant_path
        media_root_relative_path = Path(settings.MEDIA_ROOT) / variant_path
        if upload_relative_path.exists() or not media_root_relative_path.exists():
            abs_path = upload_relative_path
        else:
            abs_path = media_root_relative_path

    if not abs_path.exists():
        return None
    return os.path.relpath(abs_path, settings.MEDIA_ROOT).replace("\\", "/")


def _apply_prepared_mediafile_variants(mf: MediaFile, row, output_dir: Path) -> bool:
    changed_fields = []
    for column_name, field_name in MEDIAFILE_VARIANT_CSV_COLUMNS.items():
        if column_name not in row:
            continue
        rel_path = _resolve_variant_rel_path(row[column_name], output_dir)
        if rel_path is None:
            continue
        field_file = getattr(mf, field_name)
        if field_file.name != rel_path:
            setattr(mf, field_name, rel_path)
            changed_fields.append(field_name)

    if changed_fields:
        mf.save(update_fields=changed_fields)
        return True
    return False


def _mediafile_variants_exist(mf: MediaFile) -> bool:
    for field_name in MEDIAFILE_VARIANT_CSV_COLUMNS.values():
        field_file = getattr(mf, field_name)
        if not field_file or not field_file.name:
            return False
        if not (Path(settings.MEDIA_ROOT) / field_file.name).exists():
            return False
    return True


def update_uploaded_archive_by_metadata_csv(
    uploaded_archive: UploadedArchive,
    thumbnail_width: int = 400,
    create_missing: bool = True,
    extract_identites: bool = False,
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

    locality = get_locality(uploaded_archive.owner, str(uploaded_archive.locality_at_upload))

    status_counts = StatusCounts()
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Updating database"):
        try:
            status = _update_database_by_one_row_of_metadata(
                df,
                index,
                row,
                create_missing,
                extract_identites,
                locality,
                output_dir,
                thumbnail_width,
                uploaded_archive,
            )
            status_counts.increment(status)
        except Exception:
            logger.error(f"Error during processing row {index}: {row}")
            logger.error(traceback.format_exc())
            status_counts.increment("error")

    logger.debug(f"{status_counts=}")
    # parallel calculation have problem:
    # joblib.externals.loky.process_executor.BrokenProcessPool:
    # A task has failed to un-serialize. Please ensure that the arguments of the function
    # are all picklable.

    # num_cores = multiprocessing.cpu_count()
    # Parallel(n_jobs=num_cores)(
    #    delayed(_update_database_by_one_row_of_metadata)
    #    (df, index, row, create_missing, extract_identites, locality,
    #    output_dir, thumbnail_width, uploaded_archive)
    #    for index, row in tqdm(df.iterrows())
    # )

    # get minimum and maximum datetime from df["datetime"]
    # convert datetime as string to datetime object
    # starts_at = pd.to_datetime(df["datetime"]).min()
    # ends_at = pd.to_datetime(df["datetime"]).max()
    # logger.debug(f"{starts_at=}, {ends_at=}")
    # uploaded_archive.starts_at = str(starts_at)
    # uploaded_archive.ends_at = str(ends_at)
    uploaded_archive.locality_at_upload_object = uploaded_archive.locality
    uploaded_archive.save(update_fields=["locality_at_upload_object"])


def _update_database_by_one_row_of_metadata(
    df,
    index,
    row,
    create_missing,
    extract_identites,
    locality,
    output_dir,
    thumbnail_width,
    uploaded_archive,
    orientation_score_threshold=0.5,
) -> str:
    # rel_pth, _ = _get_rel_and_abs_paths_based_on_csv_row(row, output_dir)
    logger.debug(f"Processing row {index}")
    print(f"Processing row {index}")
    image_abs_pth = output_dir / "images" / row["image_path"]
    image_rel_pth = image_abs_pth.relative_to(settings.MEDIA_ROOT)
    media_abs_pth = Path(row["absolute_media_path"])
    media_rel_pth = media_abs_pth.relative_to(settings.MEDIA_ROOT)
    captured_at = row["datetime"]
    row_locality = get_locality_from_metadata_row(uploaded_archive, row, locality)
    # if no timzone is given, we assume it is the local time zone
    try:
        # captured_at = pd.to_datetime(captured_at, utc=True)
        captured_at = pd.to_datetime(captured_at)
        # Pokud chybí časová zóna, nastav aktuální časovou zónu
        # If the TZ is missing, use the local timezone
        if captured_at.tzinfo is None or captured_at.tzinfo.utcoffset(captured_at) is None:
            # turn timezone in string into timezone object
            # local_timezone = pytz.timezone(settings.TIME_ZONE)
            # local_timezone = django.utils.timezone.get_current_timezone()

            # captured_at = local_timezone.localize(captured_at)
            local_timezone = ZoneInfo(settings.TIME_ZONE)
            captured_at = captured_at.replace(tzinfo=local_timezone)
    except Exception as e:
        # logger.debug(f"{captured_at=}")
        logger.debug(str(row))
        logger.warning(traceback.format_exc())
        logger.error(f"Error during parsing datetime: {e}")
        captured_at = None

    # Pokud je captured_at prázdný řetězec, NaN nebo NaT, nastavíme na None
    if (captured_at == "") or (isinstance(captured_at, float) and np.isnan(captured_at)) or pd.isnull(captured_at):
        captured_at = None

    def _get_row_value(column_name):
        if column_name not in row:
            return None
        value = row[column_name]
        if value is None or pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return value

    def _parse_mediafile_location():
        latitude = _get_row_value("latitude")
        longitude = _get_row_value("longitude")
        if latitude is None or longitude is None:
            return None
        try:
            lat = round(float(latitude), 3)
            lon = round(float(longitude), 3)
        except (TypeError, ValueError):
            logger.warning("Invalid latitude/longitude in row %s: lat=%s lon=%s", index, latitude, longitude)
            return None
        return f"{lat},{lon}"

    def _resolve_identity():
        unique_name = _get_row_value("unique_name")
        code = _get_row_value("code")
        juv_code = _get_row_value("juv_code")
        if unique_name is None and code is None and juv_code is None:
            return None

        workgroup = uploaded_archive.owner.workgroup
        identity = None
        if code is not None:
            identity = get_unique_code(code, workgroup=workgroup)
        elif unique_name is not None:
            identity = get_unique_name(str(unique_name), workgroup=workgroup)

        if identity is None:
            return None

        identity_updated = False
        if unique_name is not None and identity.name != str(unique_name):
            identity.name = str(unique_name)[:100]
            identity_updated = True
        if code is not None and identity.code != str(code):
            identity.code = str(code)[:50]
            identity_updated = True
        if juv_code is not None and identity.juv_code != str(juv_code):
            identity.juv_code = str(juv_code)[:50]
            identity_updated = True
        if identity_updated:
            identity.save()
        return identity

    identity = _resolve_identity()
    mediafile_location = _parse_mediafile_location()

    mf = uploaded_archive.mediafile_set.filter(mediafile=str(media_rel_pth)).first()
    if mf is None and media_rel_pth != image_rel_pth:
        mf = uploaded_archive.mediafile_set.filter(mediafile=str(image_rel_pth)).first()

    if mf is not None:
        # logger.debug("Using Mediafile generated before")
        status = "found"
    else:
        # convert pandas row to json
        if create_missing:
            # logger.debug(f"{row['detection_results']=}")

            mf = MediaFile(
                parent=uploaded_archive,
                # mediafile=str(image_rel_pth),
                mediafile=str(media_rel_pth),
                image_file=str(image_rel_pth),
                captured_at=captured_at,
                locality=row_locality,
                media_type=row["media_type"],
                # metadata_json=row["detection_results"],
                # metadata_json=metadata_json,
            )
            mf.save()

            # logger.debug(f"{uploaded_archive.contains_identities=}")
            # logger.debug(f"{uploaded_archive.contains_single_taxon=}")
            if uploaded_archive.contains_identities and uploaded_archive.contains_single_taxon:
                identity_is_representative = True
                mf.identity_is_representative = identity_is_representative
            if "original_path" in row:
                mf.original_filename = row["original_path"]
            # if "media_type" in row:
            #     mf.media_type = str(row["media_type"])
            # logger.debug(f"{mf.identity_is_representative}")

            mf.save()
            # observation.save()
            # logger.debug(f"Created new Mediafile {mf}")
            status = "created"
        else:
            df.loc[index, "deleted"] = True
            logger.debug(f"Mediafile {image_rel_pth} not found. Skipping.")
            # continue
            status = "deleted"
            return status

    _apply_prepared_mediafile_variants(mf, row, output_dir)
    if not _mediafile_variants_exist(mf):
        # Fallback for legacy CSVs or manually repaired media directories.
        mf.make_thumbnail_for_mediafile_if_necessary(thumbnail_width=thumbnail_width)

    metadata_json = row.to_dict()
    # remove None and NaN values
    metadata_json = {k: v for k, v in metadata_json.items() if v is not None and not pd.isna(v)}
    # logger.debug(f"{metadata_json=}")
    mf.metadata_json = metadata_json

    # if the mediafile was updated by user, we believe into users input
    if mf.updated_by is None:
        status = status + " and not updated by user"
        # logger.debug(f"{row.keys()=}")
        # logger.debug(f"{uploaded_archive.contains_identities=}")
        # logger.debug(f"{row['predicted_category']=}")
        predicted_taxon = None
        predicted_taxon_confidence = None

        # if archive is uploaded with known taxon, then do not use the predicted taxon.
        if uploaded_archive.contains_single_taxon and uploaded_archive.taxon_for_identification:
            # mf.first_observation.taxon = uploaded_archive.taxon_for_identification
            taxon = uploaded_archive.taxon_for_identification
        else:
            taxon = get_taxon(row["predicted_category"])  # remove this
            # mf.first_observation.taxon = get_taxon(row["predicted_category"])  # remove this
        if captured_at is not None:
            mf.captured_at = captured_at
        if row_locality is not None:
            mf.locality = row_locality
        if mediafile_location is not None:
            mf.location = mediafile_location
        if "predicted_category_raw" in row:
            predicted_taxon = get_taxon(row["predicted_category_raw"])
            predicted_taxon_confidence = float(row["predicted_prob_raw"])
            mf.predicted_taxon = predicted_taxon
            mf.predicted_taxon_confidence = predicted_taxon_confidence
        # if len(mf.observations.all()) == 0:
        #     ao = mf.observations.create(
        #         mediafile=mf,
        #         taxon=taxon,
        #         # metadata_json=row.to_dict(),
        #     )
        # else:
        #     ao = mf.observations.first()
        #     # ao.metadata_json = row.to_dict()
        #     ao.taxon = taxon
        #     ao.save()
        if identity is not None:
            mf.identity = identity
        logger.debug("  update mediafile in db with row of metadata")

        try:
            if ("detection_results" in row) and (row["detection_results"] is not None):
                detection_results = ast.literal_eval(row["detection_results"])
                logger.debug(f"detection_results={detection_results}")
                if len(detection_results) > 0:
                    kv = {"back": "B", "front": "F", "left": "F", "right": "R", "unknown": "U"}
                    if mf.observations.count() > 1:
                        # remove all observations with the exception of the first one
                        # todo probably we should map the observation results to observations
                        # this is porcessed only if the values are not changed by user.
                        mf.observations.exclude(id=mf.first_observation.id).delete()
                    for i, one_detection_result in enumerate(detection_results):
                        ao: models.AnimalObservation
                        if mf.observations.exists() and len(detection_results) > 0:
                            ao = mf.observations.first()
                        else:
                            ao = mf.observations.create(
                                # mediafile=mf,
                                # metadata_json=row.to_dict(),
                            )
                        logger.debug(f"i={i}, detection_result={one_detection_result}")

                        if i == 0:
                            ao.identity = identity

                        ao.taxon = taxon
                        ao.predicted_taxon = predicted_taxon
                        ao.predicted_taxon_confidence = predicted_taxon_confidence
                        x_min, y_min, x_max, y_max = one_detection_result["bbox"]
                        h, w = one_detection_result["size"]
                        logger.debug(f"bbox: {x_min=}, {y_min=}, {x_max=}, {y_max=}")

                        ao.bbox_x_center = ((x_min + x_max) / 2) / w
                        ao.bbox_y_center = ((y_min + y_max) / 2) / h
                        ao.bbox_width = (x_max - x_min) / w
                        ao.bbox_height = (y_max - y_min) / h

                        orientation = detection_results[i]["orientation"]
                        orientation_score = detection_results[i]["orientation_score"]
                        if orientation in kv:
                            if orientation_score < orientation_score_threshold:
                                orientation = "unknown"

                            mf.orientation = kv[orientation]
                            ao.orientation = kv[orientation]
                        else:
                            logger.warning(f"Unknown orientation: {orientation} in {mf.mediafile}")
                        ao.save()
        except Exception as e:
            logger.warning(f"Error during setting orientation in media file {mf.mediafile}: {e}")
            logger.debug(traceback.format_exc())

        mf.save()
        # logger.debug(f"identity={mf.identity}")
    return status

    # logger.debug(f"{mf}")


def get_locality_from_metadata_row(uploaded_archive: UploadedArchive, row, fallback_locality):
    """Prefer per-media locality parsed from path/spreadsheet over upload-wide locality."""
    for column in ("locality_name", "vanilla_location"):
        if column not in row:
            continue
        value = row[column]
        if value is None or pd.isna(value) or str(value).strip() == "":
            continue
        return get_locality(uploaded_archive.owner, str(value).strip())
    return fallback_locality


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

    # _sync_metadata_by_checking_enlisted_mediafiles(csv_file, output_dir, uploaded_archive)
    _sync_metadata_by_creating_from_mediafiles(csv_file, output_dir, uploaded_archive)

    uploaded_archive.output_updated_at = django.utils.timezone.now()
    uploaded_archive.save()


def _sync_metadata_by_creating_from_mediafiles(csv_file, output_dir, uploaded_archive):
    mediafile_set = uploaded_archive.mediafile_set.all()
    if metadata_json_are_consistent(mediafile_set):
        df = create_dataframe_from_mediafiles(mediafile_set)
        df.to_csv(csv_file, encoding="utf-8-sig")
    else:
        csv_file.unlink()


def metadata_json_are_consistent(mediafiles: Generator[MediaFile, None, None]) -> bool:
    """Check if metadata JSONs are consistent."""
    for mf in mediafiles:
        metadata_row = copy.copy(mf.metadata_json)
        logger.debug(f"{metadata_row=}, {type(metadata_row)=}")
        if (metadata_row is None) or ("predicted_category" not in metadata_row):
            logger.debug("No enough information stored in webapp. " "The CSV file will be removed to be recreated.")
            return False
    return True


def create_dataframe_from_mediafiles(mediafiles: Generator[MediaFile, None, None]) -> pd.DataFrame:
    """Create DataFrame from MediaFiles."""
    records = []
    model_backed_export_fields = [
        "unique_name",
        "code",
        "juv_code",
        "locality name",
        "locality coordinates",
        "latitude",
        "longitude",
        "original_path",
        "datetime",
    ]
    # go over mediafiles in set
    for mf in mediafiles:
        # logger.debug(f"{mf.metadata_json=}, {type(mf.metadata_json)=}")
        metadata_row = copy.copy(mf.metadata_json)
        # logger.debug(f"{metadata_row=}, {type(metadata_row)=}")
        if (metadata_row is None) or ("predicted_category" not in metadata_row):
            metadata_row = {}
        for field_name in model_backed_export_fields:
            metadata_row.pop(field_name, None)

        if mf.taxon:
            metadata_row["predicted_category"] = mf.taxon.name
        if mf.identity:
            metadata_row["unique_name"] = mf.identity.name
        if mf.locality:
            metadata_row["locality name"] = mf.locality.name
            if mf.locality.location:
                metadata_row["locality coordinates"] = str(mf.locality.location)
        effective_location = mf.effective_location
        if effective_location and "," in str(effective_location):
            latitude, longitude = [part.strip() for part in str(effective_location).split(",", 1)]
            metadata_row["latitude"] = latitude
            metadata_row["longitude"] = longitude
        if mf.original_filename:
            metadata_row["original_path"] = mf.original_filename
        if mf.identity:
            if mf.identity.code:
                metadata_row["code"] = mf.identity.code
            if mf.identity.juv_code:
                metadata_row["juv_code"] = mf.identity.juv_code
        if mf.captured_at:
            metadata_row["datetime"] = mf.captured_at.isoformat()
        metadata_row["uploaded_archive"] = mf.parent.name
        if mf.parent.locality_check_at:
            metadata_row["locality_check_at"] = mf.parent.locality_check_at

        records.append(metadata_row)
    df = pd.DataFrame.from_records(records)
    return df


def create_dataframe_from_mediafiles_NDOP(
    mediafiles: Generator[MediaFile, None, None],
) -> pd.DataFrame:
    """Create DataFrame from MediaFiles for NDOP and AOPK.

    NDOP = Nálezová databáze ochrany přírody
    AOPK = Agentura ochrany přírody a krajiny
    """
    records = []
    # go over mediafiles in set
    for i, mf in enumerate(mediafiles):
        # logger.debug(f"{mf.metadata_json=}, {type(mf.metadata_json)=}")
        # metadata_row = copy.copy(mf.metadata_json)
        # logger.debug(f"{metadata_row=}, {type(metadata_row)=}")
        metadata_row = {}
        # if (metadata_row is None) or ("predicted_category" not in metadata_row):
        #     metadata_row = {}

        metadata_row["PORADI"] = i + 1
        metadata_row["ID_NALEZ"] = mf.id

        if mf.taxon:
            metadata_row["DRUH"] = mf.taxon.name
            metadata_row["CESKE_JMENO"] = ""
        if mf.parent and mf.parent.owner and mf.parent.owner.user:
            if mf.parent.owner.user.first_name and mf.parent.owner.user.last_name:
                metadata_row["AUTOR"] = mf.parent.owner.user.first_name + " " + mf.parent.owner.user.last_name
            else:
                metadata_row["AUTOR"] = mf.parent.owner.user.username
        # capture date
        if mf.captured_at:
            metadata_row["DATUM_OD"] = mf.captured_at.strftime("%Y%m%d")
            metadata_row["DATUM_DO"] = mf.captured_at.strftime("%Y%m%d")
        else:
            metadata_row["DATUM_OD"] = ""
            metadata_row["DATUM_DO"] = ""
            # if mf.parent.locality_check_at:
            #     metadata_row["locality_check_at"] = mf.parent.locality_check_at
        if mf.locality:
            metadata_row["NAZ_LOKAL"] = mf.locality.name
            metadata_row["ID_LOKAL"] = mf.locality.id
            if mf.locality.location:
                try:
                    lat, lon = str(mf.locality.location).split(",")

                    metadata_row["latitude"] = lat
                    metadata_row["longitude"] = lon

                    from pyproj import Transformer

                    # from WGS84 to S-JTSK
                    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)
                    x, y = transformer.transform(lon, lat)
                    metadata_row["X"] = x
                    metadata_row["Y"] = y

                except Exception as e:
                    logger.debug(traceback.format_exc())
                    logger.error(f"Error ({str(e)} during parsing location: {mf.locality.name}")
        metadata_row["ZDROJ"] = "caid.kky.zcu.cz"
        metadata_row["NEGATIV"] = 0
        metadata_row["POCET"] = ""
        metadata_row["POCITANO"] = ""

        records.append(metadata_row)
    df = pd.DataFrame.from_records(records)
    return df


def _sync_metadata_by_checking_enlisted_mediafiles(csv_file, output_dir, uploaded_archive):
    update_csv = False
    df = pd.read_csv(csv_file, index_col=0)
    df.rename(
        columns={
            "locality_name": "locality name",
            "locality_coordinates": "locality coordinates",
        },
        inplace=True,
    )
    logger.debug(f"{len(df)=}")
    df["deleted"] = True
    df["locality name"] = ""
    df["locality coordinates"] = ""
    # for fn in df["image_path"]:
    for index, row in df.iterrows():
        rel_pth, _ = _get_rel_and_abs_paths_based_on_csv_row(row, output_dir)

        try:
            mf = uploaded_archive.mediafile_set.get(mediafile=str(rel_pth))
            df.loc[index, "deleted"] = False
            logger.debug("Using Mediafile generated before")
        except MediaFile.DoesNotExist:
            df.loc[index, "deleted"] = True
            update_csv = True
            logger.debug(f"Mediafile {rel_pth} not found. The row will be removed from CSV.")
            continue
        # generate thumbnail if necessary
        mf.make_thumbnail_for_mediafile_if_necessary()

        ao = mf.observations.first()
        if ao:
            if mf.taxon:
                df.loc[index, "predicted_category"] = ao.taxon.name
                update_csv = True
            if mf.identity:
                df.loc[index, "unique_name"] = ao.identity.name
                update_csv = True
        if mf.locality:
            df.loc[index, "locality name"] = mf.locality.name
            if mf.locality.location:
                df.loc[index, "locality coordinates"] = str(mf.locality.location)
    # delete rows with missing mediafiles
    df = df[df["deleted"] == False]  # noqa: E712
    if update_csv:
        logger.debug(f"{len(df)=}")
        df.to_csv(csv_file, encoding="utf-8-sig")
        logger.debug(f"CSV updated. path={csv_file}")
    # return df, update_csv


@shared_task
def init_identification_on_success(*args, **kwargs):
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug(f"{args=}")
    logger.debug(f"{kwargs=}")
    # models.Notification(message=f"Identification initialization finished. {args=} {kwargs=}").save()
    workgroup_id = kwargs.pop("workgroup_id")
    statistic_id = kwargs.pop("statistic_id", None)
    workgroup = WorkGroup.objects.get(id=workgroup_id)
    models.Notification.create_for(
        message=f"Identification initialization finished. {args=} {kwargs=}",
        workgroups=[workgroup],
        level=models.Notification.INFO,
    )
    output: dict = args[0]
    status = output["status"]
    finish_identification_run_statistic(
        statistic_id,
        "finished" if status == "DONE" else status.lower(),
        workgroup.identification_scheduled_init_task_id or "",
    )
    status = "Finished" if status == "DONE" else status
    workgroup.identification_init_status = status
    if "message" in output:
        message = output["message"]
    elif "error" in output:
        message = output["error"]
    else:
        message = ""
    workgroup.identification_init_message = message
    now = django.utils.timezone.now()
    workgroup.identification_init_at = now
    workgroup.identification_scheduled_init_task_id = None
    workgroup.identification_scheduled_init_eta = None
    workgroup.save(
        update_fields=[
            "identification_init_status",
            "identification_init_message",
            "identification_init_at",
            "identification_scheduled_init_task_id",
            "identification_scheduled_init_eta",
        ]
    )
    logger.debug(f"{message=}")
    logger.debug(f"{workgroup=}")
    logger.debug(f"{workgroup.identification_init_at=}")
    logger.debug(f"{workgroup.hash=}")

    logger.debug("init_identification done.")
    schedule_reid_identification_for_workgroup(workgroup, delay_minutes=2)


@shared_task
def train_identification_on_success(*args, **kwargs):
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug(f"{args=}")
    logger.debug(f"{kwargs=}")
    caiduser = None
    if "user_id" in kwargs:
        caiduser_id = kwargs.pop("caiduser_id")
        caiduser = models.CaIDUser.objects.get(id=caiduser_id)
    models.Notification.create_for(
        message=f"Task finished with error. {args=} {kwargs=}", users=[caiduser], level=models.Notification.INFO
    )
    workgroup_id = kwargs.pop("workgroup_id")
    workgroup = WorkGroup.objects.get(id=workgroup_id)
    output: dict = args[0]
    status = output["status"]
    status = "Finished" if status == "DONE" else status
    workgroup.identification_init_status = status
    if "message" in output:
        message = output["message"]
    elif "error" in output:
        message = output["error"]
    else:
        message = ""
    workgroup.identification_init_message = message
    now = django.utils.timezone.now()
    workgroup.identification_init_at = now
    workgroup.save()
    logger.debug(f"{message=}")
    logger.debug(f"{workgroup=}")
    logger.debug(f"{workgroup.identification_init_at=}")
    logger.debug(f"{workgroup.hash=}")

    logger.debug("init_identification done.")


@shared_task
def init_identification_on_error(*args, **kwargs):
    """Callback invoked after failing init_identification function in inference worker."""
    workgroup_id = kwargs.pop("workgroup_id", None)
    statistic_id = kwargs.pop("statistic_id", None)
    finish_identification_run_statistic(statistic_id, "failed")
    if workgroup_id is not None:
        workgroup = WorkGroup.objects.get(id=workgroup_id)
        workgroup.identification_init_status = "Failed"
        workgroup.identification_init_at = django.utils.timezone.now()
        workgroup.identification_init_message = f"Initialization failed. {args=} {kwargs=}"
        workgroup.identification_scheduled_init_task_id = None
        workgroup.identification_scheduled_init_eta = None
        workgroup.save(
            update_fields=[
                "identification_init_status",
                "identification_init_at",
                "identification_init_message",
                "identification_scheduled_init_task_id",
                "identification_scheduled_init_eta",
            ]
        )
    caiduser = None
    kwargs = dict(
        message=f"Task finished with error. {args=} {kwargs=}",
        level=models.Notification.ERROR,
    )
    if "user_id" in kwargs:
        caiduser_id = kwargs.pop("caiduser_id")
        caiduser = models.CaIDUser.objects.get(id=caiduser_id)
        kwargs["users"] = [caiduser]

    models.Notification.create_for(**kwargs).save()
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
    statistic_id = kwargs.pop("statistic_id", None)
    finish_identification_run_statistic(statistic_id, "failed", uuid)
    kwargs = dict(
        message=f"Upload processing finished with error. {self} {args=} {kwargs=}",
        level=models.Notification.ERROR,
        json_message=dict(self=self, args=args, kwargs=kwargs),
    )
    models.Notification(**kwargs)
    # logger.debug(f"dir(self)={dir(self)}")


@shared_task(bind=True)
def log_output(self, output: dict, *args, **kwargs):
    """Callback invoked after running init_identification function in inference worker."""
    logger.debug("log_output")
    logger.debug(f"{output=}")
    logger.debug(f"{args=}")
    logger.debug(f"{kwargs=}")


#  remove this function  - it is not used
# @shared_task(bind=True)
# def detection_on_success_after_species_prediction(self, output: dict, *args, **kwargs):
#     """Finish detection and set status after species is predicted."""
#     logger.debug("detection on success")
#     logger.debug(f"{output=}")
#     logger.debug(f"{args=}")
#     logger.debug(f"{kwargs=}")
#     uploaded_archive_id: int = kwargs.pop("uploaded_archive_id")
#     uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
#     if "status" not in output:
#         logger.critical(f"Unexpected error {output=} is missing 'status' field.")
#         uploaded_archive.taxon_status = "U"
#     elif output["status"] == "DONE":
#         uploaded_archive.taxon_status = "TAID"
#         uploaded_archive.status_message = str(uploaded_archive.status_message) + " Detection done."
#     else:
#         uploaded_archive.taxon_status = "F"
#         if "error" in output:
#             logger.error(f"{output['error']=}")
#             uploaded_archive.status_message = output["error"]
#         uploaded_archive.finished_at = django.utils.timezone.now()
#     uploaded_archive.save()


# @shared_task(bind=True)
# def detection_on_success(self, output: dict, *args, **kwargs):
#     """Callback invoked after running init_identification function in inference worker."""
#     logger.debug("detection on success")
#     logger.debug(f"{output=}")
#     logger.debug(f"{args=}")
#     logger.debug(f"{kwargs=}")
#
#     uploaded_archive_id: int = kwargs.pop("uploaded_archive_id")
#     uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
#     uploaded_archive.taxon_status = "...detection done"
#     uploaded_archive.save()
#     identify_signature = signature(
#         "identify",
#         kwargs=kwargs,
#     )
#     identify_task = identify_signature.apply_async(
#         link=identify_on_success.s(
#             uploaded_archive_id=uploaded_archive_id,
#         ),
#         link_error=on_error_in_upload_processing.s(),
#     )
#     logger.debug(f"{identify_task=}")


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
    statistic_id = kwargs.pop("statistic_id", None)
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    # uploaded_archive.identification_status = "IAID"
    uploaded_archive.save()
    owner = uploaded_archive.owner
    workgroup = owner.workgroup

    try:
        if "status" not in output:
            finish_identification_run_statistic(statistic_id, "unknown")
            msg = f"Unexpected error {output=} is missing 'status' field."
            logger.critical(msg)
            uploaded_archive.identification_status = "U"
            uploaded_archive.status_message = msg
            uploaded_archive.save()
            workgroup.identification_reid_status = "Finished"
            workgroup.identification_reid_at = now()
            workgroup.identification_reid_message = msg
            workgroup.identification_scheduled_run_task_id = None
            workgroup.identification_scheduled_run_eta = None
            workgroup.save(
                update_fields=[
                    "identification_reid_status",
                    "identification_reid_at",
                    "identification_reid_message",
                    "identification_scheduled_run_task_id",
                    "identification_scheduled_run_eta",
                ]
            )

            # TODO - should the app return some error response to the user?
        elif output["status"] == "DONE":
            finish_identification_run_statistic(statistic_id, "finished")
            # load output file
            output_json_file = output["output_json_file"]
            with open(output_json_file, "r") as f:
                data = json.load(f)
            # logger.trace(f"Loaded output data: {data=}")
            assert "mediafile_ids" in data
            assert "pred_image_paths" in data
            assert "pred_class_ids" in data
            assert "pred_labels" in data
            assert "scores" in data
            assert "keypoints" in data

            media_root = Path(settings.MEDIA_ROOT)
            deleted_queue_count, deleted_suggestion_count = clear_identification_queue_for_uploaded_archive(uploaded_archive)
            logger.info(
                "Cleared identification queue after successful rerun for upload %s: mediafiles=%s suggestions=%s",
                uploaded_archive.id,
                deleted_queue_count,
                deleted_suggestion_count,
            )

            mediafile_ids = data["mediafile_ids"]
            len_mediafile_ids = len(mediafile_ids)
            for i, mediafile_id in enumerate(mediafile_ids):

                _prepare_mediafile_for_identification(data, i, media_root, mediafile_id)

            uploaded_archive.identification_status = "IAID"
            uploaded_archive.status_message = f"Identification suggestions ready for {len_mediafile_ids} media files."
            uploaded_archive.save()
            logger.debug("Identication suggestions done.")
            # go over all uploaded archives of the workgroup
            if UploadedArchive.objects.filter(owner__workgroup=workgroup, identification_status="IAIP").count() == 0:
                # if there is no archive in the workgroup with status "IAIP", we can start identification
                workgroup.identification_reid_status = "Finished"
                workgroup.identification_reid_at = now()
                workgroup.identification_reid_message = (
                    f"Identification suggestions ready for {len_mediafile_ids} media files."
                )
                workgroup.identification_scheduled_run_task_id = None
                workgroup.identification_scheduled_run_eta = None
                workgroup.save(
                    update_fields=[
                        "identification_reid_status",
                        "identification_reid_at",
                        "identification_reid_message",
                        "identification_scheduled_run_task_id",
                        "identification_scheduled_run_eta",
                    ]
                )
                logger.debug(f"Workgroup {workgroup} identification status set to 'IAID'.")

        else:
            finish_identification_run_statistic(statistic_id, "failed")
            # identification failed
            uploaded_archive.identification_status = "F"
            uploaded_archive.save()
            message = "Identification failed. "
            if "error" in output:
                logger.error(f"{output['error']=}")
                message += output["error"]
                if output["error"] == "Input data is empty.":
                    message += " Try to check the taxa in the input data."

            uploaded_archive.status_message = message
            workgroup.identification_reid_status = "Finished"
            workgroup.identification_reid_at = now()
            workgroup.identification_reid_message = message
            workgroup.identification_scheduled_run_task_id = None
            workgroup.identification_scheduled_run_eta = None
            workgroup.save(
                update_fields=[
                    "identification_reid_status",
                    "identification_reid_at",
                    "identification_reid_message",
                    "identification_scheduled_run_task_id",
                    "identification_scheduled_run_eta",
                ]
            )
            logger.debug(f"{output=}")
            logger.error("Identification failed.")

    except Exception as e:
        finish_identification_run_statistic(statistic_id, "failed")
        uploaded_archive.identification_status = "F"
        uploaded_archive.status_message = f"Error during identification. {str(e)}"
        uploaded_archive.save()
        workgroup.identification_reid_status = "Finished"
        workgroup.identification_reid_at = now()
        workgroup.identification_reid_message = f"Error during identification. {str(e)}"
        workgroup.identification_scheduled_run_task_id = None
        workgroup.identification_scheduled_run_eta = None
        workgroup.save(
            update_fields=[
                "identification_reid_status",
                "identification_reid_at",
                "identification_reid_message",
                "identification_scheduled_run_task_id",
                "identification_scheduled_run_eta",
            ]
        )
        logger.error(f"Error during identification: {e}")
        logger.error(traceback.format_exc())

        # TODO - should the app return some error response to the user?


@shared_task(bind=True)
def identify_bulk_on_success(self, output: dict, *args, **kwargs):
    """Callback invoked after running one bulk identification job for multiple uploads."""
    status = output.get("status", "unknown")
    logger.info("Bulk identification task finished with status '%s'.", status)

    workgroup_id: int = kwargs.pop("workgroup_id")
    uploaded_archive_ids: list[int] = kwargs.pop("uploaded_archive_ids")
    statistic_id = kwargs.pop("statistic_id", None)
    workgroup = WorkGroup.objects.get(id=workgroup_id)
    uploaded_archives = list(
        UploadedArchive.objects.filter(id__in=uploaded_archive_ids, owner__workgroup=workgroup).order_by("id")
    )

    try:
        if "status" not in output:
            finish_identification_run_statistic(statistic_id, "unknown")
            msg = f"Unexpected error {output=} is missing 'status' field."
            logger.critical(msg)
            for uploaded_archive in uploaded_archives:
                uploaded_archive.identification_status = "U"
                uploaded_archive.status_message = msg
                uploaded_archive.save(update_fields=["identification_status", "status_message"])
        elif output["status"] == "DONE":
            finish_identification_run_statistic(statistic_id, "finished")
            output_json_file = output["output_json_file"]
            with open(output_json_file, "r") as f:
                data = json.load(f)
            assert "mediafile_ids" in data
            assert "pred_image_paths" in data
            assert "pred_class_ids" in data
            assert "pred_labels" in data
            assert "scores" in data
            assert "keypoints" in data

            media_root = Path(settings.MEDIA_ROOT)
            mediafile_ids = data["mediafile_ids"]
            mediafiles = list(MediaFile.objects.filter(id__in=mediafile_ids).select_related("parent"))
            count_by_archive_id = {}
            for mediafile in mediafiles:
                if mediafile.parent_id is None:
                    continue
                count_by_archive_id[mediafile.parent_id] = count_by_archive_id.get(mediafile.parent_id, 0) + 1

            for uploaded_archive in uploaded_archives:
                deleted_queue_count, deleted_suggestion_count = clear_identification_queue_for_uploaded_archive(uploaded_archive)
                logger.info(
                    "Cleared identification queue after successful bulk rerun for upload %s: mediafiles=%s suggestions=%s",
                    uploaded_archive.id,
                    deleted_queue_count,
                    deleted_suggestion_count,
                )

            for i, mediafile_id in enumerate(mediafile_ids):
                _prepare_mediafile_for_identification(data, i, media_root, mediafile_id)

            for uploaded_archive in uploaded_archives:
                processed_count = count_by_archive_id.get(uploaded_archive.id, 0)
                uploaded_archive.identification_status = "IAID"
                uploaded_archive.status_message = (
                    f"Identification suggestions ready for {processed_count} media files."
                )
                uploaded_archive.save(update_fields=["identification_status", "status_message"])

            workgroup.identification_reid_status = "Finished"
            workgroup.identification_reid_at = now()
            workgroup.identification_reid_message = (
                f"Identification suggestions ready for {len(mediafile_ids)} media files "
                f"across {len(uploaded_archives)} uploads."
            )
            workgroup.identification_scheduled_run_task_id = None
            workgroup.identification_scheduled_run_eta = None
            workgroup.save(
                update_fields=[
                    "identification_reid_status",
                    "identification_reid_at",
                    "identification_reid_message",
                    "identification_scheduled_run_task_id",
                    "identification_scheduled_run_eta",
                ]
            )
        else:
            finish_identification_run_statistic(statistic_id, "failed")
            message = "Identification failed. "
            if "error" in output:
                logger.error(f"{output['error']=}")
                message += output["error"]
                if output["error"] == "Input data is empty.":
                    message += " Try to check the taxa in the input data."
            for uploaded_archive in uploaded_archives:
                uploaded_archive.identification_status = "F"
                uploaded_archive.status_message = message
                uploaded_archive.save(update_fields=["identification_status", "status_message"])
            workgroup.identification_reid_status = "Finished"
            workgroup.identification_reid_at = now()
            workgroup.identification_reid_message = message
            workgroup.identification_scheduled_run_task_id = None
            workgroup.identification_scheduled_run_eta = None
            workgroup.save(
                update_fields=[
                    "identification_reid_status",
                    "identification_reid_at",
                    "identification_reid_message",
                    "identification_scheduled_run_task_id",
                    "identification_scheduled_run_eta",
                ]
            )
    except Exception as e:
        finish_identification_run_statistic(statistic_id, "failed")
        for uploaded_archive in uploaded_archives:
            uploaded_archive.identification_status = "F"
            uploaded_archive.status_message = f"Error during identification. {str(e)}"
            uploaded_archive.save(update_fields=["identification_status", "status_message"])
        workgroup.identification_reid_status = "Finished"
        workgroup.identification_reid_at = now()
        workgroup.identification_reid_message = f"Error during identification. {str(e)}"
        workgroup.identification_scheduled_run_task_id = None
        workgroup.identification_scheduled_run_eta = None
        workgroup.save(
            update_fields=[
                "identification_reid_status",
                "identification_reid_at",
                "identification_reid_message",
                "identification_scheduled_run_task_id",
                "identification_scheduled_run_eta",
            ]
        )
        logger.error(f"Error during bulk identification: {e}")
        logger.error(traceback.format_exc())


@shared_task(bind=True)
def identify_bulk_on_error(self, uuid, *args, **kwargs):
    """Error callback for one bulk identification job serving multiple uploads."""
    logger.error("Bulk identification finished with error.")
    result = self.AsyncResult(uuid)
    error_message = result.result if result.failed() else "No error message available"
    logger.error(f"Bulk identification error: {error_message}")

    workgroup_id: int = kwargs.pop("workgroup_id")
    uploaded_archive_ids: list[int] = kwargs.pop("uploaded_archive_ids")
    statistic_id = kwargs.pop("statistic_id", None)
    finish_identification_run_statistic(statistic_id, "failed", uuid)
    workgroup = WorkGroup.objects.get(id=workgroup_id)
    uploaded_archives = UploadedArchive.objects.filter(id__in=uploaded_archive_ids, owner__workgroup=workgroup)
    for uploaded_archive in uploaded_archives:
        uploaded_archive.identification_status = "F"
        uploaded_archive.status_message = f"Identification failed. {error_message}"
        uploaded_archive.save(update_fields=["identification_status", "status_message"])

    workgroup.identification_reid_status = "Finished"
    workgroup.identification_reid_at = now()
    workgroup.identification_reid_message = f"Identification failed. {error_message}"
    workgroup.identification_scheduled_run_task_id = None
    workgroup.identification_scheduled_run_eta = None
    workgroup.save(
        update_fields=[
            "identification_reid_status",
            "identification_reid_at",
            "identification_reid_message",
            "identification_scheduled_run_task_id",
            "identification_scheduled_run_eta",
        ]
    )


def _prepare_mediafile_for_identification(data, i, media_root, mediafile_id):
    """Prepare media files for i-th queried image."""
    reid_top_k_class_ids = data["pred_class_ids"][i]
    reid_top_k_labels = data["pred_labels"][i]
    reid_top_k_image_paths = data["pred_image_paths"][i]
    reid_top_k_scores = data["scores"][i]
    unknown_mediafile = MediaFile.objects.get(id=mediafile_id)

    # update mediafile.metadata_json (model.JsonField) with the new data
    metadata_json = unknown_mediafile.metadata_json
    metadata_json["reid_top_k_class_ids"] = reid_top_k_class_ids
    metadata_json["reid_top_k_labels"] = reid_top_k_labels
    metadata_json["reid_top_k_image_paths"] = reid_top_k_image_paths
    metadata_json["reid_top_k_scores"] = reid_top_k_scores
    unknown_mediafile.metadata_json = metadata_json
    unknown_mediafile.save()

    # old processing
    if (reid_top_k_scores[0]) > settings.IDENTITY_MANUAL_CONFIRMATION_THRESHOLD:

        identity_id = reid_top_k_class_ids[0]  # top-1
        unknown_mediafile.identity = IndividualIdentity.objects.get(id=identity_id)
        logger.debug(
            f"{unknown_mediafile} is {unknown_mediafile.identity.name} with score={reid_top_k_scores[0]}. "
            + "No need of manual confirmation."
        )
        if unknown_mediafile.identity.name != reid_top_k_labels[0]:  # top-1
            logger.warning(f"Identity name mismatch: {unknown_mediafile.identity.name} != {reid_top_k_labels[0]}")

        unknown_mediafile.save()

    else:
        mfi, _ = MediafilesForIdentification.objects.get_or_create(
            mediafile=unknown_mediafile,
        )

        # new processing
        # delete mediafile suggestions related to mediafile for identification - mfi
        models.MediafileIdentificationSuggestion.objects.filter(for_identification=mfi).delete()

        # top_k_class_ids = data["pred_class_ids"][i]
        # top_k_labels = data["pred_labels"][i]
        # top_k_paths = data["pred_image_paths"][i]
        # top_k_scores = data["scores"][i]
        #
        paired_points_for_k_images = data["keypoints"][i]

        for identity_id, top_score, top_name, top_path, top_paired_points in zip(
            reid_top_k_class_ids,
            reid_top_k_scores,
            reid_top_k_labels,
            reid_top_k_image_paths,
            paired_points_for_k_images,
        ):
            try:
                top_abspath = Path(top_path)
                top_relpath = top_abspath.relative_to(media_root)
                top_mediafile = MediaFile.objects.get(image_file=str(top_relpath))
                # top_mediafile = MediaFile.objects.get(mediafile=str(top_relpath))

                identity = IndividualIdentity.objects.get(id=identity_id)
                if identity.name != top_name:
                    logger.warning(f"Identity name mismatch: {identity.name} != {top_name} for {unknown_mediafile=}")

                mfi_suggestion = models.MediafileIdentificationSuggestion(
                    for_identification=mfi,
                    mediafile=top_mediafile,
                    identity=identity,
                    score=top_score,
                    paired_points=top_paired_points,
                    name=top_name or "Unknown",
                )

                mfi_suggestion.save()
            except Exception as e:
                logger.debug(f"{reid_top_k_image_paths=}")
                logger.debug(f"{top_path=}")
                logger.debug(traceback.format_exc())
                logger.error(f"Error during identification of {unknown_mediafile}: {e}: {traceback.format_exc()}")


# def _identity_mismatch_waning(
#     top1_mediafile: MediaFile,
#     top2_mediafile: MediaFile,
#     top3_mediafile: MediaFile,
#     top_k_labels: list,
# ) -> None:
#     """Warn if the identity mismatch is detected."""
#     if top1_mediafile.identity.name != top_k_labels[0]:
#         logger.warning(f"Identity mismatch: {top1_mediafile.identity.name} != {top_k_labels[0]}")
#     if top2_mediafile.identity.name != top_k_labels[1]:
#         logger.warning(f"Identity mismatch: {top2_mediafile.identity.name} != {top_k_labels[1]}")
#     if top3_mediafile.identity.name != top_k_labels[2]:
#         logger.warning(f"Identity mismatch: {top3_mediafile.identity.name} != {top_k_labels[2]}")


@shared_task(bind=True)
def simple_log(self, *args, **kwargs):
    """Simple log task."""
    logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
    return {"status": "DONE"}


def schedule_reid_identification_for_workgroup(workgroup: models.WorkGroup, delay_minutes: int = 15):
    """Schedule reiidentification suggestions for a workgroup."""
    # Zruš předchozí naplánovaný task
    if workgroup.identification_scheduled_run_task_id:
        current_app.control.revoke(workgroup.identification_scheduled_run_task_id, terminate=True)

    eta = now() + timedelta(minutes=delay_minutes)
    if (workgroup.identification_scheduled_run_eta is None) or (eta > workgroup.identification_scheduled_run_eta):
        # if there is already scheduled re-identification, use that time
        task = run_identification_on_unidentified_for_workgroup_task.apply_async(args=[workgroup.id], eta=eta)

        workgroup.identification_scheduled_run_task_id = task.id
        workgroup.identification_scheduled_run_eta = eta
        workgroup.identification_reid_status = "Scheduled"
        workgroup.save(
            update_fields=[
                "identification_scheduled_run_task_id",
                "identification_scheduled_run_eta",
                "identification_reid_status",
            ]
        )


@shared_task
def run_identification_on_unidentified_for_workgroup_task(workgroup_id: int):
    """Run identification on unidentified media files for a workgroup (task wrapper)."""
    return run_identification_on_unidentified_for_workgroup(workgroup_id)


def run_identification_on_unidentified_for_workgroup(workgroup_id: int, request=None):
    """Run identification suggestions in one batch for all eligible uploads in a workgroup."""
    logger.debug(f"Running identification suggestions for workgroup {workgroup_id}...")
    from .views import run_identification_bulk

    workgroup = WorkGroup.objects.get(pk=workgroup_id)
    uploaded_archives = get_uploaded_archives_pending_identification(workgroup)
    upload_count = uploaded_archives.count()

    workgroup.identification_reid_status = "Processing"
    workgroup.identification_reid_at = now()
    workgroup.identification_reid_message = (
        f"Preparing identification batch for {upload_count} uploads."
    )
    workgroup.save(update_fields=["identification_reid_status", "identification_reid_at", "identification_reid_message"])
    models.Notification.create_for(
        message=f"Starting identification for workgroup {workgroup_id}...",
        workgroups=[workgroup],
        level=models.Notification.DEBUG,
    )

    status_ok = run_identification_bulk(workgroup, uploaded_archives=uploaded_archives)
    if request:
        from django.contrib import messages

        if status_ok:
            messages.info(request, f"Identification started for {upload_count} uploads.")
        else:
            messages.error(request, "No records for identification with the expected taxon.")

    if not status_ok:
        workgroup.identification_reid_status = "Finished"
        workgroup.identification_reid_at = now()
        workgroup.identification_reid_message = "No records available for identification."
        workgroup.identification_scheduled_run_task_id = None
        workgroup.identification_scheduled_run_eta = None
        workgroup.save(
            update_fields=[
                "identification_reid_status",
                "identification_reid_at",
                "identification_reid_message",
                "identification_scheduled_run_task_id",
                "identification_scheduled_run_eta",
            ]
        )
        models.Notification.create_for(
            message=f"No records for identification in workgroup {workgroup}.",
            workgroups=[workgroup],
            level=models.Notification.ERROR,
        )

    logger.info(
        "Identification batch dispatched for workgroup %s: started=%s total=%s",
        workgroup.id,
        int(bool(status_ok)),
        upload_count,
    )


def schedule_init_identification_for_workgroup(workgroup: models.WorkGroup, delay_minutes: int = 10):
    """Schedule initialization of identification for a workgroup."""
    logger.debug(f"Scheduling init_identification for {workgroup=} in {delay_minutes} minutes.")
    # Cancel previously scheduled task
    if workgroup.identification_scheduled_init_task_id:
        current_app.control.revoke(workgroup.identification_scheduled_init_task_id, terminate=True)

    eta = now() + timedelta(minutes=delay_minutes)
    task = init_identification.apply_async(args=[workgroup.id], eta=eta)

    workgroup.identification_scheduled_init_task_id = task.id
    workgroup.identification_scheduled_init_eta = eta
    workgroup.identification_init_status = "Scheduled"

    workgroup.save(
        update_fields=[
            "identification_scheduled_init_task_id",
            "identification_scheduled_init_eta",
            "identification_init_status",
        ]
    )

    schedule_reid_identification_for_workgroup(workgroup, delay_minutes=delay_minutes + 50)


def schedule_init_identification_after_representative_upload(uploaded_archive: UploadedArchive) -> bool:
    """Schedule one final initialization after an identified base dataset finishes importing."""
    has_representatives = uploaded_archive.mediafile_set.filter(
        identity__isnull=False,
        identity_is_representative=True,
    ).exists()
    if not uploaded_archive.import_finished or not has_representatives:
        return False

    logger.debug(
        "Scheduling identification initialization after completed representative upload: upload=%s",
        uploaded_archive.id,
    )
    schedule_init_identification_for_workgroup(uploaded_archive.owner.workgroup)
    return True


@shared_task
def init_identification(workgroup_id: int, selection: dict | None = None):
    """Initialize identification for a workgroup."""
    workgroup = WorkGroup.objects.get(pk=workgroup_id)

    process_for_message = "initialization"
    called_function_name = "init_identification"
    selection = selection or {}
    mediafiles_qs = workgroup.mediafiles_for_identification(
        uploaded_archive_ids=selection.get("uploaded_archive_ids"),
        sequence_ids=selection.get("sequence_ids"),
        mediafile_ids=selection.get("mediafile_ids"),
        # TODO select only files with finished import
        # require_import_finished=selection.get("require_import_finished", True),
        representative_only=selection.get("representative_only", True),
        require_identity=selection.get("require_identity", True),
        observation_taxon=selection.get(
            "observation_taxon",
            workgroup.default_taxon_for_identification if workgroup.check_taxon_before_identification else None,
        ),
        require_observations=selection.get("require_observations", False),
    )

    # mark these mediafiles as used for init identification
    mediafiles_qs.update(used_for_init_identification=True)
    image_number, video_number = count_identification_media_types(mediafiles_qs)
    statistic = create_identification_run_statistic(
        workgroup=workgroup,
        operation="init",
        image_number=image_number,
        video_number=video_number,
    )

    # set attribute media_file_used_for_init_identification
    logger.debug("Generating CSV for init_identification...")
    output_dir = Path(settings.MEDIA_ROOT) / workgroup.name
    output_dir.mkdir(exist_ok=True, parents=True)
    csv_data = _prepare_dataframe_for_identification(mediafiles_qs)
    identity_metadata_file = output_dir / "init_identification.csv"
    pd.DataFrame(csv_data).to_csv(identity_metadata_file, index=False)
    logger.debug(f"{identity_metadata_file=}")
    workgroup.identification_init_at = django.utils.timezone.now()
    workgroup.identification_init_status = "Processing"
    workgroup.identification_init_model_path = str(workgroup.identification_model.model_path)
    workgroup.identification_init_message = (
        f"Using {len(csv_data['image_path'])}" + f"representative images for identification {process_for_message}."
    )
    workgroup.save()
    logger.debug(f"Calling {process_for_message} identification...")
    kwargs = {
        "input_metadata_file": str(identity_metadata_file),
        "organization_id": workgroup.id,
        # csv file should contain image_path, class_id, label
        "identification_model": {
            "name": workgroup.identification_model.name,
            "path": workgroup.identification_model.model_path,
        },
    }
    logger.debug(f"{kwargs=}")
    sig = signature(
        called_function_name,
        # "init_identification",
        kwargs=kwargs,
    )
    # task =
    task = sig.apply_async(
        link=init_identification_on_success.s(
            workgroup_id=workgroup.id,
            statistic_id=statistic.id,
            # uploaded_archive_id=uploaded_archive.id,
            # zip_file=os.path.relpath(str(output_archive_file), settings.MEDIA_ROOT),
            # csv_file=os.path.relpath(str(output_metadata_file), settings.MEDIA_ROOT),
        ),
        link_error=init_identification_on_error.s(
            workgroup_id=workgroup.id,
            statistic_id=statistic.id,
            # uploaded_archive_id=uploaded_archive.id
        ),
    )
    statistic.task_id = task.id
    statistic.save(update_fields=["task_id"])
    workgroup.identification_scheduled_init_task_id = task.id
    workgroup.identification_scheduled_init_eta = None
    workgroup.save(
        update_fields=[
            "identification_scheduled_init_task_id",
            "identification_scheduled_init_eta",
        ]
    )


def _find_mediafiles_for_identification(
    mediafile_paths: list,
) -> MediafilesForIdentification:
    """Find mediafiles for identification.

    :param mediafile_paths: List of paths of mediafiles to identify.
    :return: MediafilesForIdentification object.
    """
    pass


def _ensure_date_format(date_str: str) -> str:
    if len(date_str) == 8:  # Format YYYYMMDD
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    else:
        date = date_str
    return date


def _iterate_over_locality_checks(path: Path, caiduser: CaIDUser) -> Generator[SimpleNamespace, None, None]:
    import re
    from itertools import chain

    params = user_has_access_filter_params(caiduser, "owner")
    archives = [str(archive) for archive in UploadedArchive.objects.filter(**params)]

    paths_of_locality_check = chain(
        path.glob("./????????/*"),
        path.glob("./????-??-??/*"),
        path.glob("./*/????????"),
        path.glob("./*/????-??-??"),
        path.glob("./*/????????.zip"),
        path.glob("./*/????-??-??.zip"),
        path.glob("./*"),
    )
    # paths_of_locality_check = chain(path.glob("./*_????-??-??"), path.glob("./*_????-??-??.zip"))
    # paths_of_locality_check = path.glob("./*")
    base_path = path

    checked_subdirs = []
    for path_of_locality_check in paths_of_locality_check:
        parent_dir_to_be_deleted = False
        # is this a directory inside base_path?
        is_first_level_dir = path_of_locality_check.parent == base_path
        # is_second_level_dir = path_of_locality_check.parent.parent == base_path

        # remove extension if any
        pth_no_suffix = path_of_locality_check.with_suffix("")
        # check if name is in format {locality_name}_YYYY-MM-DD
        # match0 = re.match(r"([0-9]{4}-?[0-9]{2}-?[0-9]{2})_(.*)", pth_no_suffix.name)
        match1 = re.match(r"[0-9]{4}-?[0-9]{2}-?[0-9]{2}", pth_no_suffix.parts[-2])
        match2 = re.match(r"([0-9]{4}-?[0-9]{2}-?[0-9]{2})", pth_no_suffix.parts[-1])

        dt, loc = fs_data.get_date_and_locality_from_filename(path_of_locality_check)
        if (loc is not None) and is_first_level_dir:

            # date_str, locality = match0.groups()
            date = _ensure_date_format(dt)
            locality = loc
            # split name and date, date is in the end of the name in format YYYY-MM-DD,
            # locality is in the beginning of dir or file name separated from date by underscore
            # date, locality = pth_no_suffix.parts[-1].split("_", 1)
            # locality is everything after the last underscore

            error_message = None
        elif match1:
            # Mediafiles are organized in directory structure DATE / LOCALITY
            # date is the parent directory and locality is the leaf directory
            date_str = pth_no_suffix.parts[-2]
            date = _ensure_date_format(date_str)

            locality = pth_no_suffix.parts[-1]
            error_message = None
            checked_subdirs.append(pth_no_suffix.parts[-2])
        elif match2 and not is_first_level_dir:
            # Mediafiles are organized in directory structure LOCALITY / DATE
            # date is the parent directory and locality is the leaf directory
            grps = match2.groups()
            date_str = grps[0]
            date = _ensure_date_format(date_str)

            locality = pth_no_suffix.parts[-2]
            error_message = None
            checked_subdirs.append(pth_no_suffix.parts[-2])
        elif pth_no_suffix.parts[-1] in checked_subdirs:
            parent_dir_to_be_deleted = True
            # the parent directory which was already checked
            error_message = (None,)
            locality = ""
            date = ""
            # continue
        else:
            logger.debug(f"Path withouth the suffix: {pth_no_suffix=}")
            logger.debug("Name of the directory or file is not in format {YYYY-MM-DD}_{locality_name}." + "Skipping.")
            error_message = "Name of the directory or file is not in correct format. " + "Skipping."
            locality = ""
            date = ""

        # locality = path_of_locality_check.parts[-2]
        # date = path_of_locality_check.parts[-1]
        # logger.debug(f"{path_of_locality_check.parts=}")

        # remove diacritics and spaces from zip_name

        zip_name = model_tools.remove_diacritics(f"{locality}_{date}.zip").replace(" ", "_")

        relative_path = path_of_locality_check.relative_to(path)
        is_already_processed = relative_path.parts[0] in (
            "_imported",
            "#recycle",
            "_trash_bin",
            "_del_me",
        )

        yield_dict = SimpleNamespace(
            date=date,
            locality=locality,
            locality_exists=len(Locality.objects.filter(name=locality, **params)) > 0,
            zip_name_exists=zip_name in archives,
            is_already_processed=is_already_processed,
            path_of_locality_check=path_of_locality_check,
            path=str(relative_path),
            error_message=error_message,
            zip_name=zip_name,
            parent_dir_to_be_deleted=parent_dir_to_be_deleted,
        )

        yield yield_dict


def assign_unidentified_to_identification(caiduser: CaIDUser):
    """Assign unidentified media files to identification for the workgroup of the given user."""
    kwargs = {}
    taxon_str = ""
    if caiduser.workgroup is None:
        logger.error("CaIDUser has no workgroup assigned. Cannot assign unidentified media files to identification.")
    if caiduser.workgroup.default_taxon_for_identification is not None:
        taxon_str = caiduser.workgroup.default_taxon_for_identification.name
        kwargs["taxon__name"] = taxon_str
    else:
        logger.error("Workgroup has no default taxon for identification assigned. Using all media files.")
        # kwargs = {}

    logger.debug(
        f"Assigning unidentified media files to identification for {caiduser.workgroup} and taxon {taxon_str}."
    )
    # unused_mediafiles
    logger.debug(f"{models.MediaFile.objects.filter(identity__isnull=True).count()=}")
    logger.debug(f"{models.MediaFile.objects.filter(taxon__name=taxon_str).count()=}")
    logger.debug(f"{models.MediaFile.objects.filter(parent__owner__workgroup=caiduser.workgroup).count()=}")
    mf_taxon = models.MediaFile.objects.filter(parent__owner__workgroup=caiduser.workgroup, taxon__name=taxon_str)
    logger.debug(f"{mf_taxon.count()=}")
    mf_no_identity = models.MediaFile.objects.filter(parent__owner__workgroup=caiduser.workgroup, identity__isnull=True)
    logger.debug(f"{mf_no_identity.count()=}")
    mf_taxon_and_no_identity = models.MediaFile.objects.filter(
        parent__owner__workgroup=caiduser.workgroup, identity__isnull=True, taxon__name=taxon_str
    )
    logger.debug(f"{mf_taxon_and_no_identity.count()=}")

    existing_mfi_ids = MediafilesForIdentification.objects.values_list("mediafile_id", flat=True)
    logger.debug(f"Number of Mediafiles already in identification: {existing_mfi_ids.count()}")

    base_qs = models.MediaFile.objects.filter(
        parent__owner__workgroup=caiduser.workgroup, identity__isnull=True, **kwargs
    )
    logger.debug(f"Base queryset count (before exclude): {base_qs.count()}")

    mediafiles = base_qs.exclude(id__in=list(existing_mfi_ids)).select_related(
        "parent", "taxon", "predicted_taxon", "locality", "identity", "updated_by", "sequence"
    )

    # mediafiles = models.MediaFile.objects.filter(
    #     parent__owner__workgroup=caiduser.workgroup,
    #     taxon__name=taxon_str,
    #     identity__isnull=True,
    # ).exclude(
    #     id__in=Subquery(MediafilesForIdentification.objects.values('mediafile_id'))
    # )

    identities = models.IndividualIdentity.objects.filter(
        owner_workgroup=caiduser.workgroup,
    )

    for unknown_mediafile in mediafiles:
        # check if mediafile is already in MediafilesForIdentification
        mfi, _ = MediafilesForIdentification.objects.get_or_create(
            mediafile=unknown_mediafile,
        )
        # place for some similarity between identities and filename of current file
        # try to find some of identity name in unknown_mediafile.original_filename

        orig_fn = str(unknown_mediafile.original_filename).lower()
        for identity in identities:
            if len(identity.name) > 0:
                score = 0
                if identity.name.lower() in orig_fn:
                    score += 0.1
                if identity.code and (identity.code.lower() in orig_fn):
                    score += 0.1
                if score > 0:
                    identity_mediafile = identity.mediafile_set.filter(
                        identity_is_representative=True,
                    ).first()
                    mfi_suggestion = models.MediafileIdentificationSuggestion(
                        for_identification=mfi,
                        mediafile=identity_mediafile,
                        identity=identity,
                        score=score,
                        name=identity.name,
                    )
                    mfi_suggestion.save()
    logger.debug(f"Unidentified media files added to the list. Found {mediafiles.count()} media files.")


@shared_task(bind=True)
def refresh_identities_suggestions_task(self, workgroup_id, limit=100):
    """Refresh identities suggestions task."""
    def report_progress(**progress):
        self.update_state(state="PROGRESS", meta=progress)

    result_id = compute_identity_suggestions(workgroup_id, limit, progress_callback=report_progress)
    return {"result_id": result_id}


@shared_task(bind=True)
def compute_identity_code_suggestions_task(self, workgroup_id: int):
    """Compute identity code suggestions with progress updates."""
    queryset = IndividualIdentity.objects.filter(owner_workgroup_id=workgroup_id).order_by("id")
    total = queryset.count()
    suggestion_ids: list[int] = []

    if total == 0:
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 0, "matches": 0, "message": "No identities to scan."},
        )
        return {"suggestion_ids": [], "total": 0, "matches": 0}

    for index, identity in enumerate(queryset.iterator(chunk_size=200), start=1):
        suggested_code = identity.suggested_code_from_name()
        if suggested_code:
            suggestion_ids.append(identity.id)

        if index == 1 or index == total or index % 50 == 0:
            self.update_state(
                state="PROGRESS",
                meta={
                    "current": index,
                    "total": total,
                    "matches": len(suggestion_ids),
                    "message": f"Checked {index} of {total} identities.",
                },
            )

    return {"suggestion_ids": suggestion_ids, "total": total, "matches": len(suggestion_ids)}


def run_identification_outlier_detection_for_workgroup(
    workgroup: WorkGroup,
) -> models.IdentificationOutlierSuggestionResult:
    """Prepare metadata and start identification outlier detection for a workgroup."""
    mediafiles_qs = (
        MediaFile.objects.filter(
            parent__owner__workgroup=workgroup,
            identity__isnull=False,
        )
        .select_related("identity", "locality", "sequence", "parent")
        .order_by("id")
    )

    # output_dir = Path(settings.MEDIA_ROOT) / workgroup.name / workgroup.hash
    # output_dir.mkdir(exist_ok=True, parents=True)
    # metadata_file = output_dir / "identification_outliers.csv"
    metadata_file = workgroup.file_path("identification_outliers.csv")
    metadata_file.parent.mkdir(exist_ok=True, parents=True)

    csv_data = _prepare_dataframe_for_identification(mediafiles_qs)
    pd.DataFrame(csv_data).to_csv(metadata_file, index=False)

    result = models.IdentificationOutlierSuggestionResult.objects.create(
        workgroup=workgroup,
        status="processing",
        message=f"Prepared {len(csv_data['image_path'])} identified media files for outlier detection.",
        suggestions=[],
    )

    media_root = Path(settings.MEDIA_ROOT)
    mediafile_paths = [str(media_root / mf.image_file.name) for mf in mediafiles_qs]

    sig = signature(
        "detect_identification_outliers",
        kwargs={
            "organization_id": workgroup.id,
            "input_metadata_file": str(metadata_file),
            "mediafile_paths": mediafile_paths,
            "identification_model": {
                "name": workgroup.identification_model.name if workgroup.identification_model else "",
                "path": (
                    str(workgroup.identification_model.model_path)
                    if workgroup.identification_model and workgroup.identification_model.model_path
                    else "hf-hub:strakajk/LynxV4-MegaDescriptor-v2-T-256"
                ),
            },
        },
    )
    sig.apply_async(
        link=identification_outlier_detection_on_success.s(result_id=result.id),
        link_error=identification_outlier_detection_on_error.s(result_id=result.id),
    )
    return result, metadata_file


@shared_task
def identification_outlier_detection_on_success(output: dict, *args, **kwargs):
    """Persist worker output of identification outlier detection."""
    result_id = kwargs.pop("result_id")
    result = models.IdentificationOutlierSuggestionResult.objects.get(id=result_id)
    workgroup = result.workgroup

    status = output.get("status", "unknown")
    result.status = "done" if status == "DONE" else str(status).lower()
    result.message = output.get("message", "")
    result.suggestions = _normalize_identification_outlier_suggestions(output.get("suggestions", []))
    result.save(update_fields=["status", "message", "suggestions"])

    if workgroup is not None:
        models.Notification.create_for(
            message=f"Identification outlier detection finished for workgroup {workgroup}.",
            workgroups=[workgroup],
            level=models.Notification.INFO,
        )

    return result.id


def _resolve_mediafile_from_worker_path(path_str: str) -> MediaFile | None:
    """Resolve worker image path back to MediaFile when possible."""
    if not path_str:
        return None

    media_root = Path(settings.MEDIA_ROOT)
    normalized_path_str = str(path_str).replace("/masked_images/", "/images/")
    try:
        relpath = Path(normalized_path_str).relative_to(media_root)
    except Exception:
        return None

    return MediaFile.objects.filter(image_file=str(relpath)).first()


def _normalize_identification_outlier_suggestions(raw_suggestions: list[dict]) -> list[dict]:
    """Enrich worker suggestions with API-side mediafile ids for the current UI."""
    normalized = []
    for raw_item in raw_suggestions or []:
        item = dict(raw_item)

        if not item.get("suspicious_mediafile_id") and item.get("suspicious_path"):
            suspicious_mediafile = _resolve_mediafile_from_worker_path(item["suspicious_path"])
            if suspicious_mediafile is not None:
                item["suspicious_mediafile_id"] = suspicious_mediafile.id

        candidate_items = []
        for raw_candidate in item.get("suggestions", []):
            candidate = dict(raw_candidate)
            if not candidate.get("mediafile_id"):
                candidate_path = candidate.get("mediafile_path") or candidate.get("candidate_path")
                if candidate_path:
                    candidate_mediafile = _resolve_mediafile_from_worker_path(candidate_path)
                    if candidate_mediafile is not None:
                        candidate["mediafile_id"] = candidate_mediafile.id
            candidate_items.append(candidate)

        item["suggestions"] = candidate_items
        normalized.append(item)

    return normalized


@shared_task(bind=True)
def identification_outlier_detection_on_error(self, task_id: str, *args, **kwargs):
    """Persist worker error of identification outlier detection."""
    result_id = kwargs.pop("result_id")
    result = models.IdentificationOutlierSuggestionResult.objects.get(id=result_id)
    async_result = self.AsyncResult(task_id)
    error_message = str(async_result.result) if async_result.failed() else "Unknown worker error."

    result.status = "error"
    result.message = error_message
    result.save(update_fields=["status", "message"])

    if result.workgroup is not None:
        models.Notification.create_for(
            message=f"Identification outlier detection failed for workgroup {result.workgroup}: {error_message}",
            workgroups=[result.workgroup],
            level=models.Notification.ERROR,
        )

    return result.id
