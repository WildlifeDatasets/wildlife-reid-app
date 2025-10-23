import copy
import ast
import datetime
import json
import logging
import os
import os.path
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Generator
import tempfile
import traceback
import tqdm

import django
import numpy as np
import pandas as pd
from celery import chain, shared_task, signature, current_app
from django.conf import settings
from zoneinfo import ZoneInfo

from . import fs_data, model_tools, views
from .fs_data import make_thumbnail_from_file
from .log_tools import StatusCounts
import Levenshtein
from django.contrib.auth import get_user_model
from .model_extra import compute_identity_suggestions
from . import models
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
    get_unique_name, user_has_access_filter_params,
)

# from joblib import Parallel, delayed
# from tqdm import tqdm

logger = logging.getLogger("app")


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
    print(f"Taxon classification finished with status {status}")
    logger.info(f"Taxon classification finished with status '{status}'. Updating database record.")
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)

    try:
        if "status" not in output:
            logger.critical(f"Unexpected error {output=} is missing 'status' field.")
            uploaded_archive.taxon_status = "U"
            uploaded_archive.identification = "U"
        elif output["status"] == "DONE":
            uploaded_archive.zip_file = zip_file
            uploaded_archive.csv_file = csv_file
            uploaded_archive.import_error_spreadsheet = str(Path(csv_file).with_suffix(".failed.csv"))
            make_thumbnail_for_uploaded_archive(uploaded_archive)
            # update_metadata_csv_by_uploaded_archive(uploaded_archive)
            # create missing take effect only if the processing is done for the first time
            # in other cases the file should be removed from CSV before the processing is run
            logger.debug(f"{uploaded_archive.contains_identities=}")
            update_uploaded_archive_by_metadata_csv(
                uploaded_archive, create_missing=True, extract_identites=extract_identites
            )
            if uploaded_archive.taxon_for_identification:
                assign_unidentified_to_identification(caiduser=uploaded_archive.owner)
            uploaded_archive.mediafiles_imported = True
            uploaded_archive.taxon_status = "TAID"
            uploaded_archive.identification_status = "IR"  # Ready for identification
            uploaded_archive.status_message = "Taxon classification finished."
            uploaded_archive.finished_at = django.utils.timezone.now()
            uploaded_archive.save()
            uploaded_archive.update_earliest_and_latest_captured_at()
            uploaded_archive.make_sequences()
            logger.debug("Running async detection on success taxon classification")
            # run_detection_async(uploaded_archive)  # this is probably not necessary
        else:
            uploaded_archive.taxon_status = "F"
            uploaded_archive.identification_status = "F"
            uploaded_archive.finished_at = django.utils.timezone.now()
            if "error" in output:
                logger.error(f"{output['error']=}")
                uploaded_archive.status_message = output["error"]
            uploaded_archive.save()
    except Exception as e:
        logger.debug(str(traceback.format_exc()))
        logger.error(f"Error during on_success_predict_taxon: {e}")
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
        csv_data["image_path"][i] = str(media_root / mediafile.image_file.name)
        csv_data["mediafile_id"][i] = mediafile.id
        csv_data["class_id"][i] = int(mediafile.identity.id) if mediafile.identity else None
        csv_data["label"][i] = str(mediafile.identity.name) if mediafile.identity else None
        csv_data["locality_id"][i] = int(mediafile.locality.id) if mediafile.locality else None
        csv_data["locality_name"][i] = str(mediafile.locality.name) if mediafile.locality else ""
        csv_data["locality_coordinates"][i] = (
            str(mediafile.locality.location) if (mediafile.locality and mediafile.locality.location) else None
        )
        csv_data["sequence_number"][i] = mediafile.sequence.local_id if mediafile.sequence else None
        # logger.debug(f"{mediafile.metadata_json=}")
        if mediafile.metadata_json and "detection_results" in mediafile.metadata_json:
            detection_results = mediafile.metadata_json["detection_results"]
        else:
            detection_results = None
        csv_data["detection_results"][i] = detection_results

    return csv_data


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
        self, caiduser_id: int,
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
        logger.debug(
            f"{yield_dict.path_of_locality_check=}, {yield_dict.path_of_locality_check.exists()=}"
        )
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
def do_cloud_import_for_user_async(caiduser: CaIDUser,
    contains_identities: bool = False,
    contains_single_taxon: bool = False
                                   ):
    """Run cloud import asynchronously."""
    # sig = do_cloud_import_for_user.s(caiduser=caiduser)
    # run async
    # sig.apply_async()
    caiduser.dir_import_status = "Processing"
    caiduser.save()
    sig = signature("caidapp.tasks.do_cloud_import_for_user",
                    kwargs={
                        "caiduser_id": caiduser.id,
                        "contains_single_taxon": contains_single_taxon,
                        "contains_identities": contains_identities,
                    })
    sig.apply_async()

@shared_task
def create_mediafiles_zip(user_hash, mediafiles, abs_zip_path):
    """Create a zip file for media files in the background."""
    logger.debug(f"Creating zip file for {len(mediafiles)} media files.")
    abs_zip_path = Path(abs_zip_path)
    abs_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdirname:
        mediafiles_dir = Path(tmpdirname) / "images"
        mediafiles_dir.mkdir()
        for mediafile in tqdm.tqdm(mediafiles):
            src = Path(settings.MEDIA_ROOT) / mediafile["path"]
            dst = mediafiles_dir / mediafile["output_name"]
            shutil.copy(src, dst)
        # Assume `make_zipfile` is your custom function to create a zip
        logger.debug(f"Creating zip file {abs_zip_path}")
        make_zipfile(abs_zip_path, mediafiles_dir)
    logger.debug(f"Zip file created: {abs_zip_path}")
    return str(abs_zip_path)

@shared_task
def clean_old_mediafile_zips(dirpath:str, glob_pattern: str = "mediafiles_*.zip", max_age_days: int = 7):
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

    shutil.make_archive(
        output_filename.parent / output_filename.stem, archive_type, root_dir=source_dir
    )


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
    else:
        sequence_time_limit_s = 120

    # send celery message to the data worker
    logger.info("Sending request to inference worker.")
    logger.debug(f"{uploaded_archive.contains_identities=}")
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
            "force_init": force_init,
            "sequence_time_limit_s": sequence_time_limit_s
        },
    )

    task = sig.apply_async(
        link=link,
        link_error=link_error,
    )
    logger.info(f"Created worker task with id '{task.task_id}'.")


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

    time_to_process = datetime.timedelta(seconds=10) + (
        (time_per_image * image_count) + (time_per_video * video_count)
    )
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
        except:
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
    uploaded_archive.locality_at_upload_object = locality
    uploaded_archive.save()


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
    image_abs_pth = output_dir / "images" / row["image_path"]
    image_rel_pth = image_abs_pth.relative_to(settings.MEDIA_ROOT)
    media_abs_pth = Path(row["absolute_media_path"])
    media_rel_pth = media_abs_pth.relative_to(settings.MEDIA_ROOT)
    captured_at = row["datetime"]
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
    try:
        mf = uploaded_archive.mediafile_set.get(mediafile=str(image_rel_pth))
        # logger.debug("Using Mediafile generated before")
        status = "found"
    except MediaFile.DoesNotExist:
        # convert pandas row to json
        if create_missing:
            # logger.debug(f"{row['detection_results']=}")

            # TODO use media_rel_pth instead of image_rel_pth
            mf = MediaFile(
                parent=uploaded_archive,
                # mediafile=str(image_rel_pth),
                mediafile=str(media_rel_pth),
                image_file=str(image_rel_pth),
                captured_at=captured_at,
                locality=locality,
                media_type=row["media_type"],
                # metadata_json=row["detection_results"],
                # metadata_json=metadata_json,
            )
            mf.save()
            observation = mf.observations.create(
                # mediafile=mf,
            )

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

    # generate thumbnail if necessary
    mf.make_thumbnail_for_mediafile_if_necessary(thumbnail_width=thumbnail_width)

    metadata_json = row.to_dict()
    # remove None and NaN values
    metadata_json = {k: v for k, v in metadata_json.items() if v is not None and not pd.isna(v)}
    # logger.debug(f"{metadata_json=}")
    mf.metadata_json = metadata_json

    # if the mediafile was updated by user, we believe into users input
    if mf.updated_by is None:
        status = status + " and updated by user"
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

        try:
            if ("detection_results" in row) and (row["detection_results"] is not None):
                detection_results = ast.literal_eval(row["detection_results"])
                if len(detection_results) > 0:
                    kv = {'back': "B", 'front': "F", 'left': "F", 'right': "R", "unknown": "U"}
                    if len(mf.first_observation) > 1:
                        # remove all observations with the exception of the first one
                        # todo probably we should map the observation results to observations
                        mf.observations.exclude(id=mf.first_observation.id).delete()
                    for i, one_detection_result in enumerate(detection_results):
                        if len(mf.observations) > 0 and len(detection_results) > 0:
                            ao = mf.observations.first()
                        else:
                            ao = mf.observations.create(
                                # mediafile=mf,
                                # metadata_json=row.to_dict(),
                            )

                        ao.taxon = taxon
                        ao.predicted_taxon = taxon
                        ao.predicted_taxon_confidence = taxon_confidence
                        ao.save()

                        orientation = detection_results[i]["orientation"]
                        orientation_score = detection_results[i]["orientation_score"]
                    if orientation in kv:
                        if orientation_score < orientation_score_threshold:
                            orientation = "unknown"

                        mf.orientation = kv[orientation]
                        ao.orientation = kv[orientation]
                    else:
                        logger.warning(f"Unknown orientation: {orientation} in {mf.mediafile}")
        except Exception as e:
            logger.warning(f"Error during setting orientation in media file {mf.mediafile}: {e}")
            logger.debug(traceback.format_exc())


        if extract_identites:
            mf.identity = get_unique_name(
                row["unique_name"], workgroup=uploaded_archive.owner.workgroup
            )
        mf.save()
        # logger.debug(f"identity={mf.identity}")
    return status

    # logger.debug(f"{mf}")


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
            logger.debug(
                "No enough information stored in webapp. "
                "The CSV file will be removed to be recreated."
            )
            return False
    return True


def create_dataframe_from_mediafiles(mediafiles: Generator[MediaFile, None, None]) -> pd.DataFrame:
    """Create DataFrame from MediaFiles."""
    records = []
    # go over mediafiles in set
    for mf in mediafiles:
        # logger.debug(f"{mf.metadata_json=}, {type(mf.metadata_json)=}")
        metadata_row = copy.copy(mf.metadata_json)
        # logger.debug(f"{metadata_row=}, {type(metadata_row)=}")
        if (metadata_row is None) or ("predicted_category" not in metadata_row):
            metadata_row = {}

        if mf.taxon:
            metadata_row["predicted_category"] = mf.taxon.name
        if mf.identity:
            metadata_row["unique_name"] = mf.identity.name
        if mf.locality:
            metadata_row["locality name"] = mf.locality.name
            if mf.locality.location:
                metadata_row["locality coordinates"] = str(mf.locality.location)
        if mf.original_filename:
            metadata_row["original_path"] = mf.original_filename
        if mf.identity:
            if mf.identity.code:
                metadata_row["code"] = mf.identity.code
            if mf.identity.juv_code:
                metadata_row["juv_code"] = mf.identity.juv_code
        metadata_row["uploaded_archive"] = mf.parent.name
        if mf.parent.locality_check_at:
            metadata_row["locality_check_at"] = mf.parent.locality_check_at

        records.append(metadata_row)
    df = pd.DataFrame.from_records(records)
    return df

def create_dataframe_from_mediafiles_NDOP(mediafiles: Generator[MediaFile, None, None]) -> pd.DataFrame:
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
    df.rename(columns={
        "locality_name": "locality name",
        "locality_coordinates": "locality coordinates",
    }, inplace=True)
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

        if mf.taxon:
            df.loc[index, "predicted_category"] = mf.taxon.name
            update_csv = True
        if mf.identity:
            df.loc[index, "unique_name"] = mf.identity.name
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
    models.Notification(message=f"Identification initialization finished. {args=} {kwargs=}").save()
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
    workgroup.identification_scheduled_run_eta = None
    workgroup.save()
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
    models.Notification(
        user=caiduser,
        message=f"Task finished with error. {args=} {kwargs=}",
    ).save()
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
    caiduser = None
    if "user_id" in kwargs:
        caiduser_id = kwargs.pop("caiduser_id")
        caiduser = models.CaIDUser.objects.get(id=caiduser_id)
    models.Notification(
        user=caiduser,
        message=f"Task finished with error. {args=} {kwargs=}",
    ).save()
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
    uploaded_archive = UploadedArchive.objects.get(id=uploaded_archive_id)
    # uploaded_archive.identification_status = "IAID"
    uploaded_archive.save()
    owner = uploaded_archive.owner
    workgroup = owner.workgroup


    try:
        if "status" not in output:
            msg = f"Unexpected error {output=} is missing 'status' field."
            logger.critical(msg)
            uploaded_archive.identification_status = "U"
            uploaded_archive.identification_message = msg
            uploaded_archive.save()

            # TODO - should the app return some error response to the user?
        elif output["status"] == "DONE":
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
                workgroup.save()
                logger.debug(f"Workgroup {workgroup} identification status set to 'IAID'.")

        else:
            # identification failed
            uploaded_archive.identification_status = "F"
            uploaded_archive.save()
            message = "Identification failed. "
            if "error" in output:
                logger.error(f"{output['error']=}")
                uploaded_archive.identification_message = output["error"]
                message += output["error"]
                if output["error"] == "Input data is empty.":
                    message += " Try to check the taxa in the input data."

            uploaded_archive.status_message = message
            logger.debug(f"{output=}")
            logger.error("Identification failed.")

    except Exception as e:
        uploaded_archive.identification_status = "F"
        uploaded_archive.status_message = f"Error during identification. {str(e)}"
        uploaded_archive.save()
        logger.error(f"Error during identification: {e}")
        logger.error(traceback.format_exc())


        # TODO - should the app return some error response to the user?


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
            logger.warning(
                f"Identity name mismatch: {unknown_mediafile.identity.name} != {reid_top_k_labels[0]}"
            )

        unknown_mediafile.save()

    else:
        mfi, _ = MediafilesForIdentification.objects.get_or_create(
            mediafile=unknown_mediafile,
        )
        # this try - except could be deleted
        try:
            top1_abspath = Path(reid_top_k_image_paths[0])
            top1_relpath = top1_abspath.relative_to(media_root)
            top1_mediafile = MediaFile.objects.get(mediafile=str(top1_relpath))

            top2_abspath = Path(reid_top_k_image_paths[1])
            top2_relpath = top2_abspath.relative_to(media_root)
            top2_mediafile = MediaFile.objects.get(mediafile=str(top2_relpath))

            top3_abspath = Path(reid_top_k_image_paths[2])
            top3_relpath = top3_abspath.relative_to(media_root)
            top3_mediafile = MediaFile.objects.get(mediafile=str(top3_relpath))



            mfi.top1mediafile = top1_mediafile
            mfi.top1score = reid_top_k_scores[0]
            mfi.top1name = reid_top_k_labels[0] or "Unknown"
            mfi.top2mediafile = top2_mediafile
            mfi.top2score = reid_top_k_scores[1]
            mfi.top2name = reid_top_k_labels[1] or "Unknown"
            mfi.top3mediafile = top3_mediafile
            mfi.top3score = reid_top_k_scores[2]
            mfi.top3name = reid_top_k_labels[2] or "Unknown"
            mfi.paired_points = data["keypoints"][i]
            # identification_output["query_image_path"] = query_image_path
            # identification_output["query_masked_path"] = query_masked_path
        except Exception as e:
            # logger.debug(f"{reid_top_k_image_paths=}")
            # logger.debug(traceback.format_exc())
            logger.error(f"Error during identification of {unknown_mediafile}: {e}")


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
                reid_top_k_class_ids, reid_top_k_scores, reid_top_k_labels, reid_top_k_image_paths, paired_points_for_k_images
        ):
            try:
                top_abspath = Path(top_path)
                top_relpath = top_abspath.relative_to(media_root)
                top_mediafile = MediaFile.objects.get(image_file=str(top_relpath))
                # top_mediafile = MediaFile.objects.get(mediafile=str(top_relpath))


                identity = IndividualIdentity.objects.get(id=identity_id)
                if identity.name != top_name:
                    logger.warning(
                        f"Identity name mismatch: {identity.name} != {top_name} for {unknown_mediafile=}"
                    )

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

        mfi.save()
        # _identity_mismatch_waning(top1_mediafile, top2_mediafile, top3_mediafile, top_k_labels)


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

from django.utils.timezone import now
from datetime import timedelta
from caidapp.models import WorkGroup

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
        workgroup.save(update_fields=[
            "identification_scheduled_run_task_id",
            "identification_scheduled_run_eta",
            "identification_reid_status"
        ])


@shared_task
def run_identification_on_unidentified_for_workgroup_task(workgroup_id: int):
    return run_identification_on_unidentified_for_workgroup.s(workgroup_id)

def run_identification_on_unidentified_for_workgroup(workgroup_id: int, request=None):
    from .views import run_identification
    workgroup = WorkGroup.objects.get(pk=workgroup_id)

    workgroup.identification_reid_status = "Processing"
    workgroup.save()

    uploaded_archives = UploadedArchive.objects.filter(
        owner__workgroup=workgroup,
        identification_status="IR",  # Ready for identification
        # contains_single_taxon=True,
        taxon_for_identification__isnull=False,
        contains_identities=False,
    ).all()

    for uploaded_archive in uploaded_archives:
        status_ok = run_identification(uploaded_archive, workgroup=workgroup)
        if request:
            from django.contrib import messages
            if status_ok:
                messages.info(request, f"Identification started for {uploaded_archive.name}.")
            else:
                # message is generated in run identification
                messages.error(request, f"No records for identification with the expected taxon for {uploaded_archive.name}.")

def schedule_init_identification_for_workgroup(workgroup: models.WorkGroup, delay_minutes: int = 15):

    # Zruš předchozí naplánovaný task
    if workgroup.identification_scheduled_init_task_id:
        current_app.control.revoke(workgroup.identification_scheduled_init_task_id, terminate=True)

    eta = now() + timedelta(minutes=delay_minutes)
    task = init_identification.apply_async(args=[workgroup.id], eta=eta)


    workgroup.identification_scheduled_init_task_id = task.id
    workgroup.identification_scheduled_init_eta = eta
    workgroup.identification_init_status = "Scheduled"

    workgroup.save(update_fields=[
        "identification_scheduled_init_task_id",
        "identification_scheduled_init_eta",
        "identification_init_status"
    ])

    schedule_reid_identification_for_workgroup(workgroup, delay_minutes=delay_minutes+50)


@shared_task
def init_identification(workgroup_id: int):
    from .views import _get_mediafiles_for_train_or_init_identification
    workgroup = WorkGroup.objects.get(pk=workgroup_id)

    process_for_message = "initialization"
    called_function_name = "init_identification"
    mediafiles_qs = _get_mediafiles_for_train_or_init_identification(
        workgroup
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
            f"Using {len(csv_data['image_path'])}"
            + f"representative images for identification {process_for_message}."
    )
    workgroup.save()
    logger.debug(f"Calling {process_for_message} identification...")
    sig = signature(
        called_function_name,
        # "init_identification",
        kwargs={
            # csv file should contain image_path, class_id, label
            "input_metadata_file": str(identity_metadata_file),
            "organization_id": workgroup.id,
            "identification_model": {
                "name": workgroup.identification_model.name,
                "path": workgroup.identification_model.model_path,
            }
        },
    )
    # task =
    sig.apply_async(
        link=init_identification_on_success.s(
            workgroup_id=workgroup.id,
            # uploaded_archive_id=uploaded_archive.id,
            # zip_file=os.path.relpath(str(output_archive_file), settings.MEDIA_ROOT),
            # csv_file=os.path.relpath(str(output_metadata_file), settings.MEDIA_ROOT),
        ),
        link_error=init_identification_on_error.s(
            # uploaded_archive_id=uploaded_archive.id
        ),
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


def _iterate_over_locality_checks(
    path: Path, caiduser: CaIDUser
) -> Generator[SimpleNamespace, None, None]:
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
            logger.debug(
                "Name of the directory or file is not in format {YYYY-MM-DD}_{locality_name}."
                + "Skipping."
            )
            error_message = "Name of the directory or file is not in correct format. " + "Skipping."
            locality = ""
            date = ""

        # locality = path_of_locality_check.parts[-2]
        # date = path_of_locality_check.parts[-1]
        # logger.debug(f"{path_of_locality_check.parts=}")

        # remove diacritics and spaces from zip_name
        zip_name = fs_data.remove_diacritics(f"{locality}_{date}.zip").replace(" ", "_")

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

def assign_unidentified_to_identification(caiduser:CaIDUser):
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

    from django.db.models import Subquery

    logger.debug(f"Assigning unidentified media files to identification for {caiduser.workgroup} and taxon {taxon_str}.")
    # unused_mediafiles
    logger.debug(f"{models.MediaFile.objects.filter(identity__isnull=True).count()=}")
    logger.debug(f"{models.MediaFile.objects.filter(taxon__name=taxon_str).count()=}")
    logger.debug(f"{models.MediaFile.objects.filter(parent__owner__workgroup=caiduser.workgroup).count()=}")
    logger.debug(f"{models.MediaFile.objects.filter(parent__owner__workgroup=caiduser.workgroup, taxon__name=taxon_str).count()=}")
    logger.debug(f"{models.MediaFile.objects.filter(parent__owner__workgroup=caiduser.workgroup, identity__isnull=True).count()=}")
    logger.debug(f"{models.MediaFile.objects.filter(parent__owner__workgroup=caiduser.workgroup, identity__isnull=True, taxon__name=taxon_str).count()=}")

    existing_mfi_ids = MediafilesForIdentification.objects.values_list('mediafile_id', flat=True)
    logger.debug(f"Number of Mediafiles already in identification: {existing_mfi_ids.count()}")

    base_qs = models.MediaFile.objects.filter(
        parent__owner__workgroup=caiduser.workgroup,
        identity__isnull=True,
        **kwargs
    )
    logger.debug(f"Base queryset count (before exclude): {base_qs.count()}")

    mediafiles = base_qs.exclude(id__in=list(existing_mfi_ids)).select_related(
        "parent",
        "taxon",
        "predicted_taxon",
        "locality",
        "identity",
        "updated_by",
        "sequence"
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
                if identity.code.lower() in orig_fn:
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


@shared_task
def refresh_identities_suggestions_task(workgroup_id, limit=100):
    return compute_identity_suggestions(workgroup_id, limit)

