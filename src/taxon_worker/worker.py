import json
import logging
import shutil
import time
import traceback
from hashlib import sha256
from pathlib import Path

import pandas as pd

# Celery adds the working directory to sys.path only while loading this module.
from bbox_detection import detect_file
from celery import Celery
from celery.exceptions import Ignore
from detection_utils import inference_detection
from detection_utils.errors import DetectorUnavailableError
from detection_utils.inference_video import create_image_from_video
from progress import ProgressReporter

try:
    from taxon_utils import data_processing_pipeline, dataset_tools
    from taxon_utils.config import RABBITMQ_URL, REDIS_URL
    from taxon_utils.log import setup_logging
except ModuleNotFoundError as exc:
    if exc.name != "taxon_utils":
        raise
    from .taxon_utils import data_processing_pipeline, dataset_tools
    from .taxon_utils.config import RABBITMQ_URL, REDIS_URL
    from .taxon_utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")
logger = logging.getLogger(__file__)
logger.debug(f"{RABBITMQ_URL=}")
logger.debug(f"{REDIS_URL=}")
logger.debug("--------------------worker.py------------------logger.debug-------------")
try:
    taxon_worker = Celery("taxon_worker", broker=RABBITMQ_URL, backend=REDIS_URL)
except Exception as e:
    print(traceback.format_exc())
    raise e

MEDIA_DIR_PATH = Path("/shared_data/media")


@taxon_worker.task(bind=True, name="redetect_bboxes", soft_time_limit=300, time_limit=360)
def redetect_bboxes(self, relative_path, options):
    """Run bbox detection for one media file."""
    task_id = self.request.id
    logger.info(
        "BBox detection started task_id=%s source=%s detector=%s confidence=%s",
        task_id,
        Path(relative_path).name,
        options.get("detector"),
        options.get("confidence"),
    )
    try:
        result = detect_file(relative_path, options, task_id=task_id)
        logger.info(
            "BBox detection finished task_id=%s status=%s detector=%s detections=%d",
            task_id,
            result.get("status"),
            result.get("detector"),
            len(result.get("detections", [])),
        )
        return result
    except DetectorUnavailableError as exc:
        logger.exception("BBox detector unavailable task_id=%s source=%s", task_id, Path(relative_path).name)
        return {"status": "error", "message": str(exc)}
    except Exception:
        logger.exception("BBox detection failed task_id=%s source=%s", task_id, Path(relative_path).name)
        return {"status": "error", "message": "Detection failed. Check the taxon worker logs and model availability."}
    finally:
        try:
            if not inference_detection.KEEP_DETECTION_MODEL_LOADED:
                inference_detection.del_detection_model()
            inference_detection.del_sam3_model()
            logger.info("BBox model cleanup finished task_id=%s", task_id)
        except Exception:
            logger.exception("BBox model cleanup failed task_id=%s", task_id)


def infer_prepared_media(metadata, progress=None, *, save_detection_images=True):
    """Detect and classify only the prepared media rows supplied by the caller."""
    if metadata.empty:
        raise ValueError("No prepared media files to process.")
    metadata = metadata.reset_index(drop=True).copy()
    metadata["detection_results"] = None
    on_sam3_download = None
    if progress is not None:
        progress.stage("detection", "Detecting animals")
        on_sam3_download = progress.report_sam3_download
    metadata = inference_detection.detect_animal_on_metadata(
        metadata,
        progress_callback=progress.callback("Detecting animals") if progress is not None else None,
        save_detection_images=save_detection_images,
        on_sam3_download=on_sam3_download,
    )
    if progress is not None:
        progress.stage("classification", "Classifying taxa")
    data_processing_pipeline.run_taxon_classification_inference(metadata)
    return metadata


@taxon_worker.task(bind=True, name="predict_prepared_media")
def predict_prepared_media(self, files=None, *, archive=None):
    """Return inference for selected media IDs and their prepared image paths."""
    if archive is not None:
        if files is not None:
            raise ValueError("Supply either media entries or an archive manifest.")
        return _predict_prepared_manifest(self, **archive)

    media_root = MEDIA_DIR_PATH.resolve()
    rows = []
    seen = set()

    def resolve_path(value, media_id):
        relative_path = Path(value)
        path = (media_root / relative_path).resolve()
        if relative_path.is_absolute() or not path.is_relative_to(media_root) or not path.is_file():
            raise ValueError(f"Missing or invalid prepared path for media ID {media_id}.")
        return str(path)

    for entry in files:
        media_id = entry["media_id"]
        if type(media_id) is not int or media_id <= 0 or media_id in seen:
            raise ValueError("Media IDs must be positive and unique.")
        media_type = entry["media_type"]
        if media_type not in {"image", "video"}:
            raise ValueError(f"Unsupported media type for media ID {media_id}.")
        image_path = resolve_path(entry["image_path"], media_id)
        media_path = resolve_path(entry["media_path"], media_id) if media_type == "video" else image_path
        if media_type == "video" and media_path == image_path:
            raise ValueError(f"A prepared video frame is required for media ID {media_id}.")
        seen.add(media_id)
        rows.append(
            {
                "media_id": media_id,
                "media_type": media_type,
                "full_image_path": image_path,
                "absolute_media_path": media_path,
            }
        )

    metadata = infer_prepared_media(pd.DataFrame(rows), save_detection_images=False)
    results = metadata.drop(columns=["full_image_path", "absolute_media_path"])
    return json.loads(results.to_json(orient="records", double_precision=15))


def _prepare_archive_metadata(
    input_archive_file,
    output_dir,
    output_metadata_file,
    progress,
    *,
    do_init,
    contains_identities,
    sequence_time_limit_s=None,
    path_structure_regex=None,
    path_structure_mapping=None,
):
    output_images_dir = output_dir / "images"
    post_update_csv_path = output_dir / "mediafile.post_update.csv"
    num_cores = 1
    logger.info("Import pipeline stage: prepare metadata")
    if do_init:
        progress.stage("prepare_metadata", "Preparing uploaded media")
        shutil.rmtree(output_images_dir, ignore_errors=True)
        post_update_csv_path.unlink(missing_ok=True)
        metadata, _ = data_processing_pipeline.data_preprocessing(
            input_archive_file,
            media_dir_path=output_images_dir,
            num_cores=num_cores,
            contains_identities=contains_identities,
            post_update_csv_path=post_update_csv_path,
            sequence_time_limit_s=sequence_time_limit_s,
            path_structure_regex=path_structure_regex,
            path_structure_mapping=path_structure_mapping,
        )
        metadata, df_failing0 = data_processing_pipeline.keep_correctly_loaded_images(metadata)
        logger.debug("Prepared metadata from archive: %s rows", len(metadata))
        logger.info("Import pipeline stage: build video previews")
        progress.stage("video_previews", "Extracting preview images from videos")
        metadata["full_image_path"] = metadata["image_path"].apply(lambda x: str(output_images_dir / x))
        metadata["absolute_media_path"] = [pth for pth in metadata["full_image_path"]]
        metadata["detection_results"] = [None] * len(metadata)
        metadata = create_image_from_video(
            metadata,
            progress_callback=progress.callback("Extracting preview images from videos"),
        )
        metadata, df_failing1 = data_processing_pipeline.keep_correctly_loaded_images(metadata)
        pd.concat([df_failing0, df_failing1]).to_csv(
            output_metadata_file.with_suffix(".failed.csv"), encoding="utf-8-sig"
        )
    else:
        progress.stage("load_metadata", "Loading prepared metadata")
        logger.info("Using existing metadata file: %s", output_metadata_file)
        logger.debug(f"{output_metadata_file=}, {output_metadata_file.stat().st_size=}")
        metadata = pd.read_csv(output_metadata_file, index_col=0)
        metadata["full_image_path"] = metadata["image_path"].apply(lambda x: str(output_images_dir / x))
        if "absolute_media_path" not in metadata:
            metadata["absolute_media_path"] = metadata["full_image_path"]
        metadata["detection_results"] = None
        metadata, df_failing = data_processing_pipeline.keep_correctly_loaded_images(metadata)
        if not df_failing.empty:
            df_failing.to_csv(output_metadata_file.with_suffix(".failed.csv"), encoding="utf-8-sig")
        logger.debug("Loaded existing metadata rows: %s", len(metadata))

    if metadata.empty:
        raise FileNotFoundError("No readable media files remain after preparing the upload.")

    return metadata


def _finish_archive_metadata(metadata, output_dir, output_archive_file, output_metadata_file, progress, *, do_init):
    progress.stage("previews", "Creating media previews")
    data_processing_pipeline.make_previews(
        metadata, output_dir, force=do_init, progress_callback=progress.callback("Creating media previews")
    )
    spreadsheet = output_dir / "mediafile.post_update.csv"
    if spreadsheet.exists():
        metadata = post_update_with_spreadsheet(metadata, spreadsheet)
    progress.stage("finalize", "Writing metadata and output archive")
    metadata.to_csv(output_metadata_file, encoding="utf-8-sig")
    dataset_tools.make_zipfile_with_categories(output_archive_file, output_dir / "images", metadata)
    progress.update(1, 1, message="Handing results back for import", force=True)


def _archive_progress(task, output_dir, *, do_init):
    task_id = task.request.id
    marker = None
    if task_id:
        marker = Path(output_dir) / f".taxon-cancelled-{sha256(task_id.encode()).hexdigest()}"
    return ProgressReporter(task, do_init=do_init, cancellation_marker=marker)


@taxon_worker.task(bind=True, name="prepare_taxon_archive")
def prepare_taxon_archive(
    self, input_archive_file, output_dir, output_metadata_file, manifest_path, contains_identities=False,
    force_init=False, path_structure_regex=None, path_structure_mapping=None,
):
    """Prepare archive media and hand the manifest to the API for registration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_file = Path(output_metadata_file)
    do_init = force_init or not output_metadata_file.exists()
    metadata = _prepare_archive_metadata(
        Path(input_archive_file),
        output_dir,
        output_metadata_file,
        _archive_progress(self, output_dir, do_init=do_init),
        do_init=do_init, contains_identities=contains_identities, path_structure_regex=path_structure_regex,
        path_structure_mapping=path_structure_mapping,
    )
    metadata.to_csv(manifest_path, encoding="utf-8-sig")
    return {"status": "PREPARED"}


def _predict_prepared_manifest(task, manifest_path, output_dir, output_archive_file, output_metadata_file):
    """Run inference and produce legacy export artifacts from the registered media manifest."""
    metadata = pd.read_csv(manifest_path, index_col=0)
    if metadata["media_id"].isna().any() or not metadata["media_id"].is_unique:
        raise ValueError("Each prepared media row must have a unique registered media ID.")
    progress = _archive_progress(task, output_dir, do_init=True)
    metadata = infer_prepared_media(metadata, progress)
    if metadata["detection_results"].isna().any():
        raise ValueError("Detection did not produce results for all registered media.")
    _finish_archive_metadata(
        metadata, Path(output_dir), Path(output_archive_file), Path(output_metadata_file), progress, do_init=True,
    )
    return {"status": "DONE"}


# TODO(remove-legacy-predict): Remove this block with the cleanup in TODO.md.
# @taxon_worker.task(bind=True, name="predict")
def predict_legacy_archive(
    self,
    input_archive_file: str,
    output_dir: str,
    output_archive_file: str,
    output_metadata_file: str,
    contains_identities: bool = False,
    force_init: bool = False,
    sequence_time_limit_s: int | None = None,
    detection_model_path: str = None,
    detection_model_architecture: str = None,
    path_structure_regex: str = None,
    path_structure_mapping: dict = None,
    **kwargs,
):
    """Prepare import data and species classification inference.

    What it does:
    - unzips input_archive_file
    - prepares metadata dataframe
    - extract date/time from EXIF if possible
    - runs species classification inference
    - save images into .webp format
    - creates preview images
    - saves output archive and metadata

    If the output_metadata_file does not exist, the metadata is
    created based on the content of input_archive_file and saved to output_metadata_file.
    If the output_metadata_file exists, it is directly used as input for the inference.
    """
    # TODO implement detection_model_path and detection_model_architecture parameters
    # if empty string is given, select whole media file for detection.
    # detection_model_path=r"https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt",
    # detection_model_architecture="ultralytics/yolov5:915bbf2",
    try:
        task_started_at = time.monotonic()
        logger.info(
            "Applying species identification task with args: "
            + f"{input_archive_file=}, {output_dir=}, {contains_identities=}."
        )
        logger.debug(f"celery {self.request.id=}")
        # prepare input and output file names
        input_archive_file = Path(input_archive_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_archive_file = Path(output_archive_file)
        output_metadata_file = Path(output_metadata_file)
        do_init = force_init or (not output_metadata_file.exists())
        progress = _archive_progress(self, output_dir, do_init=do_init)

        metadata = _prepare_archive_metadata(
            input_archive_file, output_dir, output_metadata_file, progress, do_init=do_init,
            contains_identities=contains_identities, sequence_time_limit_s=sequence_time_limit_s,
            path_structure_regex=path_structure_regex, path_structure_mapping=path_structure_mapping,
        )
        metadata = infer_prepared_media(metadata, progress)
        _finish_archive_metadata(
            metadata, output_dir, output_archive_file, output_metadata_file, progress, do_init=do_init,
        )
        logger.info("Finished processing in %.1fs", time.monotonic() - task_started_at)
        out = {"status": "DONE"}
    except Ignore:
        raise
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out


def post_update_with_spreadsheet(metadata, post_update_csv_path):
    """Update metadata dataframe with values from post_update_csv_path."""
    # this is not well tested
    metadata_post_update = pd.read_csv(post_update_csv_path, index_col=0).drop(columns=["media_id"], errors="ignore")
    if "original_path" in metadata_post_update.columns:
        if "media_id" in metadata:
            metadata_post_update = metadata_post_update.drop_duplicates("original_path", keep="last")
        # join on column "original_path" if the column exists, use the column from post_update_metadata
        logger.debug(f"{metadata['original_path']=}")
        logger.debug(f"{metadata_post_update['original_path']=}")
        logger.debug(f"{metadata.columns=}")
        logger.debug(f"{metadata_post_update.columns=}")

        # Perform an inner join to ensure only rows from metadata are retained
        merged_df = metadata.merge(
            metadata_post_update,
            on="original_path",
            how="left",  # Keeps all rows from `metadata`, matching only from `metadata_post_update`
            suffixes=("", "_post_update"),
        )
        logger.debug("Merging metadata with post_update_csv.")
        logger.debug(f"{merged_df.shape=}")
        # print sample of 5 records
        logger.debug(f"{merged_df.head(5)=}")

        # Overwrite columns from `metadata` with those from `metadata_post_update` if they exist
        for col in metadata_post_update.columns:
            if col != "original_path":  # Skip the join column
                if col in metadata.columns:  # Only overwrite common columns
                    if col == "datetime" and "datetime_source" in metadata.columns:
                        merged_df.loc[merged_df["datetime_post_update"].notna(), "datetime_source"] = "Spreadsheet"
                    merged_df[col] = merged_df[f"{col}_post_update"].combine_first(merged_df[col])
                    merged_df.drop(columns=[f"{col}_post_update"], inplace=True)
        metadata = merged_df
        logger.debug("Merged metadata with post_update_csv.")
        logger.debug(f"{metadata.shape=}")
    else:
        logger.warning("Column 'original_path' not found in post_update_csv. Skipping.")
        logger.debug(f"{post_update_csv_path=}")
        logger.debug(f"{metadata_post_update.columns=}")
    return metadata
