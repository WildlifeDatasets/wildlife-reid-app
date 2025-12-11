import logging
import shutil
import traceback
from pathlib import Path

import pandas as pd
from celery import Celery
from detection_utils import inference_detection
from detection_utils.inference_video import create_image_from_video

try:
    from taxon_utils import data_processing_pipeline, dataset_tools
    from taxon_utils.config import RABBITMQ_URL, REDIS_URL
    from taxon_utils.log import setup_logging
except ModuleNotFoundError:
    from .taxon_utils import data_processing_pipeline, dataset_tools
    from .taxon_utils.config import RABBITMQ_URL, REDIS_URL
    from .taxon_utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")
logger.debug(f"{RABBITMQ_URL=}")
logger.debug(f"{REDIS_URL=}")
logger.debug("--------------------worker.py------------------logger.debug-------------")
try:
    taxon_worker = Celery("taxon_worker", broker=RABBITMQ_URL, backend=REDIS_URL)
except Exception as e:
    print(traceback.format_exc())
    raise e

MEDIA_DIR_PATH = Path("/shared_data/media")


@taxon_worker.task(bind=True, name="predict")
def predict(
    self,
    input_archive_file: str,
    output_dir: str,
    output_archive_file: str,
    output_metadata_file: str,
    contains_identities: bool = False,
    force_init: bool = False,
    sequence_time_limit_s: int = 120,
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
    try:
        logger.info(
            "Applying species identification task with args: "
            + f"{input_archive_file=}, {output_dir=}, {contains_identities=}."
        )
        logger.debug(f"celery {self.request.id=}")
        num_cores = 1

        # prepare input and output file names
        input_archive_file = Path(input_archive_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir = output_dir / "images"
        output_archive_file = Path(output_archive_file)
        output_metadata_file = Path(output_metadata_file)
        do_init = force_init or (not output_metadata_file.exists())
        post_update_csv_name: str = "mediafile.post_update.csv"
        post_update_csv_path = output_dir / post_update_csv_name

        # if spreadsheet is provided in uploaded_archive, save it to post_update_csv_path
        # and use it for data update

        if do_init:
            shutil.rmtree(output_images_dir, ignore_errors=True)
            # create metadata dataframe
            metadata, _ = data_processing_pipeline.data_preprocessing(
                input_archive_file,
                media_dir_path=output_images_dir,
                num_cores=num_cores,
                contains_identities=contains_identities,
                post_update_csv_path=post_update_csv_path,
                sequence_time_limit_s=sequence_time_limit_s,
            )
            metadata, df_failing0 = data_processing_pipeline.keep_correctly_loaded_images(metadata)
            # image_path is now relative to output_images_dir
            metadata["full_image_path"] = metadata["image_path"].apply(lambda x: str(output_images_dir / x))
            metadata["absolute_media_path"] = [pth for pth in metadata["full_image_path"]]
            metadata["detection_results"] = [None] * len(metadata)
            metadata = create_image_from_video(metadata)
            metadata, df_failing1 = data_processing_pipeline.keep_correctly_loaded_images(metadata)
            pd.concat([df_failing0, df_failing1]).to_csv(
                output_metadata_file.with_suffix(".failed.csv"), encoding="utf-8-sig"
            )
        else:
            logger.debug(
                f"Using existing metadata file: {output_metadata_file}. " + f"{output_metadata_file.exists()=}"
            )
            # print size of file in bytes
            logger.debug(f"{output_metadata_file=}, {output_metadata_file.stat().st_size=}")
            # read file as str
            metadata = pd.read_csv(output_metadata_file, index_col=0)
            metadata["full_image_path"] = metadata["image_path"].apply(lambda x: str(output_images_dir / x))
            metadata["absolute_media_path"] = [pth for pth in metadata["full_image_path"]]
            metadata["detection_results"] = [None] * len(metadata)

        logger.debug(f"Metadata file: {output_metadata_file}. {output_metadata_file.exists()=}")
        logger.debug(f"{len(metadata['image_path'])=}")
        if len(metadata["image_path"]) > 0:
            logger.debug(f"{metadata['image_path'][0]=}, {Path(metadata['image_path'][0]).exists()=}")
            logger.debug(f"{metadata['full_image_path'][0]=}, " f"{Path(metadata['full_image_path'][0]).exists()=}")

        metadata = inference_detection.detect_animal_on_metadata(metadata)
        data_processing_pipeline.run_taxon_classification_inference(metadata)
        data_processing_pipeline.make_previews(metadata, output_dir, force=do_init)

        # Update metadata with post_update_csv if it exists
        # find and read zip or xlsx file in temp dir

        # if spreadsheet is provided in uploaded_archive, use it to update metadata
        if post_update_csv_path.exists():
            metadata = post_update_with_spreadsheet(metadata, post_update_csv_path)

        metadata.to_csv(output_metadata_file, encoding="utf-8-sig")

        logger.debug("Preparing output archive.")
        dataset_tools.make_zipfile_with_categories(output_archive_file, output_images_dir, metadata)
        logger.debug(f"{contains_identities=}")
        logger.debug(f"{output_archive_file=}")

        # dataset_tools.make_zipfile(output_archive_file, output_images_dir)

        logger.info("Finished processing.")
        out = {"status": "DONE"}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out


def post_update_with_spreadsheet(metadata, post_update_csv_path):
    """Update metadata dataframe with values from post_update_csv_path."""
    # this is not well tested
    metadata_post_update = pd.read_csv(post_update_csv_path, index_col=0)
    if "original_path" in metadata_post_update.columns:
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
