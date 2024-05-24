import logging
import shutil
import traceback
from pathlib import Path

import detection_utils.inference
import detection_utils.video_inference
import pandas as pd
from celery import Celery
from detection_utils.video_inference import create_image_from_video
from inference_utils import data_processing_pipeline, dataset_tools
from inference_utils.config import RABBITMQ_URL, REDIS_URL
from inference_utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")
logger.debug(f"{RABBITMQ_URL=}")
logger.debug(f"{REDIS_URL=}")
taxon_worker = Celery("taxon_worker", broker=RABBITMQ_URL, backend=REDIS_URL)
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
    **kwargs,
):
    """Main method called by Celery broker.

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

        if do_init:
            shutil.rmtree(output_images_dir, ignore_errors=True)
            # create metadata dataframe
            metadata, _ = data_processing_pipeline.data_preprocessing(
                input_archive_file,
                output_images_dir,
                num_cores=num_cores,
                contains_identities=contains_identities,
            )
            metadata = data_processing_pipeline.keep_correctly_loaded_images(metadata)
            # image_path is now relative to output_images_dir
            metadata["full_image_path"] = metadata["image_path"].apply(
                lambda x: str(output_images_dir / x)
            )
            metadata["absolute_media_path"] = [pth for pth in metadata["full_image_path"]]
            metadata["detection_results"] = [None] * len(metadata)
            metadata = create_image_from_video(metadata)
            metadata = data_processing_pipeline.keep_correctly_loaded_images(metadata)
        else:
            logger.debug(
                f"Using existing metadata file: {output_metadata_file}. "
                + f"{output_metadata_file.exists()=}"
            )
            # print size of file in bytes
            logger.debug(f"{output_metadata_file=}, {output_metadata_file.stat().st_size=}")
            # read file as str
            metadata = pd.read_csv(output_metadata_file, index_col=0)
            metadata["full_image_path"] = metadata["image_path"].apply(
                lambda x: str(output_images_dir / x)
            )
            metadata["absolute_media_path"] = [pth for pth in metadata["full_image_path"]]
            metadata["detection_results"] = [None] * len(metadata)

        logger.debug(f"Metadata file: {output_metadata_file}. {output_metadata_file.exists()=}")
        logger.debug(f"{len(metadata['image_path'])=}")
        if len(metadata["image_path"]) > 0:
            logger.debug(
                f"{metadata['image_path'][0]=}, {Path(metadata['image_path'][0]).exists()=}"
            )
            logger.debug(
                f"{metadata['full_image_path'][0]=}, "
                f"{Path(metadata['full_image_path'][0]).exists()=}"
            )

        metadata = detection_utils.inference.detect_animal_on_metadata(metadata)
        data_processing_pipeline.run_inference(metadata)
        metadata.to_csv(output_metadata_file, encoding="utf-8-sig")

        # process data
        # if output_images_dir.exists():
        #     # TODO turn of the following line
        #     metadata = pd.read_csv(output_metadata_file, index_col=0)
        # else:
        #     metadata = data_processing_pipeline.data_processing(
        #         input_archive_file,
        #         output_images_dir,
        #         output_metadata_file,
        #         num_cores=1,
        #         contains_identities=contains_identities,
        #     )
        # detect_and_segment_animal_on_metadata(metadata)
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
