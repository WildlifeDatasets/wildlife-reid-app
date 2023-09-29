import logging
import traceback
from pathlib import Path

from celery import Celery
from utils import data_processing_pipeline, dataset_tools
from utils.config import RABBITMQ_URL, REDIS_URL
from utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")
logger.debug(f"{RABBITMQ_URL=}")
logger.debug(f"{REDIS_URL=}")
inference_worker = Celery("inference_worker", broker=RABBITMQ_URL, backend=REDIS_URL)


@inference_worker.task(bind=True, name="predict")
def predict(
    self,
    input_archive_file: str,
    output_dir: str,
    output_archive_file: str,
    output_metadata_file: str,
    **kwargs,
):
    """Main method called by Celery broker."""
    try:
        logger.info(f"Applying inference task with args: {input_archive_file=}, {output_dir=}.")

        # prepare input and output file names
        input_archive_file = Path(input_archive_file)
        # assert input_archive_file.suffix.lower() in (".tar", ".tar.gz", ".zip")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir = output_dir / "images"
        output_archive_file = Path(output_archive_file)
        output_metadata_file = Path(output_metadata_file)

        # process data
        metadata = data_processing_pipeline.data_processing(
            input_archive_file,
            output_images_dir,
            output_metadata_file,
            num_cores=1,
        )
        logger.debug("")
        dataset_tools.make_zipfile_with_categories(output_archive_file, output_images_dir, metadata)
        # dataset_tools.make_zipfile(output_archive_file, output_images_dir)

        logger.info("Finished processing.")
        out = {"status": "DONE"}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out
