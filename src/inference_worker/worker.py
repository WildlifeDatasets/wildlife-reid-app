import logging
import os
import traceback
from pathlib import Path

from celery import Celery
from utils import data_processing_pipeline, dataset_tools
from utils.log import setup_logging

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
REDIS_URL = os.environ["REDIS_URL"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]

setup_logging()
logger = logging.getLogger("app")
inference_worker = Celery("inference_worker", broker=RABBITMQ_URL, backend=REDIS_URL)

logger.info(f"{RABBITMQ_URL=}; {REDIS_URL=}")  # TODO - tmp

# def run_processing_test(uploaded_archive: UploadedArchive):
#     """TODO add docstring."""
#     outputdir = Path(uploaded_archive.outputdir)
#     if outputdir.exists() and outputdir.is_dir():
#         shutil.rmtree(outputdir, ignore_errors=True)
#     outputdir.mkdir(parents=True, exist_ok=True)
#     log_format = loguru._defaults.LOGURU_FORMAT
#     logger_id = logger.add(
#         str(Path(uploaded_archive.outputdir) / "log.txt"),
#         format=log_format,
#         level="DEBUG",
#         rotation="1 week",
#         backtrace=True,
#         diagnose=True,
#     )
#     logger.debug("Processing finished")
#     logger.remove(logger_id)


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
        output_dir.mkdir(parents=True, exist_ok=False)
        output_images_dir = output_dir / "images"
        output_archive_file = Path(output_archive_file)
        output_metadata_file = Path(output_metadata_file)

        # process data
        data_processing_pipeline.data_processing(
            input_archive_file,
            output_images_dir,
            output_metadata_file,
            num_cores=1,
            wandb_api_key=WANDB_API_KEY,
        )
        dataset_tools.make_zipfile(output_archive_file, output_images_dir)

        logger.info("Finished processing.")
        # out = {
        #     "zip_file": str(output_archive_file),
        #     "csv_file": str(output_metadata_file),
        # }
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        # out = {"zip_file": None, "csv_file": None}
    # return out
