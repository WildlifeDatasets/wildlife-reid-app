import logging
import traceback
from pathlib import Path

import pandas as pd
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
    contains_identities: bool = False,
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

        # prepare input and output file names
        input_archive_file = Path(input_archive_file)
        # assert input_archive_file.suffix.lower() in (".tar", ".tar.gz", ".zip")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_images_dir = output_dir / "images"
        output_archive_file = Path(output_archive_file)
        output_metadata_file = Path(output_metadata_file)

        # process data
        if output_images_dir.exists():
            metadata = pd.read_csv(output_metadata_file, index_col=0)
        else:
            metadata = data_processing_pipeline.data_processing(
                input_archive_file,
                output_images_dir,
                output_metadata_file,
                num_cores=1,
                contains_identities=contains_identities,
            )
        logger.debug("Preparing output archive.")
        dataset_tools.make_zipfile_with_categories(output_archive_file, output_images_dir, metadata)
        # dataset_tools.make_zipfile(output_archive_file, output_images_dir)

        logger.info("Finished processing.")
        out = {"status": "DONE"}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out
