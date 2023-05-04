import logging
import os
from pathlib import Path

from celery import Celery
from utils import data_processing_pipeline, dataset_tools
from utils.log import setup_logging

RABBITMQ_URL = os.environ["RABBITMQ_URL"]
REDIS_URL = os.environ["REDIS_URL"]

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
def predict(self, input_archive_file: str, output_dir: str, **kwargs):
    """Main method called by Celery broker."""
    logger.info(f"Applying inference task with agrs: {input_archive_file=}, {output_dir=}.")

    # prepare input and output file names
    input_archive_file = Path(input_archive_file)
    assert input_archive_file.suffix.lower() in (".tar", ".tar.gz", ".zip")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    output_images_dir = output_dir / "images"
    output_metadata_file = output_dir / "metadata.csv"
    output_archive_file = output_dir / "images.zip"

    # process data
    data_processing_pipeline.data_processing(
        input_archive_file, output_images_dir, output_metadata_file, num_cores=1
    )
    dataset_tools.make_zipfile(output_archive_file, output_images_dir)
    logger.info("Finished processing.")

    # for video_pth in outputdir.glob("*.avi"):
    #     input_video_file = video_pth
    #     output_video_file = video_pth.with_suffix(".mp4")
    #     logger.debug(f"input_video_file={input_video_file}")
    #     logger.debug(f"outout_video_file={output_video_file}")
    #     if output_video_file.exists():
    #         output_video_file.unlink()
    #     _convert_avi_to_mp4(str(input_video_file), str(output_video_file))
    # add_generated_images(uploaded_archive)
    # make_zip(uploaded_archive)

    return {
        "zip_file": output_archive_file,
        "csv_file": output_metadata_file,
    }
