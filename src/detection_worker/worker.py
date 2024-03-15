import logging
import traceback
from pathlib import Path

import pandas as pd
from celery import Celery
from PIL import Image
from tqdm import tqdm

from worker_utils import config
from worker_utils.inference import detect_animal, segment_animal
from worker_utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")

logger.debug(f"{config.RABBITMQ_URL=}")
logger.debug(f"{config.REDIS_URL=}")

detection_worker = Celery("detection_worker", broker=config.RABBITMQ_URL, backend=config.REDIS_URL)

# device = torch.device("0" if torch.cuda.is_available() else "cpu")
# logger.info(f"Using device: {device} ({os.environ.get('CUDA_VISIBLE_DEVICES')})")
# device_names = "; ".join(
#    [f"{i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
# )
# logger.info(f"Device names: {device_names}")


# @detection_worker.task(bind=True, name="preprocessing")
# def preprocessing(
#         self,
#         input_archive_file: str,
#         output_dir: str,
#         output_archive_file: str, # TODO remove this parameter
#         output_metadata_file: str,
#         contains_identities: bool = False,
#         **kwargs,
# ):
#     """Main method called by Celery broker.
#
#     If the output_metadata_file does not exist, the metadata is
#     created based on the content of input_archive_file and saved to output_metadata_file.
#     If the output_metadata_file exists, it is directly used as input for the inference.
#     """
#     num_cores=1
#     try:
#         logger.info(
#             "Applying species identification task with args: "
#             + f"{input_archive_file=}, {output_dir=}, {contains_identities=}."
#         )
#
#         # prepare input and output file names
#         input_archive_file = Path(input_archive_file)
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
#         output_images_dir = output_dir / "images"
#         output_archive_file = Path(output_archive_file)
#         output_metadata_file = Path(output_metadata_file)
#
#         # process data
#         if output_images_dir.exists():
#             # TODO turn of the following line
#             metadata = pd.read_csv(output_metadata_file, index_col=0)
#         else:
#
#
#
#             # metadata = data_processing_pipeline.data_processing(
#             #     input_archive_file,
#             #     output_images_dir,
#             #     output_metadata_file,
#             #     num_cores=1,
#             #     contains_identities=contains_identities,
#             # )
#
#             metadata = data_processing_pipeline.data_preprocessing(
#                 input_archive_file,
#                 output_images_dir,
#                 num_cores=num_cores,
#                 contains_identities=contains_identities,
#             )
#             logger.debug(f"len(metadata)={len(metadata)}")
#             metadata = metadata[metadata["media_type"] == "image"].reset_index(drop=True)
#             logger.debug(f"len(metadata)={len(metadata)}")
#             metadata = metadata[metadata["read_error"] == ""].reset_index(drop=True)
#             logger.debug(f"len(metadata)={len(metadata)}")
#         # logger.debug("Preparing output archive.")
#         # dataset_tools.make_zipfile_with_categories(
#         #     output_archive_file, output_images_dir, metadata)
#         # logger.debug(f"{contains_identities=}")
#         # logger.debug(f"{output_archive_file=}")
#
#         # dataset_tools.make_zipfile(output_archive_file, output_images_dir)
#
#         logger.info("Finished processing.")
#         out = {"status": "DONE"}
#     except Exception:
#         error = traceback.format_exc()
#         logger.critical(f"Returning unexpected error output: '{error}'.")
#         out = {"status": "ERROR", "error": error}
#     return out
#


@detection_worker.task(bind=True, name="detect")
def detect(
    self,
    input_metadata_path: str,
    output_metadata_path: str,
    **kwargs,
):
    """Process and store Reference Image records in the database."""
    try:
        logger.info(f"Applying detection task with args: {input_metadata_path=}.")
        from celery import current_app

        tasks = current_app.tasks.keys()
        logger.debug(f"tasks={tasks}")

        # read metadata file
        metadata = pd.read_csv(input_metadata_path)
        detect_and_segment_animal(metadata)
        metadata.to_csv(output_metadata_path, index=None)

        logger.info("Finished processing. Metadata saved.")

        out = {"status": "DONE", "output_metadata_file": output_metadata_path}

    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}

    return out


def detect_and_segment_animal(metadata):
    """Do the detection and segmentation on images in metadata."""
    assert "image_path" in metadata
    masked_images = []
    for image_path in tqdm(metadata["image_path"]):
        try:
            results = detect_animal(image_path)
            # logger.debug(f"results={results}")

            save_path = None
            if results["class"] == "animal":
                cropper_animal = segment_animal(image_path, results["bbox"])

                base_path = Path(image_path).parent.parent / "masked_images"
                save_path = base_path / Path(image_path).name
                base_path.mkdir(exist_ok=True, parents=True)
                Image.fromarray(cropper_animal).convert("RGB").save(save_path)
            masked_images.append(str(save_path))
        except Exception:
            logger.warning(
                f"Cannot process image '{image_path}'. Exception: {traceback.format_exc()}"
            )
            masked_images.append("")
    metadata["masked_image_path"] = masked_images
