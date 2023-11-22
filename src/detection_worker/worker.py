import json
import logging
import os
import traceback

import pandas as pd
import torch
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

device = torch.device("0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device} ({os.environ.get('CUDA_VISIBLE_DEVICES')})")
device_names = "; ".join(
    [f"{i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
)
logger.info(f"Device names: {device_names}")


@detection_worker.task(bind=True, name="detect")
def detect(
    self,
    input_metadata_file: str,
    output_json_file: str,
    **kwargs,
):
    """Process and store Reference Image records in the database."""
    try:
        logger.info(f"Applying detection task with args: {input_metadata_file=}.")

        # read metadata file
        metadata = pd.read_csv(input_metadata_file)
        assert "image_path" in metadata

        procesed_images = []

        for image_path in tqdm(metadata["image_path"]):
            results = detect_animal(image_path)

            # TODO: Add confidence check
            if results["class"] == 1:
                masked_image = segment_animal(image_path, results["bbox"])

                base_path = os.path.join(image_path.rsplit("/", 1)[0], "masked_images")
                os.makedirs(base_path, exist_ok=True)
                save_path = os.path.join(base_path, image_path.rsplit("/", 1)[0])
                Image.fromarray(masked_image).convert("RGB").save(save_path)

        output_data = dict(bboxes=bboxes, scores=scores, labels=labels, class_ids=class_ids)

        # save output to json
        with open(output_json_file, "w") as f:
            json.dump(output_data, f)

        logger.info("Finished processing.")
        out = {"status": "DONE", "output_json_file": output_json_file}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out
