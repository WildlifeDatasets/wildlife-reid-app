import json
import logging
import os
import traceback
from pathlib import Path
from pprint import pformat, pprint

import cv2
import numpy as np
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
        assert "image_path" in metadata

        masked_images = []

        for image_path in tqdm(metadata["image_path"]):
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

        metadata["masked_image_path"] = masked_images
        metadata.to_csv(output_metadata_path, index=None)

        logger.info("Finished processing. Metadata saved.")

        out = {"status": "DONE", "output_metadata_file": output_metadata_path}

    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}

    return out
