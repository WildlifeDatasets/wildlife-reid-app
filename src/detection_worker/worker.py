import json
import logging
import os
import traceback

import pandas as pd
import numpy as np
import torch
from celery import Celery
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from pprint import pprint, pformat
import cv2

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


@detection_worker.task(bind=True, name="detectionsimplelog")
def detection_simple_log(
        self,
        *args,
        **kwargs,
):
    logger.debug("detectionsimplelog called")
    try:
        logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
        return {"status": "DONE"}
    except Exception as e:
        logger.debug(traceback.format_exc())
        return {"status": "ERROR", "error": str(e)}


@detection_worker.task(bind=True, name="detect_and_crop_mediafile")
def detect_and_crop_mediafile(
        self,
        input_metadata_file_path: str,
        cropped_metadata_file_path: str,
        cropped_mediafile_dir: str= "../images_cropped",
        **kwargs,
):
    logger.debug("detect_and_crop_mediafile called")
    try:
        metadata = pd.read_csv(input_metadata_file_path)
        assert "image_path" in metadata
        output_data = do_detection(input_metadata_file_path, **kwargs)
        # save output to json
        # with open(output_json_file_path, "w") as f:
        #     json.dump(output_data, f)

        logger.debug(f"data from detection ={output_data}")
        # logger.debug(f"output_data={pformat(output_data)}")

        new_image_paths = []
        bbox_per_image = []
        for i, image_path in tqdm(enumerate(metadata["image_path"])):
            bbox = output_data["bboxes"][i]
            image = cv2.imread(image_path)
            logger.debug(f"loaded image shape = {image.shape}")
            new_path = Path(image_path).parent / cropped_mediafile_dir / Path(image_path).name
            new_path = new_path.resolve()
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if bbox is None:
                pass
                bbox_per_image.append(None)
            else:
                bbox = [int(x) for x in bbox]
                # image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                image = image[bbox[1]:, bbox[0]:]
                logger.debug(f"cropped image shape = {image.shape}")
                bbox_per_image.append(bbox)
            cv2.imwrite(str(new_path), image)
            new_image_paths.append(str(new_path))

        metadata["cropped_image_path"] = new_image_paths
        metadata["bbox"] = bbox_per_image
        logger.debug(f"metadata after cropping = {metadata}")
        # save output to csv
        metadata.to_csv(cropped_metadata_file_path, index=False)


        logger.info("Finished processing.")

        # crop
        out = {"status": "DONE", "output_metadata_file_path": cropped_metadata_file_path}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out


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
