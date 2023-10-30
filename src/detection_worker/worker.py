import json
import logging
import os
import traceback

import pandas as pd
import torch
from celery import Celery
from tqdm import tqdm

from worker_utils import config
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

        # load detector
        model = torch.hub.load(
            "ultralytics/yolov5",  # repo_or_dir
            "custom",  # model
            "resources/md_v5a.0.0.pt",  # args for callable model
            force_reload=True,
            device=device,
        )

        # run detection
        bboxes = []
        scores = []
        labels = []
        class_ids = []
        for image_path in tqdm(metadata["image_path"]):
            results = model(image_path)
            id2label = results.names
            results = results.xywh[0].cpu().numpy()
            if len(results) != 1:
                bbox = results[0, :4].tolist()
                score = float(results[0, 4])
                class_id = int(results[0, 5])
                label = id2label[class_id]
            else:
                bbox = None
                score = None
                class_id = None
                label = None
            bboxes.append(bbox)
            scores.append(score)
            labels.append(label)
            class_ids.append(class_id)

        # create output dictionary
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
