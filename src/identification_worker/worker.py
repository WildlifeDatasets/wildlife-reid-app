import logging
import traceback

import numpy as np
import pandas as pd
from celery import Celery

from utils import config
from utils.database import get_db_connection, init_db_connection
from utils.inference import encode_images, identify
from utils.log import setup_logging
from pathlib import Path

setup_logging()
logger = logging.getLogger("app")

logger.debug(f"{config.RABBITMQ_URL=}")
logger.debug(f"{config.REDIS_URL=}")
logger.debug(f"{config.POSTGRES_URL=}")

identification_worker = Celery(
    "identification_worker", broker=config.RABBITMQ_URL, backend=config.REDIS_URL
)
init_db_connection(db_url=config.POSTGRES_URL)


@identification_worker.task(bind=True, name="init_identification")
def init(
    self,
    input_metadata_file: str,
    organization_id: int,
    **kwargs,
):
    """Process and store Reference Image records in the database."""
    try:
        logger.info(f"Applying init task with args: {input_metadata_file=}, {organization_id=}.")

        # read metadata file
        metadata = pd.read_csv(input_metadata_file)
        assert "image_path" in metadata
        assert "class_id" in metadata
        assert "label" in metadata

        # generate embeddings
        features = encode_images(image_paths=metadata["image_path"])
        metadata["embedding"] = features.tolist()

        # save embeddings and class ids into the database
        logger.info("Storing feature vectors into the database.")
        db_connection = get_db_connection()
        db_connection.reference_image.create_reference_images(organization_id, metadata)

        logger.info("Finished processing.")
        out = {"status": "DONE"}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out


@identification_worker.task(bind=True, name="identify")
def predict(
    self,
    input_metadata_file: str,
    organization_id: int,
    **kwargs,
):
    """Process and compare input samples with Reference Image records from the database."""
    try:
        logger.info(f"Applying init task with args: {input_metadata_file=}, {organization_id=}.")

        # read metadata file
        metadata = pd.read_csv(input_metadata_file)
        assert "image_path" in metadata
        assert Path(metadata["image_path"][0]).exists(), f"File '{metadata['image_path'][0]}' does not exist."
        logger.debug(f"first image = {metadata['image_path'][0]}, {Path(metadata['image_path'][0]).exists()}")

        # generate embeddings
        features = encode_images(image_paths=metadata["image_path"])

        # fetch embeddings of reference samples from the database
        logger.info("Loading reference feature vectors from the database.")
        db_connection = get_db_connection()
        reference_images = db_connection.reference_image.get_reference_images(organization_id)
        reference_features = np.array(reference_images["embedding"].tolist())
        reference_class_ids = reference_images["class_id"]
        id2label = dict(zip(reference_images["class_id"], reference_images["label"]))

        # make predictions by comparing the embeddings using k-NN
        logger.info("Making predictions using .")
        pred_class_ids, scores = identify(features, reference_features, reference_class_ids)
        pred_labels = [id2label[x] for x in pred_class_ids]
        output_data = dict(pred_class_ids=pred_class_ids.tolist(),
                           pred_labels=pred_labels,
                           scores=np.asarray(scores).tolist())

        logger.info("Finished processing.")
        out = {"status": "DONE", "data": output_data}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out
