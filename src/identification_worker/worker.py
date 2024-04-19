import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from celery import Celery, shared_task

from utils import config
from utils.database import get_db_connection, init_db_connection
from utils.inference import encode_images, identify
from utils.log import setup_logging
from utils.sequence_identification import extend_df_with_datetime, extend_df_with_sequence_id

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

        # remove all unused columns
        metadata = metadata[["image_path", "class_id", "label", "detection_results"]]

        # generate embeddings
        features = encode_images(metadata)
        metadata["embedding"] = features.tolist()

        # save embeddings and class ids into the database
        logger.info("Storing feature vectors into the database.")
        db_connection = get_db_connection()
        logger.debug(metadata.tail(3).to_string())
        db_connection.reference_image.create_reference_images(organization_id, metadata)

        logger.info("Finished processing.")
        out = {
            "status": "DONE",
            "message": f"Identification initiated with {len(metadata['image_path'])} images.",
        }
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out


@identification_worker.task(bind=True, name="iworker_simple_log")
def iworker_simple_log(self, *args, **kwargs):
    """Simple log task."""
    logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
    return {"status": "DONE"}


@shared_task(bind=True, name="shared_simple_log")
def shared_simple_log(self, *args, **kwargs):
    """Simple log task."""
    logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
    return {"status": "DONE"}


@identification_worker.task(bind=True, name="identify")
def predict(
    self,
    input_metadata_file_path: str,
    organization_id: int,
    output_json_file_path: str,
    top_k: int = 1,
    sequence_time: str = "480s",
    **kwargs,
):
    """Process and compare input samples with Reference Image records from the database."""
    try:
        logger.info(
            f"Applying init task with args: {input_metadata_file_path=}, {organization_id=}."
        )

        # read metadata file
        metadata = pd.read_csv(input_metadata_file_path)
        if len(metadata) == 0:
            logger.info("Input data is empty. Finishing the job.")
            out = {"status": "ERROR", "error": "Input data is empty."}
        else:
            assert "image_path" in metadata
            assert "mediafile_id" in metadata
            first_image_path = metadata["image_path"].iloc[0]
            assert Path(first_image_path).exists(), f"File '{first_image_path}' does not exist."
            logger.debug(f"first image = {first_image_path}, {Path(first_image_path).exists()}")

            # fetch embeddings of reference samples from the database
            logger.info("Loading reference feature vectors from the database.")
            db_connection = get_db_connection()
            reference_images = db_connection.reference_image.get_reference_images(organization_id)
            if len(reference_images) == 0:
                logger.info(
                    f"Identification worker was not initialized for {organization_id=}. "
                    "Finishing the job."
                )
                out = {
                    "status": "ERROR",
                    "error": "Identification worker was not initialized.",
                }
            else:

                # estimate sequence id
                if "location_name" in metadata:
                    metadata["location"] = metadata["location_name"]
                    metadata = extend_df_with_datetime(metadata)
                    metadata = extend_df_with_sequence_id(metadata, sequence_time)
                    metadata["sequence_number"] = np.where(
                        metadata["location"].isna(), -1, metadata["sequence_number"]
                    )
                logger.debug(f"{list(metadata.image_path)}")
                query_image_path = list(metadata.image_path)
                query_masked_path = [p.replace("/images/", "/masked_images/") for p in query_image_path]

                # generate embeddings
                features = encode_images(metadata)


                # get reference embeddings
                reference_features = np.array(reference_images["embedding"].tolist())
                id2label = dict(zip(reference_images["class_id"], reference_images["label"]))

                # make predictions by comparing the embeddings using k-NN
                logger.info("Making predictions using .")
                identification_output = identify(
                    features,
                    reference_features,
                    reference_image_paths=reference_images["image_path"],
                    reference_class_ids=reference_images["class_id"],
                    metadata=metadata,
                    top_k=top_k,
                )
                pred_labels = [[id2label[x] for x in row] for row in identification_output["pred_class_ids"]]
                identification_output["mediafile_ids"] = metadata["mediafile_id"].tolist()
                identification_output["pred_labels"] = pred_labels
                identification_output["query_image_path"] = query_image_path
                identification_output["query_masked_path"] = query_masked_path

                # save output to json
                with open(output_json_file_path, "w") as f:
                    json.dump(identification_output, f)

                logger.info("Finished processing.")
                out = {"status": "DONE", "output_json_file": output_json_file_path}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out
