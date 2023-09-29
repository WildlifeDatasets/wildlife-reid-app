import logging
import traceback
from pathlib import Path

from celery import Celery
from utils import data_processing_pipeline, dataset_tools
from utils.config import RABBITMQ_URL, REDIS_URL
from utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")
logger.debug(f"{RABBITMQ_URL=}")
logger.debug(f"{REDIS_URL=}")
identification_worker = Celery("identification_worker", broker=RABBITMQ_URL, backend=REDIS_URL)


@identification_worker.task(bind=True, name="init_identification")
def init(
    self,
    input_metadata_file: str,
    **kwargs,
):
    # generate embeddings
    # save embeddings and class ids into the database
    pass


@identification_worker.task(bind=True, name="identify")
def predict(
    self,
    input_metadata_file: str,
    **kwargs,
):
    # generate embeddings
    # fetch embeddings of reference samples from the database
    # make predictions by comparing the embeddings using k-NN
    pass
