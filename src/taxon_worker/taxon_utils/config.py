import os

# general
RABBITMQ_URL = os.environ["RABBITMQ_URL"]
REDIS_URL = os.environ["REDIS_URL"]

# W&B related variables
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_ARTIFACT_PATH = os.environ["WANDB_ARTIFACT_PATH"]
WANDB_ARTIFACT_PATH_CROPPED = os.environ["WANDB_ARTIFACT_PATH_CROPPED"]

SHARED_DATA_PATH = os.environ["SHARED_DATA_PATH"]
RESOURCES_DIR = os.path.join(SHARED_DATA_PATH, "resources")
