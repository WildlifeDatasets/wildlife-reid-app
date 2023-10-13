import os

# general
RABBITMQ_URL = os.environ["RABBITMQ_URL"]
REDIS_URL = os.environ["REDIS_URL"]

# W&B related variables
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
WANDB_ARTIFACT_PATH = os.environ["WANDB_ARTIFACT_PATH"]

SHARED_DATA_PATH = os.environ["SHARED_DATA_PATH"]
RESOURCES_DIR = os.path.join(SHARED_DATA_PATH, "resources")

# path to the private database
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASS = os.environ["POSTGRES_PASS"]
POSTGRES_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASS}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
