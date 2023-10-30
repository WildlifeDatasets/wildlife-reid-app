import os

# general
RABBITMQ_URL = os.environ["RABBITMQ_URL"]
REDIS_URL = os.environ["REDIS_URL"]

SHARED_DATA_PATH = os.environ["SHARED_DATA_PATH"]
RESOURCES_DIR = os.path.join(SHARED_DATA_PATH, "resources")
