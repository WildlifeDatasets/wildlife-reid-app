import skimage.transform
import skimage.io
import random
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__file__)

def make_thumbnail_from_file(directory:Path, thumbnail_path:Path):
    """
    Create thumbnail based on random image file from the directory.
    """

    image = skimage.io.imread(
        random.choice(list(directory.glob("**/*.jpg")))
    )
    width = 800
    scale = float(width) / image.shape[1]
    scale = [scale, scale, 1]
    logger.info(f"{scale=}")
    image_rescaled = skimage.transform.rescale(image, scale=scale, anti_aliasing=True)
    logger.info(f"{image_rescaled.shape=}")
    image_rescaled = (image_rescaled * 255).astype(np.uint8)
    logger.info(f"{image_rescaled.dtype}")
    thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
    skimage.io.imsave(thumbnail_path, image_rescaled)


def get_images_from_csv(csv_file:Path) -> list:
    df = pd.read_csv(csv_file)
    return list(df["image_path"])
