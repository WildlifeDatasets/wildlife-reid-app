import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
import skimage.transform

logger = logging.getLogger(__file__)

# def make_thumbnail_from_directory(directory: Path, thumbnail_path: Path, width: int = 800):
#     """Create thumbnail based on random image file from the directory."""
#     image_path = random.choice(
#         list(directory.glob("**/*.jpg"))
#         + list(directory.glob("**/*.jpeg"))
#         + list(directory.glob("**/*.png"))
#         + list(directory.glob("**/*.JPG"))
#         + list(directory.glob("**/*.JPEG"))
#         + list(directory.glob("**/*.PNG"))
#     )
#     make_thumbnail_from_file(image_path, thumbnail_path, width)


def make_thumbnail_from_file(image_path: Path, thumbnail_path: Path, width: int = 800) -> bool:
    """Create small thumbnail image from input image.

    Returns:
        True if the processing is ok.

    """
    try:
        image = skimage.io.imread(image_path)
        scale = float(width) / image.shape[1]
        scale = [scale, scale, 1]
        logger.info(f"{scale=}")
        image_rescaled = skimage.transform.rescale(image, scale=scale, anti_aliasing=True)
        logger.info(f"{image_rescaled.shape=}")
        image_rescaled = (image_rescaled * 255).astype(np.uint8)
        logger.info(f"{image_rescaled.dtype}")
        thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
        skimage.io.imsave(thumbnail_path, image_rescaled)
        return True
    except Exception:
        logger.warning(
            f"Cannot create thumbnail from file '{image_path}'. Exception: {traceback.format_exc()}"
        )
        return False


def get_images_from_csv(csv_file: Path) -> list:
    """Get list of images from CSV file."""
    df = pd.read_csv(csv_file)
    return list(df["image_path"])
