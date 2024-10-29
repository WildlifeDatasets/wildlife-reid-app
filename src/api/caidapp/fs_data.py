import logging
import traceback
import zipfile
from pathlib import Path
from typing import Union

import cv2
import pandas as pd
import skimage.io
import skimage.transform

logger = logging.getLogger(__file__)

import unicodedata


def remove_diacritics(input_str):
    """Removes diacritics (accents) from the given Unicode string.

    The function decomposes the string into its combining characters, removes
    any diacritic characters, and then recomposes the string.
    """
    # Normalize the input string to NFD (Normalization Form Decomposed)
    normalized = unicodedata.normalize("NFD", input_str)

    # Filter out combining characters (those in category 'Mn')
    filtered = "".join(c for c in normalized if unicodedata.category(c) != "Mn")

    # Return the normalized string
    return unicodedata.normalize("NFC", filtered)


def make_thumbnail_from_file(image_path: Path, thumbnail_path: Path, width: int = 800) -> bool:
    """Create small thumbnail image from input image.

    Returns:
        True if the processing is ok.

    """
    try:
        image = skimage.io.imread(image_path)
        scale = float(width) / image.shape[1]
        scale = [scale, scale, 1]
        # TODO use opencv to resize image
        image_rescaled = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1])
        # image_rescaled = skimage.transform.rescale(image, scale=scale, anti_aliasing=True)
        # image_rescaled = (image_rescaled * 255).astype(np.uint8)
        logger.info(f"{image_rescaled.shape=}, {image_rescaled.dtype=}")
        thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
        if thumbnail_path.suffix.lower() in (".jpg", ".jpeg"):
            quality = 85
        else:
            quality = None
        skimage.io.imsave(thumbnail_path, image_rescaled, quality=quality)
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


def count_files_in_archive(zip_path: Union[str, Path]) -> dict:
    """Count files in the zip archive."""
    # Open the zip file in read mode
    image_count = 0
    video_coung = 0
    with zipfile.ZipFile(zip_path, "r") as zip:
        # List all files and directories in the zip
        all_files = zip.namelist()
        # Count only the files, excluding directories
        file_count = sum(1 for f in all_files if not f.endswith("/"))
        for f in all_files:
            if (
                f.lower().endswith(".jpg")
                or f.lower().endswith(".jpeg")
                or f.lower().endswith(".tif")
                or f.lower().endswith(".tiff")
                or f.lower().endswith(".bmp")
                or f.lower().endswith(".png")
            ):
                image_count += 1
            elif (
                f.lower().endswith(".mp4")
                or f.lower().endswith(".avi")
                or f.lower().endswith(".mov")
                or f.lower().endswith(".mkv")
                or f.lower().endswith(".webm")
                or f.lower().endswith(".flv")
                or f.lower().endswith(".wmv")
                or f.lower().endswith(".m4v")
            ):
                video_coung += 1
        return {"file_count": file_count, "image_count": image_count, "video_count": video_coung}
