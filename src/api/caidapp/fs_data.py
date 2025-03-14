import logging
import traceback
import zipfile
from pathlib import Path
from typing import Union

import cv2
import pandas as pd
import skimage.io
import skimage.transform
from typing import Tuple
import subprocess
import numpy as np
from PIL import Image

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

def resize_images(input_image: np.ndarray, new_height: int = 360) -> np.ndarray:
    """Resize image to match new height and conserve aspect ratio."""
    org_height = input_image.shape[0]
    new_width = int(np.round((input_image.shape[1] * new_height) / org_height))
    resized_image = cv2.resize(input_image, (new_width, new_height))
    return resized_image


def save_gif(images, path: str, fps:float):
    """Save frames as gif using PIL library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".gif":
        format = "GIF"
    elif path.suffix.lower() == ".webp":
        format = "WEBP"
    else:
        logger.warning(f"Unsupported file type: {path.suffix}")
        format = None
    # if it is int like type, convert to uint8
    # if images[0].dtype != np.uint8:
    #     logger.error(f"Unsupported image type: {images[0].dtype}")
    #     raise ValueError(f"Unsupported image type: {images[0].dtype}")

    frame_one = Image.fromarray(images[0])
    frames = [Image.fromarray(image) for image in images[1:]]
    duration = int(1000 / fps)
    frame_one.save(
        str(path),
        format=format,
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
        optimize=True,
    )
    jpg_path = str(path) + ".jpg"
    frame_one = Image.fromarray(images[0])
    frame_one.save(jpg_path)


# def make_thumbnail_from_video_file(video_path: Path, thumbnail_path: Path, width: int = 800, frame_id=0) -> bool:
#     try:
#         frame = get_frame_from_video(video_path, frame_id)
#         scale = float(width) / frame.shape[1]
#         scale = [scale, scale, 1]
#         frame_rescaled = cv2.resize(frame, (0, 0), fx=scale[0], fy=scale[1])
#         thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
#         cv2.imwrite(str(thumbnail_path), frame_rescaled)
#         return True
#     except Exception:
#         logger.debug(traceback.format_exc())
#         logger.warning(
#             f"Cannot create thumbnail from video file '{video_path}'."
#         )
#         return False


def make_gif_from_video_file(video_path: Path, gif_path: Path, width: int = 800, num_frames:int=30, start_frame_id:int=0) -> bool:
    """Create small gif image from input video.

    Returns:
        True if the processing is ok.

    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)

        fps = cap.get(cv2.CAP_PROP_FPS)

        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Cannot read frame from video file '{video_path}'")
            return False

        # get the number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame_id
        frames = []
        for i in range(num_frames):
            frame_idx = start_frame_id + int(i * frame_count / num_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            scale = float(width) / frame.shape[1]
            scale = [scale, scale, 1]
            frame = cv2.resize(frame, (0, 0), fx=scale[0], fy=scale[1])
            # turn to rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frames = np.array(frames)

        gif_path.parent.mkdir(exist_ok=True, parents=True)
        # the whole video is shorten to 3 seconds per 10 frames
        fps = 10.
        save_gif(frames, str(gif_path), fps=fps)
        return True
    except Exception:
        logger.debug("Problem in video creation.")
        logger.warning(traceback.format_exc())
        logger.warning(
            f"aaaa Cannot create thumbnail from video file '{video_path}'."
        )
        return False


def get_frame_from_video(video_path: Path, frame_id: int = 0) -> np.ndarray:
    """Get frame from video file."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        logger.warning(f"Cannot read frame {frame_id} from video file '{video_path}'")
        return None

    # turn to rgb
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def make_thumbnail_from_file(image_path: Path, thumbnail_path: Path, width: int = 800, frame_id=0) -> bool:
    """Create small thumbnail image from input image.

    Returns:
        True if the processing is ok.

    """
    image_path = Path(image_path)
    try:
        # if input is video get first frame
        if image_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"):
            image = get_frame_from_video(image_path, frame_id)
        else:
            image = skimage.io.imread(image_path)
        save_thumbnail(image, thumbnail_path, width)
        return True
    except Exception:
        logger.warning(traceback.format_exc())
        logger.warning(
            f"Cannot create thumbnail from file '{image_path}'."
        )
        return False


def save_thumbnail(image:np.array, thumbnail_path:Path, width:int=800):
    thumbnail_path = Path(thumbnail_path)
    scale = float(width) / image.shape[1]
    scale = [scale, scale, 1]
    image_rescaled = cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1])
    # image_rescaled = skimage.transform.rescale(image, scale=scale, anti_aliasing=True)
    # image_rescaled = (image_rescaled * 255).astype(np.uint8)
    # logger.info(f"{image_rescaled.shape=}, {image_rescaled.dtype=}")
    thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
    if thumbnail_path.suffix.lower() in (".jpg", ".jpeg"):
        quality = 85
        if image_rescaled.shape[2] == 4:
            image_rescaled = cv2.cvtColor(image_rescaled, cv2.COLOR_RGBA2RGB)
    elif thumbnail_path.suffix.lower() == ".webp":
        quality = 85
    else:
        quality = None
    # check if image_rescaled is rgba. If so, convert to rgb
    skimage.io.imsave(thumbnail_path, image_rescaled, quality=quality)


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

def is_string_date(date: str) -> bool:
    """Check if the string is a valid date.

    The date should be in the format: "YYYY-MM-DD" or "YYYYMMDD".
    """
    if len(date) not in (8, 10):
        return False
    if len(date) == 10:
        date = date.replace("-", "")
    try:
        year, month, day = int(date[:4]), int(date[4:6]), int(date[6:])
        if year < 1900 or year > 2100 or month < 1 or month > 12 or day < 1 or day > 31:
            return False
        return True
    except ValueError:
        return False


def get_date_and_locality_from_filename(filename: Union[Path, str]) -> Tuple[str, str]:
    """Extract date and locality from the filename.

    The filename should be in the format: "{date}_{locality}.{ext}" or "{locality}_{date}.{ext}".
    The date should be in the format: "YYYY-MM-DD" or "YYYYMMDD".
    """
    filename = Path(filename).stem
    date = None
    locality = None
    parts = filename.split("_")
    if len(parts) >= 2:
        if is_string_date(parts[0]):
            date = parts[0]
            locality = "_".join(parts[1:])
        elif is_string_date(parts[-1]):
            date = parts[-1]
            locality = "_".join(parts[:-1])

    return date, locality



def convert_to_mp4(input_video_path: Path, output_video_path, force_rewrite=False) -> None:
    """Convert video to MP4 format."""
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)

    if not force_rewrite and output_video_path.exists():
        logger.debug(f"Output file '{output_video_path}' already exists. Skipping conversion.")
        return

    if not input_video_path.exists():
        raise FileNotFoundError(f"The input file '{input_video_path}' does not exist.")

    # ffmpeg command to convert video to MP4 (H.264 + AAC)
    command = [
        "ffmpeg",
        "-i",
        str(input_video_path),  # Input video file
        "-c:v",
        "libx264",  # Set the video codec to H.264
        "-c:a",
        "aac",  # Set the audio codec to AAC
        "-b:a",
        "192k",  # Audio bitrate (you can adjust this)
        "-strict",
        "experimental",  # For using AAC
        str(output_video_path),  # Output video file
    ]

    try:
        # Run the ffmpeg command
        subprocess.run(command, check=True)
        logger.debug(f"Conversion successful! Output saved at '{str(output_video_path)}'")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during conversion: {e}")

