import logging
import os
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .inference_detection import detect_animals_in_one_image

logger = logging.getLogger("app")


def load_video(path):
    """Load all frames from video."""
    frames = []

    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    video = np.array(frames)

    return video


def select_images(images, predictions, selection_method):
    """Selects images and predictions using provided selection method."""
    idxs = selection_method(predictions)

    images = images[idxs]
    predictions = np.array(predictions)
    predictions = predictions[idxs]

    return (
        images,
        predictions,
    )


def ratio_selection(predictions, threshold):
    """Select images with the largest ratio between height and width."""
    idxs = []

    ratios = []
    for pred in predictions:
        if pred is None:
            ratios.append(0)
            continue

        x0, y0, x1, y1 = pred["bbox"]
        w, h = x1 - x0, y1 - y0
        ratio = w / h
        ratios.append(ratio)

    if isinstance(threshold, int):
        sort_idxs = np.argsort(ratios)[::-1]
        idxs = [i for i in sort_idxs if ratios[i] > 0]
        idxs = idxs[:threshold]
    elif isinstance(threshold, float):
        idxs = [i for i, a in enumerate(ratios) if a > threshold]

    return idxs


def area_selection(predictions, threshold):
    """Select predictions with the largest area."""
    idxs = []

    areas = []
    for pred in predictions:
        if pred is None:
            areas.append(0)
            continue

        x0, y0, x1, y1 = pred["bbox"]
        w, h = x1 - x0, y1 - y0
        iw, ih = pred["size"]
        relative_area = (w * h) / (iw * ih)
        areas.append(relative_area)

    if isinstance(threshold, int):
        sort_idxs = np.argsort(areas)[::-1]
        idxs = [i for i in sort_idxs if areas[i] > 0]
        idxs = idxs[:threshold]
    elif isinstance(threshold, float):
        idxs = [i for i, a in enumerate(areas) if a > threshold]

    return idxs


def get_gif_frames(images, center_frame_idx, num_frames):
    """Select frames from video."""
    if num_frames > len(images):
        return images

    start_frame = int(center_frame_idx - np.ceil(num_frames / 2))
    end_frame = int(center_frame_idx + np.floor(num_frames / 2))

    if start_frame < 0:
        end_frame += np.abs(start_frame)
        start_frame = 0

    if end_frame > len(images) - 1:
        start_frame += (len(images) - 1) - end_frame
        end_frame = len(images) - 1

    return images[start_frame:end_frame]


def resize_images(input_image: np.ndarray, new_height: int = 360) -> np.ndarray:
    """Resize image to match new height and conserve aspect ratio."""
    org_height = input_image.shape[0]
    new_width = int(np.round((input_image.shape[1] * new_height) / org_height))
    resized_image = cv2.resize(input_image, (new_width, new_height))
    return resized_image


def save_gif(images, path: str):
    """Save frames as gif using PIL library."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame_one = Image.fromarray(images[0])
    frames = [Image.fromarray(image) for image in images[1:]]
    duration = 1 / 24 * len(images)
    frame_one.save(
        path,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
        optimize=True,
    )
    jpg_path = path + ".jpg"
    frame_one = Image.fromarray(images[0])
    frame_one.save(jpg_path)


def make_gif(
    images: np.ndarray, path: str, center_frame_idx: int, num_frames: int = 48, height: int = 360
):
    """Create gif from few frames of the video around center_frame_idx."""
    gif_frames = get_gif_frames(images, center_frame_idx, num_frames)
    if height > 0:
        gif_frames = np.array([resize_images(image, height) for image in gif_frames])
    save_gif(gif_frames, path)


def create_image_from_video(
    metadata: pd.DataFrame,
    selection_methods: Tuple[str] = ("area", "ratio"),
    selection_thresholds: Tuple[float] = (24, 1),
    gif_height: int = 240,
) -> pd.DataFrame:
    """
    Create image from video.

    Use full_image_path to get the input file, media_type indicates the type of media.
    Keep the video path in absolute_media_path and change the full_image_path and image_path
    new image.
    """
    skip_counter = 0
    process_counter = 0
    no_detection_counter = 0
    for row_idx, row in tqdm(metadata.iterrows(), desc="Video to image"):
        full_path = row["full_image_path"]

        video_name = os.path.basename(full_path)
        video_name = ".".join(video_name.split(".")[:-1])

        # new_full_path = os.path.join(os.path.dirname(full_path), f"{video_name}.png")
        # gif_path = os.path.join(os.path.dirname(full_path), f"{video_name}.gif")
        new_full_path = str(Path(full_path).with_suffix(".jpg"))
        gif_path = str(Path(full_path).parent.parent / "thumbnails" / f"{video_name}.gif")

        if row["media_type"] != "video":
            # logger.debug(f"{os.path.basename(full_path)} is image - skipping")
            skip_counter += 1
            continue
        process_counter += 1

        images = load_video(full_path)
        logger.debug(f"video frames: {images.shape}")
        if len(images) == 0:
            logger.debug(f"Problem loading video: {os.path.basename(full_path)} - skipping")
            row["read_error"] = "Problem loading video - skipping."
            # row["image_path"] = os.path.basename(new_full_path)
            # row["full_image_path"] = new_full_path
            row["suffix"] = f".{new_full_path.split('.')[-1]}"
            metadata.loc[row_idx] = row
            continue
        all_images = np.array([resize_images(image, gif_height) for image in images.copy()])

        logger.info("Running detection inference on video.")
        # detect
        predictions = []
        for frame_id, image in tqdm(
            enumerate(images), desc="Detection in video", total=len(images)
        ):
            prediction = detect_animals_in_one_image(image)

            if prediction is not None:
                # use only prediction with max confidence
                # TODO: how to handle multiple detections in one image?
                confidence = [p["confidence"] for p in prediction]
                idx = np.argmax(confidence)
                prediction = prediction[idx]

                prediction["frame"] = frame_id
            predictions.append(prediction)

        # check if any detection
        if len([1 for p in predictions if p is not None]) == 0:
            logger.debug(f"no detection in video: {os.path.basename(full_path)} - skipping")
            no_detection_counter += 1
            make_gif(all_images, gif_path, 0, height=gif_height)

            image = images[0]
            cv2.imwrite(new_full_path, image[..., ::-1])
            row["image_path"] = os.path.basename(new_full_path)
            row["full_image_path"] = new_full_path
            row["suffix"] = f".{new_full_path.split('.')[-1]}"
            metadata.loc[row_idx] = row
            continue

        # select
        for selection_method_name, selection_threshold in zip(
            selection_methods, selection_thresholds
        ):
            if selection_method_name == "area":
                selection_method = partial(area_selection, threshold=selection_threshold)
            elif selection_method_name == "ratio":
                selection_method = partial(ratio_selection, threshold=selection_threshold)

            images, predictions = select_images(images, predictions, selection_method)
        logger.debug(f"selected frames: {images.shape}")
        image = images[0]
        prediction = predictions[0]

        # save image
        make_gif(all_images, gif_path, prediction["frame"], height=gif_height)
        logger.debug(f"selected 1 image forme video, saving to: {new_full_path}")
        cv2.imwrite(new_full_path, image[..., ::-1])

        # update row
        row["image_path"] = os.path.basename(new_full_path)
        row["full_image_path"] = new_full_path
        row["suffix"] = f".{new_full_path.split('.')[-1]}"
        metadata.loc[row_idx] = row
    logger.debug(f"Processed {process_counter} videos, skipped {no_detection_counter} "
                 f"videos with no detection and  {skip_counter} images.")

    return metadata
