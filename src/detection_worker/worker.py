import json
import logging
import os
import traceback

import pandas as pd
import torch
from celery import Celery
from tqdm import tqdm
from pathlib import Path
from pprint import pprint, pformat
import cv2

from worker_utils import config
from worker_utils.log import setup_logging

setup_logging()
logger = logging.getLogger("app")

logger.debug(f"{config.RABBITMQ_URL=}")
logger.debug(f"{config.REDIS_URL=}")

detection_worker = Celery("detection_worker", broker=config.RABBITMQ_URL, backend=config.REDIS_URL)

device = torch.device("0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device} ({os.environ.get('CUDA_VISIBLE_DEVICES')})")
device_names = "; ".join(
    [f"{i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
)
logger.info(f"Device names: {device_names}")


def download_file(url: str, output_file: str):
    """Download file from url."""
    import requests


    # r = requests.get(url, allow_redirects=True)
    # with open(output_file, "wb") as f:
    #     f.write(r.content)

    # download file from url with tqdm progressbar
    # https://stackoverflow.com/a/37573701/4419811
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(output_file, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        logger.error("ERROR, something went wrong")



def download_file_if_does_not_exists(url: str, output_file: str):
    """Download file from url."""
    if not os.path.exists(output_file):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, output_file)

@detection_worker.task(bind=True, name="detectionsimplelog")
def detection_simple_log(
        self,
        *args,
        **kwargs,
):
    logger.debug("detectionsimplelog called")
    try:
        logger.info(f"Applying simple log task with args: {args=}, {kwargs=}.")
        return {"status": "DONE"}
    except Exception as e:
        logger.debug(traceback.format_exc())
        return {"status": "ERROR", "error": str(e)}

@detection_worker.task(bind=True, name="detect")
def detect(
        self,
        input_metadata_file_path: str,
        output_json_file_path: str,
        **kwargs,
):

    try:
        output_data = do_detection(input_metadata_file_path)
        # save output to json
        with open(output_json_file_path, "w") as f:
            json.dump(output_data, f)

        logger.info("Finished processing.")
        out = {"status": "DONE", "output_json_file": output_json_file_path}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out


@detection_worker.task(bind=True, name="detect_and_crop_mediafile")
def detect_and_crop_mediafile(
        self,
        input_metadata_file_path: str,
        cropped_metadata_file_path: str,
        cropped_mediafile_dir: str= "../images_cropped",
        **kwargs,
):
    logger.debug("detect_and_crop_mediafile called")
    try:
        metadata = pd.read_csv(input_metadata_file_path)
        assert "image_path" in metadata
        output_data = do_detection(input_metadata_file_path, **kwargs)
        # save output to json
        # with open(output_json_file_path, "w") as f:
        #     json.dump(output_data, f)

        logger.debug(f"data from detection ={output_data}")
        # logger.debug(f"output_data={pformat(output_data)}")

        new_image_paths = []
        bbox_per_image = []
        for i, image_path in tqdm(enumerate(metadata["image_path"])):
            bbox = output_data["bboxes"][i]
            image = cv2.imread(image_path)
            logger.debug(f"loaded image shape = {image.shape}")
            new_path = Path(image_path).parent / cropped_mediafile_dir / Path(image_path).name
            new_path = new_path.resolve()
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if bbox is None:
                pass
                bbox_per_image.append(None)
            else:
                bbox = [int(x) for x in bbox]
                # image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                image = image[bbox[1]:, bbox[0]:]
                logger.debug(f"cropped image shape = {image.shape}")
                bbox_per_image.append(bbox)
            cv2.imwrite(str(new_path), image)
            new_image_paths.append(str(new_path))

        metadata["cropped_image_path"] = new_image_paths
        metadata["bbox"] = bbox_per_image
        logger.debug(f"metadata after cropping = {metadata}")
        # save output to csv
        metadata.to_csv(cropped_metadata_file_path, index=False)


        logger.info("Finished processing.")

        # crop
        out = {"status": "DONE", "output_metadata_file_path": cropped_metadata_file_path}
    except Exception:
        error = traceback.format_exc()
        logger.critical(f"Returning unexpected error output: '{error}'.")
        out = {"status": "ERROR", "error": error}
    return out



# @detection_worker.task(bind=True, name="detect")
def do_detection(
    input_metadata_file_path: str,
) -> dict:
    """Process and store Reference Image records in the database."""
    logger.info(f"Applying detection task with args: {input_metadata_file_path=}.")
    from celery import current_app
    tasks = current_app.tasks.keys()
    logger.debug(f"tasks={tasks}")

    # read metadata file
    metadata = pd.read_csv(input_metadata_file_path)
    assert "image_path" in metadata
    model_url = r"https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt"
    model_file = Path("resources/md_v5a.0.0.pt")
    download_file_if_does_not_exists(model_url, model_file)

    # load detector
    model = torch.hub.load(
        "ultralytics/yolov5",  # repo_or_dir
        "custom",  # model
        path=model_file,
        # "resources/md_v5a.0.0.pt",  # args for callable model
        force_reload=True,
        device=device,
    )

    # run detection
    bboxes = []
    scores = []
    labels = []
    class_ids = []
    for image_path in tqdm(metadata["image_path"]):
        results = model(image_path)
        id2label = results.names
        results = results.xywh[0].cpu().numpy()
        logger.debug(f"image_path={image_path}")
        logger.debug(f"detection results (bboxes) = {results}")
        if len(results) == 1:
            bbox = results[0, :4].tolist()
            score = float(results[0, 4])
            class_id = int(results[0, 5])
            label = id2label[class_id]
        else:
            bbox = None
            score = None
            class_id = None
            label = None
        bboxes.append(bbox)
        scores.append(score)
        labels.append(label)
        class_ids.append(class_id)

    # create output dictionary
    output_data = dict(
        image_path=metadata["image_path"], bboxes=bboxes,
        scores=scores, labels=labels, class_ids=class_ids)
    logger.debug(f"output_data={pformat(output_data)}")
    return output_data

