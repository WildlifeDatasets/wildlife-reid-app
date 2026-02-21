import logging
import re
import traceback
import typing
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import exiftool
import numpy as np
import pandas as pd
import pytesseract
import scipy.stats
import skimage
import skimage.color
from joblib import Parallel, delayed
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logger = logging.getLogger(__name__)


# EXIFTOOL_EXECUTABLE = "/webapps/piglegsurgery/Image-ExifTool-13.00/exiftool"
EXIFTOOL_EXECUTABLE = None

DATETIME_BLACKLIST = [
    # "0000-00-00 00:00:00",
    "2015-05-21 15:29:12"
]


class DatasetEventIdManager:
    """
    A tool for getting unique event ID based on time differences between two following images.

    The images close enough are grouped into one event.
    ...

    Attributes
    ----------
    event_id : int
        Unique ID of the fallowing event

    Methods
    -------
    create_event_id(delta_datetime: timedelta, time_limit: [str, datetime])
        delta_datetime is time difference to previous image
        time_limit is the minimum time between two images required to create new event ID.

    """

    def __init__(self, time_limit: typing.Union[str, datetime] = "120s"):
        """Prepare parameters for sequence ID extraction.

        Parameters
        ----------
        time_limit: [str, datetime]
            The minimum time between two images required to create new event ID.
        """
        self._first_record: bool = True
        self.event_id: int = int(0)
        self.time_limit = time_limit

    def create_event_id(self, delta_datetime: timedelta):
        """Prepare new EventID if there is big enough time difference.

        Parameters
        ----------
        delta_datetime: timedelta
            Time difference to previous image


        Returns
        -------
            ID of the event based on time_delta

        """
        if self._first_record:
            self._first_record = False
            return 0
        if pd.isnull(delta_datetime):
            return None
        if delta_datetime < pd.Timedelta(0) or delta_datetime > pd.Timedelta(self.time_limit):
            self.event_id += int(1)
        return self.event_id


def replace_colon_in_exif_datetime(exif_datetime: str) -> str:
    """Turn strange EXIF datetime format (containing ':' in date) into standard datetime.

    Parameters
    ----------
    exif_datetime : str
        Input string with datetime in EXIF format i.e. "2022:10:05 10:11:56"


    Returns
    -------
    string :

    """
    replaced = exif_datetime
    if isinstance(exif_datetime, str):
        exif_ex = re.findall(
            r"([0-9]{4}):([0-9]{2}):([0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2}.*)",
            exif_datetime,
        )
        if len(exif_ex) == 1:
            ex = exif_ex[0]
            replaced = f"{ex[0]}-{ex[1]}-{ex[2]} {ex[3]}"

    return replaced


def get_datetime_using_exif_or_ocr(
    filename: typing.Union[Path, str], exiftool_metadata: dict
) -> typing.Tuple[str, str, str, dict]:
    """Extract datetime from EXIF in file and check if image is ok.

    Parameters
    ----------
    exiftool_metadata : dict extracted from file using exiftool
    filename : name of the file

    Returns
    -------
    str1:
        String with datetime in forma YYYY-MM-DD HH:MM:SS or zero length string if no EXIF is
        available.

    str2:
        Error type or zero length string if file is ok.

        The function also checks if image or video is ok for read.
    """
    filename = Path(filename)
    dt_source = ""
    in_worst_case_dt = None
    in_worst_case_dt_source = None
    opened_with_fail = False
    read_error = ""
    if not filename.exists():
        return "", "File does not exist", ""

    opened_sucessfully, opening_error, media_file_type = check_file_by_opening(filename)

    if opening_error:
        return "", str(opening_error), ""

    dt_str = ""
    try:
        checked_keys = [
            "QuickTime:MediaCreateDate",
            "QuickTime:CreateDate",
            "EXIF:CreateDate",
            "EXIF:ModifyDate",
            "EXIF:DateTimeOriginal",
            "EXIF:DateTimeCreated",
            # "File:FileModifyDate",
            # "File:FileCreateDate",
        ]

        d = exiftool_metadata
        dt_str = ""
        is_ok = False
        dt_source = ""
        for k in checked_keys:
            if k in d:
                dt_str = d[k]
                is_ok = True
                dt_source = k
                break
        # if no key was found log the metadata
        if not is_ok:
            logger.debug(f"date not found, exif: {str(d)=}")

        # dt_str, is_ok, dt_source = get_datetime_exiftool(filename)
        dt_str = replace_colon_in_exif_datetime(dt_str)
        if dt_source.startswith("QuickTime"):
            in_worst_case_dt = dt_str
            in_worst_case_dt_source = dt_source
            # df_str = ""
            dt_source = ""
        if dt_str in DATETIME_BLACKLIST:
            logger.debug("blacklisted datetime")
            dt_str = ""
            dt_source = ""
        read_error = ""
    except Exception as e:
        dt_str = ""
        read_error = str(e)
        logger.warning(f"Error while reading EXIF from {filename}")
        logger.exception(traceback.format_exc())
        opened_with_fail = True

    if read_error == "":
        if dt_str == "":
            try:
                dt_str, dt_source = get_datetime_from_ocr(filename)
                read_error = ""
                opened_sucessfully = True
            except Exception as e:
                dt_str = ""
                read_error = "OCR failed"

                logger.warning(f"Error while reading OCR from {filename}")
                logger.debug(e)
                logger.debug(traceback.format_exc())
                opened_with_fail = True

        if (dt_str == "") and (in_worst_case_dt is not None):
            dt_str = in_worst_case_dt
            dt_source = in_worst_case_dt_source

        if dt_str == "":
            dtm = min(filename.stat().st_mtime, filename.stat().st_ctime, filename.stat().st_atime)
            dt_str = datetime.fromtimestamp(dtm).strftime("%Y-%m-%d %H:%M:%S")
            read_error = ""
            dt_source = "File system"

    # this is just for debugging
    if not opened_sucessfully and not opened_with_fail:
        logger.error(f"File {filename} was not opened.")

    return dt_str, read_error, dt_source


def check_file_by_opening(filename):
    """Check if image or video file can be opened."""
    opened_sucessfully = False
    opening_error = None
    frame = None
    media_file_type = "unknown"
    # check if file is ok
    if filename.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        media_file_type = "image"
        try:
            image = Image.open(filename)
            image.verify()
            opened_sucessfully = True
        except Exception as e:
            opening_error = str(e)
            # return "", str(e), ""
    elif filename.suffix.lower() in (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".flv",
        ".wmv",
        ".m4v",
    ):
        media_file_type = "video"
        # import cv2
        try:
            cap = cv2.VideoCapture(str(filename))
            ret, frame = cap.read()
            cap.release()
            opened_sucessfully = True
        except Exception as e:
            opening_error = str(e)
            # return "", str(e), ""
    return opened_sucessfully, opening_error, media_file_type


def get_datetime_exiftool(video_pth: Path, checked_keys: Optional[list] = None) -> typing.Tuple[str, bool, str]:
    """Get datetime from video using exiftool."""
    if checked_keys is None:
        checked_keys = [
            "QuickTime:MediaCreateDate",
            "QuickTime:CreateDate",
            "EXIF:CreateDate",
            "EXIF:ModifyDate",
            "EXIF:DateTimeOriginal",
            "EXIF:DateTimeCreated",
            # "File:FileModifyDate",
            # "File:FileCreateDate",
        ]
    # files = [png", "c.tif"]
    files = [video_pth]
    with exiftool.ExifToolHelper(executable=EXIFTOOL_EXECUTABLE) as et:
        metadata = et.get_metadata(files)
        for d in metadata:
            for k in checked_keys:
                if k in d:
                    return d[k], True, k
            # if no key was found log the metadata
            logger.debug(str(d))
            # print(d)

    return "", False, ""


def get_datetime_from_ocr(filename: Path) -> typing.Tuple[str, str]:
    """Get datetime from image using OCR."""
    import cv2

    # if it is image

    if filename.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        frame_bgr = cv2.imread(str(filename))
    else:
        # read video frame
        cap = cv2.VideoCapture(str(filename))
        ret, frame_bgr = cap.read()
        cap.release()

    # from matplotlib import pyplot as plt
    # plt.imshow(frame_bgr[:, :, ::-1])
    # plt.show()
    date_str, is_cuddleback1, ocr_result = _check_if_it_is_cuddleback1(frame_bgr)
    if not is_cuddleback1:
        date_str, is_cuddleback_corner, ocr_result_corner = _check_if_it_is_cuddleback_corner(frame_bgr)
        ocr_result += "; " + ocr_result_corner
        if not is_cuddleback_corner:
            date_str = ""

    # remove non printable characters
    ocr_result = "".join([c for c in ocr_result if c.isprintable()])
    return date_str, f"OCR: {ocr_result}"


def _check_if_it_is_cuddleback1(frame_bgr: np.nan) -> Tuple[str, bool, str]:
    ocr_result = ""
    try:

        if frame_bgr.shape[0] != 720 or frame_bgr.shape[1] != 1280:
            # this is not Cuddleback
            return "", False, ""
        # Preprocess the frame: Convert to grayscale and apply thresholding
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # maybe, the thresholding is not necessary, but it works now
        _, processed_frame = cv2.threshold(gray_frame, 140, 255, cv2.THRESH_BINARY)
        # Crop the frame to the area where the date is expected
        processed_frame = processed_frame[300:500, :]

        # Use Tesseract to perform OCR on the processed frame
        ocr_result = pytesseract.image_to_string(processed_frame)
        logger.debug(f"OCR: {ocr_result=}")
        # Define a regex pattern to match date and time format:
        # MM/DD/YYYY hh:mm AM
        date_pattern = r"\b(\d{1,2})[-\/s.](\d{1,2})[-\/s.](\d{4}) (\d{1,2}):(\d{1,2}) ?([AP]M)"

        # Search for dates in the OCR result
        dates = re.findall(date_pattern, ocr_result)
        if len(dates) == 0:
            date_str = ""
            is_ok = False
            logger.debug(f"OCR result: {ocr_result}")
            logger.debug(f"{scipy.stats.describe(frame_bgr.ravel())=}")
            return date_str, is_ok, ""

        # fix AM and PM
        if dates[0][5] == "PM":
            hour = str(int(dates[0][3]) + 12)
        else:
            hour = dates[0][3]
        # turn the date into a string in format strftime("%Y-%m-%d %H:%M:%S")
        date_str = f"{dates[0][2]}-{dates[0][0]}-{dates[0][1]} {hour}:{dates[0][4]}:00"
        return date_str, True, ocr_result
    except Exception as e:
        date_str = ""
        logger.debug(e)
        logger.debug(traceback.format_exc())
        logger.warning(f"Error while processing OCR result: {ocr_result}")
        return date_str, False, ""


def _check_if_it_is_cuddleback_corner(frame_bgr: np.array) -> Tuple[str, bool, str]:
    ocr_result = ""
    try:

        frame_hsv = skimage.color.rgb2hsv(frame_bgr[:, :, ::-1])

        yellow_prototype_rgb = np.array([255, 255, 0]) / 255.0
        yellow_prototype_hsv = skimage.color.rgb2hsv(yellow_prototype_rgb)

        dist = np.sqrt(np.sum((frame_hsv - yellow_prototype_hsv) ** 2, axis=2))
        thresholded_255 = ((dist < 0.1) * 255).astype(np.uint8)

        ocr_result = pytesseract.image_to_string(thresholded_255)
        # Define a regex pattern to match date and time format:
        # MM/DD/YYYY hh:mm AM
        date_pattern = r"\d{1,3}Sec (\d{4})/(\d{2})/(\d{2}) (\d{1,2}):(\d{1,2}):(\d{1,2})"

        # Search for dates in the OCR result
        dates = re.findall(date_pattern, ocr_result)
        if len(dates) == 0:
            date_str = ""
            is_ok = False
            logger.debug(f"{np.mean(frame_hsv, axis=(0, 1))=}")
            logger.debug(f"{np.mean(frame_bgr, axis=(0, 1))=}")
            logger.debug(f"{yellow_prototype_hsv=}")
            logger.debug(f"{scipy.stats.describe(frame_bgr.ravel())=}")
            logger.debug(f"OCR result: {ocr_result}")
            logger.debug(f"{scipy.stats.describe(dist.ravel())=}")
            return date_str, is_ok, ""

        hour = dates[0][3]
        # turn the date into a string in format strftime("%Y-%m-%d %H:%M:%S")
        date_str = f"{dates[0][0]}-{dates[0][1]}-{dates[0][2]} {hour}:{dates[0][4]}:{dates[0][4]}"
        return date_str, True, ocr_result
    except Exception as e:
        date_str = ""
        logger.debug(e)
        logger.debug(traceback.format_exc())
        logger.warning(f"Error while processing OCR result: {ocr_result}")
        return date_str, False, ""


# TODO update to use with the exiftool
def get_datetime_from_exif(filename: Path) -> typing.Tuple[str, str]:
    """Extract datetime from EXIF in file and check if image is ok.

    Parameters
    ----------
    filename : name of the file

    Returns
    -------
    str1:
        String with datetime or zero length string if no EXIF is available.

    str2:
        Error type or zero length string if file is ok.


    """
    if filename.exists() and filename.suffix.lower() in (".jpg", ".jpeg", ".png"):
        try:
            image = Image.open(filename)
            image.verify()
            if filename.suffix.lower() in (".jpg", ".jpeg"):
                exifdata = image.getexif()
                tag_id = 306  # DateTimeOriginal
                dt_str = str(exifdata.get(tag_id))
            else:
                dt_str = ""
            read_error = ""
        except UnidentifiedImageError:
            dt_str = ""
            read_error = "UnidentifiedImageError"
        except OSError:
            dt_str = ""
            read_error = "OSError"
    #             logger.warning(traceback.format_exc())
    else:
        dt_str = ""
        read_error = ""

    dt_str = replace_colon_in_exif_datetime(dt_str)
    return dt_str, read_error


def extend_df_with_datetime(df: pd.DataFrame, exiftool_executable=None) -> pd.DataFrame:
    """Extends dataframe with datetime based on image exif information."""
    assert "image_path" in df
    logger.debug("Getting EXIFs")
    with exiftool.ExifToolHelper(executable=exiftool_executable) as et:
        exifs = et.get_metadata(list(df.image_path))
    logger.debug("EXIFs collected")
    dates = []
    for image_path, exif in df.image_path, exifs:
        # date, err = get_datetime_from_exif(Path(image_path))
        date, err, source = get_datetime_using_exif_or_ocr(Path(image_path), exif)
        dates.append(date)
    df["datetime"] = pd.to_datetime(dates, errors="coerce")

    return df


def extend_df_with_sequence_id(df: pd.DataFrame, time_limit: typing.Union[str, datetime] = "120s"):
    """Return resorted dataframe with sequence number based on datetime information.

    Parameters
    ----------
    df: DataFrame with fields: location, datetime, and optionally annotated

    Returns
    -------
    Input dataframe with added column "sequence_number" and "delta_time"

    """
    # generate sequnce_number based on datetime diff
    sort_keys = ["location", "datetime"]
    ascending = [False, True]
    assert len(sort_keys) == len(ascending)
    if "annotated" in df:
        sort_keys.append("annotated")
        ascending.append(False)
    df = df.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)
    df["delta_datetime"] = pd.NaT
    df["delta_datetime"] = np.array(df["delta_datetime"]).astype(np.timedelta64)

    # df[df.media_type=='image'].delta_datetime=df[df.media_type=='image'].datetime.diff()
    df.loc[~df.datetime.isna(), "delta_datetime"] = df[~df.datetime.isna()].datetime.diff()
    tqdm.pandas(desc="sequence_number")
    event_id_manager = DatasetEventIdManager(time_limit=time_limit)
    df["sequence_number"] = df.delta_datetime.progress_map(event_id_manager.create_event_id)
    return df


def add_datetime_from_exif_in_parallel(
    original_paths: List[Path],
    dataset_basedir: Optional[Path] = None,
    exiftool_executable=None,
    num_cores: int = 1,
) -> Tuple[list, list, list, list]:
    """Get list of datetimes from EXIF.

    The EXIF information is extracted in single-core way but with the help of ExifTool.
    """
    logger.debug(f"Getting EXIFs from {len(original_paths)} files.")
    # Collect EXIF info
    if dataset_basedir:
        full_paths = [dataset_basedir / original_path for original_path in original_paths]
    else:
        full_paths = original_paths
    try:
        with exiftool.ExifToolHelper(executable=exiftool_executable) as et:
            # Your code to interact with ExifTool
            exifs = et.get_metadata(full_paths)
    except exiftool.exceptions.ExifToolExecuteError:
        logger.debug(traceback.format_exc())
        logger.warning(f"Error while batch reading EXIFs from {full_paths}.")
        logger.info("Trying to process per file (slow).")

        # do it per file
        exifs = []
        for path in full_paths:
            try:
                with exiftool.ExifToolHelper(executable=None) as et:
                    exif = et.get_metadata(path)
                exifs.append(exif)
            except exiftool.exceptions.ExifToolExecuteError:
                logger.debug(traceback.format_exc())
                logger.error(f"Error while reding EXIF from {str(path)}")
                exifs.append({})

        exifs = [{} for _ in full_paths]

    assert len(exifs) == len(
        full_paths
    ), f"Number of EXIFs ({len(exifs)}) is not equal to number of files ({len(full_paths)}."
    logger.debug("EXIFs collected")

    # Evaluate exif info and use OCR if necessary
    if num_cores > 1:
        datetime_list = Parallel(n_jobs=num_cores)(
            delayed(get_datetime_using_exif_or_ocr)(full_path, exif)
            for full_path, exif in tqdm(list(zip(full_paths, exifs)), desc="datetime from EXIF or OCR parallel")
        )
    else:
        datetime_list = [
            get_datetime_using_exif_or_ocr(full_path, exif)
            for full_path, exif in tqdm(list(zip(full_paths, exifs)), desc="datetime from EXIF or OCR")
        ]

    datetime_list, error_list, source_list = zip(*datetime_list)
    return datetime_list, error_list, source_list, exifs
