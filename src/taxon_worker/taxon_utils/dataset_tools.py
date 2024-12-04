import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import tarfile
import traceback
import typing
import unicodedata
import uuid
from datetime import datetime, timedelta
from hashlib import sha256
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
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .inout import extract_archive
from .sequence_identification import get_datetime_from_exif_or_ocr

logger = logging.getLogger("app")

# this is first level preprocessing fixing the most obvious typos.
_species_czech_preprocessing = {
    None: "nevime",
    "clovek": "clovek moudry",
    "lidi": "clovek moudry",
    "lide": "clovek moudry",
    "divocak": "prase divoke",
    "divoke prase": "prase divoke",
    "divoƒak": "prase divoke",
    "prase": "prase divoke",
    "auta": "auto",
    "stroje": "stroj",
    "jeleni": "jelen evropsky",
    "jelen": "jelen evropsky",
    "kone": "kun domaci",
    "kun": "kun domaci",
    "danek": "danek evropsky",
    "danci": "danek evropsky",
    "srnci": "srnec obecny",
    "srnec": "srnec obecny",
    "liska": "liska obecna",
    "zajic": "zajic polni",
    "rys": "rys ostrovid",
    "jezevec": "jezevec lesni",
    "veverka": "veverka obecna",
    "vlk": "vlk obecny",
    "pes": "pes domaci",
    "tetrev": "tetrev hlusec",
    "krava": "tur domaci",
    "kuna asi lesni": "kuna",
    "kuna asi skalni": "kuna",
    "kuna neid": "kuna",
    "kuna neidentifikovatelna": "kuna",
    "ovce": "ovce domaci",
    "los": "los evropsky",
    "ptak neidentifikovatelny": "ptaci",
    "sika": "jelen sika",
    "ptaci asi kos": "ptaci",
    "ptaci -asi drozd": "ptaci",
    "ptak": "ptaci",
    "kuna nevim": "kuna",
    "motyli": "motyl",
    "kos": "kos cerny",
    "cervenka": "cervenka obecna",
    "datel": "datel cerny",
    "kocka": "kocka",
    "kocka divoka": "kocka divoka",
    "koƒka divoká": "kocka divoka",
    "kocka divoka!!!": "kocka divoka",
    "kozy": "koza",
    "kocka divoka asi": "kocka",
}


def get_species_substitution_latin(
    latin_to_taxonomy_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load transcription table from czech species to scientific with taxonomy.

    Returns
    -------
    species_substitution_latin: Dictionary with czech species names as keys and

    """
    dir_with_this_file = Path(__file__).parent
    if latin_to_taxonomy_csv_path is None:
        species_substitution_path = (
            # dir_with_this_file.parent.parent.parent / "resources/Sumava/species_substitution.csv"
            dir_with_this_file
            / "species_substitution.csv"
        )
    else:
        species_substitution_path = latin_to_taxonomy_csv_path
    if species_substitution_path.exists():
        species_substitution_latin = pd.read_csv(species_substitution_path)
    else:
        species_substitution_latin = pd.DataFrame({"species_czech": [], "species": []})
        logger.warning("Put species_substitution.csv into resources/Sumava/")

    st = species_substitution_latin.copy()
    keys = list(st.keys())
    species_substitution_latin[
        species_substitution_latin.columns
    ] = species_substitution_latin.apply(lambda x: x.str.strip())
    species_substitution_latin.species_czech = species_substitution_latin.species_czech.map(
        strip_accents
    ).str.lower()
    assert len(set(species_substitution_latin.species_czech)) == len(
        species_substitution_latin.species_czech
    ), "The keys are not unique"

    for key in keys:
        stk1 = st[~st[key].isna()]
        stk2 = species_substitution_latin[~species_substitution_latin[key].isna()]
        summm = np.sum(stk1[key] != stk2[key])
        assert summm == 0, "Bad labels in species table. Contain not stripped or no lowered fields"

    species_substitution_latin = species_substitution_latin.rename(
        columns={"species": "Species", "species_czech": "czech_label"}
    )
    return species_substitution_latin


def strip_accents(string: str) -> str:
    """Remove accents from string.

    Parameters
    ----------
    string: str
        String containing accents.

    Returns
    -------
    str :
        String with all accents substituted by ASCII-like character.

    """
    return "".join(
        c for c in unicodedata.normalize("NFD", string) if unicodedata.category(c) != "Mn"
    )


# # TODO use from sequence_identification
# def replace_colon_in_exif_datetime(exif_datetime: str) -> str:
#     """Turn strange EXIF datetime format (containing ':' in date) into standard datetime.
#
#     Parameters
#     ----------
#     exif_datetime : str
#         Input string with datetime in EXIF format i.e. "2022:10:05 10:11:56"
#
#
#     Returns
#     -------
#     string :
#
#     """
#     replaced = exif_datetime
#     if isinstance(exif_datetime, str):
#         exif_ex = re.findall(
#             r"([0-9]{4}):([0-9]{2}):([0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2})",
#             exif_datetime,
#         )
#         if len(exif_ex) == 1:
#             ex = exif_ex[0]
#             replaced = f"{ex[0]}-{ex[1]}-{ex[2]} {ex[3]}"
#
#     return replaced


# TODO use from sequence_identification
# def get_datetime_from_exif_or_ocr(filename: Path) -> typing.Tuple[str, str, str]:
#     """Extract datetime from EXIF in file and check if image is ok.
#
#     Parameters
#     ----------
#     filename : name of the file
#
#     Returns
#     -------
#     str1:
#         String with datetime in forma YYYY-MM-DD HH:MM:SS or zero length string if no EXIF is
#         available.
#
#     str2:
#         Error type or zero length string if file is ok.
#
#         The function also checks if image or video is ok for read.
#     """
#     dt_source = ""
#     in_worst_case_dt = None
#     in_worst_case_dt_source = None
#     opened_sucessfully = False
#     opened_with_fail = False
#     if filename.exists():
#         try:
#             checked_keys = [
#                 "QuickTime:MediaCreateDate",
#                 "QuickTime:CreateDate",
#                 "EXIF:CreateDate",
#                 "EXIF:ModifyDate",
#                 # "File:FileModifyDate",
#             ]
#             dt_str, is_ok, dt_source = get_datetime_exiftool(filename)
#             dt_str = replace_colon_in_exif_datetime(dt_str)
#             if dt_source.startswith("QuickTime"):
#                 in_worst_case_dt = dt_str
#                 in_worst_case_dt_source = dt_source
#                 df_str = ""
#                 dt_source = ""
#             read_error = ""
#         except Exception as e:
#             dt_str = ""
#             read_error = str(e)
#             logger.warning(f"Error while reading EXIF from {filename}")
#             logger.exception(traceback.format_exc())
#             opened_with_fail = True
#     else:
#         return "", "File does not exist", ""
#
#     # check if file is ok
#     if filename.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
#         try:
#             image = Image.open(filename)
#             image.verify()
#             opened_sucessfully = True
#         except Exception as e:
#             return "", str(e), ""
#     elif filename.suffix.lower() in (
#         ".mp4",
#         ".avi",
#         ".mov",
#         ".mkv",
#         ".webm",
#         ".flv",
#         ".wmv",
#         ".m4v",
#     ):
#         # import cv2
#         try:
#             cap = cv2.VideoCapture(str(filename))
#             ret, frame = cap.read()
#             cap.release()
#             opened_sucessfully = True
#         except Exception as e:
#             return "", str(e), ""
#
#     if filename.exists() and read_error == "":
#         if dt_str == "":
#             try:
#                 dt_str, dt_source = get_datetime_from_ocr(filename)
#                 read_error = ""
#                 opened_sucessfully = True
#             except Exception as e:
#                 dt_str = ""
#                 read_error = "OCR failed"
#
#                 logger.warning(f"Error while reading OCR from {filename}")
#                 logger.debug(e)
#                 logger.debug(traceback.format_exc())
#                 opened_with_fail = True
#
#         if (dt_str == "") and (in_worst_case_dt is not None):
#             dt_str = in_worst_case_dt
#             dt_source = in_worst_case_dt_source
#
#         if dt_str == "":
#             dtm = min(filename.stat().st_mtime, filename.stat().st_ctime, filename.stat().st_atime)
#             dt_str = datetime.fromtimestamp(dtm).strftime("%Y-%m-%d %H:%M:%S")
#             read_error = ""
#             dt_source = "File system"
#
#     # this is just for debugging
#     if not opened_sucessfully and not opened_with_fail:
#         logger.error(f"File {filename} was not opened.")
#
#     return dt_str, read_error, dt_source
#
#
# def get_datetime_exiftool(video_pth: Path, checked_keys: Optional[list]=None) -> typing.Tuple[str, bool, str]:
#     """Get datetime from video using exiftool."""
#     if checked_keys is None:
#         checked_keys = [
#             "QuickTime:MediaCreateDate",
#             "QuickTime:CreateDate",
#             "EXIF:CreateDate",
#             "EXIF:ModifyDate",
#             "EXIF:DateTimeOriginal",
#             "EXIF:DateTimeCreated",
#             # "File:FileModifyDate",
#             # "File:FileCreateDate",
#         ]
#     # files = [png", "c.tif"]
#     files = [video_pth]
#     with exiftool.ExifToolHelper() as et:
#         metadata = et.get_metadata(files)
#         for d in metadata:
#             for k in checked_keys:
#                 if k in d:
#                     return d[k], True, k
#             # if no key was found log the metadata
#             logger.debug(str(d))
#             # print(d)
#
#     return "", False, ""
#
#
# def get_datetime_from_ocr(filename: Path) -> typing.Tuple[str, str]:
#     """Get datetime from image using OCR."""
#     import cv2
#
#     # if it is image
#
#     if filename.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
#         frame_bgr = cv2.imread(str(filename))
#     else:
#         # read video frame
#         cap = cv2.VideoCapture(str(filename))
#         ret, frame_bgr = cap.read()
#         cap.release()
#
#     date_str, is_cuddleback1, ocr_result = _check_if_it_is_cuddleback1(frame_bgr)
#     if not is_cuddleback1:
#         date_str, is_cuddleback_corner, ocr_result_corner = _check_if_it_is_cuddleback_corner(
#             frame_bgr
#         )
#         ocr_result += "; " + ocr_result_corner
#         if not is_cuddleback_corner:
#             date_str = ""
#
#     # remove non printable characters
#     ocr_result = "".join([c for c in ocr_result if c.isprintable()])
#     return date_str, f"OCR: {ocr_result}"
#
#
# def _check_if_it_is_cuddleback1(frame_bgr: np.nan) -> Tuple[str, bool, str]:
#     ocr_result = ""
#     try:
#
#         # Preprocess the frame: Convert to grayscale and apply thresholding
#         gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#         # maybe, the thresholding is not necessary, but it works now
#         _, processed_frame = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
#
#         # Use Tesseract to perform OCR on the processed frame
#         ocr_result = pytesseract.image_to_string(processed_frame)
#         # Define a regex pattern to match date and time format:
#         # MM/DD/YYYY hh:mm AM
#         date_pattern = r"\b(\d{1,2})[-\/s.](\d{1,2})[-\/s.](\d{4}) (\d{1,2}):(\d{1,2}) ([AP]M)"
#
#         # Search for dates in the OCR result
#         dates = re.findall(date_pattern, ocr_result)
#         if len(dates) == 0:
#             date_str = ""
#             is_ok = False
#             logger.debug(f"OCR result: {ocr_result}")
#             logger.debug(f"{scipy.stats.describe(frame_bgr.ravel())=}")
#             return date_str, is_ok, ""
#
#         # fix AM and PM
#         if dates[0][5] == "PM":
#             hour = str(int(dates[0][3]) + 12)
#         else:
#             hour = dates[0][3]
#         # turn the date into a string in format strftime("%Y-%m-%d %H:%M:%S")
#         date_str = f"{dates[0][2]}-{dates[0][0]}-{dates[0][1]} {hour}:{dates[0][4]}:00"
#         return date_str, True, ocr_result
#     except Exception as e:
#         date_str = ""
#         logger.debug(e)
#         logger.debug(traceback.format_exc())
#         logger.warning(f"Error while processing OCR result: {ocr_result}")
#         return date_str, False, ""
#
#
# def _check_if_it_is_cuddleback_corner(frame_bgr: np.array) -> Tuple[str, bool, str]:
#     ocr_result = ""
#     try:
#
#         frame_hsv = skimage.color.rgb2hsv(frame_bgr[:, :, ::-1])
#
#         yellow_prototype_rgb = np.array([255, 255, 0]) / 255.0
#         yellow_prototype_hsv = skimage.color.rgb2hsv(yellow_prototype_rgb)
#
#         dist = np.sqrt(np.sum((frame_hsv - yellow_prototype_hsv) ** 2, axis=2))
#         thresholded_255 = ((dist < 0.1) * 255).astype(np.uint8)
#
#         ocr_result = pytesseract.image_to_string(thresholded_255)
#         # Define a regex pattern to match date and time format:
#         # MM/DD/YYYY hh:mm AM
#         date_pattern = r"\d{1,3}Sec (\d{4})/(\d{2})/(\d{2}) (\d{1,2}):(\d{1,2}):(\d{1,2})"
#
#         # Search for dates in the OCR result
#         dates = re.findall(date_pattern, ocr_result)
#         if len(dates) == 0:
#             date_str = ""
#             is_ok = False
#             logger.debug(f"{np.mean(frame_hsv, axis=(0,1))=}")
#             logger.debug(f"{np.mean(frame_bgr, axis=(0,1))=}")
#             logger.debug(f"{yellow_prototype_hsv=}")
#             logger.debug(f"{scipy.stats.describe(frame_bgr.ravel())=}")
#             logger.debug(f"OCR result: {ocr_result}")
#             logger.debug(f"{scipy.stats.describe(dist.ravel())=}")
#             return date_str, is_ok, ""
#
#         hour = dates[0][3]
#         # turn the date into a string in format strftime("%Y-%m-%d %H:%M:%S")
#         date_str = f"{dates[0][0]}-{dates[0][1]}-{dates[0][2]} {hour}:{dates[0][4]}:{dates[0][4]}"
#         return date_str, True, ocr_result
#     except Exception as e:
#         date_str = ""
#         logger.debug(e)
#         logger.debug(traceback.format_exc())
#         logger.warning(f"Error while processing OCR result: {ocr_result}")
#         return date_str, False, ""


def get_date_from_path_structure(filename: str) -> str:
    """Extract date from the directory structure of the Sumava dataset.

    Parameters
    ----------
    filename: str
        Path containing the date in YYYY-MM-DD format in one of the path parts.

    Returns
    -------
    str :
        Date in string

    """
    filename = Path(filename)
    x = re.findall(r"[/\\][0-9]{4}-[0-9]{2}-[0-9]{2}[/\\]", str(filename))
    if len(x) > 0:
        dt_str = x[0][1:-1]
    else:
        dt_str = filename.parents[1].name
    return dt_str


def get_relative_paths_in_dir(basedir_path: Path, path_group: Path, mask: str) -> List[str]:
    """
    Get the list of all files in directory based on mask.

    Parameters
    ----------
    basedir_path: Path
        base directory of dataset
    path_group: Path
        full path to the directory in the dataset
    mask: str
        mask used in internal glob function

    Returns
    -------
    list[str]


    """
    original_paths = []
    for pthi in path_group.glob(mask):
        pthir = pthi.relative_to(basedir_path)
        original_paths.append(str(pthir))
    return original_paths


def make_hash(filename: str, prefix: Optional[str] = "media_data") -> str:
    """Prepare hash based relative path of the filename.

    Parameters
    ----------
    filename: Input file path
    prefix: Prefix of output file path

    Returns
    -------
    hash of the file path

    """
    suffix = Path(filename).suffix
    hash_filename = sha256(str(filename).encode()).hexdigest()

    hash_filename += suffix.lower()
    if prefix and len(prefix) > 0:
        if prefix is None:
            hash_filename = str(hash_filename)

        else:
            hash_filename = str(Path(prefix) / hash_filename)
    return hash_filename


# assert make_hash(filename).startswith("media_data/ae4cd8")
# multiplatform check of the function


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
    df.loc[~df.datetime.isna(), "delta_datetime"] = df[~df.datetime.isna()].datetime.diff()
    tqdm.pandas(desc="sequence_number")
    event_id_manager = DatasetEventIdManager(time_limit=time_limit)
    df["sequence_number"] = df.delta_datetime.progress_map(event_id_manager.create_event_id)
    return df


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


def get_lynx_id_as_parent_name(relative_path: str) -> Optional[str]:
    """Lynx ID is generated from the directory name."""
    path = Path(relative_path)
    if len(path.parts) > 1:
        return path.parent.name
    else:
        return None


def get_lynx_id_in_sumava(relative_path: str) -> Optional[str]:
    """
    Get linx ID based on directory path in Sumava dataset.

    Function is dependent on Sumava dataset directory structure.

    Parameters
    ----------
    relative_path: str
        Relative path to the media file.

    Returns
    -------
    str :
        Identification of the lynx as a string or None if not applicable.

    """
    path = Path(relative_path)
    if (path.parts[0] not in ("TRIDENA", "NETRIDENA")) and len(path.parts) > 2:
        return path.parts[2]
    else:
        return None


def make_zipfile(output_filename: Path, source_dir: Path):
    """Make archive (zip, tar.gz) from a folder.

    Parameters
    ----------
    output_filename: Path of output file
    source_dir: Path to input directory
    """
    output_filename = Path(output_filename)
    source_dir = Path(source_dir)
    archive_type = "zip"

    shutil.make_archive(
        output_filename.parent / output_filename.stem, archive_type, root_dir=source_dir
    )


def make_tarfile(output_filename, source_dir):
    """Create tar package from direcotry."""
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def make_dataset(
    dataframe: typing.Optional[pd.DataFrame],
    dataset_name: typing.Optional[str],
    dataset_base_dir: Path,
    output_path: Path,
    hash_filename: bool = False,
    make_tar: bool = False,
    copy_files: bool = False,
    move_files: bool = False,
    create_csv: bool = False,
    tqdm_desc: typing.Optional[str] = None,
) -> pd.DataFrame:
    """Prepare the '.tar.gz' and '.csv' file based on the dataframe with list of the files.

    Parameters
    ----------
    dataframe: DataFrame
        Pandas DataFrame with 'original_path' column.
    dataset_name : str
        Name for output '.csv' and '.tar.gz'
    dataset_base_dir : Path
        Base dir of the dataset. First subdirs should be "TRIDENA" and "NETRIDENA".
    output_path
        Output directory.
    hash_filename : bool:
        Should be the filenames changed according to the hash?
    make_tar : bool:
        Creating of the tar file is time-consuming. It can be skipped by this parameter.
    copy_files : bool
        Should be the '.tar.gz' and '.csv' files copied to the output directory?
    create_csv : bool
        Create CSV file. The filename is based on dataset name.

    Returns
    -------
    dataframe: DataFrame
        The original input dataframe extended by 'image_file' column.
    """
    assert not (copy_files and move_files), "Onle one arg 'copy_files' or 'move_files' can be True."
    if tqdm_desc is None:
        tqdm_desc = dataset_name

    if hash_filename:
        dataframe["image_path"] = dataframe["original_path"].apply(make_hash, prefix=dataset_name)
    else:
        dataframe["image_path"] = dataframe["original_path"].apply(
            lambda filename: os.path.join(dataset_name, filename)
        )

    output_path.mkdir(parents=True, exist_ok=True)
    if create_csv:
        dataframe.to_csv(output_path / f"{dataset_name}.csv", encoding="utf-8-sig")

    if copy_files or move_files:
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=tqdm_desc):
            input_file_path = (dataset_base_dir / row["original_path"]).resolve()
            output_file_path = (output_path / Path(row["image_path"])).resolve()
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            if input_file_path.is_file():
                try:
                    if copy_files:
                        shutil.copyfile(input_file_path, output_file_path)
                    else:
                        shutil.move(input_file_path, output_file_path)
                except Exception as e:
                    error = traceback.format_exception(e)
                    logger.critical(f"Error while copying/moving file:\n{error}")

    if make_tar:
        logger.info("Creating '.tar.gz' archive.")
        make_tarfile(
            output_path / f"{dataset_name}.tar.gz",
            output_path / f"media_{dataset_name}/",
        )

    return dataframe


class SumavaInitialProcessing:
    """Do slow list of paths and extraction of date and time from EXIF in parallel if necessary.

    Parameters
    ----------
    dataset_basedir
    cache_file
    filelist_path
    group_mask
    num_cores
        Default None. If None, the number of available CPUs is used. If 1 the Parallel is not used.
    """

    def __init__(
        self,
        dataset_basedir: Path,
        cache_file: Optional[Path] = None,
        filelist_path: Optional[Path] = None,
        group_mask: str = "./*/*/*",
        num_cores: Optional[int] = None,
    ):
        if cache_file is None:
            cache_file = Path("cache.json")
        if filelist_path is None:
            filelist_path = Path("carnivoreid_sumava_paths.csv")
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()

        self.cache_file = Path(cache_file)
        self.filelist_path = Path(filelist_path)
        self.group_mask = group_mask
        self.num_cores = num_cores
        self.path_groups = None
        self.dataset_basedir = Path(dataset_basedir)
        self.cache = {}
        self.filelist_df = None
        self.metadata: pd.DataFrame = None
        self.latin_to_taxonomy_csv_path: Optional[Path] = None

    def is_update_necessary(self):
        """Check updates in dataset based on number of subdirectories."""
        self.path_groups = list(self.dataset_basedir.glob(self.group_mask))

        cache_file = self.cache_file

        if cache_file.exists():
            with open(cache_file, "r") as f:
                cache = json.load(f)
        else:
            cache = {}

        self.cache = cache

        if not self.filelist_path.exists():
            logger.warning(
                f"The change of dataset detectet because {str(self.filelist_path)} does not exists."
            )
            return True

        elif "len_of_path_groups" not in cache:
            logger.warning(
                f"The change of dataset detectet because len_of_path_groups \
                is not in cache={cache.keys()} does not exists."
            )
            return True
        elif cache["len_of_path_groups"] != len(self.path_groups):
            logger.warning(
                f"The change of dataset detectet because len_of_path_groups is not equal \
                ({cache['len_of_path_groups']}!={len(self.path_groups)} does not exists."
            )
            return True
        else:
            return False

    def get_paths_from_dir_parallel(self, mask, exclude: Optional[list] = None) -> list:
        """Get list of paths in parallel way.

        Parameters
        ----------
        mask: Mask for filename. For any file in any subdirectory the mask is '**/*.*'
        exclude: List of suffixes to be excluded. The process is not case sensitive.

        Returns
        -------
        original_paths: List of files in folder.
        """
        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]
        exclude = [suffix.lower() for suffix in exclude]

        gmask = "./*"
        list_of_files = set()
        if self.path_groups is None:
            for _ in range(0, 10):
                group_of_dirs = list(self.dataset_basedir.glob(gmask))
                list_of_files = list_of_files.union(group_of_dirs)
                if len(group_of_dirs) > self.num_cores:
                    break
                gmask = gmask + "/*"

            self.path_groups = group_of_dirs
        if self.num_cores > 1:
            original_path_groups = Parallel(n_jobs=self.num_cores)(
                delayed(get_relative_paths_in_dir)(self.dataset_basedir, path_group, mask)
                for path_group in tqdm(self.path_groups, desc="getting file list")
            )
        else:
            # single processor version to avoid error:
            #   Error: 'demonic processes are not allowed to have children'
            logger.debug("Using single CPU")
            original_path_groups = [
                get_relative_paths_in_dir(self.dataset_basedir, path_group, mask)
                for path_group in tqdm(self.path_groups, desc="getting file list")
            ]
        list_of_files = [
            item.relative_to(self.dataset_basedir)
            for item in list_of_files
            if item.suffix not in exclude
        ]
        original_paths = list_of_files + [
            item
            for sublist in original_path_groups
            for item in sublist
            # if item.suffix.lower() not in exclude
        ]
        return original_paths

    def make_metadata_csv(self, path: Path):
        """Extract information based on filelist from prev step."""
        if self.filelist_df is None:
            raise ValueError("First, run make_paths_and_exifs_parallel()")
        self.metadata = extract_information_from_dir_structure(
            self.filelist_df, latin_to_taxonomy_csv_path=self.latin_to_taxonomy_csv_path
        )
        self.metadata.to_csv(path)

    def make_paths_and_exifs_parallel(
        self,
        mask,
        make_exifs: bool = True,
        make_csv: bool = True,
        # n_jobs: Optional[int] = None,
        exclude: Optional[list] = None,
    ) -> pd.DataFrame:
        """Get list of paths based on EXIF in parallel way.

        Parameters
        ----------
        mask: Mask for filename. For any file in any subdirectory the mask is '**/*.*'
        make_exifs
        make_csv
        exclude: List of suffixes to be excluded. The process is not case sensitive.

        Returns
        -------
        dataframe: DataFrame may contain original_path and datetime column.
        """
        output_dict = {}
        output_dict["original_path"] = self.get_paths_from_dir_parallel(mask, exclude)

        if make_exifs:
            datetime_list, read_error_list, source_list = self.add_datetime_from_exif_in_parallel(
                output_dict["original_path"]
            )

            output_dict["datetime"] = datetime_list
            output_dict["read_error"] = read_error_list
            output_dict["datetime_source"] = source_list

        df = pd.DataFrame(output_dict)
        self.filelist_df = df
        if make_csv:
            df.to_csv(str(self.filelist_path))

        # save the number of 2nd order dirs to allow fast detection of dataset change
        self.cache.update(dict(len_of_path_groups=len(self.path_groups)))
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

        return df

    def add_datetime_from_exif_in_parallel(self, original_paths: list):
        """Get list of datetimes from EXIF."""
        if self.num_cores > 1:
            datetime_list = Parallel(n_jobs=self.num_cores)(
                delayed(get_datetime_from_exif_or_ocr)(self.dataset_basedir / original_path)
                for original_path in tqdm(original_paths, desc="getting EXIFs")
            )
        else:
            datetime_list = [
                get_datetime_from_exif_or_ocr(self.dataset_basedir / original_path)
                for original_path in tqdm(original_paths, desc="getting EXIFs")
            ]

        datetime_list, error_list, source_list = zip(*datetime_list)
        return datetime_list, error_list, source_list


def add_column_with_lynx_id(df: pd.DataFrame, contain_identities: bool = False) -> pd.DataFrame:
    """Create column with lynx id based on directory structure."""
    if contain_identities:
        df["unique_name"] = df["original_path"].apply(get_lynx_id_as_parent_name)
    else:
        # If we don't know about identity, we can check if the structure is similar to SUMAVA
        # Get ID of lynx from directories in basedir beside "TRIDENA" and "NETRIDENA"
        df["unique_name"] = df["original_path"].apply(get_lynx_id_in_sumava)
    return df


def extract_information_from_dir_structure(
    df_filelist: pd.DataFrame, latin_to_taxonomy_csv_path: Optional[Path] = None
) -> pd.DataFrame:
    """Get the information from path structure in files in input dataframe.

    Parameters
    ----------
    df_filelist: DataFrame with field 'original_path'

    Returns
    -------
    metadata: DataFrame containing metadata extracted from path structure.
    If the Sumava directory structure is used the locality, species, and date is extracted.

    Example of expected path
    NETRIDENA/LY2019/PRACHATICE/2019-07-01/2019-07-01_12-00-00_0001.jpg
    """
    data = dict(
        filename=[],
        original_path=[],
        suffix=[],
        media_type=[],
        annotated=[],
        vanilla_location=[],
        location=[],
        data_code=[],
        date=[],
        path_len=[],
        czech_label=[],
        vanilla_species=[],
    )
    species_substitution_latin = get_species_substitution_latin(
        latin_to_taxonomy_csv_path=latin_to_taxonomy_csv_path
    )
    species_substitution_czech = list(species_substitution_latin.czech_label)
    with logging_redirect_tqdm():
        for pthistr in tqdm(
            list(df_filelist["original_path"]),
            # list(zip(list(df_filelist["original_path"]), list(df_filelist["datetime"]))),
            desc="general columns",
        ):
            pthir = Path(pthistr)

            if pthir.suffix.lower() in (".avi", ".m4v", ".mp4", ".mov"):
                media_type = "video"
            elif pthir.suffix.lower() in (".jpg", ".png", ".jpeg"):
                media_type = "image"
            else:
                media_type = "unknown"
            data["media_type"].append(media_type)
            data["filename"].append(pthir.name)
            data["original_path"].append(str(pthir))
            data["suffix"].append(pthir.suffix)

            if len(pthir.parts) < 3:

                data["location"].append(None)
                data["annotated"].append(None)
                data["vanilla_location"].append(None)
                data["data_code"].append(None)
                data["date"].append(None)
                data["path_len"].append(None)
                data["czech_label"].append(None)
                data["vanilla_species"].append(None)
                continue
            if pthir.parts[-3] == "image":
                # Duha dataset
                data["location"].append(pthir.parts[-2])
                data["annotated"].append(None)
                data["vanilla_location"].append(None)
                data["data_code"].append(None)
                data["date"].append(None)
                data["path_len"].append(None)
                data["czech_label"].append(None)
                data["vanilla_species"].append(None)
                continue

            data["annotated"].append(True if pthir.parts[0] == "TRIDENA" else False)
            data["data_code"].append(pthir.parts[1])
            data["vanilla_location"].append(pthir.parts[2])
            data["date"].append(get_date_from_path_structure(str(pthir)))
            data["path_len"].append(len(pthir.parts))

            cleaned_location = (
                strip_accents(pthir.parts[2])
                .lower()
                .replace(" ", "")
                .replace("druhy_", "")
                .replace("_", "")
                .replace("3l-", "")
                .replace("-", "")
            )
            data["location"].append(cleaned_location)
            # data["datetime"].append(datetimei)
            # TRIDENA/LY2019/PRACHATICE/2019-07-01/2019-07-01_12-00-00_0001.jpg

            if pthir.parts[0] not in ("TRIDENA", "NETRIDENA"):
                # We are in lynx id dir
                species = "rys ostrovid"
                vanilla_species = None
                data["date"][-1] = None
                data["location"][-1] = None
                data["vanilla_location"][-1] = None

            elif not data["annotated"][-1]:
                species = None
                vanilla_species = None
            elif len(pthir.parts) == 4:
                species = None
                vanilla_species = None
            elif len(pthir.parts) == 5:
                # The most dirty class. Sometimes the
                species = strip_accents(pthir.parents[0].name).lower()
                vanilla_species = pthir.parents[0].name
            else:
                species = strip_accents(pthir.parents[0].name).lower()
                vanilla_species = pthir.parents[0].name

            species = (
                _species_czech_preprocessing[species]
                if species in _species_czech_preprocessing
                else species
            )
            species = species if species in species_substitution_czech else "nevime"
            data["czech_label"].append(species)
            data["vanilla_species"].append(vanilla_species)

        lengths = {key: len(data[key]) for key in data.keys()}
        mx_len = np.max(list(lengths.values()))
        data = {key: data[key] for key in data if (len(data[key]) == mx_len)}

        df = pd.DataFrame(data)

        return df


def make_all_images_in_directory_smaller(
    dirpath: Path, output_dir: Path, image_width=400, image_quality=70
) -> List[Path]:
    """Make all images in directory smaller.

    Parameters:
    -----------
    dirpath: Path
        Path to directory with images.
    output_dir: Path
        Path to directory where to save the images.
    image_width: int
        Width of the image in pixels.
    image_quality: int
        Quality of the JPG image in percent.

    Returns:
    --------
    filelist: List[Path]
        List of paths to the output images.

    """
    filelist = []
    for pth in tqdm(list(dirpath.glob("**/*"))):
        filelist.append(pth)
        if not pth.is_dir():
            new_pth = output_dir / pth.relative_to(dirpath)
            new_pth.parent.mkdir(exist_ok=True, parents=True)
            if pth.suffix.lower() in (".jpg", ".jpeg"):
                img = Image.open(pth)
                scale = image_width / img.size[0]
                img = img.resize((image_width, int(img.size[1] * scale)), Image.ANTIALIAS)
                if "exif" in img.info:
                    img.save(new_pth, "JPEG", quality=image_quality, exif=img.info["exif"])
                else:
                    img.save(new_pth, "JPEG", quality=image_quality)
            elif pth.suffix.lower() in (".png"):
                img = Image.open(pth)
                scale = image_width / img.size[0]
                img = img.resize((image_width, int(img.size[1] * scale)), Image.ANTIALIAS)
                img.save(new_pth, "png")
            else:
                shutil.copy(pth, new_pth)

    return filelist


def hash_file_content(filename: [Path, str]) -> Optional[str]:
    """Make hash from file."""
    filename = Path(filename)
    if filename.is_file():
        with open(filename, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    else:
        return ""


def hash_file_content_for_list_of_files(basedir: Path, filelist: List[Path], num_cores: int = 1):
    """Make hash from file."""
    if num_cores > 1:
        hashes = Parallel(n_jobs=num_cores)(
            delayed(hash_file_content)(basedir / f)
            for f in tqdm(filelist, desc="hashing file content parallel")
        )
    else:
        hashes = [
            hash_file_content(basedir / f)
            for f in tqdm(filelist, desc="hashing file content single core")
        ]
    return hashes


def find_unique_names_between_duplicate_files(
    metadata: pd.DataFrame, basedir: Path, num_cores: int = 1
):
    """Find unique_name in dataframe and extend it to all files with same hash.

    Parameters:
    -----------
    basedir: Path
        Path to directory with dataset.
    metadata: pd.DataFrame
        Metadata dataframe containing column "unique_name" and "original_path".
    num_cores: int
        Number of cores to use.

    Returns:
    --------
    metadata: pd.DataFrame
        Metadata dataframe with updated "unique_name" column.
    hashes: np.array
        Array of hashes for each file in metadata.

    """
    hashes = hash_file_content_for_list_of_files(
        basedir, metadata.original_path, num_cores=num_cores
    )
    hashes = np.array(hashes)
    # metadata["content_hash"] = hashes
    # uns = metadata["unique_name"].unique()
    # # remove None from unique names
    # uns = uns[uns != None].astype("str")
    un_hashes, counts = np.unique(hashes, return_counts=True)
    for unh, count in tqdm(list(zip(un_hashes, counts)), desc="finding duplicates"):
        if count > 1:
            unique_names_with_same_hash = metadata[hashes == unh]["unique_name"].unique()
            # remove None from unique names
            unique_names_with_same_hash = unique_names_with_same_hash[
                unique_names_with_same_hash != None  # noqa 'not None' does not work for numpy array
            ].astype("str")
            if len(unique_names_with_same_hash) == 0:
                continue
            # select longest name from unique names with same hash
            longest_unique_name = sorted(unique_names_with_same_hash)[-1]
            metadata[hashes == unh]["unique_name"] = longest_unique_name

    return metadata, hashes


def analyze_dataset_directory(
    dataset_dir_path: Path,
    num_cores: Optional[int] = None,
    latin_to_taxonomy_csv_path: Optional[Path] = None,
    contains_identities: bool = False,
):
    """Get species, locality, datetime and sequence_id from directory with media files.

    Parameters
    ----------
    dataset_dir_path
        Input directory. First subdirs should be "TRIDENA" and "NETRIDENA", if
        the directory is SUMAVA dataset.

    Returns
    -------
    metadata: DataFrame
        Image and video metadata.
    duplicates: DataFrame
        List of duplicit files.
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    init_processing = SumavaInitialProcessing(dataset_dir_path, num_cores=num_cores)
    df0 = init_processing.make_paths_and_exifs_parallel(
        mask="**/*.*", make_exifs=True, make_csv=False
    )

    df = extract_information_from_dir_structure(
        df0, latin_to_taxonomy_csv_path=latin_to_taxonomy_csv_path
    )

    df["datetime"] = pd.to_datetime(df0.datetime, errors="coerce")
    df["read_error"] = list(df0["read_error"])
    df["datetime_source"] = list(df0["datetime_source"])

    df.loc[:, "sequence_number"] = None

    df = add_column_with_lynx_id(df, contain_identities=contains_identities)
    df, hashes = find_unique_names_between_duplicate_files(
        df, basedir=Path(dataset_dir_path), num_cores=num_cores
    )
    df["content_hash"] = hashes

    df = extend_df_with_sequence_id(df, time_limit="120s")

    # Turn NaN int None
    df = df.where(pd.notnull(df), None)

    # remove directory paths (by empty hash)
    df = df[df.content_hash != ""].reset_index(drop=True)

    # Create list of duplicates based on the same EXIF time
    # duplicates = df[df.delta_datetime == pd.Timedelta("0s")]
    # duplicates = duplicates.copy().reset_index(drop=True)
    # duplicates.to_csv(
    #     "../../../resources/Sumava/list_of_duplicities.csv"
    # )

    # Remove duplicities
    # does not work if the images with unique name are also in TRIDENA or NETRIDENA
    # df = df[df.delta_datetime != pd.Timedelta("0s")].reset_index(drop=True)
    # df = df.drop_duplicates(subset=["content_hash"], keep="first").reset_index(drop=True)

    df = df.sort_values(
        by=["annotated", "location", "datetime"], ascending=[False, False, True]
    ).reset_index(drop=True)
    duplicates_bool = df.duplicated(subset=["content_hash"], keep="first")
    duplicates = df[duplicates_bool].copy().reset_index(drop=True)
    metadata = df[~duplicates_bool].copy().reset_index(drop=True)

    return metadata, duplicates


def data_preprocessing(
    zip_path: Path,
    media_dir_path: Path,
    num_cores: Optional[int] = None,
    contains_identities: bool = False,
    post_update_csv_path: Path = Path("mediafile.post_update.csv"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocessing of data in zip file.

    If the Sumava data dir structure is present, the additional information is extracted.
    Sumava data dir structure: "TRIDENA/SEASON/LOCATION/DATE/SPECIES"

    Parameters
    ----------
    zip_path: file with zipped images
    media_dir_path: output dir for media files with hashed names
    csv_path: Path to csv file
    post_update_csv_path: str - Name of file where will be stored any CSV or XLSX file

    Returns
    -------
    metadata: DataFrame - Image and video metadata

    duplicates: DataFrame - List of duplicate files
    """
    # create temporary directory
    import tempfile
    post_update_csv_path = Path(post_update_csv_path)

    tmp_dir = Path(tempfile.gettempdir()) / str(uuid.uuid4())
    tmp_dir.mkdir(exist_ok=False, parents=True)

    # extract files to the temporary directory
    extract_archive(zip_path, output_dir=tmp_dir)

    # create metadata directory
    df, duplicates = analyze_dataset_directory(
        tmp_dir, num_cores=num_cores, contains_identities=contains_identities
    )
    # post_update CSV is used for updating the metadata after all files are processed

    find_any_spreadsheet_and_save_as_csv(tmp_dir, post_update_csv_path)

    # df["original_path"].map(lambda fn: dataset_tools.make_hash(fn, prefix="media_data"))
    df = make_dataset(
        dataframe=df,
        dataset_name=None,
        dataset_base_dir=tmp_dir,
        output_path=media_dir_path,
        hash_filename=True,
        make_tar=False,
        move_files=True,
        create_csv=False,
        tqdm_desc="copying files",
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return df, duplicates


def find_any_spreadsheet_and_save_as_csv(tmp_dir, csv_path):
    post_update_path = sorted(list(tmp_dir.glob("**/*.csv")) + list(tmp_dir.glob("**/*.xlsx")))
    post_update_path = post_update_path[-1] if len(post_update_path) > 0 else None
    logger.debug(f"{post_update_path=}")
    if post_update_path is not None:
        if post_update_path.suffix == ".csv":
            df_post_update = pd.read_csv(post_update_path)
        elif post_update_path.suffix == ".xlsx":
            df_post_update = pd.read_excel(post_update_path)
        else:
            df_post_update = None
        if df_post_update is not None:
            logger.debug(f"{csv_path=}, {csv_path.parent.exists()=}")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_post_update.to_csv(csv_path, encoding="utf-8-sig")


def make_zipfile_with_categories(
    zip_path: Path,
    media_dir_path: Path,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame]:
    """Put mediafiles into zip according to their category and create updated metadata file.

    The predicted category or 'class_id' is used for the directory name.
    The metadata file contains updated image paths.
    Metadata and media files are saved into zip file.

    Parameters
    ----------
    zip_path: output file with zipped images
    media_dir_path: dir containing media files with hashed names
    metadata: DataFrame - Image and video metadata with predicted category or class_id
    """
    import tempfile

    tmp_dir = Path(tempfile.gettempdir()) / str(uuid.uuid4())
    tmp_dir.mkdir(exist_ok=False, parents=True)
    metadata = metadata.copy(deep=True)

    # create category subdirectories and move images based on prediction
    new_image_paths = []
    for i, row in metadata.iterrows():
        if pd.notnull(row["predicted_category"]):
            predicted_category = row["predicted_category"]
        else:
            predicted_category = f"class_{row['predicted_class_id']}"

        image_path = Path(media_dir_path) / row["image_path"]
        target_dir = Path(tmp_dir, predicted_category)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image_path = target_dir / row["image_path"]
        shutil.copy(image_path, target_image_path)
        new_image_paths.append(os.path.join(predicted_category, row["image_path"]))
    metadata["image_path"] = new_image_paths

    # save metadata file
    metadata.to_csv(tmp_dir / "metadata.csv", encoding="utf-8-sig")

    zip_path.unlink(missing_ok=True)
    make_zipfile(zip_path, tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return metadata
