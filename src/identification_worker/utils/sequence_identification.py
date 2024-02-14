import re
import typing
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


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
    if type(exif_datetime) == str:
        exif_ex = re.findall(
            r"([0-9]{4}):([0-9]{2}):([0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2})", exif_datetime
        )
        if len(exif_ex) == 1:
            ex = exif_ex[0]
            replaced = f"{ex[0]}-{ex[1]}-{ex[2]} {ex[3]}"

    return replaced


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


def extend_df_with_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Extends dataframe with datetime based on image exif information"""
    assert "image_path" in df
    dates = []
    for image_path in df.image_path:
        date, err = get_datetime_from_exif(Path(image_path))
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
