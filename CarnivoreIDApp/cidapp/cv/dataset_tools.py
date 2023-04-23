import json
import multiprocessing
import os.path
import re
import shutil
import tarfile
import typing
import unicodedata
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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


def get_species_substitution_latin():
    """Load transcription table from czech species to scientific with taxonomy.

    Returns
    -------
    species_substitution_latin: Dictionary with czech species names as keys and

    """
    dir_with_this_file = Path(__file__).parent
    species_substitution_path = (
        dir_with_this_file.parent.parent.parent / "resources/Sumava/species_substitution.csv"
    )
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


def get_datetime_from_exif(filename: Path) -> str:
    """Extract datetime from EXIF in file.

    Parameters
    ----------
    filename : name of the file

    Returns
    -------
    str :
        String with datetime or zero length string if no EXIF is available.

    """
    if filename.exists() and filename.suffix.lower() in (".jpg", ".jpeg"):
        try:
            image = Image.open(filename)
            exifdata = image.getexif()
            tag_id = 306  # DateTimeOriginal
            dt_str = exifdata.get(tag_id)
        except UnidentifiedImageError:
            dt_str = ""
        except OSError:
            dt_str = ""
    #             logger.warning(traceback.format_exc())
    else:
        dt_str = ""

    dt_str = replace_colon_in_exif_datetime(dt_str)
    return dt_str


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
    vanilla_paths = []
    for pthi in path_group.glob(mask):
        pthir = pthi.relative_to(basedir_path)
        vanilla_paths.append(str(pthir))
    return vanilla_paths


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
    if "annotated" in df:
        sort_keys.append("annotated")
        ascending.append(False)
    df = df.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)
    df["delta_datetime"] = pd.NaT
    # df[df.media_type=='image'].delta_datetime=df[df.media_type=='image'].datetime.diff()
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
        if pd.isnull(delta_datetime):
            return None
        if delta_datetime < pd.Timedelta(0) or delta_datetime > pd.Timedelta(self.time_limit):
            self.event_id += int(1)
        return self.event_id


def get_lynx_id_in_sumava(relative_path: str) -> str:
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
    dataframe: pd.DataFrame,
    dataset_name: str,
    dataset_base_dir: Path,
    output_path: Path,
    hash_filename: bool = False,
    make_tar: bool = False,
    copy_files: bool = False,
    create_csv: bool = False,
) -> pd.DataFrame:
    """Prepare the '.tar.gz' and '.csv' file based on the dataframe with list of the files.

    Parameters
    ----------
    dataframe: DataFrame
        Pandas DataFrame with 'vanilla_path' column.
    dataset_name : str
        Name for output '.csv' and '.tar.gz'
    dataset_base_dir : Path
        Base dir of the dataset.
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
    if hash_filename:
        # dataframe["image_path"] = dataframe["vanilla_path"].map(
        #     lambda filename: make_hash(filename, prefix="media_{dataset_name}")
        # )

        dataframe.loc[:, "image_path"] = dataframe.loc[:, "vanilla_path"].map(
            lambda filename: make_hash(filename, prefix=dataset_name)
        )
    else:
        # dataframe["image_path"] = dataframe["vanilla_path"].map(
        #     lambda filename: str(Path(f"media_{dataset_name}") / filename)
        # )
        # dataframe["image_path"] = dataframe["vanilla_path"].map(
        #     lambda fn: make_hash(fn, prefix="media_{dataset_name}")
        # )
        new_image_path = dataframe.loc[:, "vanilla_path"].map(
            lambda filename: str(Path(f"{dataset_name}") / filename)
        )
        # dataframe.loc[:, "image_path"] = list(new_image_path)
        dataframe["image_path"] = list(new_image_path)

    output_path.mkdir(parents=True, exist_ok=True)
    if create_csv:
        dataframe.to_csv(output_path / f"{dataset_name}.csv")

    if copy_files:
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"{dataset_name}"):
            input_file_path = dataset_base_dir / row["vanilla_path"]

            output_file_path = output_path / Path(row["image_path"])

            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"copyfile {input_file_path}, {output_file_path}")
            try:
                shutil.copyfile(input_file_path, output_file_path)
            except Exception as e:
                import traceback

                logger.warning(traceback.format_exception(e))

    if make_tar:
        logger.info("... preparing .tar.gz")
        make_tarfile(output_path / f"{dataset_name}.tar.gz", output_path / f"media_{dataset_name}/")
    return dataframe


class SumavaInitialProcessing:
    """Do slow list of paths and extraction of date and time from EXIF in parallel if necessary."""

    def __init__(
        self,
        dataset_basedir: Path,
        cache_file: Optional[Path] = None,
        filelist_path: Optional[Path] = None,
        group_mask: str = "./*/*/*",
        num_cores=None,
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
        vanilla_paths: List of files in folder.
        """
        if exclude is None:
            exclude = []
        elif type(exclude) == str:
            exclude = [exclude]
        exclude = [suffix.lower() for suffix in exclude]

        gmask = "./*"
        list_of_files = set()
        if self.path_groups is None:
            # self.path_groups = list(self.dataset_basedir.glob(self.group_mask))
            for i in range(0, 10):
                group_of_dirs = list(self.dataset_basedir.glob(gmask))
                list_of_files = list_of_files.union(group_of_dirs)
                if len(group_of_dirs) > self.num_cores:
                    break
                gmask = gmask + "/*"

            self.path_groups = group_of_dirs
        vanilla_path_groups = Parallel(n_jobs=self.num_cores)(
            delayed(get_relative_paths_in_dir)(self.dataset_basedir, path_group, mask)
            for path_group in tqdm(self.path_groups, desc="getting file list")
        )
        list_of_files = [
            item.relative_to(self.dataset_basedir)
            for item in list_of_files
            if item.suffix not in exclude
        ]
        vanilla_paths = list_of_files + [
            item
            for sublist in vanilla_path_groups
            for item in sublist
            # if item.suffix.lower() not in exclude
        ]
        return vanilla_paths

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
        dataframe: DataFrame may contain vanilla_path and datetime column.
        """
        output_dict = {}
        output_dict["vanilla_path"] = self.get_paths_from_dir_parallel(mask, exclude)

        if make_exifs:
            output_dict["datetime"] = self.add_datetime_from_exif_in_parallel(
                output_dict["vanilla_path"]
            )

        df = pd.DataFrame(output_dict)
        if make_csv:
            df.to_csv(str(self.filelist_path))

        # save the number of 2nd order dirs to allow fast detection of dataset change
        self.cache.update(dict(len_of_path_groups=len(self.path_groups)))
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

        return df

    def add_datetime_from_exif_in_parallel(self, vanilla_paths: list):
        """Get list of datetimes from EXIF."""
        datetime_list = Parallel(n_jobs=self.num_cores)(
            delayed(get_datetime_from_exif)(self.dataset_basedir / vanilla_path)
            for vanilla_path in tqdm(vanilla_paths, desc="getting EXIFs")
        )
        return datetime_list


# def extract_information_from_filename()
def extract_information_from_dir_structure(df_filelist: pd.DataFrame) -> pd.DataFrame:
    """Get the information from path structure in files in input dataframe.

    Parameters
    ----------
    df_filelist: DataFrame with field 'vanilla_path'

    Returns
    -------
    metadata: DataFrame containing metadata extracted from path structure.
    If the Sumava directory structure is used the locality, species, and date is extracted.

    """
    data = dict(
        filename=[],
        vanilla_path=[],
        suffix=[],
        media_type=[],
        annotated=[],
        vanilla_location=[],
        location=[],
        data_code=[],
        date=[],
        path_len=[],
        czech_label=[],
        # species=[],
        vanilla_species=[],
        # datetime=[],
    )
    species_substitution_latin = get_species_substitution_latin()
    species_substitution_czech = list(species_substitution_latin.czech_label)
    with logging_redirect_tqdm():
        for pthistr in tqdm(
            list(df_filelist["vanilla_path"]),
            # list(zip(list(df_filelist["vanilla_path"]), list(df_filelist["datetime"]))),
            desc="general columns",
        ):
            # pthi = Path(pthistr)
            # pthir = pthi.relative_to(pth)
            pthir = Path(pthistr)

            media_type = (
                "video"
                if pthir.suffix.lower() in (".avi", ".m4v")
                else "image"
                if pthir.suffix.lower() in (".jpg", "png")
                else "unknown"
            )
            data["media_type"].append(media_type)
            data["filename"].append(pthir.name)
            data["vanilla_path"].append(str(pthir))
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

            if pthir.parts[0] not in ("TRIDENA", "NETRIDENA"):
                # We are in lynx id dir
                species = "rys ostrovid"
                vanilla_species = None
                data["date"][-1] = None
                data["location"][-1] = None
                data["vanilla_location"][-1] = None

            elif not data["annotated"][-1]:
                species = None
                # species = "nevime"
                vanilla_species = None
            elif len(pthir.parts) == 4:
                species = None
                # species = "nevime"
                vanilla_species = None
            elif len(pthir.parts) == 5:
                # The most dirty class. Sometimes the
                species = strip_accents(pthir.parents[0].name).lower()
                # species = species if (
                #     (species in ok_species) or (species in species_replacement)
                #     ) else None
                vanilla_species = pthir.parents[0].name
            else:
                species = strip_accents(pthir.parents[0].name).lower()
                vanilla_species = pthir.parents[0].name

            #     species = replace_multiple(species, species_replacement)
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
