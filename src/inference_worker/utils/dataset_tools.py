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
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import List, Optional
import skimage.io
import skimage.transform

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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
    assert len(sort_keys) == len(ascending)
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
    dataframe: typing.Optional[pd.DataFrame],
    dataset_name: typing.Optional[str],
    dataset_base_dir: Path,
    output_path: Path,
    hash_filename: bool = False,
    make_tar: bool = False,
    copy_files: bool = False,
    move_files: bool = False,
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

    if hash_filename:
        dataframe["image_path"] = dataframe["vanilla_path"].apply(make_hash, prefix=dataset_name)
    else:
        dataframe["image_path"] = dataframe["vanilla_path"].apply(
            lambda filename: os.path.join(dataset_name, filename)
        )

    output_path.mkdir(parents=True, exist_ok=True)
    if create_csv:
        dataframe.to_csv(output_path / f"{dataset_name}.csv")

    if copy_files or move_files:
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"{dataset_name}"):
            input_file_path = (dataset_base_dir / row["vanilla_path"]).resolve()
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
        make_tarfile(output_path / f"{dataset_name}.tar.gz", output_path / f"media_{dataset_name}/")

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
        if self.num_cores > 1:
            vanilla_path_groups = Parallel(n_jobs=self.num_cores)(
                delayed(get_relative_paths_in_dir)(self.dataset_basedir, path_group, mask)
                for path_group in tqdm(self.path_groups, desc="getting file list")
            )
        else:
            # single processor version to avoid error:
            #   Error: 'demonic processes are not allowed to have children'
            logger.debug("Using single CPU")
            vanilla_path_groups = [
                get_relative_paths_in_dir(self.dataset_basedir, path_group, mask)
                for path_group in tqdm(self.path_groups, desc="getting file list")
            ]
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

    def make_metadata_csv(self, path: Path):
        """Extract information based on filelist from prev step."""
        if self.filelist_df is None:
            raise ValueError("First, run make_paths_and_exifs_parallel()")
        self.metadata = extract_information_from_dir_structure(self.filelist_df)
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
        dataframe: DataFrame may contain vanilla_path and datetime column.
        """
        output_dict = {}
        output_dict["vanilla_path"] = self.get_paths_from_dir_parallel(mask, exclude)

        if make_exifs:
            datetime_list, read_error_list = self.add_datetime_from_exif_in_parallel(
                output_dict["vanilla_path"]
            )

            output_dict["datetime"] = datetime_list
            output_dict["read_error"] = read_error_list

        df = pd.DataFrame(output_dict)
        self.filelist_df = df
        if make_csv:
            df.to_csv(str(self.filelist_path))

        # save the number of 2nd order dirs to allow fast detection of dataset change
        self.cache.update(dict(len_of_path_groups=len(self.path_groups)))
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

        return df

    def add_datetime_from_exif_in_parallel(self, vanilla_paths: list):
        """Get list of datetimes from EXIF."""
        if self.num_cores > 1:
            datetime_list = Parallel(n_jobs=self.num_cores)(
                delayed(get_datetime_from_exif)(self.dataset_basedir / vanilla_path)
                for vanilla_path in tqdm(vanilla_paths, desc="getting EXIFs")
            )
        else:
            datetime_list = [
                get_datetime_from_exif(self.dataset_basedir / vanilla_path)
                for vanilla_path in tqdm(vanilla_paths, desc="getting EXIFs")
            ]

        datetime_list, error_list = zip(*datetime_list)
        return datetime_list, error_list


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
        vanilla_species=[],
    )
    species_substitution_latin = get_species_substitution_latin()
    species_substitution_czech = list(species_substitution_latin.czech_label)
    with logging_redirect_tqdm():
        for pthistr in tqdm(
            list(df_filelist["vanilla_path"]),
            # list(zip(list(df_filelist["vanilla_path"]), list(df_filelist["datetime"]))),
            desc="general columns",
        ):
            pthir = Path(pthistr)

            media_type = (
                "video"
                if pthir.suffix.lower() in (".avi", ".m4v")
                else "image"
                if pthir.suffix.lower() in (".jpg", "png", ".jpeg")
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
                vanilla_species = None
            elif len(pthir.parts) == 4:
                species = None
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


def make_all_images_in_directory_smaller(dirpath:Path, output_dir:Path, image_width=400, image_quality=70) -> List[Path]:
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
                scale = image_width/img.size[0]
                img = img.resize((image_width, int(img.size[1]*scale)), Image.ANTIALIAS)
                if 'exif' in img.info:
                    img.save(new_pth, 'JPEG', quality=image_quality, exif=img.info['exif'])
                else:
                    img.save(new_pth, 'JPEG', quality=image_quality)
            elif pth.suffix.lower() in (".png"):
                img = Image.open(pth)
                scale = image_width/img.size[0]
                img = img.resize((image_width, int(img.size[1]*scale)), Image.ANTIALIAS)
                img.save(new_pth, 'png')
            else:
                shutil.copy(pth, new_pth)

    return filelist


