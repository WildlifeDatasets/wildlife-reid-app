import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

try:
    from src.inference_worker.inference_utils import dataset_tools
except ModuleNotFoundError:
    from jupyter_notebooks.datasets.sumava import dataset_tools

logger = logging.getLogger(__file__)

CAID_DATASET_BASEDIR = Path(os.getenv("CAID_DATASET_BASEDIR", r"H:\biology\orig\CarnivoreID"))
CAID_DATASET_BASEDIR = Path(
    os.getenv(
        "CAID_DATASET_BASEDIR",
        r"C:\Users\Jirik\my_bc_data\data\biology\orig\CarnivoreID",
    )
)
CI = os.getenv("CI", False)


def test_make_hash():
    """Test hash function."""
    assert (
        dataset_tools.make_hash("myfilenameverylongtohide.myextension")
        .replace("\\", "/")
        .startswith("media_data/e86ec")
    )


@pytest.mark.skipif(CI, reason="We do not have the dataset on CI server")
def test_dataset_sumava():
    """Test whole dataset preparation on small dataset."""
    metadata_path = Path("sumava_metadata")
    # sumava_processing = dataset_tools.SumavaDatasetProcessing(
    sumava_processing = dataset_tools.SumavaInitialProcessing(
        CAID_DATASET_BASEDIR / "DATA_SUNAP_tiny_test_subset_smaller",  # file_mask="./**/*.*"
    )

    if sumava_processing.filelist_path.exists():
        sumava_processing.filelist_path.unlink()
    if sumava_processing.cache_file.exists():
        sumava_processing.cache_file.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    # remove cached data
    assert ~sumava_processing.cache_file.exists()
    assert ~sumava_processing.filelist_path.exists()

    # check the update-check function
    assert sumava_processing.is_update_necessary()

    # do the slow parts
    sumava_processing.make_paths_and_exifs_parallel(mask="./**/*.*")
    assert sumava_processing.cache_file.exists()
    assert sumava_processing.filelist_path.exists()

    sumava_processing.make_metadata_csv(metadata_path)
    assert metadata_path.exists()


def test_species_substitution():
    """Test czech species substitution."""
    species_czech_preprocessing = dataset_tools._species_czech_preprocessing

    assert species_czech_preprocessing[None] == "nevime"

    txt = "jelen"
    replaced_str = species_czech_preprocessing[txt] if txt in species_czech_preprocessing else txt
    assert replaced_str == "jelen evropsky"


@pytest.mark.skipif(CI, reason="We do not have the dataset on CI server")
def test_make_tar_dataset():
    """Prepare dataset from tiny test subset."""
    sumava_dataset_basedir = CAID_DATASET_BASEDIR / "DATA_SUNAP_tiny_test_subset_smaller"
    output_test_dir = CAID_DATASET_BASEDIR / "tests/DATA_SUNAP_tiny_test_subset_output"
    sumava_processing = dataset_tools.SumavaInitialProcessing(sumava_dataset_basedir)
    sumava_processing.make_paths_and_exifs_parallel(mask="**/*.*", make_exifs=True, make_csv=False)
    metadata_path = Path("sumava_metadata.csv")
    if not metadata_path.exists():
        sumava_processing.make_metadata_csv(metadata_path)

    df = pd.read_csv(metadata_path, index_col=0)
    dataset_name = "sumava"
    dataframe_images = df[df["media_type"] == "image"]
    dataset_tools.make_dataset(
        dataframe_images,
        dataset_name,
        dataset_base_dir=sumava_processing.dataset_basedir,
        output_path=output_test_dir,
        hash_filename=True,
        make_tar=True,
        copy_files=True,
    )
    assert (output_test_dir / f"{dataset_name}.csv").exists(), "Output file does not exist"


def test_make_dataset_smaller():
    """Test making dataset smaller by resizing images."""
    dir_path = CAID_DATASET_BASEDIR / "DATA_SUNAP_tiny_test_subset"
    dir_path = CAID_DATASET_BASEDIR / "lynx_ids_small"
    # dir_path = CAID_DATASET_BASEDIR / "DUHA_tiny_test_subset"
    output_dir_path = dir_path.parent / (dir_path.name + "_smaller")
    output_files = dataset_tools.make_all_images_in_directory_smaller(dir_path, output_dir_path)
    assert len(output_files) > 0
    for output_file in output_files:
        assert output_file.exists(), "Output file does not exist"


def test_species_preprocessing():
    """Test transcription table for preprocessing of czech typos."""
    assert dataset_tools._species_czech_preprocessing[None] == "nevime"

    txt = "jelen"
    replaced_str = (
        dataset_tools._species_czech_preprocessing[txt]
        if txt in dataset_tools._species_czech_preprocessing
        else txt
    )
    assert replaced_str == "jelen evropsky"


def test_hash():
    """Test filename to hash function."""
    assert (
        dataset_tools.make_hash("myfilenameverylongtohide.myextension")
        .replace("\\", "/")
        .startswith("media_data/e86ec")
    )


@pytest.mark.parametrize(
    "dataset", ["DATA_SUNAP_tiny_test_subset_smaller", "DUHA_tiny_test_subset_smaller"]
)
def test_analyze_dir(dataset):
    """Test dataset directory analysis."""
    dir_path = CAID_DATASET_BASEDIR / dataset
    metadata, duplicates = dataset_tools.analyze_dataset_directory(dir_path, num_cores=2)

    metadata.to_csv("test_metadata.csv", encoding="utf-8-sig")
    duplicates.to_csv("test_duplicates.csv", encoding="utf-8-sig")
    assert len(metadata) > 3
    assert len(metadata.location.unique()) > 1, "There should be some localities."


def test_analyze_dir_unique_names_as_parent_name():
    """Test dataset directory analysis."""
    dir_path = CAID_DATASET_BASEDIR / "lynx_ids_FeCuMa_smaller"
    metadata, duplicates = dataset_tools.analyze_dataset_directory(
        dir_path, num_cores=2, contains_identities=True
    )

    metadata.to_csv("test_metadata.csv", encoding="utf-8-sig")
    duplicates.to_csv("test_duplicates.csv", encoding="utf-8-sig")
    assert len(metadata) > 3
    assert len(metadata.unique_name.unique()) > 1, "There should be some unique names."


@pytest.mark.parametrize(
    "dataset",
    [
        "DATA_SUNAP_tiny_test_subset_smaller",
        # "DUHA_tiny_test_subset_smaller"
    ],
)
def test_analyze_dir_sumava_unique_names(dataset):
    """Test dataset directory analysis."""
    dir_path = CAID_DATASET_BASEDIR / dataset
    metadata, duplicates = dataset_tools.analyze_dataset_directory(dir_path, num_cores=2)

    metadata.to_csv("test_metadata.csv", encoding="utf-8-sig")
    duplicates.to_csv("test_duplicates.csv", encoding="utf-8-sig")
    assert len(metadata) > 3
    assert len(metadata.location.unique()) > 1, "There should be some localities."
    assert len(metadata.unique_name.unique()) > 1, "There should be some unique names."
    assert len(duplicates) > 0
    rows_with_duplicate = metadata[metadata.content_hash == duplicates.content_hash[0]]
    assert len(rows_with_duplicate) > 0
    assert rows_with_duplicate.annotated.iat[0] == True  # noqa


def test_data_preprocessing_parallel():
    """Try the whole processing starting from .tar.gz file."""
    tarfile_path = CAID_DATASET_BASEDIR / "test_micro_data.zip"
    media_dir_path = Path("./test_pipeline/media/")
    csv_path = Path("./test_pipeline/metadata.csv")

    # if media_dir_path.exists():
    shutil.rmtree(media_dir_path, ignore_errors=True)
    csv_path.unlink(missing_ok=True)
    assert not media_dir_path.exists()

    metadata, duplicates = dataset_tools.data_preprocessing(
        tarfile_path, media_dir_path, num_cores=1
    )
    logger.debug(metadata)
    assert (
        len(list(media_dir_path.glob("**/*"))) > 0
    ), "There should be some files in media dir path"


def test_make_input_tarfile():
    """Test create input tar file."""
    dir_path = CAID_DATASET_BASEDIR / "test_micro_data"
    output_tarfile = Path("images.zip")
    output_tarfile.unlink(missing_ok=True)
    dataset_tools.make_zipfile(output_tarfile, dir_path)
    # shutil.make_archive(output_tarfile.parent / output_tarfile.stem, "zip", root_dir=dir_path)
    # make_tarfile(output_tarfile, dir_path)
    assert output_tarfile.exists()
