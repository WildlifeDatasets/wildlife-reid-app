import os
import shutil
from pathlib import Path

import pytest
from CarnivoreIDApp.cidapp.cv import data_processing_pipeline
from CarnivoreIDApp.cidapp.cv.dataset_tools import (  # make_tarfile,
    _species_czech_preprocessing,
    make_hash,
    make_zipfile,
)

# try:
#     from . import data_processing_pipeline
# except ImportError:
#     import data_processing_pipeline


CID_DATASET_BASEDIR = Path(os.getenv("CARNIVOREID_DATASET_BASEDIR", r"H:\biology\orig\CarnivoreID"))
CI = os.getenv("CI", False)


def test_make_input_tarfile():
    """Test create input tar file."""
    dir_path = CID_DATASET_BASEDIR / "DATA_SUNAP_tiny_test_subset"
    output_tarfile = Path("images.zip")
    output_tarfile.unlink(missing_ok=True)
    make_zipfile(output_tarfile, dir_path)
    # shutil.make_archive(output_tarfile.parent / output_tarfile.stem, "zip", root_dir=dir_path)
    # make_tarfile(output_tarfile, dir_path)
    assert output_tarfile.exists()


def test_species_preprocessing():
    """Test transcription table for preprocessing of czech typos."""
    assert _species_czech_preprocessing[None] == "nevime"

    txt = "jelen"
    replaced_str = _species_czech_preprocessing[txt] if txt in _species_czech_preprocessing else txt
    assert replaced_str == "jelen evropsky"


def test_hash():
    """Test filename to hash function."""
    assert (
        make_hash("myfilenameverylongtohide.myextension")
        .replace("\\", "/")
        .startswith("media_data/e86ec")
    )


@pytest.mark.parametrize("dataset", ["DATA_SUNAP_tiny_test_subset", "DUHA_tiny_test_subset"])
def test_analyze_dir(dataset):
    """Test dataset directory analysis."""
    dir_path = CID_DATASET_BASEDIR / dataset
    metadata, duplicates = data_processing_pipeline.analyze_dataset_directory(dir_path)

    assert len(metadata) > 3
    assert len(metadata.location.unique()) > 1, "There should be some localities."


def test_data_preprocessing_parallel():
    """Try the whole processing starting from .tar.gz file."""
    tarfile_path = Path("images.tar.gz")
    tarfile_path = Path("images.zip")
    media_dir_path = Path("./test_pipeline/media/")
    csv_path = Path("./test_pipeline/metadata.csv")

    # if media_dir_path.exists():
    shutil.rmtree(media_dir_path, ignore_errors=True)
    csv_path.unlink(missing_ok=True)
    assert not media_dir_path.exists()

    metadata, duplicates = data_processing_pipeline.data_preprocessing(
        tarfile_path, media_dir_path, num_cores=1
    )
    assert (
        len(list(media_dir_path.glob("**/*"))) > 0
    ), "There should be some files in media dir path"


def test_data_processing():
    """Try the whole processing starting from .tar.gz file."""
    # to make it faster - find just one subdir with jpg file
    dir_path = list((CID_DATASET_BASEDIR / "DATA_SUNAP_tiny_test_subset").glob("**/*.jpg"))[
        0
    ].parent
    tarfile_path = Path("few_images.zip")
    tarfile_path.unlink(missing_ok=True)
    make_zipfile(tarfile_path, dir_path)
    media_dir_path = Path("./test_pipeline/media_few/")
    csv_path = Path("./test_pipeline/metadata_few.csv")

    # if media_dir_path.exists():
    shutil.rmtree(media_dir_path, ignore_errors=True)
    csv_path.unlink(missing_ok=True)
    assert not media_dir_path.exists()

    data_processing_pipeline.data_processing(tarfile_path, media_dir_path, csv_path)
    assert (
        len(list(media_dir_path.glob("**/*"))) > 0
    ), "There should be some files in media dir path"

    assert csv_path.exists()
