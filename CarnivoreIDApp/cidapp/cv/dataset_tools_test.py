import os
from pathlib import Path

import pandas as pd
import pytest

try:
    import dataset_tools
except ImportError:
    from . import dataset_tools


CARNIVORE_DATASET_BASEDIR = Path(
    os.getenv("CARNIVOREID_DATASET_BASEDIR", r"H:\biology\orig\CarnivoreID")
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
def test_dataset_Sumava():
    """Test whole dataset preparation on small dataset."""
    metadata_path = Path("sumava_metadata")
    sumava_processing = dataset_tools.SumavaDatasetProcessing(
        CARNIVORE_DATASET_BASEDIR / "DATA_SUNAP_tiny_test_subset", file_mask="./**/*.*"
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
    assert sumava_processing.is_update_recommended()

    # do the slow parts
    sumava_processing.make_temp_initial_paths_and_exifs_parallel()
    assert sumava_processing.cache_file.exists()
    assert sumava_processing.filelist_path.exists()

    sumava_processing.make_metadata_csv(metadata_path)
    assert metadata_path.exists()


def test_species_substitution():
    """Test czech species substitution."""
    # Preprocessing of czech species
    sumava_initial_processing = dataset_tools.SumavaDatasetProcessing(
        CARNIVORE_DATASET_BASEDIR / "sumava_test"
    )
    species_czech_preprocessing = sumava_initial_processing.species_czech_preprocessing

    assert species_czech_preprocessing[None] == "nevime"

    txt = "jelen"
    replaced_str = species_czech_preprocessing[txt] if txt in species_czech_preprocessing else txt
    assert replaced_str == "jelen evropsky"


@pytest.mark.skipif(CI, reason="We do not have the dataset on CI server")
def test_make_tar_dataset():
    """Prepare dataset from tiny test subset."""
    sumava_dataset_basedir = CARNIVORE_DATASET_BASEDIR / "DATA_SUNAP_tiny_test_subset"
    output_test_dir = CARNIVORE_DATASET_BASEDIR / "tests/DATA_SUNAP_tiny_test_subset_output"
    sumava_processing = dataset_tools.SumavaDatasetProcessing(sumava_dataset_basedir)
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
    dataframe_images
    assert (output_test_dir / f"{dataset_name}.csv").exists(), "Output file does not exist"
