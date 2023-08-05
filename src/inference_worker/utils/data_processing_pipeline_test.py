import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from src.inference_worker.utils import data_processing_pipeline
from src.inference_worker.utils.dataset_tools import (  # make_tarfile,
    _species_czech_preprocessing,
    make_hash,
    make_zipfile,
)

logger = logging.getLogger(__file__)


CAID_DATASET_BASEDIR = Path(os.getenv("CAID_DATASET_BASEDIR", r"H:\biology\orig\CarnivoreID"))
CI = os.getenv("CI", False)


def test_make_input_tarfile():
    """Test create input tar file."""
    dir_path = CAID_DATASET_BASEDIR / "test_micro_data"
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


@pytest.mark.parametrize(
    "dataset", ["DATA_SUNAP_tiny_test_subset_smaller", "DUHA_tiny_test_subset_smaller"]
)
def test_analyze_dir(dataset):
    """Test dataset directory analysis."""
    dir_path = CAID_DATASET_BASEDIR / dataset
    metadata, duplicates = data_processing_pipeline.analyze_dataset_directory(dir_path)

    assert len(metadata) > 3
    assert len(metadata.location.unique()) > 1, "There should be some localities."
    assert len(metadata.unique_name.unique()) > 1, "There should be some unique names."


def test_data_preprocessing_parallel():
    """Try the whole processing starting from .tar.gz file."""
    tarfile_path = CAID_DATASET_BASEDIR / "test_micro_data.zip"
    media_dir_path = Path("./test_pipeline/media/")
    csv_path = Path("./test_pipeline/metadata.csv")

    # if media_dir_path.exists():
    shutil.rmtree(media_dir_path, ignore_errors=True)
    csv_path.unlink(missing_ok=True)
    assert not media_dir_path.exists()

    metadata, duplicates = data_processing_pipeline.data_preprocessing(
        tarfile_path, media_dir_path, num_cores=1
    )
    logger.debug(metadata)
    assert (
        len(list(media_dir_path.glob("**/*"))) > 0
    ), "There should be some files in media dir path"


def test_data_processing():
    """Try the whole processing starting from .tar.gz file."""
    # to make it faster - find just one subdir with jpg file
    # dir_path = list((CAID_DATASET_BASEDIR / "DATA_SUNAP_tiny_test_subset").glob("**/*.jpg"))[
    #     0
    # ].parent
    dir_path = CAID_DATASET_BASEDIR / "test_micro_data"
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
    metadata = pd.read_csv(csv_path)
    assert metadata["sequence_number"][0] == 0


def test_confidence_thresholding():
    """Test confidence thresholding and creation of new class for unidentified samples.

    There are generated artificial data. Few samples has low confidence.
    """
    n_classes = 4
    n_samples = 10
    n_uncertain_samples = 3
    np.random.seed(42)

    targs = np.random.random_integers(low=0, high=n_classes - 1, size=n_samples)
    targs.sort()

    uncertain_samples = np.random.random_integers(
        low=0, high=n_samples - 1, size=n_uncertain_samples
    )
    uncertain_samples[0] = 0

    _values = np.random.uniform(0.5, 1, size=[n_classes])
    _keys = list(range(0, _values.shape[0]))
    id2threshold = dict(zip(_keys, _values))

    logits = np.random.normal(0, 1.0, size=[n_samples, n_classes])
    for i in range(n_classes):
        logits[targs == i, i] += 1

    # make values in uncertain samples smaller
    logits[uncertain_samples, :] *= 0.1
    probs = softmax(logits, 1)

    class_ids, top_probs = data_processing_pipeline.do_thresholding_on_probs(probs, id2threshold)

    assert class_ids[uncertain_samples[0]] == n_classes, "The 0th prediction should be uncertain."
