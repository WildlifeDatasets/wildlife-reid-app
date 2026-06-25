import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

try:
    from src.taxon_worker.taxon_utils import data_processing_pipeline
    from src.taxon_worker.taxon_utils.dataset_tools import make_zipfile  # make_tarfile,
except ModuleNotFoundError:
    try:
        from taxon_utils import data_processing_pipeline
        from taxon_utils.dataset_tools import make_zipfile
    except ModuleNotFoundError:
        from jupyter_notebooks.datasets.sumava import data_processing_pipeline
        from jupyter_notebooks.datasets.sumava.dataset_tools import make_zipfile

logger = logging.getLogger(__file__)


CAID_DATASET_BASEDIR = Path(os.getenv("CAID_DATASET_BASEDIR", r"H:\biology\orig\CarnivoreID"))
CI = os.getenv("CI", False)


def test_keep_correctly_loaded_images_reports_missing_prepared_file(tmp_path):
    existing = tmp_path / "existing.webp"
    existing.write_bytes(b"image")
    missing = tmp_path / "missing.webp"
    metadata = pd.DataFrame(
        [
            {
                "media_type": "image",
                "read_error": "",
                "full_image_path": str(existing),
            },
            {
                "media_type": "image",
                "read_error": "",
                "full_image_path": str(missing),
            },
        ]
    )

    correct, failing = data_processing_pipeline.keep_correctly_loaded_images(metadata)

    assert len(correct) == 1
    assert len(failing) == 1
    assert str(missing) in failing.iloc[0]["read_error"]


def test_make_previews_creates_media_variants_once_for_duplicate_media_rows(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "first.webp"
    image_path.write_bytes(b"image")
    calls = []

    def fake_thumbnail(source_path, target_path, width=800):
        calls.append(("thumbnail", Path(source_path), Path(target_path), width))
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(b"variant")
        return True

    monkeypatch.setattr(data_processing_pipeline, "make_thumbnail_from_file", fake_thumbnail)

    metadata = pd.DataFrame(
        [
            {
                "image_path": "first.webp",
                "absolute_media_path": str(image_path),
                "full_image_path": str(image_path),
                "media_type": "image",
            },
            {
                "image_path": "first.webp",
                "absolute_media_path": str(image_path),
                "full_image_path": str(image_path),
                "media_type": "image",
            },
        ]
    )

    result = data_processing_pipeline.make_previews(metadata, output_dir, preview_width=1200, thumbnail_width=400)

    assert list(result["preview_path"]) == ["previews/first.webp", "previews/first.webp"]
    assert list(result["thumbnail_path"]) == ["thumbnails/first.webp", "thumbnails/first.webp"]
    assert list(result["static_thumbnail_path"]) == ["static_thumbnails/first.webp", "static_thumbnails/first.webp"]
    assert len(calls) == 3


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
    assert len(list(media_dir_path.glob("**/*"))) > 0, "There should be some files in media dir path"

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

    uncertain_samples = np.random.random_integers(low=0, high=n_samples - 1, size=n_uncertain_samples)
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
