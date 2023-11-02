import django

django.setup()
import logging
import os
from pathlib import Path

from .models import UploadedArchive
from .tasks import make_thumbnail_for_uploaded_archive, sync_mediafiles_uploaded_archive_with_csv

logger = logging.getLogger(__file__)

ROOT_DIR = Path(__file__).parent.parent.parent.parent
CAID_DATASET_BASEDIR = Path(
    os.getenv("CARNIVOREID_DATASET_BASEDIR", r"H:\biology\orig\CarnivoreID")
)


def test_prepare_thumbnail():
    """Check thumbnail generation."""
    output_dir = ROOT_DIR / "src/tests/prepare_thumbnail"
    thumbnail_path = output_dir / "thumbnail.jpg"

    uploaded_archive = UploadedArchive()
    uploaded_archive.outputdir = output_dir
    uploaded_archive.thumbnail = thumbnail_path
    uploaded_archive.save()

    make_thumbnail_for_uploaded_archive(
        uploaded_archive=uploaded_archive,
    )


def test_add_mediafiles_from_csv():
    """Test mediafiles extraction from CSV."""
    output_dir = CAID_DATASET_BASEDIR / "test_mini_data_output"
    csv_file = output_dir / "metadata.csv"

    uploaded_archive = UploadedArchive()
    uploaded_archive.outputdir = output_dir
    uploaded_archive.csv_file = csv_file
    # uploaded_archive.save()

    sync_mediafiles_uploaded_archive_with_csv(uploaded_archive)
    media_files = uploaded_archive.mediafile_set.all()
    logger.debug(media_files)
    assert len(media_files) > 1
