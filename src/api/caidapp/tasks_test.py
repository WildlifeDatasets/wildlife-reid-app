import pytest
from .models import UploadedArchive
from pathlib import Path
from .tasks import make_thumbnail_for_uploaded_archive

ROOT_DIR = Path(__file__).parent.parent.parent.parent

def test_prepare_thumbnail():
    image_dir = ROOT_DIR/"test_mini_data"
    output_dir = ROOT_DIR / "src/tests/prepare_thumbnail"
    thumbnail_path = output_dir/"thumbnail.jpg"

    uploaded_archive = UploadedArchive()
    uploaded_archive.outputdir = output_dir
    uploaded_archive.thumbnail = thumbnail_path
    uploaded_archive.save()

    make_thumbnail_for_uploaded_archive(uploaded_archive=uploaded_archive, )
