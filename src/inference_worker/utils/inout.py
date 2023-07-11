import tarfile
from pathlib import Path
from zipfile import ZipFile


def extract_tarfile(tar_file: Path, output_dir: Path):
    """Extract content of the tar file."""
    with tarfile.open(tar_file, "r") as tar_ref:
        tar_ref.extractall(path=output_dir)


def extract_zipfile(zip_file: Path, output_dir: Path):
    """Extract content of the zip file."""
    with ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(path=output_dir)


def extract_archive(archive_file: Path, output_dir: Path):
    """Extract content of the zip or tar file."""
    assert archive_file.suffix.lower() in (".tar", ".tar.gz", ".zip")
    if archive_file.suffix.lower() in (".tar", ".tar.gz"):
        extract_tarfile(archive_file, output_dir)
    else:
        extract_zipfile(archive_file, output_dir)
