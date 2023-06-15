import os
from pathlib import Path
from . import fs_data

root_dir = Path(__file__).parent.parent.parent.parent

CAID_DATASET_BASEDIR = Path(
    os.getenv("CARNIVOREID_DATASET_BASEDIR", r"H:\biology\orig\CarnivoreID")
)


def test_thumbnail():
    """Test thumbnail generation."""
    image_dir = CAID_DATASET_BASEDIR / "test_mini_data"
    thumbnail_path = root_dir / "src/tests/thumbnail.jpg"
    thumbnail_path.unlink(missing_ok=True)

    assert image_dir.exists()
    fs_data.make_thumbnail_from_file(image_dir, thumbnail_path)
    assert thumbnail_path.exists()
    # im = skimage.io.imread(thumbnail_path)
    # plt.imshow(im)
    # plt.show()


def test_get_filenames_csv():
    """Get image path from metadata.csv and check if exists."""
    image_dir = CAID_DATASET_BASEDIR / "test_mini_data_output/images"
    metadata_path = CAID_DATASET_BASEDIR / "test_mini_data_output/metadata.csv"
    image_paths = fs_data.get_images_from_csv(metadata_path)
    assert type(image_paths) == list
    assert (Path(image_dir) / image_paths[0]).exists()
