import pytest
import skimage.io

from . import fs_data
from pathlib import Path
from matplotlib import pyplot as plt
import skimage.io

root_dir = Path(__file__).parent.parent.parent.parent

def test_thumbnail():
    image_dir = root_dir/"test_mini_data"
    thumbnail_path = root_dir/"src/tests/thumbnail.jpg"
    thumbnail_path.unlink(missing_ok=True)

    assert image_dir.exists()
    fs_data.make_thumbnail_from_file(image_dir, thumbnail_path)
    assert thumbnail_path.exists()
    # im = skimage.io.imread(thumbnail_path)
    # plt.imshow(im)
    # plt.show()


    pass
