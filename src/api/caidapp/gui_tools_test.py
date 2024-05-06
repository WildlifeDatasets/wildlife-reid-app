import numpy as np

from . import gui_tools


def test_show_pair():
    """Test the show_pair function."""
    kp0 = (np.random.rand(10, 2) * 100).tolist()
    kp1 = (np.random.rand(10, 2) * 100).tolist()

    img0 = np.random.rand(200, 200, 3) * 100 + 50
    img1 = np.random.rand(200, 200, 3) * 100 + 50

    query_name = "query"
    database_name = "database"

    gui_tools.create_match_img_src(kp0, kp1, img0, img1, query_name, database_name)
    # gui_tools.create_match_image_plotly(kp0, kp1, img0, img1, query_name, database_name)
    assert True
