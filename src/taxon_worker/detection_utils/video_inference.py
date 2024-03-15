
import pandas as pd



def create_image_from_video(metadata:pd.DataFrame) -> pd.DataFrame:
    """
    Create image from video.

    Use full_image_path to get the input file, media_type indicates the type of media.
    Keep the video path in full_orig_media_path and change the full_image_path and image_path
    new image.
    """

    return metadata