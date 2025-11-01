import logging
import secrets
import string
import unicodedata
from datetime import datetime
from hashlib import sha1 as sha_constructor
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.utils import timezone
from django.utils.timesince import timesince

logger = logging.getLogger("database")


def convert_datetime_to_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timezone-aware datetime to naive datetime."""
    for col in df.columns:
        if df[col].dtype == "datetime64[ns, UTC]":
            df[col] = df[col].dt.tz_convert(None)
    return df


def generate_sha1(string, salt=None):
    """
    Generates a sha1 hash for supplied string.

    :param string:
        The string that needs to be encrypted.

    :param salt:
        Optionally define your own salt. If none is supplied, will use a random
        string of 5 characters.

    :return: Tuple containing the salt and hash.

    """
    string = str(string)
    if not salt:
        salt = str(sha_constructor(str(secrets.random())).hexdigest()[:5])

    # import hashlib
    # >> > sha = hashlib.sha256()
    # >> > sha.update('somestring'.encode())
    # >> > sha.hexdigest()
    hash_str = sha_constructor((salt + string).encode()).hexdigest()

    return hash_str


def random_string(stringLength=16):
    """TODO add docstring."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(stringLength))


def random_string12():
    """TODO add docstring."""
    return random_string(12)


def random_string8():
    """Generate a random string of length 8."""
    return random_string(8)


def get_output_dir():
    """TODO add docstring."""
    output_directory_path = Path(settings.MEDIA_ROOT) / "output"
    datetimestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = str(output_directory_path / (datetimestr + "_" + random_string(12)) / datetimestr)
    return filename


# def _get_zip_path_in_unqiue_folder(instance, filename):
def get_zip_path_in_unique_folder(instance, filename):
    """Uploads a file to a unique generated Path to keep the original filename."""
    logger.debug("upload_to_unique_folder")
    logger.debug(instance)
    logger.debug(filename)
    logger.debug(instance.uploaded_at)
    hash_str = generate_sha1(instance.uploaded_at, "_")

    # instance_filename = Path(instance.imagefile.path).stem
    # sometimes the instance.imagefile does not exist
    instance_filename = Path(filename).stem

    datetimestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = datetimestr + "_" + instance_filename + "_" + hash_str

    # path cannot be absolute or contain "..", otherwise django will raise an error
    return f"./upload/{unique_id}/{filename}"


# def remove_diacritics(text: str) -> str:
#     """Odstraní diakritiku ze stringu."""
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', text)
#         if unicodedata.category(c) != 'Mn'
#     )


def remove_diacritics(input_str: str):
    """Removes diacritics (accents) from the given Unicode string.

    The function decomposes the string into its combining characters, removes
    any diacritic characters, and then recomposes the string.
    """
    # Normalize the input string to NFD (Normalization Form Decomposed)
    normalized = unicodedata.normalize("NFD", input_str)

    # Filter out combining characters (those in category 'Mn')
    filtered = "".join(c for c in normalized if unicodedata.category(c) != "Mn")

    # Return the normalized string
    return unicodedata.normalize("NFC", filtered)


def order_identity_by_mediafile_count(identity1, identity2):
    """Order identity by mediafile count.

    The identity with fewer media files is the first one."""

    count_media_files_identity1 = identity1.mediafile_set.count()
    count_media_files_identity2 = identity2.mediafile_set.count()
    if count_media_files_identity1 < count_media_files_identity2:
        identity_a = identity1
        identity_b = identity2
    else:
        identity_a = identity2
        identity_b = identity1
    return identity_a, identity_b


def timesince_now(started_at: datetime) -> str:
    if timezone.is_naive(started_at):
        started_at = timezone.make_aware(started_at)

    now = timezone.now()
    description = timesince(started_at, now)
    return description
