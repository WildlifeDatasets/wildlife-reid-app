import logging
import secrets
import string
from datetime import datetime
from hashlib import sha1 as sha_constructor
from pathlib import Path

from django.conf import settings

logger = logging.getLogger("database")


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
def _get_zip_path_in_unique_folder(instance, filename):
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


