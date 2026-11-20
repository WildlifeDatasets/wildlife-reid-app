from datetime import timezone as datetime_timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath

from caidapp.model_tools import random_string
from caidapp.models import MediaFile, UploadedArchive
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone
from PIL import Image


def append_image(upload_id, image_file, *, addition_key=None, captured_at=None):
    """Store an image and schedule inference after its registration commits."""
    from caidapp.tasks import queue_added_image_inference

    if addition_key is not None and (not addition_key.strip() or len(addition_key) > 255):
        raise ValidationError("The addition key must contain between 1 and 255 characters.")

    storage = MediaFile._meta.get_field("mediafile").storage
    saved_path = None
    try:
        with transaction.atomic(durable=True):
            upload = UploadedArchive.objects.select_for_update().get(pk=upload_id)
            mediafiles = upload.mediafile_set
            if addition_key is not None:
                existing = mediafiles.filter(addition_key=addition_key).first()
                if existing is not None:
                    return existing
            if not upload.import_finished or upload.taxon_status == "TAIP":
                raise ValidationError("Wait for the upload's current import and taxon processing to finish.")

            filename = PureWindowsPath(image_file.name).name
            extension = PurePosixPath(filename).suffix.lower()
            if extension not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
                raise ValidationError("Unsupported image extension.")
            try:
                with Image.open(image_file) as image:
                    image.verify()
            except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
                raise ValidationError("The file is not a readable image.") from exc
            image_file.seek(0)

            root = Path(settings.MEDIA_ROOT).resolve()
            output_dir = (root / upload.outputdir).resolve()
            if output_dir == root or not output_dir.is_relative_to(root):
                raise ValidationError("The upload has an invalid media directory.")

            timestamp = timezone.now().astimezone(datetime_timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            for _ in range(20):
                directory = f"added_{timestamp}_{random_string(6)}"
                relative_path = f"{directory}/{filename}"
                if len(relative_path) > MediaFile._meta.get_field("original_filename").max_length:
                    raise ValidationError("The original file name is too long.")
                if (
                    mediafiles.filter(added_directory=directory).exists()
                    or mediafiles.filter(original_filename__startswith=f"{directory}/").exists()
                ):
                    continue
                # Match the ZIP worker's hash of the full original relative path.
                hashed_name = sha256(relative_path.encode()).hexdigest() + extension
                storage_path = (output_dir.relative_to(root) / "images" / hashed_name).as_posix()
                saved_path = storage.save(storage_path, image_file, max_length=500)
                if saved_path != storage_path:
                    storage.delete(saved_path)
                    saved_path = None
                    image_file.seek(0)
                    continue
                media = mediafiles.create(
                    mediafile=saved_path,
                    image_file=saved_path,
                    preview=saved_path,
                    media_type="image",
                    original_filename=relative_path,
                    metadata_json={"original_path": relative_path, "original_file_is_mediafile": True},
                    added_directory=directory,
                    addition_key=addition_key,
                    captured_at=captured_at,
                    locality=upload.locality_at_upload_object,
                )
                transaction.on_commit(lambda: queue_added_image_inference(media.pk), robust=True)
                return media
            raise ValidationError("Could not allocate a unique path for the image.")
    except Exception:
        if saved_path is not None:
            storage.delete(saved_path)
        raise
