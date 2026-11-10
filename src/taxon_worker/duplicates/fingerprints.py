"""Incremental exact hashes for an explicit worker batch; never search or mutate a dataset."""

from pathlib import Path

from PIL import Image

from .engine import file_hash, pixel_hash, read_image

HASH_VERSION = "exact-v1"


def _path(root, name):
    path = (root / name).resolve()
    if not name or path == root or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Source media file is unavailable")
    return path


def _stat(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def hash_files(items, media_root, progress=None):
    """Hash only the supplied files, distinguishing whole videos from their reference frames."""
    root = Path(media_root).resolve()
    records = []
    for index, item in enumerate(items):
        record = dict(item)
        record.update(file_sha256="", pixel_sha256="", reference_sha256="")
        try:
            source = _path(root, item["source"])
            record["source_stat"] = _stat(source)
            record["file_sha256"] = file_hash(source)
            visual_name = item["reference"] if item["media_type"] == "video" else item["source"]
            try:
                visual = _path(root, visual_name)
                record["visual_stat"] = _stat(visual)
                pixels = read_image(visual)
                field = "reference_sha256" if item["media_type"] == "video" else "pixel_sha256"
                record[field] = pixel_hash(pixels)
                record["size"] = [pixels.shape[1], pixels.shape[0]]
                if _stat(visual) != record["visual_stat"]:
                    raise ValueError("Visual source changed during hashing")
            except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                record["visual_error"] = str(exc)
                record["pixel_sha256"] = record["reference_sha256"] = ""
            if _stat(source) != record["source_stat"]:
                raise ValueError("Source media file changed during hashing")
        except (OSError, ValueError) as exc:
            record["error"] = str(exc)
            record["file_sha256"] = record["pixel_sha256"] = record["reference_sha256"] = ""
        records.append(record)
        if progress:
            progress(index + 1, len(items))
    return {"version": HASH_VERSION, "items": records}
