"""Comparable media metadata, excluding known import/runtime provenance only."""

import ast
import json


GENERATED_PATH_FIELDS = frozenset({
    "absolute_media_path", "full_image_path", "image_path", "preview_path",
    "thumbnail_path", "static_thumbnail_path",
})
RUNTIME_REFERENCE_FIELDS = frozenset({"sequence_number", "reid_observation_id"})
EXIF_RUNTIME_FIELDS = frozenset({
    "SourceFile", "File:Directory", "File:FileAccessDate", "File:FileModifyDate",
    "File:FileInodeChangeDate", "File:FilePermissions",
})


def _comparable_exif(value):
    """Support the existing Python-dict string, JSON string and object formats."""
    original = value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError, RecursionError):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError, TypeError, RecursionError):
                return original
    if not isinstance(value, dict):
        return original
    return {key: item for key, item in value.items() if key not in EXIF_RUNTIME_FIELDS}


def comparable_metadata(value):
    """Return a comparison copy; original metadata and signed audit snapshots stay intact."""
    if not isinstance(value, dict):
        return value
    result = {key: item for key, item in value.items() if key not in GENERATED_PATH_FIELDS | RUNTIME_REFERENCE_FIELDS}
    if "exif" in result:
        result["exif"] = _comparable_exif(result["exif"])
    return result
