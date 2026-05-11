import io
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import pandas as pd
from django.core.files.base import ContentFile
from django.utils import timezone

logger = logging.getLogger(__name__)

SPREADSHEET_SUFFIXES = {".csv", ".xls", ".xlsx"}
ARCHIVE_SUFFIXES = {".zip"}

COLUMN_ALIASES = {
    "original path": "original_path",
    "original_path": "original_path",
    "mediafile": "original_path",
    "media file": "original_path",
    "unique name": "unique_name",
    "unique_name": "unique_name",
    "category": "taxon",
    "taxon": "taxon",
    "location_name": "locality_name",
    "locality name": "locality_name",
    "locality_name": "locality_name",
    "lat": "latitude",
    "latitude": "latitude",
    "lon": "longitude",
    "longitude": "longitude",
    "datetime": "datetime",
}


@dataclass
class SpreadsheetSummary:
    filename: str = ""
    columns: list[str] = field(default_factory=list)
    normalized_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ZipBuildResult:
    file: Any
    filename: str
    archive_name: str
    import_mapping: dict[str, Any]
    import_log: str
    spreadsheet_summary: SpreadsheetSummary


def load_relative_path_manifest(raw_manifest: str) -> list[dict[str, Any]]:
    if not raw_manifest:
        return []
    try:
        data = json.loads(raw_manifest)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _safe_zip_path(path: str) -> str:
    path = str(path).replace("\\", "/").strip("/")
    pure_path = PurePosixPath(path)
    safe_parts = [part for part in pure_path.parts if part not in ("", ".", "..")]
    return "/".join(safe_parts)


def _manifest_path_for_file(manifest: list[dict[str, Any]], index: int, filename: str) -> str:
    if index < len(manifest):
        path = manifest[index].get("relative_path") or manifest[index].get("filename")
        if path:
            return _safe_zip_path(path)
    return _safe_zip_path(filename)


def read_spreadsheet_summary(spreadsheet_file) -> SpreadsheetSummary:
    summary = SpreadsheetSummary(filename=spreadsheet_file.name)
    suffix = PurePosixPath(spreadsheet_file.name).suffix.lower()
    if suffix not in SPREADSHEET_SUFFIXES:
        raise ValueError("Only CSV, XLS and XLSX spreadsheets are supported.")

    spreadsheet_file.seek(0)
    if suffix == ".csv":
        df = pd.read_csv(spreadsheet_file, encoding="utf-8-sig", nrows=20)
    else:
        df = pd.read_excel(spreadsheet_file, nrows=20)
    spreadsheet_file.seek(0)

    summary.columns = [str(column) for column in df.columns]
    summary.normalized_columns = [COLUMN_ALIASES.get(column.strip().lower(), column) for column in summary.columns]
    if "original_path" not in summary.normalized_columns:
        summary.warnings.append("Spreadsheet has no recognized media path column.")
    return summary


def split_upload_files(upload_files):
    media_files = []
    spreadsheet_file = None
    for upload_file in upload_files:
        suffix = PurePosixPath(upload_file.name).suffix.lower()
        if suffix in SPREADSHEET_SUFFIXES and spreadsheet_file is None:
            spreadsheet_file = upload_file
        else:
            media_files.append(upload_file)
    return media_files, spreadsheet_file


def parse_json_mapping(raw_mapping: str) -> dict[str, Any]:
    if not raw_mapping:
        return {}
    try:
        data = json.loads(raw_mapping)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_directory_structure(structure: str) -> list[str]:
    if not structure:
        return []
    return [part.strip("{}") for part in structure.strip("/").split("/") if part]


def validate_directory_mapping(relative_paths: list[str], structure: str) -> list[str]:
    if not structure:
        return []
    parts = parse_directory_structure(structure)
    if not parts:
        return []
    warnings = []
    for path in relative_paths:
        path_parts = [part for part in _safe_zip_path(path).split("/") if part]
        directory_parts = path_parts[:-1]
        if len(directory_parts) < len(parts):
            warnings.append(f"Path '{path}' is too shallow for '{structure}'.")
            if len(warnings) >= 5:
                warnings.append("Further directory mapping warnings omitted.")
                break
    return warnings


def build_archive_name(has_single_zip_upload: bool, upload_files, relative_paths: list[str]) -> str:
    timestamp = timezone.now().strftime("%Y%m%d-%H%M%S")
    fallback_name = f"upload_{timestamp}"

    if has_single_zip_upload and upload_files:
        stem = PurePosixPath(upload_files[0].name).stem.strip()
        return stem or fallback_name

    top_level_parts = []
    for path in relative_paths:
        safe_path = _safe_zip_path(path)
        path_parts = [part for part in safe_path.split("/") if part]
        if len(path_parts) > 1:
            top_level_parts.append(path_parts[0])

    unique_top_level_parts = {part for part in top_level_parts if part}
    if len(unique_top_level_parts) == 1:
        return next(iter(unique_top_level_parts))

    return fallback_name


def build_path_regex_from_directory_mapping(directory_mapping: dict[str, Any]) -> str:
    if not directory_mapping:
        return ""

    normalized_mapping: dict[str, int] = {}
    for role, raw_position in directory_mapping.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        if position < 0:
            continue
        normalized_mapping[str(role)] = position

    if not normalized_mapping:
        return ""

    role_patterns = {
        "check_date": r"(?P<check_date>\d{4}-?\d{2}-?\d{2})",
        "locality": r"(?P<locality>[^/]+)",
        "taxon": r"(?P<taxon>[^/]+)",
        "identity": r"(?P<identity>[^/]+)",
    }
    max_position = max(normalized_mapping.values())
    parts = []
    for position in range(max_position + 1):
        role = next((name for name, mapped_position in normalized_mapping.items() if mapped_position == position), None)
        parts.append(role_patterns.get(role, r"[^/]+"))
    return "^" + "/".join(parts) + r"/[^/]+$"


def build_upload_zip(
    upload_files,
    spreadsheet_file=None,
    relative_path_manifest=None,
    directory_structure="",
    directory_mapping=None,
    path_regex="",
    spreadsheet_column_mapping=None,
):
    upload_files = list(upload_files)
    detected_media_files, detected_spreadsheet = split_upload_files(upload_files)
    if spreadsheet_file is None:
        spreadsheet_file = detected_spreadsheet
        upload_files = detected_media_files

    if not upload_files:
        raise ValueError("Upload must contain a ZIP archive or at least one media file.")

    relative_path_manifest = relative_path_manifest or []
    has_single_zip_upload = len(upload_files) == 1 and PurePosixPath(upload_files[0].name).suffix.lower() in ARCHIVE_SUFFIXES
    relative_paths = [
        _manifest_path_for_file(relative_path_manifest, index, upload_file.name)
        for index, upload_file in enumerate(upload_files)
    ]
    directory_mapping = directory_mapping or {}
    if not path_regex:
        path_regex = build_path_regex_from_directory_mapping(directory_mapping)
    spreadsheet_column_mapping = spreadsheet_column_mapping or {}
    directory_warnings = [] if has_single_zip_upload else validate_directory_mapping(relative_paths, directory_structure)
    if directory_structure and not has_single_zip_upload and not any("/" in path for path in relative_paths):
        directory_warnings.append("Directory structure was selected, but no relative paths were detected.")

    spreadsheet_summary = SpreadsheetSummary()
    if spreadsheet_file:
        spreadsheet_summary = read_spreadsheet_summary(spreadsheet_file)

    import_mapping = {
        "spreadsheet": {
            "filename": spreadsheet_summary.filename,
            "columns": spreadsheet_summary.columns,
            "normalized_columns": spreadsheet_summary.normalized_columns,
            "column_mapping": spreadsheet_column_mapping,
        },
        "directory_structure": directory_structure,
        "directory_mapping": directory_mapping,
        "path_regex": path_regex,
        "path_source": "archive_path" if has_single_zip_upload else ("relative_path" if any("/" in path for path in relative_paths) else "filename"),
    }
    log_messages = [*directory_warnings, *spreadsheet_summary.warnings]

    if (
        has_single_zip_upload
        and spreadsheet_file is None
    ):
        archive_name = build_archive_name(has_single_zip_upload, upload_files, relative_paths)
        return ZipBuildResult(
            file=upload_files[0],
            filename=upload_files[0].name,
            archive_name=archive_name,
            import_mapping=import_mapping,
            import_log="\n".join(log_messages),
            spreadsheet_summary=spreadsheet_summary,
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        if has_single_zip_upload:
            upload_files[0].seek(0)
            with zipfile.ZipFile(upload_files[0], "r") as source_zip:
                for name in source_zip.namelist():
                    zip_file.writestr(name, source_zip.read(name))
            upload_files[0].seek(0)
        else:
            for index, upload_file in enumerate(upload_files):
                upload_file.seek(0)
                zip_file.writestr(relative_paths[index], upload_file.read())
                upload_file.seek(0)

        if spreadsheet_file:
            spreadsheet_file.seek(0)
            zip_file.writestr(_safe_zip_path(spreadsheet_file.name), spreadsheet_file.read())
            spreadsheet_file.seek(0)

    buffer.seek(0)
    archive_name = build_archive_name(has_single_zip_upload, upload_files, relative_paths)
    filename = f"{archive_name}.zip"
    return ZipBuildResult(
        file=ContentFile(buffer.read(), name=filename),
        filename=filename,
        archive_name=archive_name,
        import_mapping=import_mapping,
        import_log="\n".join(log_messages),
        spreadsheet_summary=spreadsheet_summary,
    )
