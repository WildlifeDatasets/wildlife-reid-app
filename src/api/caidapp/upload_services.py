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

SPREADSHEET_SUFFIXES = {".csv", ".xlsx"}
ARCHIVE_SUFFIXES = {".zip"}
NORMALIZED_SPREADSHEET_FILENAME = "mediafile.post_update.csv"

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
    "code": "code",
    "juv_code": "juv_code",
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


@dataclass
class ZipSpreadsheetSource:
    filename: str
    bytes: bytes


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


def _read_spreadsheet_dataframe(spreadsheet_file, filename: str, nrows: int | None = 20) -> pd.DataFrame:
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix not in SPREADSHEET_SUFFIXES:
        raise ValueError("Only CSV and XLSX spreadsheets are supported.")
    if suffix == ".csv":
        return pd.read_csv(spreadsheet_file, encoding="utf-8-sig", nrows=nrows)
    return pd.read_excel(spreadsheet_file, nrows=nrows)


def _build_spreadsheet_summary(filename: str, df: pd.DataFrame) -> SpreadsheetSummary:
    summary = SpreadsheetSummary(filename=filename)
    summary.columns = [str(column) for column in df.columns]
    summary.normalized_columns = [COLUMN_ALIASES.get(column.strip().lower(), column) for column in summary.columns]
    if "original_path" not in summary.normalized_columns:
        summary.warnings.append("Spreadsheet has no recognized media path column.")
    return summary


def read_spreadsheet_summary(spreadsheet_file) -> SpreadsheetSummary:
    spreadsheet_file.seek(0)
    df = _read_spreadsheet_dataframe(spreadsheet_file, spreadsheet_file.name, nrows=20)
    spreadsheet_file.seek(0)
    return _build_spreadsheet_summary(spreadsheet_file.name, df)


def _normalize_spreadsheet_columns(df: pd.DataFrame, spreadsheet_column_mapping: dict[str, Any]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        column_name = str(column)
        normalized_name = COLUMN_ALIASES.get(column_name.strip().lower())
        if normalized_name:
            rename_map[column_name] = normalized_name

    for target_name, source_name in spreadsheet_column_mapping.items():
        if source_name:
            rename_map[str(source_name)] = str(target_name)

    if not rename_map:
        return df.copy()
    return df.rename(columns=rename_map)


def _apply_path_adjustment(value: Any, spreadsheet_path_adjustment: dict[str, Any]) -> Any:
    if value is None or pd.isna(value):
        return value
    normalized_value = str(value).replace("\\", "/").strip()
    remove_prefix = str(spreadsheet_path_adjustment.get("remove_prefix") or "").replace("\\", "/")
    add_prefix = str(spreadsheet_path_adjustment.get("add_prefix") or "").replace("\\", "/")

    if remove_prefix:
        normalized_remove_prefix = remove_prefix.strip("/")
        if normalized_value == normalized_remove_prefix:
            normalized_value = ""
        elif normalized_value.startswith(f"{normalized_remove_prefix}/"):
            normalized_value = normalized_value[len(normalized_remove_prefix) + 1 :]

    if add_prefix:
        normalized_add_prefix = add_prefix.strip("/")
        if normalized_value:
            normalized_value = f"{normalized_add_prefix}/{normalized_value}"
        else:
            normalized_value = normalized_add_prefix

    return normalized_value


def _apply_spreadsheet_path_adjustment(
    df: pd.DataFrame,
    spreadsheet_path_adjustment: dict[str, Any],
) -> pd.DataFrame:
    if "original_path" not in df.columns:
        return df
    if not spreadsheet_path_adjustment:
        return df
    adjusted_df = df.copy()
    adjusted_df["original_path"] = adjusted_df["original_path"].apply(
        lambda value: _apply_path_adjustment(value, spreadsheet_path_adjustment)
    )
    return adjusted_df


def _build_normalized_spreadsheet_csv(
    spreadsheet_file,
    filename: str,
    spreadsheet_column_mapping: dict[str, Any],
    spreadsheet_path_adjustment: dict[str, Any],
) -> bytes:
    spreadsheet_file.seek(0)
    df = _read_spreadsheet_dataframe(spreadsheet_file, filename, nrows=None)
    spreadsheet_file.seek(0)
    normalized_df = _normalize_spreadsheet_columns(df, spreadsheet_column_mapping)
    normalized_df = _apply_spreadsheet_path_adjustment(normalized_df, spreadsheet_path_adjustment)
    buffer = io.StringIO()
    normalized_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")


def split_upload_files(upload_files):
    media_files = []
    spreadsheet_file = None
    for upload_file in upload_files:
        suffix = PurePosixPath(upload_file.name).suffix.lower()
        if suffix == ".xls":
            raise ValueError("XLS spreadsheets are no longer supported. Please convert them to XLSX or CSV.")
        if suffix in SPREADSHEET_SUFFIXES and spreadsheet_file is None:
            spreadsheet_file = upload_file
        else:
            media_files.append(upload_file)
    return media_files, spreadsheet_file


def _preferred_zip_spreadsheet_name(names: list[str]) -> str | None:
    normalized_candidates = [
        name for name in names if PurePosixPath(name).name == NORMALIZED_SPREADSHEET_FILENAME
    ]
    if normalized_candidates:
        return normalized_candidates[0]
    for suffix in (".csv", ".xlsx"):
        for name in names:
            if PurePosixPath(name).suffix.lower() == suffix:
                return name
    return None


def read_zip_spreadsheet_source(upload_file) -> ZipSpreadsheetSource | None:
    upload_file.seek(0)
    with zipfile.ZipFile(upload_file, "r") as source_zip:
        names = [
            name
            for name in source_zip.namelist()
            if PurePosixPath(name).suffix.lower() in SPREADSHEET_SUFFIXES
        ]
        preferred_name = _preferred_zip_spreadsheet_name(names)
        if preferred_name is None:
            upload_file.seek(0)
            return None
        spreadsheet_bytes = source_zip.read(preferred_name)
    upload_file.seek(0)
    return ZipSpreadsheetSource(filename=preferred_name, bytes=spreadsheet_bytes)


def read_zip_spreadsheet_summary(upload_file) -> SpreadsheetSummary:
    source = read_zip_spreadsheet_source(upload_file)
    if source is None:
        return SpreadsheetSummary()
    df = _read_spreadsheet_dataframe(io.BytesIO(source.bytes), source.filename, nrows=20)
    return _build_spreadsheet_summary(source.filename, df)


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
        "unique_name": r"(?P<unique_name>[^/]+)",
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
    spreadsheet_path_adjustment=None,
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
    spreadsheet_path_adjustment = spreadsheet_path_adjustment or {}
    directory_warnings = [] if has_single_zip_upload else validate_directory_mapping(relative_paths, directory_structure)
    if directory_structure and not has_single_zip_upload and not any("/" in path for path in relative_paths):
        directory_warnings.append("Directory structure was selected, but no relative paths were detected.")

    spreadsheet_summary = SpreadsheetSummary()
    zip_spreadsheet_source = None
    if spreadsheet_file:
        spreadsheet_summary = read_spreadsheet_summary(spreadsheet_file)
    elif has_single_zip_upload:
        spreadsheet_summary = read_zip_spreadsheet_summary(upload_files[0])
        zip_spreadsheet_source = read_zip_spreadsheet_source(upload_files[0])

    normalized_spreadsheet_bytes = None
    if spreadsheet_file:
        normalized_spreadsheet_bytes = _build_normalized_spreadsheet_csv(
            spreadsheet_file,
            spreadsheet_file.name,
            spreadsheet_column_mapping,
            spreadsheet_path_adjustment,
        )
    elif zip_spreadsheet_source is not None:
        normalized_spreadsheet_bytes = _build_normalized_spreadsheet_csv(
            io.BytesIO(zip_spreadsheet_source.bytes),
            zip_spreadsheet_source.filename,
            spreadsheet_column_mapping,
            spreadsheet_path_adjustment,
        )

    import_mapping = {
        "spreadsheet": {
            "filename": spreadsheet_summary.filename,
            "columns": spreadsheet_summary.columns,
            "normalized_columns": spreadsheet_summary.normalized_columns,
            "column_mapping": spreadsheet_column_mapping,
            "path_adjustment": spreadsheet_path_adjustment,
            "normalized_csv_filename": NORMALIZED_SPREADSHEET_FILENAME if normalized_spreadsheet_bytes else "",
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
        and normalized_spreadsheet_bytes is None
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
                    if normalized_spreadsheet_bytes and PurePosixPath(name).name == NORMALIZED_SPREADSHEET_FILENAME:
                        continue
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

        if normalized_spreadsheet_bytes:
            zip_file.writestr(NORMALIZED_SPREADSHEET_FILENAME, normalized_spreadsheet_bytes)

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
