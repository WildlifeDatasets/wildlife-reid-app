"""Read and write ZIP file times without changing media bytes."""

from datetime import datetime, timedelta, timezone
from struct import pack, unpack_from
from zipfile import ZipInfo

NTFS_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _fields(data):
    while len(data) >= 4:
        tag, size = unpack_from("<HH", data)
        if len(data) < size + 4:
            break
        yield tag, data[4 : size + 4]
        data = data[size + 4 :]


def _local_extra(stream, info):
    offset = stream.tell()
    try:
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            return b""
        name_size, extra_size = unpack_from("<HH", header, 26)
        stream.seek(name_size, 1)
        return stream.read(extra_size)
    finally:
        stream.seek(offset)


def read_zip_file_times(stream, info):
    """Read modification and creation times from central and local ZIP headers."""
    extended = {}
    ntfs = {}
    for extra in (info.extra, _local_extra(stream, info)):
        for tag, data in _fields(extra):
            if tag == 0x5455 and data:
                offset = 1
                for bit, field in ((1, "modified"), (2, None), (4, "created")):
                    if data[0] & bit:
                        if offset + 4 > len(data):
                            break
                        seconds = unpack_from("<i", data, offset)[0]
                        if field:
                            value = UNIX_EPOCH + timedelta(seconds=seconds)
                            extended[field] = {"value": value.isoformat(), "source": "zip_extended"}
                        offset += 4
            if tag == 0x000A:
                for marker, values in _fields(data[4:]):
                    if marker != 1 or len(values) < 24:
                        continue
                    modified, _, created = unpack_from("<QQQ", values)
                    for field, ticks in (("modified", modified), ("created", created)):
                        if ticks:
                            try:
                                value = NTFS_EPOCH + timedelta(microseconds=ticks // 10)
                            except OverflowError:
                                continue
                            ntfs[field] = {"value": value.isoformat(), "source": "zip_ntfs", "filetime": ticks}
    try:
        wall_time = datetime(*info.date_time).isoformat()
    except ValueError:
        wall_time = None
    fallback = {"modified": {"value": wall_time, "source": "zip_dos"}} if wall_time else {}
    times = {**fallback, **extended, **ntfs}
    if wall_time and times["modified"]["source"] != "zip_dos":
        times["zip_dos_modified"] = wall_time
    return times


def zip_info_with_times(path, times):
    """Write local ZIP time and available absolute times into an export entry."""
    modified = times.get("modified", {})
    created = times.get("created", {})
    header_time = times.get("zip_dos_modified") or modified.get("value")
    dt = datetime.fromisoformat(header_time) if header_time else None
    info = ZipInfo(path)
    if dt is not None and 1980 <= dt.year <= 2107:
        info.date_time = dt.timetuple()[:6]

    def filetime(entry):
        if not entry:
            return 0
        if "filetime" in entry:
            return entry["filetime"]
        aware = datetime.fromisoformat(entry["value"])
        if aware.tzinfo is None:
            return 0
        delta = aware - NTFS_EPOCH
        return max(0, ((delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds) * 10)

    modified_ticks, created_ticks = filetime(modified), filetime(created)
    if modified_ticks or created_ticks:
        info.extra = pack("<HHIHHQQQ", 0x000A, 32, 0, 1, 24, modified_ticks, 0, created_ticks)
    return info
