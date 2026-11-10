"""Bounded duplicate scans; no imports from the classification or detection pipeline."""

import hashlib
import itertools
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

VERSION = "duplicates-v1"
MAX_FILES = 100
MAX_PIXELS = 40_000_000


def file_hash(path):
    """Hash the actual media file, including videos, without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path):
    """Canonical v1 pixels: EXIF orientation, lossless 8-bit RGBA expansion, no ICC conversion.

    Higher-bit-depth and animated images are deliberately unsupported, rather than
    being silently reduced to an allegedly exact 8-bit or single-frame match.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF"}:
                raise ValueError("Unsupported pixel-comparison image format")
            if image.format == "PNG":
                with open(path, "rb") as stream:
                    header = stream.read(25)
                if len(header) != 25 or header[24] > 8:
                    raise ValueError("High-bit-depth PNG pixel comparison is unsupported")
            if image.format == "TIFF":
                bits = image.tag_v2.get(258, (8,))
                bits = (bits,) if isinstance(bits, int) else bits
                if max(bits) > 8:
                    raise ValueError("High-bit-depth TIFF pixel comparison is unsupported")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 40 megapixel scan limit")
            if getattr(image, "n_frames", 1) != 1 or image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                raise ValueError("Pixel comparison supports single-frame 8-bit images only")
            return np.asarray(ImageOps.exif_transpose(image).convert("RGBA"))


def pixel_hash(pixels):
    """Include canonicalization version, dimensions, and channels in the digest."""
    header = json.dumps([VERSION, pixels.shape, str(pixels.dtype)]).encode()
    return hashlib.sha256(header + pixels.tobytes()).hexdigest()


def features(pixels):
    """Keep a bounded SIFT representation, with coordinates in the feature raster."""
    height, width = pixels.shape[:2]
    scale = min(1.0, 1200 / max(height, width))
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)
    gray = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))))
    keypoints, descriptors = cv2.SIFT_create(nfeatures=1600).detectAndCompute(gray, None)
    return {
        "points": np.float32([point.pt for point in keypoints]).reshape(-1, 2),
        "descriptors": descriptors,
        "size": [gray.shape[1], gray.shape[0]],
    }


def crop_match(query, source, query_path, source_path):
    """Find an axis-aligned crop, allowing uniform resize and recompression.

    RANSAC alone is not evidence of a copy: verify dense, local image differences
    too, to reject matching camera-trap backgrounds with different animals.
    """
    qdesc, sdesc = query["descriptors"], source["descriptors"]
    if qdesc is None or sdesc is None or min(len(qdesc), len(sdesc)) < 12:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(qdesc, sdesc, k=2)
    reverse = matcher.match(sdesc, qdesc)
    mutual = {match.queryIdx: match.trainIdx for match in reverse}
    good = [
        pair[0]
        for pair in forward
        if len(pair) == 2
        and pair[0].distance < 0.7 * pair[1].distance
        and mutual.get(pair[0].trainIdx) == pair[0].queryIdx
    ]
    if len(good) < 12:
        return None
    qpoints = np.float32([query["points"][match.queryIdx] for match in good])
    spoints = np.float32([source["points"][match.trainIdx] for match in good])
    matrix, inliers = cv2.estimateAffinePartial2D(qpoints, spoints, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None or not np.isfinite(matrix).all():
        return None
    count = int(inliers.sum())
    if count < 12 or count / len(good) < 0.7:
        return None
    scale = matrix[0, 0]
    if scale <= 0 or abs(matrix[0, 1]) > scale * 0.005 or abs(matrix[1, 0]) > scale * 0.005:
        return None
    qw, qh = query["size"]
    sw, sh = source["size"]
    normalized = np.diag([1 / sw, 1 / sh, 1]) @ np.vstack([matrix, [0, 0, 1]]) @ np.diag([qw, qh, 1])
    corners = np.float32([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) @ normalized.T
    if corners[:, :2].min() < -0.003 or corners[:, :2].max() > 1.003:
        return None
    hull = cv2.convexHull(qpoints[inliers.ravel().astype(bool)])
    coverage = float(cv2.contourArea(hull) / (qw * qh))
    if coverage < 0.15:
        return None
    # Verify on a separate, finer raster, not just the matched keypoints.
    query_pixels, source_pixels = read_image(query_path), read_image(source_path)
    h, w = query_pixels.shape[:2]
    factor = min(1.0, 1600 / max(h, w))
    vw, vh = max(1, round(w * factor)), max(1, round(h * factor))
    target = cv2.resize(query_pixels, (vw, vh), interpolation=cv2.INTER_AREA)
    real_transform = (
        np.diag([source_pixels.shape[1], source_pixels.shape[0], 1]) @ normalized @ np.diag([1 / vw, 1 / vh, 1])
    )
    aligned = cv2.warpAffine(
        source_pixels,
        real_transform[:2],
        (vw, vh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    difference = np.abs(target.astype(np.float32) - aligned.astype(np.float32)) / 255
    error = difference.max(axis=2)
    tiles = [
        float(tile.mean())
        for row in np.array_split(error, min(12, vh))
        for tile in np.array_split(row, min(12, vw), axis=1)
    ]
    mean_error = float(error.mean())
    changed_fraction = float((error > 0.15).mean())
    if mean_error > 0.025 or max(tiles) > 0.065 or changed_fraction > 0.01:
        return None
    return {
        "left_to_right": normalized.tolist(),
        "polygon": corners[:, :2].tolist(),
        "inliers": count,
        "coverage": coverage,
        "mean_error": mean_error,
        "max_tile_error": max(tiles),
        "changed_fraction": changed_fraction,
    }


def scan(items, media_root, include_crops=True, progress=None):
    """Exhaustive bounded MVP; report unsupported inputs and never mutate media files."""
    if not 2 <= len(items) <= MAX_FILES:
        raise ValueError(f"Select between 2 and {MAX_FILES} media files")
    root = Path(media_root).resolve()
    records, paths, descriptors = [], {}, {}
    for index, item in enumerate(items):
        record = {"id": item["id"], "name": item["name"], "media_type": item["media_type"]}
        try:
            path = (root / item["source"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Media file is unavailable")
            before = path.stat()
            record["sha256"] = file_hash(path)
            record["source"] = item["source"]
            record["stat"] = [before.st_size, before.st_mtime_ns]
            paths[item["id"]] = path
            if item["media_type"] == "image":
                try:
                    pixels = read_image(path)
                    record["pixel_hash"] = pixel_hash(pixels)
                    record["size"] = [pixels.shape[1], pixels.shape[0]]
                    if include_crops:
                        descriptors[item["id"]] = features(pixels)
                except (
                    OSError,
                    ValueError,
                    cv2.error,
                    Image.DecompressionBombError,
                    Image.DecompressionBombWarning,
                ) as exc:
                    record["pixel_error"] = str(exc)
            after = path.stat()
            if [after.st_size, after.st_mtime_ns] != record["stat"]:
                raise ValueError("Media file changed during scanning")
        except (OSError, ValueError) as exc:
            record["error"] = str(exc)
            descriptors.pop(item["id"], None)
        records.append(record)
        if progress:
            progress("fingerprints", index + 1, len(items))
    valid = [record for record in records if "error" not in record]
    # Hash buckets produce equivalence groups without materializing O(n²) exact pairs.
    groups, visited = [], set()
    for field, kind in [("sha256", "byte_exact"), ("pixel_hash", "pixel_exact")]:
        buckets = {}
        for record in valid:
            if record.get(field):
                buckets.setdefault(record[field], []).append(record["id"])
        for members in buckets.values():
            if len(members) > 1:
                groups.append({"kind": kind, "members": members, "evidence": {"hash": field}})
                visited.update(itertools.combinations(sorted(members), 2))
    # Prefer complete pixel groups over byte subgroups contained in them.
    groups = [
        group
        for group in groups
        if group["kind"] != "byte_exact"
        or not any(
            other["kind"] == "pixel_exact" and set(group["members"]).issubset(other["members"]) for other in groups
        )
    ]
    pairs = list(itertools.combinations(sorted(descriptors), 2))
    for index, (left, right) in enumerate(pairs):
        if (left, right) not in visited:
            for query, source in [(left, right), (right, left)]:
                evidence = crop_match(descriptors[query], descriptors[source], paths[query], paths[source])
                if evidence:
                    evidence["source_id"] = source
                    evidence["crop_id"] = query
                    groups.append({"kind": "crop", "members": [query, source], "evidence": evidence})
                    break
        if progress:
            progress("crops", index + 1, len(pairs))
    return {"version": VERSION, "items": records, "proposals": groups}
