"""Detection-only task: never runs classification or modifies observations."""

import logging
from pathlib import Path
from time import monotonic

import cv2

logger = logging.getLogger("app")


def detect_file(relative_path, options, *, task_id=None):
    from detection_utils import inference_detection as detection

    root = Path("/shared_data/media").resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path is outside the media directory")
    logger.info("BBox source read started task_id=%s source=%s", task_id, path.name)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Cannot read the image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    logger.info("BBox source read finished task_id=%s width=%d height=%d", task_id, width, height)
    backend = options["detector"]
    if backend not in {"auto", "megadetector", "sam3"}:
        raise ValueError("Unknown detector")
    used = "megadetector"
    if backend == "sam3":
        started = monotonic()
        logger.info("SAM3 detection started task_id=%s mode=forced", task_id)
        detections = detection.detect_animals_with_sam3(image, strict=True) or []
        used = "sam3"
        logger.info(
            "SAM3 detection finished task_id=%s detections=%d duration_s=%.2f",
            task_id,
            len(detections),
            monotonic() - started,
        )
    else:
        started = monotonic()
        logger.info("MegaDetector started task_id=%s", task_id)
        detections = detection.detect_animals_with_megadetector(image) or []
        # This action detects animals only; people/vehicles must not suppress the fallback.
        animals = [d for d in detections if d["class"] == "animal"]
        max_confidence = max((d["confidence"] for d in animals), default=0)
        logger.info(
            "MegaDetector finished task_id=%s detections=%d animal_detections=%d max_animal_confidence=%.3f duration_s=%.2f",
            task_id,
            len(detections),
            len(animals),
            max_confidence,
            monotonic() - started,
        )
        if backend == "auto" and max_confidence < detection.SAM3_FALLBACK_CONF:
            started = monotonic()
            logger.info(
                "SAM3 fallback started task_id=%s reason=max_animal_confidence_below_threshold threshold=%.3f",
                task_id,
                detection.SAM3_FALLBACK_CONF,
            )
            detections = detection.detect_animals_with_sam3(image, strict=True) or []
            used = "sam3"
            logger.info(
                "SAM3 fallback finished task_id=%s detections=%d duration_s=%.2f",
                task_id,
                len(detections),
                monotonic() - started,
            )
    results = []
    for entry in detections:
        if entry["class"] != "animal" or entry["confidence"] < options["confidence"]:
            continue
        x0, y0, x1, y1 = entry["bbox"]
        box = [max(0, min(1, float(v) / scale)) for v, scale in zip((x0, y0, x1, y1), (width, height, width, height))]
        if box[0] < box[2] and box[1] < box[3]:
            results.append({"bbox": box, "confidence": float(entry["confidence"])})
    logger.info(
        "BBox detections normalized task_id=%s detector=%s detections=%d confidence_threshold=%.3f",
        task_id,
        used,
        len(results),
        options["confidence"],
    )
    return {"status": "ok", "detector": used, "detections": results}
