"""Dedicated entrypoint: never import worker.py or classification models."""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
from celery import Celery
from duplicates.engine import scan
from duplicates.fingerprints import HASH_VERSION, hash_files

duplicate_worker = Celery("duplicate_worker", broker=os.environ["RABBITMQ_URL"], backend=os.environ["REDIS_URL"])
duplicate_worker.conf.worker_prefetch_multiplier = 1
SHARED_ROOT = Path(os.environ.get("SHARED_DATA_PATH", "/shared_data"))
ROOT = SHARED_ROOT / "media"
logger = logging.getLogger(__name__)


@duplicate_worker.task(
    bind=True,
    name="duplicates.hash_files",
    soft_time_limit=3500,
    time_limit=3600,
    acks_late=True,
    reject_on_worker_lost=True,
)
def hash_media_files(self, batch_uuid):
    """Compute a bounded batch of exact hashes, independent of ReID initialization and crop search."""
    cv2.setNumThreads(1)
    batch_uuid = str(uuid.UUID(batch_uuid))
    directory = (
        Path(os.environ.get("DUPLICATE_SCAN_ROOT", str(SHARED_ROOT / "duplicate_scans"))) / "hash_batches" / batch_uuid
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    result_path = directory / "result.json"
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("version") == HASH_VERSION and [item["id"] for item in cached["items"]] == [
            item["id"] for item in manifest["items"]
        ]:
            logger.info("Reusing completed exact hash batch=%s; replaying callback only", batch_uuid)
            return {"batch_uuid": batch_uuid}
    completed = 0
    stopped = threading.Event()

    def heartbeat():
        temporary = directory / f"progress-{os.getpid()}-{threading.get_ident()}.tmp"
        temporary.write_text(json.dumps({"updated_at": time.time(), "completed": completed}), encoding="utf-8")
        temporary.replace(directory / "progress.json")

    def keep_alive():
        while not stopped.wait(15):
            try:
                heartbeat()
            except OSError:
                logger.exception("Could not write exact hash heartbeat batch=%s", batch_uuid)

    def progress(current, total):
        nonlocal completed
        completed = current
        heartbeat()
        if current == 1 or current % 10 == 0 or current == total:
            logger.info("Exact hash batch=%s progress=%s/%s", batch_uuid, current, total)

    logger.info("Starting exact hash batch=%s files=%s", batch_uuid, len(manifest["items"]))
    heartbeat()
    heartbeat_thread = threading.Thread(target=keep_alive, daemon=True)
    heartbeat_thread.start()
    try:
        result = hash_files(manifest["items"], ROOT, progress=progress)
        temporary = directory / f"result-{os.getpid()}-{uuid.uuid4()}.tmp"
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(result_path)
    finally:
        stopped.set()
        heartbeat_thread.join(timeout=2)
    logger.info("Finished exact hash batch=%s files=%s", batch_uuid, len(manifest["items"]))
    return {"batch_uuid": batch_uuid}


@duplicate_worker.task(bind=True, name="duplicates.scan", soft_time_limit=3500, time_limit=3600)
def scan_duplicates(self, scan_uuid):
    """Read a server-created manifest and atomically publish a result on shared storage."""
    cv2.setNumThreads(1)
    scan_uuid = str(uuid.UUID(scan_uuid))
    directory = Path(os.environ.get("DUPLICATE_SCAN_ROOT", str(SHARED_ROOT / "duplicate_scans"))) / scan_uuid
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    def progress(stage, current, total):
        self.update_state(state="PROGRESS", meta={"stage": stage, "current": current, "total": total})

    result = scan(manifest["items"], ROOT, include_crops=manifest["include_crops"], progress=progress)
    temporary = directory / f"result-{self.request.id}.tmp"
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(directory / "result.json")
    return {"scan_uuid": scan_uuid}
