"""Periodically report GPU availability from the identification worker."""

import logging
import threading

import torch
from celery import signals


logger = logging.getLogger("app")

GPU_HEARTBEAT_INTERVAL_SECONDS = 5 * 60
GPU_HEARTBEAT_TASK = "caidapp.tasks.record_identification_worker_gpu_heartbeat"

_monitor_thread: threading.Thread | None = None
_stop_event = threading.Event()


def collect_gpu_heartbeat() -> dict:
    """Return a CUDA runtime health sample without letting diagnostics crash a task."""
    device = "cuda:0"
    sample = {
        "device": device,
        "available": False,
        "device_name": "",
        "free_memory_gb": None,
        "total_memory_gb": None,
        "error_message": "",
    }
    try:
        if not torch.cuda.is_available():
            sample["error_message"] = "torch.cuda.is_available() returned False"
            return sample

        free_memory, total_memory = torch.cuda.mem_get_info(device)
        sample.update(
            available=True,
            device_name=torch.cuda.get_device_name(device),
            free_memory_gb=round(free_memory / 1024**3, 3),
            total_memory_gb=round(total_memory / 1024**3, 3),
        )
    except Exception as exc:  # CUDA failures may use several RuntimeError subclasses.
        sample["error_message"] = f"{type(exc).__name__}: {exc}"
    return sample


def _publish_heartbeat(celery_app) -> None:
    sample = collect_gpu_heartbeat()
    try:
        celery_app.send_task(GPU_HEARTBEAT_TASK, kwargs=sample)
    except Exception:
        # Identification processing must continue if the API worker or broker is unavailable.
        logger.exception("Could not publish identification-worker GPU heartbeat")
        return

    if sample["available"]:
        logger.debug(
            "Identification-worker GPU heartbeat: %s (%s/%s GiB free)",
            sample["device"],
            sample["free_memory_gb"],
            sample["total_memory_gb"],
        )
    else:
        logger.warning("Identification-worker GPU heartbeat failed: %s", sample["error_message"])


def start_gpu_monitor(celery_app) -> None:
    """Start one daemon thread after the Celery worker is ready."""
    global _monitor_thread
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return

    _stop_event.clear()

    def monitor() -> None:
        while not _stop_event.is_set():
            _publish_heartbeat(celery_app)
            _stop_event.wait(GPU_HEARTBEAT_INTERVAL_SECONDS)

    _monitor_thread = threading.Thread(target=monitor, name="identification-gpu-monitor", daemon=True)
    _monitor_thread.start()


def stop_gpu_monitor() -> None:
    _stop_event.set()


def register_gpu_monitor(celery_app) -> None:
    """Attach monitor lifecycle to one Celery application."""

    @signals.worker_ready.connect(weak=False)
    def _start_monitor(**_kwargs) -> None:
        start_gpu_monitor(celery_app)

    @signals.worker_shutdown.connect(weak=False)
    def _stop_monitor(**_kwargs) -> None:
        stop_gpu_monitor()
