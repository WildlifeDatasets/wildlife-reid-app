import logging
import time


logger = logging.getLogger(__name__)


class ProgressReporter:
    """Map progress within pipeline stages to throttled Celery updates."""

    INIT_STAGE_WEIGHTS = (
        ("prepare_metadata", 30),
        ("video_previews", 15),
        ("detection", 20),
        ("classification", 15),
        ("previews", 10),
        ("finalize", 10),
    )
    RESUME_STAGE_WEIGHTS = (
        ("load_metadata", 5),
        ("detection", 30),
        ("classification", 20),
        ("previews", 25),
        ("finalize", 20),
    )

    def __init__(self, task, do_init: bool):
        weights = self.INIT_STAGE_WEIGHTS if do_init else self.RESUME_STAGE_WEIGHTS
        total_weight = sum(weight for _, weight in weights)
        offset = 0.0
        self.ranges = {}
        for name, weight in weights:
            width = 99.0 * weight / total_weight
            self.ranges[name] = (offset, offset + width)
            offset += width
        self.task = task
        self.stage_name = weights[0][0]
        self.message = "Starting processing"
        self.last_percent = -1
        self.last_update_at = 0.0

    def stage(self, name: str, message: str):
        self.stage_name = name
        self.message = message
        self.update(0, 1, force=True)

    def callback(self, message: str):
        return lambda completed, total: self.update(completed, total, message=message)

    def update(self, completed, total, message: str = None, force: bool = False):
        start, end = self.ranges[self.stage_name]
        fraction = 0.0 if not total else min(max(float(completed) / float(total), 0.0), 1.0)
        percent = min(int(start + ((end - start) * fraction)), 99)
        now = time.monotonic()
        if not force and percent < self.last_percent + 1 and now - self.last_update_at < 1.0:
            return
        self.message = message or self.message
        try:
            self.task.update_state(
                state="PROGRESS",
                meta={
                    "percent": max(percent, self.last_percent),
                    "stage": self.stage_name,
                    "message": self.message,
                },
            )
        except Exception:
            logger.warning("Could not publish taxon worker progress", exc_info=True)
        self.last_percent = max(percent, self.last_percent)
        self.last_update_at = now
