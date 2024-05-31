import logging

logger = logging.getLogger(__name__)


class StatusCounts:
    def __init__(self):
        self.counts = {}

    def increment(self, status: str):
        if status not in self.counts:
            self.counts[status] = 0
        self.counts[status] += 1

    def __str__(self):
        return str(self.counts)