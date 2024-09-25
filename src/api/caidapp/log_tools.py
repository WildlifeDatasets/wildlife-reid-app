import logging

logger = logging.getLogger(__name__)


class StatusCounts:
    def __init__(self):
        self.counts = {}

    def increment(self, status: str):
        """Increment the count for the given status."""
        if status not in self.counts:
            self.counts[status] = 0
        self.counts[status] += 1

    def __str__(self):
        return str(self.counts)


# filter which will count the number of same log messages
class LogCounterFilter(logging.Filter):
    def __init__(self):
        super(LogCounterFilter, self).__init__()
        self.status_counts = StatusCounts()

    def filter(self, record):
        """Filter log messages."""
        self.status_counts.increment(record.getMessage())
        return True

    def __str__(self):
        return str(self.status_counts)


class DuplicateFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_log = None

    def filter(self, record):
        """Filter log messages."""
        current_log = (record.module, record.levelno, record.msg)
        if current_log != self.last_log:
            self.last_log = current_log
            return True
        return False


# Example
# logger = logging.getLogger()
# logger.addFilter(DuplicateFilter())
