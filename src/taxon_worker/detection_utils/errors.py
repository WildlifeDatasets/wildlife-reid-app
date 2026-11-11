"""Actionable detector errors that are safe to display without a traceback."""


class DetectorUnavailableError(RuntimeError):
    """Report a detector configuration problem without exposing credentials or paths."""
