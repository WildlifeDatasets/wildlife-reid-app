from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from celery.exceptions import Ignore
from progress import ProgressReporter


def test_progress_is_monotonic_and_stops_below_completion():
    """Report monotonic progress without claiming the upload is complete."""
    task = Mock()
    reporter = ProgressReporter(task, do_init=True)

    reporter.stage("detection", "Detecting animals")
    reporter.update(8, 10)
    reporter.update(3, 10, force=True)
    reporter.stage("finalize", "Building output archive")
    reporter.update(1, 1, force=True)

    percentages = [call.kwargs["meta"]["percent"] for call in task.update_state.call_args_list]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 99


def test_resume_profile_omits_initialization_stages():
    """Start a resumed run with its metadata loading stage."""
    reporter = ProgressReporter(Mock(), do_init=False)

    assert "prepare_metadata" not in reporter.ranges
    assert "video_previews" not in reporter.ranges
    assert reporter.ranges["load_metadata"][0] == 0


def test_backend_failure_does_not_interrupt_processing():
    """Keep processing when the Celery result backend is unavailable."""
    task = Mock()
    task.update_state.side_effect = RuntimeError("Redis unavailable")
    reporter = ProgressReporter(task, do_init=True)

    reporter.stage("prepare_metadata", "Preparing uploaded media")

    assert reporter.last_percent == 0


def test_reset_is_detected_between_media_files():
    """Check cancellation even when the next progress update would be throttled."""
    with TemporaryDirectory() as directory:
        marker = Path(directory) / "cancelled"
        task = Mock()
        reporter = ProgressReporter(task, do_init=True, cancellation_marker=marker)
        reporter.stage("detection", "Detecting animals")
        marker.touch()
        try:
            reporter.update(0, 20)
        except Ignore:
            pass
        else:
            raise AssertionError("A cancelled run should stop at its next progress check")
        assert task.update_state.call_count == 1


def test_sam3_download_explains_why_detection_progress_is_waiting():
    """Show the download message at the current detection percentage."""
    task = Mock()
    reporter = ProgressReporter(task, do_init=True)
    reporter.stage("detection", "Detecting animals")
    reporter.update(1, 20, force=True)
    percentage = reporter.last_percent
    reporter.report_sam3_download()
    payload = task.update_state.call_args.kwargs["meta"]
    assert payload["percent"] == percentage
    assert payload["message"] == "Downloading SAM3 model weights"
