from unittest.mock import Mock

from progress import ProgressReporter


def test_progress_is_monotonic_and_stops_below_completion():
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
    reporter = ProgressReporter(Mock(), do_init=False)

    assert "prepare_metadata" not in reporter.ranges
    assert "video_previews" not in reporter.ranges
    assert reporter.ranges["load_metadata"][0] == 0


def test_backend_failure_does_not_interrupt_processing():
    task = Mock()
    task.update_state.side_effect = RuntimeError("Redis unavailable")
    reporter = ProgressReporter(task, do_init=True)

    reporter.stage("prepare_metadata", "Preparing uploaded media")

    assert reporter.last_percent == 0
