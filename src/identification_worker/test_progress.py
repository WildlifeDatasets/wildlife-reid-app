from unittest.mock import Mock

from progress import ProgressReporter


def test_identification_progress_reports_weighted_stage():
    task = Mock()
    reporter = ProgressReporter(task, operation="identify")

    reporter.stage("identify", "Comparing images")
    reporter.update(1, 2, force=True)

    task.update_state.assert_called_with(
        state="PROGRESS",
        meta={
            "percent": 51,
            "stage": "identify",
            "message": "Comparing images",
        },
    )


def test_identification_progress_is_monotonic():
    task = Mock()
    reporter = ProgressReporter(task, operation="init")

    reporter.stage("finalize", "Finalizing")
    reporter.update(1, 1, force=True)
    reporter.stage("load_metadata", "Late stale update")
    reporter.update(0, 1, force=True)

    last_meta = task.update_state.call_args.kwargs["meta"]
    assert last_meta["percent"] == 99
