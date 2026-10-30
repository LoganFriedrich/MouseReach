"""A video that failed on the processing server is run again later, while it
has retries left, and only when its files are still here.

WHY (Logan, 2026-10-30, on an algorithm crash): "fail it, then do it again
another time". Until this, nothing ever selected a 'failed' row again.
"""
from datetime import datetime, timedelta
from pathlib import Path

from mousereach.watcher.orchestrator import ProcessingOrchestrator

STEM = "20990101_CNT9901_P1"
NOW = datetime(2026, 10, 30, 12, 0, 0)


class _Cfg:
    max_retries = 3


def _orch(tmp_path):
    o = object.__new__(ProcessingOrchestrator)
    o.config = _Cfg()
    o.processing_dir = tmp_path
    return o


def _files(tmp_path, pose=True, video=True):
    if video:
        (tmp_path / f"{STEM}.mp4").write_bytes(b"x")
    if pose:
        (tmp_path / f"{STEM}DLC_resnet101_MPSAOct27shuffle3_100000.h5").write_bytes(b"x")


def _row(**kw):
    r = {"video_id": STEM, "state": "failed", "error_count": 1,
         "last_error_at": (NOW - timedelta(hours=2)).isoformat(), "dlc_output_path": None}
    r.update(kw)
    return r


def test_due_after_the_wait_with_files_present(tmp_path):
    _files(tmp_path)
    assert _orch(tmp_path)._failed_retry_due(_row(), now=NOW) is True


def test_not_due_until_an_hour_has_passed(tmp_path):
    _files(tmp_path)
    o = _orch(tmp_path)
    assert o._failed_retry_due(_row(last_error_at=(NOW - timedelta(minutes=59)).isoformat()), now=NOW) is False
    assert o._failed_retry_due(_row(last_error_at=(NOW - timedelta(minutes=61)).isoformat()), now=NOW) is True


def test_no_retries_left_means_it_stays_failed(tmp_path):
    _files(tmp_path)
    assert _orch(tmp_path)._failed_retry_due(_row(error_count=3), now=NOW) is False


def test_missing_video_or_pose_is_not_retried(tmp_path):
    _files(tmp_path, pose=False)
    assert _orch(tmp_path)._failed_retry_due(_row(), now=NOW) is False
    for f in tmp_path.iterdir():
        f.unlink()
    _files(tmp_path, video=False)
    assert _orch(tmp_path)._failed_retry_due(_row(), now=NOW) is False


def test_a_row_with_no_error_time_is_due(tmp_path):
    _files(tmp_path)
    assert _orch(tmp_path)._failed_retry_due(_row(last_error_at=None), now=NOW) is True


def test_an_unjudgeable_row_is_not_retried(tmp_path):
    assert _orch(tmp_path)._failed_retry_due({"video_id": None}, now=NOW) is False
