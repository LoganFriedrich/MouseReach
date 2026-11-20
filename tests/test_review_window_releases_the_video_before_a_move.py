"""The triage window lets go of the video before it moves it.

The window reads the mp4 through _LazyVideo, which keeps a reader open per
thread. On Windows a file held open cannot be moved, so Escalate moved the
rest of the bundle into deep review and left the mp4 in the triage folder
(WinError 32). _LazyVideo.close releases every reader, and a later read
opens a new one. Built under tmp_path with a tiny generated video.
"""
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _tiny_video(path, n=5, h=16, w=16):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n):
        out.write(np.full((h, w, 3), i * 40, np.uint8))
    out.release()
    assert path.stat().st_size > 0


def test_close_releases_the_file_and_a_later_read_reopens(tmp_path):
    from mousereach.review.causal_review_widget import _LazyVideo
    mp4 = tmp_path / "20990101_ABC9901_P1.mp4"
    _tiny_video(mp4)
    video = _LazyVideo(mp4, 5, 16, 16, max_cache=0)
    assert video[1].shape == (16, 16, 3)        # a reader is now open
    video.close()
    moved = tmp_path / "moved.mp4"
    os.rename(mp4, moved)                        # fails on Windows while held open
    assert moved.is_file()
    # A read after close opens a new reader on the path it was given.
    os.rename(moved, mp4)
    assert video[2].shape == (16, 16, 3)
    video.close()


def test_escalate_releases_the_video_before_it_moves_it(monkeypatch, tmp_path):
    """Escalate lets go of the mp4 before the route moves the bundle."""
    from types import SimpleNamespace
    from mousereach.review import causal_review_widget as crw
    import mousereach.watcher.review_gate as rg
    order = []
    monkeypatch.setattr(rg, "escalate_to_deep_review",
                        lambda *a, **k: order.append("route"))
    monkeypatch.setattr(crw, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(crw, "show_error", lambda *a, **k: order.append("error"))
    fake = SimpleNamespace(
        _video_stem="20990101_ABC9901_P1", _bundle_dir=str(tmp_path),
        _save_review=lambda: True,
        _watcher_db_for_move=lambda: (None, ""),
        _release_video_file=lambda: order.append("release"),
        _status_label=SimpleNamespace(setText=lambda s: None),
        _load_next_video=lambda: order.append("next"))
    crw.CausalReviewWidget._escalate_to_deep_review(fake)
    assert order == ["release", "route", "next"]
