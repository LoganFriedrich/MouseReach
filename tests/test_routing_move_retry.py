"""Bundle routing absorbs transient Windows file locks.

WHY: the pipeline routes a bundle seconds after its own readers finish with
the mp4; the OS handle is sometimes still closing (WinError 32). _safe_move
used to warn-and-skip, silently leaving the mp4 behind in Processing while
the rest of the bundle moved -- 81 part-moved bundles on 2026-09-09 alone.
"""
import pytest

import mousereach.pipeline.fsutil as fsutil
import mousereach.watcher.review_routing as rrt


def test_safe_move_retries_transient_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    src = tmp_path / "a.mp4"
    src.write_text("video", encoding="utf-8")
    dst = tmp_path / "bundle" / "a.mp4"
    dst.parent.mkdir()

    real_move = rrt.shutil.move
    fails = {"n": 2}

    def flaky_move(s, d):
        if fails["n"] > 0:
            fails["n"] -= 1
            raise PermissionError(13, "being used by another process")
        return real_move(s, d)

    monkeypatch.setattr(rrt.shutil, "move", flaky_move)

    rrt._safe_move(src, dst)

    assert dst.read_text(encoding="utf-8") == "video"
    assert not src.exists()


def test_safe_move_still_raises_a_stuck_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    src = tmp_path / "a.mp4"
    src.write_text("video", encoding="utf-8")
    dst = tmp_path / "bundle" / "a.mp4"
    dst.parent.mkdir()

    def stuck_move(s, d):
        raise PermissionError(13, "being used by another process")

    monkeypatch.setattr(rrt.shutil, "move", stuck_move)

    with pytest.raises(PermissionError):
        rrt._safe_move(src, dst)
    assert src.exists()                      # nothing half-done
