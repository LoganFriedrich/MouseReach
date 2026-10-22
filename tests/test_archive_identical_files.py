"""Re-archiving leaves files identical to the archived ones where they are.

WHY (2026-10-22): a re-run copies the video and pose down from the archive and
hands them back unchanged. They were swept into the versioned Archive as a
"previous generation" and moved back -- a duplicate pose stored for every
re-run and 60-100 s of copying. Only files that actually changed are a new
generation.
"""
import os

import mousereach.archive.core as core
import mousereach.archive.supersede as supersede

V = "20250625_CNT0106_P1"


def test_unchanged_files_are_not_superseded_and_changed_ones_are(tmp_path, monkeypatch):
    proc, dest = tmp_path / "Processing", tmp_path / "Analyzed" / "CNT01"
    proc.mkdir(parents=True)
    dest.mkdir(parents=True)
    for name, old, new in ((f"{V}.mp4", b"video", b"video"),
                           (f"{V}DLC_resnet101_X.h5", b"pose", b"pose"),
                           (f"{V}_features.json", b'{"v": 1}', b'{"v": 2}')):
        (dest / name).write_bytes(old)
        (proc / name).write_bytes(new)
        st = (dest / name).stat()
        if new == old:                  # copied down and back: copy2 keeps the time
            os.utime(proc / name, (st.st_atime, st.st_mtime))
        else:                           # rewritten by the re-run: a later time
            os.utime(proc / name, (st.st_atime, st.st_mtime + 60))
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)
    swept = {}

    def fake_supersede(video_id, source_dir, archive_root=None, *, only=None, **kw):
        swept["only"] = set(only)
        for name in only:
            (source_dir / name).unlink()
        return {"algo": sorted(only), "pose": [], "failed": []}

    monkeypatch.setattr(supersede, "supersede_video_outputs", fake_supersede)
    import mousereach.index as index_pkg

    class _NoIndex:                     # never touch the real pipeline index
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass

    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)

    r = core.archive_video(V, dry_run=False, verbose=False, skip_ready_check=True,
                           source_dir=proc)

    assert r["success"], r
    assert swept["only"] == {f"{V}_features.json"}, "only the changed file is a new generation"
    assert (dest / f"{V}_features.json").read_bytes() == b'{"v": 2}'
    assert (dest / f"{V}.mp4").read_bytes() == b"video"
    assert not any(proc.iterdir()), "local copies are dropped either way"


def test_same_size_different_time_is_settled_by_content(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"diff")
    os.utime(b, (1, 1))
    os.utime(c, (1, 1))
    assert core._same_file(a, b) is True
    assert core._same_file(a, c) is False
