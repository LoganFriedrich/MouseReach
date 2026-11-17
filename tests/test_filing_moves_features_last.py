"""Filing moves the features file and the manifest last, and only when every
other file is in.

WHY: the features file is what the database integrator takes from the filed
folder and what the watcher reads as "this run is filed"; the manifest is the
provenance the version scan reads. Moved in listing order, one file that could
not move (a handle still open) left new kinematics filed beside missing or old
outputs -- or a video filed without its .mp4, which the next attempt read as
"no kinematics" and marked failed.
"""
import shutil

import pytest

import mousereach.archive.core as core

V = "20990101_ZZZ0101_P2"
NAMES = (f"{V}.mp4", f"{V}DLC_resnet50_X.h5", f"{V}_segments.json",
         f"{V}_features.json", f"{V}_processing_manifest.json")


@pytest.fixture
def folders(tmp_path, monkeypatch):
    proc, dest = tmp_path / "Processing", tmp_path / "Analyzed" / "ZZZ01"
    proc.mkdir(parents=True)
    dest.mkdir(parents=True)
    for n in NAMES:
        (proc / n).write_text(n)
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    import mousereach.index as index_pkg

    class _NoIndex:
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass

    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)
    return proc, dest


def test_a_file_that_will_not_move_keeps_features_and_manifest_back(folders, monkeypatch):
    proc, dest = folders
    real_move = shutil.move

    def move(src, dst):
        if src.endswith(".mp4"):
            raise PermissionError(32, "in use")
        return real_move(src, dst)

    monkeypatch.setattr(shutil, "move", move)
    r = core.archive_video(V, dry_run=False, verbose=False,
                           skip_ready_check=True, source_dir=proc)
    assert not r["success"]
    assert not (dest / f"{V}_features.json").exists(), "features filed without the video"
    assert not (dest / f"{V}_processing_manifest.json").exists()
    assert (proc / f"{V}_features.json").exists() and (proc / f"{V}.mp4").exists()

    # The retry, once the file is free, files everything.
    monkeypatch.setattr(shutil, "move", real_move)
    r2 = core.archive_video(V, dry_run=False, verbose=False,
                            skip_ready_check=True, source_dir=proc)
    assert r2["success"], r2
    assert sorted(p.name for p in dest.iterdir()) == sorted(NAMES)
    assert not any(proc.iterdir())


def test_features_is_the_very_last_file_moved(folders, monkeypatch):
    proc, dest = folders
    order = []
    real_move = shutil.move

    def move(src, dst):
        order.append(src.rsplit("\\", 1)[-1].rsplit("/", 1)[-1])
        return real_move(src, dst)

    monkeypatch.setattr(shutil, "move", move)
    r = core.archive_video(V, dry_run=False, verbose=False,
                           skip_ready_check=True, source_dir=proc)
    assert r["success"], r
    assert order[-2:] == [f"{V}_processing_manifest.json", f"{V}_features.json"], order
