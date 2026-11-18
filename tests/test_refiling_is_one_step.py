"""Re-filing a filed video is one step: the filed folder holds the previous
generation, complete, or the new one, complete -- never a mix.

WHY: filing used to sweep the previous generation into the Archive first and
then move the new files in one by one. One new file that would not move (a
handle still open on the share) left the filed folder with some new files,
some missing and no complete generation until the retry, up to an hour later;
a nightly import in that window read a folder that matched no run. Filing now
copies every new file in under a temporary name, keeps a verified copy of the
previous generation in the Archive, and only then swaps the files in by
renames, undoing them if one fails (archive/core.py, "FILING IS ONE STEP").

Built under tmp_path with invented ids; no real folder or database.
"""
import json
import os
import shutil
from pathlib import Path

import pytest

import mousereach.archive.core as core
from mousereach.config import Paths

V = "20990101_ZZZ0101_P3"
# The new generation is a different size from the old one: filing takes a file
# of the same size and time for the one already filed (archive.core._same_file).
NEW = "the newer generation"
OUTPUTS = (f"{V}_segments.json", f"{V}_reaches.json", f"{V}_pellet_outcomes.json",
           f"{V}_processing_manifest.json", f"{V}_features.json")


def _content(gen, name):
    if name.endswith("_processing_manifest.json"):
        return json.dumps({"gen": gen, "dlc_model": {"dlc_scorer": "DLC_x_Testshuffle1_1"},
                           "pipeline_versions": {"segmenter": gen}})
    return json.dumps({"gen": gen, "file": name})


def _gen_of(path):
    return json.loads(Path(path).read_text())["gen"]


@pytest.fixture
def folders(tmp_path, monkeypatch):
    analyzed = tmp_path / "Analyzed"
    proc, dest = tmp_path / "Processing", analyzed / "ZZZ01"
    proc.mkdir(parents=True)
    dest.mkdir(parents=True)
    video = b"the same video in both places"
    (dest / f"{V}.mp4").write_bytes(video)
    shutil.copy2(dest / f"{V}.mp4", proc / f"{V}.mp4")
    for n in OUTPUTS:
        (dest / n).write_text(_content("old", n))
        (proc / n).write_text(_content(NEW, n))
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed)
    monkeypatch.setattr(core, "get_archive_destination", lambda vid: dest)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    import mousereach.pipeline.fsutil as fsutil
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    import mousereach.index as index_pkg

    class _NoIndex:
        def load(self): pass
        def remove_video(self, video_id): pass
        def save(self): pass

    monkeypatch.setattr(index_pkg, "PipelineIndex", _NoIndex)
    return proc, dest, analyzed / "Archive"


def _file(proc):
    return core.archive_video(V, dry_run=False, verbose=False,
                              skip_ready_check=True, source_dir=proc)


def _assert_old_generation_stands(proc, dest):
    for n in OUTPUTS:
        assert (dest / n).is_file(), f"{n} missing from the filed folder"
        assert _gen_of(dest / n) == "old", f"{n} in the filed folder is not the old one"
    assert (dest / f"{V}.mp4").is_file()
    assert [p.name for p in dest.iterdir() if p.name.startswith(".")] == [], \
        "a temporary file was left in the filed folder"
    for n in OUTPUTS:
        assert _gen_of(proc / n) == NEW, f"{n} left Processing"


def test_a_new_file_that_cannot_be_filed_leaves_the_old_generation_standing(
        folders, monkeypatch):
    proc, dest, _ = folders
    real_move, real_copy = shutil.move, shutil.copy2

    def refuse_new_reaches(src):
        if Path(src).parent == proc and Path(src).name.endswith("_reaches.json"):
            raise PermissionError(32, "in use by another program")

    def move(src, dst, *a, **k):
        refuse_new_reaches(src)
        return real_move(src, dst, *a, **k)

    def copy2(src, dst, *a, **k):
        refuse_new_reaches(src)
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "move", move)
    monkeypatch.setattr(shutil, "copy2", copy2)
    r = _file(proc)
    assert not r["success"]
    assert r["error"]
    _assert_old_generation_stands(proc, dest)

    # Once the file is free the retry files the whole new generation.
    monkeypatch.setattr(shutil, "move", real_move)
    monkeypatch.setattr(shutil, "copy2", real_copy)
    r2 = _file(proc)
    assert r2["success"], r2
    for n in OUTPUTS:
        assert _gen_of(dest / n) == NEW
    assert not any(proc.iterdir())


def test_a_rename_that_fails_part_way_is_undone(folders, monkeypatch):
    proc, dest, _ = folders
    real_replace = os.replace

    def replace(src, dst):
        # The new reaches file refused at the moment it would take its place
        # (a file system error that retrying does not clear).
        if (Path(dst) == dest / f"{V}_reaches.json"
                and Path(src).name.startswith(".")
                and _gen_of(src) == NEW):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    r = _file(proc)
    assert not r["success"]
    assert "put back" in r["error"]
    _assert_old_generation_stands(proc, dest)


def test_a_reader_never_sees_a_features_file_beside_another_generation(
        folders, monkeypatch):
    """Looked at after every rename of the switch: whenever the filed folder
    holds a features file, every output in it is of that file's generation."""
    proc, dest, _ = folders
    real_replace = os.replace
    mixed = []

    def look():
        feats = dest / f"{V}_features.json"
        if not feats.is_file():
            return
        gen = _gen_of(feats)
        for n in OUTPUTS:
            p = dest / n
            if p.is_file() and _gen_of(p) != gen:
                mixed.append(f"{n} is {_gen_of(p)} beside {gen} features")

    def replace(src, dst):
        real_replace(src, dst)
        look()

    monkeypatch.setattr(os, "replace", replace)
    r = _file(proc)
    assert r["success"], r
    assert mixed == []
    for n in OUTPUTS:
        assert _gen_of(dest / n) == NEW


def test_the_previous_generation_is_kept_in_the_archive(folders):
    proc, dest, archive = folders
    r = _file(proc)
    assert r["success"], r
    kept = {p.name: p for p in archive.rglob("*") if p.is_file() and p.name != "README.txt"}
    for n in OUTPUTS:
        assert n in kept, f"no copy of the previous {n} in the Archive"
        assert _gen_of(kept[n]) == "old"
    assert [p.name for p in dest.iterdir() if p.name.startswith(".")] == []
    assert not any(proc.iterdir())


def test_a_retry_after_everything_was_filed_but_processing_kept_a_copy(
        folders, monkeypatch):
    """The last step removes the copies in Processing. One that cannot be
    removed makes the attempt report failure; the retry finds the files filed
    and finishes without touching the filed folder again."""
    proc, dest, _ = folders
    stuck = proc / f"{V}_segments.json"
    real_unlink = Path.unlink

    def unlink(self, *a, **k):
        if self == stuck:
            raise PermissionError(13, "in use")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", unlink)
    r = _file(proc)
    assert not r["success"]
    for n in OUTPUTS:
        assert _gen_of(dest / n) == NEW
    monkeypatch.setattr(Path, "unlink", real_unlink)
    r2 = _file(proc)
    assert r2["success"], r2
    assert not any(proc.iterdir())
    for n in OUTPUTS:
        assert _gen_of(dest / n) == NEW
