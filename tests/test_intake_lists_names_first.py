"""Finding a video's files in a big shared folder asks the share about names first.

WHY: intake finds a posed video's files in the posed-video folder, which holds
every posed video waiting for analysis with its pose files (about 5,000
entries). It used to ask the share "is this a file?" for every entry before
looking at the name: about 120 s per video, of an intake whose copying takes
4 s (2026-10-21). The result must be exactly what it was.
"""
from types import SimpleNamespace

import mousereach.watcher.orchestrator as orch
from mousereach.watcher.orchestrator import ProcessingOrchestrator

VID = "20240101_ABC0101_P1"


def _o():
    return object.__new__(ProcessingOrchestrator)


def test_it_finds_exactly_the_videos_files(tmp_path):
    for name in (f"{VID}.mp4", f"{VID}DLC_model.h5", f"{VID}DLC_model.csv",
                 f"{VID}DLC_model.h5.part", f"{VID}.tmp",
                 "20240101_ABC0102_P1.mp4", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / f"{VID}_folder").mkdir()
    got = sorted(p.name for p in _o()._get_associated_files(tmp_path, VID))
    assert got == [f"{VID}.mp4", f"{VID}DLC_model.csv", f"{VID}DLC_model.h5"]


def test_a_missing_folder_is_no_files(tmp_path):
    assert _o()._get_associated_files(tmp_path / "gone", VID) == []


def test_only_entries_with_the_videos_name_are_asked_about(tmp_path, monkeypatch):
    asked = []

    class Entry:
        def __init__(self, name):
            self.name = name
            self.path = str(tmp_path / name)

        def is_file(self):
            asked.append(self.name)
            return True

    names = [f"2023010{i}_ABC01{j:02d}_P1.mp4" for i in range(1, 10) for j in range(50)]
    names += [f"{VID}.mp4", f"{VID}DLC_model.h5"]

    class Listing:
        def __enter__(self):
            return iter(Entry(n) for n in names)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(orch.os, "scandir", lambda d: Listing())
    got = _o()._get_associated_files(tmp_path, VID)
    assert sorted(asked) == sorted([f"{VID}.mp4", f"{VID}DLC_model.h5"])
    assert len(got) == 2
