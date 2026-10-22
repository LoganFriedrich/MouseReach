"""Optional mode: an open recording program pauses the node only while a
recording is being WRITTEN.

WHY (G12, 2026-10): the recording program is routinely left open after the
day's recording, and a node that pauses whenever it is open then idles until a
person closes it -- once for two days. With ``pause_only_while_writing_to``
set, the files in the recording folder decide. Off by default: an empty list
must behave exactly as before (the program being open is enough).
"""
import os
import time

from mousereach.config import WatcherConfig
from mousereach.watcher.recording_guard import RecordingGuard


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _guard(folders, running=("recorder.exe",), quiet=120, wall=None):
    clock = Clock()
    g = RecordingGuard(["recorder.exe"], grace_seconds=120, clock=clock,
                       scan=lambda names: list(running), cache_seconds=0,
                       write_folders=folders, quiet_seconds=quiet,
                       wall_clock=wall or time.time)
    return g, clock


def _file(folder, name, age_s, size=10):
    p = folder / name
    p.write_bytes(b"x" * size)
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


def test_without_folders_an_open_program_pauses_exactly_as_before(tmp_path):
    g, _ = _guard([])
    assert g.reason() == "recorder.exe is running"


def test_an_open_program_with_no_recent_write_does_not_pause(tmp_path):
    _file(tmp_path, "yesterday.mkv", age_s=3600)
    g, _ = _guard([str(tmp_path)])
    assert g.reason() is None


def test_a_recent_write_pauses(tmp_path):
    _file(tmp_path, "now.mkv", age_s=5)
    g, _ = _guard([str(tmp_path)])
    r = g.reason()
    assert r.startswith("recorder.exe is recording") and "resuming 120 s" in r
    r.encode("ascii")


def test_a_file_that_grows_pauses_even_when_its_listed_time_is_old(tmp_path):
    """Windows can report an open, growing file's time as it was when last
    closed; growth between two looks counts as writing."""
    p = _file(tmp_path, "growing.mkv", age_s=3600, size=10)
    g, _ = _guard([str(tmp_path)])
    assert g.reason() is None                     # first look: old, nothing to compare
    with open(p, "ab") as f:
        f.write(b"more")
    t = time.time() - 3600
    os.utime(p, (t, t))                           # time still looks old
    assert g.reason().startswith("recorder.exe is recording")


def test_an_unreadable_folder_keeps_the_node_paused(tmp_path):
    g, _ = _guard([str(tmp_path / "missing")])
    r = g.reason()
    assert "cannot be checked" in r


def test_with_the_program_closed_the_folders_are_not_consulted(tmp_path):
    _file(tmp_path, "now.mkv", age_s=1)
    g, _ = _guard([str(tmp_path)], running=())
    assert g.reason() is None


def test_quiet_period_is_the_resume_delay_after_the_last_write(tmp_path):
    _file(tmp_path, "just_stopped.mkv", age_s=90)
    g, _ = _guard([str(tmp_path)], quiet=120)
    assert g.reason().startswith("recorder.exe is recording")
    g2, _ = _guard([str(tmp_path)], quiet=60)
    assert g2.reason() is None


def test_the_config_reads_and_writes_the_two_keys():
    cfg = WatcherConfig({"pause_while_running": ["recorder.exe"],
                         "pause_only_while_writing_to": "D:/Recordings",
                         "pause_recording_quiet_seconds": "90"})
    assert cfg.pause_only_while_writing_to == ["D:/Recordings"]
    assert cfg.pause_recording_quiet_seconds == 90
    d = cfg.to_dict()
    assert d["pause_only_while_writing_to"] == ["D:/Recordings"]
    assert d["pause_recording_quiet_seconds"] == 90
    plain = WatcherConfig({"pause_while_running": ["recorder.exe"]}).to_dict()
    assert "pause_only_while_writing_to" not in plain, "absent unless chosen"
