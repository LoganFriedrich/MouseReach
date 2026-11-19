"""Run chosen videos by hand: run_one.run_chosen_videos, the Run Pipeline
tab's "Run chosen videos..." and mousereach-run-videos (owner requirement
MRR-15a, 2026-11-19: everything runnable by hand, easily, CLI and GUI, with
the watcher's own steps).

Pinned here:
  * a chosen video in Processing runs the watcher's own handlers and is filed,
    a filed one is refused (the processing-machine path of run_one_video,
    reused from test_run_all_steps_runs_the_watcher's fixture);
  * on a single machine a chosen video with no pose yet is posed FIRST by the
    posing half's own handlers (adopt -> pose -> hand on), in that order;
  * a chosen multi-animal recording on a single machine is cut by the posing
    half's own handler and each child is run;
  * the command prints one line per video and fails when one did not finish.
Video ids are invented.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_run_all_steps_runs_the_watcher import env, _video_folder, V  # noqa: F401


def test_a_chosen_video_runs_the_watchers_steps_and_a_filed_one_is_refused(env):
    from mousereach.watcher.run_one import run_chosen_videos
    proc = env["roots"]["PROCESSING"]
    mp4 = _video_folder(proc)
    db = env["db"]()
    db.register_video(video_id=V, source_path=str(mp4))
    db.force_state(V, "processing", reason="test fixture")
    filed = _video_folder(env["roots"]["ANALYZED_OUTPUT"] / "PRJ" / "C01")
    res = run_chosen_videos([mp4, filed])
    assert res[0]["done"] is True and env["dispatched"] == ["pipeline", "archive"]
    assert "filed" in (res[1]["refused"] or "")


class _FakeDB:
    def __init__(self, state):
        self.rows = {"V1": {"video_id": "V1", "state": state}}
        self.collages = {}

    def get_video(self, vid):
        r = self.rows.get(vid)
        return dict(r) if r else None

    def get_collage(self, name):
        return self.collages.get(name)


class _FakePosing:
    NEXT = {"validated": "dlc_queued", "dlc_queued": "dlc_complete",
            "dlc_complete": "archived"}

    def __init__(self, db, log):
        self.db, self.log = db, log

    def _run_once_blocked(self, when):
        return False

    def _scan_phase(self):
        self.log.append("scan")

    def _dispatch_work(self, work):
        self.log.append(work["type"])
        if work["type"] == "collage":
            self.db.collages[work["id"]]["state"] = "cropped"
            return True
        row = self.db.rows[work["id"]]
        row["state"] = self.NEXT[row["state"]]
        return True

    def _expected_children(self, name):
        return [{"offspring_stem": "20200101_ABC0101_P1"},
                {"offspring_stem": "20200101_ABC0102_P1"}]


def test_on_a_single_machine_an_unposed_video_is_posed_first_by_the_posing_half(monkeypatch):
    from mousereach.watcher import run_one
    log = []
    db = _FakeDB("validated")
    monkeypatch.setattr(run_one, "_posing_half", lambda cfg: _FakePosing(db, log))
    why = run_one._pose_on_this_machine("V1", SimpleNamespace(), lambda st, msg: None)
    assert why is None
    assert log == ["adopt_single", "single_dlc", "stage_to_nas"]
    assert db.rows["V1"]["state"] == "archived"      # handed on to the posed folder


def test_a_pose_that_does_not_move_on_stops_with_the_reason(monkeypatch):
    from mousereach.watcher import run_one
    db = _FakeDB("dlc_queued")
    posing = _FakePosing(db, [])
    posing._dispatch_work = lambda work: db.rows["V1"].update(error_message="no GPU")
    monkeypatch.setattr(run_one, "_posing_half", lambda cfg: posing)
    why = run_one._pose_on_this_machine("V1", SimpleNamespace(), lambda *a: None)
    assert "single_dlc" in why and "no GPU" in why


def test_a_chosen_recording_on_a_single_machine_is_cut_and_each_video_run(tmp_path, monkeypatch):
    from mousereach.config import Paths, WatcherConfig
    from mousereach.watcher import run_one, singleton
    intake = tmp_path / "Unanalyzed" / "Multi-Animal"
    intake.mkdir(parents=True)
    rec = intake / "20200101_ABC0101,ABC0102_P1.mp4"
    rec.write_bytes(b"x")
    monkeypatch.setattr(Paths, "MULTI_ANIMAL_SOURCE", intake, raising=False)
    monkeypatch.setattr(WatcherConfig, "load", classmethod(
        lambda cls: WatcherConfig({"mode": "standalone"})))
    monkeypatch.setattr(singleton, "acquire_watcher_lock", lambda *a, **k: object())
    monkeypatch.setattr(singleton, "release_watcher_lock", lambda h: None)
    log = []
    db = _FakeDB("validated")
    db.collages[rec.name] = {"filename": rec.name, "state": "stable"}
    monkeypatch.setattr(run_one, "_posing_half", lambda cfg: _FakePosing(db, log))
    ran = []
    monkeypatch.setattr(run_one, "run_one_video",
                        lambda v, progress=None: ran.append(v) or {"video": v, "done": True})
    res = run_one.run_chosen_videos([rec])
    assert log == ["collage"]
    assert ran == ["20200101_ABC0101_P1", "20200101_ABC0102_P1"]
    assert all(r["done"] for r in res)


def test_a_recording_is_not_cut_off_a_single_machine(tmp_path, monkeypatch):
    from mousereach.config import Paths, WatcherConfig
    from mousereach.watcher import run_one
    intake = tmp_path / "Multi-Animal"
    intake.mkdir()
    monkeypatch.setattr(Paths, "MULTI_ANIMAL_SOURCE", intake, raising=False)
    monkeypatch.setattr(WatcherConfig, "load", classmethod(
        lambda cls: WatcherConfig({"mode": "processing_server"})))
    res = run_one.run_chosen_videos([intake / "20200101_A,B_P1.mp4"])
    assert "graphics-card PC" in res[0]["refused"]


def test_the_command_prints_a_line_per_video(monkeypatch, capsys):
    from mousereach.watcher import run_one
    monkeypatch.setattr(run_one, "run_chosen_videos", lambda vids, progress=None: [
        {"video": "A", "done": True}, {"video": "B", "refused": "filed"}])
    assert run_one.main(["A.mp4", "B.mp4"]) == 1
    out = capsys.readouterr().out
    assert "[OK] A: filed" in out and "[NOT RUN] B: filed" in out


def test_the_run_all_steps_tab_is_retired():
    from mousereach import launcher
    src = Path(launcher.__file__).read_text(encoding="utf-8")
    assert "RunAllStepsWidget" not in src
    with pytest.raises(ImportError):
        import mousereach.pipeline.run_all_widget  # noqa: F401
