"""A critical finding no person can fix starts the video completely over.

Lab decisions 2026-11-19 (D2-10, D2-7, D2-11 and both corrections): an
unusable pose, or a step output that is missing or unreadable, restarts the
video from a NEW CUT of its multi-animal recording, then pose and every later
step. Everything MouseReach made for the failed attempt is deleted; the
recording and everything a person made are never touched. Restarts are
unlimited, except: 5 in a row ending in the same error stop and list red; a
recording that cannot be read is a recording problem at once; a recording that
cannot be found is listed red. A boundary count or an uncertain outcome is a
question for a person and keeps holding the video as before.

Built under tmp_path with invented ids and a synthetic recording made with
ffmpeg; no real folder or database.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mousereach.config import Paths
from mousereach.watcher import restart_from_recrop as rr
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import ProcessingOrchestrator

VID = "20250101_ABC0102_P1"
COLLAGE = "20250101_ABC0101,ABC0102,ABC0001,ABC0104,ABC0105,ABC0106,ABC0107,ABC0108_P1.mkv"


def _ffmpeg():
    from mousereach.video_prep.core.rebuild import ffmpeg_exe
    return ffmpeg_exe()


@pytest.fixture
def nas(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    (nas / "Unanalyzed" / "Multi-Animal").mkdir(parents=True)
    (nas / "Processing" / "Repose_Queue").mkdir(parents=True)
    (nas / "Processing" / "Posed").mkdir(parents=True)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(Paths, "REPOSE_QUEUE", nas / "Processing" / "Repose_Queue", raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", nas / "Processing" / "Posed", raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", nas / "Analyzed", raising=False)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", nas / "Processing" / "Review", raising=False)
    return nas


def _make_recording(nas, name=COLLAGE):
    p = nas / "Unanalyzed" / "Multi-Animal" / name
    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=1920x1080:rate=10:duration=1", str(p)], check=True)
    return p


# --- the pieces -------------------------------------------------------------

def test_the_position_is_read_from_the_recordings_name_and_a_blank_box_is_never_one():
    assert rr.position_in(COLLAGE, VID) == 2
    assert rr.position_in(COLLAGE, "20250101_ABC0001_P1") is None     # empty box
    assert rr.position_in(COLLAGE, "20250101_ABC0102_P2") is None     # another tray


def test_the_same_error_ignores_numbers_and_paths():
    a = r"step output missing or unreadable: Missing reaches file: C:\x\20250101_ABC0102_P1_reaches.json"
    b = r"step output missing or unreadable: Missing reaches file: D:\y\20250101_ABC0102_P1_reaches.json"
    assert rr.same_error_key(a) == rr.same_error_key(b)
    assert rr.same_error_key(a) != rr.same_error_key("pose unusable: no SABL")


def test_five_restarts_with_the_same_error_stop(nas):
    for _ in range(4):
        rr.append_history(nas, VID, "pose unusable: no SABL", "restarted")
    found = lambda *a: Path("x.mkv")
    assert rr.plan(VID, "pose unusable: no SABL", nas_root=nas, find=found)["decision"] == "restart"
    rr.append_history(nas, VID, "pose unusable: no SABL", "restarted")
    p = rr.plan(VID, "pose unusable: no SABL", nas_root=nas, find=found)
    assert p["decision"] == "stopped" and "5 restarts" in p["says"]
    # A different error is a new run of errors: restarts are otherwise unlimited.
    assert rr.plan(VID, "step output missing or unreadable: Missing reaches file: r",
                   nas_root=nas, find=found)["decision"] == "restart"


def test_a_missing_recording_is_listed(nas):
    p = rr.plan(VID, "pose unusable: x", nas_root=nas, find=lambda *a: None)
    assert p["decision"] == "recording_missing" and "never deleted" in p["says"]


def test_the_recording_is_found_and_the_single_is_cut_again(nas, tmp_path):
    rec = _make_recording(nas)
    assert rr.find_source_recording(VID, nas) == rec
    out = rr.cut_single(rec, VID, tmp_path / "out")
    assert out.name == f"{VID}.mp4" and out.stat().st_size > 0
    import cv2
    cap = cv2.VideoCapture(str(out))
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (480, 540)
    cap.release()


def test_a_recording_that_cannot_be_read_is_unreadable(nas, tmp_path):
    bad = nas / "Unanalyzed" / "Multi-Animal" / COLLAGE
    bad.write_bytes(b"not a video at all")
    with pytest.raises(rr.RecordingUnreadable):
        rr.cut_single(bad, VID, tmp_path / "out")
    assert not (tmp_path / "out" / f"{VID}.mp4").exists()


def test_only_machine_files_are_chosen_for_deletion(tmp_path):
    d = tmp_path / "proc"
    d.mkdir()
    machine = [f"{VID}.mp4", f"{VID}DLC_resnet50_Xshuffle1_100.h5", f"{VID}_reaches.json",
               f"{VID}_pellet_outcomes.json", f"{VID}_triage.json"]
    for n in machine:
        (d / n).write_text("{}")
    (d / f"{VID}_segments.json").write_text(json.dumps({"boundary_source": "human"}))
    (d / f"{VID}_causal_review.json").write_text("{}")
    (d / f"{VID}_unified_ground_truth.json").write_text("{}")
    (d / "20250101_ABC0102_P11_reaches.json").write_text("{}")       # another video
    chosen = sorted(p.name for p in rr.machine_files(VID, [d]))
    assert chosen == sorted(machine)


# --- the watcher: a restart end to end ---------------------------------------

def _orch(nas, tmp_path, state_path=("validated", "dlc_queued", "dlc_running",
                                      "dlc_complete", "processing")):
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.processing_dir = tmp_path / "Processing"
    o.processing_dir.mkdir()
    o.staging_dir = nas / "Processing" / "Posed"
    o.hostname = "TEST-HOST"
    o.config = SimpleNamespace(max_retries=3)
    o.db.register_video(video_id=VID, source_path="x")
    for st in state_path:
        o.db.update_state(VID, st)
    return o


def _red(nas):
    f = nas / "attention" / "TEST-HOST_restart_stopped.json"
    return json.loads(f.read_text(encoding="utf-8"))["items"] if f.exists() else []


def test_a_restart_deletes_the_machine_files_keeps_a_persons_and_asks_for_a_pose(nas, tmp_path, monkeypatch):
    import mousereach.watcher.repose as repose
    monkeypatch.setattr(repose, "declared_scorer", lambda: "DLC_new")
    rec = _make_recording(nas)
    before = rec.read_bytes()
    o = _orch(nas, tmp_path)
    proc, posed = o.processing_dir, o.staging_dir
    old_pose = f"{VID}DLC_resnet50_Xshuffle1_100.h5"
    for n in (f"{VID}.mp4", old_pose):
        (proc / n).write_bytes(b"old attempt")
    for n in (f"{VID}_segments.json", f"{VID}_reaches.json"):
        (proc / n).write_text("{}")     # (an unreadable one would count as a person's)
    for n in (f"{VID}.mp4", old_pose):
        (posed / n).write_bytes(b"old attempt")
    (proc / f"{VID}_causal_review.json").write_text('{"a person": true}')

    assert o._restart_from_recrop(VID, "pose unusable: Missing SABL or SABR") is True

    assert rec.read_bytes() == before                                 # recording untouched
    assert (proc / f"{VID}_causal_review.json").read_text() == '{"a person": true}'
    assert sorted(p.name for p in proc.iterdir()) == sorted(
        [f"{VID}.mp4", f"{VID}_causal_review.json"])                 # the new cut + the person's
    assert (proc / f"{VID}.mp4").read_bytes() != b"old attempt"
    assert list(posed.iterdir()) == []
    parked = nas / "Processing" / "Repose_Queue" / "_parked" / VID / f"{VID}.mp4"
    assert parked.is_file()
    req = json.loads((nas / "Processing" / "Repose_Queue" / f"{VID}.json").read_text())
    assert req["video_path"] == str(parked) and req["declared_scorer"] == "DLC_new"
    row = o.db.get_video(VID)
    assert row["state"] == "outdated" and row["reprocess_scope"] == "full"
    assert row["mark_reason"].startswith("restart from re-crop: pose unusable")
    hist = rr.read_history(nas, VID)
    assert [h["outcome"] for h in hist] == ["restarted"]
    assert _red(nas) == []


def test_a_recording_that_cannot_be_read_is_a_recording_problem_at_once(nas, tmp_path):
    (nas / "Unanalyzed" / "Multi-Animal" / COLLAGE).write_bytes(b"broken")
    o = _orch(nas, tmp_path)
    assert o._restart_from_recrop(VID, "pose unusable: x") is True
    row = o.db.get_video(VID)
    assert row["state"] == "failed" and "recording problem" in row["error_message"]
    assert row["error_count"] >= 3                                    # no retry into it
    assert [i["video_id"] for i in _red(nas)] == [VID]
    assert (nas / "Unanalyzed" / "Multi-Animal" / COLLAGE).read_bytes() == b"broken"


def test_the_fifth_restart_with_the_same_error_is_listed_red(nas, tmp_path):
    for _ in range(5):
        rr.append_history(nas, VID, "pose unusable: no SABL", "restarted")
    o = _orch(nas, tmp_path)
    assert o._restart_from_recrop(VID, "pose unusable: no SABL") is True
    row = o.db.get_video(VID)
    assert row["state"] == "failed" and "restart stopped" in row["error_message"]
    assert [i["video_id"] for i in _red(nas)] == [VID]


def test_a_filed_video_is_never_started_over(nas, tmp_path):
    _make_recording(nas)
    o = _orch(nas, tmp_path)
    o.db.update_state(VID, "processed")
    o.db.update_state(VID, "archiving")
    o.db.update_state(VID, "archived")
    assert o._restart_from_recrop(VID, "pose unusable: x") is False
    assert o.db.get_video(VID)["state"] == "archived"


# --- which quality findings start a video over --------------------------------

def _triage_with(o, monkeypatch, flags):
    import mousereach.pipeline.triage as tri
    result = tri.TriageResult(video_id=VID, verdict="needs_review",
                              flags=[tri.TriageFlag(*f) for f in flags])
    monkeypatch.setattr(o, "_restart_from_recrop", lambda vid, err: o._restarts.append(err) or True)
    o._restarts = []
    return o._run_quality_triage(VID, o.processing_dir, None, lambda *a, **k: result)


def test_a_missing_step_output_starts_over(nas, tmp_path, monkeypatch):
    from mousereach.watcher.orchestrator import QC_RESTARTED
    o = _orch(nas, tmp_path)
    v = _triage_with(o, monkeypatch, [("structural", "critical",
                                       f"Missing reaches file: {VID}_reaches.json")])
    assert v == QC_RESTARTED
    assert o._restarts and o._restarts[0].startswith("step output missing or unreadable")


@pytest.mark.parametrize("flag", [
    ("structural", "critical", "Unexpected boundary count: 3 (expected 20-22)"),
    ("cross_step", "critical", "Outcome detector reported uncertain (conf 0.40)"),
    ("cross_step", "critical", "Outcome confidence extremely low (0.10)"),
])
def test_a_question_a_person_can_answer_still_holds_for_a_person(nas, tmp_path, monkeypatch, flag):
    o = _orch(nas, tmp_path)
    v = _triage_with(o, monkeypatch, [flag])
    assert v == "needs_review" and o._restarts == []
