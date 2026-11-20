"""A pose the steps cannot use is a problem of the pose step.

A pose file that would not load, or lacked the points the steps need, used to
fail segmentation and be routed to the re-segmentation queue; its quality
flags ("Could not read DLC file", "Missing SABL or SABR") were routed to a
review queue too. Nothing a person can do in a review window fixes a pose.

Only one case is settled: a FILED video whose unusable pose is not from the
current model is sent for a new pose, never to a review queue. Every other
case (a video not filed yet; a filed video whose unusable pose is the current
model's own) is an open lab decision, and the video goes on exactly as it did
before the check -- the check only records what it found. Built under
tmp_path with invented ids; no real folder or database.
"""
import json

import numpy as np
import pandas as pd
import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import ProcessingOrchestrator

VID = "20990101_ABC9901_P1"
POSE = f"{VID}DLC_resnet50_TestModelJan1shuffle1_100.h5"


def _pose(path, points):
    cols = pd.MultiIndex.from_product([["scorer"], points, ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    pd.DataFrame(np.ones((10, 3 * len(points))), columns=cols).to_hdf(
        path, key="df_with_missing")
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    # The shared storage is there (reachable); a storage folder that is not
    # there at all means the filed tree cannot be checked for publication
    # locks, and routing is refused (pipeline/publication_lock.py).
    nas.mkdir()
    review = nas / "Review"
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", review / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", review / "deep_review")
    monkeypatch.setattr(Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", nas / "Analyzed", raising=False)
    monkeypatch.setattr(Paths, "DLC_STAGING", None, raising=False)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", tmp_path / "proc_root", raising=False)
    o = ProcessingOrchestrator.__new__(ProcessingOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "w.db")
    o.processing_dir = proc
    o.hostname = "TEST-HOST"
    o._hand_score_precheck = lambda vid, d: True
    o.db.register_video(video_id=VID, source_path="x")
    for st in ("validated", "dlc_queued", "dlc_running", "dlc_complete", "processing"):
        o.db.update_state(VID, st)
    (proc / f"{VID}.mp4").write_bytes(b"video")
    return o, proc, tmp_path


def _no_queue(tmp_path):
    review = tmp_path / "nas" / "Review"
    return not any((review / q / VID).exists() for q in ("triage", "deep_review"))


def _waiting(tmp_path):
    f = tmp_path / "nas" / "attention" / "TEST-HOST_waiting_for_pose.json"
    return json.loads(f.read_text(encoding="utf-8"))["items"] if f.exists() else []


def _red(tmp_path):
    f = tmp_path / "nas" / "attention" / "TEST-HOST_restart_stopped.json"
    return json.loads(f.read_text(encoding="utf-8"))["items"] if f.exists() else []


def test_an_unfiled_videos_unusable_pose_starts_over_and_never_reaches_segmentation(
        env, monkeypatch):
    """Not filed: the video is started completely over from a new cut of its
    recording (lab decision 2026-11-19; watcher/restart_from_recrop.py). Here
    the recording is not on the share, so it is listed red ("source recording
    missing") and failed with no retries -- never segmented on the bad pose,
    never sent to a review queue."""
    o, proc, tmp_path = env
    _pose(proc / POSE, ["Nose", "BOXL", "BOXR"])
    import mousereach.segmentation.core.batch as seg
    reached = []
    monkeypatch.setattr(seg, "process_single",
                        lambda *a, **k: reached.append(1) or
                        {"success": False, "error": "no SABL in the pose"})
    row = o.db.get_video(VID)
    try:
        o._run_pipeline({"id": VID, "data": dict(row, dlc_output_path=str(proc / POSE))})
    except Exception:
        pass                                            # whatever happened before
    assert reached == []                                # stopped before segmentation
    row = o.db.get_video(VID)
    assert row["state"] == "failed"
    assert "source recording missing" in (row.get("error_message") or "")
    assert [i["video_id"] for i in _red(tmp_path)] == [VID]
    assert _no_queue(tmp_path)
    assert _waiting(tmp_path) == []


def test_an_unreadable_pose_of_an_unfiled_video_starts_over(env):
    """The quality check's 'Could not read DLC file' on a video not filed:
    started over from a new cut (here: its recording is missing, so listed
    red), never held in a review queue."""
    o, proc, tmp_path = env
    from mousereach.pipeline.triage import triage_video
    from mousereach.watcher.review_gate import DECISION_CLEAN, run_gate
    (proc / f"{VID}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [0, 100]}))
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"segments": [{"segment_num": 1, "outcome": "untouched"}]}))
    bad = proc / POSE
    bad.write_bytes(b"this is not a pose table")
    held = o._run_checks_before_kinematics(VID, bad, 1.0, triage_video,
                                           run_gate, DECISION_CLEAN)
    assert held is True                                     # no kinematics
    row = o.db.get_video(VID)
    assert row["state"] == "failed"
    assert [i["video_id"] for i in _red(tmp_path)] == [VID]
    assert _no_queue(tmp_path)
    assert _waiting(tmp_path) == []


def test_a_usable_pose_passes_the_check(tmp_path):
    from mousereach.pipeline.pose_check import check_pose
    ok = _pose(tmp_path / POSE, ["SABL", "SABR", "BOXL", "BOXR", "Nose"])
    assert check_pose(ok, VID) == (True, "")
    usable, why = check_pose(_pose(tmp_path / "b.h5", ["SABL", "SABR", "Nose"]), VID)
    assert usable is False and "BOXL" in why
    # An Easy tray is not segmented: the ruler points are enough.
    assert check_pose(_pose(tmp_path / "c.h5", ["SABL", "SABR"]),
                      "20990101_ABC9901_E1")[0] is True
    assert check_pose(tmp_path / "missing.h5", VID)[0] is None


DECLARED = "DLC_resnet50_NewModelFeb2shuffle1_200"


def _filed(o, tmp_path, monkeypatch, archive_poses):
    """The video is filed: its video and ``archive_poses`` are in its archive
    folder, the row has archived_at, and the declared model is DECLARED."""
    from mousereach.watcher import repose
    archive = tmp_path / "nas" / "Analyzed" / "PROJ" / "PROJ99"
    archive.mkdir(parents=True)
    (archive / f"{VID}.mp4").write_bytes(b"video")
    for name in archive_poses:
        _pose(archive / name, ["Nose"])                    # unusable: no SABL
    monkeypatch.setattr(repose, "archive_folder", lambda vid: archive)
    monkeypatch.setattr(repose, "declared_scorer", lambda: DECLARED)
    o.db.set_fields(VID, archived_at="2099-01-01T00:00:00")
    return archive


def _run_on(o, proc, pose_name, monkeypatch):
    import mousereach.segmentation.core.batch as seg
    monkeypatch.setattr(seg, "process_single",
                        lambda *a, **k: {"success": False, "error": "no SABL in the pose"})
    _pose(proc / pose_name, ["Nose"])
    row = o.db.get_video(VID)
    o._run_pipeline({"id": VID, "data": dict(row, dlc_output_path=str(proc / pose_name))})


def test_a_filed_videos_old_model_pose_gets_a_real_re_pose_request(env, monkeypatch):
    """The unusable pose is an older model's: the video is marked for a new
    pose and the re-pose round trip writes a request for it -- it is not
    narrowed back onto the same pose."""
    from mousereach.watcher import repose
    o, proc, tmp_path = env
    _filed(o, tmp_path, monkeypatch, [POSE])
    _run_on(o, proc, POSE, monkeypatch)
    row = o.db.get_video(VID)
    assert _no_queue(tmp_path)
    assert (row["state"], row["reprocess_scope"]) == ("outdated", "full")
    [item] = json.loads((tmp_path / "nas" / "attention" / "TEST-HOST_waiting_for_pose.json")
                        .read_text(encoding="utf-8"))["items"]
    assert item["needs_person"] is False                    # a request is really made
    queue = tmp_path / "nas" / "Processing" / "Repose_Queue"
    out = repose.publish_pending(o.db, [row], hostname="TEST-HOST", staging_dir=False,
                                 repose_dir=queue)
    assert out["published"] == [VID] and out["narrowed"] == []
    assert (queue / f"{VID}.json").is_file()
    row = o.db.get_video(VID)
    assert (row["state"], row["reprocess_scope"]) == ("outdated", "full")


def test_a_filed_videos_unusable_current_model_pose_goes_on_as_before(
        env, monkeypatch):
    """The unusable pose IS the declared model's. The re-pose round trip
    would find this same file in the archive (by its name) and never write a
    request, and a new pose from the same model may be the same pose. What to
    do is for the lab to decide, so nothing is pretended: no new pose is
    requested, the video is not failed by the check, and it is not listed as
    waiting for a pose."""
    o, proc, tmp_path = env
    current = f"{VID}{DECLARED}.h5"
    _filed(o, tmp_path, monkeypatch, [current])
    _run_on(o, proc, current, monkeypatch)
    row = o.db.get_video(VID)
    assert row["state"] != "outdated" and row.get("reprocess_scope") != "full"
    assert o.db.get_videos_in_state("outdated") == []       # nothing for the round trip
    assert not str(row.get("mark_reason") or "").startswith("pose unusable")
    assert _waiting(tmp_path) == []
