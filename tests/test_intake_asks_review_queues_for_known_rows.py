"""The processing server's intake asks the review queues before taking in a
posed video it ALREADY has a row for.

The posed-folder scan asks "held in a review queue, or filed?" only for a posed
video with no row (test_posed_scan_asks_held_or_filed). A row put back to
'dlc_complete' some other way -- a person's reset, an intake parked for a
missing file -- was taken in without asking, analyzed again and routed at the
queue folder that already held a person's bundle. Now the intake records such
a row in that queue's state and copies nothing.

A row a lifted hold sent back to the posed folder is still taken in: the
leftover kept for a person (it holds the left-over note) is not a hold. And a
filed video's row put back to 'dlc_complete' is taken in as before: every
writer of that state for an existing row asks for a fresh run.
tmp_path only (the fixtures of test_singles_intake_claim).
"""
import json
from types import SimpleNamespace

import mousereach.watcher.orchestrator as orch
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.review_routing import LEFT_NOTE

from test_singles_intake_claim import POSE, _archive, env  # noqa: F401

VID = "20240101_ABC9901_P1"


def _server(env):
    (env.dlc_staging / f"{VID}.mp4").write_bytes(b"video")
    (env.dlc_staging / f"{VID}{POSE}.h5").write_bytes(b"pose")
    db = WatcherDB(db_path=env.tmp / "server.db")
    db.register_video(video_id=VID, source_path=str(env.dlc_staging / f"{VID}.mp4"))
    # How a person's reset (or the posed-folder scan's re-drive) leaves it.
    db.force_state(VID, "dlc_complete", reason="test setup",
                   current_path=str(env.dlc_staging / f"{VID}.mp4"),
                   dlc_output_path=str(env.dlc_staging / f"{VID}{POSE}.h5"))
    s = object.__new__(orch.ProcessingOrchestrator)
    s.db = db
    s.processing_dir = env.processing
    s.staging_dir = env.dlc_staging
    s.config = SimpleNamespace(max_retries=3)
    s._claim_video = lambda video_id: True
    s._release_claim = lambda video_id: None
    return s


def _intake(s):
    return s._intake_from_staging({"id": VID, "data": s.db.get_video(VID)})


def _copied(env):
    return sorted(p.name for p in env.processing.glob(f"{VID}*"))


def test_a_known_row_whose_bundle_is_in_triage_is_recorded_held_not_taken_in(env):
    s = _server(env)
    (env.triage_review / VID).mkdir()
    (env.triage_review / VID / f"{VID}_triage.json").write_text("{}")
    _intake(s)
    assert s.db.get_video(VID)["state"] == "triage"
    assert _copied(env) == [], "a video held in triage was copied in for a second run"
    # The person's bundle and the posed copy are untouched.
    assert (env.triage_review / VID / f"{VID}_triage.json").is_file()
    assert (env.dlc_staging / f"{VID}.mp4").is_file()


def test_a_known_row_being_routed_into_deep_review_is_recorded_held(env):
    s = _server(env)
    inc = env.deep_review / ".incoming" / VID
    inc.mkdir(parents=True)
    (inc / f"{VID}_segments.json").write_text("{}")
    _intake(s)
    assert s.db.get_video(VID)["state"] == "deep_review"
    assert _copied(env) == []


def test_a_known_row_with_no_hold_is_taken_in(env):
    s = _server(env)
    _intake(s)
    assert s.db.get_video(VID)["state"] == "processing"
    assert _copied(env) == sorted([f"{VID}.mp4", f"{VID}{POSE}.h5"])


def test_the_leftover_of_a_lifted_hold_does_not_bounce_the_video_back(env):
    """_release_to_posed puts the row to 'dlc_complete' and leaves a person's
    file in the queue folder with the left-over note: that folder is not the
    video's live copy, so the fresh run the release asked for goes ahead."""
    s = _server(env)
    left = env.deep_review / VID
    left.mkdir()
    (left / f"{VID}_ground_truth.json").write_text("{}")
    (left / LEFT_NOTE).write_text("{}")
    _intake(s)
    assert s.db.get_video(VID)["state"] == "processing"


def test_a_filed_video_put_back_for_a_fresh_run_is_still_taken_in(env):
    s = _server(env)
    d = _archive(VID)
    (d / f"{VID}_processing_manifest.json").write_text(
        json.dumps({"dlc_model": {"dlc_scorer": POSE}}), encoding="ascii")
    _intake(s)
    assert s.db.get_video(VID)["state"] == "processing"
