"""A route never pours a video's files into a bundle it already has in that queue.

A video held in a queue whose older generation is still filed could be routed
again (mousereach-route-to-queue, or a re-run reaching the gate): the move
poured the source's files into the existing bundle, replacing the held run's
video, pose, results and a review a person had saved there with older copies.
Now route_to_queue refuses before anything moves, and the route command
reports such a video as "held" and touches neither the held bundle's files nor
the filed copy. A question it was asked about the video is not dropped: it is
flagged inside the held bundle, and when it cannot be, the command fails.

Built under tmp_path with invented video ids; no real folder or database.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher import route_cli
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.review_gate import route_to_queue

VID = "20990101_ABC9901_P1"


@pytest.fixture
def held(tmp_path, monkeypatch):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed")
    bundle = triage / VID
    bundle.mkdir()
    for n, body in ((f"{VID}.mp4", "held video"),
                    (f"{VID}_segments.json", json.dumps({"note": "held cuts"})),
                    (f"{VID}_pellet_outcomes.json",
                     json.dumps({"segments": [{"segment_num": 3, "outcome": "retrieved"}]})),
                    (f"{VID}_causal_review.json", json.dumps({"reviewer": "person-a", "segments": []}))):
        (bundle / n).write_text(body, encoding="utf-8")
    filed = tmp_path / "Analyzed" / "PROJ" / "PROJ01"
    filed.mkdir(parents=True)
    outcomes = filed / f"{VID}_pellet_outcomes.json"
    outcomes.write_text(json.dumps({"segments": [{"segment_num": 3}]}), encoding="utf-8")
    for n, body in ((f"{VID}.mp4", "filed video"),
                    (f"{VID}_segments.json", "filed cuts"),
                    (f"{VID}_causal_review.json", json.dumps({"reviewer": "person-b", "segments": []}))):
        (filed / n).write_text(body, encoding="utf-8")
    monkeypatch.setattr(route_cli, "_find_outcomes", lambda analyzed, vid: outcomes)
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    return bundle, filed, outcomes, db


def _contents(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def test_route_to_queue_refuses_to_merge_and_moves_nothing(held):
    bundle, filed, _, db = held
    db.force_state(VID, "triage", reason="test")       # the bundle is the live copy
    b0, f0 = _contents(bundle), _contents(filed)
    refused = None
    try:
        route_to_queue(VID, filed, Paths.TRIAGE_REVIEW, "bench check", db=db,
                       db_state="triage")
    except Exception as e:
        refused = e
    assert _contents(bundle) == b0                # the held copy, byte for byte
    assert _contents(filed) == f0                 # nothing taken from the source
    assert type(refused).__name__ == "AlreadyInQueueError"


def _held_outcomes(bundle):
    return json.loads((bundle / f"{VID}_pellet_outcomes.json").read_text(encoding="utf-8"))


def test_route_command_reports_a_bundle_in_the_queue_as_held(held):
    bundle, filed, outcomes, db = held
    b0, f0 = _contents(bundle), _contents(filed)
    res = route_cli.route_video(VID, "triage", "bench check", [3], db=db)
    assert _contents(filed) == f0                 # the filed copy: untouched
    assert "flagged_for_review" not in outcomes.read_text(encoding="utf-8")
    after = _contents(bundle)
    assert {n: b for n, b in after.items() if not n.endswith("_pellet_outcomes.json")}         == {n: b for n, b in b0.items() if not n.endswith("_pellet_outcomes.json")}
    seg = _held_outcomes(bundle)["segments"][0]   # the question, in the held bundle
    assert seg["flagged_for_review"] is True and seg["flag_reason"] == "bench check"
    assert res.get("held") and not res["routed"] and res["flagged"] == [3]
    assert not res.get("unrecorded") and not res["error"]


def test_a_held_bundle_that_cannot_be_read_fails_the_question(held):
    bundle, _, _, db = held
    (bundle / f"{VID}_segments.json").write_text("cut off", encoding="utf-8")
    res = route_cli.route_video(VID, "triage", "bench check", [3], db=db)
    assert res["held"] and res["unrecorded"] == [3]
    assert "question not recorded" in res["error"]


def test_route_command_reports_a_held_row_as_held(held):
    bundle, filed, outcomes, db = held
    # The bundle sits elsewhere (routed from another machine, say); the row
    # alone says the video is held.
    import shutil
    shutil.rmtree(bundle)
    db.force_state(VID, "deep_review", reason="test")
    f0 = _contents(filed)
    res = route_cli.route_video(VID, "deep_review", "segmentation wrong", db=db)
    assert _contents(filed) == f0
    assert not (Paths.DEEP_REVIEW / VID).exists()
    assert res.get("held") and not res["routed"]


def test_held_is_not_a_failure_for_the_exit_code(held, monkeypatch, capsys):
    bundle, _, _, db = held
    monkeypatch.setattr(route_cli, "_open_watcher_db", lambda out: db)
    code = route_cli.main([VID, "--queue", "triage", "--reason", "x",
                           "--flag-segments", "3", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert _held_outcomes(bundle)["segments"][0]["flagged_for_review"] is True
    assert out[0].get("held") and not out[0]["routed"]
    assert code == 0


def test_a_question_that_cannot_reach_a_held_video_fails(held, monkeypatch, capsys):
    """Held in deep review: its results are made again after re-segmentation,
    so a flag would be thrown away. The question is reported as not recorded
    and the command fails, so the caller asks again later."""
    bundle, filed, _, db = held
    import shutil
    shutil.rmtree(bundle)
    db.force_state(VID, "deep_review", reason="test")
    (Paths.DEEP_REVIEW / VID).mkdir()
    f0 = _contents(filed)
    monkeypatch.setattr(route_cli, "_open_watcher_db", lambda out: db)
    code = route_cli.main([VID, "--queue", "deep_review", "--reason", "x",
                           "--flag-segments", "3", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert _contents(filed) == f0
    assert out[0]["held"] and out[0]["unrecorded"] == [3]
    assert "question not recorded" in out[0]["error"]
    assert code == 1
