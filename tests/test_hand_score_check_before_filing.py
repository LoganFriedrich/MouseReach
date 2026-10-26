"""Before a video is filed, its pellets are checked against the hand scores.

WHY (Logan, 2026-10-26): "Videos should be checked to see if they disagree
with bench immediately as a part of the pipeline. If they do, they should
move to triage ... They shouldn't move out of the processing phase until
they have for sure got no issues." Filing means done. A video whose
session's hand scores are not there yet waits, unfiled, for the sheet's next
import; if that import has none, the session was never scored by hand and the
video is filed ("If can't compare to manual, then can't mismatch manual",
Logan, 2026-10-26).
"""
import json

import pytest

import mousereach.watcher.hand_scores as hs
import mousereach.watcher.route_cli as rc
from mousereach.watcher.db import WatcherDB

VID = "20250709_ABC0207_P1"


def _setup(tmp_path, monkeypatch, scores=None, required=("ABC",), last_import=None,
           outcomes=None, answered=()):
    f = tmp_path / "hand.csv"
    lines = ["video_key,pellet,score"]
    for p, s in (scores or {}).items():
        lines.append("%s,%d,%d" % (VID, p, s))
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    f.with_suffix(".json").write_text(json.dumps(
        {"last_successful_import": {"ABC_02": last_import} if last_import else {}}), encoding="utf-8")
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f), "required_projects": list(required)})
    hs._CACHE.update(key=None)
    folder = tmp_path / "work"
    folder.mkdir(exist_ok=True)
    segs = [{"segment_num": n, "outcome": o} for n, o in (outcomes or {}).items()]
    (folder / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": segs}), encoding="utf-8")
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set(answered))
    return folder


def test_every_pellet_agreeing_files_the_video(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2, 2: 0, 3: 1},
                    outcomes={1: "retrieved", 2: "untouched", 3: "displaced_outside"})
    assert hs.check(VID, folder).status == hs.AGREES


def test_a_disagreeing_pellet_is_reported_and_one_a_person_answered_is_not(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2, 2: 0, 3: 1},
                    outcomes={1: "displaced_sa", 2: "retrieved", 3: "displaced_sa"}, answered={2})
    v = hs.check(VID, folder)
    assert v.status == hs.DISAGREES
    assert v.disagreements == {1: "the hand score says retrieved; the video says displaced_sa"}


def test_no_scores_waits_only_in_a_required_project(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={}, last_import="2026-10-26T20:05:00")
    v = hs.check(VID, folder)
    assert v.status == hs.WAITING and v.cohort_id == "ABC_02"
    # an import after the video was ready that did not bring them: never scored
    assert hs.scores_never_came(v, "2026-10-26T10:00:00")
    assert not hs.scores_never_came(v, "2026-10-26T21:00:00")
    folder = _setup(tmp_path, monkeypatch, scores={}, required=())
    assert hs.check(VID, folder).status == hs.NOT_CHECKED


def test_easy_flat_and_an_unconfigured_machine_are_not_checked(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2}, outcomes={1: "untouched"})
    assert hs.check("20250709_ABC0207_E1", folder).status == hs.NOT_CHECKED
    monkeypatch.setattr(hs, "settings", lambda: {})
    assert hs.check(VID, folder).status == hs.NOT_CHECKED


def _orch(tmp_path, monkeypatch):
    import mousereach.watcher.orchestrator as o
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.db.force_state(VID, "processed", reason="test fixture")
    orch.hostname = "HOST"
    return orch


def test_a_disagreement_flags_the_pellet_and_routes_to_triage(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2}, outcomes={1: "untouched"})
    routed = []
    import mousereach.watcher.review_gate as rg
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append((a, k)))
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_scores_allow_filing(VID, folder) is False
    seg = json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())["segments"][0]
    assert seg["flagged_for_review"] is True
    assert seg["flag_reason"] == ("disagrees with the hand score: the hand score says "
                                  "retrieved; the video says untouched")
    assert routed and routed[0][1]["db_state"] == "triage"


def test_a_waiting_video_is_not_filed_listed_and_not_picked_until_the_file_changes(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={}, last_import="2000-01-01T00:00:00")
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_scores_allow_filing(VID, folder) is False
    assert orch._waiting_for_hand_scores(VID)
    listed = json.loads((tmp_path / "nas" / "attention" / "HOST_hand_scores.json").read_text())
    assert [i["video_id"] for i in listed["items"]] == [VID]
    assert listed["items"][0]["needs_person"] is False      # waiting is never an alarm
    # the scores arrive: the file changes, the video is picked and filed
    f = tmp_path / "hand.csv"
    f.write_text("video_key,pellet,score\n%s,1,0\n" % VID, encoding="utf-8")
    import os
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 10))
    assert not orch._waiting_for_hand_scores(VID)
    assert orch._hand_scores_allow_filing(VID, folder) is True
    listed = json.loads((tmp_path / "nas" / "attention" / "HOST_hand_scores.json").read_text())
    assert listed["items"] == []


def test_a_check_that_cannot_run_never_files_the_video(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2}, outcomes={1: "untouched"})
    (folder / f"{VID}_pellet_outcomes.json").write_text("{not json", encoding="utf-8")
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_scores_allow_filing(VID, folder) is False
    # and a person is told, in red: a held video must never be silent
    listed = json.loads((tmp_path / "nas" / "attention" / "HOST_hand_scores.json").read_text())
    item = listed["items"][0]
    assert item["needs_person"] is True and item["kind"] == "hand_score_check_failed"
    assert "could not run" in item["says"]


def test_a_session_the_next_import_has_no_scores_for_is_filed_without_comparison(tmp_path, monkeypatch):
    """Never scored by hand: nothing to compare, so nothing can disagree."""
    folder = _setup(tmp_path, monkeypatch, scores={}, last_import="2999-01-01T00:00:00")
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_scores_allow_filing(VID, folder) is True
    assert not orch._waiting_for_hand_scores(VID)


def test_a_cohort_with_no_import_waits_for_its_sheet_or_for_an_import_of_every_sheet(tmp_path, monkeypatch):
    """A cohort with a sheet waits for that sheet's first import; one with no
    sheet is settled by the next import of every sheet -- never by the file's
    own write time, which a hand import of another cohort also moves."""
    folder = _setup(tmp_path, monkeypatch, scores={}, last_import=None)
    v = hs.check(VID, folder)
    assert v.status == hs.WAITING and v.last_import is None and v.has_sheet is None
    v.full_import_at = "2026-10-26T20:05:00"
    assert not hs.scores_never_came(v, "2026-10-26T10:00:00")     # not known: wait
    v.has_sheet = True
    assert not hs.scores_never_came(v, "2026-10-26T10:00:00")     # its sheet is coming
    v.has_sheet = False
    assert hs.scores_never_came(v, "2026-10-26T10:00:00")         # no sheet, all imported since
    assert not hs.scores_never_came(v, "2026-10-26T21:00:00")     # not since it was ready


def test_the_check_reads_which_cohorts_have_a_sheet_from_the_manifest(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={})
    (tmp_path / "hand.json").write_text(json.dumps({
        "last_successful_import": {}, "sheets_seen": ["ABC_01"],
        "full_import_at": "2026-10-26T20:05:00"}), encoding="utf-8")
    hs._CACHE.update(key=None)
    import os
    f = tmp_path / "hand.csv"
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))   # the file changed
    v = hs.check(VID, folder)
    assert v.has_sheet is False and v.full_import_at == "2026-10-26T20:05:00"
    assert hs.scores_never_came(v, "2026-10-26T10:00:00")


def test_times_are_compared_as_times_and_an_unreadable_one_means_wait():
    """Two tools write these times; text order equals time order only while
    both write the same shape."""
    v = hs.Verdict(hs.WAITING, "no hand scores", last_import="2026-10-26 20:05:00")
    assert hs.scores_never_came(v, "2026-10-26T10:00:00.123456")    # space vs T: still a time
    v.last_import = "2026-10-26T20:05:00Z"
    assert not hs.scores_never_came(v, "2026-10-26T10:00:00")        # zone vs none: not known
    v.last_import = "yesterday"
    assert not hs.scores_never_came(v, "2026-10-26T10:00:00")        # unreadable: wait
