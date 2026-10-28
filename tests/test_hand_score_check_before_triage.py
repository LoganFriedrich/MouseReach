"""After analysis and before the review gate, a video's pellets are checked
against the hand scores (watcher/hand_scores.py, _hand_score_precheck).

WHY (Logan, 2026-10-26): "Videos should be checked to see if they disagree
with bench immediately as a part of the pipeline. If they do, they should
move to triage ... They shouldn't move out of the processing phase until
they have for sure got no issues." Since 2026-10-28 the check runs BEFORE the
gate, so a disagreeing pellet joins the video's other open questions and the
gate sends it to triage once. A video whose session's hand scores are not
there yet is held in 'processing' for the sheet's next import; if that import
has none, the session was never scored by hand and the video goes on ("If
can't compare to manual, then can't mismatch manual", Logan, 2026-10-26).
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
    assert v.disagreements == {1: "the hand score says retrieved; the algorithm says displaced_sa"}


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
    orch.db.force_state(VID, "processing", reason="test fixture")
    orch.hostname = "HOST"
    return orch


def _started_at(tmp_path, when):
    """Set the row's processing_started_at (the time a wait is measured from)."""
    import sqlite3
    con = sqlite3.connect(str(tmp_path / "local.db"))
    con.execute("UPDATE videos SET processing_started_at = ? WHERE video_id = ?", (when, VID))
    con.commit()
    con.close()


def _seg(folder, n_segments=3):
    """A segmentation the gate accepts (well-spaced cuts), written BEFORE the
    outcomes so the gate does not read the outcomes as stale."""
    import os
    p = folder / f"{VID}_segments.json"
    p.write_text(json.dumps({"overall_confidence": 0.9,
                             "boundaries": [1000 * i for i in range(n_segments + 1)]}),
                 encoding="utf-8")
    old = (folder / f"{VID}_pellet_outcomes.json").stat().st_mtime - 60
    os.utime(p, (old, old))


def _attention(tmp_path):
    return json.loads((tmp_path / "nas" / "attention" / "HOST_hand_scores.json").read_text())


def test_a_disagreement_only_flags_and_the_gate_sends_it_to_triage(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2, 2: 0, 3: 1},
                    outcomes={1: "untouched", 2: "untouched", 3: "displaced_sa"})
    _seg(folder)
    routed = []
    import mousereach.watcher.review_gate as rg
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append((a, k)))
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is True       # on to the gate
    assert routed == []                                         # the check routes nothing
    doc = json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())
    seg = doc["segments"][0]
    assert seg["flagged_for_review"] is True
    assert seg["flag_reason"] == ("bench disagreement: the hand score says "
                                  "retrieved; the algorithm says untouched")
    assert not doc["segments"][1].get("flagged_for_review")
    assert doc["hand_scores"]["verdict"] == "disagrees"
    assert doc["hand_scores"]["pellets"] == {"1": 2, "2": 0, "3": 1}
    # the gate, which routes once, sends it to triage for that pellet
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_TRIAGE and reason.endswith(": 1")
    # the review tool recognises it, labels it and opens on the algorithm's answer
    from mousereach.review.causal_review_widget import is_bench_mismatch, plain_flag_reason
    assert is_bench_mismatch(seg["flag_reason"])
    assert plain_flag_reason(seg["flag_reason"]) == "hand score mismatch (hand: retrieved, algo: untouched)"
    assert orch.db.get_video(VID)["state"] == "processing"


def test_a_reason_the_pipeline_already_had_is_kept_after_the_bench_one(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2}, outcomes={1: "untouched"})
    p = folder / f"{VID}_pellet_outcomes.json"
    doc = json.loads(p.read_text())
    doc["segments"][0].update(flagged_for_review=True, flag_reason="outcome_uncertain: low margin")
    p.write_text(json.dumps(doc), encoding="utf-8")
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is True
    reason = json.loads(p.read_text())["segments"][0]["flag_reason"]
    assert reason == ("bench disagreement: the hand score says retrieved; the algorithm "
                      "says untouched (also: outcome_uncertain: low margin)")
    from mousereach.review.causal_review_widget import is_bench_mismatch, plain_flag_reason
    assert is_bench_mismatch(reason)
    assert plain_flag_reason(reason) == "hand score mismatch (hand: retrieved, algo: untouched)"
    # checked again (a re-run), the bench reason is not stacked on itself
    assert orch._hand_score_precheck(VID, folder) is True
    assert json.loads(p.read_text())["segments"][0]["flag_reason"] == reason


def test_a_pellet_a_person_answered_is_never_flagged(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2, 2: 0},
                    outcomes={1: "untouched", 2: "untouched"}, answered={1})
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is True
    doc = json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())
    assert not any(s.get("flagged_for_review") for s in doc["segments"])
    assert doc["hand_scores"]["verdict"] == "agrees"


def test_agreeing_and_unchecked_videos_go_on_and_their_scores_are_kept(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2}, outcomes={1: "retrieved"})
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is True
    assert json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())[
        "hand_scores"]["pellets"] == {"1": 2}
    monkeypatch.setattr(hs, "settings", lambda: {})             # nothing configured
    assert orch._hand_score_precheck(VID, folder) is True


def test_a_waiting_video_is_held_in_processing_listed_and_not_picked_until_the_file_changes(
        tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={}, last_import="2000-01-01T00:00:00",
                    outcomes={1: "untouched"})
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is False
    assert orch.db.get_video(VID)["state"] == "processing"      # held where it is
    assert (folder / f"{VID}_pellet_outcomes.json").exists()
    assert orch._waiting_for_hand_scores(VID) and orch._held_for_hand_scores(VID)
    item = _attention(tmp_path)["items"][0]
    assert item["video_id"] == VID and item["needs_person"] is False   # never an alarm
    assert "held before triage and kinematics" in item["says"]
    item["says"].encode("ascii")
    # logged once as a step, not on every check
    assert orch._hand_score_precheck(VID, folder) is False
    import sqlite3
    con = sqlite3.connect(str(tmp_path / "local.db"))
    n = con.execute("SELECT COUNT(*) FROM processing_log WHERE video_id = ? AND "
                    "step = 'hand_score_check'", (VID,)).fetchone()[0]
    con.close()
    assert n == 1
    assert "hand_scores" not in json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())
    # the scores arrive: the file changes, the video is picked and goes on
    f = tmp_path / "hand.csv"
    f.write_text("video_key,pellet,score\n%s,1,0\n" % VID, encoding="utf-8")
    import os
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 10))
    assert not orch._waiting_for_hand_scores(VID)
    assert orch._held_for_hand_scores(VID)          # still not counted for intake
    assert orch._hand_score_precheck(VID, folder) is True
    assert not orch._held_for_hand_scores(VID)
    assert _attention(tmp_path)["items"] == []


def test_a_check_that_cannot_run_holds_the_video_and_says_so_in_red(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2}, outcomes={1: "untouched"})
    (folder / f"{VID}_pellet_outcomes.json").write_text("{not json", encoding="utf-8")
    orch = _orch(tmp_path, monkeypatch)
    assert orch._hand_score_precheck(VID, folder) is False
    item = _attention(tmp_path)["items"][0]
    assert item["needs_person"] is True and item["kind"] == "hand_score_check_failed"
    assert "could not run" in item["says"]


def test_never_scored_is_measured_from_when_the_video_entered_processing(tmp_path, monkeypatch):
    """A held video is never 'processed', so its wait is measured from
    processing_started_at."""
    folder = _setup(tmp_path, monkeypatch, scores={}, last_import="2026-10-26T20:05:00",
                    outcomes={1: "untouched"})
    orch = _orch(tmp_path, monkeypatch)
    _started_at(tmp_path, "2026-10-26T21:00:00")                # after that import: wait
    assert orch._hand_score_precheck(VID, folder) is False
    assert orch._score_waits[VID]["ready_since"] == "2026-10-26T21:00:00"
    orch._score_waits.clear()
    _started_at(tmp_path, "2026-10-26T10:00:00")                # before it: never scored
    assert orch._hand_score_precheck(VID, folder) is True
    assert not orch._held_for_hand_scores(VID)
    assert json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())[
        "hand_scores"]["verdict"] == "never_scored"


def test_the_first_waits_ready_time_is_kept(tmp_path, monkeypatch):
    """A re-run must not move the time a wait is measured from."""
    folder = _setup(tmp_path, monkeypatch, scores={}, last_import="2026-10-26T20:05:00",
                    outcomes={1: "untouched"})
    orch = _orch(tmp_path, monkeypatch)
    _started_at(tmp_path, "2026-10-26T10:00:00")
    orch._score_waits = {VID: {"stamp": None, "cohort_id": "ABC_02",
                               "ready_since": "2026-10-26T10:00:00"}}
    _started_at(tmp_path, "2026-10-26T23:00:00")                # re-stamped by something
    assert orch._hand_score_precheck(VID, folder) is True


# ---------------------------------------------------------------------------
# The work loop: a held video is not picked, and does not use up intake
# ---------------------------------------------------------------------------

class _FakeDB:
    def __init__(self, by_state):
        self.by_state = by_state

    def get_videos_in_state(self, state):
        return [dict(v) for v in self.by_state.get(state, [])]

    def get_collages_in_state(self, state):
        return []


def _row(vid):
    return {"video_id": vid, "tray_type": None, "animal_id": None}


def _server(by_state, max_local_pending):
    from types import SimpleNamespace
    from mousereach.watcher.orchestrator import ProcessingOrchestrator
    orch = object.__new__(ProcessingOrchestrator)
    orch.config = SimpleNamespace(max_local_pending=max_local_pending, also_process=False,
                                  work_priority=None)
    orch.db = _FakeDB(by_state)
    orch._get_priority_animal = lambda: None
    orch._archive_reachable = lambda: True
    return orch


@pytest.fixture
def _no_lab_order(monkeypatch):
    from mousereach.watcher import work_priority as wp
    monkeypatch.setattr(wp, "read_lab_priority", lambda *a, **k: None)
    wp.invalidate_lab_policy()
    yield
    wp.invalidate_lab_policy()


def test_a_held_video_is_not_picked_and_does_not_count_against_intake(
        tmp_path, monkeypatch, _no_lab_order):
    monkeypatch.setattr(hs, "scores_file_stamp", lambda: 5.0)
    held = "20250709_ABC0207_P1"
    new = "20250709_ABC0208_P1"
    orch = _server({"processing": [_row(held)], "dlc_complete": [_row(new)]},
                   max_local_pending=1)
    orch._score_waits = {held: {"stamp": 5.0, "cohort_id": "ABC_02"}}
    work = orch._select_work_item(admit_deferred=True)
    assert work["type"] == "intake" and work["id"] == new
    # nothing new to take in: the held video is still not picked
    orch.db.by_state["dlc_complete"] = []
    assert orch._select_work_item(admit_deferred=True) is None
    # the scores file changes: it is picked again, to be checked again
    monkeypatch.setattr(hs, "scores_file_stamp", lambda: 6.0)
    work = orch._select_work_item(admit_deferred=True)
    assert work["type"] == "pipeline" and work["id"] == held
    # and a video that is NOT held still counts against the limit
    orch._score_waits = {}
    orch.db.by_state["dlc_complete"] = [_row(new)]
    work = orch._select_work_item(admit_deferred=True)
    assert work["type"] == "pipeline" and work["id"] == held


def test_a_node_that_also_analyses_skips_a_held_video_too(monkeypatch, _no_lab_order):
    from types import SimpleNamespace
    from mousereach.watcher.orchestrator import DLCOrchestrator
    monkeypatch.setattr(hs, "scores_file_stamp", lambda: 5.0)
    orch = object.__new__(DLCOrchestrator)
    orch.config = SimpleNamespace(also_process=True, work_priority=None)
    orch.db = _FakeDB({"processing": [_row(VID)]})
    orch._get_priority_animal = lambda: None
    orch._archive_backoff_active = lambda video_id: False
    orch._score_waits = {VID: {"stamp": 5.0}}
    picked = orch._select_work_item(admit_deferred=True)
    assert picked is None or picked.get("id") != VID
    monkeypatch.setattr(hs, "scores_file_stamp", lambda: 6.0)
    picked = orch._select_work_item(admit_deferred=True)
    assert picked["type"] == "local_pipeline" and picked["id"] == VID


def test_a_held_video_picked_again_reuses_its_analysis(tmp_path, monkeypatch):
    """When the scores file changes the held video runs again only to be
    checked again: segmentation, reaches and outcomes are not redone."""
    import mousereach.watcher.orchestrator as o
    import mousereach.segmentation.core.batch as segb
    import mousereach.reach.core.batch as reachb
    import mousereach.outcomes.core.batch as outb
    import mousereach.assignment.run as assign
    import mousereach.pipeline.manifest as man
    import mousereach.pipeline.triage as tri

    def _must_not_run(*a, **k):
        raise AssertionError("a held video's analysis was redone")

    folder = _setup(tmp_path, monkeypatch, scores={}, last_import="2000-01-01T00:00:00",
                    outcomes={1: "untouched"})
    for name in ("_segments.json", "_reaches.json", "_reach_assignments.json"):
        (folder / f"{VID}{name}").write_text("{}", encoding="utf-8")
    pose = folder / f"{VID}DLC_test.h5"
    pose.write_bytes(b"")
    monkeypatch.setattr(o, "resolve_pose_input", lambda *a, **k: pose)
    monkeypatch.setattr(segb, "process_single", _must_not_run)
    monkeypatch.setattr(reachb, "process_single", _must_not_run)
    monkeypatch.setattr(outb, "process_single", _must_not_run)
    monkeypatch.setattr(assign, "assign_reaches_for_video", _must_not_run)
    monkeypatch.setattr(man, "create_processing_manifest", lambda **k: {})
    monkeypatch.setattr(tri, "triage_video", lambda **k: (_ for _ in ()).throw(RuntimeError("no QC here")))
    import mousereach.watcher.review_gate as rg
    monkeypatch.setattr(rg, "run_gate", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the gate ran on a held video")))
    orch = _orch(tmp_path, monkeypatch)
    orch.processing_dir = folder
    orch._score_waits = {VID: {"stamp": -1.0, "cohort_id": "ABC_02",
                               "ready_since": "2026-10-26T10:00:00"}}
    orch._run_pipeline({"id": VID, "data": {"video_id": VID, "dlc_output_path": str(pose)}})
    assert orch._held_for_hand_scores(VID)                       # held again
    assert orch.db.get_video(VID)["state"] == "processing"


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


# ---------------------------------------------------------------------------
# Every pellet's score is kept with the video (hand_scores.record)
# ---------------------------------------------------------------------------

def _kept(folder):
    return json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())


def test_record_keeps_every_pellet_and_every_other_key(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2, 2: 0, 3: 1},
                    outcomes={1: "retrieved", 2: "untouched", 3: "displaced_outside"},
                    last_import="2026-10-26T20:05:00")
    v = hs.check(VID, folder)
    assert v.status == hs.AGREES
    assert hs.record(folder, VID, v, 123.0) is True
    doc = _kept(folder)
    assert [s["segment_num"] for s in doc["segments"]] == [1, 2, 3]   # untouched
    kept = doc["hand_scores"]
    assert kept["pellets"] == {"1": 2, "2": 0, "3": 1}
    assert kept["verdict"] == "agrees"
    assert kept["source_mtime"] == 123.0
    assert kept["cohort_last_import"] == "2026-10-26T20:05:00"
    assert kept["read_at"]
    assert not list(folder.glob("*.tmp"))


def test_record_keeps_the_scores_when_a_person_answered_every_pellet(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2, 2: 0},
                    outcomes={1: "untouched", 2: "retrieved"}, answered={"all"})
    v = hs.check(VID, folder)
    assert v.status == hs.AGREES and v.reason == "every pellet answered by a person"
    hs.record(folder, VID, v, 1.0)
    assert _kept(folder)["hand_scores"]["pellets"] == {"1": 2, "2": 0}


def test_record_keeps_disagreeing_and_unscored_sessions_too(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2}, outcomes={1: "untouched"})
    v = hs.check(VID, folder)
    hs.record(folder, VID, v, 1.0)
    assert _kept(folder)["hand_scores"]["verdict"] == "disagrees"
    # not a required project, no row: the sheet was read, pellets empty
    folder = _setup(tmp_path, monkeypatch, scores={}, required=(), outcomes={1: "untouched"})
    v = hs.check(VID, folder)
    hs.record(folder, VID, v, 1.0)
    assert _kept(folder)["hand_scores"] == dict(_kept(folder)["hand_scores"],
                                                pellets={}, verdict="not_required")
    # a waiting session whose scores never came
    folder = _setup(tmp_path, monkeypatch, scores={}, outcomes={1: "untouched"})
    v = hs.check(VID, folder)
    assert v.status == hs.WAITING
    hs.record(folder, VID, v, 1.0)
    assert _kept(folder)["hand_scores"]["verdict"] == "never_scored"


def test_nothing_is_recorded_when_the_sheet_was_not_read(tmp_path, monkeypatch):
    folder = _setup(tmp_path, monkeypatch, scores={1: 2}, outcomes={1: "untouched"})
    v = hs.check("20250709_ABC0207_E1", folder)                # Easy tray
    assert hs.record(folder, "20250709_ABC0207_E1", v, 1.0) is False
    monkeypatch.setattr(hs, "settings", lambda: {})            # no scores file
    v = hs.check(VID, folder)
    assert hs.record(folder, VID, v, 1.0) is False
    assert "hand_scores" not in _kept(folder)


def test_score_from_outcomes_reads_one_pellet_and_never_raises():
    doc = {"hand_scores": {"pellets": {"3": 1}, "verdict": "agrees"}}
    assert hs.score_from_outcomes(doc, 3) == (1, "")
    assert hs.score_from_outcomes(doc, 4)[0] is None
    score, why = hs.score_from_outcomes({"segments": []}, 3)       # analysed before
    assert score is None and "no hand scores kept" in why
    score, why = hs.score_from_outcomes(
        {"hand_scores": {"pellets": {}, "verdict": "never_scored"}}, 3)
    assert score is None and "no hand scores" in why
    assert hs.score_from_outcomes(None, 3)[0] is None
    assert hs.score_from_outcomes({"hand_scores": {"pellets": "junk"}}, 3)[0] is None


def test_the_census_says_how_many_analysed_videos_wait_for_hand_scores(capsys):
    """A night of held videos must not read as a stalled pipeline."""
    from mousereach.census import runner
    c = {"generated_at": "t", "scan_seconds": 0,
         "totals": {"expected": 10, "finished_files": 5, "unfinished": 5},
         "by_element": {"mousereach": 4, "analyzed": 5}, "by_project": {},
         "held_for_hand_scores": 3,
         "eta": {"basis": "b", "machine_backlog": 4, "human_backlog": 0},
         "diagnostics": {}}
    runner._print_table(c)
    out = capsys.readouterr().out
    assert "of which waiting for hand scores: 3" in out
    out.encode("ascii")
    assert set(runner.HAND_SCORE_HOLD_KINDS) == {"hand_scores_waiting", "hand_score_check_failed"}
