"""A video whose hand scores cannot be found is held as a defect, never filed
uncompared.

WHY: in a project whose sessions are all scored by hand (required_projects),
every session was scored at the bench before its videos entered the pipeline.
So when the cohort's sheet has been imported after a video was ready and still
has no scores for it, the scores are MISSING -- a sheet row that is absent or
misnamed, or not matched to the video -- not "never scored". The video used
to go on as never scored and be filed without the comparison, and nothing
ever compared it later. Now it stays held, listed in red for a person, and
goes on by itself once its scores arrive.
"""
import json
import os
import sqlite3

import mousereach.watcher.hand_scores as hs
import mousereach.watcher.route_cli as rc
from mousereach.watcher.db import WatcherDB

VID = "20250709_ABC0207_P1"


def _setup(tmp_path, monkeypatch, manifest):
    f = tmp_path / "hand.csv"
    f.write_text("video_key,pellet,score\n", encoding="utf-8")            # no scores for it
    f.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(hs, "settings", lambda: {"path": str(f), "required_projects": ["ABC"]})
    hs._CACHE.update(key=None)
    folder = tmp_path / "work"
    folder.mkdir()
    (folder / f"{VID}_pellet_outcomes.json").write_text(
        json.dumps({"segments": [{"segment_num": 1, "outcome": "retrieved"}]}), encoding="utf-8")
    monkeypatch.setattr(rc, "answered_segments_of", lambda v, d: set())

    import mousereach.watcher.orchestrator as o
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path / "nas", raising=False)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = WatcherDB(tmp_path / "local.db")
    orch.db.register_video(video_id=VID, source_path="x")
    orch.db.force_state(VID, "processing", reason="test fixture")
    orch.hostname = "HOST"
    con = sqlite3.connect(str(tmp_path / "local.db"))
    con.execute("UPDATE videos SET processing_started_at = ? WHERE video_id = ?",
                ("2026-01-05T10:00:00", VID))
    con.commit()
    con.close()
    return f, folder, orch


def _items(tmp_path):
    return json.loads((tmp_path / "nas" / "attention" / "HOST_hand_scores.json").read_text())["items"]


def _steps(tmp_path):
    con = sqlite3.connect(str(tmp_path / "local.db"))
    rows = con.execute("SELECT status FROM processing_log WHERE video_id = ? AND "
                       "step = 'hand_score_check'", (VID,)).fetchall()
    con.close()
    return [r[0] for r in rows]


def test_scores_missing_after_the_import_hold_the_video_in_red(tmp_path, monkeypatch):
    f, folder, orch = _setup(tmp_path, monkeypatch,
                             {"last_successful_import": {"ABC_02": "2026-01-05T20:00:00"}})
    assert orch._hand_score_precheck(VID, folder) is False          # held, not filed
    assert orch.db.get_video(VID)["state"] == "processing"
    item = _items(tmp_path)[0]
    assert item["kind"] == "hand_scores_missing" and item["needs_person"] is True
    assert "ABC_02" in item["says"] and "no hand scores" in item["says"]
    item["says"].encode("ascii")
    # The remedy is the sheet. A held video sits in this watcher's own
    # Processing folder and nothing renames it there: renamed files would
    # belong to no video and the hold would never clear.
    assert "Fix the sheet" in item["says"]
    assert "name)" not in item["says"] and "rename" not in item["says"].lower()
    # nothing kept as if the session had been settled
    assert "hand_scores" not in json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())
    # logged once, not on every look
    assert orch._hand_score_precheck(VID, folder) is False
    assert _steps(tmp_path) == ["missing"]


def test_a_wait_that_becomes_missing_turns_red_and_scores_that_arrive_release_it(tmp_path, monkeypatch):
    f, folder, orch = _setup(tmp_path, monkeypatch,
                             {"last_successful_import": {"ABC_02": "2026-01-04T20:00:00"}})
    assert orch._hand_score_precheck(VID, folder) is False
    assert _items(tmp_path)[0]["kind"] == "hand_scores_waiting"     # import not yet run: grey
    f.with_suffix(".json").write_text(json.dumps(
        {"last_successful_import": {"ABC_02": "2026-01-05T20:00:00"}}), encoding="utf-8")
    hs._CACHE.update(key=None)
    assert orch._hand_score_precheck(VID, folder) is False
    assert _items(tmp_path)[0]["kind"] == "hand_scores_missing"     # imported, none: red
    assert _steps(tmp_path) == ["waiting", "missing"]
    # a person fixes the sheet; the next import brings the scores
    f.write_text("video_key,pellet,score\n%s,1,2\n" % VID, encoding="utf-8")
    st = f.stat()
    os.utime(f, (st.st_atime, st.st_mtime + 10))
    assert not orch._waiting_for_hand_scores(VID)                    # picked again
    assert orch._hand_score_precheck(VID, folder) is True            # compared, goes on
    assert _items(tmp_path) == []
    kept = json.loads((folder / f"{VID}_pellet_outcomes.json").read_text())["hand_scores"]
    assert kept["verdict"] == "agrees" and kept["pellets"] == {"1": 2}


def test_a_cohort_with_no_sheet_is_held_after_an_import_of_every_sheet(tmp_path, monkeypatch):
    f, folder, orch = _setup(tmp_path, monkeypatch,
                             {"last_successful_import": {}, "sheets_seen": ["XYZ_01"],
                              "full_import_at": "2026-01-05T20:00:00"})
    assert orch._hand_score_precheck(VID, folder) is False
    assert _items(tmp_path)[0]["kind"] == "hand_scores_missing"


def test_a_cohort_with_no_sheet_is_told_to_add_the_sheet_not_to_fix_a_row(tmp_path, monkeypatch):
    """WHY: with no sheet for the cohort, "the ABC_02 sheet was imported ...
    fix the sheet's row" is false -- no such sheet was imported -- and sends
    the person looking for a sheet and a row that do not exist, so the hold
    never clears. The message names the real fix, and so does the log."""
    f, folder, orch = _setup(tmp_path, monkeypatch,
                             {"last_successful_import": {}, "sheets_seen": ["XYZ_01"],
                              "full_import_at": "2026-01-05T20:00:00"})
    assert orch._hand_score_precheck(VID, folder) is False
    item = _items(tmp_path)[0]
    says = item["says"]
    says.encode("ascii")
    assert item["needs_person"] is True and item.get("no_sheet") is True
    assert "no tracking sheet was found for cohort ABC_02" in says
    assert "Add the cohort's tracking sheet" in says
    assert "correct the cohort" in says
    assert "was imported after the video was ready" not in says
    assert "Fix the sheet" not in says and "correct its row" not in says
    con = sqlite3.connect(str(tmp_path / "local.db"))
    msgs = [r[0] for r in con.execute(
        "SELECT message FROM processing_log WHERE video_id = ? AND step = 'hand_score_check'",
        (VID,)).fetchall()]
    con.close()
    assert len(msgs) == 1 and "no tracking sheet found for cohort ABC_02" in msgs[0]
    # The same text after a watcher restart: the kind of defect is kept in
    # the attention list the holds are rebuilt from.
    restored = orch._restore_score_waits()
    assert restored[VID]["missing"] is True and restored[VID]["no_sheet"] is True


def test_a_cohort_whose_sheet_exists_keeps_the_fix_the_row_message(tmp_path, monkeypatch):
    f, folder, orch = _setup(tmp_path, monkeypatch,
                             {"last_successful_import": {"ABC_02": "2026-01-05T20:00:00"},
                              "sheets_seen": ["ABC_02"],
                              "full_import_at": "2026-01-05T20:00:00"})
    assert orch._hand_score_precheck(VID, folder) is False
    item = _items(tmp_path)[0]
    assert not item.get("no_sheet") and "Fix the sheet" in item["says"]
    assert "no tracking sheet" not in item["says"]


def test_the_census_counts_a_missing_score_as_a_hand_score_hold():
    from mousereach.census import runner
    assert "hand_scores_missing" in runner.HAND_SCORE_HOLD_KINDS
