"""Census traps -- each found by measurement against a live corpus, each of
which silently corrupted a census before it was guarded. Identifiers here are
SYNTHETIC (project ABC/XYZ, cohort/subject 0101, trays P/E/F): assert the
behaviour, never a corpus count.
"""
import json

import pytest

from mousereach.census.expected_sessions import (
    classify_trailer, expected_sessions, select_sources, session_key,
)
from mousereach.census.locate_sessions import (
    DatabaseViewUnavailable, invariant_violations, resolve_elements, tally,
    tray_from_stem,
)
from mousereach.census.review_completeness import (
    completeness_of_document, scan_queue,
)
from mousereach.census.runner import bundles_in, estimate_eta, ids_in_dir


# --- Trap 1: the dead completion field ------------------------------------
def test_completeness_ignores_answers_reviewed():
    """answers.reviewed has never been written by anything; completeness must
    come from human.outcome or it reports 'no reviews exist' forever."""
    doc = {"segments": (
        [{"answers": {"reviewed": False}, "human": {"outcome": "retrieved"}}] * 15
        + [{"answers": {"reviewed": False}, "human": {"outcome": None}}] * 5
    )}
    answered, total, _, _ = completeness_of_document(doc)
    assert (answered, total) == (15, 20)


# --- Trap 2: blank camera slots are not sessions --------------------------
def _offspring_two_blanks(stem):
    out = []
    for i in range(1, 9):
        blank = i in (4, 8)
        out.append({"position": i, "animal_id": "ABC%04d" % i,
                    "offspring_stem": None if blank else "20240101_ABC%04d_P1" % i,
                    "blank": blank})
    return out


def test_blank_slots_excluded_and_counted(tmp_path):
    (tmp_path / "20240101_a,b,c,d,e,f,g,h_P1.mkv").touch()
    exp = expected_sessions([tmp_path], _offspring_two_blanks)
    assert len(exp["sessions"]) == 6
    assert exp["diagnostics"]["blank_camera_slots_excluded"] == 2


# --- Trap 3: the intake folder is not a backlog signal --------------------
def test_artifact_only_sessions_join_the_denominator(tmp_path):
    """(b) a session with artifacts but NO surviving collage must still count."""
    exp = expected_sessions([tmp_path], lambda s: [],
                            found_sessions={"20240101_XYZ0101_P1"})
    assert "20240101_XYZ0101_P1" in exp["sessions"]
    assert exp["sessions"]["20240101_XYZ0101_P1"]["source"] == "artifact"
    assert exp["diagnostics"]["sessions_with_artifact_but_no_collage"] == 1


def test_finished_children_are_not_not_started():
    """(a) a lingering collage's finished children resolve past unanalyzed."""
    sid = "20240101_ABC0101_P1"
    els = resolve_elements(
        {sid: {"tray": "P"}}, {sid: {"_reaches.json", "_pellet_outcomes.json",
                                     "_reach_assignments.json", "_features.json"}},
        {}, in_database=None,
        analysis_outputs=("_reaches.json", "_pellet_outcomes.json",
                          "_reach_assignments.json", "_features.json"))
    assert els[sid] != "unanalyzed"


def test_extra_collages_join_the_denominator(tmp_path):
    """Collages living outside the scanned roots (e.g. retired into the final
    output tree, or archived uncropped) still contribute their children."""
    extra = tmp_path / "20240102_a,b_F1.mkv"
    extra.touch()

    def off(stem):
        return [{"position": 1, "animal_id": "ABC0101",
                 "offspring_stem": stem.replace("a,b", "ABC0101"),
                 "blank": False}]

    exp = expected_sessions([], off, extra_collages=[extra])
    assert "20240102_ABC0101_F1" in exp["sessions"]


# --- Trap 4: a collage that cannot be parsed is a problem, not an absence -
def test_unparsable_collage_surfaces(tmp_path):
    name = "20240101_ABC0101,ABC0102,ABC0103,ABC0104,ABC0105,ABC0106,ABC0107_P1.mkv"
    (tmp_path / name).touch()
    exp = expected_sessions([tmp_path], lambda s: [])
    assert name in exp["diagnostics"]["collages_that_parsed_to_nothing"]


# --- Trap 5: trailer-level malformations stay visible ---------------------
# (id-level malformations -- fused ids, wrong digit counts -- are the
#  provenance parser's domain and are tested with it; the census surfaces
#  them through trap 4's parse-failure channel.)
def test_split_recording_and_dual_tray_are_variants():
    assert classify_trailer("P1,1") == "variant"
    assert classify_trailer("P1,2") == "variant"
    assert classify_trailer("XP3,YP1") == "variant"
    assert classify_trailer("P1") == "clean"


def test_variant_files_collapse_when_a_clean_twin_exists(tmp_path):
    a = tmp_path / "20240101_ABC0101,x_P1.mkv"
    b = tmp_path / "20240101_ABC0101,x_P1 uncropped.mkv"
    a.touch(), b.touch()
    kept, dropped = select_sources([a, b])
    assert kept == [a]
    assert dropped.get("variant") == 1


# --- Trap 6: never trust the tray column alone ----------------------------
def test_tray_falls_back_to_the_stem():
    assert tray_from_stem("20240101_ABC0101_P1", db_tray=None) == "P"
    assert tray_from_stem("20240101_ABC0101_P1", db_tray="E") == "E"


# --- Trap 7: a queue beats the final output tree --------------------------
def test_queue_outranks_analyzed():
    sid = "20240101_ABC0101_P1"
    req = ("_reaches.json", "_pellet_outcomes.json",
           "_reach_assignments.json", "_features.json")
    els = resolve_elements({sid: {"tray": "P"}}, {sid: set(req)},
                           {"triage": {sid}},
                           in_database=lambda s: True, analysis_outputs=req)
    assert els[sid] == "triage"


# --- Trap 8: outcome-free trays terminate legitimately --------------------
def test_outcome_free_tray_reaches_session_only():
    sid = "20240101_ABC0101_E1"
    els = resolve_elements({sid: {"tray": "E"}}, {sid: {"_reaches.json"}}, {},
                           in_database=lambda s: True,
                           outcome_free_trays=("E", "F"),
                           analysis_outputs=("_reaches.json",
                                             "_pellet_outcomes.json"),
                           session_only_outputs=("_reaches.json",))
    assert els[sid] == "session_only"


# --- Trap 9: count videos, not files --------------------------------------
def test_ids_in_dir_counts_videos_not_files(tmp_path):
    for i in range(5):
        (tmp_path / ("20240101_ABC010%d_P1.mp4" % i)).touch()
        (tmp_path / ("20240101_ABC010%d_P1.quarantine.json" % i)).touch()
    assert len(ids_in_dir(tmp_path)) == 5


def test_ids_in_dir_normalizes_dlc_and_full_names(tmp_path):
    (tmp_path / "20240101_ABC0101_P1_full.mp4").touch()
    (tmp_path / "20240101_ABC0101_P1DLCsomething.mp4").touch()
    assert ids_in_dir(tmp_path) == {"20240101_ABC0101_P1"}


# --- Trap 10: scratch directories are not review bundles ------------------
def test_non_bundle_directories_are_skipped(tmp_path):
    (tmp_path / "20240101_ABC0101_P1").mkdir()
    (tmp_path / ".scratch").mkdir()
    (tmp_path / "_Problematic").mkdir()
    (tmp_path / "notes").mkdir()
    assert bundles_in(tmp_path) == {"20240101_ABC0101_P1"}


def test_scan_queue_skips_dot_dirs(tmp_path):
    b = tmp_path / "20240101_ABC0101_P1"
    b.mkdir()
    (b / "20240101_ABC0101_P1_causal_review.json").write_text(json.dumps(
        {"segments": [{"human": {"outcome": "retrieved"}}]}), encoding="utf-8")
    (tmp_path / ".scratch").mkdir()
    rows, skipped = scan_queue(tmp_path)
    assert [r["stem"] for r in rows] == ["20240101_ABC0101_P1"]
    assert ".scratch" not in skipped


# --- The invariant refuses to lie -----------------------------------------
def test_invariant_requires_a_database_view():
    """Without the database the 'violations' would be the whole finished
    corpus -- wrong in the alarming direction. Refusing is the only safe
    answer, and it must be structural (an exception), not a flag."""
    with pytest.raises(DatabaseViewUnavailable):
        invariant_violations({}, {}, {}, in_database=None)


def test_invariant_flags_finished_but_not_landed():
    sid = "20240101_ABC0101_P1"
    req = ("_reaches.json", "_features.json")
    idx = {sid: set(req)}
    els = resolve_elements({sid: {"tray": "P"}}, idx, {},
                           in_database=lambda s: False, analysis_outputs=req)
    v = invariant_violations({sid: {"tray": "P"}}, els, idx,
                             in_database=lambda s: False, analysis_outputs=req)
    assert sid in v
    v2 = invariant_violations({sid: {"tray": "P"}}, els, idx,
                              in_database=lambda s: True, analysis_outputs=req)
    assert sid not in v2


def test_tally_without_database_view_reports_unavailable_not_zero():
    t = tally({"s": {}}, {"s": "mousereach"}, database_view=False)
    assert t["by_element"].get("analyzed") is None
    assert t["caveats"]


# --- Estimates never invent a pace ----------------------------------------
def test_eta_without_pace_reports_no_dates():
    eta = estimate_eta({"unanalyzed": 100, "triage": 10}, None, None, 14)
    assert eta["machine_backlog"] == 100
    assert eta["human_backlog"] == 10
    assert "machine_date" not in eta and "human_date" not in eta


def test_eta_with_pace_projects_dates():
    eta = estimate_eta({"unanalyzed": 100}, 10.0, 5.0, 14)
    assert eta["machine_days"] == 10.0
    assert eta["machine_date"]


# --- Session identity survives which file describes it --------------------
def test_session_key_collapses_clean_and_variant():
    assert (session_key("20240101", "a,b", "P1")
            == session_key("20240101", "a,b", "P1 uncropped"))
