"""mousereach-results: the FILED results at reach, pellet, session, day or
animal level, from MouseReach alone (export/results.py).

WHY: every earlier export read one flat folder (Processing by default), which
filing empties; none had a session or animal level counted from pellets; and
build-database's session "retrieval_rate" was the share of REACH rows whose
pellet was retrieved, so a pellet with no reach vanished from the counts.
These tests build a small filed tree (invented ids) and pin the counting
rules, the level-first column names and the walk of the filed tree.
"""
import json
from pathlib import Path

import pytest


def _reach(rid, causal=False, extent=10.0):
    return {"reach_id": rid, "reach_num": rid, "segment_num": 1, "outcome": None,
            "causal_reach": causal, "start_frame": rid * 10, "end_frame": rid * 10 + 5,
            "max_extent_mm": extent, "duration_frames": 5,
            "extended": {"some_kinematic": 1.5}}


def _seg(num, outcome, reaches, causal_id=None):
    return {"segment_num": num, "start_frame": num * 100, "end_frame": num * 100 + 99,
            "outcome": outcome, "n_reaches": len(reaches), "causal_reach_id": causal_id,
            "reaches": reaches, "attention_score": 50.0, "n_reaches_left_out": 0}


def _write(folder: Path, video: str, segments, session_only=False):
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"video_name": video, "segments": segments,
           "summary": {"session_only": session_only}}
    (folder / f"{video}_features.json").write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def filed(tmp_path):
    root = tmp_path / "Analyzed"
    c = root / "PRJ" / "PRJ_01"
    # Tray 1: retrieved (causal reach 2 of 2), untouched with no reach at all,
    # displaced_sa (causal 3), no_pellet.
    _write(c, "20200101_ABC0101_P1", [
        _seg(1, "retrieved", [_reach(1), _reach(2, causal=True)], causal_id=2),
        _seg(2, "untouched", []),
        _seg(3, "displaced_sa", [_reach(3, causal=True, extent=20.0)], causal_id=3),
        _seg(4, "no_pellet", []),
    ])
    # Tray 2 same day: one pellet moved with NO credited reach.
    _write(c, "20200101_ABC0101_P2", [
        _seg(1, "displaced_outside", [_reach(1)]),
        _seg(2, "retrieved", [_reach(2, causal=True)], causal_id=2),
    ])
    # An easy tray: reaches, no pellet outcomes.
    _write(c, "20200101_ABC0101_E1", [_seg(1, None, [_reach(1), _reach(2)])],
           session_only=True)
    # Another day, another animal.
    _write(c, "20200102_ABC0102_P1", [_seg(1, "untouched", [_reach(1)])])
    # A superseded generation under Analyzed/Archive is NOT a result.
    _write(root / "Archive" / "old", "20200101_ABC0101_P1", [_seg(1, "retrieved", [])])
    return root


def _rows(root, level):
    from mousereach.export.results import load_videos, build_rows
    videos, problems = load_videos(root)
    assert problems == []
    return build_rows(videos, level)


def test_session_counts_pellets_not_reaches(filed):
    rows = {r["session_id"]: r for r in _rows(filed, "session")}
    p1 = rows["20200101_ABC0101_P1"]
    assert p1["session_pellets_presented"] == 3        # no_pellet not presented
    assert p1["session_untouched"] == 1                # a pellet with no reach counts
    assert p1["session_retrieved"] == 1 and p1["session_displaced_sa"] == 1
    assert p1["session_success_rate"] == pytest.approx(1 / 3)
    assert p1["session_contact_rate"] == pytest.approx(2 / 3)
    assert p1["session_n_reaches"] == 3
    assert p1["session_mean_reach_max_extent_mm"] == pytest.approx(40 / 3)
    # identity from the video name
    assert (p1["animal_id"], p1["date"], p1["tray_type"], p1["tray_run"]) == \
        ("ABC0101", "2020-01-01", "P", 1)
    # an easy tray: reaches counted, pellet columns empty
    e1 = rows["20200101_ABC0101_E1"]
    assert e1["session_n_reaches"] == 2 and e1["session_pellets_presented"] is None


def test_day_and_animal_add_up_the_pellets(filed):
    day = {(r["animal_id"], r["date"]): r for r in _rows(filed, "day")}
    d = day[("ABC0101", "2020-01-01")]
    assert d["day_pellets_presented"] == 5
    assert d["day_retrieved"] == 2
    assert d["day_success_rate"] == pytest.approx(2 / 5)
    assert d["day_n_reaches"] == 3 + 2 + 2               # easy tray reaches included
    assert d["day_n_sessions"] == 3
    animals = {r["animal_id"]: r for r in _rows(filed, "animal")}
    assert animals["ABC0102"]["animal_pellets_presented"] == 1
    assert animals["ABC0102"]["animal_success_rate"] == 0
    assert animals["ABC0101"]["animal_n_days"] == 1


def test_pellet_rows_repeat_the_levels_above(filed):
    rows = [r for r in _rows(filed, "pellet") if r["session_id"] == "20200101_ABC0101_P1"]
    assert [r["segment_outcome"] for r in rows] == \
        ["retrieved", "untouched", "displaced_sa", "no_pellet"]
    assert all(r["session_pellets_presented"] == 3 for r in rows)
    assert all(r["day_pellets_presented"] == 5 for r in rows)
    assert all("animal_success_rate" in r for r in rows)
    assert rows[0]["segment_attention_score"] == 50.0


def test_a_reach_carries_its_own_outcome_apart_from_its_pellets(filed):
    rows = _rows(filed, "reach")
    by = {(r["session_id"], r["segment_number"], r["reach_id"]): r for r in rows}
    r1 = by[("20200101_ABC0101_P1", 1, 1)]
    r2 = by[("20200101_ABC0101_P1", 1, 2)]
    assert r1["segment_outcome"] == r2["segment_outcome"] == "retrieved"
    assert r1["reach_outcome"] == "missed" and r2["reach_outcome"] == "retrieved"
    assert r2["reach_is_causal"] is True
    # moved with nobody credited -> undetermined, never a guess
    assert by[("20200101_ABC0101_P2", 1, 1)]["reach_outcome"] == "undetermined"
    # untouched pellet -> missed
    assert by[("20200102_ABC0102_P1", 1, 1)]["reach_outcome"] == "missed"
    # kinematics, level-first
    assert r1["reach_max_extent_mm"] == 10.0 and r1["reach_some_kinematic"] == 1.5
    assert r1["session_success_rate"] == pytest.approx(1 / 3)


def test_column_names_are_level_first(filed):
    from mousereach.export.results import LEVELS
    ident = {"animal_id", "lab_animal_id", "lab_session", "date", "session_id", "tray_type",
             "tray_run", "segment_number", "reach_number", "reach_id", "source_file"}
    for level in LEVELS:
        for r in _rows(filed, level):
            for c in r:
                assert c in ident or c.split("_")[0] in (
                    "reach", "segment", "session", "day", "animal"), (level, c)


def test_the_archive_is_skipped_and_both_formats_are_written(filed, tmp_path):
    from mousereach.export.results import export_results
    res = export_results("session", root=filed, out=tmp_path / "out" / "s.csv")
    assert res["videos"] == 4 and res["rows"] == 4
    import pandas as pd
    assert len(pd.read_csv(res["path"])) == 4
    res = export_results("pellet", root=filed, out=tmp_path / "out" / "p", fmt="xlsx")
    assert res["path"].endswith(".xlsx")
    book = pd.read_excel(res["path"], sheet_name=None)
    assert set(book) == {"results", "columns"}
    meanings = dict(zip(book["columns"]["column"], book["columns"]["meaning"]))
    assert "retrieved / pellets_presented" in meanings["session_success_rate"].lower()


def test_nothing_is_written_into_the_results_it_reads(filed):
    from mousereach.export.results import export_results
    with pytest.raises(ValueError):
        export_results("session", root=filed, out=filed / "PRJ" / "x.csv")


def test_level_names_and_aliases():
    from mousereach.export.results import normalize_level
    assert normalize_level("segment") == "pellet"
    assert normalize_level("Tray") == "session"
    with pytest.raises(ValueError):
        normalize_level("cohort")


def test_the_command_runs_end_to_end(filed, tmp_path, capsys):
    from mousereach.export.results import main
    assert main(["--level", "animal", "--root", str(filed),
                 "--out", str(tmp_path / "a.csv")]) == 0
    assert "[OK] 2 animal row(s) from 4 filed video(s)" in capsys.readouterr().out


def test_a_pellet_moved_by_something_else_is_presented_and_its_reaches_missed(tmp_path):
    """'abnormal' / 'abnormal_exception': a person saw something other than a
    reach move the pellet -- decided, not undetermined; no reach credited."""
    root = tmp_path / "Analyzed"
    _write(root / "PRJ" / "PRJ_02", "20200103_ABC0201_P1", [
        _seg(1, "abnormal_exception", [_reach(1)]),
        _seg(2, "missed", [_reach(2)]),
    ])
    s = _rows(root, "session")[0]
    assert s["session_pellets_presented"] == 2
    assert s["session_moved_not_by_reach"] == 1 and s["session_untouched"] == 1
    assert s["session_undetermined"] == 0 and s["session_success_rate"] == 0
    assert [r["reach_outcome"] for r in _rows(root, "reach")] == ["missed", "missed"]


def test_the_reach_the_segment_names_is_credited_even_without_its_own_flag(tmp_path):
    root = tmp_path / "Analyzed"
    _write(root / "PRJ" / "PRJ_03", "20200104_ABC0301_P1",
           [_seg(1, "retrieved", [_reach(1), _reach(2)], causal_id=2)])
    rows = _rows(root, "reach")
    assert [r["reach_outcome"] for r in rows] == ["missed", "retrieved"]
    assert rows[1]["reach_is_causal"] is True
