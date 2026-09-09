"""A hand-fixed segmentation must void its downstream, and the gate must not
judge a video whose outcomes predate its segments.

WHY: the fixer saved corrected boundaries and touched nothing else; reaches/
outcomes/assignments/features kept describing the OLD cuts, kinematics-scope
reruns reused them through the stage-reuse check, and the gate re-routed
corrected videos to triage on stale outcomes -- reviewers re-reviewed work
that was already right, and a causal reach could have been attributed to the
wrong pellet (2026-09-08, reviewer diagnosis).
"""
import json
import os
import time

import mousereach.watcher.review_gate as rg
from mousereach.review.fix_segmentation_widget import invalidate_stale_downstream


def test_invalidate_archives_and_removes(tmp_path):
    seg = tmp_path / "20240101_ABC0101_P1_segments.json"
    seg.write_text("{}", encoding="utf-8")
    arch = tmp_path / "arch"
    arch.mkdir()
    made = []
    for suf in ("_reaches.json", "_pellet_outcomes.json", "_reach_assignments.json"):
        f = tmp_path / ("20240101_ABC0101_P1" + suf)
        f.write_text('{"old": true}', encoding="utf-8")
        made.append(f)
    # note: _features.json deliberately absent -- must be skipped, not an error
    removed = invalidate_stale_downstream(seg, "20240101_ABC0101_P1", arch)
    assert len(removed) == 3
    for f in made:
        assert not f.exists()                      # gone from the working dir
        assert (arch / f.name).exists()            # archived, never deleted
    assert seg.exists()                            # the segments file is untouched


def test_gate_returns_stale_when_outcomes_predate_segments(tmp_path, monkeypatch):
    vid = "20240101_ABC0101_P1"
    out = tmp_path / (vid + "_pellet_outcomes.json")
    out.write_text('{"segments": []}', encoding="utf-8")
    old = time.time() - 500
    os.utime(out, (old, old))
    (tmp_path / (vid + "_segments.json")).write_text(
        json.dumps({"overall_confidence": 1.0, "boundaries": [1, 2]}),
        encoding="utf-8")

    class _St:
        seg_failed = False

    monkeypatch.setattr(rg, "triage_status", lambda d, v: _St())
    monkeypatch.setattr(rg, "_gt_certification", lambda v: (False, set()))
    decision, reason, _ = rg.evaluate_gate(vid, tmp_path, "auto_approved")
    assert decision == rg.DECISION_STALE
    assert "predate" in reason


def test_gate_stale_by_in_file_dates_despite_fresh_mtimes(tmp_path, monkeypatch):
    """The triage validation rewrite refreshes every file's mtime seconds
    before the gate, defeating the mtime test -- but the in-file stamps
    survive: corrected_at newer than detected_at is stale, full stop."""
    vid = "20240101_ABC0101_P1"
    (tmp_path / (vid + "_segments.json")).write_text(json.dumps(
        {"overall_confidence": 1.0, "boundaries": [1, 2],
         "boundary_source": "human",
         "corrected_at": "2026-09-01T15:15:00"}), encoding="utf-8")
    (tmp_path / (vid + "_reaches.json")).write_text(json.dumps(
        {"detected_at": "2026-08-11T10:00:00", "segments": []}),
        encoding="utf-8")
    # outcomes written LAST: newest mtime, so the mtime test passes it
    (tmp_path / (vid + "_pellet_outcomes.json")).write_text(
        '{"segments": []}', encoding="utf-8")

    class _St:
        seg_failed = False

    monkeypatch.setattr(rg, "triage_status", lambda d, v: _St())
    monkeypatch.setattr(rg, "_gt_certification", lambda v: (False, set()))
    decision, reason, _ = rg.evaluate_gate(vid, tmp_path, "auto_approved")
    assert decision == rg.DECISION_STALE
    assert "in-file dates" in reason


def test_run_gate_stale_routes_nothing_touches_nothing(tmp_path, monkeypatch):
    vid = "20240101_ABC0101_P1"
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    monkeypatch.setattr(rg, "evaluate_gate",
                        lambda *a, **k: (rg.DECISION_STALE, "why", object()))

    class RecDB:
        calls = []
        def __getattr__(self, name):
            def rec(*a, **k):
                RecDB.calls.append(name)
            return rec

    decision = rg.run_gate(vid, tmp_path, RecDB())
    assert decision == rg.DECISION_STALE
    assert not routed                              # never moved to a queue
    assert not RecDB.calls                         # never touched the db
