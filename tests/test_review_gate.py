"""Gate precedence: a human's completed deep review outranks the QC self-check.

Logan's rule (2026-08-31): "if a human completed all that needed to be
reviewed, then it moves on." Without it, a deterministic QC critical (e.g. an
unstable landmark ruler, re-raised identically by every re-run) ping-ponged
cleared videos between deep review and reprocessing forever.
"""
import json

from mousereach.watcher.review_gate import (
    DECISION_CLEAN, DECISION_DEEP, evaluate_gate,
)

VID = "20990101_CNT9901_P1"


def _write_ok_seg(d):
    # overall_confidence > 0 and no anomalies -> segmentation not failed;
    # no outcome/assignment docs -> nothing triaged, nothing unresolved.
    (d / f"{VID}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.9, "boundaries": [100, 200]}),
        encoding="utf-8")


def test_qc_needs_review_holds_without_a_clear(tmp_path):
    _write_ok_seg(tmp_path)
    decision, reason, _ = evaluate_gate(VID, tmp_path, qc_verdict="needs_review")
    assert decision == DECISION_DEEP
    assert reason == "qc_needs_review"


def test_human_clear_outranks_qc_needs_review(tmp_path):
    _write_ok_seg(tmp_path)
    (tmp_path / f"{VID}_deep_review_cleared.json").write_text(
        json.dumps({"cleared_by": "test", "cleared_at": "2026-08-31"}),
        encoding="utf-8")
    decision, reason, _ = evaluate_gate(VID, tmp_path, qc_verdict="needs_review")
    assert decision == DECISION_CLEAN


def test_auto_approved_still_clean(tmp_path):
    _write_ok_seg(tmp_path)
    decision, reason, _ = evaluate_gate(VID, tmp_path, qc_verdict="auto_approved")
    assert decision == DECISION_CLEAN
    assert reason == "clean"


def _write_failed_seg(d):
    # overall_confidence <= 0 == uniform fallback == segmentation failed.
    (d / f"{VID}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.0, "boundaries": [100, 200]}),
        encoding="utf-8")


def test_seg_failed_holds_without_a_clear(tmp_path):
    _write_failed_seg(tmp_path)
    decision, reason, _ = evaluate_gate(VID, tmp_path, qc_verdict="auto_approved")
    assert decision == DECISION_DEEP
    assert reason == "segmentation_failed"


def test_human_clear_outranks_seg_failed(tmp_path):
    # The seg self-check re-raises identically on every re-run; the human's
    # completed deep review outranks it, same rule as the QC branch.
    _write_failed_seg(tmp_path)
    (tmp_path / f"{VID}_deep_review_cleared.json").write_text(
        json.dumps({"cleared_by": "test", "cleared_at": "2026-09-01"}),
        encoding="utf-8")
    decision, reason, _ = evaluate_gate(VID, tmp_path, qc_verdict="auto_approved")
    assert decision == DECISION_CLEAN
