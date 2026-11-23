"""A person's cuts (boundary_source "human") or a person's confirmation of the
cuts (cuts_confirmed) are never questioned by an automated check of segment
length, spacing or count (owner, 2026-11-23; causal_review_io.cuts_taken_as_made).
"""
import json

from mousereach.pipeline.triage import check_structural
from mousereach.review.causal_review_io import cuts_confirmed, cuts_taken_as_made
from mousereach.review.triage_status import _too_short_segments, segmentation_failed
from mousereach.segmentation.core.triage import (classify_segments,
                                                  classify_segments_graduated)

STEM = "20250627_CNT0103_P3"
CONFIRMED = {"by": "owner", "at": "2026-11-23T12:00:00", "note": "irregular recording"}


def _cuts(n=21, short_at=None):
    """n cuts ~1800 frames apart; with ``short_at``, a 3-frame segment there."""
    b = [100 + 1800 * i for i in range(n)]
    if short_at is not None:
        b[short_at + 1] = b[short_at] + 2
        b = sorted(set(b))
    return b


def _algo(boundaries, **extra):
    doc = {"boundaries": boundaries, "overall_confidence": 0.0,
           "anomalies": ["Very short interval B5->B6: 2 frames (0.0s)"]}
    doc.update(extra)
    return doc


def test_rule():
    assert not cuts_taken_as_made(_algo(_cuts()))
    assert cuts_taken_as_made(_algo(_cuts(), boundary_source="human"))
    assert cuts_taken_as_made(_algo(_cuts(), cuts_confirmed=CONFIRMED))
    # A confirmation must say who made it.
    assert not cuts_confirmed(_algo(_cuts(), cuts_confirmed={"note": "x"}))


def test_too_short_segment_not_raised_on_person_cuts():
    b = _cuts(short_at=4)
    assert _too_short_segments(_algo(b)), "control: algorithm cuts are questioned"
    assert _too_short_segments(_algo(b, boundary_source="human")) == {}
    assert _too_short_segments(_algo(b, cuts_confirmed=CONFIRMED)) == {}


def test_segmentation_not_failed_on_person_cuts():
    assert segmentation_failed(_algo(_cuts()))
    assert not segmentation_failed(_algo(_cuts(), boundary_source="human"))
    assert not segmentation_failed(_algo(_cuts(), cuts_confirmed=CONFIRMED))


def _structural(tmp_path, doc):
    (tmp_path / f"{STEM}_segments.json").write_text(json.dumps(doc), encoding="utf-8")
    for s in ("reaches", "pellet_outcomes"):
        (tmp_path / f"{STEM}_{s}.json").write_text("{}", encoding="utf-8")
    return [f.description for f in check_structural(STEM, tmp_path)]


def test_boundary_count_not_raised_on_person_cuts(tmp_path):
    few = _cuts(n=12)
    assert any("boundary count" in d for d in _structural(tmp_path, _algo(few)))
    assert _structural(tmp_path, _algo(few, boundary_source="human")) == []
    assert _structural(tmp_path, _algo(few, cuts_confirmed=CONFIRMED)) == []


def test_segmentation_triage_classifiers_accept_person_cuts(tmp_path):
    p = tmp_path / f"{STEM}_segments.json"
    p.write_text(json.dumps(_algo(_cuts(n=12))), encoding="utf-8")
    assert classify_segments(p, check_gt=False)[0] == "failed"
    assert classify_segments_graduated(p)[0] == "failed"
    p.write_text(json.dumps(_algo(_cuts(n=12), cuts_confirmed=CONFIRMED)), encoding="utf-8")
    assert classify_segments(p, check_gt=False)[0] == "good"
    assert classify_segments_graduated(p)[0] == "auto_approved"
