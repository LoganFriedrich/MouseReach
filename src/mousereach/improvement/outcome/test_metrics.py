"""
Unit tests for outcome classification accuracy metrics.

Tests cover the pure helper functions and the verdict logic.
No I/O, no file system access (except the full-pipeline test which
uses temporary directories).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mousereach.improvement.outcome.metrics import (
    derive_gt_causal_reach,
    _compute_verdict,
    _match_causal_reaches,
    compute_outcome_metrics,
    TOUCHED_OUTCOMES,
)


# ---------------------------------------------------------------------------
# derive_gt_causal_reach
# ---------------------------------------------------------------------------

class TestDeriveGtCausalReach:
    """Tests for derive_gt_causal_reach."""

    def test_interaction_inside_reach_window(self):
        """interaction_frame inside a reach [start, end] -> picks that reach."""
        gt_seg = {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 300}
        gt_reaches = [
            {"reach_id": 1, "segment_num": 1, "start_frame": 250, "end_frame": 260},
            {"reach_id": 2, "segment_num": 1, "start_frame": 290, "end_frame": 310},
            {"reach_id": 3, "segment_num": 1, "start_frame": 400, "end_frame": 420},
        ]
        assert derive_gt_causal_reach(gt_seg, gt_reaches) == 2

    def test_interaction_between_reaches_picks_closer(self):
        """interaction_frame between two reaches -> picks the closer one."""
        gt_seg = {"segment_num": 1, "outcome": "displaced_sa", "interaction_frame": 350}
        gt_reaches = [
            {"reach_id": 1, "segment_num": 1, "start_frame": 300, "end_frame": 310},
            {"reach_id": 2, "segment_num": 1, "start_frame": 400, "end_frame": 420},
        ]
        # reach 1 start=300, dist=50; reach 2 start=400, dist=50 -> tie, first wins
        # Actually: |300 - 350| = 50, |400 - 350| = 50 -> tie, but iteration order
        # matters -- first one encountered wins if equal distance
        result = derive_gt_causal_reach(gt_seg, gt_reaches)
        assert result in (1, 2)  # Either is acceptable for tie

    def test_interaction_closer_to_second_reach(self):
        """interaction_frame closer to second reach -> picks second."""
        gt_seg = {"segment_num": 1, "outcome": "displaced_outside", "interaction_frame": 390}
        gt_reaches = [
            {"reach_id": 1, "segment_num": 1, "start_frame": 300, "end_frame": 310},
            {"reach_id": 2, "segment_num": 1, "start_frame": 400, "end_frame": 420},
        ]
        # |300 - 390| = 90, |400 - 390| = 10 -> reach 2
        assert derive_gt_causal_reach(gt_seg, gt_reaches) == 2

    def test_untouched_returns_none(self):
        """Untouched segment -> no causal reach."""
        gt_seg = {"segment_num": 1, "outcome": "untouched", "interaction_frame": None}
        gt_reaches = [
            {"reach_id": 1, "segment_num": 1, "start_frame": 100, "end_frame": 120},
        ]
        assert derive_gt_causal_reach(gt_seg, gt_reaches) is None

    def test_no_reaches_in_segment(self):
        """No reaches with matching segment_num -> None."""
        gt_seg = {"segment_num": 2, "outcome": "retrieved", "interaction_frame": 300}
        gt_reaches = [
            {"reach_id": 1, "segment_num": 1, "start_frame": 250, "end_frame": 310},
        ]
        assert derive_gt_causal_reach(gt_seg, gt_reaches) is None

    def test_null_interaction_frame(self):
        """Touched outcome with null interaction_frame -> None."""
        gt_seg = {"segment_num": 1, "outcome": "retrieved", "interaction_frame": None}
        gt_reaches = [
            {"reach_id": 1, "segment_num": 1, "start_frame": 100, "end_frame": 120},
        ]
        assert derive_gt_causal_reach(gt_seg, gt_reaches) is None


# ---------------------------------------------------------------------------
# _compute_verdict
# ---------------------------------------------------------------------------

class TestComputeVerdict:
    """Tests for verdict logic."""

    def test_both_untouched(self):
        assert _compute_verdict("untouched", "untouched", False) == "label_correct_untouched"

    def test_label_match_with_reach_match(self):
        assert _compute_verdict("retrieved", "retrieved", True) == "label_and_reach_correct"

    def test_label_match_without_reach_match(self):
        assert _compute_verdict("retrieved", "retrieved", False) == "label_correct_wrong_reach"

    def test_label_wrong(self):
        assert _compute_verdict("retrieved", "untouched", False) == "label_wrong"

    def test_abstained(self):
        assert _compute_verdict("retrieved", "uncertain", False) == "abstained"

    def test_abstained_unknown(self):
        assert _compute_verdict("displaced_sa", "unknown", False) == "abstained"

    def test_both_uncertain_not_abstained(self):
        """If GT is also uncertain, algo isn't abstaining (both uncommitted)."""
        # GT uncertain + algo uncertain -> label match (both same)
        # Since uncertain is not in COMMITTED_OUTCOMES, this goes to label match path
        result = _compute_verdict("uncertain", "uncertain", False)
        # uncertain == uncertain and uncertain is not untouched, causal_match=False
        assert result == "label_correct_wrong_reach"

    def test_displaced_sa_match(self):
        assert _compute_verdict("displaced_sa", "displaced_sa", True) == "label_and_reach_correct"


# ---------------------------------------------------------------------------
# _match_causal_reaches
# ---------------------------------------------------------------------------

class TestMatchCausalReaches:
    """Tests for causal reach start-proximity matching."""

    def test_exact_match(self):
        assert _match_causal_reaches(100, 100) is True

    def test_within_window(self):
        assert _match_causal_reaches(105, 100) is True
        assert _match_causal_reaches(95, 100) is True

    def test_at_boundary(self):
        assert _match_causal_reaches(110, 100) is True
        assert _match_causal_reaches(90, 100) is True

    def test_outside_window(self):
        assert _match_causal_reaches(111, 100) is False
        assert _match_causal_reaches(89, 100) is False

    def test_none_algo(self):
        assert _match_causal_reaches(None, 100) is False

    def test_none_gt(self):
        assert _match_causal_reaches(100, None) is False


# ---------------------------------------------------------------------------
# Full pipeline integration tests (using temp dirs)
# ---------------------------------------------------------------------------

def _make_gt_file(path: Path, video_name: str, segments: list,
                  reaches: list, exhaustive: bool = True):
    """Write a minimal unified GT file."""
    data = {
        "video_name": video_name,
        "type": "unified_ground_truth",
        "schema_version": "1.0",
        "segmentation": {"boundaries": []},
        "reaches": {
            "exhaustive": exhaustive,
            "reaches": reaches,
        },
        "outcomes": {
            "exhaustive": True,
            "n_segments": len(segments),
            "segments": segments,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _make_algo_file(path: Path, video_name: str, segments: list,
                    version: str = "3.0.0"):
    """Write a minimal pellet_outcomes.json."""
    data = {
        "detector_version": version,
        "video_name": video_name,
        "total_frames": 40000,
        "n_segments": len(segments),
        "segments": segments,
        "summary": {},
        "detected_at": "2026-04-24T00:00:00",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _make_algo_reaches_file(path: Path, video_name: str, segments_reaches: list):
    """Write a minimal _reaches.json."""
    data = {
        "detector_version": "6.0",
        "video_name": video_name,
        "segments": segments_reaches,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class TestComputeOutcomeMetricsPipeline:
    """Integration tests for compute_outcome_metrics."""

    def test_perfect_match(self):
        """Every segment: correct label + correct causal reach -> all label_and_reach_correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            gt_dir = d / "gt"
            algo_dir = d / "algo"
            out_dir = d / "out"

            vid = "20250701_TEST_P1"

            # GT: 3 segments, all retrieved
            gt_segments = [
                {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 300, "causal_reach_id": None},
                {"segment_num": 2, "outcome": "displaced_sa", "interaction_frame": 2100, "causal_reach_id": None},
                {"segment_num": 3, "outcome": "untouched", "interaction_frame": None, "causal_reach_id": None},
            ]
            gt_reaches = [
                {"reach_id": 1, "segment_num": 1, "start_frame": 290, "end_frame": 310},
                {"reach_id": 2, "segment_num": 2, "start_frame": 2090, "end_frame": 2110},
            ]
            _make_gt_file(gt_dir / f"{vid}_unified_ground_truth.json", vid, gt_segments, gt_reaches)

            # Algo: matching outcomes and causal reaches
            algo_segments = [
                {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 302,
                 "causal_reach_id": 1, "causal_reach_frame": 302},
                {"segment_num": 2, "outcome": "displaced_sa", "interaction_frame": 2105,
                 "causal_reach_id": 2, "causal_reach_frame": 2105},
                {"segment_num": 3, "outcome": "untouched", "interaction_frame": None,
                 "causal_reach_id": None, "causal_reach_frame": None},
            ]
            _make_algo_file(algo_dir / f"{vid}_pellet_outcomes.json", vid, algo_segments)

            # Algo reaches with matching starts
            algo_reaches_segs = [
                {"segment_num": 1, "reaches": [
                    {"reach_id": 1, "start_frame": 292, "end_frame": 312},
                ]},
                {"segment_num": 2, "reaches": [
                    {"reach_id": 2, "start_frame": 2093, "end_frame": 2113},
                ]},
            ]
            _make_algo_reaches_file(algo_dir / f"{vid}_reaches.json", vid, algo_reaches_segs)

            scalars = compute_outcome_metrics(gt_dir, algo_dir, out_dir)

            assert scalars["n_videos"] == 1
            assert scalars["n_segments_paired"] == 3
            # seg 1: retrieved+retrieved, reach matched (|292-290|=2 <= 10) -> label_and_reach_correct
            # seg 2: displaced_sa+displaced_sa, reach matched (|2093-2090|=3 <= 10) -> label_and_reach_correct
            # seg 3: untouched+untouched -> label_correct_untouched
            assert scalars["causal_reach"]["overall"].get("label_and_reach_correct", 0) == 2
            assert scalars["outcome_label"]["strict_accuracy"] is not None

    def test_correct_label_wrong_reach(self):
        """Same label but algo points at a different reach -> label_correct_wrong_reach."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            gt_dir = d / "gt"
            algo_dir = d / "algo"
            out_dir = d / "out"

            vid = "20250701_TEST_P1"

            gt_segments = [
                {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 300, "causal_reach_id": None},
            ]
            gt_reaches = [
                {"reach_id": 1, "segment_num": 1, "start_frame": 290, "end_frame": 310},
                {"reach_id": 2, "segment_num": 1, "start_frame": 400, "end_frame": 420},
            ]
            _make_gt_file(gt_dir / f"{vid}_unified_ground_truth.json", vid, gt_segments, gt_reaches)

            # Algo says retrieved but causal_reach points at reach 2 (start=400)
            algo_segments = [
                {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 302,
                 "causal_reach_id": 2, "causal_reach_frame": 410},
            ]
            _make_algo_file(algo_dir / f"{vid}_pellet_outcomes.json", vid, algo_segments)

            # Algo reach 2 starts at 500 (way off from GT reach 1 at 290)
            algo_reaches_segs = [
                {"segment_num": 1, "reaches": [
                    {"reach_id": 2, "start_frame": 500, "end_frame": 520},
                ]},
            ]
            _make_algo_reaches_file(algo_dir / f"{vid}_reaches.json", vid, algo_reaches_segs)

            scalars = compute_outcome_metrics(gt_dir, algo_dir, out_dir)

            assert scalars["causal_reach"]["overall"].get("label_correct_wrong_reach", 0) == 1

    def test_wrong_label(self):
        """Algo predicts untouched when GT says retrieved -> label_wrong."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            gt_dir = d / "gt"
            algo_dir = d / "algo"
            out_dir = d / "out"

            vid = "20250701_TEST_P1"

            gt_segments = [
                {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 300, "causal_reach_id": None},
            ]
            gt_reaches = [
                {"reach_id": 1, "segment_num": 1, "start_frame": 290, "end_frame": 310},
            ]
            _make_gt_file(gt_dir / f"{vid}_unified_ground_truth.json", vid, gt_segments, gt_reaches)

            algo_segments = [
                {"segment_num": 1, "outcome": "untouched", "interaction_frame": None,
                 "causal_reach_id": None, "causal_reach_frame": None},
            ]
            _make_algo_file(algo_dir / f"{vid}_pellet_outcomes.json", vid, algo_segments)

            scalars = compute_outcome_metrics(gt_dir, algo_dir, out_dir)

            # This is a label_wrong since GT=retrieved, algo=untouched
            # But it's in the confusion matrix, not causal_reach (label mismatch)
            cm = scalars["outcome_label"]["confusion_matrix"]
            assert cm.get("retrieved__untouched", 0) == 1

    def test_abstention(self):
        """Algo predicts uncertain when GT committed -> abstained."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            gt_dir = d / "gt"
            algo_dir = d / "algo"
            out_dir = d / "out"

            vid = "20250701_TEST_P1"

            gt_segments = [
                {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 300, "causal_reach_id": None},
            ]
            gt_reaches = [
                {"reach_id": 1, "segment_num": 1, "start_frame": 290, "end_frame": 310},
            ]
            _make_gt_file(gt_dir / f"{vid}_unified_ground_truth.json", vid, gt_segments, gt_reaches)

            algo_segments = [
                {"segment_num": 1, "outcome": "uncertain", "interaction_frame": None,
                 "causal_reach_id": None, "causal_reach_frame": None},
            ]
            _make_algo_file(algo_dir / f"{vid}_pellet_outcomes.json", vid, algo_segments)

            scalars = compute_outcome_metrics(gt_dir, algo_dir, out_dir)

            assert scalars["outcome_label"]["abstention_rate"] == 1.0

    def test_untouched_both(self):
        """Untouched in both -> label_correct_untouched, no reach evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            gt_dir = d / "gt"
            algo_dir = d / "algo"
            out_dir = d / "out"

            vid = "20250701_TEST_P1"

            gt_segments = [
                {"segment_num": 1, "outcome": "untouched", "interaction_frame": None, "causal_reach_id": None},
            ]
            gt_reaches = []
            _make_gt_file(gt_dir / f"{vid}_unified_ground_truth.json", vid, gt_segments, gt_reaches)

            algo_segments = [
                {"segment_num": 1, "outcome": "untouched", "interaction_frame": None,
                 "causal_reach_id": None, "causal_reach_frame": None},
            ]
            _make_algo_file(algo_dir / f"{vid}_pellet_outcomes.json", vid, algo_segments)

            scalars = compute_outcome_metrics(gt_dir, algo_dir, out_dir)

            # Untouched segments don't go into causal_reach breakdown
            assert scalars["causal_reach"]["overall"] == {}
            assert scalars["outcome_label"]["strict_accuracy"] == 1.0

    def test_abstention_accuracy_math(self):
        """Verify strict vs committed denominators differ correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            gt_dir = d / "gt"
            algo_dir = d / "algo"
            out_dir = d / "out"

            vid = "20250701_TEST_P1"

            # 4 segments: 2 correct, 1 wrong, 1 abstained
            gt_segments = [
                {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 300, "causal_reach_id": None},
                {"segment_num": 2, "outcome": "untouched", "interaction_frame": None, "causal_reach_id": None},
                {"segment_num": 3, "outcome": "displaced_sa", "interaction_frame": 2100, "causal_reach_id": None},
                {"segment_num": 4, "outcome": "retrieved", "interaction_frame": 3500, "causal_reach_id": None},
            ]
            gt_reaches = [
                {"reach_id": 1, "segment_num": 1, "start_frame": 290, "end_frame": 310},
                {"reach_id": 3, "segment_num": 3, "start_frame": 2090, "end_frame": 2110},
                {"reach_id": 5, "segment_num": 4, "start_frame": 3490, "end_frame": 3510},
            ]
            _make_gt_file(gt_dir / f"{vid}_unified_ground_truth.json", vid, gt_segments, gt_reaches)

            algo_segments = [
                # Seg 1: correct label, correct reach
                {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 302,
                 "causal_reach_id": 1, "causal_reach_frame": 302},
                # Seg 2: correct untouched
                {"segment_num": 2, "outcome": "untouched", "interaction_frame": None,
                 "causal_reach_id": None, "causal_reach_frame": None},
                # Seg 3: wrong label
                {"segment_num": 3, "outcome": "untouched", "interaction_frame": None,
                 "causal_reach_id": None, "causal_reach_frame": None},
                # Seg 4: abstained
                {"segment_num": 4, "outcome": "uncertain", "interaction_frame": None,
                 "causal_reach_id": None, "causal_reach_frame": None},
            ]
            _make_algo_file(algo_dir / f"{vid}_pellet_outcomes.json", vid, algo_segments)

            algo_reaches_segs = [
                {"segment_num": 1, "reaches": [
                    {"reach_id": 1, "start_frame": 292, "end_frame": 312},
                ]},
            ]
            _make_algo_reaches_file(algo_dir / f"{vid}_reaches.json", vid, algo_reaches_segs)

            scalars = compute_outcome_metrics(gt_dir, algo_dir, out_dir)

            # Total: 4 segments
            # Correct (strict): seg1 (label_and_reach_correct) + seg2 (label_correct_untouched) = 2
            # Abstained: seg4 = 1
            # Committed: 4 - 1 = 3
            # strict_accuracy = 2/4 = 0.5
            # committed_accuracy = 2/3 = 0.6667
            # abstention_rate = 1/4 = 0.25
            assert scalars["n_segments_paired"] == 4
            assert scalars["outcome_label"]["strict_accuracy"] == 0.5
            assert abs(scalars["outcome_label"]["committed_accuracy"] - 0.6667) < 0.001
            assert scalars["outcome_label"]["abstention_rate"] == 0.25
