"""A ground-truth file holding only some cuts places each cut by its index.

"Save as Ground Truth" writes only the boundaries a person determined. Numbered
by their position in that shorter list, cuts 3, 4 and 9 became "segment 1"
(cuts 3-4) and "segment 2" (cuts 4-9), so a pellet-1 outcome was matched to
pellet 4's footage.
"""
from mousereach.review.causal_review_io import gt_segments, index_gt_outcomes_by_segment

# Cuts every 100 frames: index i at frame 100 * (i + 1); segment N = cut N-1 .. cut N.
CURRENT = [{"segment_num": n, "start_frame": 100 * n, "end_frame": 100 * (n + 1) - 1}
           for n in range(1, 11)]


def _gt():
    return {
        "segmentation": {"boundaries": [
            {"index": i, "frame": 100 * (i + 1), "determined": True} for i in (3, 4, 9)]},
        "outcomes": {"segments": [
            {"segment_num": 1, "outcome": "untouched", "determined": True},
            {"segment_num": 4, "outcome": "retrieved", "determined": True},
        ]},
    }


def test_only_segments_with_both_cuts_have_frames():
    assert gt_segments(_gt()) == [{"segment_num": 4, "start_frame": 400, "end_frame": 499}]


def test_outcomes_land_on_their_own_pellets():
    by_seg, _ = index_gt_outcomes_by_segment(_gt(), CURRENT)
    assert by_seg[4]["outcome"] == "retrieved"
    assert by_seg[1]["outcome"] == "untouched"


def test_a_file_without_indexes_is_numbered_by_position():
    gt = {"segmentation": {"boundaries": [{"frame": 100}, {"frame": 200}, {"frame": 300}]}}
    assert [s["segment_num"] for s in gt_segments(gt)] == [1, 2]
