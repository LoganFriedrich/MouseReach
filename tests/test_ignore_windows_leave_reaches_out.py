"""A reviewer's ignore window takes the reaches it touches out of the
kinematics.

WHY (2026-10-30): the review tool's "Ignore windows" box marks frame ranges
where something other than a reach moved the pellet. The answer was saved and
carried onto the segment as ``abnormal_ranges`` -- and read by nothing, so the
windows changed no measurement (ISSUES.md 2026-10-27, item 2). Logan,
2026-10-30: a reach overlapping a window is LEFT OUT of the measurements.
"""
from mousereach.kinematics.core.feature_extractor import split_reaches_by_ignore_windows

R = [
    {"reach_id": 1, "start_frame": 100, "end_frame": 110},
    {"reach_id": 2, "start_frame": 200, "end_frame": 215},
    {"reach_id": 3, "start_frame": 300, "end_frame": 305},
]


def test_no_windows_keeps_every_reach():
    kept, out = split_reaches_by_ignore_windows(R, None)
    assert kept == R and out == []
    kept, out = split_reaches_by_ignore_windows(R, [])
    assert kept == R and out == []


def test_a_reach_overlapping_a_window_is_left_out_and_recorded():
    kept, out = split_reaches_by_ignore_windows(R, [{"start_frame": 205, "end_frame": 260, "reason": ""}])
    assert [r["reach_id"] for r in kept] == [1, 3]
    assert out == [{"reach_id": 2, "start_frame": 200, "end_frame": 215,
                    "ignore_window": {"start_frame": 205, "end_frame": 260}}]


def test_overlap_is_inclusive_by_one_frame_either_side():
    kept, _ = split_reaches_by_ignore_windows(R, [{"start_frame": 110, "end_frame": 150}])
    assert [r["reach_id"] for r in kept] == [2, 3]          # window starts on the reach's last frame
    kept, _ = split_reaches_by_ignore_windows(R, [{"start_frame": 50, "end_frame": 100}])
    assert [r["reach_id"] for r in kept] == [2, 3]          # window ends on the reach's first frame
    kept, _ = split_reaches_by_ignore_windows(R, [{"start_frame": 111, "end_frame": 199}])
    assert [r["reach_id"] for r in kept] == [1, 2, 3]       # the gap between two reaches


def test_a_window_inside_a_reach_removes_it():
    kept, out = split_reaches_by_ignore_windows(R, [{"start_frame": 302, "end_frame": 303}])
    assert [r["reach_id"] for r in kept] == [1, 2] and out[0]["reach_id"] == 3


def test_a_malformed_window_is_ignored_and_a_reversed_one_is_righted():
    kept, _ = split_reaches_by_ignore_windows(R, [{"start_frame": None, "end_frame": 5}, "junk"])
    assert kept == R
    kept, _ = split_reaches_by_ignore_windows(R, [{"start_frame": 215, "end_frame": 200}])
    assert [r["reach_id"] for r in kept] == [1, 3]
