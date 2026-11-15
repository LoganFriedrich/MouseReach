"""Ground-truth reaches go to the segment their FRAMES lie in.

They were bucketed by the segment number they were labelled with. After a
re-cut renumbers the segments, that number names other footage: the reach
joined -- or, with an exhaustive reach set, replaced -- the reaches of a
different pellet presentation.
"""
from mousereach.review.truth_resolver import _apply_gt_reaches


def _reaches(segments):
    return {"segments": [{"segment_num": n, "start_frame": a, "end_frame": b,
                          "reaches": [dict(r) for r in reaches]}
                         for n, a, b, reaches in segments]}


# Today's cuts put one more segment in front: the ground truth's segment 2
# (frames 200-299) is today's segment 3.
TODAY = [(1, 50, 99, []), (2, 100, 199, []),
         (3, 200, 299, [{"reach_id": 7, "start_frame": 240, "end_frame": 250}]),
         (4, 300, 399, [])]


def _gt(exhaustive, reaches):
    return {"reaches": {"exhaustive": exhaustive, "reaches": reaches}}


def _seg(out, n):
    return next(s for s in out["segments"] if s["segment_num"] == n)


def test_a_labelled_reach_joins_the_segment_holding_its_frames():
    gt = _gt(False, [{"segment_num": 2, "start_frame": 260, "end_frame": 270}])
    out = _apply_gt_reaches(_reaches(TODAY), gt)
    assert [(r["start_frame"], r["reach_source"]) for r in _seg(out, 3)["reaches"]] == \
        [(240, "algo"), (260, "ground_truth")]
    assert _seg(out, 2)["reaches"] == []


def test_an_exhaustive_reach_set_replaces_the_reaches_of_the_same_footage_only():
    gt = _gt(True, [{"segment_num": 2, "start_frame": 241, "end_frame": 251}])
    out = _apply_gt_reaches(_reaches(TODAY), gt)
    assert [(r["start_frame"], r["reach_source"]) for r in _seg(out, 3)["reaches"]] == \
        [(241, "ground_truth")]
    assert _seg(out, 2)["reaches"] == []


def test_without_frames_on_the_segments_the_label_is_used():
    data = {"segments": [{"segment_num": 2, "reaches": []}]}
    gt = _gt(False, [{"segment_num": 2, "start_frame": 260, "end_frame": 270}])
    out = _apply_gt_reaches(data, gt)
    assert [r["start_frame"] for r in out["segments"][0]["reaches"]] == [260]
