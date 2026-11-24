"""The review window never shows one pellet's algorithm results on another.

In the re-segmentation queue a person saves new cuts into the bundle's
segments file, while the reaches, outcomes and reach assignment there are still
the run on the old cuts. The window took each pellet's results from those files
by segment NUMBER, so after the re-cut it showed (and let a person agree with)
another pellet's outcome and reaches. algo_files_on_other_cuts tells the window
when the files describe other cuts; it then shows no algorithm call.
"""
import mousereach.review.causal_review_io as cio

OLD = [453, 2288, 4123, 5959, 6957, 7094, 7790, 9624]
NEW = [453, 2288, 4123, 5959, 7788, 9624]


def _reaches(cuts, detected_at="2099-01-01T10:00:00"):
    ends = cuts[1:] + [cuts[-1] + 1800]          # the detector's extra last segment
    return {"detected_at": detected_at,
            "segments": [{"segment_num": i + 1, "start_frame": a, "end_frame": b, "reaches": []}
                         for i, (a, b) in enumerate(zip(cuts, ends))]}


def test_reaches_made_on_other_cuts_are_noticed():
    why = cio.algo_files_on_other_cuts({"boundaries": NEW}, _reaches(OLD))
    assert why and "other cuts" in why


def test_reaches_made_on_these_cuts_are_used():
    assert cio.algo_files_on_other_cuts({"boundaries": NEW}, _reaches(NEW)) is None


def test_without_segment_frames_the_times_decide():
    seg = {"boundaries": NEW, "boundary_source": "human", "corrected_at": "2099-01-02T10:00:00"}
    assert cio.algo_files_on_other_cuts(seg, {"detected_at": "2099-01-01T10:00:00",
                                              "reaches": []})
    assert cio.algo_files_on_other_cuts(seg, {"detected_at": "2099-01-03T10:00:00",
                                              "reaches": []}) is None


def _window(reaches_doc):
    from types import SimpleNamespace
    stub = SimpleNamespace(
        _video_stem="20990101_TEST0101_P1", _segments_data={"boundaries": NEW},
        # The old run's results, numbered by the OLD cuts.
        _outcomes_data={"segments": [{"segment_num": n, "outcome": "retrieved"}
                                     for n in range(1, len(OLD))]},
        _assignments_data={"reaches": [{"segment_num": 6, "is_causal": True,
                                        "start_frame": 7539, "end_frame": 7555,
                                        "reach_id": 36}]},
        _reaches_data=reaches_doc,
        _status_label=SimpleNamespace(setText=lambda *_: None),
        _triage_only=False, _triaged_indices=[])
    stub._recompute_triaged_indices = lambda: None
    return stub


def test_the_window_shows_no_algorithm_call_from_other_cuts():
    from mousereach.review.causal_review_widget import CausalReviewWidget
    doc = _reaches(OLD)
    doc["segments"][5]["reaches"] = [{"reach_id": 36, "start_frame": 7539, "end_frame": 7555}]
    stub = _window(doc)
    CausalReviewWidget._build_segment_list(stub)
    segs = stub._segment_list
    assert [s["outcome"] for s in segs] == ["triaged"] * (len(NEW) - 1)   # every pellet asked
    assert all(s["causal_reach"] is None for s in segs)
    # reach 7539-7555 is shown with the pellet whose frames hold it (pellet 4, 5959-7787),
    # not with today's pellet 6 (old pellet 6's number)
    assert [r["reach_id"] for r in segs[3]["all_reaches"]] == [36]
    assert all(not s["all_reaches"] for i, s in enumerate(segs) if i != 3)


def test_the_window_uses_results_made_on_these_cuts():
    from mousereach.review.causal_review_widget import CausalReviewWidget
    stub = _window(_reaches(NEW))
    CausalReviewWidget._build_segment_list(stub)
    assert [s["outcome"] for s in stub._segment_list] == ["retrieved"] * (len(NEW) - 1)
