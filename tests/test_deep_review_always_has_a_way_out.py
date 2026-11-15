"""Every bundle in the re-segmentation queue has a visible way out.

A pillar video leaves the queue only after a person has set or confirmed its
cuts since it arrived (review/resegmented.py). Three kinds of bundle used to
have no tool that could give it that:

- cuts set by hand on an EARLIER visit, then sent back: the fixer hid it
  (``boundary_source == "human"``) although those cuts no longer count;
- sent for a reason that is not segmentation (a flagged session): the fixer
  never listed it, although the proof is still required;
- re-cut and fully answered but sent for another reason: the release check
  held it back, the window would not open it and the fixer would not list it.

And a ground-truth file saved by the current ground-truth tool could never
count as the proof, because only the old field name was read.
"""
import json

from mousereach.review.fix_segmentation_widget import FixSegmentationWidget
from mousereach.review.release_cli import classify_queue
from mousereach.review.resegmented import resegmented_since_entry

STEM = "20990101_ABC9901_P1"
ROUTED = "2099-01-10T09:00:00"
BEFORE = "2099-01-05T09:00:00"
AFTER = "2099-01-11T09:00:00"


def _bundle(root, stem=STEM, reason="reviewer escalated from triage: bad segmentation",
            seg_extra=None, review=None, gt_bounds=None):
    b = root / stem
    b.mkdir(parents=True)
    (b / f"{stem}_routing.json").write_text(
        json.dumps({"routed_at": ROUTED, "routed_reason": reason}), encoding="utf-8")
    seg = {"boundaries": [100, 200, 300]}
    seg.update(seg_extra or {})
    (b / f"{stem}_segments.json").write_text(json.dumps(seg), encoding="utf-8")
    if review is not None:
        (b / f"{stem}_causal_review.json").write_text(
            json.dumps({"segments": review}), encoding="utf-8")
    if gt_bounds is not None:
        (b / f"{stem}_unified_ground_truth.json").write_text(
            json.dumps({"segmentation": {"boundaries": gt_bounds}}), encoding="utf-8")
    return b


def _listed(bundle):
    """What the segmentation fixer decides for this bundle (self is unused)."""
    seg = json.loads((bundle / f"{bundle.name}_segments.json").read_text(encoding="utf-8"))
    return FixSegmentationWidget._needs_reseg(None, bundle, seg)


def test_fixer_lists_a_video_whose_hand_cuts_predate_its_return(tmp_path):
    b = _bundle(tmp_path, seg_extra={"boundary_source": "human", "corrected_at": BEFORE})
    assert not resegmented_since_entry(b, STEM)[0]
    assert _listed(b)


def test_fixer_lists_a_flagged_session_with_machine_cuts(tmp_path):
    b = _bundle(tmp_path, reason="reviewer flagged session ABC9901_20990101")
    assert _listed(b)


def test_fixer_lists_a_quality_hold(tmp_path):
    assert _listed(_bundle(tmp_path, reason="qc_needs_review"))


def test_fixer_does_not_list_a_video_recut_since_it_arrived(tmp_path):
    b = _bundle(tmp_path, seg_extra={"boundary_source": "human", "corrected_at": AFTER})
    assert not _listed(b)


def test_easy_and_flat_trays_are_never_listed(tmp_path):
    from mousereach.review.resegmented import needs_recut
    stem = "20990101_ABC9901_E1"
    b = _bundle(tmp_path, stem=stem)
    assert needs_recut(b, stem) == (False, "Easy/Flat tray: not segmented")


def test_ground_truth_from_the_current_tool_counts_as_the_recut(tmp_path):
    every = [{"index": i, "frame": f, "determined": True, "determined_at": AFTER}
             for i, f in enumerate((100, 200, 300))]
    b = _bundle(tmp_path, gt_bounds=every)
    assert resegmented_since_entry(b, STEM)[0]
    assert not _listed(b)


def test_ground_truth_with_an_undetermined_boundary_does_not_count(tmp_path):
    some = [{"frame": 100, "determined": True, "determined_at": AFTER},
            {"frame": 200, "determined": False, "determined_at": None}]
    assert not resegmented_since_entry(_bundle(tmp_path, gt_bounds=some), STEM)[0]


def test_ground_truth_determined_before_the_video_arrived_does_not_count(tmp_path):
    every = [{"frame": f, "determined": True, "determined_at": BEFORE} for f in (100, 200)]
    assert not resegmented_since_entry(_bundle(tmp_path, gt_bounds=every), STEM)[0]


def _answered(n):
    return {"segment_num": n, "human": {"outcome": "retrieved"}, "answers": {}}


def test_recut_and_fully_answered_is_complete_whatever_it_was_sent_for(tmp_path):
    _bundle(tmp_path, reason="qc_needs_review",
            seg_extra={"boundary_source": "human", "corrected_at": AFTER},
            review=[_answered(1), _answered(2)])
    cls = classify_queue(tmp_path)
    assert cls["complete"] == [(STEM, 2, 2)]
    assert cls["fixed_held"] == []


def test_recut_for_another_reason_with_open_pellets_is_still_held(tmp_path):
    _bundle(tmp_path, reason="qc_needs_review",
            seg_extra={"boundary_source": "human", "corrected_at": AFTER},
            review=[_answered(1), {"segment_num": 2, "answers": {"reviewed": False},
                                   "human": {"outcome": None}}])
    cls = classify_queue(tmp_path)
    assert [s for s, _ in cls["fixed_held"]] == [STEM]
    assert cls["complete"] == []
