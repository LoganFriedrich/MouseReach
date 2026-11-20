"""Ground truth inside a triage bundle answers its flagged pellets (MRR-03d).

Owner decision 2026-11-19: "an answer a human gives is always true so long as
it was given about the correct pellet/segment". The ground-truth tool's
``*_unified_ground_truth.json`` is a person's answer. Before this, the release
check read ground truth only through the corpus index, which does not cover
the shared review queues, so a triage bundle carrying its own ground truth
was neither released nor shown (the window skips a bundle with ground truth).

Now:
- a flagged pellet the in-bundle ground truth determined counts as answered
  (matched by the pellet's frames in the ground truth's own cuts, as every
  review answer is), so a bundle whose every flagged pellet is so answered
  is released by the ordinary rule;
- when a re-cut set the current cuts AFTER the ground truth was made, an
  answer stands only where the pellet's frames are unchanged; otherwise it is
  asked again (the re-cut rule, answer_survives_recut); ground truth made
  after the re-cut stands;
- the review window shows a bundle with ground truth only while a flagged
  pellet is still open, and walks only those pellets.

Built under tmp_path with invented ids.
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.review import causal_review_io as cio
from mousereach.review.triage_status import triage_status

VID = "20990101_ABC9901_P2"
CUTS = [100, 400, 700, 1000]          # pellets 1-3
# The ground-truth tool's own cuts: a person places them a few frames off the
# segmenter's -- still the same pellets.
GT_CUTS = [102, 398, 703, 1000]


@pytest.fixture(autouse=True)
def _no_corpus_index(monkeypatch, tmp_path):
    # The corpus index must not see these files: only the bundle holds them.
    monkeypatch.setattr(cio, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "records")


def _bundle(tmp_path, cuts=CUTS, flagged=(2,), gt_outcomes=None, recut=None,
            gt_when="2026-01-01T00:00:00"):
    b = tmp_path / "triage" / VID
    b.mkdir(parents=True)
    seg = {"overall_confidence": 0.9,
           "boundaries": [{"index": i, "frame": f} for i, f in enumerate(cuts)]}
    if recut:
        # A person re-cut the video at ``recut`` (fix-segmentation save).
        seg.update({"boundary_source": "human", "corrected_at": recut,
                    "algo_boundaries": list(CUTS)})
    (b / f"{VID}_segments.json").write_text(json.dumps(seg), encoding="utf-8")
    (b / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": n, "outcome": "triaged" if n in flagged else "untouched"}
        for n in (1, 2, 3)]}), encoding="utf-8")
    outs = gt_outcomes if gt_outcomes is not None else [
        {"segment_num": n, "outcome": "retrieved", "determined": True,
         "determined_at": gt_when} for n in flagged]
    gt = {"last_modified_at": gt_when,
          "segmentation": {"boundaries": [
              {"index": i, "frame": f, "determined": True} for i, f in enumerate(GT_CUTS)]},
          "reaches": {"exhaustive": False, "reaches": []},
          "outcomes": {"exhaustive": False, "segments": outs}}
    (b / f"{VID}_unified_ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")
    return b


def test_ground_truth_inside_the_bundle_releases_it(tmp_path):
    b = _bundle(tmp_path)
    st = triage_status(b, VID)
    assert st.triaged == {2}
    assert st.unresolved == set() and st.fully_resolved and st.clean


def test_only_the_pellets_ground_truth_determined_count(tmp_path):
    b = _bundle(tmp_path, flagged=(1, 3), gt_outcomes=[
        {"segment_num": 1, "outcome": "untouched", "determined": True},
        {"segment_num": 3, "outcome": "retrieved", "determined": False}])
    st = triage_status(b, VID)
    assert st.unresolved == {3}


def test_ground_truth_made_before_a_recut_that_moved_the_pellet_is_asked_again(tmp_path):
    # A person re-cut the video after the ground truth was made and moved
    # pellet 2's frames: its ground-truth answer no longer describes them.
    b = _bundle(tmp_path, cuts=[100, 450, 700, 1000], recut="2026-06-01T00:00:00")
    assert triage_status(b, VID).unresolved == {2}


def test_ground_truth_made_after_the_recut_stands(tmp_path):
    b = _bundle(tmp_path, cuts=[100, 450, 700, 1000], recut="2026-06-01T00:00:00",
                gt_when="2026-07-01T00:00:00")
    assert triage_status(b, VID).unresolved == set()


def test_window_shows_a_ground_truthed_bundle_only_while_a_pellet_is_open(tmp_path):
    from mousereach.review.causal_review_widget import CausalReviewWidget as W
    stub = SimpleNamespace(_deep_review=False, _bundle_reviewed=lambda b: False)
    answered = _bundle(tmp_path / "a")
    assert not W._bundle_needs_review(stub, answered, tmp_path)
    still_open = _bundle(tmp_path / "b", flagged=(1, 2), gt_outcomes=[
        {"segment_num": 1, "outcome": "untouched", "determined": True}])
    assert W._bundle_needs_review(stub, still_open, tmp_path)


def test_window_does_not_walk_a_pellet_ground_truth_answered(tmp_path):
    from mousereach.review.causal_review_widget import ground_truth_answered
    b = _bundle(tmp_path, flagged=(1, 2), gt_outcomes=[
        {"segment_num": 1, "outcome": "untouched", "determined": True}])
    cur = cio.segments_from_segmentation(json.loads(
        (b / f"{VID}_segments.json").read_text(encoding="utf-8")))
    assert ground_truth_answered(VID, [b], cur, False) == {1}
