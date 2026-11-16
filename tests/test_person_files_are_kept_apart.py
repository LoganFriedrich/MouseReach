"""Clean-up moves keep a person's files apart from machine output.

A bundle holds the video and pose, what the pipeline made, and what a person
did (a review, ground truth, a clear marker, cuts set by hand). Two clean-up
paths treated everything alike: a run moved into an existing bundle replaced
same-named files -- a person's hand-set cuts included -- and a stale bundle was
filed away to _Problematic, where no reader looks, whatever a person had
written in it. Built under tmp_path with invented ids.
"""
import json

import pytest

import mousereach.watcher.review_return as rr
from mousereach.watcher import review_routing as routing

VID = "20990101_ABC9901_P1"


def test_which_files_are_a_persons(tmp_path):
    human = tmp_path / f"{VID}_segments.json"
    human.write_text(json.dumps({"boundary_source": "human"}), encoding="utf-8")
    algo = tmp_path / "algo" / f"{VID}_segments.json"
    algo.parent.mkdir()
    algo.write_text(json.dumps({"boundary_source": "algo"}), encoding="utf-8")
    assert routing.is_person_file(human, VID)
    assert not routing.is_person_file(algo, VID)
    for name in (f"{VID}_causal_review.json", f"{VID}_unified_ground_truth.json",
                 f"{VID}_deep_review_cleared.json", f"{VID}_seg_validation.json",
                 f"{VID}_deep_review_cleared_superseded_20990101T000000.json"):
        assert routing.is_person_file(tmp_path / name, VID), name
    for name in (f"{VID}_reaches.json", f"{VID}_pellet_outcomes.json",
                 f"{VID}_triage.json", f"{VID}_routing.json", f"{VID}_manifest.json",
                 f"{VID}.mp4", f"{VID}DLC_model_x.h5"):
        assert not routing.is_person_file(tmp_path / name, VID), name


def test_a_run_never_replaces_cuts_a_person_set(tmp_path):
    queue = tmp_path / "deep_review"
    bundle = queue / VID
    bundle.mkdir(parents=True)
    hand = json.dumps({"boundary_source": "human", "boundaries": [1, 2, 3]})
    (bundle / f"{VID}_segments.json").write_text(hand, encoding="utf-8")
    src = tmp_path / "Processing"
    src.mkdir()
    (src / f"{VID}_segments.json").write_text(
        json.dumps({"boundary_source": "algo", "boundaries": [9]}), encoding="utf-8")
    (src / f"{VID}_reaches.json").write_text("{}", encoding="utf-8")

    _, moved = routing.move_video_bundle(VID, src, queue, "why")

    assert (bundle / f"{VID}_segments.json").read_text(encoding="utf-8") == hand
    assert (src / f"{VID}_segments.json").is_file()          # left at the source
    assert f"{VID}_reaches.json" in moved
    record = json.loads((bundle / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert [f["name"] for f in record["failed_files"]] == [f"{VID}_segments.json"]


@pytest.fixture
def review_root(tmp_path, monkeypatch):
    root = tmp_path / "Review"
    (root / "triage").mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", root)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", root / "triage")
    return root


def test_a_stale_bundle_holding_ground_truth_is_kept_for_a_person(review_root):
    """Not filed away among the retired bundles, where nobody looks: kept
    whole in _kept_for_a_person with a note (and listed for a person)."""
    bundle = review_root / "triage" / VID
    bundle.mkdir()
    (bundle / f"{VID}_segments.json").write_text("{}", encoding="utf-8")
    gt = bundle / f"{VID}_unified_ground_truth.json"
    gt.write_text('{"by": "person-a"}', encoding="utf-8")

    assert rr._retire_stale_bundle(bundle, VID, "archived")

    kept = review_root / "_Problematic" / "_kept_for_a_person" / VID
    assert (kept / gt.name).read_text(encoding="utf-8") == '{"by": "person-a"}'
    assert (kept / "SET_ASIDE.json").is_file()
    assert not list((review_root / "_Problematic").glob("_stale_seg_failed_bundles_*"))


def test_a_stale_machine_only_bundle_is_still_filed_away(review_root):
    bundle = review_root / "triage" / VID
    bundle.mkdir()
    (bundle / f"{VID}_segments.json").write_text("{}", encoding="utf-8")
    assert rr._retire_stale_bundle(bundle, VID, "archived")
    assert not bundle.exists()
