"""The kinematics step applies the ground truth beside the run, and the
processing manifest records the ground truth it really applied.

WHY: the ground-truth index (causal_review_io.gt_index) is built once per
process. In a long-running watcher it predates a ground-truth file copied down
for a filed video's re-run -- the very file the re-run exists to apply -- so the
kinematics were made without it, while the manifest was stamped with the file
beside the run as if it had been applied. The video then counted as brought up
to date for good and the person's answers never reached its features.

The kinematics-only start of such a re-run is enough for ground truth: the
feature extractor layers ground-truth REACH BOUNDARIES over the algorithm's
reaches itself (truth_resolver._apply_gt_reaches, reach_source
'ground_truth'), so re-running reach detection would add nothing -- it does not
read ground truth. Shown below on a reach whose frames the ground truth moves.

Built under tmp_path with invented ids; no real folder. Names that only this
change adds are imported inside the test that needs them, so the behaviour
tests fail on code without the change by what they assert, not on import.
"""
import pickle
import json

import numpy as np
import pandas as pd
import pytest

import mousereach.review.causal_review_io as crio
from mousereach.kinematics.core.feature_extractor import FeatureExtractor
from mousereach.pipeline import manifest as man

VID = "20990101_ABC0901_P1"
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A processing folder with one segment and one algorithm reach (frames
    20-30), and a corpus index built BEFORE any ground truth existed."""
    monkeypatch.setattr(crio, "_GT_INDEX_CACHE", {})        # stale: built earlier
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [], raising=False)
    n = 100
    cols, data = [], []
    for p in POINTS:
        for c, v in (("x", 100.0), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(n, v) + (np.arange(n) * 0.5 if c == "y" and p.startswith("R") else 0))
    pose = tmp_path / f"{VID}DLC_x.h5"
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(pose, key="df")
    # The video's frame rate, recorded beside the pose as DeepLabCut does
    # (kinematics/core/frame_rate.py); without one no kinematics are written.
    with open(pose.with_name(pose.stem + "_meta.pickle"), "wb") as _f:
        pickle.dump({"data": {"fps": 60.0}}, _f)
    reaches = {"video_name": VID, "total_frames": n, "n_segments": 1,
               "segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 99, "ruler_pixels": 34.0,
                             "n_reaches": 1,
                             "reaches": [{"reach_id": 1, "reach_num": 1, "start_frame": 20,
                                          "end_frame": 30, "apex_frame": 25,
                                          "duration_frames": 10}]}]}
    outcomes = {"video_name": VID, "segments": [{"segment_num": 1, "outcome": "untouched",
                                                 "start_frame": 0, "end_frame": 99}]}
    (tmp_path / f"{VID}_reaches.json").write_text(json.dumps(reaches), encoding="utf-8")
    (tmp_path / f"{VID}_pellet_outcomes.json").write_text(json.dumps(outcomes), encoding="utf-8")
    return tmp_path, pose


def _gt(folder):
    p = folder / f"{VID}_unified_ground_truth.json"
    p.write_text(json.dumps({
        "video_name": VID,
        "reaches": {"exhaustive": False, "reaches": [
            {"reach_id": 1, "segment_num": 1, "start_frame": 22, "end_frame": 40,
             "start_determined": True, "end_determined": True,
             "apex_frame": 31}]},
    }), encoding="utf-8")
    return p


def _extract(folder, pose):
    fx = FeatureExtractor()
    feats = fx.extract(pose, folder / f"{VID}_reaches.json",
                       folder / f"{VID}_pellet_outcomes.json")
    return fx, feats.to_dict()


def _reach_frames(feats):
    out = []
    for seg in feats.get("segments", []):
        for r in seg.get("reaches", []):
            out.append((r.get("start_frame"), r.get("end_frame"), r.get("reach_source")))
    return out


def test_ground_truth_copied_beside_the_run_after_the_index_was_built_is_applied(run_dir):
    folder, pose = run_dir
    gt = _gt(folder)
    fx, feats = _extract(folder, pose)
    frames = _reach_frames(feats)
    assert (22, 40) in [(s, e) for s, e, _ in frames]       # the ground truth's boundaries
    assert (20, 30) not in [(s, e) for s, e, _ in frames]   # not the algorithm's
    assert fx.applied_ground_truth == gt                    # and that is what is recorded


def test_with_no_ground_truth_anywhere_the_extractor_says_none(run_dir):
    folder, pose = run_dir
    fx, feats = _extract(folder, pose)
    assert fx.applied_ground_truth is None
    assert [(s, e) for s, e, _ in _reach_frames(feats)] == [(20, 30)]


# --- what the manifest records ------------------------------------------------

def _manifest(folder):
    p = folder / f"{VID}_processing_manifest.json"
    p.write_text(json.dumps({"video_id": VID, "created_at": "2099-01-01T10:00:00",
                             "pipeline_versions": {}}), encoding="utf-8")
    return p


def test_the_stamp_is_the_file_applied_or_null_or_nothing(tmp_path, monkeypatch):
    # The shared version index is never written by a test.
    import mousereach.pipeline.version_index as vi
    from types import SimpleNamespace
    from mousereach.pipeline.manifest import NOT_RESOLVED
    monkeypatch.setattr(vi, "VersionIndex", lambda *a, **k: SimpleNamespace(
        upsert_from_manifest=lambda *a, **k: None))
    p = _manifest(tmp_path)
    gt = _gt(tmp_path)
    man.record_kinematic_version(VID, tmp_path, "9.9", ground_truth_path=gt)
    assert json.loads(p.read_text())["applied_ground_truth"]["file"] == gt.name
    man.record_kinematic_version(VID, tmp_path, "9.9", ground_truth_path=None)
    assert json.loads(p.read_text())["applied_ground_truth"] is None
    man.record_kinematic_version(VID, tmp_path, "9.9", ground_truth_path=NOT_RESOLVED)
    assert "applied_ground_truth" not in json.loads(p.read_text())


def test_kinematics_made_with_no_ground_truth_owe_a_file_filed_later(tmp_path):
    """A null stamp is a fact: none was applied. A ground-truth file filed
    beside the video now is owed a re-run, whatever the clocks say."""
    gt = _gt(tmp_path)
    m = {"created_at": "2999-01-01T00:00:00", "applied_ground_truth": None}
    assert man.ground_truth_applied(m, gt) is False


def test_a_stamp_matching_the_filed_file_is_applied(tmp_path):
    gt = _gt(tmp_path)
    from mousereach.pipeline.fsutil import sha256_file
    m = {"applied_ground_truth": {"file": gt.name, "sha256": sha256_file(gt)}}
    assert man.ground_truth_applied(m, gt) is True
