"""The kinematics extractor knows when a person set the segmentation.

WHY (2026-10-23): the check read ``reach_path``, a name that does not exist in
FeatureExtractor.extract (the parameter is ``reaches_path``). The NameError was
swallowed by the surrounding try, so ``human_segmentation`` was always False:
since f3cae91 (2026-10-21) a person's short segments were meant to count when
their review answers are applied, and in the kinematics they never did.
"""
import json

import numpy as np
import pandas as pd
import pytest

import mousereach.review.truth_resolver as tr
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

VID = "20990101_CNT9901_P1"


class Stop(BaseException):
    pass


def test_a_person_s_segmentation_reaches_the_truth_layering(tmp_path, monkeypatch):
    cols = pd.MultiIndex.from_tuples([("s", "Nose", c) for c in ("x", "y", "likelihood")])
    pose = tmp_path / f"{VID}DLC_x.h5"
    pd.DataFrame(np.ones((10, 3)), columns=cols).to_hdf(pose, key="df")
    reaches = tmp_path / f"{VID}_reaches.json"
    reaches.write_text(json.dumps({"video_name": VID, "total_frames": 10,
                                   "n_segments": 0, "segments": []}), encoding="utf-8")
    outcomes = tmp_path / f"{VID}_pellet_outcomes.json"
    outcomes.write_text(json.dumps({"segments": []}), encoding="utf-8")
    (tmp_path / f"{VID}_segments.json").write_text(
        json.dumps({"boundary_source": "human", "boundaries": [0]}), encoding="utf-8")
    seen = {}

    def record(*a, **k):
        seen.update(k)
        raise Stop
    monkeypatch.setattr(tr, "resolve_truth_layers", record)

    with pytest.raises(Stop):
        FeatureExtractor().extract(pose, reaches, outcomes)
    assert seen["human_segmentation"] is True
