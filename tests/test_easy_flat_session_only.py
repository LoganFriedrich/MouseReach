"""Easy and Flat trays are session-only: not segmented, no outcomes, never held.

WHY (Logan, 2026-10-23): "Easy and flat should never get triaged"; "Pellet
outcomes don't matter for easy and flats"; "They also don't get segmented"
(pellets are presented, "we just don't care about it"); "Easy and flat don't
need kinematics". What they need is the session's reaches. Before this, the
segmenter cut them into pellet windows, and a failed cut or a quality flag
held them for a person. Built under tmp_path.
"""
import json

import numpy as np
import pandas as pd
import pytest

from mousereach.config import is_outcome_free_tray
from mousereach.pipeline import triage
from mousereach.segmentation.core import batch as seg_batch
from mousereach.watcher.review_gate import (
    DECISION_CLEAN, DECISION_STALE, evaluate_gate,
)

EF = "20990101_CNT9901_F1"
EASY = "20990101_CNT9902_E3"
PILLAR = "20990101_CNT9903_P1"
SCORER = "DLC_resnet101_MPSAOct27shuffle3_100000"


def test_which_videos_are_easy_or_flat():
    assert is_outcome_free_tray(EF) and is_outcome_free_tray(EASY)
    assert not is_outcome_free_tray(PILLAR)
    assert not is_outcome_free_tray("20990101_CNT9901")       # no tray in the name


# ------------------------------------------------------------------ the gate

def _seg(d, vid, failed=False, **extra):
    doc = {"overall_confidence": 0.0 if failed else 0.9, "boundaries": [0]}
    doc.update(extra)
    (d / f"{vid}_segments.json").write_text(json.dumps(doc), encoding="utf-8")


@pytest.mark.parametrize("vid", [EF, EASY])
def test_the_gate_never_holds_easy_or_flat(tmp_path, vid):
    _seg(tmp_path, vid, failed=True)
    decision, reason, _ = evaluate_gate(vid, tmp_path, qc_verdict="needs_review")
    assert decision == DECISION_CLEAN
    assert "never held" in reason


def test_a_mid_flight_easy_or_flat_still_runs_again(tmp_path):
    # The stale-output check sends a video round again, not to a person.
    _seg(tmp_path, EF, corrected_at="2026-10-23T10:00:00")
    (tmp_path / f"{EF}_reaches.json").write_text(
        json.dumps({"detected_at": "2026-10-23T09:00:00"}), encoding="utf-8")
    decision, _, _ = evaluate_gate(EF, tmp_path)
    assert decision == DECISION_STALE


# ---------------------------------------------------------- the segmentation

def _pose(d, vid, n=300):
    cols = pd.MultiIndex.from_tuples(
        [("scorer", bp, c) for bp in ("SABL", "SABR") for c in ("x", "y", "likelihood")])
    df = pd.DataFrame(np.ones((n, len(cols))), columns=cols)
    p = d / f"{vid}{SCORER}.h5"
    df.to_hdf(p, key="df")
    return p


@pytest.mark.parametrize("human", [False, True])
def test_easy_or_flat_is_one_whole_video_span(tmp_path, human):
    pose = _pose(tmp_path, EF, n=300)
    if human:   # a hand-set cut is no more used than a machine one
        (tmp_path / f"{EF}_segments.json").write_text(
            json.dumps({"boundary_source": "human", "boundaries": [0, 100, 200]}),
            encoding="utf-8")
    result = seg_batch.process_single(pose)
    assert result["success"] and result["whole_video_span"]
    doc = json.loads((tmp_path / f"{EF}_segments.json").read_text(encoding="utf-8"))
    assert doc["boundaries"] == [0]
    assert doc["total_frames"] == 300
    assert doc["segmenter_algorithm"] == seg_batch.WHOLE_VIDEO_ALGORITHM
    assert doc["segmenter_version"] == seg_batch.SEGMENTER_VERSION


def test_the_quality_check_does_not_count_easy_or_flat_segments(tmp_path):
    _seg(tmp_path, EF)
    (tmp_path / f"{EF}_reaches.json").write_text("{}", encoding="utf-8")
    flags = triage.check_structural(EF, tmp_path)
    assert not [f for f in flags if "boundary count" in f.description]


# ---------------------------------------------------- the session file

def test_the_session_file_has_reaches_and_attention_and_no_kinematics(tmp_path):
    from mousereach.kinematics.core.feature_extractor import FeatureExtractor
    n = 200
    cols = pd.MultiIndex.from_tuples(
        [("scorer", bp, c) for bp in ("BOXR", "Nose") for c in ("x", "y", "likelihood")])
    data = np.zeros((n, 6))
    data[:, 1] = 300.0                       # BOXR_y
    data[:, 2] = 0.99
    data[:, 4] = np.where(np.arange(n) < 50, 100.0, 290.0)   # Nose_y: at the tray 150/200
    data[:, 5] = 0.99
    pose = tmp_path / f"{EF}{SCORER}.h5"
    pd.DataFrame(data, columns=cols).to_hdf(pose, key="df")
    reaches = {"video_name": EF, "total_frames": n, "n_segments": 1, "segments": [{
        "segment_num": 1, "start_frame": 0, "end_frame": n - 1, "ruler_pixels": 32.0,
        "n_reaches": 2, "reaches": [
            {"reach_id": 1, "reach_num": 1, "start_frame": 60, "apex_frame": 64,
             "end_frame": 70, "duration_frames": 11, "max_extent_pixels": 12.0},
            {"reach_id": 2, "reach_num": 2, "start_frame": 120, "apex_frame": 125,
             "end_frame": 131, "duration_frames": 12, "max_extent_pixels": 9.0}]}]}
    rp = tmp_path / f"{EF}_reaches.json"
    rp.write_text(json.dumps(reaches), encoding="utf-8")

    d = FeatureExtractor().extract_session_only(pose, rp).to_dict()

    seg = d["segments"][0]
    assert d["n_segments"] == 1 and seg["n_reaches"] == 2
    assert seg["attention_score"] == pytest.approx(75.0, abs=1.0)
    assert seg["outcome"] is None
    r = seg["reaches"][0]
    assert (r["reach_id"], r["start_frame"], r["end_frame"]) == (1, 60, 70)
    assert r["max_extent_pixels"] is None and r["velocity_at_apex_mm_per_sec"] is None
    assert r["outcome"] is None and r["causal_reach"] is False


# ------------------------------------------------------- filed videos re-run

class StubDB:
    def __init__(self, archived):
        self.archived = list(archived)
        self.forced = []

    def _get_connection(self):
        rows = self.archived

        class C:
            def execute(self, *a):
                return [(r,) for r in rows]

            def close(self):
                pass
        return C()

    def get_videos_in_state(self, state):
        if state == "archived":
            return [{"video_id": v, "state": "archived"} for v in self.archived]
        return []

    def force_state(self, video_id, state, **k):
        self.forced.append((video_id, state, k.get("reprocess_scope")))

    def update_state(self, *a, **k):
        pass


def test_a_filed_easy_or_flat_without_its_session_file_is_rerun_once(tmp_path, monkeypatch):
    import mousereach.pipeline.versions as versions_mod
    import mousereach.review.causal_review_io as crio
    from mousereach.watcher.reprocessor import ReprocessingScanner
    current = {"versions": {"dlc_scorer": SCORER, "segmenter": "9.9.9"}}   # Pillar is current
    coh = tmp_path / "Analyzed" / "Connectome" / "CNT99"
    coh.mkdir(parents=True)
    for vid in (EF, EASY, PILLAR):
        (coh / f"{vid}_processing_manifest.json").write_text(json.dumps(
            {"dlc_model": {"dlc_scorer": "an older model"},                    # stale for all
             "pipeline_versions": {"segmenter": "1.0.0"}}), encoding="utf-8")
        (coh / f"{vid}.mp4").write_text("x", encoding="utf-8")
    (coh / f"{EASY}_features.json").write_text("{}", encoding="utf-8")       # has its file
    monkeypatch.setattr(versions_mod, "get_current_versions", lambda root=None: current)
    monkeypatch.setattr(versions_mod, "declaration_drift", lambda cur: [])
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {})
    db = StubDB(archived=(EF, EASY))

    ReprocessingScanner(db, tmp_path).scan(mark_outdated=True, adopt_orphans=False)

    # Only the Easy/Flat video without a session file; never a re-pose for
    # version staleness.
    assert db.forced == [(EF, "outdated", "segmentation")]
