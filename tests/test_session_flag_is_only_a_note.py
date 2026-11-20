"""A session flag is a note and nothing else (lab decision 2026-11-19, D5-10).

"Flags are there so we could in theory find them." Flag Session in the review
window records who flagged the mouse+day session, when and why. It moves no
video, holds none and leaves none out. The record is carried into every
features file of the session (``session_flag``), where a database integrator
finds it as a marker; a filed video of a session flagged after its kinematics
were made is owed a kinematics-only re-run so the marker reaches it.

Built under tmp_path with invented ids; no real folder or database.
"""
import json
import os
from types import SimpleNamespace

import pytest

import mousereach.review.causal_review_io as crio
import mousereach.review.causal_review_widget as crw
import mousereach.watcher.review_gate as rg

VID = "20990101_ABC9901_P2"
KEY = "20990101_ABC9901"


def _window(bundle, root, said, advanced):
    return SimpleNamespace(
        _video_stem=VID, _bundle_dir=bundle,
        _review_root=lambda: root,
        _save_review=lambda: None,
        _notes_edit=SimpleNamespace(toPlainText=lambda: "cage door open all day"),
        _status_label=SimpleNamespace(setText=lambda text: said.append(text)),
        _load_next_video=lambda: advanced.append(True),
        _watcher_db_for_move=lambda: (None, ""),
    )


def test_flag_session_records_who_when_why_and_moves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(rg.Paths, "REVIEW_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(rg.Paths, "DEEP_REVIEW", tmp_path / "deep", raising=False)
    root = tmp_path
    bundle = tmp_path / "triage" / VID
    bundle.mkdir(parents=True)
    (bundle / f"{VID}.mp4").write_bytes(b"x")
    before = sorted(p.name for p in bundle.iterdir())
    moved, said, advanced, infos = [], [], [], []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: moved.append(a))
    monkeypatch.setattr(crw, "show_info", lambda msg, *a, **k: infos.append(msg))
    monkeypatch.setattr(crw, "show_error", lambda msg, *a, **k: infos.append("ERR " + msg))

    crw.CausalReviewWidget._flag_session(_window(bundle, root, said, advanced))

    assert moved == [] and advanced == []                 # nothing moved, stays on the video
    assert sorted(p.name for p in bundle.iterdir()) == before
    assert not (tmp_path / "deep").exists()
    rec = crio.load_flagged_sessions(root, use_cache=False)[KEY]
    assert rec["reason"] == "cage door open all day"
    assert rec["flagged_by"] and rec["flagged_at"]
    assert "requires_human_review" not in rec             # it requires nothing
    assert any("nothing was moved" in m for m in infos)


def test_the_flag_is_stamped_into_the_features(tmp_path, monkeypatch):
    from mousereach.watcher import orchestrator as o
    monkeypatch.setattr(o.Paths, "REVIEW_ROOT", tmp_path, raising=False)
    assert o.with_session_flag({"a": 1}, VID) == {"a": 1, "session_flag": None}
    crio.flag_session(VID, tmp_path, reason="why", flagged_by="someone")
    out = o.with_session_flag({"a": 1}, "20990101_ABC9901_P4")   # same mouse+day
    assert out["a"] == 1
    assert out["session_flag"]["flagged_by"] == "someone"
    assert out["session_flag"]["reason"] == "why"
    assert o.with_session_flag({}, "20990102_ABC9901_P1")["session_flag"] is None


def test_the_kinematics_step_writes_the_flag_into_the_features_file(tmp_path, monkeypatch):
    from mousereach.watcher import orchestrator as o
    import mousereach.kinematics.core.feature_extractor as fe
    import mousereach.pipeline.manifest as man
    monkeypatch.setattr(o.Paths, "REVIEW_ROOT", tmp_path / "Review", raising=False)
    crio.flag_session(VID, tmp_path / "Review", reason="why", flagged_by="someone")
    work = tmp_path / "work"
    work.mkdir()
    for n in ("processing_manifest", "reaches", "pellet_outcomes"):
        (work / f"{VID}_{n}.json").write_text("{}")

    class F:
        n_segments = 0

        def to_dict(self):
            return {"made": True}

    class X:
        VERSION = "1"

        def extract(self, *a, **k):
            return F()
    monkeypatch.setattr(fe, "FeatureExtractor", X)
    monkeypatch.setattr(man, "record_kinematic_version", lambda *a, **k: None)
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.db = SimpleNamespace(log_step=lambda *a, **k: None)
    assert orch._make_features(VID, work, None, skip_outcomes=False) is None
    feats = json.loads((work / f"{VID}_features.json").read_text())
    assert feats["made"] is True and feats["session_flag"]["reason"] == "why"


# --- the version scan: a flag newer than the filed features --------------------

class DB:
    def __init__(self, archived):
        self.rows = {s: {"video_id": s, "state": "archived"} for s in archived}

    def get_videos_in_state(self, state):
        return [dict(r) for r in self.rows.values() if r["state"] == state]

    def _get_connection(self):
        raise RuntimeError("no db in this test")

    def force_state(self, vid, state, reason=None, **kw):
        self.rows[vid].update(kw, state=state)

    def update_state(self, vid, state, **kw):
        self.force_state(vid, state, **kw)

    def set_fields(self, vid, **kw):
        self.rows[vid].update(kw)


@pytest.fixture
def filed(tmp_path, monkeypatch):
    import mousereach.config as config
    import mousereach.pipeline.version_index as vi
    from mousereach.watcher.reprocessor import ReprocessingScanner
    monkeypatch.setattr(config, "filing_rules", lambda: {
        "ABC": {"folder": "Project_A", "cohort_folder": "{CODE}{NN}"}})
    nas = tmp_path / "nas"
    cohort = nas / "Analyzed" / "Project_A" / "ABC99"
    cohort.mkdir(parents=True)
    versions = {"segmenter": "1", "reach_detector": "1", "outcome_detector": "1",
                "assignment": "1", "kinematic_extractor": "1"}
    (nas / "pipeline_versions.json").write_text(json.dumps(
        {"versions": dict(versions, dlc_scorer="S")}), encoding="utf-8")
    (cohort / f"{VID}_processing_manifest.json").write_text(json.dumps(
        {"video_stem": VID, "dlc_model": {"dlc_scorer": "S"},
         "pipeline_versions": versions}), encoding="utf-8")
    feats = cohort / f"{VID}_features.json"
    feats.write_text("{}", encoding="utf-8")
    os.utime(feats, (1_900_000_000, 1_900_000_000))     # 2030
    monkeypatch.setattr(vi, "VersionIndex", lambda *a, **k: SimpleNamespace(
        upsert_from_manifest=lambda *a, **k: None))
    monkeypatch.setattr(crio, "review_path_index", lambda stems: {})
    monkeypatch.setattr(crio, "resolve_review_path", lambda *a, **k: None)

    def scan():
        db = DB([VID])
        s = ReprocessingScanner(db, nas, cache_path=tmp_path / "c.json").scan(
            mark_outdated=True, adopt_orphans=False)
        return db, s
    return nas, scan


def _flag_at(nas, when):
    root = nas / "Processing" / "Review"
    root.mkdir(parents=True, exist_ok=True)
    (root / "flagged_sessions.json").write_text(json.dumps({"sessions": {KEY: {
        "session_key": KEY, "reason": "r", "flagged_by": "p", "flagged_at": when}}}))


def test_a_flag_after_the_filed_kinematics_owes_a_kinematics_only_rerun(filed):
    nas, scan = filed
    _flag_at(nas, "2031-01-01T10:00:00")
    db, s = scan()
    assert db.rows[VID]["state"] == "outdated"
    assert db.rows[VID]["reprocess_scope"] == "kinematics"
    assert s["outdated_videos"][0]["stale_components"] == ["session_flag"]


def test_a_flag_the_filed_kinematics_already_carry_owes_nothing(filed):
    nas, scan = filed
    _flag_at(nas, "2029-01-01T10:00:00")
    db, _ = scan()
    assert db.rows[VID]["state"] == "archived"
