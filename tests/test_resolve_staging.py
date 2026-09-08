"""The return path's pose resolver consults DLC_Complete staging -- last, and
declared-scorer only.

WHY: a GPU node's freshly staged pose lives in DLC_Complete before intake
pulls it. A human-cleared video whose only pose was the staged copy refused
to return in a loop every ~2-3 minutes (2026-09-08); and 8 videos whose
Analyzed trees hold only OLD-model poses were one bundle-omission away from
silently re-running on the wrong pose. Staging closes both -- but an
old-model pose in staging is stale by definition and must be refused.
"""
import mousereach.watcher.review_return as rr
import mousereach.pipeline.versions as vmod


def _wire(monkeypatch, tmp_path, staging):
    monkeypatch.setattr(rr.Paths, "ANALYZED_OUTPUT", None)
    monkeypatch.setattr(rr.Paths, "DLC_STAGING", staging)
    monkeypatch.setattr(rr.Paths, "NAS_ROOT", tmp_path)
    monkeypatch.setattr(vmod, "get_current_versions",
                        lambda root: {"versions": {"dlc_scorer": "DLC_newmodel"}})


def test_staged_declared_pose_is_found(tmp_path, monkeypatch):
    staging = tmp_path / "DLC_Complete"
    staging.mkdir()
    bundle = tmp_path / "b"
    bundle.mkdir()
    stem = "20240101_ABC0101_P1"
    h5 = staging / (stem + "DLC_newmodel.h5")
    h5.write_text("x", encoding="utf-8")
    (staging / (stem + ".mp4")).write_text("v", encoding="utf-8")
    _wire(monkeypatch, tmp_path, staging)
    mp4, pose = rr._resolve_inputs(bundle, stem)
    assert pose == h5
    assert mp4 is not None


def test_old_model_staged_pose_is_refused(tmp_path, monkeypatch):
    staging = tmp_path / "DLC_Complete"
    staging.mkdir()
    bundle = tmp_path / "b"
    bundle.mkdir()
    stem = "20240101_ABC0101_P1"
    (staging / (stem + "DLC_oldmodel.h5")).write_text("x", encoding="utf-8")
    _wire(monkeypatch, tmp_path, staging)
    mp4, pose = rr._resolve_inputs(bundle, stem)
    assert pose is None        # refusing beats silently posing with old model
