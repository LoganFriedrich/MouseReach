"""The return path's pose resolver consults pose staging (Processing/Posed) --
last, and declared-scorer only.

WHY: a GPU node's freshly staged pose lives in staging before intake pulls it.
A human-cleared video whose only pose was the staged copy refused to return in
a loop every ~2-3 minutes (2026-09-08); and 8 videos whose Analyzed trees hold
only OLD-model poses were one bundle-omission away from silently re-running on
the wrong pose. Staging closes both -- but an old-model pose in staging is
stale by definition and must be refused.
"""
import json

import mousereach.watcher.review_return as rr
import mousereach.pipeline.versions as vmod


def _wire(monkeypatch, tmp_path, staging):
    monkeypatch.setattr(rr.Paths, "ANALYZED_OUTPUT", None)
    monkeypatch.setattr(rr.Paths, "DLC_STAGING", staging)
    monkeypatch.setattr(rr.Paths, "NAS_ROOT", tmp_path)
    monkeypatch.setattr(vmod, "get_current_versions",
                        lambda root: {"versions": {"dlc_scorer": "DLC_newmodel"}})


def test_staged_declared_pose_is_found(tmp_path, monkeypatch):
    staging = tmp_path / "Posed"
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
    staging = tmp_path / "Posed"
    staging.mkdir()
    bundle = tmp_path / "b"
    bundle.mkdir()
    stem = "20240101_ABC0101_P1"
    (staging / (stem + "DLC_oldmodel.h5")).write_text("x", encoding="utf-8")
    _wire(monkeypatch, tmp_path, staging)
    mp4, pose = rr._resolve_inputs(bundle, stem)
    assert pose is None        # refusing beats silently posing with old model


def test_manifest_pose_path_through_a_guard_file_reads_as_not_found(tmp_path, monkeypatch):
    # A triage manifest written before the layout change can name the OLD
    # staging folder, which is now a plain guard FILE; the pose it named was set
    # aside into a leftovers folder that no resolver reads. Following that
    # pointer must read as "not found" (never raise), and the resolver must then
    # find the pose in the NEW staging folder when it is there.
    nas = tmp_path / "nas"
    posed = nas / "Processing" / "Posed"
    posed.mkdir(parents=True)
    guard = nas / "Processing" / "DLC_Complete"
    guard.write_text("retired folder", encoding="ascii")
    stem = "20240101_ABC0101_P1"
    name = stem + "DLC_newmodel.h5"
    leftovers = nas / "Processing" / "_leftovers_pending_cleanup_2026-09-14" / "DLC_Complete"
    leftovers.mkdir(parents=True)
    (leftovers / name).write_text("x", encoding="utf-8")
    bundle = nas / "Processing" / "Review" / "triage" / stem
    bundle.mkdir(parents=True)
    (bundle / f"{stem}_manifest.json").write_text(json.dumps({
        "video_stem": stem,
        "canonical_video_path": str(guard / f"{stem}.mp4"),
        "canonical_dlc_h5_path": str(guard / name),
    }), encoding="utf-8")
    _wire(monkeypatch, nas, posed)

    mp4, pose = rr._resolve_inputs(bundle, stem)
    # Leftovers are not a stage: such a bundle stays unreturnable until the
    # manifest is rewritten or the pose is back in a folder the resolver reads.
    assert (mp4, pose) == (None, None)

    staged = posed / name
    staged.write_text("x", encoding="utf-8")
    assert rr._resolve_inputs(bundle, stem)[1] == staged
