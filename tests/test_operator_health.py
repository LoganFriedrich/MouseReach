"""Operator recovery surface: the health line and honest bundle manifests.

WHY: recovering a stuck video used to require a terminal; the dashboard now
says in plain words whether the processor runs and what waits for a person.
And a routed bundle's manifest claimed self_contained with a null pose
pointer, which put one video into an every-2-minutes return-refusal loop
(2026-09-08) -- the manifest now records where the pose actually lives.
"""
import json
from pathlib import Path

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.watcher.health import health_report


class StubDB:
    def __init__(self, failed=(), unres=()):
        self._f = [{"video_id": v} for v in failed]
        self._u = [{"video_id": v} for v in unres]
    def get_videos_in_state(self, state):
        return {"failed": self._f, "unresolvable": self._u}.get(state, [])


def test_health_names_stuck_videos():
    lines = health_report(StubDB(failed=("20240101_ABC0101_P1",
                                         "20240101_ABC0102_P1"),
                                 unres=("20240101_ABC0103_P1",)))
    text = "\n".join(lines)
    assert "2 video(s) FAILED" in text
    assert "20240101_ABC0101_P1" in text
    assert "1 video(s) are parked" in text


def test_health_says_nothing_stuck():
    text = "\n".join(health_report(StubDB()))
    assert "Nothing is stuck" in text


def test_health_headline_without_db():
    lines = health_report(None)
    assert lines and lines[0].startswith("The auto-processor is")


def test_manifest_records_external_pose_and_honest_flag(tmp_path, monkeypatch):
    stem = "20240101_ABC0101_P1"
    bundle = tmp_path / stem
    bundle.mkdir()
    ext = tmp_path / (stem + "DLC_newmodel.h5")
    ext.write_text("x", encoding="utf-8")
    monkeypatch.setattr(rr, "_resolve_inputs", lambda b, s: (None, ext))
    rg._write_review_manifest(bundle, stem, "why")
    doc = json.loads((bundle / (stem + "_manifest.json")).read_text(encoding="utf-8"))
    assert doc["canonical_dlc_h5_path"] == str(ext)
    assert doc["provenance"]["self_contained"] is False


def test_manifest_in_bundle_pose_is_self_contained(tmp_path):
    stem = "20240101_ABC0101_P1"
    bundle = tmp_path / stem
    bundle.mkdir()
    h5 = bundle / (stem + "DLC_newmodel.h5")
    h5.write_text("x", encoding="utf-8")
    rg._write_review_manifest(bundle, stem, "why")
    doc = json.loads((bundle / (stem + "_manifest.json")).read_text(encoding="utf-8"))
    assert doc["canonical_dlc_h5_path"] == str(h5)
    assert doc["provenance"]["self_contained"] is True


def _staged_pose_env(tmp_path, monkeypatch, stem):
    """A share with a declared-scorer pose only in staging (Processing/Posed)
    and an empty triage bundle; the real resolver is used."""
    import mousereach.pipeline.versions as vmod
    nas = tmp_path / "nas"
    staging = nas / "Processing" / "Posed"
    staging.mkdir(parents=True)
    staged = staging / (stem + "DLC_newmodel.h5")
    staged.write_bytes(b"pose-bytes")
    bundle = nas / "Processing" / "Review" / "triage" / stem
    bundle.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "NAS_ROOT", nas)
    monkeypatch.setattr(rr.Paths, "ANALYZED_OUTPUT", None)
    monkeypatch.setattr(rr.Paths, "DLC_STAGING", staging)
    monkeypatch.setattr(vmod, "get_current_versions",
                        lambda root: {"versions": {"dlc_scorer": "DLC_newmodel"}})
    return staged, bundle


def test_manifest_never_points_at_staging_it_copies_the_pose_in(tmp_path, monkeypatch):
    # Staging is a transient handover folder: a manifest naming it went stale
    # when the pose moved on (and when the old staging folder was set aside by
    # the layout change), so the bundle refused to return every cycle. The pose
    # now travels IN the bundle, and the staged copy is left for intake.
    stem = "20240101_ABC0101_P1"
    staged, bundle = _staged_pose_env(tmp_path, monkeypatch, stem)
    rg._write_review_manifest(bundle, stem, "why")
    doc = json.loads((bundle / (stem + "_manifest.json")).read_text(encoding="utf-8"))
    local = bundle / staged.name
    assert local.read_bytes() == b"pose-bytes"
    assert doc["canonical_dlc_h5_path"] == str(local)
    assert doc["provenance"]["self_contained"] is True
    assert staged.exists()
    assert not list(bundle.glob("*.part"))
    # And the return path now finds the pose in the bundle itself.
    assert rr._resolve_inputs(bundle, stem)[1] == local


def test_manifest_pose_pointer_stays_empty_when_the_staged_copy_fails(tmp_path, monkeypatch):
    stem = "20240101_ABC0101_P1"
    staged, bundle = _staged_pose_env(tmp_path, monkeypatch, stem)

    def no_space(src, dst, *a, **k):
        Path(dst).write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(rg.shutil, "copy2", no_space)
    rg._write_review_manifest(bundle, stem, "why")
    doc = json.loads((bundle / (stem + "_manifest.json")).read_text(encoding="utf-8"))
    assert doc["canonical_dlc_h5_path"] is None
    assert doc["provenance"]["self_contained"] is False
    assert sorted(p.name for p in bundle.iterdir()) == [stem + "_manifest.json"]


def test_dashboard_widget_imports():
    import mousereach.dashboard.widget  # noqa: F401
