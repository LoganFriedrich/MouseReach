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


def test_dashboard_widget_imports():
    import mousereach.dashboard.widget  # noqa: F401
