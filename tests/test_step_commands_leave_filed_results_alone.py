"""Hand-run step and backfill commands do not rewrite filed results unseen.

WHY: a filed video has passed every check, and its files are the record of
the run that passed them. ``mousereach-grasp-analyze -i <filed folder>
--overwrite`` rewrote its features file in place (no earlier generation, no
watcher record, no crystallize lock asked), and
``mousereach-backfill-kinematic-versions --apply`` rewrote filed manifests
with no copy kept and no lock asked. One wrong folder argument changed filed
results silently. Video ids are invented.
"""
import json
import sys
from types import SimpleNamespace

import pytest

V = "20990101_CNT9901_P1"
V2 = "20990101_CNT9902_P1"


def test_grasp_analyze_refuses_to_write_into_the_filed_tree(tmp_path, monkeypatch):
    from mousereach.config import Paths
    from mousereach.kinematics import cli
    analyzed = tmp_path / "Analyzed"
    filed = analyzed / "PRJ" / "C01"
    filed.mkdir(parents=True)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed, raising=False)
    (filed / f"{V}DLC_resnet101_X_shuffle3_100000.h5").write_bytes(b"pose")
    (filed / f"{V}_reaches.json").write_text("{}", encoding="utf-8")
    (filed / f"{V}_pellet_outcomes.json").write_text("{}", encoding="utf-8")
    features = filed / f"{V}_features.json"
    features.write_text('{"filed": true}', encoding="utf-8")

    class FakeExtractor:
        def extract(self, *a, **k):
            return SimpleNamespace(to_dict=lambda: {"rewritten": True})

    monkeypatch.setattr(cli, "FeatureExtractor", FakeExtractor)
    monkeypatch.setattr(sys, "argv", ["mousereach-grasp-analyze", "-i", str(filed),
                                      "--overwrite"])
    with pytest.raises(SystemExit) as e:
        cli.main_batch()
    assert e.value.code == 1
    assert json.loads(features.read_text()) == {"filed": True}


def test_grasp_analyze_still_writes_outside_the_filed_tree(tmp_path, monkeypatch):
    from mousereach.config import Paths
    from mousereach.kinematics import cli
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    (work / f"{V}DLC_resnet101_X_shuffle3_100000.h5").write_bytes(b"pose")
    (work / f"{V}_reaches.json").write_text("{}", encoding="utf-8")
    (work / f"{V}_pellet_outcomes.json").write_text("{}", encoding="utf-8")

    class FakeExtractor:
        def extract(self, *a, **k):
            return SimpleNamespace(to_dict=lambda: {"made": True})

    monkeypatch.setattr(cli, "FeatureExtractor", FakeExtractor)
    monkeypatch.setattr(sys, "argv", ["mousereach-grasp-analyze", "-i", str(work)])
    try:
        cli.main_batch()
    except SystemExit as e:
        assert not e.code
    assert json.loads((work / f"{V}_features.json").read_text()) == {"made": True}


def test_kinematic_backfill_keeps_a_copy_and_leaves_a_crystallized_video_alone(tmp_path, monkeypatch):
    from mousereach.config import Paths
    from mousereach.pipeline import manifest
    nas = tmp_path / "nas"
    analyzed = nas / "Analyzed"
    filed = analyzed / "PRJ" / "C01"
    filed.mkdir(parents=True)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed, raising=False)
    for vid in (V, V2):
        (filed / f"{vid}_processing_manifest.json").write_text(
            json.dumps({"pipeline_versions": {"kinematic_extractor": "not_run"}}),
            encoding="utf-8")
        (filed / f"{vid}_features.json").write_text(
            json.dumps({"extractor_version": "9.9.9"}), encoding="utf-8")
    locked_before = (filed / f"{V2}_processing_manifest.json").read_bytes()

    import mousereach.pipeline.version_index as vi
    monkeypatch.setattr(vi, "VersionIndex", lambda *a, **k: SimpleNamespace(
        upsert_from_manifest=lambda *a, **k: None))
    from mousereach.watcher import db_location
    rows = {V: {"state": "archived"},
            V2: {"state": "crystallized", "crystallized_label": "PAPER_X"}}
    monkeypatch.setattr(db_location, "open_watcher_db_if_present", lambda: (
        SimpleNamespace(get_video=lambda vid: rows.get(vid)), None))
    monkeypatch.setattr(sys, "argv", ["mousereach-backfill-kinematic-versions",
                                      "--apply", "--root", str(analyzed)])
    manifest.main_backfill_kinematic_versions()

    fixed = json.loads((filed / f"{V}_processing_manifest.json").read_text())
    assert fixed["pipeline_versions"]["kinematic_extractor"] == "9.9.9"
    copies = list((nas / "_archived").rglob(f"{V}_processing_manifest.json"))
    assert len(copies) == 1, "the original is copied before it is changed"
    assert json.loads(copies[0].read_text())["pipeline_versions"]["kinematic_extractor"] == "not_run"
    assert (filed / f"{V2}_processing_manifest.json").read_bytes() == locked_before


@pytest.mark.parametrize("module, core, prog", [
    ("mousereach.segmentation.cli", "mousereach.segmentation.core", "mousereach-segment"),
    ("mousereach.reach.cli", "mousereach.reach.core", "mousereach-detect-reaches"),
    ("mousereach.outcomes.cli", "mousereach.outcomes.core.batch", "mousereach-detect-outcomes"),
])
def test_algorithm_step_commands_refuse_the_filed_tree(tmp_path, monkeypatch, module, core, prog):
    import importlib
    from mousereach.config import Paths
    mod = importlib.import_module(module)
    analyzed = tmp_path / "Analyzed"
    filed = analyzed / "PRJ" / "C01"
    filed.mkdir(parents=True)
    (filed / f"{V}DLC_resnet101_X_shuffle3_100000.h5").write_bytes(b"pose")
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed, raising=False)
    ran = []
    monkeypatch.setattr(importlib.import_module(core), "process_batch",
                        lambda *a, **k: ran.append(1) or {})
    main = getattr(mod, "main_batch", None) or getattr(mod, "main")
    monkeypatch.setattr(sys, "argv", [prog, "-i", str(filed)])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    assert ran == []
