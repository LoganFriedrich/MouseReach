"""Pipeline tools never read a superseded output under Analyzed/Archive/.

WHY: superseded outputs are moved to Analyzed/Archive/ and keep their ORIGINAL
names ({stem}_processing_manifest.json, {stem}_features.json, {stem}.mp4,
{stem}DLC_...h5). Every tool here used to walk Analyzed with rglob, so an older
generation was read as if it were live -- silently: the manifest backfills
(with --apply) rewrote archived manifests, the field audit tallied old fields,
the cohort export gained old reaches, and the reprocess path could pick an
archived "DLC Model 4.0" pose (its path contains "DLC Model 4", the preference
substring).

Each test builds a live copy AND a same-named decoy under Analyzed/Archive/,
calls the REAL function at the site, and asserts the decoy is ignored while
the live data is still found. Every test is written so it fails against the
old rglob code WHATEVER order rglob visits folders in (NTFS lists Archive
before Connectome; Python 3.13's rglob pops its stack in reverse): counts
include the decoy, or the decoy is the ONLY candidate for one of the videos,
or the decoy is the only one the old preference/filter would accept.
"""
import csv
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.config as cfg

VID = "20240101_ABC0101_P1"
VID2 = "20240102_ABC0102_P1"
VID3 = "20240103_ABC0103_P1"
POSE = "DLC_resnet101_TestNetshuffle3_100000.h5"


@pytest.fixture
def an(tmp_path, monkeypatch):
    """A tmp Analyzed tree: live cohort folder, live DLC Model 4 pose folder,
    and the superseded-output archive. Every configured root points into
    tmp_path so nothing can reach the shared drive."""
    root = tmp_path / "nas" / "Analyzed"
    live = root / "Connectome" / "ABC01"
    model4 = root / "Connectome" / "DLC Model 4" / "ABC01"
    archive = root / "Archive" / "superseded_processing_root_3.1"
    archive_model4 = root / "Archive" / "DLC Model 4.0" / "ABC01"
    for d in (live, archive, archive_model4):
        d.mkdir(parents=True)
    monkeypatch.setattr(cfg.Paths, "NAS_ROOT", tmp_path / "nas")
    monkeypatch.setattr(cfg.Paths, "ANALYZED_OUTPUT", root)
    monkeypatch.setattr(cfg.Paths, "TRIAGE_REVIEW", None)
    monkeypatch.setattr(cfg.Paths, "DEEP_REVIEW", None)
    return SimpleNamespace(root=root, live=live, model4=model4,
                           archive=archive, archive_model4=archive_model4)


def _write_json(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _fake_version_index(monkeypatch):
    """The backfills upsert into the version index, whose default home is the
    NAS. Record the upserts instead of opening any database."""
    import mousereach.pipeline.version_index as vi
    upserts = []
    monkeypatch.setattr(vi, "VersionIndex", lambda *a, **k: SimpleNamespace(
        upsert_from_manifest=lambda stem, m, p: upserts.append(Path(p))))
    return upserts


# --- manifest.py ------------------------------------------------------------

def test_backfill_manifest_versions_never_visits_archived_manifest(an, tmp_path, monkeypatch):
    import mousereach.pipeline.manifest as manifest
    upserts = _fake_version_index(monkeypatch)
    for folder in (an.live, an.archive):
        _write_json(folder / f"{VID}_processing_manifest.json",
                    {"pipeline_versions": {"segmenter": "not_run"}})
        _write_json(folder / f"{VID}_segments.json", {"segmenter_version": "2.2.2"})
    decoy = an.archive / f"{VID}_processing_manifest.json"
    decoy_before = decoy.read_bytes()

    res = manifest.backfill_manifest_versions(
        an.root, apply=True, archive_dir=tmp_path / "backup", stages=["segmenter"])

    # Old rglob: 2 manifests, 2 fixed, and the archived one rewritten.
    assert res["stats"]["manifests"] == 1
    assert res["stats"]["segmenter: fixed"] == 1
    assert decoy.read_bytes() == decoy_before
    live = json.loads((an.live / f"{VID}_processing_manifest.json").read_text())
    assert live["pipeline_versions"]["segmenter"] == "2.2.2"
    assert upserts == [an.live / f"{VID}_processing_manifest.json"]


def test_backfill_kinematic_versions_never_visits_archived_manifest(an, monkeypatch):
    import mousereach.pipeline.manifest as manifest
    upserts = _fake_version_index(monkeypatch)
    for folder in (an.live, an.archive):
        _write_json(folder / f"{VID}_processing_manifest.json",
                    {"pipeline_versions": {"kinematic_extractor": "not_run"}})
        _write_json(folder / f"{VID}_features.json", {"extractor_version": "1.4.0"})
    decoy = an.archive / f"{VID}_processing_manifest.json"
    decoy_before = decoy.read_bytes()

    res = manifest.backfill_kinematic_versions(an.root, apply=True)

    # Old rglob: 2 manifests, 2 fixed, and the archived one rewritten.
    assert res["counts"]["manifests"] == 1
    assert res["counts"]["fixed"] == 1
    assert decoy.read_bytes() == decoy_before
    live = json.loads((an.live / f"{VID}_processing_manifest.json").read_text())
    assert live["pipeline_versions"]["kinematic_extractor"] == "1.4.0"
    assert upserts == [an.live / f"{VID}_processing_manifest.json"]


def test_backfill_cli_default_backup_lands_outside_analyzed(an, tmp_path, monkeypatch, capsys):
    """backfill-manifest-versions --root Analyzed/<project> --apply. Old code
    defaulted the backup folder to Path(root).parent / "_archived", i.e.
    Analyzed/_archived/, putting an original-named manifest copy inside the
    tree every walker reads. It must land under NAS_ROOT/_archived instead."""
    import mousereach.pipeline.manifest as manifest
    _fake_version_index(monkeypatch)
    _write_json(an.live / f"{VID}_processing_manifest.json",
                {"pipeline_versions": {"segmenter": "not_run"}})
    _write_json(an.live / f"{VID}_segments.json", {"segmenter_version": "2.2.2"})
    monkeypatch.setattr(sys, "argv", [
        "mousereach-backfill-manifest-versions", "--root", str(an.root / "Connectome"),
        "--stage", "segmenter", "--apply"])

    assert manifest.main_backfill_manifest_versions() == 0

    assert not (an.root / "_archived").exists()                # old: created here
    copies = list((tmp_path / "nas" / "_archived").glob(
        f"manifests_pre_version_backfill_*/{VID}_processing_manifest.json"))
    assert len(copies) == 1
    capsys.readouterr().out.encode("ascii")


# --- field_audit.py ---------------------------------------------------------

def test_field_audit_scan_files_ignores_archived_outputs(an):
    from mousereach.pipeline.field_audit import scan_files
    _write_json(an.live / f"{VID}_features.json",
                {"segments": [{"reaches": [{"live_field": 1.5}]}]})
    _write_json(an.archive / f"{VID}_features.json",
                {"segments": [{"reaches": [{"archived_field": 2.5}]}]})

    res = scan_files(an.root)

    # Old rglob: 2 files read and archived_field tallied.
    assert res["kinematics"]["files"] == 1
    assert "live_field" in res["kinematics"]["fields"]
    assert "archived_field" not in res["kinematics"]["fields"]


# --- reach_export.py --------------------------------------------------------

def _features(start_frames):
    return {"video_name": VID,
            "segments": [{"segment_num": 1, "outcome": "retrieved",
                          "reaches": [{"reach_id": i + 1, "start_frame": s}
                                      for i, s in enumerate(start_frames)]}]}


def test_export_cohort_emits_no_rows_from_archived_features(an, tmp_path):
    from mousereach.kinematics.analysis.reach_export import export_cohort
    _write_json(an.live / f"{VID}_features.json", _features([10]))
    _write_json(an.archive / f"{VID}_features.json", _features([990, 999]))
    out = tmp_path / "cohort.csv"

    csv_path, n_rows, n_videos = export_cohort("ABC01", output_csv=out)

    # Old rglob: 3 rows (1 live + 2 archived) for the same video.
    assert (csv_path, n_rows, n_videos) == (out, 1, 1)
    with open(out, newline="") as f:
        starts = [r["start_frame"] for r in csv.DictReader(f)]
    assert starts == ["10"]


# --- reprocess_to_current.py ------------------------------------------------

def test_find_current_pose_skips_archive_and_keeps_model4_preference(an):
    from mousereach.pipeline.reprocess_to_current import find_current_pose
    # VID: the live pose sits beside the video only; the archived pose's path
    # contains "DLC Model 4". Old code prefers it whichever is walked first.
    live_vid = an.live / f"{VID}{POSE}"
    live_vid.write_bytes(b"live")
    (an.archive_model4 / f"{VID}{POSE}").write_bytes(b"archived")
    # VID2: live DLC Model 4 pose, a live cohort pose, and the archived decoy.
    # The live DLC Model 4 pose must still win (Model folders are not pruned).
    an.model4.mkdir(parents=True)
    live_vid2 = an.model4 / f"{VID2}{POSE}"
    live_vid2.write_bytes(b"live model 4")
    (an.live / f"{VID2}{POSE}").write_bytes(b"live cohort")
    (an.archive_model4 / f"{VID2}{POSE}").write_bytes(b"archived")

    assert find_current_pose(VID) == live_vid
    assert find_current_pose(VID2) == live_vid2


def test_find_video_file_skips_archive(an):
    from mousereach.pipeline.reprocess_to_current import find_video_file
    # VID: live mp4 exists only in DLC Model staging; the archived copy is the
    # only one the old "not DLC Model" filter accepts, so old code returns it.
    an.model4.mkdir(parents=True)
    staged = an.model4 / f"{VID}.mp4"
    staged.write_bytes(b"live staged")
    (an.archive / f"{VID}.mp4").write_bytes(b"archived")
    # VID2: ordinary case -- live in the cohort folder, decoy in the archive.
    live2 = an.live / f"{VID2}.mp4"
    live2.write_bytes(b"live")
    (an.archive / f"{VID2}.mp4").write_bytes(b"archived")

    assert find_video_file(VID) == staged
    assert find_video_file(VID2) == live2


def test_build_reprocess_worklist_skips_archived_poses(an):
    from mousereach.pipeline.reprocess_to_current import build_reprocess_worklist
    # No Connectome/DLC Model 4 folder, so the scan falls back to Analyzed itself.
    live_pose = an.live / f"{VID}{POSE}"
    live_pose.write_bytes(b"live")
    live_mp4 = an.live / f"{VID}.mp4"
    live_mp4.write_bytes(b"video")
    (an.archive_model4 / f"{VID}{POSE}").write_bytes(b"archived")
    # VID3 exists ONLY in the archive: old code queues it as work.
    (an.archive_model4 / f"{VID3}{POSE}").write_bytes(b"archived only")

    work = build_reprocess_worklist()

    assert work == [(VID, live_pose, live_mp4)]


# --- fix_segmentation_widget.py ---------------------------------------------

def test_fix_segmentation_load_video_skips_archived_mp4(an, tmp_path, monkeypatch):
    from mousereach.review.fix_segmentation_widget import FixSegmentationWidget
    # Stand-ins for the video decoder and the napari-backed lazy layer, so the
    # real _load_video runs headless; _add_dlc_overlay records the mp4 chosen.
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.CAP_PROP_FRAME_COUNT, fake_cv2.CAP_PROP_FRAME_HEIGHT, \
        fake_cv2.CAP_PROP_FRAME_WIDTH = 7, 4, 3
    fake_cv2.VideoCapture = lambda p: SimpleNamespace(get=lambda k: 0,
                                                      release=lambda: None)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    fake_crw = types.ModuleType("mousereach.review.causal_review_widget")
    fake_crw._LazyVideo = lambda path, n, h, w: SimpleNamespace(path=path)
    monkeypatch.setitem(sys.modules, "mousereach.review.causal_review_widget", fake_crw)

    def load(stem):
        chosen, status = [], []
        fake_self = SimpleNamespace(
            status=SimpleNamespace(setText=status.append),
            _video_layer=None, n_frames=0,
            viewer=SimpleNamespace(
                layers=[], add_image=lambda data, **k: None,
                dims=SimpleNamespace(events=SimpleNamespace(
                    current_step=SimpleNamespace(connect=lambda cb: None)))),
            _goto_spin=SimpleNamespace(setRange=lambda a, b: None),
            _add_dlc_overlay=lambda bundle, s, mp4: chosen.append(Path(mp4)),
            _bind_nav_keys=lambda: None,
            _on_frame_change=lambda *a: None)
        bundle = tmp_path / "bundle" / stem
        bundle.mkdir(parents=True)  # not self-contained: no mp4, no manifest
        FixSegmentationWidget._load_video(fake_self, bundle, stem)
        return chosen, status

    # VID exists ONLY as an archived copy: old code loads it.
    (an.archive / f"{VID}.mp4").write_bytes(b"archived")
    chosen, status = load(VID)
    assert chosen == []
    assert any("No video file found" in s for s in status)

    # VID2 has a live copy and an archived decoy: the live one loads.
    live2 = an.live / f"{VID2}.mp4"
    live2.write_bytes(b"live")
    (an.archive / f"{VID2}.mp4").write_bytes(b"archived")
    chosen, _ = load(VID2)
    assert chosen == [live2]
