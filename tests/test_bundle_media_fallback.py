"""Review bundles must open even when their manifest's absolute paths are stale.

A bundle manifest records ABSOLUTE canonical_video_path / canonical_dlc_h5_path.
Renaming the queue folder a bundle sits in (flagged_for_review -> deep_review)
left every older manifest pointing at a folder that no longer exists, and the
review tool opened that path with no fallback. These tests pin the resolver the
tool now uses: manifest path if it is a file, else the bundle's own copy, else
None. The widget wiring is tested at the bottom by calling its methods unbound
on a stub -- no Qt widget or napari viewer is built.
"""

import os
from pathlib import Path
from types import SimpleNamespace

from mousereach.review.bundle_media import resolve_bundle_pose, resolve_bundle_video

STEM = "20250101_TEST0101_P1"
SCORER_A = "DLC_resnet50_ModelAshuffle1_100000"
SCORER_B = "DLC_resnet50_ModelBshuffle3_200000"


def _touch(p, mtime=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _bundle(root, name="deep_review"):
    b = root / "Processing" / "Review" / name / STEM
    b.mkdir(parents=True)
    return b


# --------------------------------------------------------------------- video
def test_manifest_video_wins_when_it_exists(tmp_path):
    bundle = _bundle(tmp_path)
    _touch(bundle / f"{STEM}.mp4")
    canonical = _touch(tmp_path / "Analyzed" / "PROJ" / "01" / f"{STEM}.mp4")
    manifest = {"video_stem": STEM, "canonical_video_path": str(canonical)}
    assert resolve_bundle_video(manifest, bundle) == canonical


def test_stale_manifest_video_falls_back_to_bundle_copy(tmp_path):
    bundle = _bundle(tmp_path)
    local = _touch(bundle / f"{STEM}.mp4")
    stale = tmp_path / "Processing" / "Review" / "flagged_for_review" / STEM / f"{STEM}.mp4"
    manifest = {"video_stem": STEM, "canonical_video_path": str(stale)}
    assert resolve_bundle_video(manifest, bundle) == local


def test_manifest_path_through_a_plain_file_falls_back_without_raising(tmp_path):
    # The data migration leaves a plain FILE at the old folder name so old code
    # fails loudly; a stale path running through it must read as "not found".
    bundle = _bundle(tmp_path)
    local = _touch(bundle / f"{STEM}.mp4")
    old = tmp_path / "Processing" / "Review" / "flagged_for_review"
    old.write_text("moved", encoding="utf-8")
    manifest = {"video_stem": STEM,
                "canonical_video_path": str(old / STEM / f"{STEM}.mp4"),
                "canonical_dlc_h5_path": str(old / STEM / f"{STEM}{SCORER_A}.h5")}
    assert resolve_bundle_video(manifest, bundle) == local
    assert resolve_bundle_pose(manifest, bundle, expected_scorer=SCORER_A) is None


def test_stem_falls_back_to_bundle_folder_name(tmp_path):
    bundle = _bundle(tmp_path)
    local = _touch(bundle / f"{STEM}.mp4")
    assert resolve_bundle_video({"canonical_video_path": None}, bundle) == local


def test_nothing_found_is_none(tmp_path):
    bundle = _bundle(tmp_path)
    manifest = {"video_stem": STEM,
                "canonical_video_path": str(tmp_path / "gone" / f"{STEM}.mp4")}
    assert resolve_bundle_video(manifest, bundle) is None
    assert resolve_bundle_pose(manifest, bundle) is None
    assert resolve_bundle_video({}, None) is None
    assert resolve_bundle_pose(None, None) is None


def test_bundle_dir_that_is_a_file_is_none(tmp_path):
    not_a_dir = tmp_path / STEM
    not_a_dir.write_text("not a folder", encoding="utf-8")
    assert resolve_bundle_video({"video_stem": STEM}, not_a_dir) is None
    assert resolve_bundle_pose({"video_stem": STEM}, not_a_dir) is None


# ---------------------------------------------------------------------- pose
def test_manifest_pose_wins_when_it_exists(tmp_path):
    bundle = _bundle(tmp_path)
    _touch(bundle / f"{STEM}{SCORER_B}.h5")
    declared = _touch(tmp_path / "Analyzed" / "PROJ" / "01" / f"{STEM}{SCORER_A}.h5")
    manifest = {"video_stem": STEM, "canonical_dlc_h5_path": str(declared)}
    assert resolve_bundle_pose(manifest, bundle, expected_scorer=SCORER_B) == declared


def test_stale_manifest_pose_falls_back_to_bundle_pose(tmp_path):
    bundle = _bundle(tmp_path)
    local = _touch(bundle / f"{STEM}{SCORER_A}.h5")
    stale = tmp_path / "Processing" / "Review" / "flagged_for_review" / STEM / local.name
    manifest = {"video_stem": STEM, "canonical_dlc_h5_path": str(stale)}
    assert resolve_bundle_pose(manifest, bundle) == local


def test_bundle_pose_prefers_declared_model_over_newer_file(tmp_path):
    bundle = _bundle(tmp_path)
    declared = _touch(bundle / f"{STEM}{SCORER_A}.h5", mtime=1_000_000)
    _touch(bundle / f"{STEM}{SCORER_B}.h5", mtime=2_000_000)  # newer, wrong model
    manifest = {"video_stem": STEM, "canonical_dlc_h5_path": None}
    assert resolve_bundle_pose(manifest, bundle, expected_scorer=SCORER_A) == declared


def test_bundle_pose_ignores_other_videos_pose(tmp_path):
    bundle = _bundle(tmp_path)
    _touch(bundle / f"20250101_TEST0102_P1{SCORER_A}.h5")
    assert resolve_bundle_pose({"video_stem": STEM}, bundle) is None


# -------------------------------------------------------------- widget wiring
# WHY these exist: the resolver above is only useful if the review widget calls
# it. A regression back to opening Path(manifest["canonical_video_path"]) would
# make every migrated deep-review bundle unopenable with no test failing. The
# widget methods are called UNBOUND on a stub self, so no Qt widget or napari
# viewer is built.

def _widget_module():
    import mousereach.review.causal_review_widget as crw
    return crw


def _recording_self(**attrs):
    calls = {"load_video": [], "banner": 0}
    me = SimpleNamespace(**attrs)
    me._load_video = lambda path: calls["load_video"].append(path)

    def banner():
        calls["banner"] += 1

    me._update_routing_banner = banner
    return me, calls


def _retired_queue_guard(root):
    old = root / "Processing" / "Review" / "flagged_for_review"
    old.write_text("moved", encoding="utf-8")
    return old


def test_widget_opens_the_bundle_copy_when_the_manifest_video_is_stale(tmp_path, monkeypatch):
    crw = _widget_module()
    errors = []
    monkeypatch.setattr(crw, "show_error", lambda msg: errors.append(msg))
    bundle = _bundle(tmp_path)
    local = _touch(bundle / f"{STEM}.mp4")
    old = _retired_queue_guard(tmp_path)
    manifest = {"video_stem": STEM,
                "canonical_video_path": str(old / STEM / f"{STEM}.mp4")}
    me, calls = _recording_self()

    crw.CausalReviewWidget.load_from_manifest(me, manifest, bundle)

    assert calls["load_video"] == [local]
    assert calls["banner"] == 1
    assert me._manifest == manifest and me._bundle_dir == bundle
    assert errors == []


def test_widget_shows_an_error_and_keeps_its_bundle_when_no_video_resolves(tmp_path, monkeypatch):
    crw = _widget_module()
    errors = []
    monkeypatch.setattr(crw, "show_error", lambda msg: errors.append(msg))
    bundle = _bundle(tmp_path)                      # no mp4 inside
    previous_manifest = {"video_stem": "20250101_TEST0199_P1"}
    previous_bundle = tmp_path / "previous_bundle"
    me, calls = _recording_self(_manifest=previous_manifest,
                                _bundle_dir=previous_bundle)
    manifest = {"video_stem": STEM,
                "canonical_video_path": str(tmp_path / "gone" / f"{STEM}.mp4")}

    crw.CausalReviewWidget.load_from_manifest(me, manifest, bundle)

    assert calls["load_video"] == [] and calls["banner"] == 0
    # The video on screen stays paired with its own bundle.
    assert me._manifest is previous_manifest
    assert me._bundle_dir == previous_bundle
    assert len(errors) == 1 and STEM in errors[0]


def _pose_loader_self(video, manifest, bundle):
    return SimpleNamespace(video_path=video, _manifest=manifest,
                           _bundle_dir=bundle, _video_stem=STEM, dlc_df="untouched")


def test_widget_pose_loader_hands_bundle_and_stem_to_the_resolver(tmp_path, monkeypatch):
    crw = _widget_module()
    import pandas as pd
    import mousereach.review.bundle_media as bm

    bundle = _bundle(tmp_path)
    local_pose = _touch(bundle / f"{STEM}{SCORER_A}.h5")
    # The video is opened from elsewhere, with no pose beside it, so only the
    # resolver's bundle fallback can supply the pose.
    video = _touch(tmp_path / "Analyzed" / "PROJ" / "01" / f"{STEM}.mp4")
    old = _retired_queue_guard(tmp_path)
    manifest = {"video_stem": STEM,
                "canonical_dlc_h5_path": str(old / STEM / local_pose.name)}

    seen = []
    real = bm.resolve_bundle_pose

    def spy(m, b, s=None, expected_scorer=None):
        seen.append((m, b, s))
        return real(m, b, s, expected_scorer=expected_scorer)

    read = []
    cols = pd.MultiIndex.from_tuples([("scorer", "Nose", "x"), ("scorer", "Nose", "y")])

    def fake_read_hdf(path, *a, **k):
        read.append(Path(path))
        return pd.DataFrame([[1.0, 2.0]], columns=cols)

    monkeypatch.setattr(bm, "resolve_bundle_pose", spy)
    monkeypatch.setattr(pd, "read_hdf", fake_read_hdf)
    me = _pose_loader_self(video, manifest, bundle)

    crw.CausalReviewWidget._load_dlc_data(me)

    assert seen == [(manifest, bundle, STEM)]
    assert read == [local_pose]
    assert list(me.dlc_df.columns) == ["Nose_x", "Nose_y"]


def test_widget_pose_loader_leaves_no_pose_when_nothing_resolves(tmp_path, monkeypatch):
    crw = _widget_module()
    import pandas as pd

    bundle = _bundle(tmp_path)                      # no pose inside
    video = _touch(tmp_path / "Analyzed" / "PROJ" / "01" / f"{STEM}.mp4")
    manifest = {"video_stem": STEM,
                "canonical_dlc_h5_path": str(tmp_path / "gone" / f"{STEM}{SCORER_A}.h5")}
    read = []
    monkeypatch.setattr(pd, "read_hdf", lambda path, *a, **k: read.append(path))
    me = _pose_loader_self(video, manifest, bundle)

    crw.CausalReviewWidget._load_dlc_data(me)

    assert read == []
    assert me.dlc_df is None
