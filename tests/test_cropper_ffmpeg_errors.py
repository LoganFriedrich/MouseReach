"""A slot ffmpeg cannot cut says why, and leaves no partial file.

WHY (2026-10-22): the cropper ran ffmpeg through os.system, so ffmpeg's own
message went to a console nobody reads; eight slots of one collage failed in
the same instant and the log said only "ffmpeg returned 1". A failed run can
also leave a partial file under the single's name.
"""
import subprocess
from types import SimpleNamespace

import mousereach.video_prep.core.cropper as cropper

COLLAGE = ("20250101_CNT0101,CNT0001,CNT0002,CNT0003,CNT0004,CNT0005,"
           "CNT0006,CNT0007_P1.mkv")


def test_ffmpegs_message_is_kept_and_the_partial_file_removed(tmp_path, monkeypatch):
    src = tmp_path / COLLAGE
    src.write_bytes(b"collage")
    out = tmp_path / "out"

    def fake_run(cmd, shell, capture_output, text, errors):
        (out / "20250101_CNT0101_P1.mp4").write_bytes(b"half a video")
        return SimpleNamespace(returncode=1, stderr="\n[mov] moov atom not found\n"
                                                    "Invalid data found when processing input\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(cropper, "write_crop_manifest", lambda *a, **k: None, raising=False)

    results = cropper.crop_collage(src, out, verbose=False)

    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["error"] == ("ffmpeg returned 1: [mov] moov atom not found | "
                                  "Invalid data found when processing input")
    assert not (out / "20250101_CNT0101_P1.mp4").exists()


def test_a_clean_cut_is_still_a_success(tmp_path, monkeypatch):
    src = tmp_path / COLLAGE
    src.write_bytes(b"collage")
    out = tmp_path / "out"

    def fake_run(cmd, shell, capture_output, text, errors):
        (out / "20250101_CNT0101_P1.mp4").write_bytes(b"video")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    results = cropper.crop_collage(src, out, verbose=False)
    ok = [r for r in results if r["status"] == "success"]
    assert [r["animal_id"] for r in ok] == ["CNT0101"]
