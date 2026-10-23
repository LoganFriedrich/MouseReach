"""Rebuilding a lost multi-animal recording from its singles.

WHY (2026-10-23): the input to MouseReach is the recording; a single fed in by
hand has no link to the recording it came from. Where only singles survive,
the recording is rebuilt (each single back in its grid cell) and goes in like
any other. Built under tmp_path with tiny synthetic videos.
"""
import json

import cv2
import numpy as np
import pytest

from mousereach.video_prep.core import rebuild as rb

NAME = "20990101_PRJ0101,PRJ0102,PRJ0103,PRJ0104,PRJ0105,PRJ0106,PRJ0107,PRJ0001_P1"
IDS = NAME.split("_")[1].split(",")


def _video(path, n=12, size=(480, 540), shade=0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), shade, np.uint8))
    w.release()
    return path


def _singles(tmp_path, n=12, odd=None):
    out = {}
    for k, aid in enumerate(IDS[:7]):                      # position 8 is empty (PRJ0001)
        frames = n + 1 if aid == odd else n
        out[aid] = _video(tmp_path / f"20990101_{aid}_P1.mp4", frames, shade=30 * k)
    return out


def test_a_recording_is_rebuilt_cell_by_cell_with_a_record(tmp_path):
    singles = _singles(tmp_path)
    dest, records = tmp_path / "intake", tmp_path / "records"
    path = rb.rebuild(NAME, singles, dest, records, "test")
    got = rb.probe(path)
    assert (got["width"], got["height"], got["frames"]) == (1920, 1080, 12)
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok
    # position 3 (shade 60) sits in the cell the cropper cuts for position 3
    w, h, x, y = rb.CROP_COORDS[2]
    assert abs(frame[y:y + h, x:x + w].mean() - 60) < 8
    w, h, x, y = rb.CROP_COORDS[7]                          # the empty position is black
    assert frame[y:y + h, x:x + w].mean() < 8
    rec = json.loads((records / (NAME + ".json")).read_text(encoding="utf-8"))
    assert rec["frames"] == 12 and rec["positions"][7]["blank"] is True
    assert "Not an original recording" in rec["note"]


def test_singles_that_do_not_line_up_are_refused(tmp_path):
    with pytest.raises(rb.RebuildRefused, match="line up"):
        rb.plan(NAME, _singles(tmp_path, odd="PRJ0104"))


def test_a_single_of_another_size_is_refused(tmp_path):
    singles = _singles(tmp_path)
    singles["PRJ0102"] = _video(tmp_path / "other.mp4", size=(500, 540))
    with pytest.raises(rb.RebuildRefused, match="not 480x540"):
        rb.plan(NAME, singles)


def test_a_missing_single_and_an_existing_output_are_refused(tmp_path):
    singles = _singles(tmp_path)
    del singles["PRJ0105"]
    with pytest.raises(rb.RebuildRefused, match="no single for PRJ0105"):
        rb.plan(NAME, singles)
    dest = tmp_path / "intake"
    dest.mkdir()
    (dest / (NAME + ".mkv")).write_bytes(b"already here")
    with pytest.raises(rb.RebuildRefused, match="already exists"):
        rb.rebuild(NAME, _singles(tmp_path), dest, None, "test")
    assert (dest / (NAME + ".mkv")).read_bytes() == b"already here"
