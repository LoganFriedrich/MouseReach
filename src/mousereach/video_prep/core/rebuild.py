"""Rebuild a multi-animal recording from its single-animal cuts.

WHY THIS EXISTS
---------------
The input to MouseReach is the multi-animal recording: the pipeline cuts it
into single-animal videos itself, and that cut is what records which recording
and grid position each single came from (the shared record the GPU nodes read
before cutting, and the test that a recording is finished). A single fed into
the intake by hand has none of that. When a recording is lost but its singles
survive, the recording is rebuilt instead, and goes into the recording intake
like any other.

It is exactly reversible here. A recording is a 1920x1080 frame laid out as a
4x2 grid of 480x540 cells with no overlap and no margin (cropper.CROP_COORDS),
and a single is one whole cell. Placing each single back in its cell -- black
for an empty position -- rebuilds the layout pixel for pixel, and the pipeline's
own cut then returns the same singles.

WHAT IT REFUSES
---------------
* Singles that do not line up in time (different frame counts or frame rates):
  they did not come from one cut of one recording, and stitching them would
  pair frames that were never simultaneous.
* A single that is not 480x540: it was cut some other way, and fitting it into
  a cell would change what the pose model sees.
* An output name that already exists: nothing is ever replaced.

WHAT IT RECORDS
---------------
The file says it is rebuilt (container metadata), and a record beside the
pipeline (``<nas_root>/pipeline_records/rebuilt_recordings/<name>.json``) names
every single used, its size and frame count, the grid, and why. A rebuilt
recording is never an original: it must never be filed as one.

Encoding: H.264 at CRF 10 (visually lossless), so the pipeline's own cut adds
the only loss that matters, as it does for any recording.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .cropper import CROP_COORDS, is_blank_animal, parse_collage_filename

CELL_W, CELL_H = 480, 540
REBUILD_CRF = 10
TOOL = "mousereach.video_prep.core.rebuild"


class RebuildRefused(Exception):
    """The singles cannot honestly be put back together; the message says why."""


def ffmpeg_exe() -> str:
    """ffmpeg on PATH, else the one the imageio-ffmpeg package ships."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:                                   # pragma: no cover
        raise RebuildRefused(f"no ffmpeg found: {e}")


def probe(path: Path) -> Dict:
    """Width, height, frame rate (OpenCV) and exact frame count (ffmpeg
    counting packets without decoding) of one video."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RebuildRefused(f"cannot open {Path(path).name}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path), "-map", "0:v:0",
                        "-c", "copy", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    frames = re.findall(r"frame=\s*(\d+)", r.stderr)
    if r.returncode != 0 or not frames:
        raise RebuildRefused(f"cannot count the frames of {Path(path).name}: "
                             f"{(r.stderr or '').strip()[-200:]}")
    return {"width": w, "height": h, "fps": round(fps, 3), "frames": int(frames[-1])}


def plan(recording_name: str, singles: Dict[str, Path]) -> Dict:
    """Check that ``singles`` ({animal_id: path}) can rebuild ``recording_name``.

    Every non-blank animal of the name needs exactly one single; all must be
    480x540 with the same frame count and rate. Returns the plan (positions,
    probes); raises RebuildRefused otherwise. Reads only.
    """
    info = parse_collage_filename(recording_name if recording_name.endswith(".mkv")
                                  else recording_name + ".mkv")
    cells: List[Optional[Path]] = []
    probes: Dict[str, Dict] = {}
    for aid in info["animal_ids"]:
        if is_blank_animal(aid):
            cells.append(None)
            continue
        p = singles.get(aid)
        if p is None or not Path(p).is_file():
            raise RebuildRefused(f"no single for {aid}")
        probes[aid] = probe(Path(p))
        cells.append(Path(p))
    if not probes:
        raise RebuildRefused("every position is empty")
    for aid, pr in probes.items():
        if (pr["width"], pr["height"]) != (CELL_W, CELL_H):
            raise RebuildRefused(f"{aid} is {pr['width']}x{pr['height']}, not "
                                 f"{CELL_W}x{CELL_H}: it was not cut from this grid")
    counts = {pr["frames"] for pr in probes.values()}
    rates = {pr["fps"] for pr in probes.values()}
    if len(counts) != 1 or len(rates) != 1:
        raise RebuildRefused("the singles do not line up in time (frame counts "
                             f"{sorted(counts)}, frame rates {sorted(rates)}): they "
                             f"are not one cut of one recording")
    return {"info": info, "cells": cells, "probes": probes,
            "frames": counts.pop(), "fps": rates.pop()}


def rebuild(recording_name: str, singles: Dict[str, Path], dest_dir: Path,
            records_dir: Optional[Path], reason: str) -> Path:
    """Build the recording into ``dest_dir`` and write its record. Returns the
    path. The file appears under its final name only when it is complete and
    its frame count checked (built in a temporary folder, copied in as
    ``<name>.part``, renamed)."""
    p = plan(recording_name, singles)
    name = recording_name if recording_name.endswith(".mkv") else recording_name + ".mkv"
    dest = Path(dest_dir) / name
    if dest.exists():
        raise RebuildRefused(f"{name} already exists in {dest_dir}")
    fps = p["fps"]
    duration = p["frames"] / fps
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    for cell in p["cells"]:
        if cell is None:
            cmd += ["-f", "lavfi", "-t", f"{duration:.6f}", "-i",
                    f"color=c=black:s={CELL_W}x{CELL_H}:r={fps}"]
        else:
            cmd += ["-i", str(cell)]
    layout = "|".join(f"{x}_{y}" for _, _, x, y in CROP_COORDS)
    streams = "".join(f"[{i}:v]" for i in range(8))
    cmd += ["-filter_complex", f"{streams}xstack=inputs=8:layout={layout}[v]",
            "-map", "[v]", "-frames:v", str(p["frames"]), "-r", str(fps),
            "-c:v", "libx264", "-crf", str(REBUILD_CRF), "-preset", "medium",
            "-pix_fmt", "yuv420p", "-an",
            "-metadata", f"comment=MouseReach: REBUILT from single-animal cuts "
                         f"({TOOL}); not an original recording"]
    with tempfile.TemporaryDirectory(prefix="rebuild_") as tmp:
        built = Path(tmp) / name
        r = subprocess.run(cmd + [str(built)], capture_output=True, text=True,
                           errors="replace")
        if r.returncode != 0 or not built.exists():
            raise RebuildRefused(f"ffmpeg failed: {(r.stderr or '').strip()[-300:]}")
        got = probe(built)
        if got["frames"] != p["frames"] or (got["width"], got["height"]) != (1920, 1080):
            raise RebuildRefused(f"rebuilt file is {got}, expected {p['frames']} "
                                 f"frames at 1920x1080")
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        shutil.copy2(built, part)
        if part.stat().st_size != built.stat().st_size:
            part.unlink()
            raise RebuildRefused("copy into the intake came out a different size")
        os.replace(part, dest)
    if records_dir is not None:
        Path(records_dir).mkdir(parents=True, exist_ok=True)
        rec = {
            "recording": name, "rebuilt_at": datetime.now().isoformat(timespec="seconds"),
            "tool": TOOL, "reason": reason,
            "grid": {"width": 1920, "height": 1080, "cell": [CELL_W, CELL_H],
                     "layout": CROP_COORDS},
            "frames": p["frames"], "fps": fps, "crf": REBUILD_CRF,
            "positions": [
                {"position": i + 1, "animal_id": aid,
                 "single": str(cell) if cell else None,
                 "single_bytes": cell.stat().st_size if cell else None,
                 "blank": cell is None}
                for i, (aid, cell) in enumerate(zip(p["info"]["animal_ids"], p["cells"]))],
            "note": "Not an original recording: assembled from per-animal cuts "
                    "because the recording itself does not exist.",
        }
        (Path(records_dir) / (Path(name).stem + ".json")).write_text(
            json.dumps(rec, indent=2), encoding="utf-8")
    return dest
