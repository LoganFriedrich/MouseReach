"""Start a video completely over: cut it again out of its multi-animal
recording, pose it again, and run every later step again.

THE RULE (lab decisions 2026-11-19, D2-10 / D2-7 / D2-11 and their two
corrections)
------------------------------------------------------------------------
Two kinds of critical quality finding are not questions a person can answer in
a review window:

  1. the pose cannot be used -- it will not load, or the points every tray
     needs (SABL/SABR among them) are missing (pipeline/pose_check.py; the
     quality triage's POSE_UNUSABLE flags);
  2. a step's output is missing or cannot be read (the quality triage's
     "Missing <step> file" / "Could not read segments file").

For either, the video is started completely over: the single-animal video is
CUT AGAIN from its multi-animal recording (the collage), posed again on a
graphics-card PC, and every later step runs again. Everything MouseReach itself
made for the failed attempt -- the single-animal video, its pose, every step
output -- is DELETED ("No garbage sits around gathering dust"). Never deleted:
the multi-animal recording, and anything a person made (a review, ground truth,
hand-set cuts, a clear marker -- review_routing.is_person_file decides).

Restarts are unlimited, with three exceptions, each listed red for a person on
this machine's attention list ``<NAS>/attention/<host>_restart_stopped.json``:
  * the SAME error has ended 5 restarts in a row (MAX_SAME_ERROR): restarting
    stops, the video is marked failed with that error;
  * the multi-animal recording cannot be read (the cut fails): a recording
    problem, at once -- no restart can fix it;
  * the multi-animal recording cannot be found: listed red (recordings are
    never deleted, so this needs a person).
An empty box (a blank position in the recording) is intentional and never
counted as a problem: a blank position has no single-animal video at all.

The other two critical findings -- an unexpected number of boundaries, and an
uncertain or low-confidence outcome -- are NOT handled here: a person can
answer them, and they keep holding the video in review as before.

HOW THE RESTART RUNS (all on the processing server)
----------------------------------------------------
  1. the restart record ``<NAS>/Processing/Restarts/<video>.json`` gets one
     line per restart (time, machine, error); it is what the 5-in-a-row rule
     counts, and it survives watcher restarts and machine changes;
  2. the source recording is found (quarantine_purge.CollageIndex: the intake
     folder Unanalyzed/Multi-Animal, any Multi-Animal folder in Analyzed or
     Archive, the watcher database's collage rows);
  3. the failed attempt's machine files are deleted from this machine's
     processing folder and from the shared posed folder; the posed claim is
     let go;
  4. the single is cut again (ffmpeg, the position of the animal in the
     recording's name, the same crop as the graphics-card PCs use) into the
     processing folder, and copied to
     ``<NAS>/Processing/Repose_Queue/_parked/<video>/`` where a graphics-card
     PC may read it;
  5. a pose request naming that copy goes into the shared Repose_Queue, and
     the row is parked 'outdated' / scope 'full' with mark_reason
     "restart from re-crop: <error>". From there the existing re-pose round
     trip (watcher/repose.py) takes over: a graphics-card PC poses it and
     stages the pose; the server adopts it and runs every step after posing.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_SAME_ERROR = 5
RESTART_PREFIX = "restart from re-crop"
STOPPED_PREFIX = "restart stopped"
RECORDING_PROBLEM_PREFIX = "recording problem"
COLLAGE_MISSING_PREFIX = "source recording missing"
ATTENTION_KIND = "restart_stopped"
RESTARTS_DIR = ("Processing", "Restarts")

# The two finding kinds (module docstring). The words go into the restart
# record and the row, so a person reading either sees which it was.
POSE_UNUSABLE = "pose unusable"
OUTPUT_MISSING = "step output missing or unreadable"

# The quality triage's critical descriptions of kind 2 (pipeline/triage.py).
_OUTPUT_MISSING_PATTERNS = (re.compile(r"^Missing \w+ file: "),
                            re.compile(r"^Could not read segments file: "))


def is_output_missing_finding(description: str) -> bool:
    """True for a quality-triage critical finding of kind 2: a step output
    missing or unreadable."""
    return any(p.search(description or "") for p in _OUTPUT_MISSING_PATTERNS)


def same_error_key(error: str) -> str:
    """The error with numbers and paths blanked, so 'the same error' is not
    told apart by a frame count, a time or a folder name."""
    e = re.sub(r"[A-Za-z]:[\\/][^\s,;)]*|/[^\s,;)]+", "<path>", str(error or ""))
    e = re.sub(r"\d+(\.\d+)?", "#", e)
    return " ".join(e.split()).lower()


# --------------------------------------------------------------- the record

def record_path(nas_root, video_id: str) -> Path:
    return Path(nas_root).joinpath(*RESTARTS_DIR) / f"{video_id}.json"


def read_history(nas_root, video_id: str) -> List[dict]:
    p = record_path(nas_root, video_id)
    try:
        return list(json.loads(p.read_text(encoding="utf-8")).get("restarts") or [])
    except FileNotFoundError:
        return []
    except Exception as e:
        # Unreadable: counted as no history (a restart is never refused for a
        # record that cannot be read), and said.
        logger.warning(f"{video_id}: restart record {p} could not be read ({e})")
        return []


def same_error_run(history: List[dict], error: str) -> int:
    """How many restarts at the END of ``history`` ended in ``error``."""
    key = same_error_key(error)
    n = 0
    for entry in reversed(history):
        if same_error_key(entry.get("error", "")) != key:
            break
        n += 1
    return n


def append_history(nas_root, video_id: str, error: str, outcome: str) -> None:
    from mousereach.pipeline.fsutil import write_json_atomic
    p = record_path(nas_root, video_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    hist = read_history(nas_root, video_id)
    hist.append({"at": datetime.now().isoformat(timespec="seconds"),
                 "host": socket.gethostname(), "error": error, "outcome": outcome})
    write_json_atomic(p, {"video_id": video_id, "restarts": hist}, indent=1)


# --------------------------------------------------------------- the recording

def find_source_recording(video_id: str, nas_root, db=None) -> Optional[Path]:
    """The multi-animal recording this single was cut from, or None."""
    from mousereach.config import Paths
    from mousereach.watcher.quarantine_purge import CollageIndex
    parsed = parse_single(video_id)
    if parsed is None:
        return None
    quarantine = getattr(Paths, "QUARANTINE", None) or Path(nas_root) / "__no_quarantine__"
    hits = CollageIndex(Path(nas_root), Path(quarantine), db=db).source_of(parsed)
    hits = [h for h in hits if position_in(h.name, video_id) is not None]
    return hits[0] if hits else None


def parse_single(video_id: str) -> Optional[dict]:
    """{date, animal_id, tray_type, tray_run} of a single's name, or None."""
    m = re.match(r"^(\d{8})_([A-Za-z]+\d+)_([A-Za-z])(\d+)", video_id)
    if not m:
        return None
    return {"date": m.group(1), "animal_id": m.group(2),
            "tray_type": m.group(3), "tray_run": m.group(4)}


def position_in(collage_name: str, video_id: str) -> Optional[int]:
    """The single's position (1-8) in the recording, or None when the
    recording does not hold it (wrong date or tray, or a blank position)."""
    from mousereach.video_prep.core.cropper import is_blank_animal, parse_collage_filename
    try:
        info = parse_collage_filename(collage_name)
    except ValueError:
        return None
    for i, animal in enumerate(info["animal_ids"]):
        if is_blank_animal(animal):
            continue                        # an empty box is never a single
        if f"{info['date']}_{animal}_{info['last_part']}" == video_id:
            return i + 1
    return None


class RecordingUnreadable(RuntimeError):
    """The multi-animal recording could not be read or cut."""


def cut_single(collage: Path, video_id: str, dest_dir: Path) -> Path:
    """Cut ``video_id`` out of ``collage`` into ``dest_dir/<video_id>.mp4``,
    exactly as the graphics-card PCs cut it (cropper.CROP_COORDS, ffmpeg crop,
    audio copied). Written under a temporary name and renamed when ffmpeg has
    finished. Raises RecordingUnreadable when the recording cannot be opened
    or ffmpeg cannot cut it."""
    from mousereach.video_prep.core.cropper import CROP_COORDS
    from mousereach.video_prep.core.rebuild import ffmpeg_exe
    pos = position_in(collage.name, video_id)
    if pos is None:
        raise RecordingUnreadable(f"{collage.name} does not hold {video_id}")
    try:
        with open(collage, "rb") as fh:
            fh.read(1)
    except OSError as e:
        raise RecordingUnreadable(f"{collage} cannot be opened ({e})")
    w, h, x, y = CROP_COORDS[pos - 1]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{video_id}.mp4"
    part = dest_dir / f"{video_id}.recrop.part.mp4"
    proc = subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", str(collage),
                           "-filter:v", f"crop={w}:{h}:{x}:{y}", "-c:a", "copy", str(part)],
                          capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not part.is_file() or part.stat().st_size == 0:
        try:
            part.unlink()
        except OSError:
            pass
        said = " | ".join(l.strip() for l in (proc.stderr or "").splitlines() if l.strip())
        raise RecordingUnreadable(f"ffmpeg could not cut {video_id} out of "
                                  f"{collage.name} (exit {proc.returncode}): {said[-300:]}")
    part.replace(out)
    return out


# --------------------------------------------------------------- the deletion

def machine_files(video_id: str, folders) -> List[Path]:
    """Every file of ``video_id`` in ``folders`` that MouseReach made: the
    single-animal video, its pose, and the steps' outputs -- never a file a
    person made (review_routing.is_person_file reads hand-set cuts and marks
    inside the step files too)."""
    from mousereach.pipeline.fsutil import files_of_video
    from mousereach.watcher.review_routing import (is_input_file, is_machine_output,
                                                   is_person_file)
    out: List[Path] = []
    for folder in folders:
        if not folder or not Path(folder).is_dir():
            continue
        for f in files_of_video(Path(folder), video_id):
            name = f.name
            if is_input_file(video_id, name):
                out.append(f)
            elif is_machine_output(video_id, name) and not is_person_file(f, video_id):
                out.append(f)
    return out


# --------------------------------------------------------------- the plan

def plan(video_id: str, error: str, *, nas_root, db=None,
         find=find_source_recording) -> Dict:
    """What a restart of ``video_id`` for ``error`` would do, read-only:
    {"decision": "restart" | "stopped" | "recording_missing", "collage",
    "same_error_run", "says"}. (A recording that cannot be read is found only
    when the cut is tried: execute says so.)"""
    hist = read_history(nas_root, video_id)
    run = same_error_run(hist, error)
    if run >= MAX_SAME_ERROR:
        return {"decision": "stopped", "collage": None, "same_error_run": run,
                "says": (f"{video_id}: {STOPPED_PREFIX} after {run} restarts that all "
                         f"ended in the same error: {error}. Restarting it again "
                         f"would only repeat it -- a person must find why.")}
    collage = find(video_id, nas_root, db)
    if collage is None:
        return {"decision": "recording_missing", "collage": None, "same_error_run": run,
                "says": (f"{video_id}: {COLLAGE_MISSING_PREFIX} -- it must be cut again "
                         f"({error}) but its multi-animal recording was not found in "
                         f"Unanalyzed/Multi-Animal or any Multi-Animal folder. "
                         f"Recordings are never deleted: find it and put it back.")}
    return {"decision": "restart", "collage": collage, "same_error_run": run,
            "says": f"{video_id}: {RESTART_PREFIX} ({error}) from {collage.name}"}
