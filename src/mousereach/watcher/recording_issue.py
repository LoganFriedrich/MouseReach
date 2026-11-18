"""Recording issues: recordings that failed, recognized before anything cuts them.

WHAT A RECORDING ISSUE IS
-------------------------
A multi-animal recording (a "collage") that is not a usable recording at all:
the recording program stopped after a fraction of a second, the file is empty
or cannot be read as a video, it is far shorter than any real session, or its
start or end cannot be decoded. Cutting such a file gives eight useless
single-animal videos that land in a reviewer's queue, and nothing a reviewer
does can fix them: the behavior was never recorded.

THE CHECK (``assess``)
---------------------
Read the recording with ffprobe (or, where ffprobe is not installed, with
ffmpeg alone -- the copy on the PATH or the one the imageio-ffmpeg package
carries) and decide one of three things:

* ``ok``           -- a real recording; cut it as usual.
* ``issue``        -- a recording issue, with a reason in plain words, e.g.
                      "recording issue: 4 frames, 0.08 s -- the recording failed".
* ``cannot_check`` -- the check itself could not run (no ffmpeg on this
                      machine, the file could not be opened over the network,
                      the probe timed out). NOTHING is moved on this outcome:
                      a share blinking for a minute is not a failed recording.

A recording is an issue when ANY of these holds (each threshold is a setting
in the ``watcher`` section of ~/.mousereach/config.json):

* it cannot be read as a video (no duration, no video stream, ffmpeg refuses
  it -- an empty file is this case);
* it is shorter than ``recording_min_minutes`` (default 5 minutes);
* it holds fewer than ``recording_min_frames`` frames (default 3000; the
  container's own count when it has one, otherwise duration x frame rate);
* its size is implausibly small for its duration: under
  ``recording_min_kb_per_second`` kilobytes per second (default 5);
* its first frame, or a frame near its end, cannot be decoded.

WHY THESE DEFAULTS: a real session recording is many minutes long at a steady
frame rate, so five minutes and 3000 frames are far below anything real and
far above any recording that stopped by accident. A real recording of a
nearly empty scene is still tens of kilobytes per second; 5 is far below that.
A lab whose sessions are shorter lowers the settings.

WHERE A RECORDING ISSUE GOES (``set_aside``)
-------------------------------------------
Into ``<NAS root>/Processing/Review/_Problematic/`` -- the folder for cases
the pipeline cannot process at all -- with a reason file beside it:

    <collage name>.recording_issue.json

holding: the reason in plain words, the measurements, who or what flagged it
(the intake check on a named machine, or a person through
``mousereach-recording-issue``), when, where the file was, and the
single-animal sessions the recording was meant to hold. The reason file is
written FIRST and the recording moved second, so an interruption leaves a
reason file beside a recording still in the intake -- the next scan of any
watcher sees that and finishes the move (``finish_interrupted_moves``). The
recording is never cut, never reviewed and never deleted.

WHY THE REASON FILE IS THE SIGNAL TO EVERY MACHINE: each graphics-card PC
keeps its own record of every recording it has seen in the intake, and the
shared coordination database has no row for a recording until some machine
claims it. So the reason file on the shared drive is the one thing every
machine can see. Every watcher's scan (``apply_marks``) reads the reason
files and takes its own record of each named recording to the end state
``recording_issue``; nothing retries a recording in that state, and no machine
records the vanished file as a failed cut.

UNDOING ONE (a person decided it was not a failed recording)
------------------------------------------------------------
Move the recording back into Unanalyzed/Multi-Animal and move its
``.recording_issue.json`` file out of _Problematic (into an archive folder;
never delete records). The watcher then checks it again -- with the same
settings, so lower the setting that caught it first if the check was wrong.

All text written to a terminal or a log is ASCII.
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

REASON_SUFFIX = ".recording_issue.json"
STATE = "recording_issue"
KIND = "recording_issue"
FOLDER_NAME = "_Problematic"

# Collage states a reason file moves a watcher's own row out of. Every state
# that is not finished: a recording already cut ('cropped', 'archived') is
# left alone -- its singles exist and were handled on their own.
MARKABLE_STATES = ("discovered", "validated", "stable", "cropping", "failed", "quarantined")

DEFAULT_MIN_MINUTES = 5.0
DEFAULT_MIN_FRAMES = 3000
DEFAULT_MIN_KB_PER_SECOND = 5.0
PROBE_TIMEOUT_S = 120


def _ascii(text) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


# =============================================================================
# SETTINGS
# =============================================================================

@dataclass
class CheckSettings:
    """The thresholds, from the ``watcher`` section of the config file."""
    min_seconds: float = DEFAULT_MIN_MINUTES * 60.0
    min_frames: int = DEFAULT_MIN_FRAMES
    min_kb_per_second: float = DEFAULT_MIN_KB_PER_SECOND

    @classmethod
    def from_config(cls, cfg=None) -> "CheckSettings":
        """From a WatcherConfig (or anything with the same attributes); the
        machine's own config when ``cfg`` is None."""
        if cfg is None:
            try:
                from mousereach.config import WatcherConfig
                cfg = WatcherConfig.load()
            except Exception:
                cfg = None
        return cls(
            min_seconds=float(getattr(cfg, "recording_min_minutes", DEFAULT_MIN_MINUTES)) * 60.0,
            min_frames=int(getattr(cfg, "recording_min_frames", DEFAULT_MIN_FRAMES)),
            min_kb_per_second=float(getattr(cfg, "recording_min_kb_per_second",
                                            DEFAULT_MIN_KB_PER_SECOND)),
        )


# =============================================================================
# READING THE FILE
# =============================================================================

def find_tool(name: str) -> Optional[str]:
    """ffprobe or ffmpeg: on the PATH, else (ffmpeg only) imageio-ffmpeg's copy."""
    exe = shutil.which(name)
    if exe:
        return exe
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


class CannotCheck(Exception):
    """The check could not run (no tool, timeout, file not reachable)."""


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_FPS_RE = re.compile(r"Video:.*?,\s*([\d.]+)\s*fps")
_PROGRESS_TIME_RE = re.compile(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_PROGRESS_FRAME_RE = re.compile(r"frame=\s*(\d+)")
READ_TIMEOUT_S = 900


def measure_by_reading(path: Path, fps: Optional[float]) -> dict:
    """Duration and frame count found by reading the whole video stream
    (copied to nowhere, nothing decoded). WHY: a recording whose program was
    killed has a video stream but no duration in its header ("Duration:
    N/A"), and may still hold a whole session -- it must not be called a
    failed recording just because its header was never finished. Raises
    CannotCheck when the tool cannot run or reads nothing."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise CannotCheck("ffmpeg is not installed on this machine")
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(path),
                              "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=READ_TIMEOUT_S,
                             errors="replace")
    except subprocess.TimeoutExpired:
        raise CannotCheck("reading the whole file took longer than %d s" % READ_TIMEOUT_S)
    except OSError as e:
        raise CannotCheck("could not run ffmpeg: %s" % e)
    err = out.stderr or ""
    times = _PROGRESS_TIME_RE.findall(err)
    frames = _PROGRESS_FRAME_RE.findall(err)
    if not times:
        raise CannotCheck("the header has no duration and reading the file gave none")
    h, mi, sec = times[-1]
    duration = int(h) * 3600 + int(mi) * 60 + float(sec)
    n = int(frames[-1]) if frames else (int(round(duration * fps)) if fps else None)
    return {"readable": True, "duration": duration, "fps": fps, "frames": n,
            "frames_counted": bool(frames), "error": None,
            "duration_from": "reading the file (no duration in its header)"}


def _rate(text) -> Optional[float]:
    try:
        if "/" in str(text):
            a, b = str(text).split("/", 1)
            return float(a) / float(b) if float(b) else None
        v = float(text)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def probe(path: Path) -> dict:
    """What the file says about itself: {readable, duration, fps, frames,
    frames_counted, error}. ``readable`` False means the tool ran and the
    file is not a readable video. Raises CannotCheck when no tool could run."""
    path = Path(path)
    if not path.is_file():
        raise CannotCheck("the file is not there (moved or not reachable): %s" % path)
    ffprobe = find_tool("ffprobe")
    try:
        if ffprobe:
            out = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries",
                 "format=duration:stream=codec_type,avg_frame_rate,r_frame_rate,nb_frames",
                 "-of", "json", str(path)],
                capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, errors="replace")
            try:
                data = json.loads(out.stdout or "{}")
            except ValueError:
                data = {}
            video = [s for s in data.get("streams") or [] if s.get("codec_type") == "video"]
            duration = _rate((data.get("format") or {}).get("duration"))
            if video and duration is None:
                v = video[0]
                return measure_by_reading(
                    path, _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate")))
            if not video or duration is None:
                return {"readable": False, "duration": duration, "fps": None, "frames": None,
                        "frames_counted": False,
                        "error": _ascii((out.stderr or "").strip()[:200] or "no video stream")}
            v = video[0]
            fps = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))
            try:
                frames = int(v.get("nb_frames"))
                counted = True
            except (TypeError, ValueError):
                frames = int(round(duration * fps)) if fps else None
                counted = False
            return {"readable": True, "duration": duration, "fps": fps, "frames": frames,
                    "frames_counted": counted, "error": None}
        ffmpeg = find_tool("ffmpeg")
        if not ffmpeg:
            raise CannotCheck("neither ffprobe nor ffmpeg is installed on this machine")
        out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                             capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
                             errors="replace")
        err = out.stderr or ""
        m = _DURATION_RE.search(err)
        fm = _FPS_RE.search(err)
        has_video = "Video:" in err
        if has_video and not m:
            return measure_by_reading(path, _rate(fm.group(1)) if fm else None)
        if not m or not has_video:
            last = [ln.strip() for ln in err.splitlines() if ln.strip()]
            return {"readable": False, "duration": None, "fps": None, "frames": None,
                    "frames_counted": False,
                    "error": _ascii((last[-1] if last else "no video stream")[:200])}
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        fps = _rate(fm.group(1)) if fm else None
        frames = int(round(duration * fps)) if fps else None
        return {"readable": True, "duration": duration, "fps": fps, "frames": frames,
                "frames_counted": False, "error": None}
    except subprocess.TimeoutExpired:
        raise CannotCheck("reading the file took longer than %d s" % PROBE_TIMEOUT_S)
    except OSError as e:
        raise CannotCheck("could not run the video tool: %s" % e)


def decodes(path: Path, at_end: bool) -> Tuple[bool, str]:
    """(True, "") when one frame at the start (or near the end) decodes;
    (False, why) when it does not. Raises CannotCheck when no tool can run."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise CannotCheck("ffmpeg is not installed on this machine")
    cmd = [ffmpeg, "-hide_banner", "-v", "error"]
    if at_end:
        cmd += ["-sseof", "-3"]
    cmd += ["-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-f", "framemd5", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
                             errors="replace")
    except subprocess.TimeoutExpired:
        raise CannotCheck("decoding a frame took longer than %d s" % PROBE_TIMEOUT_S)
    except OSError as e:
        raise CannotCheck("could not run ffmpeg: %s" % e)
    frames = [ln for ln in (out.stdout or "").splitlines() if ln and not ln.startswith("#")]
    if out.returncode == 0 and frames:
        return True, ""
    said = " | ".join(ln.strip() for ln in (out.stderr or "").splitlines() if ln.strip())
    return False, _ascii(said[-200:] or "no frame could be decoded")


# =============================================================================
# THE VERDICT
# =============================================================================

@dataclass
class Verdict:
    outcome: str                      # 'ok' | 'issue' | 'cannot_check'
    reason: str = ""                  # plain words; "" when ok
    measurements: dict = field(default_factory=dict)

    @property
    def is_issue(self) -> bool:
        return self.outcome == "issue"


def _fmt_seconds(s: float) -> str:
    return ("%.2f s" % s) if s < 10 else ("%.0f s" % s if s < 120 else "%.1f min" % (s / 60.0))


def assess(path: Path, settings: Optional[CheckSettings] = None,
           probe_fn: Callable[[Path], dict] = probe,
           decode_fn: Callable[[Path, bool], Tuple[bool, str]] = decodes) -> Verdict:
    """Decide whether ``path`` is a usable recording. Never raises."""
    settings = settings or CheckSettings.from_config()
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as e:
        return Verdict("cannot_check", _ascii("the file cannot be reached: %s" % e))
    m = {"bytes": size, "min_seconds": settings.min_seconds,
         "min_frames": settings.min_frames,
         "min_kb_per_second": settings.min_kb_per_second}
    try:
        info = probe_fn(path)
    except CannotCheck as e:
        return Verdict("cannot_check", _ascii(str(e)), m)
    except Exception as e:                                   # pragma: no cover
        return Verdict("cannot_check", _ascii("%s: %s" % (type(e).__name__, e)), m)
    m.update({k: info.get(k) for k in ("duration", "fps", "frames", "frames_counted")})
    if info.get("duration_from"):
        m["duration_from"] = info["duration_from"]
    if not info.get("readable"):
        return Verdict("issue", _ascii(
            "recording issue: the file cannot be read as a video (%d bytes; %s) -- "
            "the recording failed" % (size, info.get("error") or "no video stream")), m)
    duration = float(info.get("duration") or 0.0)
    frames = info.get("frames")
    short = []
    if frames is not None:
        short.append("%d frames" % frames)
    short.append(_fmt_seconds(duration))
    if duration < settings.min_seconds:
        return Verdict("issue", _ascii(
            "recording issue: %s -- the recording failed (shorter than the %g-minute "
            "minimum, setting recording_min_minutes)"
            % (", ".join(short), settings.min_seconds / 60.0)), m)
    if frames is not None and frames < settings.min_frames:
        return Verdict("issue", _ascii(
            "recording issue: %s -- the recording failed (fewer than the %d-frame "
            "minimum, setting recording_min_frames)"
            % (", ".join(short), settings.min_frames)), m)
    kbps = (size / 1000.0) / duration if duration > 0 else 0.0
    m["kb_per_second"] = round(kbps, 2)
    if kbps < settings.min_kb_per_second:
        return Verdict("issue", _ascii(
            "recording issue: only %d KB for %s (%.1f KB per second; the minimum is %g, "
            "setting recording_min_kb_per_second) -- the file is too small to hold the "
            "recording" % (size // 1000, _fmt_seconds(duration), kbps,
                           settings.min_kb_per_second)), m)
    for at_end, where in ((False, "start"), (True, "end")):
        try:
            ok, why = decode_fn(path, at_end)
        except CannotCheck as e:
            return Verdict("cannot_check", _ascii(str(e)), m)
        m["decodes_at_" + where] = ok
        if not ok:
            return Verdict("issue", _ascii(
                "recording issue: the %s of the recording cannot be decoded (%s) -- the "
                "recording is broken" % (where, why)), m)
    return Verdict("ok", "", m)


class RecordingIssueError(ValueError):
    """Raised by ``refuse_if_issue``: the file is a recording issue."""


def refuse_if_issue(path: Path, settings: Optional[CheckSettings] = None) -> Verdict:
    """For every program that cuts a recording by hand (the prep window, the
    cropping command): raise RecordingIssueError instead of cutting a failed
    recording. A check that could not run does not stop the cut."""
    v = assess(path, settings)
    if v.is_issue:
        raise RecordingIssueError(
            "%s: %s. Not cut. To set it aside: mousereach-recording-issue \"%s\" "
            "--reason \"<why>\"" % (Path(path).name, v.reason, Path(path).name))
    return v


# =============================================================================
# THE FOLDER AND THE REASON FILES
# =============================================================================

def problematic_dir(nas_root=None) -> Optional[Path]:
    """<NAS root>/Processing/Review/_Problematic, or None when unconfigured."""
    if nas_root is None:
        from mousereach.config import Paths
        nas_root = Paths.NAS_ROOT
    return Path(nas_root) / "Processing" / "Review" / FOLDER_NAME if nas_root else None


def reason_path(folder: Path, collage_name: str) -> Path:
    return Path(folder) / (collage_name + REASON_SUFFIX)


def expected_sessions(collage_name: str) -> List[str]:
    """The single-animal sessions this recording was meant to hold (blank
    boxes left out), e.g. ['20250101_ABC0101_P2', ...]. [] when the name
    cannot be read."""
    try:
        from mousereach.video_prep.core.collage_provenance import expected_offspring
        return [o["offspring_stem"] for o in expected_offspring(collage_name)
                if o.get("offspring_stem") and not o.get("blank")]
    except Exception:
        return []


def read_marks(nas_root=None) -> Dict[str, dict]:
    """{collage name: reason-file contents} for every reason file in the
    folder. A file that cannot be read is listed with 'unreadable' set, so
    it is shown rather than silently skipped."""
    folder = problematic_dir(nas_root)
    out: Dict[str, dict] = {}
    if folder is None or not folder.is_dir():
        return out
    try:
        names = [e.name for e in os.scandir(folder) if e.name.endswith(REASON_SUFFIX)]
    except OSError:
        return out
    for name in sorted(names):
        collage = name[: -len(REASON_SUFFIX)]
        try:
            data = json.loads((folder / name).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
        except Exception as e:
            data = {"collage": collage, "unreadable": _ascii(str(e)),
                    "reason": "reason file could not be read (%s)" % _ascii(e)}
        data.setdefault("collage", collage)
        out[collage] = data
    return out


def _flagged_by_machine() -> str:
    return "intake check on %s" % socket.gethostname()


def set_aside(collage_path: Path, reason: str, flagged_by: str,
              measurements: Optional[dict] = None, nas_root=None) -> Path:
    """Write the reason file, then move the recording into _Problematic.
    Returns the recording's new path. Raises (nothing moved) when a different
    file of the same name is already there -- a person decides; nothing is
    renamed or overwritten."""
    collage_path = Path(collage_path)
    folder = problematic_dir(nas_root)
    if folder is None:
        raise RuntimeError("the shared pipeline folder (nas_root) is not configured")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / collage_path.name
    if dest.exists():
        raise FileExistsError(
            "a file named %s is already in %s; nothing was moved -- compare the two "
            "and keep the right one" % (collage_path.name, folder))
    record = {
        "kind": KIND,
        "collage": collage_path.name,
        "reason": _ascii(reason),
        "says": _ascii("recording failed: %s, %s" % (collage_path.name, reason)),
        "measurements": measurements or {},
        "flagged_by": _ascii(flagged_by),
        "flagged_at": datetime.now().isoformat(timespec="seconds"),
        "original_path": str(collage_path),
        "sessions": expected_sessions(collage_path.name),
        "what_it_means": ("This recording is not a usable recording, so it was never cut "
                          "into single-animal videos and is not reviewed. Its sessions "
                          "have hand scores (if they were scored) but no kinematics."),
    }
    from mousereach.pipeline.fsutil import write_json_atomic
    write_json_atomic(reason_path(folder, collage_path.name), record, indent=1)
    shutil.move(str(collage_path), str(dest))
    logger.warning("Recording issue: %s set aside in %s -- %s",
                   collage_path.name, folder, _ascii(reason))
    return dest


def finish_interrupted_moves(intake: Path, nas_root=None, marks=None) -> List[str]:
    """A reason file whose recording is still in the intake is a move that was
    interrupted: finish it. Returns the names moved. Never raises."""
    moved = []
    folder = problematic_dir(nas_root)
    if folder is None or intake is None:
        return moved
    marks = read_marks(nas_root) if marks is None else marks
    for name in marks:
        src = Path(intake) / name
        dest = folder / name
        try:
            if src.is_file() and not dest.exists():
                shutil.move(str(src), str(dest))
                moved.append(name)
                logger.warning("Recording issue: finished setting aside %s", name)
        except Exception as e:
            logger.warning("Recording issue: could not finish setting aside %s: %s",
                           name, _ascii(e))
    return moved


def mark_row(db, collage_name: str, why: str) -> bool:
    """Take this machine's own row for the recording to 'recording_issue'.
    True when a row changed. A finished row ('cropped', 'archived') or one
    already marked is left alone. Never raises."""
    try:
        row = db.get_collage(collage_name)
        if not row or row.get("state") not in MARKABLE_STATES:
            return False
        db.update_collage_state(collage_name, STATE, validation_error=_ascii(why)[:500])
        try:
            db.log_step(collage_name, "recording_check", "completed",
                        message=_ascii("recording issue: %s" % why)[:500])
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning("Recording issue: could not record %s here: %s", collage_name, _ascii(e))
        return False


def apply_marks(db, nas_root=None, intake: Optional[Path] = None) -> List[str]:
    """Once per scan, on every watcher: finish interrupted moves, then take
    every local row named by a reason file to 'recording_issue'. Returns the
    names whose row changed. Never raises."""
    try:
        marks = read_marks(nas_root)
    except Exception:
        return []
    if not marks:
        return []
    if intake is not None:
        finish_interrupted_moves(intake, nas_root, marks)
    changed = []
    for name, data in marks.items():
        if mark_row(db, name, data.get("reason") or "set aside as a recording issue"):
            changed.append(name)
    if changed:
        logger.info("Recording issues recorded here: %s", ", ".join(changed))
    return changed


def attention_items(nas_root=None) -> List[dict]:
    """One item per recording set aside, in the shape of the watchers'
    attention lists (watcher/attention.py). ``needs_person`` False: the
    recording needs nothing more from the pipeline; the lab acknowledges it
    in its database tool, which cannot write back here. A reason file that
    cannot be read IS needs_person."""
    out = []
    for name, d in read_marks(nas_root).items():
        out.append({"kind": KIND, "needs_person": bool(d.get("unreadable")),
                    "collage": name, "video_id": None, "cohort_id": None,
                    "flagged_at": d.get("flagged_at"),
                    "says": _ascii(d.get("says") or "recording failed: %s, %s"
                                   % (name, d.get("reason")))})
    return out


# =============================================================================
# COMMAND LINE: mousereach-recording-issue
# =============================================================================

def _find_in_intake(name: str) -> Optional[Path]:
    from mousereach.config import Paths
    p = Path(name)
    if p.is_file():
        return p
    if Paths.MULTI_ANIMAL_SOURCE:
        cand = Path(Paths.MULTI_ANIMAL_SOURCE) / p.name
        if cand.is_file():
            return cand
    return None


def _claim_refusal(name: str) -> Optional[str]:
    """Why a person may not set this recording aside now, from the shared
    claim; None when nothing stands in the way. An unreadable coordination
    database does not refuse (it may not exist on this machine)."""
    try:
        from mousereach.watcher.coordination import PipelineCoordinator
        coord = PipelineCoordinator()
        claim = coord.get_collage_claim(name) or {}
    except Exception:
        return None
    state = claim.get("state")
    if state in ("cropped", "archived"):
        return ("%s was already cut into single-animal videos by %s; its singles are "
                "handled on their own. Nothing was moved." % (name, claim.get("hostname")))
    if state == "cropping" and claim.get("hostname") != socket.gethostname():
        return ("%s is being cut by %s right now. Wait for it to finish, then deal with "
                "the singles. Nothing was moved." % (name, claim.get("hostname")))
    return None


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="mousereach-recording-issue",
        description=("Set a multi-animal recording aside as a RECORDING ISSUE: a recording "
                     "that failed (empty, a fraction of a second, unreadable). It is moved to "
                     "Processing/Review/_Problematic with a reason file, never cut, never "
                     "reviewed, never deleted, and every watcher stops trying to cut it."))
    p.add_argument("file", nargs="?", help="the recording's file name (in the intake "
                                            "Unanalyzed/Multi-Animal) or full path")
    p.add_argument("--reason", help="why, in plain words (required to set one aside)")
    p.add_argument("--by", help="who is setting it aside (default: your login name)")
    p.add_argument("--check", action="store_true",
                   help="only run the automatic check on FILE and say what it finds; move nothing")
    p.add_argument("--list", action="store_true", help="list every recording set aside")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.list:
        marks = read_marks()
        if not marks:
            print("No recordings are set aside as recording issues.")
        for name, d in marks.items():
            print("%s\n    %s\n    flagged by %s at %s" % (
                _ascii(name), _ascii(d.get("reason")), _ascii(d.get("flagged_by")),
                d.get("flagged_at")))
        return 0
    if not a.file:
        p.print_usage()
        return 2
    path = _find_in_intake(a.file)
    if path is None:
        print("[FAIL] %s is not in the intake (Unanalyzed/Multi-Animal) and is not a file "
              "path. Nothing was moved." % _ascii(a.file))
        return 1
    if a.check:
        v = assess(path)
        print("%s: %s%s" % (_ascii(path.name), v.outcome,
                            (" -- " + v.reason) if v.reason else ""))
        print("  measurements: %s" % json.dumps(v.measurements, sort_keys=True))
        return 0
    if not (a.reason or "").strip():
        print("[FAIL] say why with --reason \"...\". Nothing was moved.")
        return 1
    why = _claim_refusal(path.name)
    if why:
        print("[FAIL] " + _ascii(why))
        return 1
    who = (a.by or "").strip() or getpass.getuser()
    v = assess(path)
    try:
        dest = set_aside(path, a.reason.strip(), "person: %s (mousereach-recording-issue)" % who,
                         dict(v.measurements, automatic_check=v.outcome,
                              automatic_reason=v.reason))
    except Exception as e:
        print("[FAIL] %s" % _ascii(e))
        return 1
    # This machine's own record, at once; every other watcher reads the
    # reason file on its next scan.
    try:
        from mousereach.watcher.cli import _resolve_db_path
        from mousereach.watcher.db import WatcherDB
        dbp = _resolve_db_path()
        if dbp.exists():
            mark_row(WatcherDB(dbp), path.name, a.reason.strip())
    except Exception:
        pass
    print("[OK] %s set aside as a recording issue: %s" % (_ascii(path.name), _ascii(dest)))
    print("     Every watcher stops trying to cut it at its next scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
