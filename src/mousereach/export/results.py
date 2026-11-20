"""Export the FILED results at the level a person chooses: reach, pellet
(segment), session, day or animal. ``mousereach-results`` and the GUI's
"Export results" tab both call ``export_results`` -- one implementation.

WHAT IT READS
-------------
Every ``<video>_features.json`` under the filed-results folder
(``<pipeline folder>/Analyzed/<project>/<cohort>/``), walked with
``analyzed_tree.iter_files``, which skips ``Analyzed/Archive`` (superseded
generations). A features file is a video's resolved result: the algorithm's
calls with every person's answer applied (review, ground truth, ignore
windows). Nothing else is read -- no watcher, no hand-score sheet, no
database -- so a lab that only drops videos in and presses Run gets its
results from MouseReach alone.

WHY THIS EXISTS
---------------
Every earlier export read ONE flat folder, the local Processing folder by
default; filing moves a video's results out of it into Analyzed, so after
filing they found nothing. None gave a session or animal level built from
pellets: mousereach-build-database's Sessions sheet computed its
"retrieval_rate" as the share of REACH rows whose pellet was retrieved, and a
pellet nobody reached for dropped out of every count.

THE LEVELS (what one row is)
----------------------------
  reach    one reach. Carries its own results (reach_*), its pellet's
           (segment_*), its tray's (session_*), its animal-day's (day_*) and
           its animal's (animal_*) -- every level above it, repeated.
  pellet   one pellet = one segment of a tray (segment_*), with session_*,
  (segment) day_* and animal_* repeated.
  session  one tray = one video (session_*), with day_* and animal_*.
  day      one animal on one date, all its trays (day_*), with animal_*.
  animal   one animal, everything filed for it (animal_*).

COLUMN NAMES are level-first (reach_, segment_, session_, day_, animal_) so a
repeated higher-level value is never mistaken for the row's own. Identity
columns (animal_id, date, session_id, tray_type, tray_run, segment_number,
reach_number, reach_id) keep plain names. ``column_meanings()`` says what each
column means; the xlsx file carries it as a second sheet, "columns".

HOW THE PELLET NUMBERS ARE COUNTED
----------------------------------
A pellet's outcome is its segment's outcome in the features file: retrieved,
displaced_sa (displaced, stayed in the scoring area), displaced_outside,
untouched ('missed' in older files), abnormal / abnormal_exception (a person
saw something other than a reach move it: counted as presented, in its own
column moved_not_by_reach, never as retrieved or displaced), no_pellet
(nothing was presented in that segment), or anything else, counted as
undetermined (a filed video should have none: filing refuses an undecided
pellet).

  presented     = pellets whose outcome is not no_pellet
  success rate  = retrieved / presented
  contact rate  = (retrieved + displaced_sa + displaced_outside) / presented

A pellet with NO reach still counts as presented. Session, day and animal are
computed from pellets, never from reach rows. Easy and Flat trays (E, F) have
no pellet outcomes (their videos are not segmented): they add reaches to the
session/day/animal reach counts and nothing to the pellet counts; their
session rows leave the pellet columns empty.

A REACH'S OWN OUTCOME (reach_outcome) is not its pellet's. Only the reach a
person or algorithm 4 credited with the pellet (causal_reach) carries the
pellet's outcome (retrieved / displaced_sa / displaced_outside). Every other
reach at that pellet is "missed", and so is every reach at an untouched
pellet or at a pellet moved by something other than a reach. A pellet that moved with no credited reach gives its reaches
"undetermined" -- never a guess. segment_outcome is the pellet's result and
is the same on every reach row of the segment.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

LEVELS = ("reach", "pellet", "session", "day", "animal")
# "segment" is the same level as "pellet" (one segment holds one pellet).
LEVEL_ALIASES = {"segment": "pellet", "tray": "session", "video": "session",
                 "mouse": "animal", "reaches": "reach", "pellets": "pellet",
                 "sessions": "session", "days": "day", "animals": "animal"}
FORMATS = ("csv", "xlsx")

MOVED = ("retrieved", "displaced_sa", "displaced_outside")
# 'missed' is an older word for untouched. 'abnormal' / 'abnormal_exception':
# a person saw something other than a reach move the pellet (the review tool's
# "abnormal", or an ignore window with no reach credited) -- a decided answer
# about the pellet, not a gap. No reach did it, so every reach there missed.
UNTOUCHED_WORDS = ("untouched", "missed")
NOT_BY_A_REACH = ("abnormal", "abnormal_exception")
KNOWN_OUTCOMES = MOVED + UNTOUCHED_WORDS + NOT_BY_A_REACH + ("no_pellet",)

# Reach kinematics averaged per session / day / animal (over every measured
# reach). The per-reach values are in the reach file.
MEAN_KINEMATICS = (
    "max_extent_mm", "duration_frames", "peak_velocity_px_per_frame",
    "velocity_at_apex_mm_per_sec", "trajectory_straightness",
)

_VIDEO_RE = re.compile(r"^(\d{8})_(.+?)_([A-Za-z])(\d+)$")


def normalize_level(level: str) -> str:
    """The level name, or ValueError naming the choices."""
    lv = (level or "").strip().lower()
    lv = LEVEL_ALIASES.get(lv, lv)
    if lv not in LEVELS:
        raise ValueError(f"unknown level '{level}': choose one of "
                         f"{', '.join(LEVELS)} (segment = pellet)")
    return lv


def parse_video_name(video: str) -> Dict[str, Optional[str]]:
    """Identity of a single-animal video named YYYYMMDD_<animal>_<tray><run>.

    A name that does not follow the rule keeps its whole name as the session
    and an empty animal/date (it is exported, never dropped)."""
    m = _VIDEO_RE.match(video or "")
    if not m:
        return {"animal_id": None, "date": None, "tray_type": None, "tray_run": None}
    d, animal, tray, run = m.groups()
    return {"animal_id": animal, "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "tray_type": tray.upper(), "tray_run": int(run)}


def _lab_ids(video: str) -> Tuple[Optional[str], Optional[str]]:
    """(lab animal id, lab session name) where a project re-encodes its ids on
    the way in (mousereach.aspa); (None, None) otherwise. The encoded id stays
    the key; these are added for a human reader."""
    try:
        from mousereach.aspa import lab_animal_id, decode_video_stem
        a, s = lab_animal_id(video), decode_video_stem(video)
        return (a, s) if a else (None, None)
    except Exception:
        return None, None


def reach_outcome(seg_outcome: Optional[str], is_causal: bool, has_causal: bool) -> str:
    """What THIS reach did (module docstring, "A REACH'S OWN OUTCOME")."""
    if is_causal and seg_outcome in MOVED:
        return seg_outcome
    if seg_outcome in MOVED and not has_causal:
        return "undetermined"
    if seg_outcome in MOVED or seg_outcome in UNTOUCHED_WORDS + NOT_BY_A_REACH + ("no_pellet",):
        return "missed"
    return "undetermined"


# ---------------------------------------------------------------- reading

def find_features(root) -> List[Path]:
    """Every current features file under ``root`` (Analyzed/Archive skipped)."""
    from mousereach.pipeline.analyzed_tree import iter_files
    return sorted(iter_files(Path(root), "*_features.json"), key=lambda p: p.name)


def load_videos(root) -> Tuple[List[Dict], List[str]]:
    """``(videos, problems)``: each readable features file, parsed; the names
    of the ones that could not be read, with why."""
    videos, problems = [], []
    seen = {}
    for fp in find_features(root):
        stem = fp.name[: -len("_features.json")]
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            problems.append(f"{stem}: could not be read ({e})")
            continue
        if stem in seen:
            # Two filed copies of one video (two project folders): both are
            # results somebody filed; exported once, and said.
            problems.append(f"{stem}: filed twice ({seen[stem]} and {fp.parent}); "
                            f"the first is exported")
            continue
        seen[stem] = fp.parent
        doc.setdefault("video_name", stem)
        doc["_source"] = str(fp)
        videos.append(doc)
    return videos, problems


# ------------------------------------------------------------ the pellets

def _session_only(doc: Dict) -> bool:
    from mousereach.config import is_outcome_free_tray
    return (bool((doc.get("summary") or {}).get("session_only"))
            or is_outcome_free_tray(doc.get("video_name") or ""))


def _new_tally() -> Dict:
    t = {"pellets_presented": 0, "no_pellet": 0, "undetermined": 0, "moved_not_by_reach": 0,
         "n_reaches": 0, "n_reaches_left_out": 0, "n_causal_reaches": 0,
         "n_sessions": 0, "_dates": set()}
    for o in MOVED + ("untouched",):
        t[o] = 0
    for k in MEAN_KINEMATICS:
        t["_sum_" + k] = 0.0
        t["_n_" + k] = 0
    return t


def _add_reach(t: Dict, reach: Dict) -> None:
    t["n_reaches"] += 1
    if reach.get("causal_reach"):
        t["n_causal_reaches"] += 1
    for k in MEAN_KINEMATICS:
        v = reach.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            t["_sum_" + k] += float(v)
            t["_n_" + k] += 1


def _add_pellet(t: Dict, outcome: Optional[str]) -> None:
    if outcome == "no_pellet":
        t["no_pellet"] += 1
        return
    t["pellets_presented"] += 1
    if outcome in MOVED:
        t[outcome] += 1
    elif outcome in UNTOUCHED_WORDS:
        t["untouched"] += 1
    elif outcome in NOT_BY_A_REACH:
        t["moved_not_by_reach"] += 1
    else:
        t["undetermined"] += 1


def _finish(t: Dict, prefix: str, with_pellets: bool = True) -> "OrderedDict":
    """The tally as published columns, ``<prefix>_<name>``."""
    out = OrderedDict()
    p = prefix + "_"
    if with_pellets:
        pres = t["pellets_presented"]
        out[p + "pellets_presented"] = pres
        for o in MOVED:
            out[p + o] = t[o]
        out[p + "displaced"] = t["displaced_sa"] + t["displaced_outside"]
        out[p + "untouched"] = t["untouched"]
        out[p + "moved_not_by_reach"] = t["moved_not_by_reach"]
        out[p + "undetermined"] = t["undetermined"]
        out[p + "no_pellet"] = t["no_pellet"]
        out[p + "success_rate"] = (t["retrieved"] / pres) if pres else None
        out[p + "contact_rate"] = (sum(t[o] for o in MOVED) / pres) if pres else None
    else:
        for name in ("pellets_presented",) + MOVED + ("displaced", "untouched",
                                                      "moved_not_by_reach",
                                                      "undetermined", "no_pellet",
                                                      "success_rate", "contact_rate"):
            out[p + name] = None
    out[p + "n_reaches"] = t["n_reaches"]
    out[p + "n_reaches_left_out"] = t["n_reaches_left_out"]
    out[p + "n_causal_reaches"] = t["n_causal_reaches"]
    pres = t["pellets_presented"]
    out[p + "reaches_per_pellet"] = (t["n_reaches"] / pres) if (with_pellets and pres) else None
    for k in MEAN_KINEMATICS:
        n = t["_n_" + k]
        out[p + "mean_reach_" + k] = (t["_sum_" + k] / n) if n else None
    if prefix in ("day", "animal"):
        out[p + "n_sessions"] = t["n_sessions"]
    if prefix == "animal":
        out[p + "n_days"] = len(t["_dates"])
    return out


def build_rows(videos: Iterable[Dict], level: str) -> List["OrderedDict"]:
    """The export's rows at ``level`` (module docstring)."""
    level = normalize_level(level)
    videos = list(videos)
    sessions: Dict[str, Dict] = {}
    days: Dict[Tuple, Dict] = defaultdict(_new_tally)
    animals: Dict[str, Dict] = defaultdict(_new_tally)
    animal_has_pellets: Dict[str, bool] = defaultdict(bool)
    day_has_pellets: Dict[Tuple, bool] = defaultdict(bool)

    # Pass 1: tally every pellet and reach into its session, day and animal.
    for doc in videos:
        video = doc.get("video_name")
        ident = parse_video_name(video)
        animal = ident["animal_id"] or video
        day_key = (animal, ident["date"])
        only = _session_only(doc)
        st = _new_tally()
        st["n_sessions"] = 1
        for seg in doc.get("segments") or []:
            if not only:
                _add_pellet(st, seg.get("outcome"))
            st["n_reaches_left_out"] += int(seg.get("n_reaches_left_out") or 0)
            for r in seg.get("reaches") or []:
                _add_reach(st, r)
        sessions[video] = {"tally": st, "only": only, "ident": ident, "doc": doc}
        for agg, key, has in ((days, day_key, day_has_pellets),
                              (animals, animal, animal_has_pellets)):
            t = agg[key]
            for k, v in st.items():
                if k == "_dates":
                    continue
                t[k] += v
            t["_dates"].add(ident["date"])
            if not only:
                has[key] = True

    def _id(video: str) -> "OrderedDict":
        ident = sessions[video]["ident"]
        la, ls = _lab_ids(video)
        row = OrderedDict()
        row["animal_id"] = ident["animal_id"] or video
        if la:
            row["lab_animal_id"] = la
            row["lab_session"] = ls
        row["date"] = ident["date"]
        row["session_id"] = video
        row["tray_type"] = ident["tray_type"]
        row["tray_run"] = ident["tray_run"]
        return row

    def _above(video: str, upto: str) -> "OrderedDict":
        """The columns of every level above the row's own, repeated."""
        s = sessions[video]
        animal = s["ident"]["animal_id"] or video
        day_key = (animal, s["ident"]["date"])
        out = OrderedDict()
        if upto in ("reach", "pellet"):
            out.update(_finish(s["tally"], "session", with_pellets=not s["only"]))
        if upto in ("reach", "pellet", "session"):
            out.update(_finish(days[day_key], "day", with_pellets=day_has_pellets[day_key]))
        out.update(_finish(animals[animal], "animal", with_pellets=animal_has_pellets[animal]))
        return out

    rows: List[OrderedDict] = []
    if level == "animal":
        for animal in sorted(animals):
            row = OrderedDict(animal_id=animal)
            la = None
            for v, s in sessions.items():
                if (s["ident"]["animal_id"] or v) == animal:
                    la = _lab_ids(v)[0]
                    break
            if la:
                row["lab_animal_id"] = la
            row.update(_finish(animals[animal], "animal",
                               with_pellets=animal_has_pellets[animal]))
            rows.append(row)
        return rows
    if level == "day":
        for (animal, date) in sorted(days, key=lambda k: (str(k[0]), str(k[1]))):
            row = OrderedDict(animal_id=animal, date=date)
            row.update(_finish(days[(animal, date)], "day",
                               with_pellets=day_has_pellets[(animal, date)]))
            row.update(_finish(animals[animal], "animal",
                               with_pellets=animal_has_pellets[animal]))
            rows.append(row)
        return rows

    for video in sorted(sessions):
        s = sessions[video]
        if level == "session":
            row = _id(video)
            row.update(_finish(s["tally"], "session", with_pellets=not s["only"]))
            row.update(_above(video, "session"))
            row["source_file"] = s["doc"].get("_source")
            rows.append(row)
            continue
        above = _above(video, level)
        for seg in s["doc"].get("segments") or []:
            seg_outcome = None if s["only"] else seg.get("outcome")
            reaches = seg.get("reaches") or []
            causal_id = seg.get("causal_reach_id")
            has_causal = causal_id is not None or any(r.get("causal_reach") for r in reaches)
            seg_cols = OrderedDict()
            seg_cols["segment_outcome"] = seg_outcome
            seg_cols["segment_n_reaches"] = len(reaches)
            seg_cols["segment_n_reaches_left_out"] = int(seg.get("n_reaches_left_out") or 0)
            seg_cols["segment_causal_reach_id"] = causal_id
            seg_cols["segment_start_frame"] = seg.get("start_frame")
            seg_cols["segment_end_frame"] = seg.get("end_frame")
            for k, v in seg.items():
                if k in ("segment_num", "outcome", "reaches", "n_reaches",
                         "causal_reach_id", "start_frame", "end_frame",
                         "n_reaches_left_out"):
                    continue
                if isinstance(v, (dict, list)):
                    continue
                seg_cols["segment_" + k] = v
            if level == "pellet":
                row = _id(video)
                row["segment_number"] = seg.get("segment_num")
                row.update(seg_cols)
                row.update(above)
                rows.append(row)
                continue
            for r in reaches:
                row = _id(video)
                row["segment_number"] = seg.get("segment_num")
                row["reach_number"] = r.get("reach_num")
                row["reach_id"] = r.get("reach_id")
                # Credited by its own flag, or named by the segment (causal_reach_id):
                # either is the pellet's answer.
                is_causal = bool(r.get("causal_reach")) or (
                    causal_id is not None and r.get("reach_id") == causal_id)
                row["reach_outcome"] = (None if s["only"] else
                                        reach_outcome(seg_outcome, is_causal, has_causal))
                row["reach_is_causal"] = is_causal
                for k, v in r.items():
                    if k in ("reach_id", "reach_num", "segment_num", "outcome",
                             "causal_reach", "extended"):
                        continue
                    if isinstance(v, (dict, list)):
                        continue
                    row["reach_" + k] = v
                for k, v in (r.get("extended") or {}).items():
                    if not isinstance(v, (dict, list)):
                        row["reach_" + k] = v
                row.update(seg_cols)
                row.update(above)
                rows.append(row)
    return rows


# --------------------------------------------------------------- meanings

def column_meanings(columns: Iterable[str]) -> List[Tuple[str, str]]:
    """(column, meaning) for each column, for the xlsx "columns" sheet and the
    documentation."""
    fixed = {
        "animal_id": "The animal, as named in the video file (YYYYMMDD_<animal>_<tray><run>).",
        "lab_animal_id": "The animal as the lab names it, where the project re-encodes ids on the way in.",
        "lab_session": "The video name as the lab names it (same rule).",
        "date": "Recording date (YYYY-MM-DD), from the video name.",
        "session_id": "The session = one tray = one video (the video name).",
        "tray_type": "P = pillar (pellets scored); E = easy, F = flat (no pellet outcomes).",
        "tray_run": "The tray's number that day (the digit after the tray letter).",
        "segment_number": "The pellet's number within the tray (one segment = one pellet).",
        "reach_number": "The reach's number within its segment.",
        "reach_id": "The reach's id within the video.",
        "reach_outcome": ("What THIS reach did: the pellet's outcome for the credited (causal) "
                          "reach; missed for every other reach at that pellet and every reach at "
                          "an untouched pellet; undetermined where a pellet moved with no "
                          "credited reach."),
        "reach_is_causal": "True for the one reach credited with the pellet's outcome.",
        "segment_outcome": ("The pellet's result (same on every reach row of the segment): "
                            "retrieved, displaced_sa, displaced_outside, untouched, "
                            "abnormal / abnormal_exception (moved by something other than "
                            "a reach), no_pellet."),
        "segment_n_reaches": "Reaches measured in this segment.",
        "segment_n_reaches_left_out": "Reaches a person took out of the measurements (ignore window or exclusion).",
        "segment_causal_reach_id": "reach_id of the credited reach, empty if none.",
        "segment_start_frame": "First frame of the segment.",
        "segment_end_frame": "Last frame of the segment.",
        "source_file": "The features file the row was read from.",
    }
    agg = {
        "pellets_presented": "pellets presented (every pellet except no_pellet; a pellet with no reach counts)",
        "retrieved": "pellets retrieved",
        "displaced_sa": "pellets displaced, stayed in the scoring area",
        "displaced_outside": "pellets displaced outside the scoring area",
        "displaced": "pellets displaced (both kinds)",
        "untouched": "pellets untouched",
        "moved_not_by_reach": ("pellets a person saw moved by something other than a reach "
                               "(abnormal); presented, not retrieved, no reach credited"),
        "undetermined": "pellets with no decided outcome (a filed video has none)",
        "no_pellet": "segments in which no pellet was presented (not counted as presented)",
        "success_rate": "retrieved / pellets_presented (0-1)",
        "contact_rate": "(retrieved + displaced) / pellets_presented (0-1)",
        "n_reaches": "reaches measured (all trays, Easy and Flat included)",
        "n_reaches_left_out": "reaches a person took out of the measurements",
        "n_causal_reaches": "reaches credited with a pellet's outcome",
        "reaches_per_pellet": "n_reaches / pellets_presented (pillar trays)",
        "n_sessions": "trays (videos)",
        "n_days": "recording days",
    }
    level_word = {"session": "this tray", "day": "this animal on this date",
                  "animal": "this animal, everything filed"}
    out = []
    for c in columns:
        if c in fixed:
            out.append((c, fixed[c]))
            continue
        hit = None
        for lv, word in level_word.items():
            p = lv + "_"
            if c.startswith(p):
                name = c[len(p):]
                if name in agg:
                    hit = f"{agg[name][0].upper()}{agg[name][1:]} -- {word}."
                elif name.startswith("mean_reach_"):
                    hit = (f"Mean over the measured reaches of {word} of the per-reach "
                           f"value {name[len('mean_reach_'):]}.")
        if hit is None and c.startswith("segment_"):
            hit = "Per-pellet value from the features file (" + c[len("segment_"):] + ")."
        if hit is None and c.startswith("reach_"):
            hit = ("Per-reach value from the features file (" + c[len("reach_"):] + "); see "
                   "REACH_KINEMATIC_DATA_DICTIONARY.md.")
        out.append((c, hit or ""))
    return out


# ---------------------------------------------------------------- writing

def _columns(rows: List[Dict]) -> List[str]:
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    return cols


def write_rows(rows: List[Dict], out_path: Path, fmt: str) -> Path:
    """Write ``rows`` as CSV or xlsx (with a "columns" sheet of meanings)."""
    import pandas as pd
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = _columns(rows)
    df = pd.DataFrame(rows, columns=cols)
    if fmt == "xlsx":
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="results", index=False)
            pd.DataFrame(column_meanings(cols), columns=["column", "meaning"]).to_excel(
                w, sheet_name="columns", index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def default_output(level: str, fmt: str) -> Path:
    """<pipeline folder>/Exports/MouseReach_results_<level>_<time>.<fmt>."""
    from mousereach.config import Paths
    base = Paths.NAS_ROOT or Path.cwd()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / "Exports" / f"MouseReach_results_{level}_{stamp}.{fmt}"


def export_results(level: str = "session", root=None, out=None, fmt: str = "csv") -> Dict:
    """Export the filed results at ``level`` (module docstring).

    ``root``: the filed-results folder (default Analyzed in the pipeline
    folder). ``out``: the file to write (default ``default_output``); never
    inside the filed tree. ``fmt``: csv or xlsx. Returns a summary: ``path``
    (None when nothing was written), ``rows``, ``videos``, ``problems`` (videos
    that could not be read, with why), ``level``, ``root``. Raises ValueError
    for a bad level/format or a missing root."""
    from mousereach.config import Paths
    level = normalize_level(level)
    fmt = (fmt or "csv").lower().lstrip(".")
    if fmt not in FORMATS:
        raise ValueError(f"unknown format '{fmt}': choose csv or xlsx")
    root = Path(root) if root else Paths.ANALYZED_OUTPUT
    if root is None:
        raise ValueError("no filed-results folder: the pipeline folder (nas_root) is not "
                         "set. Run mousereach-setup, or give the folder with --root.")
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"the filed-results folder does not exist: {root}")
    out = Path(out) if out else default_output(level, fmt)
    if out.suffix.lower().lstrip(".") != fmt:
        out = out.with_suffix("." + fmt)
    # Never write into the filed tree: what is filed there is final.
    from mousereach.pipeline.analyzed_tree import filed_tree_refusal
    why = filed_tree_refusal(out.parent)
    try:
        if not why and (out.parent.resolve() == root.resolve()
                        or root.resolve() in out.parent.resolve().parents):
            why = "the output would sit among the results it reads; choose another folder"
    except Exception:
        pass
    if why:
        raise ValueError(f"not written inside the filed results ({out.parent}): {why}")
    videos, problems = load_videos(root)
    rows = build_rows(videos, level)
    result = {"path": None, "rows": len(rows), "videos": len(videos),
              "problems": problems, "level": level, "root": str(root)}
    if rows:
        result["path"] = str(write_rows(rows, out, fmt))
    return result


# -------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(
        prog="mousereach-results",
        description=("Export the FILED results (every video's features file under "
                     "Analyzed) at one level: reach, pellet (= segment), session (= tray "
                     "= video), day (one animal on one date) or animal. Session, day and "
                     "animal are counted from pellets: success rate = retrieved / "
                     "pellets presented, and a pellet with no reach still counts. Needs "
                     "nothing but MouseReach: no watcher, no hand-score sheet, no "
                     "database. Column meanings: docs/RESULTS_EXPORT.md (and the "
                     "'columns' sheet of an xlsx file)."))
    ap.add_argument("--level", default="session",
                    help="reach | pellet (or segment) | session | day | animal "
                         "(default session)")
    ap.add_argument("--root", default=None,
                    help="the filed-results folder (default: Analyzed in the pipeline folder)")
    ap.add_argument("--out", default=None,
                    help="file to write (default: <pipeline folder>/Exports/"
                         "MouseReach_results_<level>_<time>.<format>)")
    ap.add_argument("--format", default="csv", choices=FORMATS, help="csv (default) or xlsx")
    args = ap.parse_args(argv)
    try:
        res = export_results(args.level, args.root, args.out, args.format)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for p in res["problems"]:
        print(f"[!] {p}")
    if not res["path"]:
        print(f"Nothing exported: no filed results under {res['root']}.")
        return 1
    print(f"[OK] {res['rows']} {res['level']} row(s) from {res['videos']} filed video(s) "
          f"-> {res['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
