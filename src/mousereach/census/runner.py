"""Run the pipeline census over THIS deployment's folders.

WHY THIS EXISTS
---------------
The operator's four questions, in order (they are the acceptance spec):

1. TOTAL WORKLOAD -- how many single-animal videos will ultimately need
   analysis, derived from every unique multi-animal collage name that exists
   anywhere in the pipeline (intake, archives, retired collages), unioned
   with every session that already left an artifact on disk.
2. HOW MANY ARE NOT FINISHED yet.
3. That remainder BROKEN DOWN by where each session is -- categories that
   map to real pipeline stages AND real storage folders, so "it is in
   crop_dlc" tells a person which folder to open.
4. PERCENTAGES and TIME-TO-COMPLETE estimates: pace measured from output-file
   timestamps, projected over the remaining backlog, always labelled with its
   basis -- an estimate is reported as an estimate, never dressed as a fact.

This module is the ONLY file in the census package that knows the folder
layout (through ``mousereach.config.Paths``). The three modules beside it are
generic and carry no deployment facts; keep it that way.

The GUI is the deliverable; this headless entry point is the debugging aid
and the machine interface an integrator (a database tool's GUI) calls:

    mousereach-census --json            # machine-readable, full detail
    mousereach-census                   # human table (ASCII only)
    python -m mousereach.census.runner  # same, when console shims are absent

DATABASE MEMBERSHIP IS DELIBERATELY ABSENT HERE. "Analyzed" is three
conditions -- analysis finished AND in the database AND in the final output
tree -- and this side cannot see the database. So the JSON carries, per
session, the element WITHOUT the database condition plus a ``finished`` flag,
and the caller that owns the database view promotes finished+landed sessions
to analyzed and evaluates the finished-but-not-landed invariant. A caller
with no database view structurally cannot render an "analyzed" count from
this output alone -- that is the honest answer, and it is deliberate
(see locate_sessions.DatabaseViewUnavailable).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, timedelta
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from .expected_sessions import expected_sessions
from .locate_sessions import ELEMENT_ORDER, resolve_elements, tray_from_stem
from .review_completeness import scan_queue, summarise

# A session is DONE (filesystem sense) when the four algorithms have all
# produced output. Verified against the live corpus 2026-09-01.
REQUIRED = ("_reaches.json", "_pellet_outcomes.json", "_reach_assignments.json",
            "_features.json")
# On outcome-free trays (pellet level with the scoring area; a displaced
# pellet can be retried, so per-pellet outcomes are not meaningful) a session
# legitimately terminates with reaches only.
SESSION_ONLY = ("_reaches.json",)
OUTCOME_FREE_TRAYS = ("E", "F")
# Any of these present means analysis has at least STARTED.
PARTIAL = REQUIRED + ("_segments.json",)

SUFFIXES = tuple(sorted(set(REQUIRED) | set(SESSION_ONLY) | set(PARTIAL)))

# Real review bundles are named by video stem, which always begins with an
# 8-digit date. Queue folders also hold archive/scratch directories
# (dot-prefixed, underscore-prefixed, or plainly named) that are NOT bundles.
_BUNDLE_NAME = re.compile(r"^\d{8}_")


def walk_analyzed(root) -> tuple:
    """One walk over the final output tree.

    Returns ``(index, mtimes, collages)``: ``{stem: set(suffixes present)}``
    for the element resolver; ``{stem: {suffix: mtime}}`` for throughput (the
    timestamps ride along free, so pace measurement costs no second pass over
    the NAS); and every collage-shaped video found INSIDE the tree. The last
    matters: retired collages are moved into Analyzed, and stragglers (e.g.
    outcome-free-tray recordings archived uncropped) live in cohort subfolders
    there -- a workload denominator that missed them would under-count.

    Filters on the NAME only (no is_file stat per entry) -- over a network
    share the stat is the cost, and a directory named ``*.json`` would merely
    contribute a harmless unparsable name.
    """
    index: Dict[str, Set[str]] = {}
    mtimes: Dict[str, Dict[str, float]] = {}
    collages: list = []
    root = Path(root)
    if not root.exists():
        return index, mtimes, collages
    for p in root.rglob("*"):
        name = p.name
        low = name.lower()
        if low.endswith(".json"):
            for suf in SUFFIXES:
                if name.endswith(suf):
                    stem = name[: -len(suf)]
                    index.setdefault(stem, set()).add(suf)
                    try:
                        mtimes.setdefault(stem, {})[suf] = p.stat().st_mtime
                    except OSError:
                        pass
                    break
        elif low.endswith((".mkv", ".mp4")) and "," in p.stem:
            collages.append(p)
    return index, mtimes, collages


def ids_in_dir(d, exts=(".mp4", ".mkv")) -> Set[str]:
    """Video stems in a folder -- counts VIDEOS, not files (a quarantine
    holding ``x.mp4`` + ``x.quarantine.json`` is one video). Stems are
    normalized so a DLC output or ``_full`` copy maps to its session."""
    from mousereach.video_prep.core.collage_provenance import normalize_video_stem
    d = Path(d) if d else None
    if not d or not d.exists():
        return set()
    out = set()
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            s = normalize_video_stem(p)
            if s:
                out.add(s)
    return out


def bundles_in(d) -> Set[str]:
    """Review-bundle stems in a queue folder. Only date-named directories
    count: dot-dirs, underscore-prefixed archives and other scratch folders
    land inside queues and would otherwise be phantom videos."""
    d = Path(d) if d else None
    if not d or not d.exists():
        return set()
    return {x.name for x in d.iterdir()
            if x.is_dir() and _BUNDLE_NAME.match(x.name)}


def project_of(stem: str) -> str:
    """Leading letters of the animal token: 20250624_CNT0101_P1 -> CNT."""
    parts = stem.split("_")
    if len(parts) < 2:
        return "other"
    ident = parts[1]
    for i, ch in enumerate(ident):
        if ch.isdigit():
            return ident[:i] or "other"
    return ident or "other"


def finished_set(sessions: dict, index: Dict[str, Set[str]]) -> Set[str]:
    """Sessions whose analysis outputs are all present (filesystem sense --
    the database condition is the caller's, see the module docstring)."""
    done = set()
    for sid, info in sessions.items():
        tray = (info or {}).get("tray") or tray_from_stem(sid)
        req = SESSION_ONLY if tray in OUTCOME_FREE_TRAYS else REQUIRED
        if set(req).issubset(index.get(sid) or set()):
            done.add(sid)
    return done


def finished_times(finished: Iterable[str], sessions: dict,
                   mtimes: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """{sid: when it became finished} = the LAST required output's mtime."""
    out = {}
    for sid in finished:
        tray = (sessions.get(sid) or {}).get("tray") or tray_from_stem(sid)
        req = SESSION_ONLY if tray in OUTCOME_FREE_TRAYS else REQUIRED
        ts = [mtimes.get(sid, {}).get(s) for s in req]
        ts = [t for t in ts if t]
        if ts:
            out[sid] = max(ts)
    return out


def pace_per_day(timestamps: Iterable[float], window_days: int) -> Optional[float]:
    """Events per day over the trailing window; None when the window is empty
    (no pace is measurable -- never report a zero that reads as 'stopped')."""
    now = time.time()
    cut = now - window_days * 86400
    n = sum(1 for t in timestamps if t and t >= cut)
    return (n / float(window_days)) if n else None


def review_pace(window_days: int) -> Optional[float]:
    """Human reviews saved per day, from the durable review store's file
    timestamps. None when the store is unconfigured or quiet."""
    try:
        from mousereach.review.causal_review_io import durable_review_dir
        d = durable_review_dir()
    except Exception:
        return None
    if not d or not Path(d).exists():
        return None
    stamps = []
    for p in Path(d).glob("*_causal_review.json"):
        try:
            stamps.append(p.stat().st_mtime)
        except OSError:
            pass
    return pace_per_day(stamps, window_days)


def estimate_eta(machine_backlog: int, human_backlog: int,
                 finished_per_day: Optional[float],
                 reviews_per_day: Optional[float], window_days: int) -> dict:
    """Backlog sizes and projected completion. Estimates carry their basis in
    the output; a missing pace yields a missing estimate, never a made-up one.

    ``machine_backlog`` must count only sessions the machines still owe work
    on -- not started, in crop/pose, or mid-algorithms. Finished sessions
    awaiting their database import and sessions queued for a person are NOT
    machine work; folding them in overstated the first live estimate by ~30%.
    """
    machine, human = machine_backlog, human_backlog
    out = {
        "machine_backlog": machine,
        "human_backlog": human,
        "window_days": window_days,
        "basis": ("pace measured over the trailing %d days from output-file "
                  "timestamps; an estimate from recent pace, not a promise"
                  % window_days),
    }
    if finished_per_day:
        d = machine / finished_per_day
        out["finished_per_day"] = round(finished_per_day, 1)
        out["machine_days"] = round(d, 1)
        out["machine_date"] = (date.today() + timedelta(days=ceil(d))).isoformat()
    if reviews_per_day:
        d = human / reviews_per_day
        out["reviews_per_day"] = round(reviews_per_day, 1)
        out["human_days"] = round(d, 1)
        out["human_date"] = (date.today() + timedelta(days=ceil(d))).isoformat()
    return out


def run_census(window_days: int = 14) -> dict:
    """The whole census, as one JSON-able dict. See the module docstring for
    what is deliberately absent (the database view)."""
    from mousereach.config import Paths
    try:
        from mousereach.aspa.identity import encode_collage_stem
    except ImportError:                                   # pragma: no cover
        encode_collage_stem = None

    from mousereach.video_prep.core.collage_provenance import expected_offspring

    t0 = time.time()
    nas = Paths.NAS_ROOT

    # -- Every root this census reads, recorded and existence-checked.
    # A missing root must be a visible problem, never quietly zero sessions
    # (the relic-database lesson: a wrong path that answers plausibly costs
    # wrong conclusions; a wrong path that is REPORTED costs nothing).
    roots: Dict[str, dict] = {}
    missing = []

    def _root(label: str, p) -> Optional[Path]:
        ok = bool(p) and Path(p).exists()
        roots[label] = {"path": str(p) if p else None, "exists": ok}
        if not ok:
            missing.append(label)
        return Path(p) if ok else None

    collage_roots = [r for r in (
        _root("collages_intake", Paths.MULTI_ANIMAL_SOURCE),
        _root("collages_aspa_archive",
              (nas / "Archive" / "historical" / "ASPA") if nas else None),
    ) if r]

    analyzed_root = _root("analyzed", Paths.ANALYZED_OUTPUT)

    # -- The final output tree (the slow part). Timestamps AND any collages
    # living inside it -- retired ones, uncropped stragglers -- ride along in
    # the same walk, so the denominator sees them without a second NAS pass.
    index, mtimes, analyzed_collages = (walk_analyzed(analyzed_root)
                                        if analyzed_root else ({}, {}, []))

    # -- Where sessions sit before they finish. Elements are the operator's
    # categories; each maps to real folders, listed here so "crop_dlc" is
    # openable, not jargon. Sessions being worked on another node's local
    # disk are invisible to this scan and show in the NAS stage they were
    # claimed from -- a short-lived, self-healing discrepancy.
    locations = {
        "crop_dlc": (ids_in_dir(_root("dlc_queue_nas",
                                      (nas / "DLC_Queue") if nas else None))
                     | ids_in_dir(_root("dlc_queue_local", Paths.DLC_QUEUE))
                     | ids_in_dir(_root("single_animal",
                                        Paths.SINGLE_ANIMAL_OUTPUT))),
        "mousereach": (ids_in_dir(_root("dlc_complete", Paths.DLC_STAGING))
                       | ids_in_dir(_root("processing_local", Paths.PROCESSING))),
        "triage": bundles_in(_root("triage_queue", Paths.TRIAGE_REVIEW)),
        "deep_review": bundles_in(_root("deep_review_queue", Paths.DEEP_REVIEW)),
        "quarantined": ids_in_dir(_root("quarantine",
                                        (nas / "Processing" / "Quarantine")
                                        if nas else None)),
    }

    # -- The denominator: collage expansion UNIONED with found artifacts.
    found = set(index)
    for members in locations.values():
        found |= members
    exp = expected_sessions(collage_roots, expected_offspring,
                            found_sessions=found,
                            encode_stem=encode_collage_stem,
                            extra_collages=analyzed_collages)
    sessions, diagnostics = exp["sessions"], exp["diagnostics"]
    diagnostics["missing_roots"] = missing

    # -- One element per session (no database view on this side).
    elements = resolve_elements(
        sessions, index, locations, in_database=None,
        outcome_free_trays=OUTCOME_FREE_TRAYS,
        analysis_outputs=REQUIRED, session_only_outputs=SESSION_ONLY,
        partial_analysis_outputs=PARTIAL)

    finished = finished_set(sessions, index)
    fin_at = finished_times(finished, sessions, mtimes)

    by_element: Dict[str, int] = {}
    by_project: Dict[str, Dict[str, int]] = {}
    per_session: Dict[str, dict] = {}
    for sid, el in elements.items():
        info = sessions.get(sid) or {}
        proj = project_of(sid)
        tray = info.get("tray") or tray_from_stem(sid)
        by_element[el] = by_element.get(el, 0) + 1
        by_project.setdefault(proj, {})
        by_project[proj][el] = by_project[proj].get(el, 0) + 1
        per_session[sid] = {"element": el, "finished": sid in finished,
                            "tray": tray, "project": proj,
                            "source": info.get("source")}

    fpd = pace_per_day(fin_at.values(), window_days)
    rpd = review_pace(window_days)
    machine_backlog = sum(
        1 for s in per_session.values()
        if s["element"] in ("unanalyzed", "crop_dlc")
        or (s["element"] == "mousereach" and not s["finished"]))
    human_backlog = (by_element.get("triage", 0)
                     + by_element.get("deep_review", 0))
    eta = estimate_eta(machine_backlog, human_backlog, fpd, rpd, window_days)

    # -- Review completeness (who is finished-but-unreleased in each queue).
    review = {}
    try:
        from mousereach.review.causal_review_io import durable_review_dir
        durable = durable_review_dir()
    except Exception:
        durable = None
    for qname, qroot in (("triage", Paths.TRIAGE_REVIEW),
                         ("deep_review", Paths.DEEP_REVIEW)):
        if qroot and Path(qroot).exists():
            rows, skipped = scan_queue(qroot, durable_dir=durable)
            s = summarise(rows)
            s["no_review_document"] = len(skipped)
            review[qname] = s

    total = len(sessions)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_seconds": round(time.time() - t0, 1),
        "window_days": window_days,
        "totals": {
            "expected": total,                    # question 1: the workload
            "finished_files": len(finished),      # filesystem sense only
            "unfinished": total - len(finished),  # question 2
        },
        "by_element": {k: by_element.get(k, 0)
                       for k in ELEMENT_ORDER if by_element.get(k)},
        "by_project": by_project,
        "eta": eta,                               # question 4
        "review": review,
        "sessions": per_session,                  # per-sid, for the DB-side join
        "diagnostics": diagnostics,               # problems, never absences
        "roots": roots,                           # question 3: real folders
    }


def _print_table(c: dict) -> None:
    """Human-readable census (ASCII only -- Windows cp1252 consoles)."""
    tot = c["totals"]
    print("PIPELINE CENSUS  %s  (scan took %ss)"
          % (c["generated_at"], c["scan_seconds"]))
    print("=" * 64)
    print("Total workload (unique single-animal sessions): %d" % tot["expected"])
    pct = (100.0 * tot["finished_files"] / tot["expected"]) if tot["expected"] else 0
    print("Analysis finished (files on disk):              %d  (%.1f%%)"
          % (tot["finished_files"], pct))
    print("Not finished yet:                               %d" % tot["unfinished"])
    print()
    label = {
        "unanalyzed": "not started (collage in Unanalyzed/Multi-Animal)",
        "crop_dlc": "cropping / pose estimation (DLC_Queue, Single_Animal)",
        "mousereach": "analysis algorithms (DLC_Complete, Processing)",
        "triage": "waiting for a person: triage queue",
        "deep_review": "waiting for a person: deep review queue",
        "quarantined": "held out: quarantine (unprocessable as-is)",
        "analyzed": "finished and landed",
        "session_only": "finished (outcome-free tray)",
    }
    for el, n in c["by_element"].items():
        p = 100.0 * n / tot["expected"] if tot["expected"] else 0
        print("  %-12s %6d  (%4.1f%%)  %s" % (el, n, p, label.get(el, "")))
    print()
    projs = sorted(c["by_project"])
    if projs:
        els = [e for e in ELEMENT_ORDER if any(c["by_project"][p].get(e)
                                               for p in projs)]
        print("%-14s" % "" + "".join("%10s" % p for p in projs))
        for el in els:
            print("%-14s" % el
                  + "".join("%10d" % c["by_project"][p].get(el, 0) for p in projs))
        print()
    eta = c["eta"]
    print("PACE AND ESTIMATES (%s)" % eta["basis"])
    if eta.get("finished_per_day"):
        print("  machine backlog %d at ~%.1f finished/day -> ~%.1f days (%s)"
              % (eta["machine_backlog"], eta["finished_per_day"],
                 eta["machine_days"], eta["machine_date"]))
    else:
        print("  machine backlog %d -- no finishes inside the window, "
              "no pace measurable" % eta["machine_backlog"])
    if eta.get("reviews_per_day"):
        print("  review backlog  %d at ~%.1f reviews/day -> ~%.1f days (%s)"
              % (eta["human_backlog"], eta["reviews_per_day"],
                 eta["human_days"], eta["human_date"]))
    else:
        print("  review backlog  %d -- no saved reviews inside the window, "
              "no pace measurable" % eta["human_backlog"])
    print()
    d = c["diagnostics"]
    print("DIAGNOSTICS -- these are problems, not absences:")
    print("  collage files seen                 : %s" % d.get("collage_files_seen"))
    print("  sessions after source selection    : %s"
          % d.get("collage_sessions_after_selection"))
    print("  blank camera slots excluded        : %s"
          % d.get("blank_camera_slots_excluded"))
    print("  collages that parsed to NOTHING    : %s"
          % len(d.get("collages_that_parsed_to_nothing") or []))
    print("  sessions with artifact, no collage : %s"
          % d.get("sessions_with_artifact_but_no_collage"))
    print("  files dropped as duplicate/variant : %s" % d.get("dropped_by_reason"))
    if d.get("missing_roots"):
        print("  [!] MISSING ROOTS (scanned as empty): %s"
              % ", ".join(d["missing_roots"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Pipeline census: total workload, per-stage breakdown, "
                    "pace and completion estimates.")
    ap.add_argument("--json", action="store_true",
                    help="emit the full machine-readable census on stdout")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the JSON to this file")
    ap.add_argument("--days", type=int, default=14,
                    help="trailing window for pace measurement (default 14)")
    args = ap.parse_args(argv)

    c = run_census(window_days=args.days)
    if args.out:
        args.out.write_text(json.dumps(c, indent=0), encoding="utf-8")
    if args.json:
        json.dump(c, sys.stdout, indent=0)
        print()
    else:
        _print_table(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
