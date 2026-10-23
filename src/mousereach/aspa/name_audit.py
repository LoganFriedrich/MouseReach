"""
mousereach.aspa.name_audit - Check an archive's recording names against its own cuts.

WHY THIS EXISTS
---------------
An archive that kept its own single-animal cuts beside its recordings (ASPA's:
``{cohort}/Multi-Animal`` and ``{cohort}/Single_Animal``) names every animal
TWICE: once in the recording's name, once in each cut's. The two were typed
separately, and where they disagree, the pipeline -- which trusts the
recording's name -- files one animal's video under another animal's id. Found
2026-10-23: a recording named for one group of eight held the other group (its
seats are, pixel for pixel, the cuts named for that group), and a run of
recordings named one animal in the seat that the cuts -- and the lab's own score
sheet -- give to another.

HOW A CUT IS TRACED TO ITS RECORDING
------------------------------------
Every cut of a recording is as long as it, to the frame. A cut that is as long
as a recording that names its animal is taken as coming from it (the quick
path; nothing is in doubt). Otherwise the recordings it could have come out of
that day and tray are:

1. one as long as the cut (within ``LENGTH_TOLERANCE`` frames), or
2. one no cut is as long as -- it was trimmed in time after it was cut, or kept
   untrimmed when the cuts were made from a trimmed copy -- that names another
   cut exactly as long as this one (within ``SIBLING_TOLERANCE``): cuts of one
   recording are as long as each other even when the recording no longer is.

Length alone is NOT evidence of a conflict: two recordings of one day can be
within a few frames of each other by chance (seen: 3 frames apart). So each such
candidate is confirmed by the PICTURE: a few frames of the cut are compared with
each of the recording's eight seats (cropper.CROP_COORDS), with the recording
not shifted and shifted by its extra length (trimmed at the start or the end).
A seat matches when its mean difference is at most ``PIXEL_SAME`` (0-255 scale;
a cut against its own seat measures about 1) and every other seat differs by at
least ``PIXEL_OTHER`` (other seats measure 20 and up).

A confirmed match whose seat the recording's name gives to a different animal
(or to an empty seat) is a CONFLICT, and the recording is contested: whose
video each of its seats holds is in doubt. A cut no recording is confirmed for
is taken by its name if a recording that day names it (one no other cut is as
long as), and otherwise is a cut whose recording is lost (see :mod:`mousereach.aspa.feed`, which rebuilds those).

WHAT IS DONE WITH A CONFLICT
----------------------------
Nothing is renamed: which name is right is a fact only a person can confirm
(the renaming of a recording is the one human identity check). Instead:

* ``mousereach-aspa-import-collages`` does not bring a contested recording in;
  it names it for a person.
* ``mousereach-aspa-name-audit`` writes the record
  ``<nas_root>/pipeline_records/archive_name_conflicts.json``: each contested
  recording, the pipeline names it was or would be imported under, and the cuts
  that disagree with it. An integrator that exports the pipeline's results can
  read it to leave every video of a contested recording out of its exports until
  the conflict is resolved -- the record is recomputed by rerunning the audit,
  never edited by hand.

WHAT IT DOES NOT SEE: a cut as long as a recording that names its animal is
taken on its name (the quick path), so two animals' names swapped WITHIN such a
recording are not seen -- no picture is compared for it. A swap shows only when
the cut is in doubt by length (found once that way: a recording kept untrimmed,
whose cuts were traced through each other, had two seats' names swapped).

Lengths are read from the container headers, and pictures only for the few
cuts in doubt (a recording is a gigabyte or more; reading one through is never
needed). Read-only throughout.

CLI:
    mousereach-aspa-name-audit [--cohorts H,L] [--source PATH] [--no-record]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

LENGTH_TOLERANCE = 5
SIBLING_TOLERANCE = 1
# Mean absolute difference per pixel (0-255) between a cut and a seat. Measured
# 2026-10-23: a cut against its own seat ~1-2, against any other seat 20 and up.
PIXEL_SAME = 4.0
PIXEL_OTHER = 12.0
# Frames compared (the first that exist in both).
PIXEL_FRAMES = (500, 5000, 20000)
RECORD_NAME = "archive_name_conflicts.json"

_CUT = re.compile(r"^(\d{8})_([A-Za-z]+\d+)_([A-Za-z]+\d+)\.mp4$")


def header_frames(path: Path) -> Optional[int]:
    """A video's frame count from its header; None when it cannot be opened."""
    from mousereach.aspa.feed import recording_length
    n = recording_length(Path(path))
    return n[0] if n else None


def seat_of(cut: Path, recording: Path, cut_frames: int, rec_frames: Optional[int]
            ) -> Optional[Tuple[int, float, float]]:
    """``(seat index 0-7, its difference, the next seat's)`` when the picture
    shows ``cut`` is one seat of ``recording`` -- not shifted, or shifted by the
    recording's extra length -- else None. See the module docstring."""
    import cv2
    import numpy as np
    from mousereach.video_prep.core.cropper import CROP_COORDS

    def grab(path, idx):
        cap = cv2.VideoCapture(str(path))
        out = []
        try:
            for i in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, img = cap.read()
                out.append(img.astype(np.int16) if ok else None)
        finally:
            cap.release()
        return out

    idx = [i for i in PIXEL_FRAMES if i < cut_frames - 1] or [cut_frames // 2]
    pics = grab(cut, idx)
    shifts = {0}
    if rec_frames and rec_frames > cut_frames:
        shifts.add(rec_frames - cut_frames)
    for shift in sorted(shifts):
        seats = grab(recording, [i + shift for i in idx])
        diffs = []
        for w, h, x, y in CROP_COORDS:
            d = [float(np.abs(a - b[y:y + h, x:x + w]).mean()) for a, b in zip(pics, seats)
                 if a is not None and b is not None and a.shape == (h, w, 3)]
            diffs.append(sum(d) / len(d) if d else float("inf"))
        order = sorted(range(len(diffs)), key=lambda k: diffs[k])
        best, nxt = diffs[order[0]], diffs[order[1]]
        if best <= PIXEL_SAME and nxt >= PIXEL_OTHER:
            return order[0], round(best, 1), round(nxt, 1)
    return None


def _readings(multi: Path) -> Dict[Tuple[str, str], List[Tuple[str, List[str], str]]]:
    """``{(date, TRAY): [(file name, [animal ids in seat order], lab-form session
    stem)]}`` for every recording in ``multi`` the recording import can read."""
    from mousereach.aspa.import_collages import read_recording_stem
    out: Dict[Tuple[str, str], list] = defaultdict(list)
    if not multi.is_dir():
        return out
    for p in sorted(multi.iterdir()):
        if p.suffix.lower() not in (".mkv", ".mp4"):
            continue
        read = read_recording_stem(p.stem)
        if not read:
            continue
        for stem, _ in read[0]:
            parts = stem.split("_")
            out[(parts[0], parts[-1].upper())].append(
                (p.name, [a.upper() for a in parts[1].split(",")], stem))
    return out


def audit(source_root: Path, cohorts: Iterable[str],
          only: Optional[Set[Tuple[str, str, str]]] = None,
          length: Callable[[Path], Optional[int]] = None,
          progress: Optional[Callable[[str], None]] = None,
          picture: Callable = None) -> dict:
    """Trace every cut of ``cohorts`` to its recording (see the module docstring).

    ``only`` -- ``{(cohort, date, TRAY)}`` -- restricts the work to those days.
    Returns ``{"conflicts": [...], "consistent": n, "no_source": n,
    "unreadable": [...], "pictures_compared": n}``; each conflict is one
    contested recording with the cuts that disagree with it, the seat each was
    found in and the name the recording gives that seat. ``picture`` stands in
    for :func:`seat_of` (tests)."""
    from mousereach.aspa.identity import encode_collage_stem
    length = length or header_frames
    picture = picture or seat_of
    source_root = Path(source_root)
    contested: Dict[Tuple[str, str], dict] = {}
    res = {"conflicts": [], "consistent": 0, "no_source": 0, "unreadable": [],
           "pictures_compared": 0}
    for cohort in cohorts:
        from mousereach.aspa.feed import cuts_folder
        multi = source_root / cohort / "Multi-Animal"
        sa = cuts_folder(source_root / cohort) or (source_root / cohort / "Single_Animal")
        recs = _readings(multi)
        cuts: Dict[Tuple[str, str], list] = defaultdict(list)
        names = sorted(x.name for x in sa.iterdir() if _CUT.match(x.name)) if sa.is_dir() else []
        for n in names:
            d, aid, tray = _CUT.match(n).groups()
            if only is not None and (cohort, d, tray.upper()) not in only:
                continue
            f = length(sa / n)
            if not f:
                res["unreadable"].append(f"{cohort}/{n}")
                continue
            cuts[(d, tray.upper())].append((n, aid.upper(), f))
        if progress:
            progress(f"{cohort}: {sum(len(v) for v in cuts.values())} cuts measured")
        rec_len: Dict[str, Optional[int]] = {}
        for key, day in cuts.items():
            day_recs = recs.get(key, [])
            for r, _, _ in day_recs:
                if r not in rec_len:
                    rec_len[r] = length(multi / r)

            def as_long(r, f):
                return bool(rec_len.get(r)) and abs(rec_len[r] - f) <= LENGTH_TOLERANCE

            # A recording some cut is as long as has its cuts known by length;
            # a cut of another length did not come out of it.
            anchored = {r for r, _, _ in day_recs if any(as_long(r, f) for _, _, f in day)}
            for n, aid, f in day:
                direct = [r for r, _, _ in day_recs if as_long(r, f)]
                named = [r for r, ids, _ in day_recs if aid in ids]
                if any(r in named for r in direct):
                    res["consistent"] += 1                   # the quick path
                    continue
                cands = list(direct)
                for n2, a2, f2 in day:
                    if n2 != n and abs(f2 - f) <= SIBLING_TOLERANCE:
                        cands += [r for r, ids, _ in day_recs
                                  if a2 in ids and r not in anchored and r not in cands]
                found = None
                for r in cands:
                    res["pictures_compared"] += 1
                    hit = picture(sa / n, multi / r, f, rec_len.get(r))
                    if hit:
                        found = (r, hit, "by length" if r in direct else "through a sibling cut")
                        break
                if found is None:
                    # Taken by its name -- unless every recording naming it is
                    # accounted for by other cuts of its own length: then this
                    # cut's recording is lost.
                    if any(r not in anchored for r in named):
                        res["consistent"] += 1
                    else:
                        res["no_source"] += 1
                    continue
                r, (seat, best, nxt), how = found
                ids = next(ids for rr, ids, _ in day_recs if rr == r)
                given = ids[seat] if seat < len(ids) else ""
                if given == aid:
                    res["consistent"] += 1
                    continue
                entry = contested.setdefault((cohort, r), {
                    "cohort": cohort, "recording": r, "pipeline_recordings": [],
                    "cuts_named_differently": []})
                # the name(s) the recording import gives it in the pipeline
                entry["pipeline_recordings"] = sorted(
                    set(entry["pipeline_recordings"])
                    | ({encode_collage_stem(st) for rr, _, st in day_recs if rr == r} - {None}))
                entry["cuts_named_differently"].append({
                    "cut": n, "animal": aid, "frames": f, "traced": how,
                    "seat": seat + 1,
                    "recording_names_seat": "(empty)" if given.isdigit() or not given else given,
                    "picture_difference": best, "next_seat_difference": nxt})
    res["conflicts"] = [contested[k] for k in sorted(contested)]
    return res


def record_path(nas_root: Optional[Path]) -> Optional[Path]:
    return (Path(nas_root) / "pipeline_records" / RECORD_NAME) if nas_root else None


def write_record(result: dict, path: Path, source_root: Path, cohorts: List[str]) -> Path:
    """Write the audit as the record an integrator reads (see the module docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps({
        "schema_version": "1.0",
        "description": ("Recordings whose seats the archive's own single-animal cuts name "
                        "differently. Every video cut from one is of doubtful identity until "
                        "a person resolves the name. Recomputed by mousereach-aspa-name-audit; "
                        "never edit by hand."),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "cohorts": cohorts,
        "method": {"length_tolerance": LENGTH_TOLERANCE,
                   "sibling_tolerance": SIBLING_TOLERANCE,
                   "pixel_same": PIXEL_SAME, "pixel_other": PIXEL_OTHER,
                   "pixel_frames": list(PIXEL_FRAMES)},
        "counts": {"conflicting_recordings": len(result["conflicts"]),
                   "consistent_cuts": result["consistent"],
                   "cuts_without_a_recording": result["no_source"],
                   "unreadable_cuts": len(result["unreadable"]),
                   "pictures_compared": result["pictures_compared"]},
        "conflicts": result["conflicts"],
    }, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def main(argv=None) -> int:
    from mousereach.aspa.import_collages import IN_SCOPE_COHORTS, default_source_root
    from mousereach.config import Paths

    ap = argparse.ArgumentParser(
        prog="mousereach-aspa-name-audit",
        description="Check the ASPA archive's recording names against its own cuts, and "
                    "write the record of recordings whose seats the cuts name differently. "
                    "Reads only.")
    ap.add_argument("--cohorts", default=None,
                    help="Comma-separated cohort folders (default: %s)" % ",".join(IN_SCOPE_COHORTS))
    ap.add_argument("--source", default=None,
                    help="Archive root (default: the NAS root's Archive/historical/ASPA)")
    ap.add_argument("--no-record", action="store_true",
                    help="Print only; do not write the record")
    a = ap.parse_args(argv)

    source = Path(a.source) if a.source else default_source_root()
    cohorts = a.cohorts.split(",") if a.cohorts else list(IN_SCOPE_COHORTS)
    if source is None or not source.is_dir():
        print(f"[FAIL] archive not found: {source}")
        return 1
    print(f"Archive : {source}")
    print(f"Cohorts : {', '.join(cohorts)}")
    print("Measuring every cut and recording (headers only; this takes a while)...")
    res = audit(source, cohorts, progress=lambda m: print("  " + m, flush=True))
    print("")
    print(f"Cuts traced to a recording that names them : {res['consistent']}")
    print(f"Cuts whose recording is lost               : {res['no_source']}")
    print(f"Cuts compared by picture                   : {res['pictures_compared']}")
    print(f"Cuts whose length cannot be read           : {len(res['unreadable'])}")
    print(f"[!] Recordings whose seats the cuts name differently: {len(res['conflicts'])}")
    for c in res["conflicts"]:
        print(f"  {c['cohort']}/{c['recording']}")
        for x in c["cuts_named_differently"]:
            print(f"    seat {x['seat']}: the recording says {x['recording_names_seat']}, "
                  f"the cut {x['cut']} says {x['animal']}")
    if a.no_record:
        return 0
    out = record_path(Paths.NAS_ROOT)
    if out is None:
        print("[FAIL] no NAS root configured: run mousereach-setup (or use --no-record)")
        return 1
    write_record(res, out, source, cohorts)
    print(f"\nRecord written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
