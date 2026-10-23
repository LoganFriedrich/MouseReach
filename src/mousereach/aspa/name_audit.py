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
Every cut of a recording is as long as it, to the frame. So a cut's source
recording, that day and tray, is found:

1. directly -- a recording as long as the cut (within ``LENGTH_TOLERANCE``
   frames: each was encoded separately, and a header can be a frame or two off);
2. else through a sibling -- another cut of that day and tray exactly as long
   (within ``SIBLING_TOLERANCE``) whose animal a recording names: a recording
   trimmed in time after it was cut is no longer as long as its cuts, but its
   cuts are still as long as each other.

A cut whose source recording does not list its animal is a CONFLICT, and the
recording is contested: whose video each of its seats holds is in doubt. A cut
with no source is simply a cut whose recording is lost (see
:mod:`mousereach.aspa.feed`, which rebuilds those).

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

Lengths are read from the container headers only (a recording is a gigabyte or
more; reading it through is never needed for this). Read-only throughout.

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
RECORD_NAME = "archive_name_conflicts.json"

_CUT = re.compile(r"^(\d{8})_([A-Za-z]+\d+)_([A-Za-z]+\d+)\.mp4$")


def header_frames(path: Path) -> Optional[int]:
    """A video's frame count from its header; None when it cannot be opened."""
    from mousereach.aspa.feed import recording_length
    n = recording_length(Path(path))
    return n[0] if n else None


def _readings(multi: Path) -> Dict[Tuple[str, str], List[Tuple[str, Set[str], str]]]:
    """``{(date, TRAY): [(file name, {animal ids}, lab-form session stem)]}`` for
    every recording in ``multi`` the recording import can read."""
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
                (p.name, {a.upper() for a in parts[1].split(",")}, stem))
    return out


def audit(source_root: Path, cohorts: Iterable[str],
          only: Optional[Set[Tuple[str, str, str]]] = None,
          length: Callable[[Path], Optional[int]] = None,
          progress: Optional[Callable[[str], None]] = None) -> dict:
    """Trace every cut of ``cohorts`` to its recording (see the module docstring).

    ``only`` -- ``{(cohort, date, TRAY)}`` -- restricts the work to those days.
    Returns ``{"conflicts": [...], "ambiguous": [...], "consistent": n,
    "no_source": n, "unreadable": [...]}``; each conflict is one contested
    recording with the cuts that disagree with it."""
    from mousereach.aspa.identity import encode_collage_stem
    length = length or header_frames
    source_root = Path(source_root)
    contested: Dict[Tuple[str, str], dict] = {}
    res = {"conflicts": [], "ambiguous": [], "consistent": 0, "no_source": 0,
           "unreadable": []}
    for cohort in cohorts:
        multi, sa = source_root / cohort / "Multi-Animal", source_root / cohort / "Single_Animal"
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
                direct = {r for r, _, _ in day_recs if as_long(r, f)}
                src = set(direct)
                if not src:
                    # No recording is as long as the cut (it was trimmed after
                    # it was cut, or is lost): among the recordings no cut is as
                    # long as, the one naming this animal, and those naming a
                    # cut exactly as long as it.
                    loose = [(r, ids) for r, ids, _ in day_recs if r not in anchored]
                    src = {r for r, ids in loose if aid in ids}
                    for n2, a2, f2 in day:
                        if n2 != n and abs(f2 - f) <= SIBLING_TOLERANCE:
                            src |= {r for r, ids in loose if a2 in ids}
                if not src:
                    res["no_source"] += 1
                    continue
                if any(aid in ids for r, ids, _ in day_recs if r in src):
                    res["consistent"] += 1
                    continue
                cut = {"cut": n, "animal": aid, "frames": f,
                       "traced": "by length" if direct else "through a sibling cut"}
                if len(src) > 1:
                    res["ambiguous"].append(dict(cut, cohort=cohort, recordings=sorted(src)))
                    continue
                r = next(iter(src))
                entry = contested.setdefault((cohort, r), {
                    "cohort": cohort, "recording": r, "pipeline_recordings": [],
                    "cuts_named_differently": []})
                # the name(s) the recording import gives it in the pipeline
                entry["pipeline_recordings"] = sorted(
                    set(entry["pipeline_recordings"])
                    | ({encode_collage_stem(st) for rr, _, st in day_recs if rr == r} - {None}))
                entry["cuts_named_differently"].append(cut)
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
                   "sibling_tolerance": SIBLING_TOLERANCE},
        "counts": {"conflicting_recordings": len(result["conflicts"]),
                   "consistent_cuts": result["consistent"],
                   "cuts_without_a_recording": result["no_source"],
                   "ambiguous_cuts": len(result["ambiguous"]),
                   "unreadable_cuts": len(result["unreadable"])},
        "conflicts": result["conflicts"],
        "ambiguous": result["ambiguous"],
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
    print(f"Cuts traced to more than one recording     : {len(res['ambiguous'])}")
    print(f"Cuts whose length cannot be read           : {len(res['unreadable'])}")
    print(f"[!] Recordings whose seats the cuts name differently: {len(res['conflicts'])}")
    for c in res["conflicts"]:
        named = ",".join(sorted({x["animal"] for x in c["cuts_named_differently"]}))
        print(f"  {c['cohort']}/{c['recording']}: cut(s) named {named}")
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
