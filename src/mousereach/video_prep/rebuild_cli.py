"""mousereach-rebuild-recording -- rebuild a lost multi-animal recording from its singles.

Use it only when the recording itself exists nowhere and its single-animal
cuts survive (see video_prep/core/rebuild.py for why and how). The rebuilt
recording goes into the recording intake (Unanalyzed/Multi-Animal) and the
pipeline processes it like any other; a record of what it was built from goes
to <nas_root>/pipeline_records/rebuilt_recordings/.

    mousereach-rebuild-recording --name "<date>_<id1,...,id8>_<tray><n>" \\
        --singles-dir <folder> [--singles-dir <folder> ...] \\
        [--as "<name to write>"] [--reason "<why>"] [--apply]

--name     the recording as its singles are named: each non-empty position's
           single is <date>_<id>_<tray><n>.mp4 in one of the --singles-dir folders
--as       the name to write, when the pipeline's naming differs from the
           singles' (same positions, same order); default: --name
--apply    without it, only checks and says what it would do
"""
import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    from mousereach.config import Paths
    from mousereach.video_prep.core.cropper import parse_collage_filename, is_blank_animal
    from mousereach.video_prep.core.rebuild import RebuildRefused, plan, rebuild

    ap = argparse.ArgumentParser(prog="mousereach-rebuild-recording",
                                 description="Rebuild a lost multi-animal recording from its singles.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--singles-dir", action="append", required=True, type=Path)
    ap.add_argument("--as", dest="as_name", default=None)
    ap.add_argument("--reason", default="the recording itself does not exist")
    ap.add_argument("--dest", type=Path, default=None,
                    help="where to write it (default: the recording intake)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    src = parse_collage_filename(a.name if a.name.endswith(".mkv") else a.name + ".mkv")
    out_name = a.as_name or a.name
    out = parse_collage_filename(out_name if out_name.endswith(".mkv") else out_name + ".mkv")
    singles = {}
    for sid, oid in zip(src["animal_ids"], out["animal_ids"]):
        if is_blank_animal(oid):
            continue
        fname = f"{src['date']}_{sid}_{src['last_part']}.mp4"
        hits = [d / fname for d in a.singles_dir if (d / fname).is_file()]
        if len(hits) > 1:
            print(f"[FAIL] {fname} is in more than one folder: {[str(h) for h in hits]}")
            return 1
        if hits:
            singles[oid] = hits[0]
    dest = a.dest or Paths.MULTI_ANIMAL_SOURCE
    stem = out_name[:-4] if out_name.endswith(".mkv") else out_name
    existing = [d / f"{stem}.mkv" for d in (dest, (Paths.ANALYZED_OUTPUT / "Multi-Animal")
                                             if Paths.ANALYZED_OUTPUT else None) if d]
    for e in existing:
        if e.exists():
            print(f"[FAIL] the recording already exists: {e} -- use it, do not rebuild")
            return 1
    try:
        p = plan(stem, singles)
    except RebuildRefused as e:
        print(f"[FAIL] cannot rebuild {stem}: {e}")
        return 1
    print(f"{stem}: {len(p['probes'])} singles, {8 - len(p['probes'])} empty positions, "
          f"{p['frames']} frames at {p['fps']} fps")
    if not a.apply:
        print("dry run: nothing written (--apply to rebuild)")
        return 0
    records = (Paths.NAS_ROOT / "pipeline_records" / "rebuilt_recordings") if Paths.NAS_ROOT else None
    try:
        path = rebuild(stem, singles, dest, records, a.reason)
    except RebuildRefused as e:
        print(f"[FAIL] {e}")
        return 1
    print(f"[OK] rebuilt {path}" + (f"; record in {records}" if records else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
