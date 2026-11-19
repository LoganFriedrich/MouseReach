#!/usr/bin/env python3
"""Convert grasp_features.json files to CSV for easy viewing"""

import json
from pathlib import Path
import csv

from mousereach.config import Paths


def _lab_ids(video_name):
    """(animal_id, session) as the LAB names them, for a human-facing export.

    ASPA animals are re-encoded on the way into the pipeline (J11 -> ASPA1011) so
    every stage sees one id shape. Nothing decoded on the way out, so results
    surfaced as ASPA1011 and a reader had to know the alphabet rule to get back to
    J11. These two columns are that missing half.

    video_name itself is left alone -- it is the key that ties a row back to the
    file it came from. Returns (None, video_name) for a non-ASPA video, so every
    project can go through the same call.
    """
    try:
        from mousereach.aspa import lab_animal_id, decode_video_stem
        return lab_animal_id(video_name), decode_video_stem(video_name)
    except Exception:
        return None, video_name


def main():
    """Retired: redirected to ``mousereach-results --level reach``.

    WHY: this command looked for ``*_grasp_features.json`` in the local
    Processing folder -- a file name the pipeline no longer writes, in a
    folder that filing empties -- so on current output it found nothing. The
    per-reach export of the FILED results, with every person's answer applied
    and level-first column names, is mousereach-results (export/results.py).
    """
    import argparse
    import sys
    argparse.ArgumentParser(
        description="Retired: runs 'mousereach-results --level reach' (the filed "
                    "results, one row per reach). Extra options are passed on.",
    ).parse_known_args()
    print("mousereach-features-csv is retired (it read *_grasp_features.json, a file "
          "the pipeline no longer writes). Running: mousereach-results --level reach")
    from mousereach.export.results import main as results_main
    return results_main(["--level", "reach"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
