"""Check whether any GT segments with outcome='untouched' have non-null
interaction_frame or outcome_known_frame (stale from the tool's
'can't unset IFR once set' bug).
"""
import json
from pathlib import Path

GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\gt"
)


def main():
    total_untouched = 0
    stale_ifr = []
    stale_okf = []
    for f in sorted(GT_DIR.glob("*_unified_ground_truth.json")):
        d = json.loads(f.read_text())
        video = d["video_name"]
        for s in d["outcomes"]["segments"]:
            if s.get("outcome") != "untouched":
                continue
            total_untouched += 1
            ifr = s.get("interaction_frame")
            okf = s.get("outcome_known_frame")
            if ifr is not None:
                stale_ifr.append({
                    "video": video,
                    "seg": s["segment_num"],
                    "ifr": ifr,
                    "okf": okf,
                })
            if okf is not None and ifr is None:
                # Untouched with okf but no ifr -- also potentially stale
                stale_okf.append({
                    "video": video,
                    "seg": s["segment_num"],
                    "okf": okf,
                })

    print(f"Total untouched GT segments on this corpus: {total_untouched}")
    print(f"With non-null interaction_frame (stale candidates): {len(stale_ifr)}")
    print(f"With non-null outcome_known_frame only: {len(stale_okf)}")
    print()
    if stale_ifr:
        print("Untouched GT segments with stale interaction_frame:")
        for r in stale_ifr:
            print(f"  {r['video']} s{r['seg']:>3}: ifr={r['ifr']}, okf={r['okf']}")
    if stale_okf:
        print()
        print("Untouched GT segments with stale outcome_known_frame (no ifr):")
        for r in stale_okf:
            print(f"  {r['video']} s{r['seg']:>3}: okf={r['okf']}")


if __name__ == "__main__":
    main()
