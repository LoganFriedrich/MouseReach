"""Child-process side of an interruptible pose.

Run as:  python -m mousereach.dlc.core.interruptible_worker <args.json>

WHY A SEPARATE PROCESS: deeplabcut.analyze_videos cannot be interrupted once
it starts (about fourteen minutes per video), and a recording must never have
to compete with it. Running the pose here lets the parent
(mousereach.dlc.core.interruptible) kill this whole process the moment a
recording starts. See that module for the full story.

The args file holds: video_path, config_path, output_dir, gpu, shuffle,
result_path. This process poses that ONE video with run_dlc_batch and writes
the result dict to result_path. The result is written to a temporary name and
then renamed into place, so the parent can never read a half-written result
(WHY: a kill can land at any instant, including mid-write).

Exit code 0 whenever a result was written -- success or failure is in the
result, not the exit code. A non-zero exit means no result could be written
(unreadable args file); the parent reports that as a failed pose.
"""

import json
import os
import sys
import traceback
from pathlib import Path


def _write_result(result_path: Path, result: dict) -> None:
    tmp = result_path.with_name(result_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, result_path)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m mousereach.dlc.core.interruptible_worker <args.json>")
        return 2
    try:
        with open(argv[0], "r", encoding="utf-8") as f:
            args = json.load(f)
        result_path = Path(args["result_path"])
    except Exception as e:
        print(f"[FAIL] cannot read pose arguments {argv[0]}: {e}")
        return 2

    video = str(args.get("video_path", ""))
    try:
        # Imported here, not at the top: this is the expensive import the
        # parent deliberately never pays for.
        from mousereach.dlc.core.batch import run_dlc_batch

        output_dir = args.get("output_dir")
        results = run_dlc_batch(
            video_paths=[Path(video)],
            config_path=Path(args["config_path"]),
            output_dir=Path(output_dir) if output_dir else None,
            gpu=args.get("gpu"),
            save_as_csv=True,
            shuffle=args.get("shuffle"),
        )
        if results:
            result = dict(results[0])
            result.setdefault("video", video)
        else:
            result = {"video": video, "status": "failed",
                      "error": "DLC returned no result for this video"}
    except BaseException as e:  # noqa: BLE001 -- the parent needs to hear about ANY failure
        traceback.print_exc()
        result = {"video": video, "status": "failed",
                  "error": f"{type(e).__name__}: {e}"}

    try:
        _write_result(result_path, result)
    except Exception as e:
        print(f"[FAIL] cannot write pose result {result_path}: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
