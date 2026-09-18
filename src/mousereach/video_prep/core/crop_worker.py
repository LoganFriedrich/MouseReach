"""Child-process side of an interruptible crop.

Run as:  python -m mousereach.video_prep.core.crop_worker <args.json>

WHY A SEPARATE PROCESS: cropping a collage runs ffmpeg once per animal and
takes minutes, and while it runs nothing can stop it -- so an operator who
opened the recording program had to wait for it. Running the crop here lets the
parent (mousereach.video_prep.core.crop_interruptible) kill this process and
everything it started the moment a recording program appears. The interrupted
crop is thrown away and the collage goes back in the queue; cropping again is
cheap, a lost recording is not.

The args file holds: input_path, output_dir, result_path. This process crops
that ONE collage with crop_collage and writes {"collage", "status", "results"}
to result_path, written to a temporary name then renamed into place so the
parent can never read a half-written result.

Exit code 0 whenever a result was written -- success or failure is in the
result, not the exit code.
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
        print("usage: python -m mousereach.video_prep.core.crop_worker <args.json>")
        return 2
    try:
        with open(argv[0], "r", encoding="utf-8") as f:
            args = json.load(f)
        result_path = Path(args["result_path"])
    except Exception as e:
        print(f"[FAIL] cannot read crop arguments {argv[0]}: {e}")
        return 2

    collage = str(args.get("input_path", ""))
    try:
        # Imported here, not at the top: the parent must not pay for ffmpeg's
        # module imports just to start a child.
        from mousereach.video_prep.core.cropper import crop_collage

        results = crop_collage(input_path=Path(collage),
                               output_dir=Path(args["output_dir"]),
                               verbose=False)
        result = {"collage": collage, "status": "success", "results": list(results or [])}
    except BaseException as e:  # noqa: BLE001 -- the parent needs to hear about ANY failure
        traceback.print_exc()
        result = {"collage": collage, "status": "failed",
                  "error": f"{type(e).__name__}: {e}"}

    try:
        _write_result(result_path, result)
    except Exception as e:
        print(f"[FAIL] cannot write crop result {result_path}: {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
