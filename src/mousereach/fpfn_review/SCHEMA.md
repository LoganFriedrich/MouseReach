# FP/FN Reach Review — manifest schema

This widget consumes a per-video JSON describing the detector's TP / FP / FN
reaches against ground truth. One JSON per video.

Manifests are produced by whatever evaluation harness compares your detector
output to ground truth: group its per-event records by `video_id` and emit one
JSON per video in the shape below (a worked transform is at the bottom).

## Per-video JSON

```json
{
  "video_id": "20240301_ANIMAL01_S1",
  "video_path": "/data/videos/20240301_ANIMAL01_S1.mp4",
  "n_frames": 18000,
  "fps": 60,
  "detector_version": "v8.0.0",
  "snapshot": "detector_run_2024-03-01",
  "corpus": "validation_set_a",
  "matching_criterion": "strict_start2_span",
  "events": [
    {
      "kind": "TP",
      "detector": {"start": 1200, "end": 1245},
      "gt":       {"start": 1198, "end": 1250},
      "category": null,
      "start_delta": 2,
      "span_delta": -7
    },
    {
      "kind": "FP",
      "detector": {"start": 1450, "end": 1490},
      "gt": null,
      "category": "over_extends_end"
    },
    {
      "kind": "FN",
      "detector": null,
      "gt": {"start": 2000, "end": 2030},
      "category": "miss"
    }
  ]
}
```

## Field semantics

- **`video_id`** — string. Required. Used in widget header.
- **`video_path`** — string (absolute path, or any path resolvable on the
  machine running the widget). Optional. If present, widget auto-loads the
  video; otherwise user is prompted to pick it.
- **`n_frames`**, **`fps`** — integers. Optional but useful for the widget's
  header line. If absent, widget reads them from the loaded video.
- **`detector_version`** — string. Required. Shown in header so the user can
  tell at a glance which algorithm version produced these errors. Example
  values: `"v8.0.0"`, `"v7.2.0"`.
- **`snapshot`** — string. Optional. Name of the evaluation run the manifest
  was generated from, for traceability.
- **`corpus`** — string. Required. Free-form label for the video set this
  manifest belongs to (e.g. `"validation_set_a"`, `"holdout"`). Drives the
  corpus label in the widget, and keys ground-truth auto-resolution: add a
  `"fpfn_gt_roots": {"<corpus label>": "<GT directory>"}` map to
  `~/.mousereach/config.json` and the widget resolves
  `<GT directory>/<video_id>_unified_ground_truth.json` automatically on
  manifest load. Unconfigured corpora fall back to the Load GT button.
- **`matching_criterion`** — string. Required. Documents which match rule
  was used to classify events. Common values: `"strict_start2_span"` (v8's
  current rule: start_tol=2 AND span tolerance), `"permissive_window10"`
  (v7.1.0 style: ±10 frame window, no span check).
- **`events`** — array. Required. One entry per reach event.

### Event object

Required fields:

- **`kind`** — `"TP"` | `"FP"` | `"FN"`.
- **`detector`** — `{start, end}` integer frame indices for the algorithm's
  reach, or `null` for FN.
- **`gt`** — `{start, end}` integer frame indices for the ground-truth
  reach, or `null` for FP.

Optional fields (rendered as columns / tooltips if present):

- **`category`** — string. Failure-mode label. Suggested values:
  - For FP: `"over_extends_end"`, `"over_extends_start"`, `"within_gt"`,
    `"split_twin"`, `"phantom"` (entirely outside any GT reach),
    `"tolerance_miss"` (within-GT but failed span tolerance).
  - For FN: `"miss"` (model never fired), `"tolerance_miss_start"`,
    `"tolerance_miss_span"`, `"merged"` (model fired one reach where GT had
    two).
  - For TP: usually `null`.
  - The widget treats `category` as a free-form string; new values can be
    introduced without widget changes.
- **`start_delta`** — integer. detector_start − gt_start. TP only.
- **`span_delta`** — integer. detector_span − gt_span. TP only.

## Building manifests from your matcher's output

If your evaluation harness emits per-event records with fields like `status`
(`"TP"`/`"FP"`/`"FN"`), `video_id`, `algo_start_frame`, `algo_end_frame`,
`gt_start_frame`, `gt_end_frame`, `start_delta`, `span_delta`, transform per
record:

```python
event = {
    "kind": rec["status"],   # already "TP"/"FP"/"FN"
    "detector": {"start": rec["algo_start_frame"], "end": rec["algo_end_frame"]}
                if rec.get("algo_start_frame") is not None else None,
    "gt": {"start": rec["gt_start_frame"], "end": rec["gt_end_frame"]}
          if rec.get("gt_start_frame") is not None else None,
    "category": rec.get("category"),
}
if rec["status"] == "TP":
    event["start_delta"] = rec.get("start_delta")
    event["span_delta"] = rec.get("span_delta")
```

Group records by `video_id`, then emit one manifest per video. Place output
anywhere the widget can browse to; pairing each corpus label with a GT
directory in `fpfn_gt_roots` (see `corpus` above) makes GT loading automatic.

## Versioning

If the schema evolves, bump the snapshot's filename suffix or add a
`schema_version` top-level field. The widget tolerates extra fields, so
additive changes don't break it.
