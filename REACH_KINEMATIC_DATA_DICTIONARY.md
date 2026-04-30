# Reach Kinematic Data Dictionary

This document defines every column emitted by `mousereach-reach-export` /
`reach_kinematics.csv`. It is the canonical reference for what each value means
and how it is computed.

## Coordinate frame, units, and reach window conventions

* **Reach window.** A reach has a `start_frame`, an `apex_frame`, and an
  `end_frame`, all integer frame indices into the source video. The
  `reach_duration_frames` is `end_frame - start_frame + 1`.
* **Trajectory features are computed in the body-relative frame.** For every
  trajectory feature, the paw point's position is nose-subtracted at every
  frame: `paw_relative = paw - nose`. This isolates paw motion from incidental
  body / head motion.
* **Synthetic slit anchors bracket the trajectory.** Every trajectory feature
  is computed on an "augmented" trajectory that prepends one synthetic frame
  at the reach start and appends one at the reach end:
    * Pre-anchor (nose-relative): `(BoxL - Nose) / 2` evaluated at `start_frame`
      (i.e., the midpoint of `Nose` and `BoxL`, then nose-subtracted).
    * Post-anchor (nose-relative): `(BoxR - Nose) / 2` evaluated at `end_frame`.
  The anchors give a geometrically defined reach boundary that does not depend
  on which frame DLC happened to first/last detect the paw, and they are the
  reference endpoints for path directness and lateral deviation.
* **Image y increases downward.** Reach extension out through the slit moves
  the paw toward the bottom of the frame, i.e., positive y in image and
  nose-relative coordinates. Larger `extension_past_nose` means a paw
  positioned further out through the slit than the nose.
* **Spatial units are reported in two forms:**
    * `_px` and `_px2` — raw pixels and pixel area
    * `_mm` and `_mm2` — millimetres / square millimetres, computed as
      `pixels * 9.0 / ruler_pixels` (the 9 mm calibration ruler corresponds to
      the per-segment `ruler_pixels` distance between SABL and SABR).
* **Temporal units are frames only.** Framerate is not a measured quantity in
  this pipeline; conversion to seconds requires a separately known framerate
  and is the consumer's responsibility.
* **Speed units:** `_px_per_frame` and `_mm_per_frame`. There are no
  `_per_second` columns.
* **Tracking quality is not a gate.** If a reach was emitted by reach
  detection, the paw is considered trackable through the reach window.
  Per-feature likelihood gating is not applied; numerically robust
  accumulation (`np.nansum`, `np.nanmean`, etc.) tolerates occasional NaN
  frames within a tracked reach.

## DLC points referenced

| Name in this document | DLC label | Notes |
|---|---|---|
| paw point: `righthand` | `RightHand` | One of four peripheral paw labels. Anatomical correspondence not validated; use as a stable label, not as anatomy. |
| paw point: `rhleft` | `RHLeft` | Empirically the leftmost paw periphery point (~13 px left of `RightHand` at rest, stable across cohorts and dates). |
| paw point: `rhright` | `RHRight` | Slight upper-right of `RightHand` at rest. |
| paw point: `rhout` | `RHOut` | Slight lower-left of `RightHand` at rest; on the leading edge of the paw under reach extension. |
| Nose | `Nose` | Body-frame reference. |
| BoxL | `BOXL` | Left edge of the slit; static apparatus landmark. |
| BoxR | `BOXR` | Right edge of the slit; static apparatus landmark. |
| Pellet | `Pellet` | Used for tray-contact motion detection. |
| Pillar | `Pillar` | Used for tray-contact motion detection. |
| LeftEar / RightEar | `LeftEar` / `RightEar` | Used for head-width and head-angle posture features. |

The four paw points form a stable but non-symmetric quadrilateral. `RHLeft`
and `RHRight` are empirically the most-separated pair on every video tested
(44/44 across cohorts CNT01-CNT04), which is why the paw width proxy uses
that pair specifically.

---

## Section 1. Identifiers and reach context

| Column | Meaning |
|---|---|
| `video` | Source video name. |
| `segment_num` | Segment index. Segments 2-21 correspond to physical pellets 1-20. |
| `reach_num` | 1-indexed reach number within the segment. |
| `reach_id` | Unique reach id (assigned by reach detection). |
| `is_first_reach` / `is_last_reach` | Booleans: is this reach the first / last in its segment. |
| `n_reaches_in_segment` | Total number of reaches in this segment. |
| `source` | `"algorithm"` or `"human"` (for human-corrected reaches). Only populated by the legacy reaches.json fall-back path. |
| `human_corrected` | Boolean; only populated by the legacy fall-back path. |
| `flagged_for_review` | Boolean from the feature extractor. |
| `flag_reason` | Reason string (if flagged). |

## Section 2. Outcome linkage

| Column | Meaning |
|---|---|
| `outcome` | Segment-level outcome inherited from the outcome detector. |
| `causal_reach` | Boolean: is this the reach that produced the outcome. |
| `interaction_frame` | Frame the pellet interaction occurred (causal reach only; otherwise null). |
| `distance_to_interaction` | `interaction_frame - apex_frame`, in frames. |

## Section 3. Timing

| Column | Meaning |
|---|---|
| `start_frame` / `apex_frame` / `end_frame` | Reach boundary frames (integer indices into the video). |
| `duration_frames` | `end_frame - start_frame + 1`. |

## Section 4. Per-paw-point trajectory features

For each of the four paw points (`righthand`, `rhleft`, `rhright`, `rhout`),
the same set of trajectory features is computed on the augmented, nose-relative
trajectory. The column root below is shown for `righthand`; the same suffixes
apply to `rhleft_*`, `rhright_*`, `rhout_*`.

| Column | Meaning | Calculation |
|---|---|---|
| `righthand_apex_frame` | Frame within the reach when this point's nose-relative y is maximal. May differ between paw points. | `argmax(paw_y - nose_y)` over real reach frames. |
| `righthand_extension_past_nose_px` / `_mm` | How far past the nose this point reached at its own apex. | `paw_y - nose_y` at `righthand_apex_frame`. |
| `righthand_total_path_px` / `_mm` | Total distance this paw point traveled along its actual route. | Sum of frame-to-frame Euclidean distances on the augmented nose-relative trajectory. |
| `righthand_lateral_spread_px` / `_mm` | Side-to-side range covered. | `max(x) - min(x)` on the augmented nose-relative trajectory. |
| `righthand_swept_area_px2` / `_mm2` | 2D area enclosed by the path. | Shoelace formula on the augmented nose-relative trajectory polygon. |
| `righthand_path_directness` | How direct vs winding the path was (dimensionless 0-1, 1 = perfectly straight). | `straight_line_distance(pre_anchor, post_anchor) / total_path`. |
| `righthand_motion_smoothness` | How smooth vs jerky the motion was (dimensionless, 1 = smoothest). | `1 / (1 + mean(|jerk|))`, where jerk is the third difference of the augmented trajectory. |
| `righthand_lateral_deviation_px` / `_mm` | Maximum perpendicular distance from the synthetic-pre-to-synthetic-post line. | `max(|((p - pre) x (post - pre))| / |post - pre|)` over all augmented trajectory points. |
| `righthand_mean_speed_px_per_frame` / `_mm_per_frame` | Average speed across the reach. | `total_path / number_of_inter_frame_segments` (segments include the ones bridging the synthetic anchors). |
| `righthand_peak_speed_px_per_frame` / `_mm_per_frame` | Fastest moment. | `max(frame_to_frame_distance)`. |
| `righthand_apex_speed_px_per_frame` / `_mm_per_frame` | Speed at the per-point apex. | Frame-to-frame distance straddling the per-point apex frame. |

## Section 5. Cross-paw timing

| Column | Meaning | Calculation |
|---|---|---|
| `paw_apex_lead_frames` | How spread out the four paw apexes were. | `max(per-point apex frames) - min(per-point apex frames)`. |
| `paw_leading_point` | Which paw point apexed first. | `argmin` over per-point apex frames. One of `righthand` / `rhleft` / `rhright` / `rhout`. |
| `paw_velocity_correlation` | How synchronized the four paw points' speed profiles were. Dimensionless, in `[-1, 1]`. | Mean of pairwise Pearson r across the four points' frame-to-frame speed series, computed over real reach frames in the nose-relative coordinate frame. |

## Section 6. Inter-paw shape proxies

Three intra-paw geometric measurements, summarised at six time points and as
overall statistics. Operates on real reach frames (no synthetic anchors).

| Measure | Column root | Calculation |
|---|---|---|
| Paw width proxy | `paw_width_proxy` | Distance from `RHLeft` to `RHRight` per frame. The empirically most-separated pair on the paw outline. Foreshortens with paw rotation out of the camera plane. |
| Paw outline area | `paw_outline_area` | Shoelace polygon area of the four paw points per frame. The polygon order is determined per-frame by sorting points by angle around their centroid (handles paw rotation automatically). |
| Paw spread (max pairwise) | `paw_spread_max` | Maximum of all six pairwise distances among the four paw points per frame. A which-points-agnostic "size of paw" measure. |

For each measure, eight summary suffixes:

| Suffix | Calculation |
|---|---|
| `_at_start_*` | Value at `start_frame` (the literal first reach frame). |
| `_at_apex_*` | Value at `apex_frame` (the reach's primary apex, i.e., the wrist's). |
| `_at_end_*` | Value at `end_frame`. |
| `_at_contact_*` | Value at `interaction_frame` (causal reach only; null otherwise). |
| `_max_*` | Maximum across the reach window (`np.nanmax`). |
| `_min_*` | Minimum across the reach window. |
| `_mean_*` | Mean across the reach window. |
| `_range_*` | `max - min`. |

Column units: `_px` / `_mm` for `paw_width_proxy` and `paw_spread_max`;
`_px2` / `_mm2` for `paw_outline_area`. So the full column name pattern is
e.g. `paw_width_proxy_at_apex_mm`, `paw_outline_area_mean_mm2`,
`paw_spread_max_range_px`.

## Section 7. Tray contact

| Column | Meaning | Calculation |
|---|---|---|
| `tray_contact_duration_frames` | Frames within the reach window where pellet/tray motion was detected with the wrist not confidently tracked. Proxy for tray jiggle / paw-tray contact events. | Count of frames in `[start_frame, end_frame]` where the 4-frame rolling std of `Pellet_x` exceeds 2 px AND `RightHand_likelihood < 0.5` AND `Pillar_likelihood < 0.5`. |

This is a single integer per reach: total frames of motion-while-occluded.
Discretization into bouts (event count, longest bout) is not currently emitted.

## Section 8. Hand orientation

| Column | Meaning | Calculation |
|---|---|---|
| `hand_angle_at_apex_deg` | Paw orientation in the image plane at apex. Dimensionless angle. | `atan2(RHRight_y - RHLeft_y, RHRight_x - RHLeft_x)` at `apex_frame`. |
| `hand_rotation_total_deg` | Total in-image-plane rotation across the reach. | Sum of absolute frame-to-frame angle changes (with +/-180 wraparound handling). |

These describe the paw's heading direction in image space — distinct from
pronation, which is captured by `paw_width_proxy` foreshortening.

## Section 9. Body posture (kept from legacy schema)

| Column | Meaning |
|---|---|
| `head_width_at_apex_mm` | Ear-to-ear distance at apex. Size proxy. |
| `nose_to_slit_at_apex_mm` | Nose-to-`BOXR` distance at apex. |
| `head_angle_at_apex_deg` | Head orientation at apex (ear-to-ear angle). |
| `head_angle_change_deg` | Head rotation from reach start to apex. |

## Section 10. Per-paw visibility profile

For each of the four paw points, a likelihood-profile feature group. Likelihood
is dimensionless (DLC outputs values in `[0, 1]`). Reported as "visibility"
because dropouts often carry behavioural signal — paw rotation can occlude a
labelled point, grasp closure can hide multiple points simultaneously,
withdrawal pulls points out of the camera view, and motion blur causes
short-lived dips during fast moments.

For each point, eight columns (shown for `righthand`):

| Column | Meaning |
|---|---|
| `righthand_visibility_at_start` | Likelihood at `start_frame`. |
| `righthand_visibility_at_apex` | Likelihood at `apex_frame`. |
| `righthand_visibility_at_end` | Likelihood at `end_frame`. |
| `righthand_visibility_at_contact` | Likelihood at `interaction_frame` (causal reach only; null otherwise). |
| `righthand_visibility_max` | Best-tracked moment in the reach. |
| `righthand_visibility_min` | Worst dropout moment. |
| `righthand_visibility_mean` | Overall visibility through the reach. |
| `righthand_visibility_range` | `max - min`. |

Plus one aggregate:

| Column | Meaning |
|---|---|
| `frames_any_paw_low_confidence` | Count of frames in the reach where the minimum likelihood across the four paw points was below 0.5. |

Asymmetric dropouts between paw points (e.g., `rhleft_visibility_min` low while
`rhright_visibility_min` high) carry rotation-direction information. Low
visibility is not necessarily bad data; it is data about paw configuration.

## Deprecated columns retained for backward compatibility

These columns appear in the output but are superseded by the extended-feature
equivalents above. They will be dropped in a future cleanup once downstream
consumers migrate.

| Column | Successor |
|---|---|
| `max_extent_pixels` / `_ruler` / `_mm` | Per-paw `*_extension_past_nose_*`. |
| `mean_likelihood`, `frames_low_confidence` | Per-paw visibility profile and `frames_any_paw_low_confidence`. |

---

## Versioning

Schema version: 2.0 (extended kinematic feature set).

Bump the schema version when columns are added, removed, or have their
calculation changed. Document the change in this file in a changelog section.
