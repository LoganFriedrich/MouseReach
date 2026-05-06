# Historical Video Reprocessing Plan

## Goal

Reprocess ALL historical behavior videos through the current MouseReach pipeline
(multi-animal DLC model + latest segmentation/reach/outcome detection) so that
kinematic analyses pool data from a single, consistent processing version.

**This is a data-copying operation, not a replacement.** Old results are preserved
intact in a legacy archive. Only the raw videos are copied forward into the
current pipeline for reprocessing.

---

## Background

### Why This Is Needed

- Historical cohorts (CNT_01 through early CNT_03) were processed with ASPA
  (the old single-animal pipeline) using an older DLC model.
- The current MouseReach pipeline uses a multi-animal DLC model with improved
  segmentation, reach detection, and outcome classification.
- Pooling old-pipeline and new-pipeline kinematics contaminates analyses because
  the measurement tool changed, not just the biology.
- To do proper cross-cohort kinematic analysis (PCA, LASSO, recovery stratification),
  ALL data must come from the same processing version.

### Current Infrastructure

- **Version tracking** already exists: `pipeline_versions.json`, per-video
  `_processing_manifest.json`, `ReprocessingScanner`
- **Distributed compute** already exists: DLCLabPC, Vid&DLC1PC, Vid&DLC2PC (all
  have CUDA GPUs), Processing Server (no GPU -- seg/reach/outcomes only)
- **Watcher system** already exists: `mousereach-watch` with marker-file claiming
- **Crystallization** already exists: `mousereach-crystallize` to lock published data

### Compute Constraints

| Machine | GPU | Can Run DLC | Role in Reprocessing |
|---------|-----|-------------|---------------------|
| Processing Server (this PC) | AMD (no CUDA) | NO | Seg/reach/outcomes only, watcher coordination |
| DLCLabPC | CUDA | YES | DLC + full pipeline |
| Vid&DLC1PC | CUDA | YES | DLC + full pipeline |
| Vid&DLC2PC | CUDA | YES | DLC + full pipeline |

---

## Plan

### Phase 1: Archive Legacy Data

**Goal:** Preserve all ASPA-processed results in a clearly labeled archive.

1. Create `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Archive_ASPA\`
2. For each historical cohort processed with the old pipeline:
   - Copy (not move) the entire processed output tree into `Archive_ASPA/{cohort}/`
   - Include: DLC output, segmentation, reach data, outcome data, manifests
   - Do NOT copy raw videos (they stay in place or get copied forward)
3. Add a `README.md` in `Archive_ASPA/` explaining:
   - What pipeline version produced this data
   - When it was archived
   - That it should NOT be used for cross-cohort kinematic analysis
   - That it IS the canonical record for any publications that used ASPA data

### Phase 2: Identify and Prepare Historical Videos

**Goal:** Find all raw multi-animal videos and stage them for reprocessing.

1. Inventory all historical video files:
   - Locations: original filming directories, NAS archives
   - Formats: may include old naming conventions (pre-MouseReach standard)
   - Types: multi-animal collage videos (the pipeline input format)
2. Catalog naming convention differences:
   - Current format: `YYYYMMDD_CNTxxyy_T#_*.avi` (or similar)
   - Old formats: document what they look like, map to current conventions
3. Write a cleaning/renaming script (or mapping file) that:
   - Maps old filenames to the metadata the pipeline expects
   - Does NOT rename the originals -- creates symlinks or a lookup table
   - Handles edge cases (missing metadata, unusual naming)
4. Copy (not move) the raw videos to the current pipeline input staging area:
   - `Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing\` (or wherever
     the watchers expect new input)
   - Preserve original paths in a manifest for provenance

### Phase 3: Watcher Adjustments

**Goal:** Make the watcher system handle old-format videos gracefully.

1. **Processing Server (this PC):**
   - Cannot run DLC (no GPU)
   - Watcher changes: recognize old-format video names in the staging area
   - Route them to `DLC_Complete/` queue for GPU nodes (skip local DLC step)
   - Wait -- actually this PC can't do DLC at all, so it should:
     - Stage videos for GPU nodes to pick up
     - Handle post-DLC processing (seg/reach/outcomes) once DLC is done

2. **GPU Nodes (DLCLabPC, Vid&DLC1PC, Vid&DLC2PC):**
   - Install latest MouseReach version (ensure `pipeline_versions.json` is current)
   - Watcher must recognize old naming conventions:
     - Parse subject ID, cohort, session date from old format
     - Map to standard internal representation
   - Process: DLC (new multi-animal model) -> seg -> reach -> outcomes
   - Archive results back to `Y:` as usual

3. **Naming convention handler:**
   - Add a `legacy_name_parser.py` module (or extend existing parser)
   - Input: old-format filename
   - Output: standard metadata dict (subject_id, cohort, session_date, tray_type)
   - Include a lookup table for known historical naming patterns
   - Fall back gracefully (flag for manual review if unparseable)

### Phase 4: Trigger Reprocessing

**Goal:** Run all historical videos through the current pipeline on GPU nodes.

1. Verify all GPU nodes have:
   - Latest MouseReach installed
   - Latest DLC model downloaded
   - Correct `config.json` pointing to Y: NAS
2. Stage historical videos in the pipeline input area
3. Let the existing watcher system pick them up via marker-file claiming
4. Monitor progress via `mousereach-watch-status`
5. As results come in, verify:
   - DLC tracking quality (spot-check likelihood scores)
   - Reach detection counts (compare to ASPA for sanity)
   - Outcome classification (spot-check a sample)

### Phase 5: Validation

**Goal:** Confirm reprocessed data is usable and consistent.

1. Compare reprocessed vs ASPA results for a sample of videos:
   - Number of reaches detected
   - Outcome distribution
   - Key kinematic ranges (max extent, peak velocity)
   - Document expected differences (new model may find more/fewer reaches)
2. Run `mousereach-version-check` to confirm all videos now show current version
3. Update `connectome.db` with reprocessed reach data (reimport)
4. Re-export `reach_data.csv` to `database_dump/`

---

## Key Principles

- **COPY, never replace.** Old data stays in `Archive_ASPA/`. Raw videos are copied
  to the pipeline input area. Nothing is deleted.
- **Provenance is mandatory.** Every reprocessed video gets a `_processing_manifest.json`
  recording the new pipeline version, timestamp, and source video path.
- **GPU nodes do DLC.** This PC (Processing Server) handles only post-DLC steps.
- **Old naming conventions are a data problem, not a code problem.** Solve with a
  mapping/parser, not by renaming files.

---

## Open Questions (For the Chat That Picks This Up)

1. Where exactly are all the historical raw videos stored? (Y: archive paths, Drobo?)
2. What are the specific old naming conventions? (Need examples to write the parser)
3. Are there any historical videos that are single-animal (not collage format)?
   If so, they may need a different preprocessing step before DLC.
4. Should we crystallize the ASPA archive before starting? (Prevents accidental
   reprocessing of the archive itself)
5. How much disk space is needed for the video copies? (Estimate: N videos x avg size)
6. Priority order for cohorts? (Most recent first, or oldest first?)

---

## Timeline Estimate

- Phase 1 (Archive): ~1 session (mostly copying)
- Phase 2 (Inventory + naming): ~1-2 sessions (depends on naming complexity)
- Phase 3 (Watcher changes): ~1 session (code changes to MouseReach)
- Phase 4 (Reprocessing): Days to weeks (depends on video count and GPU availability)
- Phase 5 (Validation): ~1 session after reprocessing completes
