# Watcher database: what every field means

Verified against: 412ad28 (2026-09-14)

WHY THIS EXISTS

Knowing how the pipeline works should never require reading the source or
asking a person. It did. A check of whether roughly 1,100 finished CNT videos
needed re-posing went wrong for most of a night because three things were
written down nowhere: which file on disk is live state, what `reprocess_scope`
means, and that `archived` does not mean "fully analyzed". All three are
one-line facts. This file is those lines.

A dictionary cannot tell you the pipeline is CORRECT, only what its fields are
called and what they hold. Behaviour lives in PIPELINE_AS_BUILT.md.


## 1. WHICH FILE IS THE DATABASE

A node's live database is the `db_path` override in that machine's
`~/.mousereach/config.json`, else `processing_root/watcher.db`. On a machine
with the override set, a file named `watcher.db` may also sit beside it holding
the full schema and zero rows; it is not the live one. `_resolve_db_path`
(watcher/cli.py) is the only correct way to pick it, and it prints its choice.

  nas_root/watcher_state/<hostname>/watcher.db   BACKUP, NOT STATE
      A plain copy of one node's live database (watcher/coordination.py:175),
      written opportunistically inside the work loop, not on a timer. Read by
      exactly one caller, restore_db (:208), and only when that node's own
      database is missing or under 4 KB. Its timestamp says when that node last
      finished a cycle, not when its contents were true. Do not read it to
      answer questions about the pipeline.

  nas_root/watcher_central.db                    AUDIT LOG, NOT STATE
      Provenance written after each archive. Every row in it is 'archived' by
      construction, so "this video is archived here" carries no information
      about its current state. Has columns the per-node schema does not
      (source_machine, exported_at).

For reprocessing questions the processing server's live database decides. It is
the only role with `reprocesses_partial` true (orchestrator.py:433); GPU nodes
set it false (:728), pass `full_only=True`, and therefore mark only the scope
they can drain. `handles_reprocessing` does NOT separate the roles: the GPU
role sets it from `also_process` (:756), true on all three lab GPU machines.


## 2. TABLE: videos

Identity and provenance

  id                      autoincrement row id; not the video's identity
  video_id                the identity: YYYYMMDD_ANIMALID_TRAY, for example
                          20250316_CNT0315_P3. UNIQUE. Two files with this same
                          stem are the same video.
  collage_id              filename of the multi-animal video this was cut from
  date, animal_id, experiment, cohort, subject, tray_type, tray_position
                          parsed out of video_id; see config.AnimalID and
                          config.FilePatterns. tray_type P=Pillar (the only
                          supported one), E=Easy, F=Flat.

Where the files are

  source_path             where it was first registered from. NOT NULL.
  current_path            where this NODE last put it. A recorded path is a
                          claim, not a fact. Resolve with watcher/locate.py,
                          which searches working folders first and the archive
                          last, rather than trusting it.
  dlc_output_path         the pose file, same caveat.

State

  state                   see section 4. NOT NULL, default 'discovered'.
  claimed_by              hostname holding a multi-machine claim, else NULL
  error_message           why it failed, or for 'unresolvable' the owning host
  error_count             retries so far; not incremented for 'unresolvable'
  last_error_at

Timestamps (ISO strings, on the writing machine's clock)

  discovered_at (NOT NULL), validated_at, crop_started_at, crop_completed_at,
  dlc_started_at, dlc_completed_at, processing_started_at,
  processing_completed_at, archived_at, created_at, updated_at

  archived_at is when the video was filed. It is NOT when a reprocess_scope was
  assigned; nothing records that, so a scope cannot be dated from the row.

Version tracking (added by migration, db.py:331)

  pipeline_versions_hash  may be NULL even on current videos; the manifest
                          beside the video in Analyzed/ is the real provenance
  reprocess_scope         which stage to re-run, set when state='outdated':
                            'full'          needs a NEW pose on a GPU. Since
                                            2026-08-24 this means only "no pose
                                            from the declared model exists
                                            anywhere in the archive".
                            'segmentation'  re-run every post-DLC stage against
                                            a pose already on disk
                            'reach', 'outcome', 'kinematics'
                                            re-run from that stage onward
                          NULL when not outdated. A leftover non-NULL value on
                          an 'archived' row is stale and means nothing.
  mark_reason             NULL for a scanner mark; set for a HAND-mark. The
                          reprocessor's two-way door never un-marks a row
                          carrying a reason. That is the whole point of the
                          column: nothing else distinguishes "a person said
                          re-run this" from "the scanner noticed a version".
  crystallized_at, crystallized_by, crystallized_label
                          locked against reprocessing; use force_state to
                          unlock


## 3. TABLE: collages, and TABLE: processing_log

collages

  id, filename (UNIQUE), source_path (NOT NULL), state, date
  animal_ids              comma-separated list of the mice inside
  tray_suffix, file_size
  first_seen_at (NOT NULL), last_size_change_at, stable_since
                          a collage is 'stable' once it stops growing; that is
                          how a still-copying file is kept out of the pipeline
  validation_error, crop_started_at, crop_completed_at
  videos_created, videos_skipped, archived_at, archive_path
  retry_count             added by migration
  created_at, updated_at

processing_log: append-only, one row per step attempt

  id, video_id (NOT NULL), step (NOT NULL), status (NOT NULL), message,
  duration_seconds, machine, created_at


## 4. STATE VOCABULARY

videos.state, with the legal next states (db.VIDEO_TRANSITIONS):

  discovered    -> validated, quarantined
  quarantined   -> validated            filename failed the hard gate
  validated     -> dlc_queued, unresolvable
  dlc_queued    -> dlc_running, unresolvable
  dlc_running   -> dlc_complete, dlc_queued, failed, unresolvable
  dlc_complete  -> processing, archived, unresolvable
  processing    -> processed, failed, triage, deep_review, unresolvable
  processed     -> archiving, unresolvable
  archiving     -> archived, failed, unresolvable
  archived      -> outdated, crystallized, unresolvable
  outdated      -> dlc_queued, processing, failed, archived, unresolvable
  crystallized  -> nothing; force_state to unlock
  triage        -> outdated, processing, deep_review, failed
  deep_review   -> outdated, processing, dlc_queued, failed
  failed        -> validated, dlc_queued, dlc_complete, processing, processed,
                   unresolvable
  unresolvable  -> validated, dlc_queued, dlc_complete, processing, processed,
                   archived

READ THIS BEFORE INTERPRETING A STATE

  'archived' means THIS NODE is done with it, not "fully analyzed". A GPU
  machine running also_process=false marks 'archived' the moment it has staged
  the pose; the algorithms have not run yet. The collages table uses the same
  idiom for 'cropped'. This is the single most misread value in the schema.

  'outdated' means the version scanner found a mismatch, NOT that the work is
  bad or missing. Read reprocess_scope to learn what is actually stale, and the
  manifest in Analyzed/ to learn what actually ran.

  'unresolvable' means this node has no file for this video. Nothing went wrong
  with the data. It is deliberately not 'failed', which is a retry state and
  which reads to people as a verdict about the animal's data.

  'triage' and 'deep_review' are human-review holds. Kinematics never run on a
  held video, and it stays out of the archive and the central database until a
  person clears it.

collages.state (db.COLLAGE_TRANSITIONS):

  discovered -> validated, quarantined
  quarantined -> validated
  validated -> stable
  stable -> cropping
  cropping -> cropped, failed
  cropped -> archived
  failed -> validated


## 5. WHAT IS NOT IN HERE

The per-video `_processing_manifest.json` beside the video in
Analyzed/{project}/{cohort}/ is the authority on what actually produced a
video's results: the pose model, every stage version, and the applied_review
stamp. The database records where a video is in the pipeline; the manifest
records what was done to it. When the two disagree, the manifest is the one
that was written by the code that did the work.
