# Watcher database: what every field means

Verified against: 05de530 (2026-09-14), plus the GPU-node guards of 2026-09-15
(collage claims, finished-or-held children, Processing/Posed; see
PIPELINE_AS_BUILT.md, "Update 2026-09-15")

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
the full schema and zero rows; it is not the live one.
`resolve_watcher_db_path` (watcher/db_location.py) is the only correct way to
pick it. The CLI's `_resolve_db_path` calls it and prints its choice, and
`mousereach-route-to-queue` uses it too, refusing to route when the file is
missing (2026-09-14: that command used to open the zero-row decoy, so every
state it set was lost).

  nas_root/watcher_state/<hostname>/watcher.db   BACKUP, NOT STATE
      A plain copy of one node's live database (watcher/coordination.py:175),
      written opportunistically inside the work loop, not on a timer. Read by
      exactly one caller, restore_db (:208), and only when that node's own
      database is missing or under 4 KB. Its timestamp says when that node last
      finished a cycle, not when its contents were true. Do not read it to
      answer questions about the pipeline.

  nas_root/watcher_central.db                    TWO KINDS OF TABLE IN ONE FILE
      Tables `videos` and `processing_log`: AUDIT LOG, NOT STATE.
      Provenance written after each archive (db.py, export to central). Every
      row in `videos` is 'archived' by construction, so "this video is archived
      here" carries no information about its current state. Has columns the
      per-node schema does not (source_machine, exported_at).

      Tables `pipeline_collages` and `pipeline_videos`: LIVE CROSS-NODE
      COORDINATION between GPU nodes (watcher/coordination.py). Only
      DLCOrchestrator writes or reads them; the processing server never does.

      pipeline_collages   one row per collage a GPU node has claimed for cropping
        filename          the collage file name (PRIMARY KEY: one claim per collage)
        hostname          the node holding the claim
        state             'cropping' = claimed, crop not finished;
                          'cropped'  = the holder finished the crop (or found every
                                       child already finished or held)
        claimed_at        when the holder took or last re-used the claim, on the
                          holder's clock
        completed_at, singles_created
      Rules, each with its reason (full account in PIPELINE_AS_BUILT.md,
      "Update 2026-09-15"):
        * No claim, no crop. A node that cannot reach this table crops nothing.
          WHY: without a claim two GPU nodes crop the same collage and pose
          every child twice (~14 GPU-minutes each).
        * A 'cropping' row another host has held for more than 24 h
          (COLLAGE_CLAIM_STALE_S) is taken over -- but NOT while any child of
          that collage has a row in pipeline_videos. WHY: a stale claim does not
          mean nothing was cropped; its children may be queued on the holder,
          in Processing/Posed or on the processing server.
        * The holder deletes its row after a failed crop, unless it still
          carries a child of the collage. WHY: a claim that outlives a failed
          crop blocked that collage for every node.
        * A 'cropped' row is never taken over or released.

      pipeline_videos     one row per video a GPU node has synced
        video_id, collage_id, hostname (last node to write it), state,
        source_path, nas_path, discovered_at, dlc_completed_at, processed_at,
        staged_at, updated_at, error_message
      Written best-effort by GPU nodes as a video moves: 'dlc_queued' the
      moment a cropped child is queued (since 2026-09-15; before that the
      first write was 'dlc_running'), then 'dlc_running', 'dlc_complete',
      'processed', 'archived'. Read at a GPU node's startup by cross-node
      recovery, and by the claim takeover check above. A GPU node's 'archived'
      means that node staged or filed it, not that the analysis is finished.

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

  On a GPU node, a row can also be 'archived', 'triage' or 'deep_review' with
  source_path '(no file on this node)' for a video this node never cropped or
  posed. Since 2026-09-15 a GPU node checks the shared drive before it poses a
  collage child or a dropped single: an archive manifest means 'archived', a
  bundle in a review queue (or its .incoming build folder) means that queue's
  state, and the row is recorded that way instead of being queued for DLC.
  The processing log says why (step 'crop', 'adopt' or 'dlc', status 'skipped').

  collages.state 'cropped' on a GPU node has three meanings. Read the collage's
  processing log to tell them apart:
    * cropped HERE: step 'crop', status 'completed'; videos_created = the
      children this node queued.
    * cropped by ANOTHER node: step 'claim', status 'skipped', naming the node.
      This node has no children of it.
    * NOT cropped at all, because every child was already finished or held:
      step 'crop', status 'skipped'; videos_created = 0.
  A collage another node is cropping RIGHT NOW stays 'stable' here. This node
  skips it for 30 minutes at a time, then asks the shared claim table again.
  The wait is kept in memory, so it does not show in this table.

  'outdated' means the version scanner found a mismatch, NOT that the work is
  bad or missing. Read reprocess_scope to learn what is actually stale, and the
  manifest in Analyzed/ to learn what actually ran.

  'unresolvable' means this node has no file for this video. Nothing went wrong
  with the data. It is deliberately not 'failed', which is a retry state and
  which reads to people as a verdict about the animal's data. One case added
  2026-09-15: a GPU node's stage step finds the video already in
  Processing/Posed and nothing of it left locally ("already in NAS staging ...
  left for the processing server"). This node may well have staged it and been
  stopped before recording that; files in Posed are never taken as proof, so
  the row is not 'archived' either.

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
