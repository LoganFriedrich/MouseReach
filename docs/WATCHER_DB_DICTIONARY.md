# Watcher database: what every field means

Verified against: 05de530 (2026-09-14), plus the GPU-node guards of 2026-09-15
(collage claims, finished-or-held children, Processing/Posed; see
PIPELINE_AS_BUILT.md, "Update 2026-09-15")

WHY THIS EXISTS

Knowing how the pipeline works should never require reading the source or
asking a person. It did. A check of whether finished videos needed re-posing
once went wrong because three things were written down nowhere: which file on disk is live state, what `reprocess_scope`
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
      An export that fails (locked, busy or unreachable share) is only a warning.
      The processing server re-sends, at start and then every 30 minutes, every
      archived video whose row from IT (source_machine = its name) is missing or
      older than its own row, at most 50 per pass ("Sent N archived video(s) to
      the shared record ..."). Before 2026-10-26 this ran only at start and only
      for a video with no row from any machine, so a video a GPU PC had posed --
      which has that PC's row -- was never re-sent.

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
          every child twice.
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
      (A GPU PC or the CPU pose pool writes 'archived' here when it HANDS the
      pose on to Processing/Posed. Only the 'videos' table above, written at
      filing, means filed. mousereach-session-coverage reads it that way: an
      'archived' here is reported by where the pose is -- "taken in for
      analysis", "posed, waiting for analysis", or "handed on; not in Posed,
      review or Analyzed" -- never as "analysed".)
      Recovery moves a node's own row forward to match, but never replaces a
      'triage' or 'deep_review' row with a record written before it (since
      2026-10-22; before, an older 'archived' here cleared a newer hold).
      A working-state row is adopted only when the video's file is on this
      node; a source_path in ANOTHER machine's Single_Animal/.inflight/<host>/
      claim folder does not count, although the shared drive makes it exist
      everywhere (since 2026-10-28; before, a restarting GPU node adopted and
      re-posed every video the CPU pose pool was holding).

For reprocessing questions the processing server's live database decides. It is
the only role with `reprocesses_partial` true (orchestrator.py:433); GPU nodes
set it false (:728), pass `full_only=True`, and therefore mark only the scope
they can drain. `handles_reprocessing` does NOT separate the roles: the GPU
role sets it from `also_process` (:756).


## 2. TABLE: videos

Identity and provenance

  id                      autoincrement row id; not the video's identity
  video_id                the identity: YYYYMMDD_ANIMALID_TRAY, for example
                          20250101_CNT0101_P1. UNIQUE. Two files with this same
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
                          (the CURRENT error: reaching 'archived' clears it, and since
                          2026-10-22 each watcher start clears it on archived rows;
                          the history is in processing_log). One exception: an
                          'archived' row whose message starts "re-run of a filed
                          video failed" is a filed video whose re-run failed for
                          good; its filed results stand. That message is kept
                          across restarts and cleared when the video is filed
                          again (the processing watcher lists these rows in
                          <NAS root>/attention/<host>_failed_reruns.json).
                          On an 'outdated' row a message starting "re-run
                          could not start (attempt N)" means its files could
                          not be copied down; it is tried again an hour after
                          last_error_at, and the message clears when the
                          re-run starts.
  error_count             failures since the last reset; not incremented for
                          'unresolvable'. A person's reset
                          (mousereach-watch-reprocess) sets it back to 0 and
                          writes the old count to processing_log (step
                          'reset'), so the automatic retries apply again.
                          Without --from-step the reset reads its restart
                          step from this node's files (db.py
                          infer_reset_state); a single whose copy into the
                          DLC queue failed (error starts "could not copy it
                          from the shared singles folder") and that is back
                          at the top of the shared singles folder is reset
                          to 'validated' with its path set to that file.
  last_error_at

Timestamps (ISO strings, on the writing machine's clock)

  discovered_at (NOT NULL), validated_at, crop_started_at, crop_completed_at,
  dlc_started_at, dlc_completed_at, processing_started_at,
  processing_completed_at, archived_at, created_at, updated_at

  archived_at is when the video was filed. It is NOT when a reprocess_scope was
  assigned; nothing records that, so a scope cannot be dated from the row.

Version tracking (added by migration, db.py:331)

  pipeline_versions_hash  may be NULL even on current videos; the manifest
                          beside the video in Analyzed/ is the real provenance.
                          Written in one case: on an 'archived' row whose
                          re-run failed for good (error_message starts "re-run
                          of a filed video failed"), the fingerprint of what
                          that re-run was attempted against -- the declared
                          versions, the saved review, the filed ground truth
                          (watcher/reprocessor.py failed_rerun_stamp). The
                          version scan does not mark such a row again while
                          the fingerprint is unchanged.
  reprocess_scope         which stage to re-run, set when state='outdated':
                            'full'          needs a NEW pose on a GPU. Since
                                            2026-08-24 this means only "no pose
                                            from the declared model exists
                                            anywhere in the archive".
                            'segmentation'  re-run every post-DLC stage against
                                            a pose already on disk
                            'reach', 'outcome', 'kinematics'
                                            re-run from that stage onward
                            A 'kinematics' re-run of a filed video
                            (archived_at set) reuses every algorithm output
                            and skips the quality triage and the hand-score
                            check; the review gate still runs and asks only
                            about segments no person answered (since
                            2026-11-03; PIPELINE_AS_BUILT.md).
                          The scope stays on the row while the re-run is under
                          way. A 'kinematics' scope on a filed row is read back
                          when the run is picked without a start stage (the
                          failed-video retry, a watcher restart), so the retry
                          is kinematics-only too. A run that has to remake an
                          algorithm output after all changes 'kinematics' to
                          the stage it remakes (assignment counts as
                          'outcome') before remaking it, so a later pick runs
                          every check. It is cleared when the video
                          is filed, when a review queue returns it, when the
                          gate finds stale outputs, and whenever the row enters
                          a state other than 'outdated', 'processing',
                          'failed', 'processed' or 'archiving' (db.py
                          REPROCESS_SCOPE_STATES), so it never outlives its
                          re-run. A scope is also carried only BETWEEN those
                          five states: a row that enters one of them from any
                          other state (a review queue, 'unresolvable',
                          'archived' ...) starts with no scope unless the
                          caller gives one with the move -- update_state,
                          force_state, mark_failed, mark_unresolvable and
                          reset_failed all keep this rule. A value on a row
                          in any other state is a leftover from older code.
                          It cannot be read back (the rule above), and it is
                          cleared the next time any program opens the
                          database (WatcherDB._clear_scopes_outside_reruns).
                          The clearing cannot be undone from the database, so
                          BEFORE it happens the log of the program that
                          opened the database gets: how many rows are being
                          cleared, the count per state, and every row as
                          "<video> (<state>) = <scope>" -- enough to put a
                          value back by hand. Only the rows named in the log
                          are cleared, and a last line says how many were.
                          If the database is locked at that moment the
                          leftovers simply wait for the next open.
  held_at_step            which step sent the video to a review queue, written
                          when it is routed (review_gate.route_to_queue):
                          'segmentation' (every route into deep review),
                          'outcome' (the gate's triage), 'assignment' (triage
                          because reach assignment could not run). Also in the
                          bundle's manifest (provenance.held_at_step). Kept
                          while the video waits in the queue and through the
                          run its return starts; cleared when the video is
                          filed and whenever the row enters a state that is
                          neither in that run (resume_stage, below) nor a
                          review hold ('triage', 'deep_review'; db.py
                          HELD_AT_STEP_STATES), unless the caller sets it. A
                          new hold writes its own; a row set to its queue's
                          state because its bundle was found there keeps the
                          step its route recorded. NULL on rows held before
                          this column existed.
  resume_stage            where the run of a video a review queue returned
                          resumes: 'reach', 'assignment' or 'checks' (the
                          step after held_at_step; deep review always 'reach').
                          Written by the return, read by the pipeline (which
                          starts earlier if an earlier output is out of date;
                          watcher/resume_point.py). Kept through a retry or a
                          restart: only in 'processing', 'failed', 'processed'
                          and 'archiving' (db.py RESUME_POINT_STATES). Cleared
                          by a new hold, by filing, by a stale-outputs gate,
                          when a version re-run starts, and whenever the row
                          enters any other state unless the caller sets it --
                          update_state, force_state and reset_failed all do
                          this, so a person's reset, a send for a new pose,
                          'unresolvable' or 'outdated' ends the resume point
                          and the next run is a fresh one with every step.
                          held_at_step and resume_stage were added after the
                          first release. A database that was locked when a
                          program opened it may not have them yet (the log
                          says "column(s) ... could not be added"); they are
                          added again at that program's next state change,
                          and until then state changes go through without
                          them.
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
  archiving     -> archived, processed, failed, unresolvable
  archived      -> outdated, crystallized, unresolvable
  outdated      -> dlc_queued, processing, failed, archived, unresolvable
  crystallized  -> nothing; only mousereach-uncrystallize unlocks it (since
                   2026-11-15 force_state, mark_failed and mark_unresolvable
                   refuse it, and no
                   route, return or divert moves a crystallized video)
  triage        -> outdated, processing, deep_review, failed
  deep_review   -> outdated, processing, dlc_queued, failed
  failed        -> validated, dlc_queued, dlc_complete, processing, processed,
                   unresolvable
  unresolvable  -> validated, dlc_queued, dlc_complete, processing, processed,
                   archived

READ THIS BEFORE INTERPRETING A STATE

  Every state change (update_state, force_state, mark_failed,
  mark_unresolvable, reset_failed) reads the row and writes its new state
  under one write lock (db.WatcherDB._state_write). Two programs changing
  the same row at once no longer overwrite each other: the second waits and
  is checked against the state the first one wrote.

  'archived' means THIS NODE is done with it, not "fully analyzed". A GPU
  machine running also_process=false marks 'archived' the moment it has staged
  the pose; the algorithms have not run yet. The collages table uses the same
  idiom for 'cropped'. This is the single most misread value in the schema.

  'processed' on the processing server means analysed, passed the review
  gate, and waiting to be filed.

  'archiving' (since 2026-11-15) is written BEFORE the files move into the
  filed folder. A row left in it means a filing attempt was interrupted; the
  next watcher start puts it back to 'processed', and the next filing attempt
  records 'archived' if that run's files are already in the filed folder. A
  failed attempt that leaves the files in Processing goes back to 'processed'.

  'failed' with an error starting "not filed: no kinematics" (since
  2026-11-15) means the analysis ran but the features file could not be
  written; the video is NOT filed, its files stay in Processing, and the
  processing server's failed-video retry runs it again.

  'processing' can also mean "analysed, and HELD for its hand scores": since
  2026-10-28 the hand-score comparison runs after the analysis and before the
  review gate, and a video whose session has no hand scores yet stays in
  'processing', with its files in the local Processing folder, until the next
  import of its sheet (watcher/hand_scores.py). If that import has none, the
  scores are missing and the video STAYS in 'processing' -- it is never filed
  uncompared -- until a later import brings them. Such videos are listed in
  <NAS root>/attention/<host>_hand_scores.json (grey while waiting, red with
  kind 'hand_scores_missing' once the import came without them), and the
  processing log has one row for the wait: step 'hand_score_check', status
  'waiting', and one more, status 'missing', when it turns red. They do not
  count against max_local_pending. (From 2026-10-26 to 2026-10-28 the wait
  happened in 'processed', before filing.)

  On a GPU node, a row can also be 'archived', 'triage' or 'deep_review' with
  source_path '(no file on this node)' for a video this node never cropped or
  posed. Since 2026-09-15 a GPU node checks the shared drive before it poses a
  collage child or a dropped single: an archive manifest means 'archived', a
  bundle in a review queue (or its .incoming build folder) means that queue's
  state, and the row is recorded that way instead of being queued for DLC.
  The processing log says why (step 'crop', 'adopt' or 'dlc', status 'skipped').

  processing_log step 'dlc', status 'aborted' (since 2026-09-15): a pose was
  stopped part-way because a program listed in watcher.pause_while_running
  opened (the message is the reason, e.g. "recorder.exe is running"). The row
  went back to 'dlc_queued' with the same current_path, error_count was NOT
  increased, and the half-written pose files of that run were removed. It is
  not a failure of the video; the pose runs again once recording is over.

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

  More 'unresolvable' reasons, for singles dropped into
  Unanalyzed/Single_Animal (since 2026-09-15; watcher/single_claim.py):
    * "gone from the shared singles folder before this node took it: ..."
      (single_claim.LEFT_FOLDER_REASON). Another GPU node claimed the video
      first (the reason names it), or somebody removed the file. Only rows
      with this reason are set back to 'validated' by the intake scan,
      automatically, when a file of that name is back at the top of the folder
      and stable, and no node holds a claim on it. That happens when another
      node's copy failed or its claim went stale. WHY only these rows:
      'unresolvable' has other reasons, and a row parked for one of them must
      not be re-driven just because a same-named file exists. So a row may
      go from 'unresolvable' back to 'validated' with nobody acting: that is
      expected.
    * "no file on this node; another node had it before its pose: <node> has
      it at state '<state>'" (single_claim.ELSEWHERE_BEFORE_POSE_REASON),
      written by startup recovery when the shared record says another node
      had the video before its pose was finished. Re-driven exactly like the
      reason above. The plain "no file on this node; <node> has it at state
      ..." (another node is past the pose, or the video is filed) is never
      re-driven.
    * "posed files not all in the posed folder when intake looked (...)"
      (db.INTAKE_MISS_REASON), processing server only: its intake looked
      while a GPU node was still handing the files over. The posed-folder
      scan sets the row back to 'dlc_complete' by itself once the video and
      its pose are both in Processing/Posed.
    * "a second copy of a video another node already holds for pose: ..."
      (single_claim.DUPLICATE_OF_CLAIM_REASON). The same video was copied in
      again while another node still held it. It is never re-driven. A person
      deletes the second copy.

  processing_log step 'single_claim', status 'completed' (since 2026-09-15):
  this node removed its claimed copy of a single
  (Single_Animal/.inflight/<host>/<stem>.mp4) after the video's next copy was
  confirmed: staged, archived, or in a review bundle. The message names the
  file and how the next copy was confirmed.

  A 'failed' single whose error_message starts "could not copy it from the
  shared singles folder into the DLC queue" (single_claim.COPY_FAILED_REASON)
  had its copy onto this node fail; the video went back to the folder for any
  node. The singles intake sets the row back to 'validated' by itself an hour
  after the failure (last_error_at) while error_count is below
  watcher.max_retries, once the file is back at the top of the folder, stable
  and claimed by no node -- so this node, or another, takes it again. When
  the retries are used up it stays 'failed': fix the cause, then run
  mousereach-watch-reprocess <video_id>; with the file back at the top of
  the shared singles folder the reset puts the row to 'validated'. (Rows
  written before this rule say "could not copy ... into the DLC queue; ...
  This node does not retry it on its own" and are not offered again by
  themselves; mousereach-watch-reprocess resets them the same way.)

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
video's results: the pose model, every stage version, the applied_review
stamp and the applied_ground_truth stamp (the ground-truth file the feature
extractor actually applied, by content; null: none was applied; absent: it
could not be determined). The database records where a video is in the pipeline; the manifest
records what was done to it. When the two disagree, the manifest is the one
that was written by the code that did the work.
