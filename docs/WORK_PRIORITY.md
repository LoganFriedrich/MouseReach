# Which video the watcher picks next

Assumes no knowledge of the codebase. If you only want to know *how to make one
kind of session go first*, read "The two settings" and stop.

---

## What the watcher is choosing between

A node wakes up every 30 seconds and asks one question: of everything waiting,
what do I do next? Waiting work comes in four kinds, and they are tried in this
order (this order is not configurable, and it is not the subject of this page):

| # | Kind | What it does | Brings new work onto this node? |
|---|------|--------------|---------------------------------|
| 1 | archive | file a finished video to the data repository | no -- it is leaving |
| 2 | intake  | copy a new video in from the handoff folder | **yes** |
| 3 | pipeline | run segmentation / reach / outcome detection on a video already here | no -- it is already here |
| 4 | reprocess | re-run a video whose tools have moved on since it was analysed | **yes** |

The order is "finish what is started before starting more". What IS configurable
is which video is chosen **inside** each of those kinds, and which videos are
not *taken on* until the node has nothing else at all to do.

A GPU node has the same shape with different names: cropping a collage and
running DLC on a queued video bring new work on; staging a finished pose to the
NAS, running the local pipeline, and archiving locally do not.

---

## The two settings

Everything lives under `watcher.work_priority` in `~/.mousereach/config.json`:

```json
{
  "watcher": {
    "work_priority": {
      "order": [{"project": ["PROJECT_A"]}, {"project": ["PROJECT_B"]}],
      "idle_only": [{"tray_type": ["E", "F"]}]
    }
  }
}
```

(`PROJECT_A` and `PROJECT_B` are placeholders. Put your own project codes there.)

**`order`** is a preference. It is a list of descriptions ("selectors"), tried
top to bottom. The first one that matches something waiting wins, and the choice
is made among its matches. Anything matching nothing in the list is not
excluded -- it simply goes last. The example says: a PROJECT_A video before a
PROJECT_B video, anything else after both.

**`idle_only`** is a deferral, and within its scope it is absolute. Anything it
matches is not **taken on** by the node until a complete sweep of all four work
kinds has found nothing else whatsoever to do. The default defers the Easy (E)
and Flat (F) trays: on those trays the divot is level with the scoring area, a
displaced pellet can just be retried, and a per-pellet outcome is not a
meaningful number. Those sessions are still wanted for session-level measures.
They are wanted last.

### What `idle_only` does *not* do

It does not stop a node **finishing** a video it already holds. Once a video is
on a node's disk, the node analyses it and files it in the normal bucket order,
even if a deferred one is next in line.

That is deliberate, and it is the fix for two real failures:

* **The stampede.** Intake is visited before the pipeline that drains it. If
  both waited for an idle node, then on an idle node intake would win every
  cycle and copy a whole `max_local_pending` worth of deferred videos onto local
  disk before analysing one of them. With intake deferred and the pipeline not,
  one deferred video is taken in and the next cycle finishes it.

* **Stranding.** A node with a backlog is never idle, so "later" means "never".
  A deferred video sitting in `Processing/` would never be analysed, never be
  archived, and would hold a slot against `max_local_pending` for good -- which
  then throttles intake of the work you actually wanted first. Videos also
  arrive in `Processing/` straight from the human review queue, so this is not
  hypothetical.

"Do not take on new low-priority work" is the policy. "Refuse to finish
low-priority work already on your disk" never was.

Deferred work already on the disk is still picked **last within its bucket** by
`order`, so a Pillar video on the same disk is always analysed first.

### Restart the watcher after editing

The file is read once when the watcher starts. `mousereach-watch --dry-run`
prints the policy it resolved, in plain words, so you can check it took.

---

## What you can select on

A selector is `{field: [accepted values]}`. It matches when **every** field it
names matches; a field matches when the video presents one of the listed values.
A bare string is accepted where a list is expected.

| Field | Matches |
|-------|---------|
| `tray_type` | `P`, `E`, `F` |
| `project` | the experiment code *or* the archive folder it maps to -- either spelling selects the same videos, and a short lettered cohort id is selected by its project without listing the letters |
| `cohort` | the bare cohort number, the code plus number, or the archive's cohort folder |
| `animal_id` | a single animal |
| `state`, `reprocess_scope` | database values, for unusual cases |

Comparison ignores case. `{}` matches everything and is the honest way to write
an explicit "everything else" tier.

**A video whose value cannot be determined never matches a named field.** A
video whose tray cannot be read is not "an E tray" and is not deferred -- unknown
work stays visible, because deferring it would hide it forever.

Neither the tray nor the project is read from the database column alone. The
column is empty on every video that has come back from a human review queue --
on a node with a review backlog, most of the waiting re-runs -- so both fall back
to the video's own filename, which always carries them.

---

## What happens if the setting is missing or wrong

**Missing** is normal and supported. The shipped default is:

```json
{"order": [{"tray_type": ["P"]}], "idle_only": [{"tray_type": ["E", "F"]}]}
```

Pillar first, Easy and Flat only when idle, and **no project preference** --
which project goes first is a decision only your lab can make, so nothing is
shipped.

**Wrong** never stops the watcher. A misspelled field, a setting name this
version does not know, a value that is not a list, a selector that is not an
object: each is dropped, the shipped default is used for that part, and a
warning names the field, says what it did, and tells you which file to edit. The
warnings appear in the log at startup and in `--dry-run`.

Elsewhere in this project an unset setting stops the tool and names the command
to run. That rule is for *locations*: a path has no safe substitute, and
guessing one writes data somewhere wrong. An ordering does have a safe
substitute -- what the watcher did before this setting existed -- and refusing to
start over an unset preference would stop the lab's data for nothing. So this
one defaults, loudly.

---

## Precedence, in one list

When two rules disagree, this is the order. It is fixed, and it is written down
so that it is not something you have to work out from the code.

1. **The work kind** (archive, then intake, then pipeline, then reprocess). Not
   configurable.
2. **The priority animal** (`mousereach-watch-prioritize`) wins over everything
   below, including `idle_only`. It is a person saying "this animal, now":
   explicit, manual and temporary. If the only work for that animal is an Easy
   session, that session runs, and a log line says why. The alternative is a
   button that silently does nothing for a rehab animal.
3. **`idle_only`**, in the kinds that bring new work on (intake, reprocess; on a
   GPU node, collage cropping and DLC). A preferred project's Easy session still
   waits behind any Pillar session.
4. **`order`** decides among what is left, in every kind including the ones
   `idle_only` does not touch.
5. **Chance** decides among equals -- except for archiving, staging and intake,
   which keep the database's own newest-first order.

---

## Worked examples

**"Only Pillar matters right now; leave the rest until the queue is empty."**
The default already does this. Nothing to write.

**"PROJECT_A before PROJECT_B, and Easy/Flat last."**

```json
"work_priority": {"order": [{"project": ["PROJECT_A"]}, {"project": ["PROJECT_B"]}]}
```

`idle_only` is left out, so it keeps its default of E and F.

**"This one cohort first, then anything Pillar."**

```json
"work_priority": {"order": [{"cohort": ["PROJECT_A05"]}, {"tray_type": ["P"]}]}
```

**"Process everything in whatever order; defer nothing."**

```json
"work_priority": {"order": [], "idle_only": []}
```

---

## Where this lives in the code

- `src/mousereach/watcher/work_priority.py` -- the policy, the selectors, and
  the field resolvers.
- `BaseOrchestrator._get_next_work_item` -- the two passes.
- `<role>Orchestrator._select_work_item` -- one pass over the four work kinds,
  and the ADMIT/DRAIN split marked bucket by bucket.
- `BaseOrchestrator._pick_from_pool` -- the precedence steps above.
- `tests/test_work_priority.py` -- all of it, with no database and no network.
  `TestAdmitVersusDrain` is the stampede and the stranding, reproduced.
