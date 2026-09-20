# SPEC: Entity-Aware Footage Retrieval for VidBrisk

## 1. Why this spec exists

VidBrisk's footage engine currently finds clips by generating a search
query (one or two keywords) from a script beat, embedding it, and
taking the top match from the vector index. This has two compounding
problems:

1. **Single-shot search has no recall strategy.** One keyword phrasing
   against a large, loosely-labeled footage pool often lands a
   mediocre match, and there's no fallback or reranking — whatever
   comes back gets used, low similarity score or not.
2. **No way to target a specific known subject.** If we've ingested
   footage of a specific ship, animal, or place, there's currently no
   way to tell the engine "use footage of *this* thing" — it can only
   search by loose semantic similarity across the whole pool. Tagging
   was considered and rejected: free-text tags drift, and nobody
   remembers what tag they used six months ago.

The real workflow this needs to support: Annas ingests footage
per-subject (a folder of Cyclops footage, a YouTube documentary about
an aye-aye) and wants that footage preferentially — and reliably —
reused whenever a future script mentions that subject, while still
falling back to a general B-roll pool for generic beats (ocean waves,
establishing shots, etc.).

**This spec solves that by adding two things to the pipeline:**
a canonical entity registry (Postgres) that footage and scripts both
resolve against, and a shot-level ingest/retrieval pipeline that
filters by entity before ranking by semantic match. Multi-query
generation and reranking are included as the fix for problem 1 and
apply to both entity-filtered and general search.

**Out of scope for this spec:** UI for manual entity tagging/review,
backfilling entity assignment onto the existing unlabeled footage
library (separate migration effort), and cross-entity relationship
modeling (e.g. "aye-aye" is-a "lemur" is-a "primate" — flat entities
only for v1).

---

## 2. Current state (for context)

- Footage embeddings live in Zilliz/Milvus, indexed by X-CLIP vectors.
- Shot segmentation via PySceneDetect, transcription via WhisperX
  already run elsewhere in the pipeline (voice/caption work) — this
  spec reuses both, applied to source footage instead of narration.
- No VLM captioning step currently exists on ingested footage.
- No entity concept exists anywhere in the schema today.

---

## 3. Data model

### 3.1 Postgres — canonical registry

```sql
CREATE TABLE entities (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,              -- "USS Cyclops"
    aliases       TEXT[] NOT NULL DEFAULT '{}', -- ["Cyclops", "AC-4", "the collier"]
    entity_type   TEXT NOT NULL,               -- 'ship' | 'animal' | 'person' | 'location' | 'event'
    notes         TEXT,                        -- free-form, optional
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_entities_aliases ON entities USING GIN (aliases);

CREATE TABLE sources (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id     UUID REFERENCES entities(id),   -- nullable: general/unassigned footage
    source_type   TEXT NOT NULL,                  -- 'youtube' | 'upload' | 'stock'
    source_url    TEXT,
    source_path   TEXT,                            -- local/object-storage path if uploaded
    title         TEXT,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    status        TEXT NOT NULL DEFAULT 'pending'  -- 'pending' | 'processing' | 'ready' | 'failed'
);

CREATE INDEX idx_sources_entity ON sources (entity_id);
```

Alias resolution at ingest/retrieval time is a simple `name ILIKE` /
`aliases @> ARRAY[...]` lookup, or a fuzzy match (trigram/`pg_trgm`)
if exact/alias match fails, before falling back to "create new
entity" or "ask for confirmation" (see 4.1 and 4.3).

### 3.2 Milvus — shot-level index

```
Collection: shots
  id                  VARCHAR (PK)
  source_id           VARCHAR      -- FK to Postgres sources.id
  entity_id           VARCHAR      -- denormalized from sources.entity_id, nullable
  embedding           FLOAT_VECTOR -- X-CLIP, existing dimension
  caption             VARCHAR      -- VLM-generated shot description
  transcript_snippet  VARCHAR      -- WhisperX text overlapping this shot's timerange, if any
  start_ts            FLOAT
  end_ts              FLOAT
  duration            FLOAT
```

`entity_id` is a scalar field with a Milvus scalar index, enabling
filtered ANN search (`entity_id == "..."` as a pre-filter, not a
post-filter) in a single query — no second round trip to Postgres
needed at retrieval time for the common case.

Denormalizing `entity_id` onto the shot record (rather than joining
through `source_id` every query) means an entity reassignment in
Postgres must also patch the corresponding Milvus records — see 4.4.

---

## 4. Ingest pipeline

Trigger: a folder path, a single file upload, or a YouTube URL is
submitted for ingestion, optionally with an explicit entity ("this is
about the Aye-aye").

### 4.1 Entity resolution (before processing footage)

- If an explicit entity name is given: resolve against
  `entities.name`/`aliases`. Exact/alias match → reuse `entity_id`.
  No match → create a new `entities` row.
- If no explicit entity is given (e.g. a folder of general ocean
  footage): `entity_id` stays null, footage lands in the general pool.
- If the resolved name is close-but-not-exact to an existing entity
  (fuzzy match hit, not exact), flag for a one-line confirmation
  rather than silently creating a duplicate entity or silently
  merging into the wrong one. This is the one manual-confirm touch
  point in the whole pipeline — cheap, and it's what prevents "aye-aye"
  and "aye aye lemur" from silently becoming two entities.

### 4.2 Download / normalize

- YouTube URL → download via existing tooling, extract audio track.
- Folder/upload → use as-is.
- Register a `sources` row (`status = 'processing'`).

### 4.3 Shot segmentation and enrichment

For each source video:

1. **PySceneDetect** → list of (start_ts, end_ts) shot boundaries.
2. **Keyframe extraction** per shot (e.g. midpoint frame).
3. **VLM captioning** per keyframe → short natural-language
   description (subject, setting, notable visual detail). This is
   the field that makes text-based beat matching reliable — see 5.2.
4. **X-CLIP embedding** per shot (existing model/pipeline).
5. **WhisperX transcript overlap**, if the source has narration/audio
   — attach any transcript text whose timestamp falls inside the
   shot's range.
6. Write one `shots` record to Milvus per shot, with `entity_id`
   copied from the resolved `sources.entity_id`.

Mark `sources.status = 'ready'` on completion, `'failed'` with an
error note on any unrecoverable step failure (partial results are
still usable — a failed caption on one shot shouldn't discard the
whole source).

### 4.4 Entity edits after ingestion

If an entity is renamed, merged into another, or a source's entity
assignment is corrected after the fact, propagate the `entity_id`
change to every Milvus `shots` record with that `source_id`. This
should be a single batch update keyed by `source_id`, not a full
reindex.

---

## 5. Retrieval pipeline (per script beat)

### 5.1 Entity detection in the beat

Run entity resolution against the beat's script text (LLM extraction
of named subjects, or a lighter NER pass — either resolves against
`entities.name`/`aliases`, same lookup as ingest-time).

- **Hit** → note the `entity_id`(s) mentioned.
- **No hit** → beat is generic, skip to 5.3 with no entity filter.

### 5.2 Multi-query generation

Regardless of entity hit, generate several query variants from the
beat text rather than one:

- the literal subject/action described,
- a broader category fallback,
- a shot-type variant (wide/close/interior/exterior, if inferable),
- a mood/tone variant if the beat has strong emotional framing.

This directly targets the "one keyword search returns a mediocre
match" problem — the fallback variants give the search room to find
something usable even when the literal phrasing doesn't hit well.

### 5.3 Search

- If an entity was detected: Milvus query with `entity_id == X` as a
  pre-filter, ANN search across the multi-query embeddings within
  that filtered set only.
- If no entity: ANN search across the full/general pool
  (optionally still excluding other entities' footage, or including
  everything — recommend including everything, since a Cyclops shot
  of "ocean waves" is still valid generic footage; the entity filter
  should only apply when the beat is specifically about that entity).
- Merge and dedupe candidates across all query variants.

### 5.4 Rerank

Rerank the merged candidate pool against the original beat text (not
the search queries) using a cross-encoder or LLM-judge pass. Score
each candidate; take the top result plus its confidence score.

### 5.5 Confidence floor

If the top result's score is below a defined threshold:

- do not silently ship it,
- flag the beat for manual review, route to a generative B-roll
  fallback if available, or mark the beat visibly (e.g. a placeholder
  layer) in the Remotion timeline rather than surfacing a wrong or
  unrelated clip.

### 5.6 Multiple shots per beat

A beat may need more than one shot (e.g. an entity-specific insert
shot plus a generic establishing shot). Retrieval should support
requesting N shots per beat rather than assuming exactly one, running
5.1–5.5 independently per requested shot slot.

---

## 6. Open questions for the team

1. Threshold value for the confidence floor (5.5) — needs a small
   labeled eval set to calibrate rather than a guessed number.
2. VLM choice for captioning (4.3.3) — latency/cost per shot at
   ingest-time scale needs benchmarking against whatever's already
   in use elsewhere in the pipeline.
3. Should general-pool search (5.3, no entity hit) ever exclude
   footage belonging to *other* entities, or should everything always
   be eligible for generic beats? Spec assumes "always eligible" —
   confirm this matches intent.
4. Fuzzy-match threshold for entity resolution (4.1) — how close is
   "close enough to flag for confirmation" vs. "close enough to just
   match" vs. "different entity entirely."

---

## 7. Rollout order

1. Postgres schema (3.1) + Milvus schema addition (3.2, additive
   field, no migration of existing records required).
2. Ingest pipeline (4.1–4.3) on new/newly-ingested footage only.
3. Retrieval pipeline (5.1–5.5) — ships useful immediately for any
   footage ingested under step 2, degrades gracefully (behaves as
   today) for unlabeled legacy footage since `entity_id` is nullable.
4. Confidence floor + fallback routing (5.5) — depends on threshold
   calibration (6.1), can ship slightly behind the rest.
5. Legacy footage backfill (assign `entity_id` to existing library) —
   separate follow-up effort, not blocking the above.
