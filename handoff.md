# Handoff: Memory Engine + Grammar Engine — Session 2026-04-29

**Date:** 2026-04-29
**Author:** Engineer Manager
**Status:** COMMITTED — `0fcda41` on master
**E2E Baseline:** 54/59 (92%), 27 edges, 10 facts, 6 milestones, 64 PQs, 9 entities

---

## What shipped this session

### Memory Engine (`app/engines/memory.py`)

Rewrote `store()` from 12+ scattered UPDATEs into a three-phase pipeline:

```
Phase 1: _prepare_row()     — pure computation, no DB
Phase 2: _write_row()       — one INSERT/UPDATE + FTS5
Phase 3: side-effects       — facts, milestones, PQs, clustering, arcs
```

Full list of changes:
- Three-phase store: _prepare_row → _write_row → side-effects
- Tier router: significance alone determines fact/milestone/episode (no schema allow-list)
- Fact upsert with history chain + supersession type classification (update/correction/reversal)
- Milestone append with exact-text dedup
- Entity population in Phase 3 before affiliation scoring
- Edge embedding + predicate embedding computed at store time
- Subject/object type resolution from entities table
- Predicted queries generated and stored at store time
- Cluster assignment via entity overlap with merge on multi-match
- Arc membership via open arc matching to cluster_id
- Edge affiliation (in-group/out-group) from entity mention frequency
- FTS5 delete passes original column values (not empty strings)
- Tombstone guard prevents un-forget on re-ingestion
- Emotional valence 0.0 no longer coerced to 0.5
- Explicit conn.rollback() on store failure
- utterance_type wired from grammar → store
- is_historical, episodic_fact, emotional_target, extraction_rule columns added and wired

### Grammar Engine (`app/engines/grammar_engine.py`)

- Significance classifier rewritten: tense × aspect × verb_class (was verb_class alone)
  - present + simple → stative
  - past + simple + ACHIEVEMENT → milestone
  - past + habitual → stative
  - present + continuous → routine
- Reported speech: AUX-headed clauses now found by frame-skipper (`pos_ in ("VERB", "AUX")`)

### Schema (`app/db/models.py`)

- Migrations for: is_historical, episodic_fact, emotional_target, extraction_rule
- Trace-primary schema with source_text_hash dedup
- arcs + timers tables
- FTS5 virtual table with backfill migration

### Test

- `tests/test_e2e_writepath.py` — 20-sentence E2E test, runs full pipeline, checks every table and column

---

## Memory Engine — what's left to do

### HIGH: Fact key uses surface predicate (no contradiction tracking across synonyms)

**File:** `app/engines/memory.py`, `_upsert_fact()` ~line 880
**Problem:** Fact key is `schema::predicate::subject`. "I work at Google" → `career::work_at::Sam`. "I started at Apple" → `career::start_at::Sam`. Different predicate = different key = no history chain. Same fact, no contradiction detected.
**Fix options:**
1. Predicate normalization: map surface verbs to semantic concepts (work_at/start_at/join → EMPLOYMENT)
2. Predicate embedding similarity: if two fact keys have cosine > 0.85 on predicate_embedding, treat as same fact
3. Schema+subject key only: `career::Sam` — any career fact for Sam is the same key. Coarser but catches more contradictions.

**Risk:** Option 3 is too coarse (merges "works at Google" with "graduated from Stanford" if both are career). Option 1 needs a verb→concept mapping. Option 2 is structural but adds latency.

### MEDIUM: Entity garbage in entities table

**File:** `app/engines/memory.py`, `_upsert_entities()` ~line 760
**Problem:** Title-casing object phrases creates garbage entities like "Really Nervous About The Interview", "3 Years Old", "A Marathon". The filter only skips "user/i/me/myself".
**Fix:** Only upsert entities whose names are PROPN (proper nouns) or appear in spaCy NER spans. The grammar engine's relational_entities are already NER-filtered, but td.subject and td.object are not. Filter: skip names with >3 words, skip names starting with articles (a/an/the), or only upsert names that match a known NER label.

### MEDIUM: Cluster entity overlap could use relational_entities JSON

**File:** `app/engines/memory.py`, `_assign_cluster()` ~line 1049
**Status:** DONE — LIKE '%"entity"%' clause added. Verify performance at scale.

### LOW: _resolve_event_date is a separate UPDATE

**File:** `app/engines/memory.py`, `_resolve_event_date()` ~line 960
**Problem:** This does its own UPDATE on the relationship row. Could be folded into _write_row if the temporal engine were called in Phase 1. But the temporal engine needs the rel_id (which doesn't exist until Phase 2 writes the row), so this is structurally unavoidable.

### LOW: Dynamic SQL from dict keys

**File:** `app/engines/memory.py`, `_write_row()` ~line 572, 630
**Problem:** Column names come from dict keys via f-string. Safe (keys are internal constants) but fragile. A key with a space or special character would break the SQL.
**Fix:** Validate keys against an allow-list of known column names before building SQL.

---

## Grammar Engine — what's left to do

### HIGH: ACHIEVEMENT verb class too narrow

**File:** `app/engines/grammar_engine.py`, `_VERB_CLASS_ANCHORS` line 319
**Problem:** WordNet anchors for ACHIEVEMENT are `succeed.v.01`, `win.v.01`, `achieve.v.01`. Hypernym closure from these synsets does NOT reach `marry`, `graduate`, `have_a_baby`, `engage`. These verbs get VerbClass.UNKNOWN, so past+simple+ACHIEVEMENT → milestone never fires for them.
**Evidence:** E2E test #16 "My sister Sarah just had a baby" → `sig=routine` (not milestone). "have" → VerbClass.HAVE, not ACHIEVEMENT.
**Fix options:**
1. Add more anchor synsets: `marry.v.01`, `graduate.v.01`, `bear.v.02` (give birth)
2. Create a new VerbClass.LIFE_EVENT and add the milestone check: `if verb_class in (VerbClass.ACHIEVEMENT, VerbClass.LIFE_EVENT)`
3. Use a structural signal instead of verb class: if past+simple AND the object contains a milestone-indicating noun (baby, degree, wedding) via WordNet hypernym check

**Recommended:** Option 2 — new VerbClass.LIFE_EVENT with anchors for marriage/graduation/birth/death/engagement. Structural, scales via WordNet hypernym closure.

### HIGH: Schema misclassification

**File:** `app/engines/grammar_engine.py`, `_compute_schema()` 
**Evidence from E2E:**
- "I love playing guitar" → `schema=housing` (should be hobby/preferences)
- "I had coffee with Jake" → `schema=uncategorized` (should be social)
- "I got married" → `schema=career` (should be family/life_event — verb_class mapping uses ACHIEVEMENT→career)
- "I've been training for a marathon" → `schema=career` (should be health/fitness)
**Root cause:** Schema classification derives from verb_class → schema mapping (`_VERB_CLASS_SCHEMA_MAP`). When verb class is wrong or too broad, schema is wrong.
**Fix:** The schema mapping at line ~209-215 needs review. ACHIEVEMENT shouldn't always map to "career". Add schema inference from the OBJECT (marathon→health, guitar→hobby, wedding→family) via WordNet hypernym closure on the object noun.

### MEDIUM: Subject extraction quality

**Evidence from E2E:**
- "My mom's name is Linda" → `S=My mom 's name` (should be S=Linda or S=mom)
- "My cat Whiskers is 3 years old" → `S=My cat` (Whiskers dropped)
- "The weather was nice" → `S=The weather, O=The weather` (degenerate triple)
**Root cause:** Subject extraction uses the syntactic nsubj span, which includes possessives and determiners. Needs head-noun extraction or NER-aware subject resolution.

### MEDIUM: Imposed passive agent noise

**Evidence from E2E:**
- "I got married" → extra edge `S=Sam P=marry O=Sam` (rule=imposed_passive_agent)
- Sentence 20 → extra edge `S=Sam P=marry O=we` (rule=imposed_passive_agent)
**Root cause:** The imposed_passive_agent rule generates an extra edge when it detects a passive/ergative construction. For intransitive events like "got married", the agent IS the subject — the imposed edge is noise.
**Fix:** Guard: if the imposed agent == the existing subject, skip the imposed edge.

### LOW: "used to" not captured as temporal_expression

**Evidence:** #17 "I used to live in Boston" → `temporal_expression=None`, `is_historical=1`
**Root cause:** "used to" is a habitual-past marker, not a DATE/TIME NER entity. The temporal_expression extractor only looks at DATE/TIME NER spans. Arguably correct — "used to" conveys tense (handled by is_historical), not a specific date.

### LOW: emotional_target sometimes wrong

**Evidence:** #19 "I feel guilty about not visiting my grandmother" → `target=I` (should be "grandmother" or "visiting my grandmother")
**Root cause:** The emotional_target extractor falls back to nsubj when it can't find a pobj. The negation ("not visiting") disrupts the prep→pobj chain.

---

## Files changed

| File | Lines changed | What |
|------|--------------|------|
| `app/engines/memory.py` | +962 -188 | Three-phase store, all thread-laying |
| `app/engines/grammar_engine.py` | +75 | Significance classifier, AUX guard |
| `app/db/models.py` | +406 | Migrations, trace-primary schema |
| `tests/test_e2e_writepath.py` | +267 (new) | 20-sentence E2E baseline test |
