# Engine Code Map

This file is a code-backed map of these files only:

- [sdk/client.py](/D:/Nura/Code/nura_living_memory_code/sdk/client.py)
- [app/engines/sentence_model.py](/D:/Nura/Code/nura_living_memory_code/app/engines/sentence_model.py)
- [app/engines/grammar_engine.py](/D:/Nura/Code/nura_living_memory_code/app/engines/grammar_engine.py)
- [app/engines/memory.py](/D:/Nura/Code/nura_living_memory_code/app/engines/memory.py)
- [app/engines/temporal.py](/D:/Nura/Code/nura_living_memory_code/app/engines/temporal.py)
- [app/db/session.py](/D:/Nura/Code/nura_living_memory_code/app/db/session.py)
- [app/db/models.py](/D:/Nura/Code/nura_living_memory_code/app/db/models.py)

Every statement below is grounded in code present in those files.

## 1. SDK Entry

### `sdk/client.py`

- `35-77`
  `Kenotic.__init__` stores `user_id` and `db_path`, sets `RAYA_EMBED_DEVICE`, assigns `settings.sqlite_path`, calls `_init_schema()`, and initializes lazy engine slots.

- `81-121`
  `_run_verifier()` imports `verify_kenotic_architecture_v1`, runs it once, logs pass/fail counts, and returns the cached results list.

- `125-134`
  `_engines()` imports and caches:
  - `get_memory_engine`
  - `get_temporal_engine`
  - `get_retrieval_engine`

  It also calls `self._temporal.bind_memory(self._memory)`.

- `136-153`
  `_init_schema()`:
  - opens SQLite directly
  - runs `MIGRATIONS`
  - runs `run_schema_upgrades(conn)`
  - attempts to add `sequence_number`
  - commits and closes

- `157-214`
  `ingest(...)` calls `memory.ingest_text(...)`.
  If `model_response` is non-empty, it calls `memory.ingest_text(...)` again with `source_tag="model_comprehension"`.

- `216-251`
  `retrieve(...)`:
  - may route to `reconstruct(...)` depending on `settings.explicit_reconstruct_only` or `wh_type.is_situational`
  - otherwise calls `retrieval.retrieve(self.user_id, query)`

- `253-333`
  `forget`, `show`, `trace`, `check_proactive`, `profile` delegate to engine methods.

- `335-441`
  `process(...)`:
  - proactive-only mode if `check_proactive` and empty `text`
  - otherwise tries `grammar_engine.process(...)` to classify the utterance
  - routes:
    - backchannel -> `ProcessResult(action="skipped")`
    - command -> forget path
    - question -> retrieve or reconstruct
    - statement -> ingest

- `480-527`
  `KenoticV1(...)` is a module-level singleton wrapper around `Kenotic(...).process(...)`.

## 2. Cleanup Layer

### `app/engines/sentence_model.py`

- `35-77`
  Defines:
  - `_COEDIT_TASK_PREFIX = "Fix grammatical errors in this sentence:"`
  - filler/discourse/idiom/retraction sets

- `84-140`
  Defines model lifecycle helpers:
  - `is_enabled()`
  - `is_available()`
  - `_load()`

  `_load()` loads the local seq2seq model and tokenizer, sets device, and caches both.

- `142-196`
  `polish(sentence)`:
  - returns the input unchanged if polishing is disabled or the model is unavailable
  - otherwise runs one model generation with `_COEDIT_TASK_PREFIX`

- `198-587`
  `cleanup(text, speaker=None)`:
  - returns early for empty text
  - performs structural cleanup using spaCy
  - removes retractions
  - strips some rhetorical/discourse question prefixes
  - removes filler tokens and some scaffolding spans
  - removes some idiom-at-start frames
  - normalizes punctuation/token joins
  - if the model is enabled and available, rewrites sentence-by-sentence
  - rejects rewrites that:
    - look like tuple/triple hallucinations
    - lose all named entities from the input
    - are less than half the input length
    - introduce a `?` where the input sentence had none
  - strips some discourse-only lead-ins after rewrite
  - strips a trailing `but also` fragment

- `588`
  `reset_cache()` clears `_CACHE`.

## 3. Grammar Layer

### `app/engines/grammar_engine.py`

- `47-78`
  Defines:
  - `_get_nlp()`
  - `_get_nlp_fragment()`
  - `_ensure_wordnet()`

- `96-249`
  Defines these core types:
  - `CoarseBin`
  - `UtteranceClassification`
  - `TenseAspect`
  - `Triple`
  - `TraceDecomposition`
  - `GrammarResult`
  - `VerbClass`

- `250-408`
  Defines verb classification helpers, including `classify_verb_class(...)`.

- `409-652`
  Defines syntactic extraction helpers, including:
  - `_get_root(...)`
  - `_span_text(...)`
  - `_extract_grammatical_object(...)`
  - `_get_prep_object(...)`

- `653-1218`
  Defines utterance analysis helpers, including:
  - `detect_mood(...)`
  - `detect_negation(...)`
  - `detect_tense_aspect(...)`
  - `detect_voice(...)`
  - `resolve_pronouns(...)`
  - `classify_utterance(...)`

- `1291-2114`
  Defines trace extractors:
  - `_extract_episodic(...)`
  - `_extract_emotional(...)`
  - `_extract_temporal(...)`
  - `_extract_relational(...)`
  - `_extract_schematic(...)`

- `2115-2478`
  Defines trace assembly helpers:
  - `_build_trace_decomposition(...)`
  - `_find_content_verb(...)`
  - `_extract_traces_from_sentence(...)`

- `2479-3456`
  Defines imposed-fact helpers:
  - `_build_imposed_trace(...)`
  - `_extract_imposed_facts(...)`

- `3457-3746`
  Defines clause/triple helpers:
  - `_derive_triple(...)`
  - `_split_compound_clauses(...)`

- `3747-4004`
  `process(text, speaker=None, listener="user")`:
  - parses the input with `_get_nlp()`
  - splits compound clauses
  - classifies each clause
  - resolves pronouns
  - extracts decompositions
  - derives triples
  - returns `GrammarResult`

- `4005-4019`
  `extract_typed_triple(...)` wraps `process(...)`.

- `4020-4105`
  Defines `QueryDecomposition` and `_wh_to_return_field(...)`.

- `4105+`
  `classify_query(query_text)` parses a query and fills a `QueryDecomposition`.

## 4. Memory Layer

### `app/engines/memory.py`

- `55-87`
  Defines local dataclasses:
  - `Entity`
  - `Relationship`
  - `Traces`

- `109-147`
  `_run_ingestion_path(text, speaker=None)`:
  - returns `("", [], None)` for empty input
  - imports `sentence_model.cleanup` and `grammar_engine.process`
  - runs cleanup
  - runs grammar processing
  - flattens `GrammarResult.triples` plus `trace_decompositions`
  - returns `(final_text, rows, grammar_result)`

- `149-154`
  `clean(text)` calls `sentence_model.cleanup(...)` directly.

- `156-169`
  `extract(text)` calls `_run_ingestion_path(...)` and returns tuples:
  - `(subject, predicate, object, is_historical)`

- `175-248`
  `ingest_text(...)`:
  - returns `0` for empty input
  - calls `_run_ingestion_path(...)`
  - resolves `user` / `i` / `me` / `myself` to the speaker name when present
  - copies `speaker` into `decomp.relational_subject` when needed
  - calls `store(...)` once per flattened row
  - returns the number of stored rows

- `254-424`
  `store(...)`:
  - fills subject/predicate/object from `trace_decomposition` if missing
  - builds fallback `source_text` if needed
  - computes `source_text_hash`
  - calls `_prepare_row(...)`
  - opens a DB transaction with `get_db_context()`
  - calls `_write_row(...)`
  - if write succeeds, runs side-effects:
    - `_upsert_entities(...)`
    - `_compute_affiliation(...)`
    - type backfill from `entities`
    - `_upsert_fact(...)` or `_append_milestone(...)` depending on `_tier`
    - predicted query generation
    - `_resolve_event_date(...)`
    - `TemporalEngine.detect_supersession(...)`
    - `_assign_cluster(...)`
    - `TemporalEngine.detect_arcs(...)`
  - commits and returns `rel_id`
  - on exception, rolls back and returns `0`

- `430-516`
  `_prepare_row(...)` is pure computation.
  It fills the row dict with:
  - base relationship fields
  - trace fields from `TraceDecomposition`
  - `temporal_expression`
  - `relational_entities`
  - `edge_embedding`
  - `predicate_embedding`
  - internal `_tier`

- `522+`
  `_write_row(...)`:
  - dedups on `source_text_hash`
  - refuses to revive tombstoned rows
  - updates or inserts into `relationships`
  - updates/syncs `relationships_fts`

- `709`
  `_compute_affiliation(...)` exists.

- `791`
  `_upsert_entities(...)` exists.

- `881`
  `_upsert_fact(...)` exists.

- `970`
  `_append_milestone(...)` exists.

- `1062`
  `_assign_cluster(...)` exists.

- `1515-1568`
  Forget helpers include:
  - `forget_by_time_range(...)`
  - `forget_by_source(...)`

- `1574-1651`
  `list_by_facet(...)` exists.

- `1661-1665`
  `get_memory_engine()` returns the module singleton.

## 5. Temporal Layer

### `app/engines/temporal.py`

- `53-58`
  `_get_spacy()` loads `en_core_web_sm`.

- `65-96`
  Defines:
  - `TemporalResult`
  - `Cluster`
  - `SupersessionEvent`
  - `TemporalPattern`

- `123-133`
  `TemporalEngine.__init__` stores an optional memory engine and initializes caches.

- `131-133`
  `bind_memory(...)` stores the memory-engine reference.

- `137-176`
  Time authority helpers:
  - `now()`
  - `freeze()`
  - `unfreeze()`
  - `temporal_context()`
  - `days_until()`
  - `days_since()`

- `177-299`
  Defines:
  - `proximity(...)`
  - `is_before(...)`
  - `duration_between(...)`
  - `urgency(...)`
  - `upcoming_events(...)`
  - `_to_datetime(...)`

- `304-319`
  Defines:
  - `_anchor(...)`
  - `_cos(...)`

- `324-452`
  `resolve_event_date(text, reference_timestamp=None)`:
  - extracts DATE/TIME spans with spaCy NER
  - uses `dateparser`
  - uses embedding-based temporal direction preference
  - returns ISO datetime or `None`

- `456-564`
  `parse(text, now=None)`:
  - embeds the input
  - compares against temporal anchors
  - chooses a direction
  - calls `resolve_event_date(...)`
  - calls `_extract_duration(...)`
  - returns `TemporalResult`

- `571-717`
  `detect_supersession(...)`:
  - accepts a relationship id or object
  - reads the new edge from the DB
  - uses embedding anchors plus grammar-based predicate normalization
  - may return `SupersessionEvent`

- `720-835`
  `recluster_for_reconstruction(...)`:
  - fetches edge rows
  - groups them by shared non-generic entities
  - returns reconstruction clusters

- `841-930`
  `detect_arcs(...)`:
  - fetches the new edge
  - embeds the edge topic
  - compares against open arcs
  - attaches to an existing arc or creates a new arc

- `934-959`
  `edges_valid_at(...)` exists.

- `961-996`
  `temporal_neighbors(...)` exists.

- `1000-1058`
  `patterns(...)` exists.

- `1065-1102`
  `staleness_batch(...)` and `staleness(...)` exist.

- `1106-1117`
  `humanize(...)` exists.

- `1123-1127`
  `get_temporal_engine()` returns the module singleton.

## 6. DB Session Layer

### `app/db/session.py`

- `8-11`
  `init_wal_mode(conn)` enables WAL and commits.

- `14-19`
  `get_db_connection()`:
  - opens SQLite using `settings.sqlite_path`
  - sets `row_factory = sqlite3.Row`
  - enables WAL

- `22-29`
  `get_conn()` returns `get_db_connection()`.

- `33-46`
  `get_db_context()` yields a connection and always closes it.

- `49-66`
  `init_db(sqlite_path)`:
  - creates the parent directory
  - runs `MIGRATIONS`
  - runs `run_schema_upgrades(conn)`
  - enables WAL

- `73-139`
  Proactive cooldown helpers:
  - `get_proactive_cooldown(...)`
  - `update_proactive_cooldown(...)`

## 7. DB Schema

### `app/db/models.py`

- `28-43`
  `facts` table:
  - one row per `(user_id, key)`

- `49-64`
  `milestones` table

- `111-126`
  `entities` table

- `138-223`
  `relationships` table with:
  - source text and hash
  - trace fields
  - triple fields
  - type fields
  - temporal fields
  - embeddings
  - structural ids
  - grammar-trace fields
  - confidence/provenance fields

- `220-223`
  `relationships_fts` virtual table

- `263-276`
  `predicted_queries` table

- `281-295`
  `arcs` table

- `328+`
  `run_schema_upgrades(conn)`:
  - checks live columns with `PRAGMA table_info(...)`
  - adds missing columns/indexes
  - rebuilds/backfills `relationships_fts` when needed

## 8. Direct Cross-File Wiring

These calls are explicit in code:

- `sdk/client.py` -> `memory.ingest_text(...)`
- `sdk/client.py` -> `retrieval.retrieve(...)`
- `sdk/client.py` -> `retrieval.reconstruct(...)`
- `memory.py` -> `sentence_model.cleanup(...)`
- `memory.py` -> `grammar_engine.process(...)`
- `memory.py` -> `TemporalEngine.detect_supersession(...)`
- `memory.py` -> `TemporalEngine.detect_arcs(...)`
- `memory.py` -> `get_db_context()`
- `temporal.py` -> `get_db_context()`
- `sdk/client.py` -> `MIGRATIONS` and `run_schema_upgrades(...)`

## 9. Minimal Write Path

For statement ingestion through the SDK, the code path is:

1. `KenoticV1(...)`
2. `Kenotic.process(...)`
3. `Kenotic.ingest(...)`
4. `MemoryEngine.ingest_text(...)`
5. `MemoryEngine._run_ingestion_path(...)`
6. `sentence_model.cleanup(...)`
7. `grammar_engine.process(...)`
8. `MemoryEngine.store(...)`
9. `MemoryEngine._prepare_row(...)`
10. `MemoryEngine._write_row(...)`
11. memory side-effects

That path is directly supported by the code in these files.
