# Handoff: Retrieval Engine — Session 2026-04-29

**Date:** 2026-04-29
**Author:** Engineer Manager
**Status:** NOT STARTED — spec only, no code changes
**Prerequisite:** Memory engine write-path audit complete (`0fcda41`), E2E baseline 54/59 (92%)

---

## Current state

Retrieval reads from the relationships table via SQL keyword/entity scoping, keyword scoring with magic numbers, and recency sort. The memory engine now lays 11 threads that retrieval doesn't pull. The header of retrieval.py literally says "No cosine over everything. No predicted_queries."

## What to do: CONNECT, ENFORCE, REMOVE

---

## CONNECT (11 unread threads to retrieval consumers)

### 1. Facts table — fast-path BEFORE the pipeline

Right now retrieval runs the full pipeline for every query. For stable facts ("Where does Sam work?"), there's a one-row answer sitting in the facts table. Check it FIRST.

Step 0: Is this a fact-tier query?
  - classify_query gives schema + entity
  - `SELECT value FROM facts WHERE user_id=? AND key LIKE 'schema::verb_class::entity'`
  - If found -> return Answer immediately. Skip everything.

This handles Cat 4 single-hop factual questions in one SQL query. No scoping, no scoring, no keyword matching. LoCoMo's biggest category (42%) becomes instant.

### 2. Predicted queries — HyPE matching for Path B

When Path B fires (no entity, keyword-only), keyword matching fails on semantic gaps ("hobby" vs "enjoys reading"). The predicted_queries table bridges this.

After Path B fails (pool empty):
  - Embed the query text
  - `SELECT FROM predicted_queries WHERE user_id=? ORDER BY cosine(question_embedding, query_embedding) DESC LIMIT 5`
  - Pull the relationship_ids from top matches
  - Fetch those edges -> they ARE the pool

This is one cosine search against pre-embedded questions. The "hobby" query matches "What does Sam do for fun?" which points to the "enjoys reading" edge.

### 3. Edge embedding — cosine tiebreaker

When the pool has multiple edges after filtering, keyword scoring picks the wrong one (defect 1, 4, 5 from scale test). Replace keyword scoring with:

After scoping (entity + keyword/schema filters):
  - If pool > 1 edge:
    - Embed query text
    - For each edge in pool: `cosine(query_embedding, edge.edge_embedding)`
    - Sort by cosine DESC, then sequence_number DESC

Already fetched (line 393). `_cosine_from_blob` already defined (line 231). Just call it.

### 4. Schema scoping — add to Path B

Path B (no entity) does keyword matching only. Add schema inference:

Path B, before keyword matching:
  - `_infer_schema_from_query(query_text)` -> schema
  - If schema found: add `WHERE edge_schematic_category = ?` to all Path B queries
  - Pool shrinks from ~1,000 to ~200 -> keyword matching is accurate within the domain

`_infer_schema_from_query` already exists. It's used in Path A.2 but NOT in Path B. Wire it in.

### 5. is_historical — filter for "used to" queries

If `query_references_past(query_text)`:
  - Add `WHERE is_historical = 1` to scoping
  - Only return facts that are marked as past

This separates "Where does Sam live?" (current) from "Where did Sam used to live?" (historical).

### 6. emotional_target — filter for "how do I feel about X?"

If `wh_type == "emotional"` and query mentions a target:
  - Add `WHERE emotional_target LIKE '%target%'` to scoping
  - "How do I feel about the interview?" -> only edges with emotional_target containing "interview"

### 7. cluster_id — reconstruction queries

Already partially wired. `reconstruct()` groups by cluster_id (line 1678). Make `retrieve()` also use it for "What's going on with X?" queries — pull the entire cluster, not just keyword matches.

### 8. episodic_fact — cleaner answer text

When extracting the answer, prefer episodic_fact over raw source_text. It's the normalized fact without subject and auxiliaries. Add to SELECT and use in `_extract_by_wh_type`.

---

## ENFORCE (structural rules)

### 9. Replace keyword scoring with filter-then-sort

Remove `_keyword_score` entirely. Replace with:

Step 6 (current): `score = 10*root + 7*obj + 5*src` (magic numbers)

Step 6 (new):
  - Filter: keyword appears in predicate OR object OR source_text (boolean)
  - Sort: `cosine(query_emb, edge_embedding) DESC`, then `decay_factor DESC`, then `sequence_number DESC`

No magic numbers. No weights. The keyword is a FILTER (yes/no), not a SCORE (10/7/5). Cosine handles ranking. Decay handles recency. Sequence handles tiebreaker.

### 10. One spaCy parse per query

Currently 5+ redundant `nlp(query_text)` calls per `retrieve()`. Parse ONCE at the top, pass the doc through.

---

## REMOVE (dead code + law violations)

### 11. Remove _keyword_score function (lines 986-1004)

Replaced by cosine tiebreaker. Violates Law 5 (NO SCORES) and Law 6 (NO RANGES) with magic numbers 10/7/5/3/2/1.

### 12. Remove _AGG_SCHEMAS word list (lines 863-874)

Replace with grammar engine's `classify_query` which already infers schema structurally.

### 13. Remove _generic word list (lines 1172-1177)

Replace with POS-based filtering. Generic words are function words (DET, AUX, PRON) — filter by POS, not by list.

### 14. Remove past_patterns tuple (lines 104-123)

Replace with grammar engine's `classify_query` temporal detection. It already detects past-reference via dep tree (temporal adverbs, tense morphology). Don't substring-match the query text.

### 15. Remove confidence=1.0 hardcoded (line 1367)

Either use the edge's actual confidence or remove the field.

---

## Retrieval pipeline AFTER fixes

```
Query in
  |
  +-- Step 0: Fact fast-path
  |     classify_query -> schema + entity
  |     SELECT FROM facts WHERE key LIKE 'schema::*::entity'
  |     -> Hit? Return immediately.
  |
  +-- Step 1: Parse query (ONE spaCy parse)
  |     entities, keywords, wh_type, schema, temporal_direction
  |
  +-- Step 2: Scope (SQL filters, not Python scoring)
  |     WHERE entity matches (relational_entities / subject / object)
  |     AND edge_schematic_category = schema (if inferred)
  |     AND is_current = 1
  |     AND edge_mood = 'indicative'
  |     AND tombstoned_at IS NULL
  |     AND is_historical = ? (if past-reference detected)
  |
  +-- Step 3: If pool empty -> HyPE fallback
  |     Embed query -> cosine against predicted_queries
  |     -> Pull matching relationship_ids -> those ARE the pool
  |
  +-- Step 4: If pool empty -> StructuralRefusal
  |
  +-- Step 5: Adversarial speaker filter
  |     relational_entities check (Cat 5)
  |
  +-- Step 6: Rank (no magic numbers)
  |     cosine(query_emb, edge_embedding) DESC
  |     decay_factor DESC
  |     sequence_number DESC
  |
  +-- Step 7: Multi-hop if needed
  |
  +-- Step 8: Extract answer by wh_type
  |     Prefer episodic_fact over source_text
  |
  +-- Return Answer
```

No keyword scoring. No word lists. No magic numbers. Facts fast-path for Cat 4. HyPE for semantic gaps. Cosine for ranking. Filters for scoping.

---

## Execution order

```
Phase 1 (CONNECT — add consumers):
  1. Facts fast-path (Step 0)
  2. Edge embedding cosine tiebreaker (Step 6 replacement)
  3. Predicted queries HyPE fallback (Step 3)
  4. Schema scoping in Path B
  5. is_historical filter
  6. emotional_target filter
  7. cluster_id for reconstruction
  8. episodic_fact for answer text

Phase 2 (ENFORCE — structural rules):
  9. Replace keyword scoring with filter-then-sort
  10. One spaCy parse per query

Phase 3 (REMOVE — dead code):
  11-15. Remove word lists, magic numbers, redundant patterns
```

Phase 1 items are independent and can run in parallel. Phase 2 depends on #3 (cosine tiebreaker) being in place. Phase 3 depends on Phase 2.
