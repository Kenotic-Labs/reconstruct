# Retrieval Engine v3 — Definitive Spec

**Date:** 2026-05-04
**Status:** Spec. Implementation replaces current retrieval.py.
**Author:** Sam + Claude session diagnosis

---

## The Insight

The grammar engine already computes everything retrieval needs.
classify_query() gives: match_entity, match_schema, match_predicate, return_field, wh_word.
TraceDecomposition gives: object, temporal_expression, relational_entities, schematic_category, negated, mood.

Retrieval does NOT need cosine ranking, embedding search, or complex verification.
It needs SQL filters on the fields the grammar engine already filled.

---

## The Spec

### retrieve(user_id, query)

```
Step 0: classify_query(query) -> qd
  qd.match_entity     = "Caroline"
  qd.match_schema     = "career"
  qd.match_predicate  = "research"
  qd.return_field     = "episodic" | "temporal" | "relational" | "emotional"
  qd.wh_word          = "what" | "when" | "who" | "where" | "how"

Step 1: Facts fast path (O(1))
  key = f"{qd.match_schema}::{verb_class}::{qd.match_entity}"
  row = SELECT value FROM facts WHERE user_id=? AND key=?
  if row -> return Answer(text=row.value)

Step 2: Edge lookup (SQL, not cosine)
  SELECT * FROM relationships
  WHERE user_id = ?
    AND edge_mood = 'indicative'
    AND COALESCE(is_current, 1) = 1
    AND tombstoned_at IS NULL
  Filters (applied in order, each narrows the pool):
    a. Entity scope: relational_entities LIKE '%match_entity%'
       OR LOWER(subject) = LOWER(match_entity)
    b. Schema scope: edge_schematic_category = match_schema (if available)
    c. Predicate scope: predicate lemma overlaps match_predicate (if available)
  Order by sequence_number DESC (most recent first)

Step 3: Cat 5 speaker attribution
  For each candidate edge:
    speaker = JSON_EXTRACT(relational_entities, '$[#-1]')  -- last entry
    if speaker != match_entity AND speaker != 'user':
      REJECT (wrong speaker -- adversarial)
  If ALL edges rejected -> StructuralRefusal("not mentioned")

Step 4: Return the right field
  if return_field == "temporal":
    return edge.temporal_expression OR edge.resolved_event_date
  elif return_field == "emotional":
    return edge.edge_emotional_label
  elif return_field == "relational":
    return the entity that ISN'T match_entity (subject or object)
  else:  # episodic (default)
    return edge.object  -- THE short noun phrase answer

Step 5: Fallback to cosine (only if SQL found nothing)
  If Steps 1-2 return zero edges, fall back to current moat pipeline.
  This handles paraphrase queries where match_predicate is None.
```

### 8 Properties — Where Each Is Handled

| # | Property | How | Field |
|---|----------|-----|-------|
| 1 | Anti-hallucination | Edge must exist in DB. No edge = refuse. | SQL existence |
| 2 | Indirect reference | classify_query extracts match_entity from possessives/relcl. _resolve_indirect_references handles "speaker's girlfriend" chains. | match_entity |
| 3 | Paraphrase tolerance | classify_verb_class maps surface verbs to verb classes. "job"->WORK, "based"->LOCATION. Schema scope handles the rest. | match_schema + verb_class |
| 4 | Multi-hop | Chain: resolve entity from first hop, query second hop. "person who plays guitar" -> find entity -> query their work. | _resolve_indirect_references + recursive retrieve |
| 5 | Temporal reasoning | return_field=="temporal" -> return temporal_expression. is_historical flag for "used to". is_current for "still". | temporal_expression, is_historical, is_current |
| 6 | Counting/aggregation | SQL COUNT/GROUP BY on filtered edges. "How many people live in DC?" -> COUNT(DISTINCT subject) WHERE object LIKE '%DC%'. | SQL aggregate |
| 7 | Relational inference | "Who else works at Palantir?" -> SELECT DISTINCT subject WHERE predicate=work_at AND object=Palantir AND subject != user. Set operations on edges. | SQL set query |
| 8 | Negation/absence | Search for entity + predicate. Zero edges = "No" (CWA). negated=True on edge = "No". | negated field, SQL existence |

### What Changes from Current Engine

| Current | v3 |
|---------|-----|
| 5-stage moat pipeline (PQ + edge + FTS cosine) | SQL filter on entity + schema + predicate |
| PQ cosine as primary ranking | sequence_number as primary ordering |
| Complex verification loop with 3 tiers | Simple: entity match + speaker check |
| 16 wired methods from 3 engines | classify_query + SQL + return_field |
| exit_cosine = max(pq, edge, predicate) | No cosine in primary path |
| Cosine fallback for everything | Cosine fallback ONLY when SQL finds nothing |
| source_text fallback in rendering | NEVER return source_text. Always return the field. |
| ~3000 lines | ~500 lines |

### What Stays

- _resolve_indirect_references (5 patterns) — handles possessives, relcl, comparatives
- _try_facts_lookup — O(1) fact lookup
- _render_temporal — strip prepositions from temporal_expression
- StructuralRefusal with "not mentioned" text
- reconstruct() for situational queries
- PQ write-back on successful answer

### Test Expectations

| Test | Target |
|------|--------|
| Isolation (25 queries) | >= 22/25 |
| LOCOMO-hard (27 queries) | >= 23/27 |
| Real LOCOMO conv 0 (199 questions) | >= 40% F1 |
| Cat 4 (70 questions) | >= 30% F1 |
| Cat 5 (47 questions) | >= 80% F1 |
| Cat 2 (37 questions) | >= 25% F1 |
