# Retrieval Engine v2 -- Handoff
**Date:** 2026-05-03
**Status:** 8 capabilities implemented. 22/25 isolation (88%), 21/27 LOCOMO-hard.
**File:** app/engines/retrieval.py (2705 lines)

---

## What It Is

The retrieval engine is the **read path** of DTCM. Given a user query, it finds the correct fact from the knowledge graph and returns a short answer -- or refuses if the fact does not exist.

Two layers:
1. **Moat pipeline (Stages 1-5):** Candidate generation and ranking. Recalls from PQ cosine, edge embedding cosine, and FTS BM25, then expands by entity, groups by cluster, filters by reachability, and orders by exit cosine.
2. **Verification loop:** Anti-hallucination gate. Entity match (hard gate) + schema re-ranking + predicate alignment + temporal existence check. First verified candidate wins; all rejected = structural refusal.

## Where It Is

- app/engines/retrieval.py -- main engine (2705 lines)
- app/engines/retrieval_types.py -- Candidate dataclass
- app/engines/wh_type.py -- WH-word to expected answer type
- app/engines/type_resolver.py -- entity type resolution
- app/engines/entity_resolver.py -- query-time entity cosine resolution
- app/engines/predicted_queries.py -- PQ generation (write-path, read at query time)
- tests/test_retrieval_isolation.py -- 25-query isolation test (22/25)
- _test_locomo_hard.py -- 27-query LOCOMO-hard test (21/27)
- _locomo_data.json -- LOCOMO-hard test edges + queries

## 8 Capabilities -- Current State

1. Anti-hallucination -- DONE. _verification_loop, _check_entity_match, _check_existence
2. Indirect reference -- DONE. _resolve_indirect_references (5 patterns: speaker, possessive, relcl, comparative, who-else)
3. Paraphrase tolerance -- PARTIAL. Verb-level cosine + WordNet. Fails on: based/live, exercise/run, colleague/manager
4. Multi-hop traversal -- DONE. relcl to _find_entity_by_edge to Stage 2 expand to verification
5. Temporal reasoning -- DONE. is_current filter removed from SQL. _query_requests_historical routes temporal mode. _check_existence temporal-aware. Tombstoned edge detection for still/is-it-true queries.
6. Counting/aggregation -- DONE. _aggregate_loop collects all verified candidates. Count includes entity names.
7. Relational inference -- DONE. _find_peer_entities for who-else queries. Pattern 5 in _resolve_indirect_references.
8. Negation/absence -- DONE. Yes/no predicate absence detection with WordNet. _entity_exists CWA check.

## Verification Loop -- How It Works

candidates (sorted by exit_cosine, schema-boosted)
  Gate 1: Entity match -- edge must mention query_entity in subject, object, or relational_entities
  Gate 2: Existence check -- temporal-mode-aware (current/historical/any), tombstoned rejection
  Gate 3: Predicate alignment (yes/no queries) -- verb lemma overlap + object hint WordNet check
  ACCEPT: first passing candidate, rendered via WH-type-aware _render_answer_text
  ALL REJECTED: StructuralRefusal

## What Is To Be Done

### P0 -- Paraphrase Gap (5 LOCOMO-hard failures)

All 5 remaining failures are the same root cause: WordNet does not connect the query verb to the edge predicate, and verb-level embedding cosine is too low.

- wife vs be_girlfriend_of (semantic role: wife/partner/girlfriend)
- have vs own (light verb: have/possess/own)
- based vs live_in (location paraphrase: based/reside/live)
- exercise vs run (hypernym: exercise/physical_activity/run)
- colleague vs be_manager_of (workplace role: colleague/coworker/manager)

Approach: embedding-based fallback when lemma/WordNet fails. Among entity-matched candidates, when no candidate passes lemma/WordNet, fall back to verb-level embedding cosine argmax. No threshold -- just argmax within entity-filtered set.

### P1 -- Full LOCOMO Benchmark

The engine has NOT been tested against the real LOCOMO-10 benchmark (run_locomo.py against locomo10.json). This requires:
1. Temporal engine operational (temporal ordering, supersession)
2. Full write path through SDK (grammar engine to memory engine to DB)
3. Category 5 (adversarial/refusal) scoring fix

### P2 -- Predicate Gate for Non-Yes/No Queries

Currently predicate alignment only runs for yes/no queries. Non-yes/no queries like "Where is Tariq based?" still rely on exit_cosine ranking + schema boost. Extending predicate alignment to ALL queries would fix these but risks regressions on light-verb and category-noun queries.

### P3 -- Schema Inference Robustness

_infer_query_schema returns "career" for "graduate from" when the edge has schema="education". Schema works as secondary tiebreak but cannot be trusted as primary disambiguation.

## Vision

The retrieval engine proves Property 5 (Reconstruction) from the 7 Properties of Continuity. It does not just retrieve similar chunks -- it answers questions about the user's current living state.

The verification loop is what makes this different from RAG. RAG returns top-k similar chunks. The verification loop ensures the answer is:
1. About the right entity (not someone else's fact)
2. Temporally correct (not stale state)
3. Semantically aligned (answering what was asked, not something adjacent)
4. Grounded in the DB (the fact actually exists)

When all 8 capabilities work at production quality, the engine handles the full range of human questions about personal state -- direct lookups, indirect references, temporal queries, counting, negation, relational inference, and reconstruction.

Target: 27/27 LOCOMO-hard, then full LOCOMO-10 benchmark, then ATANT cumulative 500 stories.
