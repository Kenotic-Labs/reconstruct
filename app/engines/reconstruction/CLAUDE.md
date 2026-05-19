# RECONSTRUCTION ENGINE — AGENT REGULATIONS
**Last Updated:** 2026-05-19
**Authority:** Founder (Sam)
**Scope:** Any agent modifying reconstruction.py or any retrieval/read-path code MUST read this file first.

---

## BEFORE YOU TOUCH ANYTHING

**Step 0: Verify the DB.**

Before writing any reconstruction code, run these queries against the actual database and record the results:

```sql
-- What columns exist on the relationships table?
PRAGMA table_info(relationships);

-- What indexes exist?
SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='relationships';

-- What FTS5 tables exist?
SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%fts%';

-- How many edges total?
SELECT COUNT(*) FROM relationships WHERE tombstoned_at IS NULL;

-- How many edges per speaker?
SELECT subject, COUNT(*) FROM relationships WHERE tombstoned_at IS NULL GROUP BY subject ORDER BY COUNT(*) DESC LIMIT 20;

-- What does the object field actually look like? Sample 30 non-null objects.
SELECT id, subject, predicate, object FROM relationships WHERE object IS NOT NULL AND object != '' AND tombstoned_at IS NULL ORDER BY RANDOM() LIMIT 30;

-- How many edges have NULL or empty objects?
SELECT COUNT(*) FROM relationships WHERE (object IS NULL OR object = '') AND tombstoned_at IS NULL;

-- What does a predicted_queries row look like?
SELECT * FROM predicted_queries LIMIT 5;

-- What does a verified_queries row look like?
SELECT * FROM verified_queries LIMIT 5;

-- What embedding columns exist?
SELECT name FROM pragma_table_info('relationships') WHERE name LIKE '%embed%' OR name LIKE '%vector%';
```

**Record these results. They are your map. Build from what IS, not from what you assume.**

If the schema doesn't match what this document describes — update your understanding, not the schema. The write path is correct. The read path adapts to it.

---

## TRACES, NOT EDGES — THE FUNDAMENTAL UNIT

**This section exists because the agent keeps drifting from traces to edges. Read it. Internalize it. If you find yourself writing code that searches by subject/predicate/object fields, or reaches for cosine similarity over a single embedding, STOP. You are thinking in edges. Think in traces.**

### What an Edge Is

An edge is a row in the `relationships` table. It has columns: id, subject, predicate, object, source_text, embedding, etc. It is a flat database record. Every retrieval system the agent has seen in its training data — RAG, vector search, BM25, reranking — operates on flat units like this. Documents. Chunks. Rows. That is NOT how this system works.

### What a Trace Is

A trace is a multi-dimensional decomposition of a single experience. When someone says "I'm nervous because I have a Google interview next Tuesday at 3 PM," that is not a row with subject="I" predicate="have" object="interview." It is FIVE simultaneous dimensions of meaning:

1. **Episodic trace** — what happened: interview scheduled at Google
2. **Emotional trace** — how it felt: nervous, anxious
3. **Temporal trace** — when: next Tuesday, 3 PM, future tense
4. **Relational trace** — who's involved: Google, the speaker, possibly an interviewer
5. **Schematic trace** — what pattern this fits: career event, job preparation, milestone

These five traces are stored as columns on the same row in the database. But the ROW is not the unit. The TRACES are the unit. The row is just the container. The retrieval operates on the traces.

### Why This Distinction Matters

**Edge thinking (WRONG):**
- Query: "When is my Google interview?"
- Search: `WHERE subject = 'user' AND predicate LIKE '%interview%'`
- Or: embed the query, cosine against all edge embeddings, return top-k
- Problem: finds any row mentioning "interview" regardless of whether it's about Google, regardless of temporal context, regardless of emotional state

**Trace thinking (CORRECT):**
- Query: "When is my Google interview?"
- Decompose query into traces:
  - Episodic probe: interview
  - Emotional probe: (empty — question doesn't specify emotion)
  - Temporal probe: asking for WHEN → temporal trace is the return target
  - Relational probe: Google
  - Schematic probe: career event
- Send ALL probe dimensions to ALL stored traces simultaneously
- Each stored trace responds proportionally to how many dimensions match
- A trace that matches on episodic (interview) + relational (Google) + schematic (career) = 3 dimensions → STRONG response
- A trace that matches only on episodic (interview) but relational is "Stanford" = 1 dimension → WEAK response
- The strongest response IS the answer. Its temporal trace column provides the "when."

### The Mathematical Foundation — MINERVA 2

Hintzman (1984) formalized this. Every stored memory is a vector of attributes. A retrieval probe is also a vector of attributes. Each stored trace responds with activation:

```
A(i) = similarity(probe, trace_i) ^ 3
```

The cubing means a trace that matches on 4 dimensions is 64x stronger than a trace matching on 1 dimension. This is not a scoring threshold — it's an exponential amplification of multi-dimensional overlap. The traces self-select. The answer emerges from convergence, not from search.

Multiple Trace Theory (Nadel & Moscovitch 1997) confirms this: "every item ever encoded exists as multiple traces, where each trace is a vector of numerical attribute values distributed across multiple dimensions."

### How This Maps to the Database

The database stores traces as columns on rows:

```
Row = container
├── Episodic trace → subject, predicate, object, source_text, episodic_fact
├── Emotional trace → edge_emotional_valence, emotional_target
├── Temporal trace → edge_temporal_context, temporal_expression, resolved_event_date
├── Relational trace → relational_entities, edge_relational_type
└── Schematic trace → edge_schematic_category, verb_class
```

Retrieval does NOT search by row. It decomposes the query into probe dimensions, matches each dimension against the corresponding trace columns, and the row whose traces match on the MOST dimensions is the answer.

### Trace Convergence = The Answer

Convergence is when multiple trace dimensions point to the same row. That row is the answer. The more dimensions that converge, the higher the confidence. If no row achieves convergence across multiple dimensions — refuse.

For LOCOMO precise answers: trace convergence identifies WHICH row contains the answer. The `object` field (or `resolved_event_date` for temporal, or `emotional_target` for emotional) of that row provides the precise text to return.

**Traces find it. The row provides the text.**

The agent must stop trying to find the text through the row (edge thinking) and start finding the row through the traces (trace thinking).

### What This Means For Every Decision

When writing retrieval code, ask:

1. "Am I searching by a single field?" → WRONG. Decompose into trace dimensions.
2. "Am I using cosine on one embedding?" → WRONG. Match across multiple trace columns.
3. "Am I checking subject, then predicate, then object sequentially?" → WRONG. Check all dimensions simultaneously and count how many converge.
4. "Am I returning the first row that matches on ANY dimension?" → WRONG. Return the row that matches on the MOST dimensions.
5. "Am I using a threshold to decide if a match is good enough?" → WRONG. Convergence is comparative — the row with the most matching dimensions wins, or if no row matches on enough dimensions, refuse.

### The Anti-Pattern

Every time the agent:
- Writes `WHERE LOWER(subject) = LOWER(?)` as the primary retrieval mechanism → edge thinking
- Uses `cosine_similarity(query_embedding, edge_embedding) > threshold` → edge thinking
- Searches FTS5 with OR logic across text fields → edge thinking
- Builds a "fact statement" from S/P/O and searches for it → edge thinking

The agent should instead:
- Decompose the query into 5 probe dimensions
- For each dimension, find which rows have matching trace values
- Count convergence: which row matches on the most dimensions?
- The highest-convergence row is the answer
- If no row converges on 2+ dimensions — refuse

This is DTCM. Decomposed Trace Convergence Memory. The name IS the algorithm. Decompose → match per trace → converge.

---

## THE THESIS OF RECONSTRUCTION

Reconstruction is NOT retrieval. Retrieval says: here are some related past things. Reconstruction says: here is the answer to your question, verified against what was stored, or a refusal because the answer doesn't exist.

The brain does not search a database. It completes a pattern. You give it a partial cue — a few features of a memory — and attractor dynamics settle into the nearest stored pattern that matches. If no pattern is close enough, the system doesn't guess. It fails to converge. That's the biological analog of refusal.

**Our reconstruction engine must do the same thing:**
1. Decompose the question into a 5-dimension trace probe
2. Send the probe to all stored traces simultaneously
3. Each stored trace responds proportionally to multi-dimensional overlap
4. The trace with the highest convergence is the answer candidate
5. Verify: does this candidate's full trace pattern match the probe on enough dimensions?
6. If verified — return the precise answer from the appropriate trace column
7. If no candidate converges — refuse. "This information is not mentioned in the conversation."

---

## WHAT THE BRAIN TEACHES US

### Pattern Completion, Not Field Matching

The hippocampal CA3 region is an autoassociative network. A partial cue activates stored patterns proportional to their overlap with the cue. The highest-overlap pattern wins. This is NOT searching by subject, then by predicate, then by keywords. It's matching the probe against ALL trace dimensions simultaneously.

**Implication:** The reconstruction engine should not search one trace dimension at a time in isolation. The query probe is decomposed into 5 trace dimensions. Each stored trace responds on all 5 dimensions simultaneously. The trace that matches on the most dimensions converges. That's the answer.

### Pattern Separation, Not Just Pattern Completion

The dentate gyrus takes similar inputs and maps them to maximally different stored representations. Caroline's experiences and Melanie's experiences might share topic words, but pattern separation means they're stored distinctly.

**Implication:** The subject field is our primary separator. But source_text, temporal context, and relational entities are secondary separators. When two candidates share a subject and predicate but differ in source_text content, the source_text is the tiebreaker — not a threshold, but a structural check against the query's topic.

### Temporal Context Drifts

Howard & Kahana's Temporal Context Model says the brain maintains a slowly drifting context vector bound to every memory. Retrieving something reinstates its context, which cues temporally adjacent memories.

**Implication:** Edges stored near each other in time share temporal context. If the query asks about something that happened "after the camping trip," the system should find the camping trip edge first, then retrieve edges temporally adjacent to it. This is temporal hopping, not just date filtering.

### Reconsolidation = Supersession

When the brain retrieves a memory, it becomes editable. The updated version replaces the old one as "current" while the old one remains as "historical."

**Implication:** The `is_current` flag and the supersession chain are biologically correct. Reconstruction should ONLY return `is_current = 1` edges unless the question explicitly asks about history.

### Event Boundaries = Session Boundaries + Topic Shifts

The brain segments continuous experience at prediction error spikes. Each segment becomes a separately addressable episode.

**Implication:** Session boundaries are natural event boundaries. But within a session, topic shifts also create implicit boundaries. Edges from different topics within the same session should not be treated as the same "episode" for retrieval purposes.

### No Thresholds

The brain uses competitive dynamics — highest activation wins. Not "is this above 0.7?" MINERVA 2 returns an echo whose intensity is relative to noise, not absolute. ACT-R uses power-law activation with competitive retrieval, not a cutoff.

**Implication:** Our system is deterministic. It works or it doesn't. A candidate either verifies against the DB as a single row or it doesn't. No cosine thresholds. No evidence_strength scores. No "close enough." Binary verification.

---

## JENGA PROCESS FOR RECONSTRUCTION

**Every decision about the reconstruction engine is a Jenga move. Before touching any code:**

### Dependency Map for Reconstruction

```
classify_query() ──→ affects ALL categories (query parsing)
│
├── Tier 1 (structural SQL) ──→ affects Cat 4, Cat 5 (subject+predicate match)
├── Tier 2 (predicted query) ──→ affects Cat 4, Cat 1 (semantic match)
├── Tier 3 (FTS5 BM25) ──→ affects Cat 4, Cat 5, Cat 1 (keyword match)
├── Tier 4 (cosine) ──→ affects Cat 4, Cat 1 (embedding match)
│
├── Verification loop ──→ affects Cat 4 (accept) AND Cat 5 (reject)
│   ├── Subject gate ──→ Cat 5 CRITICAL (wrong speaker = refuse)
│   ├── Predicate gate ──→ Cat 4 accuracy (right verb class)
│   └── Object gate ──→ Cat 4 accuracy (right fact)
│
├── Temporal bypass ──→ affects Cat 2 (date resolution)
├── List aggregation ──→ affects Cat 1 (multi-hop)
└── Refusal logic ──→ affects Cat 5 (must refuse correctly)
```

**Load-bearing pieces (DO NOT TOUCH without extreme care):**
- Cat 5 refusal logic — 22.5% of benchmark weight, currently at 96%
- Subject gate in verification — the Cat 5 defense
- Temporal bypass — Cat 2 at 58%, working well

**Loose pieces (room to improve):**
- Tier selection quality — Cat 4 at ~5%, biggest gap
- Object matching in verification — Cat 4 accuracy
- Query decomposition — `classify_query()` misses predicates on possessive/copular questions

### Before Any Reconstruction Change

1. **Identify which tiers and gates the change affects**
2. **Predict all 5 category scores** — specifically, not vaguely
3. **Simulate the code path mentally** — trace a Cat 4 question AND a Cat 5 question through the modified path
4. **Only then write the code**
5. **Run full LOCOMO benchmark — confirm prediction**
6. **If Cat 5 dropped below 90% — REVERT IMMEDIATELY regardless of other gains**

---

## THE VERIFICATION DESIGN — Sam's Architecture

This is the core of the reconstruction engine. It is correct in principle. It needs to be implemented exactly as specified.

### The Rule

**Subject = from QUERY. Predicate = from QUERY. Object = from CANDIDATE.**

Check: does ONE ROW exist in the database with all three? Yes = verified. No = rejected.

### Why This Works

For Cat 4 (should answer): "What did Caroline research?" → Subject=Caroline, Predicate=research. Candidate has object="adoption agencies." The row subject=Caroline, predicate=research, object="adoption agencies" EXISTS. Verified. Return "adoption agencies."

For Cat 5 (should refuse): "What did Melanie research?" → Subject=Melanie, Predicate=research. Candidate has object="adoption agencies." The row subject=Melanie, predicate=research, object="adoption agencies" does NOT exist. Only Caroline's row exists. Rejected. All candidates rejected. Refuse.

### Why It Fails Currently

1. **Generic predicates.** "feel", "see", "go", "do", "love" — every speaker has edges with these verbs. Caroline + feel + [something] exists for 6 different edges. The verification confirms Caroline felt something — but not the specific thing the question asks about.

2. **No predicate from query.** Possessive questions ("What is Caroline's reason for...") and copular questions ("What does Melanie's necklace symbolize?") often yield no predicate from `classify_query()`. Without a predicate, the verification has nothing to constrain.

3. **Tiers surface wrong candidates.** The tiers return edges based on partial matches — subject or keywords — without considering whether the candidate is topically relevant to the full question.

### The Unsolved Problem

The verification confirms that a fact EXISTS for a subject. It does not confirm that the fact is RELEVANT to the question's topic. "How did Caroline feel while watching the meteor shower?" — Caroline has "feel" edges about sharing her story, about being proud, about art. None are about the meteor shower. But they all verify because Caroline + feel + [their object] exists as a row.

**The brain solves this with pattern completion over ALL features simultaneously.** The cue is not just "Caroline + feel" — it's "Caroline + feel + meteor + shower." The attractor only converges if a stored pattern matches on ALL those features. No stored pattern has Caroline + feel + meteor + shower → no convergence → refusal.

**This is the gap that needs to be closed to break the 44% ceiling.**

---

## CURRENT ARCHITECTURE — What Exists

### Tier Cascade with Verification-Driven Fallthrough

```
Query → classify_query() → QueryDecomposition
                              │
                              ▼
              ┌─── Tier 1: Structural SQL ───┐
              │    subject + VerbClass        │
              │    + schema                   │
              └──────────────┬────────────────┘
                             ▼
              ┌─── Tier 2: Predicted Query ──┐
              │    cosine on PQ embeddings   │
              └──────────────┬────────────────┘
                             ▼
              ┌─── Tier 3: FTS5 BM25 ────────┐
              │    entity-aware keywords     │
              └──────────────┬────────────────┘
                             ▼
              ┌─── Tier 4: Cosine ───────────┐
              │    edge_embedding similarity  │
              └──────────────┬────────────────┘
                             ▼
              Verification Loop
              (subject + predicate + object = one row?)
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                VERIFIED          ALL REJECTED
                Return object     Refuse
```

**After your fixes:** if verification rejects ALL candidates from one tier, the cascade falls through to the next tier. Only after ALL tiers are exhausted does the system refuse.

### What Each Tier Does

**Tier 1 — Structural SQL:** Exact match on subject (case-insensitive) + VerbClass + schematic category. Fast. Precise when query has a clear subject and predicate. Fails when `classify_query()` can't extract subject or predicate.

**Tier 2 — Predicted Queries:** At write time, the grammar engine generates predicted questions for each edge. At query time, embed the question and cosine-match against PQ embeddings. Good for paraphrase. Fails if PQs don't anticipate the actual question phrasing.

**Tier 3 — FTS5 BM25:** Keyword search with entity-aware filtering. Broad recall. Noisy — returns many irrelevant hits, especially for common words.

**Tier 4 — Cosine Similarity:** Embed the query, match against all edge embeddings. Good for semantic similarity. Fails on exact entity/number matching.

### What Else Exists

- **Temporal bypass:** Cat 2 questions route through temporal resolution directly
- **List aggregation:** Cat 1 multi-hop questions aggregate across multiple edges
- **Pronoun filter:** Skip candidates where object is a bare pronoun
- **Rejection memory:** Persistent tracking of rejected candidates to prevent re-trying

---

## APPROACHES TO EXPLORE — From Research

The agent should understand these methods and apply Jenga thinking to decide which ones to try. Each approach must be evaluated against ALL 5 categories before adoption.

### Brain-Inspired Approaches (Highest Priority)

**Multi-dimensional pattern matching.** Instead of checking subject, then predicate, then object separately — create a unified feature vector from the query (subject + predicate + topic words + temporal cue) and compare against unified feature vectors of stored edges. The closest match wins. No sequential filtering. This mimics CA3 pattern completion.

**Predicted query as pattern completion.** The PQ system already exists. It's the closest thing to "the brain generates what it expects and checks if it matches." Improve PQ quality at write time → Tier 2 becomes the primary retrieval path. PQs should cover the topic, not just the subject+predicate.

**Context-vector retrieval.** Maintain a context embedding that drifts over the conversation. At query time, combine the query embedding with the estimated context window. This helps temporal and multi-hop questions.

### Proven Engineering Approaches

**Hybrid BM25 + Dense with RRF.** Cognis uses this. 70% vector / 30% BM25 fused with Reciprocal Rank Fusion. Independently retrieve from both channels, fuse ranks. This compensates for BM25 missing paraphrase and dense retrieval missing exact entities.

**Cross-encoder reranking.** After the tier cascade surfaces top-20 candidates, run a cross-encoder (e.g., bge-reranker) that takes (query, candidate_source_text) as concatenated input and produces a relevance score. This is the single highest-leverage addition for Cat 4 accuracy. BUT — it adds latency and model dependency. Evaluate against the deterministic constraint.

**Entity-centric retrieval.** Extract named entities from the query. Retrieve ALL edges for those entities. Then filter by predicate and topic. This is how the brain's perirhinal cortex works — "what" first, then narrow.

### Deterministic Approaches (Most Aligned with Thesis)

**Predicate-argument template matching (QA-SRL style).** Parse the query into (who, did what, to whom, when, where). Parse each stored edge into the same template at write time. Match templates structurally. No embedding. No scoring. Templates match or they don't.

**SQL-based fact verification (current design, needs fixing).** The verification loop checks subject + predicate + object as one row. The fix needed: incorporate query topic words into the check. Not as a threshold — as a structural requirement. If the query mentions "meteor shower" and no edge for that subject mentions "meteor" or "shower" in any field — reject. This is the DG-style separation check.

**Retrieval by generation.** Generate the expected answer form from the query structure (e.g., "Caroline researched [BLANK]"), then search for rows where filling the blank produces a stored fact. This inverts the retrieval — instead of finding similar edges, you generate what the answer SHOULD look like and check if it exists.

### Methods That Must NOT Be Used

**Threshold-based scoring.** No "cosine > 0.7" or "evidence_strength > 0.4." Binary verification only.

**Token overlap as verification.** Matching scattered tokens doesn't verify facts. "Caroline" + "meteor" matching separately doesn't mean Caroline saw a meteor.

**Self-verification loops.** A candidate cannot verify against its own row in a search. Verification must check the implied fact (query subject + query predicate + candidate object) as a distinct structural lookup.

**Test contamination.** Never put LOCOMO questions into predicted queries, training data, or any cached index. An honest 44% beats a fraudulent 53%.

**Greedy retrieval.** Never take the first match. Always retrieve k≥20 and let verification select.

---

## THE SPECIFIC PROBLEMS TO SOLVE

### Problem 1: Cat 4 Returns Wrong Facts (Biggest Gap)

**Current state:** ~5% F1 on 841 questions.

**Root cause:** The tiers surface candidates that share the query's subject and a generic predicate but are about a completely different topic. Verification confirms them because the subject + predicate + object row exists — just not for the right topic.

**What the brain does:** Pattern completion over ALL features. The cue includes the topic. If no stored pattern has Caroline + feel + meteor + shower, the system doesn't converge.

**What to try (in Jenga order):**
1. Add query topic words as a structural requirement in verification — if the query mentions "meteor shower" and the candidate's source_text doesn't contain "meteor" or "shower," reject. This is a structural check, not a threshold. Simulate impact on Cat 5 first — Cat 5 questions also have topic words that won't match the wrong speaker's edges, so this should HELP Cat 5 too.
2. Improve PQ generation to include topic-specific questions — so Tier 2 finds the right edge directly.
3. Use source_text embedding similarity between query and candidate as a reranking signal within the verification loop — not as a threshold but as a preference: when multiple candidates verify, pick the one whose source_text is most similar to the query.

### Problem 2: classify_query() Misses Predicates (4 of the Remaining Cat 5 Failures)

**Current state:** Possessive questions ("What is Caroline's reason for...") return match_predicate=None.

**Root cause:** spaCy parses "Caroline's reason" as a possessive NP, and the ROOT verb is "is" (copular) which doesn't map to a meaningful VerbClass.

**What to try:** Extract the content noun as a pseudo-predicate. "Caroline's reason for getting into running" → topic="reason", content="running." Use these as retrieval cues even without a traditional verb predicate.

### Problem 3: Generic Predicates Match Everything (12 of the Remaining Cat 5 Failures)

**Current state:** "feel", "see", "go", "do" exist for every speaker.

**Root cause:** VerbClass grouping is too broad. All EXPERIENCE verbs match each other.

**What to try:** When the predicate is generic AND the subject has many edges with that VerbClass — require topic word overlap in addition to predicate match. This narrows the candidate pool without changing the verification logic.

### Problem 4: Data Quality (Upstream but Affects Everything)

**Current state:** 60-70% of edges have garbage objects (pronouns, adjectives, empty strings, speaker names).

**What to try (write-path, not reconstruction):**
- Check `is_storable` before calling `store()`
- Filter objects that are bare pronouns, lone adjectives, or speaker names
- These are ingestion fixes, not reconstruction fixes, but they reduce noise in the candidate pool

---

## SCOREBOARD

**Baseline before any reconstruction changes:**

```
┌─────────────────────┬─────────┬────────────────────────────────────┐
│      Category       │ Current │ What Reconstruction Must Do        │
├─────────────────────┼─────────┼────────────────────────────────────┤
│ Cat 1 (multi-hop)   │  ~24%   │ Aggregate across edges correctly   │
│ Cat 2 (temporal)    │  ~58%   │ Route to temporal resolution       │
│ Cat 3 (open-domain) │   ~2%   │ Mostly out of scope                │
│ Cat 4 (narrative)   │   ~5%   │ FIND THE RIGHT EDGE + RETURN IT   │
│ Cat 5 (adversarial) │  ~96%   │ REFUSE CORRECTLY — NEVER BREAK    │
│ Overall             │   44%   │ Target: 50%+                       │
└─────────────────────┴─────────┴────────────────────────────────────┘
```

**After ANY change, update this scoreboard with actual numbers. If Cat 5 < 90%, revert immediately.**

---

## THE GOLDEN RULES

1. **Cat 5 is sacred.** 96% adversarial refusal is the crown jewel. Any change that drops it below 90% is reverted. No exceptions. No "but Cat 4 improved." Cat 5 is the moat.

2. **Think in traces, not edges.** Every retrieval decision operates on the 5 trace dimensions. If you find yourself writing `WHERE subject = ?` as the primary mechanism, you are thinking in edges. Decompose. Converge.

3. **Verify from what IS stored.** The DB has trace dimensions as columns. Run the PRAGMA query. See what exists. Build from reality, not assumption.

4. **No thresholds. No scores.** Convergence is comparative — the trace matching on the most dimensions wins. No "cosine > 0.7." No "evidence_strength > 0.4." The traces converge or they don't.

5. **No LLM in the reconstruction path.** spaCy for parsing. SQL for lookup. Grammar rules for structure. Trace matching for retrieval. That's the stack. That's the moat.

6. **The benchmark confirms. It does not discover.** Simulate first. Predict all 5 scores. Then run. If the prediction was wrong, fix your understanding before the next change.

7. **Every change must answer: which continuity property does this serve?** If "none" — reject. If "it improves LOCOMO but doesn't serve a property" — reject.

8. **Traces find it. The row provides the text.** The 5 trace dimensions identify WHICH row contains the answer. The object/date/entity field of that row provides the precise text for LOCOMO scoring. These are two separate steps. Never skip the first.

---

## REMEMBER

The brain doesn't search rows. It converges traces.

The brain doesn't match one field at a time. It matches all dimensions simultaneously.

The brain doesn't threshold. The trace with the most dimensional overlap wins.

The brain doesn't guess. It either converges or it doesn't.

**Traces, not edges. Convergence, not search. Dimensions, not fields.**

Build the reconstruction engine the same way.