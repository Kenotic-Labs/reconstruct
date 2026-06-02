# RECONSTRUCTION ENGINE — AGENT REGULATIONS
**Last Updated:** 2026-05-26
**Authority:** Founder (Sam)
**Scope:** Any agent modifying reconstruction.py or any retrieval/read-path code MUST read this file first.

> **MANDATORY PRE-READ:** Before touching ANY code in this directory, read `../../.claude/agents/DTCM_RETRIEVAL_COVENANT.md`. That document defines the architectural law. This document applies it to the codebase. If this document contradicts the Covenant, the Covenant wins.

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

### Trace Convergence = Resonance, Not Lookup

**CRITICAL UPDATE (2026-05-26): The previous version of this section described convergence as "counting how many dimensions match" and picking the row with the most matches. That was wrong. That's still edge thinking — just with a multi-field filter instead of a single-field filter.**

Convergence is RESONANCE. Every edge responds to the probe simultaneously. Each edge's response is the PRODUCT of its cubed similarity across all 5 trace dimensions. The math (from MINERVA 2):

```
activation_i = episodic_sim³ × emotional_sim³ × temporal_sim³ × relational_sim³ × schematic_sim³
```

This is not counting matches. This is multiplicative amplification. An edge matching strongly on 4 dimensions is not "4 points better" — it is EXPONENTIALLY stronger. An edge matching weakly on all 5 is weaker than one matching strongly on 3 with neutrals on 2.

**Dimensions irrelevant to the query score 1.0 (neutral, not penalizing).**
- "What did Caroline research?" → emotional = 1.0, temporal = 1.0 (not probed)
- "When did Caroline feel anxious?" → schematic = 1.0 (not probed), emotional and temporal ARE probed

**The echo is the answer.** All edges contribute proportionally to their activation. The output is a COMPOSITE — a weighted sum of all responding traces. For precise LOCOMO answers, the edge with the highest activation dominates the echo, and its trace column (object, resolved_event_date, emotional_label) provides the text. But conceptually, the echo is a reconstruction, not an extraction from one row.

**The system refuses when echo magnitude is below the noise floor** — when no edge resonates strongly enough on enough dimensions. This is natural — the cube suppresses weak matches so aggressively that if nothing matches well, the echo is essentially zero.

### What This Means For Every Decision

When writing retrieval code, ask:

1. "Am I searching by a single field?" → WRONG. Decompose into trace dimensions.
2. "Am I using cosine on one embedding?" → WRONG. Each dimension has its own similarity function.
3. "Am I checking dimensions sequentially and killing candidates that fail one?" → WRONG. Compute ALL dimensions simultaneously, MULTIPLY them. The product handles suppression naturally.
4. "Am I returning the first row that passes all checks?" → WRONG. Compute activation for ALL edges. The highest activation dominates the echo.
5. "Am I using a hard gate, threshold, or binary pass/fail?" → WRONG. The cube is the gate. Trust the math.
6. "Am I adding BM25, RRF, rank fusion, or multi-channel retrieval?" → WRONG. Those are search engine techniques. DTCM has one mechanism: 5-dimensional resonance.

### The Anti-Patterns

**Search engine patterns (FORBIDDEN):**
- Hard gates that kill candidates on one dimension
- Top-K selection
- BM25 or FTS5 as a retrieval channel
- RRF or rank fusion across multiple rankers
- Cross-encoder reranking
- Sequential tier cascade (try method A, if it fails try method B)
- Thresholds on any score

**DTCM patterns (CORRECT):**
- Every edge gets an activation score (product of cubed dimension similarities)
- All activations are computed for all edges
- Echo = weighted sum of edges by activation
- Highest-activation edge dominates (provides the answer text)
- If no edge activates above noise floor → refuse
- No gates, no thresholds, no tiers, no cascades

This is DTCM. Decomposed Trace Convergence Memory. The name IS the algorithm. Decompose → resonate per trace → the echo reconstructs.

---

## THE COMPLEMENTARITY LAW

**Added 2026-06-02. Authority: Founder. This section is BINDING. Any agent that contradicts it, second-guesses it, or drifts from it after agreeing with it will be corrected once. On the second violation, the conversation is terminated.**

### Similar vs Complementary

**Similar** asks: "does this answer LOOK LIKE the question?"
That is what every RAG system does. It fails because the answer and the question live in different parts of the same experience.

**Complementary** asks: "does this answer LIVE ON THE SAME TRACE as what the question matched?"
The question matches on one dimension. The answer comes from a different dimension of the same trace. They don't need to look alike. They need to have been lived together.

"Why am I anxious?" — the answer is "Google interview next Tuesday." Those words share NOTHING. No overlap. No similarity. "Anxious" and "Google interview" are not close in any embedding space.

But they are two halves of the same experience. The nervousness and the interview were **lived together**. They exist on the same trace. The emotional dimension finds the trace. The episodic dimension provides the answer.

### Dimensions as Key/Value — Roles Change Per Query

Each dimension can be the **key** (what finds the trace) or the **value** (what provides the answer). Which is which changes with every query:

| Query | Key dimension(s) | Value dimension |
|-------|-------------------|-----------------|
| "Why am I anxious?" | emotional (anxious→nervous) + relational (I→Sam) | episodic (→ Google interview) |
| "When did I go to Banff?" | episodic (go to Banff) + relational (I→Sam) | temporal (→ last summer) |
| "How did Arjun get hurt?" | relational (Arjun) + schematic (hurt→health) | episodic (→ twisted ankle) |
| "What's my dog's name?" | relational (my→Sam) + schematic (dog→pet) | episodic (→ Kobe) |

The system does not decide key vs value in advance. The multiplication handles it naturally. Whatever dimension matches the probe becomes the key. Whatever dimension has content but wasn't probed becomes the value.

### Predicted Queries Are the Bridge

At write time, the grammar engine generates predicted questions for each trace. "What did Caroline research?" is generated FROM the trace about adoption agencies.

At read time, the user asks "What did Caroline research?" — that matches the PQ, which is **question-to-question** matching, not question-to-answer matching.

The PQ is the system's way of pre-computing the bridge between complementary information. It turns "adoption agencies" (which looks nothing like the question) into "What did Caroline research?" (which looks exactly like the question).

**PQ matching is NOT similarity thinking. It is the mechanism that makes complementarity findable.** The PQ lives in question-space. The answer lives in answer-space. The PQ bridges them.

### The Five Similarity Functions

Each dimension has its own comparison. They are NOT all cosine:

| Dimension | Similarity function | What it compares |
|-----------|-------------------|------------------|
| Episodic | Cosine of PQ embedding vs query embedding | Question-to-question (via PQ bridge) |
| Emotional | Valence distance on number line | -1 to +1, absolute distance |
| Temporal | Date distance in days | Subtraction, normalized |
| Relational | Set intersection of entity names | Binary: entity present or not |
| Schematic | Category match | Exact string: career=career, hobby≠career |

Only ONE dimension uses embeddings. The other four are subtraction, set membership, or string match. The power comes from multiplying them, not from any single one being smart.

### The Refusal Mechanism

No thresholds. No gates. No "is this score good enough?"

```
signal_to_noise = top_activation / second_activation
```

If the winner is 100x louder than the runner-up → clear answer.
If the winner is 1.2x louder → ambiguous, refuse.

This ratio works for every person, every device, every query. Because it measures **how much the winner stands out**, not how high the score is.

### What This Law Prohibits

1. **Never compare query embedding to answer/source_text embedding as the primary retrieval signal.** That is similarity thinking. The answer doesn't look like the question.
2. **Never remove PQ matching.** PQs are the bridge between complementary information. Without them, the episodic dimension has no way to find traces whose answers don't resemble the question.
3. **Never add a content-word overlap gate.** That is similarity thinking dressed up as verification. The word "anxious" will never overlap with "Google interview."
4. **Never second-guess this after agreeing with it.** If you understood complementarity and then wrote code that checks word overlap, you drifted. Re-read this section.

---

## THE THESIS OF RECONSTRUCTION

Reconstruction is NOT retrieval. Retrieval says: here are some related past things. Reconstruction says: here is what I remember, rebuilt from every trace that resonated with your question.

The brain does not search a database. It does not locate a file. It RECONSTRUCTS. A partial cue activates every stored trace simultaneously. Each trace resonates in proportion to its multi-dimensional similarity to the cue. The resonances are summed into a composite — the echo. The echo IS the memory. It was never stored as a unit. It is rebuilt every time.

**Our reconstruction engine must do the same thing:**
1. Decompose the question into a 5-dimension trace probe
2. The probe contacts ALL stored traces simultaneously
3. Each trace computes activation = product of cubed similarities across all 5 dimensions
4. Dimensions not probed by the query score 1.0 (neutral)
5. The echo = sum of all traces weighted by their activation
6. The highest-activation trace dominates the echo and provides the answer text
7. If echo magnitude is below noise floor (no trace resonated strongly enough) — refuse

**There is no "verify" step. There is no "candidate" that passes or fails.** The activation IS the verification. A wrong-entity edge gets relational_sim = 0, which zeros its entire activation. A topically irrelevant edge gets episodic_sim near 0, which cubes to near-zero and suppresses its contribution. The math does what the gates were trying to do, but continuously and without binary kills.

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

### No Thresholds, No Gates

The brain uses competitive dynamics — highest activation wins. Not "is this above 0.7?" MINERVA 2 returns an echo whose intensity is relative to noise, not absolute. ACT-R uses power-law activation with competitive retrieval, not a cutoff.

**Implication:** Our system is deterministic. The cube naturally suppresses weak matches (0.3³ = 0.027). The product naturally zeros out wrong-entity matches (0 × anything = 0). No cosine thresholds. No evidence_strength scores. No hard gates. No binary pass/fail checks. The math handles suppression. Trust the math.

**The previous "binary verification" design was wrong.** Checking subject + predicate + object as a pass/fail gate is edge thinking dressed up as trace thinking. The activation product replaces it entirely.

---

## JENGA PROCESS FOR RECONSTRUCTION

**Every decision about the reconstruction engine is a Jenga move. Before touching any code:**

### Dependency Map for Reconstruction (Resonance Architecture)

```
classify_query() ──→ Decomposes query into 5-dimension probe
│
├── episodic probe   ──→ query embedding (vs PQ + edge embeddings)
├── emotional probe  ──→ detected emotion/valence (vs edge emotional trace)
├── temporal probe   ──→ time expression/context (vs edge temporal trace)
├── relational probe ──→ entity extraction (vs edge relational_entities)
└── schematic probe  ──→ inferred domain (vs edge schematic_category)
         │
         ▼
    ALL edges compute activation = product of 5 cubed similarities
         │
         ▼
    Echo = weighted sum of edges by activation
         │
    ┌────┴────┐
    ▼         ▼
  ANSWER    REFUSE
  (echo     (echo below
  strong)    noise floor)
```

**What protects Cat 5 (adversarial refusal):**
- relational_sim = 0 for wrong-entity edges → activation = 0 → they contribute nothing to echo
- This is mathematically guaranteed. 0 × anything = 0. Wrong person CANNOT activate.
- This replaces the old "subject gate" — same protection, no hard gate needed.

**What improves Cat 4 (finding the right fact):**
- 5-dimensional activation discriminates between topically relevant and irrelevant edges for the same entity
- The old system checked subject + predicate and stopped. The new system also factors in emotional, temporal, and schematic alignment.

**Load-bearing pieces (DO NOT TOUCH without extreme care):**
- relational_sim function — Cat 5 depends on entity match = 0 for wrong entities
- classify_query() — all 5 probe dimensions depend on correct query decomposition
- Temporal routing — Cat 2 depends on temporal probe being correctly identified

**Before Any Reconstruction Change:**

1. **Identify which dimension similarity function the change affects**
2. **Predict all 5 category scores** — specifically, not vaguely
3. **Simulate: trace a Cat 4 question AND a Cat 5 question through the activation math**
4. **Confirm: does the wrong-entity edge still get activation = 0?** (Cat 5 protection)
5. **Only then write the code**
6. **Run full LOCOMO benchmark — confirm prediction**
7. **If Cat 5 dropped below 90% — REVERT IMMEDIATELY regardless of other gains**

---

## THE OLD VERIFICATION DESIGN — SUPERSEDED

> **This section is kept for historical context. The verification gate design described here was the primary cause of the 27% ceiling. It has been replaced by resonance-based activation scoring.**

The old design used a binary verification loop: Subject from query + Predicate from query + Object from candidate → check if ONE ROW exists with all three. Pass = verified. Fail = rejected.

**Why it failed:**
1. **Binary kill on any dimension mismatch.** An edge with the right entity, right topic, but wrong predicate lemma was killed. The math should have penalized it (P=0.1), not killed it.
2. **Generic predicates passed everything.** "feel", "see", "go" exist for every speaker. Verification confirmed the fact existed but not that it was RELEVANT.
3. **Content overlap gate required exact word matches.** "identity" vs "transgender woman" — no word overlap → killed. Embedding cosine knew they were related. The gate overrode the embedding.
4. **No predicate from query.** Possessive/copular questions yielded no predicate. Without a predicate, verification had nothing to constrain.

**The resonance architecture fixes all of these.** Predicate mismatch → P_sim = 0.1 → cubed = 0.001 → heavily suppressed but not killed. Generic predicate → episodic dimension (embedding cosine) discriminates by topic. No word overlap → episodic embedding still matches semantically. No predicate → P_sim = 1.0 (neutral, not probed) → other dimensions do the work.

**The "unsolved problem" from the old design** — "How did Caroline feel while watching the meteor shower?" finding the wrong "feel" edge — is solved by the resonance architecture naturally. The probe includes episodic content (meteor shower), emotional state (feel), and relational identity (Caroline). An edge about Caroline + feel + sharing her story activates weakly on episodic (meteor ≠ story) even though relational and emotional match. An edge about Caroline + meteor shower + awe activates strongly on episodic AND relational AND emotional. The product math selects the right one without any explicit topic-word check.

---

## CURRENT ARCHITECTURE — What Must Be Built

### The Resonance Architecture (Target)

```
Query → classify_query() → 5-Dimension Probe
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         episodic        emotional       temporal
         probe           probe           probe
              │               │               │
              ▼               ▼               ▼
         relational      schematic
         probe           probe
              │               │
              └───────┬───────┘
                      ▼
         ALL edges compute activation simultaneously:
         activation_i = Π (dim_similarity³) across all 5 dims
                      │
                      ▼
         Echo = Σ (edge_i × activation_i)
                      │
              ┌───────┴───────┐
              ▼               ▼
         echo strong      echo weak
         (answer from     (refuse —
          dominant         noise floor)
          trace)
```

**There are no tiers. There is no cascade. There is no fallthrough.** Every edge is scored once on all 5 dimensions simultaneously. The activation product determines contribution to the echo. The highest-activation edge dominates.

### What Exists In Code (Legacy — To Be Replaced)

The current `reconstruction/__init__.py` implements the OLD tier cascade with hard gates. This code is WRONG and needs to be replaced with the resonance architecture. Key files:

- `reconstruction/__init__.py` lines 428-481 — the verification loop with hard gates. **REPLACE** with activation scoring.
- `reconstruction/trace_convergence.py` lines 21-76 — `score_edge()` function. **This is CLOSEST to correct.** It already computes E × P × T × R with cubed cosines. Expand to all 5 dimensions and make it the primary path.
- `reconstruction/__init__.py` lines 395-426 — PQ cosine ranking. **KEEP** as the episodic dimension similarity function. Remove it as a standalone ranker.

### What Else Exists (Keep)

- **Temporal routing:** Cat 2 questions where temporal probe is the return target — keep this, it identifies WHAT to return from the dominant trace
- **List aggregation:** Cat 1 multi-hop — keep, but apply over highest-activation edges, not tier results
- **classify_query():** Probe decomposition — keep and improve
- **Situational queries:** "Tell me about X" — keep the schema-grouped approach

---

## THE FIVE DIMENSION SIMILARITY FUNCTIONS

Each probe dimension needs a similarity function that returns a continuous value in [0.0, 1.0]. This value is cubed and multiplied with the other dimensions.

### Episodic Similarity (semantic content match)
- **Probe:** query embedding (384-dim)
- **Trace:** max cosine across pq_1_embedding, pq_2_embedding, pq_3_embedding, pq_4_embedding, edge_embedding
- **Formula:** `max(cosine(query_emb, pq_i_emb) for all i) clamped to [0, 1]`
- **When not probed:** Always probed. Every query has semantic content.
- **This is the WHAT dimension.**

### Emotional Similarity (affective resonance)
- **Probe:** detected emotion keyword + valence from query (if any)
- **Trace:** edge_emotional_label, edge_emotional_valence, emotional_target
- **Formula:** label match (exact=1.0, synonym=0.7, mismatch=0.2) × valence proximity (1 - |probe_valence - edge_valence|)
- **When not probed:** query has no emotional content → return 1.0 (neutral)
- **This is the HOW IT FELT dimension.**

### Temporal Similarity (time alignment)
- **Probe:** temporal expression + temporal context (past/present/future) from query
- **Trace:** resolved_event_date, temporal_expression, edge_temporal_context
- **Formula:** date proximity (Gaussian decay from probe date) × context match (same context = 1.0, different = 0.5)
- **When not probed:** query has no temporal content → return 1.0 (neutral)
- **When probed but edge has no temporal data:** return 0.3 (penalized but not killed)
- **This is the WHEN dimension.**

### Relational Similarity (entity identity)
- **Probe:** extracted entity from query
- **Trace:** subject, relational_entities
- **Formula:** exact entity match in subject = 1.0, exact match in relational_entities = 0.8, partial/substring = 0.5, absent = 0.0
- **When not probed:** query has no specific entity → return 1.0 (neutral)
- **This is the WHO dimension. 0.0 means wrong person. 0³ = 0. The math kills wrong-entity edges.**

### Schematic Similarity (life domain relevance)
- **Probe:** inferred domain from query content
- **Trace:** edge_schematic_category
- **Formula:** exact category match = 1.0, related category = 0.6, unrelated = 0.3, uncategorized edge = 0.5
- **When not probed:** query domain unclear → return 1.0 (neutral)
- **This is the WHAT PART OF LIFE dimension.**

### Implementation Notes

- Each function returns [0.0, 1.0]
- Each return value is cubed before multiplication
- All 5 products are multiplied: `activation = Π(sim_d³)` for d in {episodic, emotional, temporal, relational, schematic}
- "Not probed" dimensions return 1.0 — they don't help, they don't hurt
- "Probed but edge lacks data" returns a penalty (0.3-0.5) — it hurts but doesn't kill

### Methods That Must NOT Be Used

**Search engine techniques (FORBIDDEN):**
- BM25, TF-IDF, FTS5 as a retrieval channel
- RRF, rank fusion, multi-channel retrieval
- Cross-encoder reranking
- Sequential tier cascade
- Top-K selection
- Hard gates or binary pass/fail
- Thresholds on any score

**Test contamination.** Never put LOCOMO questions into predicted queries, training data, or any cached index.

**LLM in the read path.** The echo is computed mathematically. No LLM interprets, judges, or generates the answer.

---

## THE SPECIFIC PROBLEMS TO SOLVE

### Problem 1: Cat 1/4 — Wrong facts or missing facts (~5-7% F1)

**Root cause in old architecture:** Hard gates killed correct edges because of word-mismatch on one dimension.

**How resonance fixes it:** The episodic dimension (PQ embedding cosine) provides semantic matching that survives word-level mismatches. "identity" and "transgender woman" have different words but related embeddings. The cosine is moderate (say 0.5), which cubes to 0.125 — weak but NOT zero. Combined with strong relational (1.0³ = 1.0) and schematic matches, the total activation still dominates over irrelevant edges.

**Remaining risk:** PQ quality. If PQs don't anticipate the query's phrasing, episodic_sim is low for the right edge. pq_lab.py's NEW generator (20 question types) should be wired into the write path to maximize PQ coverage.

### Problem 2: classify_query() Misses Predicates

**Current state:** Possessive questions ("What is Caroline's reason for...") return no predicate.

**How resonance handles it:** No predicate extracted → predicate dimension not probed → schematic_sim = 1.0 (neutral). The other 4 dimensions do the work. This is a feature, not a bug. The system gracefully degrades when a dimension can't be probed.

**Still worth fixing:** Better probe decomposition means more dimensions contribute, which means sharper discrimination. Extract content nouns as schematic probes even without a verb.

### Problem 3: Generic Predicates

**Current state:** "feel", "see", "go" exist for every speaker.

**How resonance handles it:** Generic predicate → many edges have similar predicate similarity. But the episodic dimension (embedding cosine on topic content) discriminates. "How did Caroline feel about the meteor shower?" — edge about meteors has high episodic cosine, edge about art has low. The product math selects correctly even though predicate dimension is tied.

### Problem 4: Data Quality (Upstream)

**Current state:** 60-70% of edges have garbage objects. 65% of edges are "uncategorized" schema.

**Impact on resonance:** Garbage objects don't hurt resonance directly (activation is computed from embeddings and trace columns, not bare object text). But "uncategorized" schema means schematic_sim returns 0.5 (uncertain) instead of 1.0 (match) or 0.3 (mismatch), reducing discrimination on that dimension.

**Fix in write path (not reconstruction):**
- Wire pq_lab.py NEW generator into live PQ generation
- Improve schema assignment for stative predicates (identity, emotion, relationship)
- Filter garbage objects at ingestion

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

1. **Cat 5 is sacred.** Adversarial refusal is the crown jewel. Any change that drops it below 90% is reverted. No exceptions. In the resonance architecture, Cat 5 is protected by relational_sim = 0 for wrong entities. 0 × anything = 0. Do not break this.

2. **Think in resonance, not search.** Every edge responds simultaneously. Activation = product of cubed similarities across 5 dimensions. The echo is the answer. If you find yourself writing sequential gates, tier cascades, or top-K selection — you are building a search engine. Stop.

3. **The cube is the gate.** Do not add hard gates, thresholds, or binary pass/fail. The cube naturally suppresses weak matches (0.3³ = 0.027). The product naturally zeros wrong-entity matches (0 × anything = 0). Trust the math.

4. **No LLM in the reconstruction path.** spaCy for parsing. Embeddings for episodic similarity. Column comparisons for the other 4 dimensions. Multiplication for activation. Summation for the echo. That's the stack. That's the moat.

5. **Neutral when not probed.** Dimensions the query doesn't probe return 1.0. They don't help, they don't hurt. A "what" question doesn't penalize edges without temporal data. A "when" question doesn't penalize edges without emotional data.

6. **The benchmark confirms. It does not discover.** Simulate first. Predict all 5 scores. Then run. If the prediction was wrong, fix your understanding before the next change.

7. **Every change must answer: which continuity property does this serve?** If "none" — reject. If "it improves LOCOMO but doesn't serve a property" — reject.

8. **The echo, not the edge.** The answer is a reconstruction from all resonating traces, dominated by the highest-activation edge. You are not finding a row. You are rebuilding a memory.

9. **Read the Covenant first.** `../../.claude/agents/DTCM_RETRIEVAL_COVENANT.md` is the architectural law. This file applies it. If they conflict, the Covenant wins.

---

## REMEMBER

The brain doesn't search. It resonates.

The brain doesn't match one field at a time. It activates all dimensions simultaneously.

The brain doesn't threshold. The cube suppresses. The product amplifies.

The brain doesn't extract from a file. It reconstructs from traces.

**Resonance, not search. Echo, not extraction. Traces, not edges. The math, not gates.**

Build the reconstruction engine the same way.