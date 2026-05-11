# Retrieval Verification Loop — Definitive Design
**Author:** Sam (via audit session 2026-04-29 to 2026-05-01)
**Status:** Authoritative spec. Implementation in retrieval.py is WRONG — needs rewrite to match this.

---

## Root causes diagnosed (from reading retrieval.py lines 546-745):

1. **Line 642:** verb class "be" doesn't match surface token "is" — comparing lemmas vs surface forms
2. **Line 669:** "no fact → trust cosine" is a fallback Sam explicitly said to remove
3. **Lines 660-668:** only compares object to fact value, never checks subject/predicate coherence
4. **Lines 596, 610:** regex used for entity extraction
5. **Line 556:** grammar engine imported in facts fast-path

---

## The Design

### What the verification loop does:

Cosine generates candidates. The loop checks each one. First to pass BOTH checks = answer. All rejected = refusal.

### The check (EVERY candidate, no exceptions, no branches):

**Check 1 — COHERENCE (no DB needed):**

Turn the edge into a fact. Does that fact answer the query?

- Does the edge's SUBJECT match the query's entity?
  - "Sam's dad" ≠ "Sam" → REJECT
  - "Sam" == "Sam" → PASS
- Does the edge's PREDICATE relate to what the query asks?
  - "retired from" doesn't answer "where does Sam work" → REJECT
  - "work_at" answers "where does Sam work" → PASS
  - "feel" answers "how does Sam feel" → PASS
  - "workout" ≠ "work" → REJECT

**Check 2 — EXISTENCE (DB lookup):**

Does this fact exist as a CURRENT fact in the DB?

- is_current = 1? Not tombstoned?
- For stative facts: does the facts table value match the candidate's object?
  - career::WORK::Sam = "Apple". Candidate says "Apple" → MATCH → ACCEPT
  - career::WORK::Sam = "Apple". Candidate says "Google" → old value → REJECT

---

## Critical Rules

1. **NO fallbacks.** If coherence fails → reject. Period. No "else trust cosine."
2. **NO branches** for fact-tier vs episodic. Same two checks for ALL candidates.
3. **NO grammar engine imports** in the read path. All signals come from DB fields + string ops.
4. **Rejected candidates** go into `rejected_ids` set. NEVER revisited.
5. **When the right answer is found** → WRITE BACK the query as a new PQ for that edge (self-improving retrieval).
6. **All candidates exhausted** → refusal: "This information is not mentioned in the conversation."
7. **The coherence check is COMPULSORY.** It fires on EVERY candidate. It is not a feature. It is not optional. It is the core of the loop.

---

## Walkthrough (Sam's exact example)

```
Query: "Where does Sam work?"

Candidate 1: (Sam's dad, retire_from, Boeing)
  Make into fact: "Sam's dad retired from Boeing"
  Coherence: Subject "Sam's dad" ≠ query entity "Sam" → REJECT
  (Never reaches existence check)

Candidate 2: (Sam, workout, routine)
  Make into fact: "Sam works out routinely"
  Coherence: Predicate "workout" ≠ query verb "work" → REJECT

Candidate 3: (Sam, love, woodworking)
  Make into fact: "Sam loves woodworking"
  Coherence: Predicate "love" ≠ "work" → REJECT

Candidate 4: (Sam, work_at, Google) [is_current=0]
  Make into fact: "Sam worked at Google"
  Coherence: Subject=Sam ✓, Predicate=work_at ✓ → COHERENT
  Existence: is_current=0 → NOT CURRENT → REJECT

Candidate 5: (Sam, start_at, Apple) [is_current=1]
  Make into fact: "Sam started at Apple"
  Coherence: Subject=Sam ✓, Predicate=start_at relates to work (same schema) ✓ → COHERENT
  Existence: is_current=1 ✓, facts table career::WORK::Sam = "Apple" ✓ → ACCEPT
  
  → Return "Apple"
  → Write "Where does Sam work?" as PQ for this edge
```

---

## Same check for non-fact queries

```
Query: "How does Sam feel about the interview?"

Candidate: (Sam, feel, nervous) [emotional_target=interview]
  Make into fact: "Sam feels nervous about the interview"
  Coherence: Subject=Sam ✓, Predicate=feel ✓ → COHERENT
  Existence: is_current=1, not tombstoned → EXISTS → ACCEPT

Candidate: (Sam's dad, retire_from, Boeing)
  Coherence: Subject "Sam's dad" ≠ "Sam" → REJECT
```

```
Query: "What did Sam do yesterday?"

Candidate: (Sam, have_coffee_with, Jake)
  Make into fact: "Sam had coffee with Jake"
  Coherence: Subject=Sam ✓, Predicate=have ✓ → COHERENT
  Existence: is_current=1 → ACCEPT

Candidate: (Sam's dad, retire_from, Boeing)
  Coherence: Subject "Sam's dad" ≠ "Sam" → REJECT
```

---

## What went wrong with previous implementations

1. Agent added 5 steps with fallbacks — Sam said ONE check (two sub-checks), no fallbacks
2. Agent made coherence check optional ("if fact exists") — Sam said COMPULSORY on every candidate
3. Agent used grammar engine imports — Sam said pure DB + string ops
4. Agent compared object to fact value WITHOUT checking subject/predicate coherence — missed the core
5. Verb class token matching used surface forms ("is") vs lemmas ("be") — never matched
6. Agent added regex for entity extraction — Sam said no regex in business logic
7. Agent created a separate wrapper file instead of integrating into retrieval.py

---

## Predicate relevance — how "start_at" relates to "work"

The question: how does the coherence check know start_at relates to work without grammar engine?

Answer: The facts table already resolved this at WRITE TIME. The fact key is `career::WORK::Sam`. Both "work_at" and "start_at" were classified as VerbClass.WORK by the grammar engine at write time. The read path doesn't re-derive the relationship — it compares the candidate's object against the facts table canonical value. If they match, the candidate is correct regardless of its predicate surface form.

For non-fact queries: the edge's `edge_schematic_category` was assigned at write time. Both "work_at" and "start_at" edges have schema="career". The coherence check can compare schemas without calling the grammar engine — the schema is already on the edge.

---

## Test scores (honest)

- Perfect data (isolation): 23/25 (92%) — moat pipeline only, no verification
- Hard ambiguity (50 edges): 6/30 (20%) — moat pipeline baseline
- With partial verification (value comparison only): 12/30 (40%)
- With full coherence check: NOT YET IMPLEMENTED CORRECTLY
- Target: 85-90% on hard ambiguity with correct coherence check

---

## Files

- Retrieval engine: `app/engines/retrieval.py` (~1856 lines)
- Memory engine: `app/engines/memory.py` (~1571 lines)  
- Grammar engine: `app/engines/grammar_engine.py` (~4143 lines)
- Facts table schema: `app/db/models.py`
- Predicted queries: `app/engines/predicted_queries.py`
- Tests: `tests/test_retrieval_isolation.py`, `tests/test_retrieval_ambiguity_hard.py`
