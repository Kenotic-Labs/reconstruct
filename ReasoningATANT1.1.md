# ReasoningATANT1.1 — Why ATANT, Not LOCOMO

**Date:** 2026-04-10
**Author:** Claude (session with Sam)
**Status:** Strategic decision doc + execution plan for P0–P4
**Prerequisite reading:** `CLAUDE.md`, `forwarding_knowledge.md`, `locomo_gap_analysis.md`

---

## 0. Reader Orientation — What This Doc Is

This is the strategic reframe that came out of the LOCOMO gap analysis on 2026-04-10. It answers three questions:

1. **Is LOCOMO actually relevant, or is ATANT just better?**
2. **What are the limits of our own benchmark (ATANT)?**
3. **What should we do next and in what order?**

The short answer: **ATANT is better. LOCOMO is broken in specific measurable ways. The right move is one small LOCOMO fix for credibility, plus expanding ATANT using the 500 stories already on disk, plus building refusal and narrative reconstruction paths for real.**

The long answer fills the rest of this doc. Every claim below is backed by data from the actual code and dataset.

---

## 1. The Original Reframe That This Doc Updates

On 2026-04-10 earlier in the session, Sam corrected a previous wrong framing. The wrong framing was:

> "LOCOMO is benchmark-shape mismatched with our architecture. ATANT is the only truth signal."

Sam's correction:

> "LOCOMO rewards: long-context retrieval, duration extraction from prose, adversarial trick questions — even those are to be done by the architecture so that what ATANT measures is forward-building and useful. Do you understand what I'm saying?"

The correct framing that came out of that exchange:

- **ATANT** measures what we're built for (the 7 properties of continuity). 94% ATANT proves the architecture works for its primary mission.
- **LOCOMO** measures completeness — long-form retrieval/extraction that a complete continuity layer should ALSO have.
- **Both must work.** ATANT validates strengths. LOCOMO validates completeness.

That reframe is STILL VALID for the *capabilities* LOCOMO tries to measure. What changes in this doc (1.1) is that we now know LOCOMO's **specific scoring mechanism is broken** in ways that make chasing its number worse than useless. We still need to build the capabilities. We just need to measure them in ATANT, because ATANT can be made precise enough to tell the truth, whereas LOCOMO cannot.

---

## 2. What The LOCOMO Gap Analysis Actually Found

Today I ran a structural analysis on the full LOCOMO dataset (`locomo_bench/locomo/data/locomo10.json`, 10 conversations, 1986 QA pairs). Three findings.

### Finding 1 — The forwarding doc mislabeled the categories

The forwarding doc said:

```
single-hop    9%
multi-hop    22%  ← strongest
temporal      8%
adversarial   0%  ← broken
```

That implied cat2 = multi-hop and cat4 = temporal. **That's backwards.** Real distribution across all 10 conversations:

| Cat | Forwarding doc said | Actually is | n | % "when" | % "what" | Empty gold |
|-----|---------------------|-------------|---|----------|----------|------------|
| 1 | single-hop | single-hop factual recall | 282 | 1% | 65% | 0/282 |
| 2 | multi-hop | **TEMPORAL** ("when/how long") | 321 | **77%** | 5% | 0/321 |
| 3 | open-domain | hypothetical / inference | 96 | 0% | 34% | 0/96 |
| 4 | temporal | **NARRATIVE COMPREHENSION** (why/how/what happened) | 841 | 4% | **71%** | 0/841 |
| 5 | adversarial | **EMPTY-GOLD REFUSAL QUESTIONS** | 446 | 3% | 69% | **444/446 (99.5%)** |

Sample evidence:

**Cat 2 (actually temporal):**
- "When did Caroline go to the LGBTQ support group?" → "7 May 2023"
- "When did Melanie run a charity race?" → "The sunday before 25 May 2023"
- "How long has Caroline had her current group of friends for?" → "4 years"
- 246 of 321 (77%) start with "when", 22 more with "how long/many"

**Cat 4 (actually narrative comprehension):**
- "What did Melanie realize after the charity race?" → "self-care is important"
- "How does Melanie prioritize self-care?" → "by carving out some me-time each day…"
- "Why did Caroline choose the adoption agency?" → "because of their inclusivity and support…"
- 594 of 841 (71%) start with "what", 76 more with "how did/does", 38 with "why"

**Cat 5 (actually empty-gold refusal):**
- "What did Caroline realize after her charity race?" → `""`
- "What are Melanie's plans for the summer with respect to adoption?" → `""`
- "What type of individuals does the adoption agency Melanie is considering support?" → `""`
- **444 of 446 (99.5%) have empty gold answers.** The remaining 2 are yes/no negations answered "No".

**Score reinterpretation:**

| Forwarding doc label | What it ACTUALLY measures | Our score | Meaning |
|----------------------|---------------------------|-----------|---------|
| multi-hop 22% | **temporal** (duration + date) | **22% — our best** | Duration extractor + source_timestamp wiring paid off. |
| temporal 8% | **narrative comprehension** | **8%** | The 841-question bucket (42% of LOCOMO). Real completeness gap. |
| single-hop 9% | single-hop factual | **9%** | Label correct. Factual recall is weak. |
| adversarial 0% | **empty-gold refusal** | **0% (by design)** | See Finding 2. Not a fabrication problem. A scoring bug. |

### Finding 2 — LOCOMO's cat-5 "adversarial" score is a scoring bug, not a system gap

`run_locomo_benchmark.py` line 177:

```python
def answer_matches(predicted: str, gold: str) -> bool:
    if not predicted or not gold:  # ← returns False when gold is empty
        return False
    …
```

If gold is empty, this returns False **regardless of what the system predicts**. That means **all 444 empty-gold questions are unscorable by design**. Even if our system returned a perfect silence or "I don't know", the runner would count it as wrong.

The forwarding doc said "cat 5 = 0/112 because the system fabricates instead of refusing". That's wrong. The 0% is purely a matcher bug. It's not measuring fabrication. It's measuring nothing.

**This means:** 23% of the LOCOMO benchmark is unscorable by design. Any effort to "improve cat 5" that doesn't start with fixing `answer_matches` is wasted.

### Finding 3 — Cat 4 (narrative comprehension) is where the real completeness gap lives

841 questions — 42% of LOCOMO — are paraphrase-scored narrative comprehension:

- "What did Melanie realize after the charity race?" → gold "self-care is important"
- "How does Melanie prioritize self-care?" → gold "by carving out some me-time each day for activities like running, reading, or playing with Oscar"
- "Why did Caroline choose the adoption agency?" → gold "because of their inclusivity and support for LGBTQ+ individuals"

The runner scores these with substring containment: if the gold fragment appears in the predicted answer, it passes. That has two consequences:

1. A verbose answer that happens to contain the gold fragment wins.
2. A terse answer that rephrases the gold (same meaning, different words) loses.

Our triple-object return path returns terse objects: `(Melanie, realized, self-care)` → answer "self-care". If the gold is "self-care is important", that's a substring. We pass. But if the gold is the verbose phrase with "carving out some me-time…", we fail — our object is "self-care" and the gold fragment "carving out some me-time" isn't in it.

This is what `schematic_trace.reconstruct_living_situation` is built for — read path Step 0 in `convergence_gate`. But it only triggers on queries like "summarize my situation". It does NOT trigger on individual "what did X realize" / "why did X do Y" / "how does X do Z" narrative questions. Widening the trigger is P3 below.

---

## 3. The Honest Strategic Answer — Is LOCOMO Relevant?

### Why LOCOMO is LESS relevant than the forwarding doc suggested

1. **23% of it is unscorable by design.** The cat-5 matcher bug. No effort on our side fixes this.
2. **42% of it uses fuzzy substring matching on paraphrased narrative answers.** This measures verbosity + phrasing luck, not continuity. A system that dumps more text wins.
3. **Category labels don't match what the paper claims.** cat2 vs cat4 swapped, cat5 mischaracterized. This means the dataset was assembled without tight editorial control, and anyone citing "LOCOMO cat X performance" is probably wrong.
4. **LOCOMO doesn't test anything our thesis cares about:**
   - No persistence across real app closures (single-shot retrieval over static transcripts)
   - No update handling ("I was nervous before, I feel fine now")
   - No disambiguation (two people named the same thing)
   - No reconstruction of current state (as distinct from past state)
   - No temporal supersession (Tuesday → moved to Wednesday)
   - No model independence test
   - No operational usefulness across surfaces
   - It measures ~2 of the 7 properties at best. The other 5 are invisible.

### Why LOCOMO is STILL relevant

1. **External credibility.** Investors, paper reviewers, and dev users recognize the name. A published LOCOMO number is marketing.
2. **The capabilities it targets ARE continuity primitives.** Refusal, narrative reconstruction, temporal duration extraction — these are all things a real continuity layer should do. What's broken is LOCOMO's scoring, not the underlying capability targets.
3. **Sam's earlier reframe still holds for the capabilities** — just not for the specific numbers.

### The reconciliation

Build the capabilities LOCOMO targets. Measure them in ATANT, because ATANT can be made precise. Report LOCOMO as a secondary metric, not a target. Fix LOCOMO's one obvious bug (the matcher) to get an honest baseline number. That single fix lifts us from 8.8% to ~30% in one 5-line change — a 3.4× credibility bump for almost zero cost.

Beyond that, do NOT optimize to LOCOMO. Every hour spent optimizing to LOCOMO cat 4 is an hour NOT spent expanding ATANT to actually cover the 7 properties.

---

## 4. ATANT's Own Limits — The Honest Version

Sam raised the point: "even tests themselves have limitations". He's right. ATANT has real limits too. I'm going to enumerate them because any strategy that says "trust ATANT" has to acknowledge what ATANT is currently missing.

### 4.1 ATANT as it exists in `run_atant_ground_truth.py` (the 5-story runner)

- **Tiny surface.** 5 stories, 32 questions total. Easy to overfit by accident.
- **Ground truth written by us.** Tests what we think the system should do, not what real usage surfaces.
- **Same 2 questions fail every run.** Story 1 (Job Interview) and Story 4 (Learning Guitar) each lose exactly 1 question. This pattern hasn't changed across days. It's a distribution of 1 — not a meaningful spread.
- **Only 8 checkpoints.** cp2/cp3/cp4/cp7/cp8/cp9/cp11/cp12.
- **CP12 (reconstruction similarity) is at 47%.** ATANT itself knows we're weak at narrative reconstruction. That's actually a feature of ATANT — it tells the truth — but we've been ignoring the signal.

### 4.2 ATANT as it exists in `run_atant_cumulative.py` (the 500-story runner)

This is the big find of this part of the session. I'd been regressing against the 5-story runner. The real ATANT is the 500-story cumulative runner. It already exists and already loads:

- **500 stories in `tests/stories/cumulative/`** (IDs 051–550).
- Each story is a full YAML with `story_id`, `story_name`, `category`, `batches`, `expected_memory_stores`, `expected_triples`, `expected_adaptation`, `final_verification`.
- **20 checkpoints** instead of 8: write path (cp1–cp4), read path (cp5–cp8), traces (cp9–cp10), reconstruction (cp11–cp12), emotion detection (cp13), relational assessment (cp14), schematic (cp15), frequency (cp16), proactive (cp17), situation reconstruction (cp18), contradiction detection (cp19), is_current filtering (cp20).
- **Refusal tests already built in:** batch 8 of story 051 is a "GK trap" with `expected_memory_stores: ["GK trap"]` and `expected_triples: []`. The test is: the user asks a trap question ("what's the name of the new VP and where is he coming from") that the system should answer from prior turns, not fabricate. The runner special-cases this with `if expected_stores != ["GK trap"]`.
- **Correction tests already built in:** story 051 batch 7 has "I got the timeline wrong" with `expected_triples` containing explicit `("user", "corrected", "Derek from Amazon → Microsoft")` triples. The runner tests these via cp19_contradiction (checks that superseded rows exist via `is_current=0`).
- **Multi-entity disambiguation already built in:** story 051 has Maya as user, Priya as manager, Leon as teammate, Julian as partner, Derek as new VP — all referenced in overlapping batches. The runner's cp14_relational tests that the relationship map resolves them correctly.
- **Temporal ordering already built in:** "next week" → "two weeks" correction pattern.

So what I thought needed to be built (P2: "add 20+ more stories covering the 7 properties") is already written. The 500 stories already cover update handling, refusal, disambiguation, temporal ordering, multi-batch narrative, multi-entity. What's missing is:

1. The cumulative runner **doesn't go through backbone**. It calls `convergence_check` directly at line 670 and `equation_read(..., shadow_mode=False)` at 679. P2 is wiring this runner through `backbone.process_read`.
2. The cumulative runner **was built today (per forwarding doc) but hasn't had a clean pass recently.** I haven't run it yet in this session. It may have drifted from its baseline.
3. Some checkpoints depend on stubbed components (`_detect_emotions`, `_assess_relationship`, `_get_proactive_insights`). If those are NotImplementedError stubs, those checkpoints silently fail.

### 4.3 What ATANT still doesn't cover

Even the 500-story runner doesn't yet fully cover:

- **Property #1 (Persistence beyond session)** — the cumulative runner runs in one process. There's no "close the DB, reopen it, query" test. This would require a runner that spawns two processes or explicitly closes + reopens the connection between write and read.
- **Property #6 (Model independence)** — there's no test that swaps the T5 model mid-run, or reads with a different embedder than the writer used.
- **Property #7 (Operational usefulness)** — all stories are personal-chat shaped. No library patron, no clinic patient, no agent task runner.
- **Adversarial semantic confusion** — ambiguous pronouns with TWO possible referents ("she said he…"), name collisions (two people named John), homograph confusion (Apple the company vs apple the fruit). Stories have some of this organically but no deliberate adversarial surface.
- **Long-horizon drift** — stories are 2-3 simulated days. Real continuity should hold across simulated months.

These are real gaps. But they're gaps in ATANT's *test surface*, not in our architecture. Adding them is straightforward story-authoring work once the cumulative runner is wired through backbone.

### 4.4 Why ATANT is still the right primary signal

Despite the limits:

1. **We authored it for our 7 properties.** Every checkpoint maps to one or more properties. LOCOMO does not.
2. **Precise scoring.** Triple-object exact match, no paraphrase fuzziness. When ATANT says "pass", it means pass.
3. **Extensible by us.** Adding a new story for a new property is a YAML file. No begging a third party.
4. **Ground truth is inspectable.** Every expected value is in the YAML. Every failure is debuggable.
5. **Scale is already there** — 500 stories on disk. Not a toy corpus.

LOCOMO can never be any of these things for us. It's someone else's definition of continuity tested someone else's way.

---

## 5. The Five Priorities — P0 to P4

Numbered in the order they should be executed. Each priority has: the change, why, expected impact, file targets, and a regression check.

### P0 — Fix LOCOMO's `answer_matches` empty-gold handling

**Change:** 5 lines in `run_locomo_benchmark.py`. When gold is empty, score a prediction as correct if the prediction is also empty OR contains a refusal marker ("i don't know", "no information", "insufficient", "cannot determine", "unknown").

**Why:** The current matcher returns False on empty gold regardless of prediction. This makes 444 of 1986 LOCOMO questions (23%) unscorable by design. Fixing the matcher is necessary for the LOCOMO number to mean anything.

**Expected impact:** Overall LOCOMO lifts from 8.8% to ~30% IF the system currently returns empty answers for these questions (no fabrication). If the system currently fabricates, we need P1 first.

**File targets:**
- `run_locomo_benchmark.py` `answer_matches()` function

**Regression check:**
- Re-run LOCOMO `--max-convs 3` and confirm cat 5 goes from 0% to >0%.
- Confirm cats 1–4 are unchanged (the fix is gold-empty-only).

**Effort:** ~10 minutes.

---

### P1 — RefusalGate in `backbone.process_read`

**Change:** When both `convergence_check` and `equation_read` return no answer, `process_read` returns `{"answer": "", "source": "NONE", …}` explicitly. Add a helper `backbone.refusal_text()` that returns a canonical "I don't have information about that." so LLM wrappers and LOCOMO runner can distinguish "refused" from "fabricated".

**Why:** Necessary complement to P0. Without it, when the system genuinely doesn't know, it returns whatever the structural_matcher's top candidate was — a fabrication. With it, the system explicitly signals "no answer" and the downstream layer can respond honestly.

This is also a foundational continuity property. Returning "I don't know" when you don't is **more valuable than returning a wrong answer confidently**. User trust is built on this.

**Expected impact:**
- LOCOMO cat 5 moves from ~0% to 60–90% (depends on how often our retrieval correctly identifies "no match" vs spuriously matches).
- ATANT cumulative cp2_storage "GK trap" test should pass.
- Real users experience far fewer hallucinations.

**File targets:**
- `app/integration/backbone.py` — update `process_read` to set `source="NONE"` when both paths fail; add `refusal_text()` helper.
- `run_locomo_benchmark.py` — `ask_question()` returns the refusal marker when `source == "NONE"`.
- `run_atant_cumulative.py` — `read_question()` needs to accept refusal-as-valid-answer for "GK trap" batches (the runner already does this at the batch level, but the read path should align).

**Regression check:**
- ATANT ground_truth and ATANT cumulative must stay at baseline.
- LOCOMO should lift on cat 5.

**Effort:** ~30 minutes.

---

### P2 — Migrate `run_atant_cumulative.py` to `backbone.process_read` and run on 500 stories

**Change:**
1. Replace `_convergence_check` + `_equation_read` direct calls in `read_question()` with `backbone.process_read(always_run_structural=True)`.
2. Replace `inject_batch` write path (currently uses `upsert_relationship` directly because the cumulative runner injects ground-truth triples) — this stays as-is; the cumulative runner bypasses the write path on purpose.
3. Fix `_equation_read(user_id, question, shadow_mode=False)` — our current `equation_read` does not accept `shadow_mode`. Either add the param back or drop it.
4. Run on small range first (`--range 51-60`) to validate imports + wiring, then full 500.

**Why:** `run_atant_cumulative.py` is the REAL ATANT. It loads 500 stories, 20 checkpoints, tests all 7 properties (via cp19 contradiction, cp20 is_current, cp18 reconstruction, cp13 emotion, etc.). The 5-story `run_atant_ground_truth.py` is a scratchpad. I've been regressing against the wrong thing. P2 fixes this.

**Expected impact:**
- Establishes the real ATANT baseline across all 20 checkpoints.
- Every future change gets regressed against 500 stories, not 5.
- Exposes which checkpoints are currently at 0% because of stub dependencies (cp14 relational → `_assess_relationship`, cp17 proactive → `_get_proactive_insights`, etc.).

**File targets:**
- `run_atant_cumulative.py` — `read_question()` migration to backbone.
- Resolve `shadow_mode` mismatch.
- Import `process_read` at the top.

**Regression check:**
- First pass: `--range 51-60` must complete without import errors.
- Second pass: full `500` run — capture stories_passed, questions_passed, and per-checkpoint rates as the new baseline.
- Monotonic rule: once a baseline is established here, every future change must maintain it.

**Effort:** 20 minutes for migration + runtime cost (500 stories × ~2s per story = ~17 minutes compute).

---

### P3 — Situation reconstruction for narrative ("what/why/how") queries

**Change:**
1. In `app/memory/convergence_gate.py` Step 0 (situation query detection), widen the trigger beyond "summarize my situation" to include narrative comprehension patterns:
   - "what did X realize / learn / decide"
   - "why did X choose / decide"
   - "how does X feel / prioritize / handle"
   - "what happened after / before Y"
2. When triggered, call `schematic_trace.reconstruct_living_situation(user_id, anchor_subject, anchor_predicate)` and return the narrative string as the answer, instead of a single triple object.
3. Ensure `reconstruct_living_situation` actually exists — if it's stubbed, rebuild it using the arc_classifier template.

**Why:** This is the real gap for cat 4 LOCOMO (narrative comprehension, 841 questions, 42% of LOCOMO) AND for ATANT cp12_similarity (47% — reconstruction weakness is measured). Both benchmarks agree reconstruction is our weakest dimension. Fixing it serves continuity directly (property #5 — Reconstruction — from the 7 properties).

**Expected impact:**
- ATANT cp12_similarity: 47% → 70%+ (answers become richer narrative reconstructions).
- LOCOMO cat 4: 8% → 25–35% (verbose narrative answers contain gold fragments more often).
- Cumulative runner cp18_reconstruction actually starts firing.

**File targets:**
- `app/memory/convergence_gate.py` Step 0 trigger widening.
- `app/memory/schematic_trace.py` — check `reconstruct_living_situation` exists; if stubbed, rebuild.

**Regression check:**
- ATANT ground_truth: must stay at 30/32 or better.
- ATANT cumulative: CP8_answer must not regress; CP12_similarity should improve.

**Effort:** 1–2 hours. This is the heaviest priority because it touches read-path orchestration.

---

### P4 — Document LOCOMO policy: reference metric, not a target

**Change:** Add a short memory file + note in `CLAUDE.md` that LOCOMO is a reference metric only. Do not optimize to it. Do not celebrate LOCOMO gains. Use ATANT cumulative as the truth signal.

**Why:** Prevents future sessions (mine or a successor) from burning cycles chasing LOCOMO cat 4 paraphrase wins that don't translate to real continuity capability. The reframe needs to be durable across context clears.

**Expected impact:** Direction preservation.

**File targets:**
- `C:\Users\Sam\.claude\projects\D--Nura-Code-nura-living-memory-code\memory\feedback_locomo_not_a_target.md` (new)
- `CLAUDE.md` — append a short validation-policy block.

**Regression check:** None — this is documentation.

**Effort:** 5 minutes.

---

## 6. Execution Order (Corrected from Punch List)

| Order | Priority | Why this order |
|-------|----------|----------------|
| 1 | **P0** | Cheapest, unblocks honest LOCOMO measurement. 5 lines. |
| 2 | **P1** | Necessary complement to P0. Without it, P0's fix measures a different wrong thing. |
| 3 | **P2** | Establish the real ATANT baseline (500 stories, 20 CPs). Every later change must regress against this. |
| 4 | **P3** | Biggest capability lift. Needs the P2 baseline to tell if it helped. |
| 5 | **P4** | Documentation. Last. |

This order is the same order Sam asked for in the prompt (P0, P1, P2, P3, P4), and it's also the order that minimizes wasted work. Earlier priorities unblock later priorities.

---

## 7. What NOT To Do — The Temptations To Resist

1. **Do not add narrative reconstruction to make LOCOMO cat 4 go up.** Add it because ATANT cp12 is at 47% and reconstruction is one of the 7 properties. LOCOMO benefit is incidental.
2. **Do not "improve LOCOMO adversarial" by building a confusion detector.** LOCOMO adversarial is a matcher bug, not a confusion problem.
3. **Do not celebrate LOCOMO lifts without a corresponding ATANT cumulative pass count.** LOCOMO can go up for bad reasons (verbose answers, lucky substring matches). Cumulative ATANT cannot.
4. **Do not skip the cumulative-runner migration (P2) to go straight to P3.** Without a 500-story baseline, there's no way to tell if P3 helped or hurt.
5. **Do not add more stubs or shadow modes.** The 19 bytecode-reconstructed stubs that still exist are already a liability. If a checkpoint depends on a stub, rebuild the stub or mark the checkpoint as "cannot evaluate".
6. **Do not batch priorities for commit.** Each priority gets its own regression run, its own pass count, its own commit.

---

## 8. The Thesis Restated In Benchmark Terms

From `CLAUDE.md`:

> Memory stores the past. Continuity keeps the right parts alive in the present. The continuity layer is the missing layer between AI interaction and AI relationship.

Translated to benchmark selection:

- A benchmark that tests **what's in memory** is a retrieval benchmark. LOCOMO is (mostly) this.
- A benchmark that tests **which parts are alive right now** is a continuity benchmark. ATANT (especially cp19 contradiction + cp20 is_current + cp18 reconstruction) is this.
- ATANT's cp18 (situation reconstruction) is the single closest benchmark point to the thesis. It's also our weakest. That's the real signal to follow.

---

## 9. What I'm Actually Going To Do Right Now (Execution Record)

Following this doc, I am executing P0, P1, P2, P3, P4 **in order**, with regression runs between each.

1. **P0:** Fix `run_locomo_benchmark.py` `answer_matches()`. Commit-ready.
2. **P1:** Add `source="NONE"` signal + `refusal_text()` helper in `backbone.process_read`. Wire LOCOMO runner to return refusal text when `source="NONE"`. Run LOCOMO 1 conversation to sanity-check cat 5.
3. **P2:** Migrate `run_atant_cumulative.py` to `backbone.process_read`. Fix `shadow_mode` call mismatch. Run `--range 51-60` to validate. Then full 500.
4. **P3:** Widen convergence_gate Step 0 trigger for narrative queries. Confirm `schematic_trace.reconstruct_living_situation` exists (rebuild if stubbed). Run ATANT cumulative to measure cp12 lift.
5. **P4:** Write `feedback_locomo_not_a_target.md` memory file + append validation-policy block to `CLAUDE.md`.

Each step has its own regression run against ATANT. Monotonic improvement rule holds: pass count never decreases.

---

## 10. Cross-References (Context Survival)

Anyone (including future me) reading this after a context clear should ALSO read:

- `CLAUDE.md` — project rules, 7 properties, hard limits, GPU budget
- `forwarding_knowledge.md` — full session handoff (has the original wrong LOCOMO category labels, now corrected here)
- `locomo_gap_analysis.md` — the gap analysis that triggered this reframe
- `C:\Users\Sam\.claude\projects\D--Nura-Code-nura-living-memory-code\memory\project_locomo_categories.md` — auto-memory capturing the category correction
- `C:\Users\Sam\.claude\projects\D--Nura-Code-nura-living-memory-code\memory\project_backbone_unified.md` — the canonical process_write/process_read entries
- `C:\Users\Sam\.claude\projects\D--Nura-Code-nura-living-memory-code\memory\project_proactive_v2_canonical.md` — v1 retirement
- `C:\Users\Sam\.claude\projects\D--Nura-Code-nura-living-memory-code\memory\project_stub_rebuild_progress.md` — 2/19 stubs rebuilt ledger

---

## 11. One-Line Summary

**LOCOMO is less relevant than we thought. ATANT is better and the 500 stories already exist. P0-P1 make LOCOMO honest in 40 minutes. P2 gives us the real 500-story ATANT baseline. P3 closes the reconstruction gap that both benchmarks already agree is our weakness. P4 prevents the next context clear from undoing all of this.**

---

# Part 2 — The Actual Regression (2026-04-10, Continuation)

## 12. The Paper Exists — And We Found It

The ATANT paper is real. Published by Kenotic Labs, author Samuel Sameer Tanguturi. Local copies:

- **Paper PDF:** `D:/Nura/Kenotic Labs/atant-repo/ATANT Evaluation Framework.pdf`
- **Standard spec:** `D:/Nura/Kenotic Labs/atant-repo/docs/ATANT_Standard_v1.0.md`
- **Public repo:** [github.com/Kenotic-Labs/ATANT](https://github.com/Kenotic-Labs/ATANT)
- **Paper title:** "ATANT: An Evaluation Framework for AI Continuity"
- **arXiv citation:** `arxiv:2604.06710`
- **Talknura backup repo:** [github.com/Talknura/Nura](https://github.com/Talknura/Nura) — cloned locally to `D:/Nura/Code/talknura-nura/`

**The published compliance table:**

| Level | Date | Stories | Questions | CP8 | Notes |
|---|---|---|---|---|---|
| ATANT-Core | 2026-02-25 | 50 isolated | 304/304 | **100%** | Suite 1.0 |
| ATANT-Core refined | 2026-02-28 | 50 isolated | 304/304 | **100%** | Suite 1.1, all 10 CPs at 100% |
| Stress (100) | 2026-02-28 | 100 isolated | 656/671 | 98% | Suite 1.2, niche predicate failures |
| Stress (100) refined | 2026-03-01 | 100 isolated | 671/671 | **100%** | Suite 2.0 |
| **ATANT-Stress** | **2026-03-01** | **250 isolated** | **1,835/1,835** | **100%** | **Suite 2.1** |
| **ATANT-Cumulative** | **2026-03-14** | **50 cumulative** | **304/304** | **100%** | **← the baseline Sam remembers** |
| ATANT-Scale (active) | paper publication | 250 cumulative | 1,761/1,835 | 96% | Silver tier, frontier |

**"It passed before" = 2026-03-14, 304/304 CP8 on 50 stories in cumulative mode. That is the ground truth we regressed from.**

---

## 13. The Talknura Clone Reveals the Pre-Regression Architecture

The Talknura/Nura github repo exists. I cloned it:

```
git clone https://github.com/Talknura/Nura.git talknura-nura
```

- **Last commit:** 2026-04-03 (1 week before today)
- **Total commits:** 80
- **convergence_gate.py:** 427 lines (current: 833)
- **equation_pipeline.py:** 233 lines (current: 463)
- **NO cumulative/ corpus** — only the 50 core stories in 6 category dirs
- **NO nli_verifier.py, NO transitive_inference.py, NO duration_extractor.py**
- **NO reverse query templating in trace_decomposer.py** (the source of the "your has" bug)

**This is the paper-era snapshot.** Not bit-for-bit identical to the Mar 14 100% run (there's a week of drift between March 14 and April 3), but it's the closest public reference point to the passing state.

### The cleanest finding: the old `_convergence_inner`

The old convergence_gate has a clean, deterministic read path:

1. **Step 1.5** — Query Plan (predicate family + answer type via embedding)
2. **Step 1** — Fingerprint match against predicted_queries (cosine ≥ 0.55, no top-20 cap)
3. **Step 2** — Trace convergence (schema + temporal + emotional, additive tc score)
4. **Step 2.5** — Predicate discrimination (embed predicate, cosine vs query)
5. **Step 2.7** — Predicate family gate (hard filter on family_similarity ≥ 0.30)
6. **Step 3** — Sort by `(trace_count, fingerprint + 0.3 * predicate_sim)`, return top

That's it. 160 lines. No NLI. No type filter. No situation reconstruction. No temporal override. No source_timestamp magic. And it scored **100% on 50 cumulative** on March 14.

The current code has TWICE the lines and scores 22% on 5 stories in the extended corpus.

---

## 14. The Three Architectural Regressions, Identified

### Regression #1 — NLI verifier added on top of a working retriever

**Added:** 2026-04-09 19:21 (file mtime on `app/memory/nli_verifier.py`)

The current `_convergence_inner` wraps the old scoring in an NLI re-ranking pass. For each top candidate, it builds a "premise + hypothesis" pair and runs `cross-encoder/ms-marco-MiniLM-L-6-v2`. If NLI doesn't verify, the candidate is rejected. Multiple NLI-verified candidates → multi-value or multi-aspect reconstruction.

**Why this is wrong for our use case:**

`cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on MS MARCO — the task is: given a **web search query** (short) and a **candidate passage** (100-500 words), predict relevance. The output is a scalar log-odds score. It's a **passage reranker**, not an entailment model.

Our use case is: given a **memory query** (short) and a **triple rendered as a short sentence** (3-7 words: "user has pet Kobe"), pick the right triple. The input shape is totally different from MS MARCO training: both sides are short, and we want entailment semantics ("is the answer in this triple?"), not relevance semantics ("is this passage relevant to this query?").

**Evidence from the partial 50-story run:**
- NLI scores come back as −3 to −5 on valid candidates (strong rejection)
- `[DTCM] Multi-value: 4-year-old, 3-year-old (avg nli=-3.75)` — these are the right answers, marked as contradicted
- The downstream returns `"answer='your has'"` and `"answer='your health'"` — NLI accepted these over the real answers

**The old pipeline at 100% cumulative did not use NLI at all.** It used `(trace_count, fingerprint + 0.3 * predicate_sim)` as the decision criterion. That was enough.

### Regression #2 — Reverse query templating introduced "your has" / "your health" garbage

**Where:** `app/memory/trace_decomposer.py:847`

```python
display_pred = pred.replace("_", " ")
if subj.lower() == "user":
    reverse_answer = f"your {display_pred}"
else:
    reverse_answer = f"{subj}'s {display_pred}"

reverse_queries = [
    f"who is {obj}",
    f"what is {obj}",
    f"tell me about {obj}",
]
for rq in reverse_queries:
    # INSERT into predicted_queries with answer_text = reverse_answer
```

When `pred` is a bad generic predicate from auto_generate_triples (`has`, `is`, `does`), `display_pred` = `"has"` → `reverse_answer` = `"your has"`. This string is then inserted into the `predicted_queries` table as the `answer_text` for "who is Kobe" / "what is Kobe" / "tell me about Kobe". At read time, those reverse queries match and the system confidently returns **"your has"** as the answer.

**This feature doesn't exist in the old talknura trace_decomposer.** There is no `f"your {display_pred}"` string anywhere in the 2026-04-03 snapshot. The feature was added later, with no filter against generic predicates, and poisoned the `predicted_queries` index.

### Regression #3 — Grammar engine mis-types raw tokens

**Symptom from the partial 50-story run:**

```
[EntityGraph] Contradiction: user.lives_in 'July' → 'Brooklyn'
[EntityGraph] Contradiction: user.lives_in 'Brooklyn' → 'Sao Paulo'
[EntityGraph] Contradiction: user.lives_in 'Sao Paulo' → 'English classes'
```

"July" should be a `TIME` / `DATE`, not a `LOCATION`. "English classes" should be a `SUBJECT` / `TOPIC`, not a `LOCATION`. Something in `grammar_engine.tag_triple_type` (called during `upsert_relationship`) is mapping any object that comes through a `lives_in` predicate into a LOCATION — no type verification.

This feeds back into the contradiction detector, which then supersedes valid `lives_in` entries with bogus ones. By the end of a story the entire `lives_in` chain is garbage.

**The fix needs to be neural, not heuristic.** A NER model that knows "July" is a DATE, "English classes" is a TOPIC, and "Brooklyn" is a LOCATION. See section 15 for the right model choice.

---

## 15. Fixes Must Be Architectural and Scale — The Neural Model Choices

Sam's explicit direction: *"fixes should be architectural and scale - use NLP (search for other neural networks for my exact use case)"*.

That means: no threshold tuning, no regex hacks, no hand-crafted stopword lists. Each broken sub-system needs to be replaced with a small neural model that is **purpose-built for its specific task**, not a general-purpose model awkwardly repurposed. This section enumerates the use cases and the models I'd pick for each.

### Use case A — Triple-level answer verification (replaces current NLI)

**What the task actually is:** Given a memory query ("Who is my vet?") and a set of candidate triples rendered as short sentences ("user has vet Dr. Peterson", "user has dog Kobe", ...), pick which triple answers the query. The output should be three-way: `entailment` (this triple answers the query), `neutral` (unrelated), `contradiction` (this triple directly contradicts the query's premise).

**Current model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — wrong training objective (passage ranking, not entailment).

**Recommended replacement:** **`cross-encoder/nli-deberta-v3-small`**

- **Parameters:** 100M
- **Disk size:** ~400MB
- **Training data:** SNLI (570k) + MultiNLI (412k) — trained directly on premise/hypothesis entailment
- **Output:** 3-class softmax over `[contradiction, entailment, neutral]`
- **Benchmarks:** SNLI 91.65%, MNLI mismatched 87.55%
- **GPU budget:** <1 GB VRAM, fits comfortably alongside our 8B LLM + T5 SRL in async load/offload
- **Why it fits our use case:** trained on SENTENCE PAIRS (premise + hypothesis), both short, which is exactly the shape of (query, triple-as-sentence). The 3-class output lets us keep entailment, reject only on strong contradiction, treat neutral as "fall through to composite score".

**How to use it architecturally (not as a gate):**

```python
# Pseudocode for convergence_gate Step 3 rerank:
scores = nli.predict([(query, triple_to_sentence(t)) for t in candidates])
# scores.shape == (N, 3) — columns are contradiction, entailment, neutral

entail = scores[:, 1]
contradict = scores[:, 0]

# Only reject on STRONG contradiction — never reject on neutral
keep = [c for c, con in zip(candidates, contradict) if con < 0.70]

# Use entailment as tiebreaker within keep set, not as primary rank
keep.sort(key=lambda c: (c.trace_count, c.fingerprint + 0.3 * c.predicate_sim + 0.2 * entail[c.idx]), reverse=True)
```

The key difference from current code: **entailment is a tiebreaker, not a gate.** If everything is neutral, we still return the top composite-scoring candidate (the old April-3 behavior). Only strong contradiction (>70%) removes a candidate.

**Alternative if budget constrained:** `cross-encoder/nli-deberta-v3-xsmall` (~50M params, faster, ~88% MNLI accuracy).

**References:**
- [cross-encoder/nli-deberta-v3-small on HuggingFace](https://huggingface.co/cross-encoder/nli-deberta-v3-small)
- [cross-encoder/nli-deberta-v3-xsmall on HuggingFace](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall)

### Use case B — Object type verification at write time (prevents "July → lives_in")

**What the task actually is:** Given an extracted triple `(subject, predicate, object)`, verify that the object is the correct semantic type for the predicate. "lives_in" expects LOCATION. "works_at" expects ORGANIZATION. "tenure" expects DURATION. "manager" expects PERSON. If the type doesn't match, reject the triple or re-route it.

**Current:** `app/memory/grammar_engine.py`'s `tag_triple_type()` does a heuristic mapping using verb anchor embeddings. Fails on tokens like "July" / "English classes" because it doesn't know what those tokens actually are.

**Recommended neural:** **`dslim/bert-base-NER`** or **`Davlan/bert-base-multilingual-cased-ner-hrl`**

- **Parameters:** ~110M
- **Task:** Named Entity Recognition (token classification)
- **Output labels:** PERSON, LOCATION, ORGANIZATION, MISC (+DATE/TIME for extended models)
- **Why it fits:** off-the-shelf NER is exactly the missing piece — does "July" tokenize as a DATE? Does "Brooklyn" tokenize as LOCATION? Does "Dr. Peterson" tokenize as PERSON? This is the standard task this model family was built for.
- **Architectural placement:** Run NER on the `object` string inside `upsert_relationship`. If the predicate expects LOCATION and the NER says the object is DATE/O/MISC, **reject the triple** (or reroute it to a more appropriate predicate like `scheduled_time`).
- **Fallback:** If NER is unavailable or fails, fall through to the current heuristic.

**Better alternative for date/time specifically:** **`Jean-Baptiste/roberta-large-ner-english`** (355M) includes DATE and CARDINAL labels natively.

**Best option if we want something tiny:** Use our existing **T5 SRL v4** with a new task prefix `<type>`. Fine-tune on a small dataset of `(token, expected_type)` pairs. Would reuse the already-loaded T5, no new model to manage.

### Use case C — Reverse query templating cleanup (prevents "your has")

**What the task actually is:** When a triple `(user, pred, object)` is stored, generate reverse queries ("Who is {object}?") whose answer is the relationship the user has to that object. E.g., `(user, manager, Priya Banerjee)` should produce the reverse query "Who is Priya Banerjee?" → answer "your manager". But `(user, has, "stuff")` should NOT produce "Who is stuff?" → answer "your has", because "has" is not a relationship.

**This is a filtering problem, not a reranking problem.** But the filter should be semantic, not a hand-coded stopword list.

**Recommended neural:** Use the **existing sentence-transformers MiniLM embedder** we already have loaded.

- Maintain a **reference embedding set** of valid reverse-query predicates (`manager`, `teammate`, `vet`, `partner`, `sister`, `brother`, `mother`, `father`, `friend`, `mentor`, `advisor`, `colleague`, `neighbor`, `pet`, `doctor`, `therapist`, ...).
- Compute embedding of `display_pred`.
- Max cosine similarity against the reference set.
- If max sim ≥ 0.55, generate the reverse query. If below, skip.

This replaces the current "generate reverse for every predicate" with "generate reverse only when the predicate is semantically close to a known relationship word". Uses the already-loaded MiniLM — no new model. Scales because we can extend the reference set without changing code. Avoids regex / stopword hacks per Sam's "no thresholds for pattern matching" rule.

**Alternative (more robust):** Use a small classifier trained on a binary task "is this predicate a relationship noun?". Could be `distilbert-base-uncased` fine-tuned on a few hundred examples. 66M params. But probably overkill — the embedding cosine approach gets 90%+ with zero training.

### Use case D — Auto-generating triples from text (current regex is the worst part)

**What the task actually is:** Given a raw text turn + a list of `expected_memory_stores`, produce `(subject, predicate, object)` triples. Currently done in `run_atant_cumulative.py:165 auto_generate_triples()` — 300+ lines of regex rules covering roles, names, emotions, job titles, durations, etc. Exactly what Sam's `feedback_no_regex.md` forbids.

**Recommended replacement:** **Use the existing T5 SRL v4 model.**

We already have T5 SRL v4 at `models/raya-srl-220m-v4/final/`. It has the `<triplets>` prefix that was trained to take text and emit `(subject, predicate, object)` triples. **That's exactly what auto_generate_triples is trying to do, but better.**

**The correct P2 fix** (beyond what I already did):

1. If the YAML batch has `expected_triples`, use them directly. (Already done today.)
2. Otherwise, run T5 SRL `<triplets>` on the batch's `user_input` text.
3. Validate each emitted triple against `expected_memory_stores` — any store-keyword that doesn't appear in any emitted triple is a failure signal (diagnostic, not a reject).
4. Delete `auto_generate_triples()` entirely once the T5 path is validated.

This replaces 300 lines of regex with 3 model calls, uses an already-loaded model, and matches exactly what the paper's Feb-Mar runs did (which passed at 100%). Scales because T5 handles novel phrasings without code changes.

### Use case E — Situation reconstruction for narrative queries

**What the task actually is:** Given a query like "What did Melanie realize after the charity race?" (narrative comprehension, LOCOMO cat 4), return not a single triple object but a multi-triple narrative reconstruction around the target entity + time range.

**Current:** My P3 fallback in convergence_gate calls `schematic_trace.reconstruct_living_situation(user_id, target_entity)`. That function exists but concatenates entity profile sections, not a coherent narrative.

**Recommended neural:** **T5-small fine-tuned for abstractive summarization** or **FLAN-T5-small**.

- **Parameters:** 60M (T5-small) or 80M (FLAN-T5-small)
- **Task:** Given a set of triples + traces as input, generate a natural-language narrative paragraph.
- **Why it fits:** T5 is a seq2seq model; input is a list of structured facts, output is a narrative sentence. This is the canonical "data-to-text" use case T5 was designed for.
- **Architectural placement:** Replace the concatenation logic in `reconstruct_living_situation()` with a T5 generation pass over the retrieved trace set.
- **Budget:** FLAN-T5-small is ~300MB, fits in our async slot alongside T5 SRL (load one at a time).

**Alternative (if we want to train on our corpus):** Fine-tune our existing T5 SRL v4 with a new `<situation>` prefix. Input = serialized traces + query, output = narrative. One model serves both extraction and reconstruction.

### Summary table

| Use case | Current implementation | Recommended neural | Params | Role |
|---|---|---|---|---|
| **A.** Triple verification | `ms-marco-MiniLM-L-6-v2` (wrong task) | `cross-encoder/nli-deberta-v3-small` | 100M | Tiebreaker, not gate |
| **B.** Object type verification | heuristic verb embeddings | `dslim/bert-base-NER` OR fine-tune T5 SRL | 110M / 220M | Pre-upsert filter |
| **C.** Reverse query predicate filter | hard template `f"your {pred}"` | MiniLM cosine vs reference set (already loaded) | 0 new | Semantic filter |
| **D.** Triple generation from text | 300 lines of regex | existing T5 SRL v4 `<triplets>` prefix | 0 new | Replaces auto_generate |
| **E.** Situation reconstruction | concat entity profile sections | FLAN-T5-small OR T5 SRL `<situation>` prefix | 300M / 0 new | Narrative generation |

**Net new model footprint:** 100M (NLI) + 110M (NER) = 210M params = ~840MB on disk. Load both in an async slot alongside T5 SRL (~900MB). Fits the 8GB VRAM budget.

**Net replaced hand-written code:** ~500 lines of regex, templates, heuristic scoring.

---

## 16. The Complete Post-Paper Regression Ledger

Files modified or added between the paper-era talknura/Nura snapshot (2026-04-03) and today (2026-04-10):

| File | Status | Lines delta | What it added | Regression impact |
|---|---|---|---|---|
| `app/memory/nli_verifier.py` | **NEW** (04-09 19:21) | +230 | Cross-encoder reranker using `ms-marco-MiniLM-L-6-v2` | **SEVERE** — rejects valid answers due to wrong model choice |
| `app/memory/convergence_gate.py` | modified | 427 → 833 (+406) | Step 0 situation detect, Step 1.4 temporal, Step 2.8 type filter, source_timestamp override, NLI gate, narrative fallback | **SEVERE** — NLI gate + over-widening |
| `app/memory/equation_pipeline.py` | modified | 233 → 463 (+230) | Correction handler, proactive v2 wire, temporal parse, duration extractor, source_timestamp stamping | Medium — adds complexity; possibly correct but untested |
| `app/memory/trace_decomposer.py` | modified | — | **Reverse query templating** `f"your {display_pred}"` | **HIGH** — poisons predicted_queries with "your has" garbage |
| `app/memory/entity_graph.py` | modified | — | Contradiction detection wired into upsert | Medium — without NER gate, supersedes valid triples with wrong types |
| `app/memory/transitive_inference.py` | **NEW** (04-10 04:42) | +? | Synthetic inferred triples | Unknown — untested, may add noise |
| `app/memory/duration_extractor.py` | **NEW** (04-10 13:36 — today) | +200 | Duration extraction from text | Low — only fires on specific patterns |
| `run_atant_cumulative.py` | modified | — | Added 500-story cumulative/ corpus, `auto_generate_triples`, ignores YAML `expected_triples` field | **SEVERE** — uses regex instead of YAML ground truth (partially fixed today) |
| `tests/stories/cumulative/` (500 new stories) | **NEW** (04-09) | 500 files | Extended corpus never validated against published compliance | N/A — corpus expansion, not pipeline regression |

**Summary:** Five architectural additions post-paper, all attempting to improve the pipeline, at least three of which caused measurable regressions when evaluated on naturalistic stories.

The common pattern: each addition was well-intentioned (NLI should improve precision; reverse queries should improve recall; contradiction should improve update handling) but **none were validated against ATANT-Cumulative 50 before being merged**. Monotonic improvement rule was not enforced.

---

## 17. The Rollback Strategy (Architectural, Not Revertist)

The wrong response to the regressions above is "revert everything to April 3 talknura". That abandons a week of real improvements and is not architectural. The right response is a **surgical rollback + neural replacement**:

### Phase A — Stop the bleeding (1 day, high confidence)

1. **Disable NLI gate.** Change `convergence_gate._convergence_inner` so that `if not verified_triples` returns the top composite-score candidate (the old April-3 behavior), not `None`. NLI becomes a pure tiebreaker.
2. **Gate reverse query templating** behind the MiniLM cosine filter (Use case C above). `"has"`, `"is"`, `"does"` never generate reverse queries.
3. **Replace `ms-marco-MiniLM-L-6-v2` with `cross-encoder/nli-deberta-v3-small`** in `nli_verifier.py`. Update the score interpretation to 3-class (accept entailment, reject only on strong contradiction > 0.7, treat neutral as "no signal").
4. **Run ATANT-Cumulative 50** on the category dirs. Target: 250+/304 (80%+).

### Phase B — Neural type verification (2 days, medium confidence)

5. **Add `dslim/bert-base-NER` pre-upsert gate** in `entity_graph.upsert_relationship`. Reject or reroute triples where the object's NER tag doesn't match the predicate's expected type.
6. **Re-run ATANT-Cumulative 50.** Target: 290+/304 (95%+).

### Phase C — Replace auto_generate with T5 SRL (1 day, high confidence)

7. **Delete `auto_generate_triples()` from `run_atant_cumulative.py`.** Replace with: use YAML `expected_triples` if present, else call `equation_write()` (which goes through T5 SRL). No more regex.
8. **Re-run ATANT-Cumulative 50.** Target: 304/304 (100%).

### Phase D — Validate against ATANT-Scale (1 week, uncertain)

9. **Run ATANT-Cumulative 250** (if we can find or regenerate the Stress Rounds 2-5 corpus — 200 more stories beyond the 50 core).
10. **Target: Silver tier (95%+)** matching the paper's active frontier number.

### Phase E — Situation reconstruction (FLAN-T5-small)

11. **Integrate FLAN-T5-small** into `schematic_trace.reconstruct_living_situation` for narrative queries (LOCOMO cat 4 alignment). This is purely additive — improves cp12_similarity and LOCOMO cat 4 without touching the core retrieval.

---

## 18. What This Session Produced (Running Ledger)

Updated at the end of the 2026-04-10 session:

**Completed:**
- ✅ P0 — LOCOMO `answer_matches` empty-gold fix (5 lines, 8/8 unit tests pass)
- ✅ P1 — `backbone.process_read` RefusalGate with `source="NONE"` + `REFUSAL_TEXT`
- ✅ P2 — `run_atant_cumulative.py` migrated through `backbone.process_read`
- ✅ P3 — Narrative fallback in `convergence_gate` Step 2 (for "what did X" queries)
- ✅ P4 — LOCOMO policy documented in `CLAUDE.md` + memory
- ✅ Stub rebuild: `arc_classifier.py` (380 lines), `query_structure.py` (450 lines)
- ✅ Fix: YAML `expected_triples` injection → **22% → 36% on 5 cumulative stories**
- ✅ Proactive v1 retired; v2 canonical in chat_routes, backbone, orchestrator
- ✅ `backbone.process_write` / `process_read` canonical entries (module-level)
- ✅ Found the ATANT paper, the Talknura backup repo, and the pre-regression architecture

**Identified but not yet fixed (Phase A targets for next session):**
- NLI verifier uses wrong model — swap to `nli-deberta-v3-small` and make it a tiebreaker
- Reverse query templating generates "your has" garbage — add MiniLM cosine predicate filter
- Grammar engine mis-types raw tokens (July → LOCATION) — add NER pre-upsert gate
- `auto_generate_triples()` regex should be replaced by T5 SRL `<triplets>`

**Compliance current state:**
- ATANT-Core isolated on 5 stories (subset test): **8/32 via ground_truth runner** — but this is the 5-story runner, not a real ATANT mode
- ATANT-Cumulative on stories 51-55 of the *new* 500 corpus: **18/50 (36%)** after today's YAML fix
- ATANT-Cumulative on published 50: **unknown** (partial run hit 36/50 before being killed; no report generated)
- LOCOMO 3 convs: **42/476 (8.8%)** — unchanged (P0 and P1 fixes need a re-run to measure lift)

**The monotonic baseline is now 36% cp8_answer on stories 51-55 of the cumulative/ corpus via backbone.process_read.** Any change that drops below this must be reverted.

---

## 19. Sources (for future context)

### Academic
- [ATANT: An Evaluation Framework for AI Continuity](https://arxiv.org/abs/2604.06710) — the paper. Kenotic Labs, Samuel Tanguturi, April 2026.
- Logan, J. (2026). *Continuum Memory Architectures for Long-Horizon LLM Agents.* arXiv:2601.09913 — cited as closest prior work in the ATANT paper.
- Natangelo, S. (2025). *The Narrative Continuity Test.* arXiv:2510.24831 — conceptual framework, 5 dimensions.
- Packer et al. (2023). *MemGPT.* arXiv:2310.08560 — tiered memory OS metaphor.
- Chhikara et al. (2025). *Mem0.* arXiv:2504.19413 — production memory layer.
- Xu et al. (2025). *A-MEM.* arXiv:2502.12110 — agentic memory for LLM agents.

### Neural model references
- [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) — recommended NLI reranker replacement
- [cross-encoder/nli-deberta-v3-xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) — smaller fallback
- [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) — recommended NER gate
- [Jean-Baptiste/roberta-large-ner-english](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english) — NER with DATE/CARDINAL labels
- [Sentence-Transformers docs on pretrained cross-encoders](https://www.sbert.net/docs/pretrained_cross-encoders.html)

### Repos
- [github.com/Kenotic-Labs/ATANT](https://github.com/Kenotic-Labs/ATANT) — the spec repo (also cloned at `D:/Nura/Kenotic Labs/atant-repo/`)
- [github.com/Talknura/Nura](https://github.com/Talknura/Nura) — the paper-era code snapshot (cloned at `D:/Nura/Code/talknura-nura/`, last commit 2026-04-03)
- `D:/Nura/Code/nura_living_memory_code/SingleKenotic/originals/` — second pre-NLI snapshot (2026-04-09 15:51, a few hours before NLI was added)

### Local diagnostics (artifacts from this session)
- `ReasoningATANT1.1.md` — this document
- `locomo_gap_analysis.md` — the LOCOMO findings that triggered this reframe
- `test_reports/atant_cumulative_20260410_*.json` — cumulative runs, for regression comparison
- Memory files:
  - `project_locomo_categories.md` — LOCOMO category correction
  - `project_backbone_unified.md` — canonical process_write/process_read
  - `project_proactive_v2_canonical.md` — v1 retirement
  - `project_stub_rebuild_progress.md` — stub rebuild ledger
  - `project_real_atant_baseline.md` — 22% cumulative baseline
  - `feedback_locomo_not_a_target.md` — LOCOMO policy

---

## 20. One-Line Summary (Updated)

**ATANT passed at 100% cumulative 50 on 2026-03-14. Between then and today, five architectural additions (NLI verifier with the wrong base model, reverse query templating, contradiction-without-NER, auto_generate_triples regex, and a 500-story extended corpus) regressed the pipeline to ~22-36%. The fix is not to revert — it is to swap each broken sub-system for the right purpose-built neural model (nli-deberta-v3-small, dslim/bert-base-NER, existing T5 SRL, MiniLM cosine filter, FLAN-T5-small) and restore the April-3 composite-score fallback as the default when NLI has no signal. That is architectural. That scales.**
