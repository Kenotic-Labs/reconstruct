# KENOTIC LABS — AGENT REGULATIONS
**Last Updated:** 2026-03-31
**Authority:** Founder (Sam)
**Scope:** Every agent working on this project MUST read this file before touching any code.

---

## THE THESIS

Most AI today is intelligent per session. Continuity is what makes AI coherent across life.

**Session-based AI:**
You say something. It responds. The moment ends. Whatever survives is prompt context, chat history, or retrieved notes.

**Life-based AI:**
You say something. The system determines what matters. That thing remains alive beyond the session. If the situation changes, the system updates it. If it becomes relevant later, the system reconstructs it.

That is continuity.

**Memory stores the past. Continuity keeps the right parts alive in the present.**

The continuity layer is the missing layer between AI interaction and AI relationship.

---

## WHAT WE ARE

Kenotic Labs builds the **continuity layer** for AI systems.

We are an infrastructure company. Not a product company. Not building a companion. Not building a chatbot.

Human life is not made of isolated prompts. It is made of unfinished situations, changing states, recurring concerns, relationships, timing, logistics, moods, plans, identities, and commitments. Most AI systems are structurally bad at that. Good at answering in the moment. Weak at carrying a life forward.

Continuity is the missing layer between intelligence-in-the-moment and presence-over-time.

**What continuity is NOT:**
- "The AI remembers my name"
- "The AI remembers my favorite car"
- "The AI can search old messages"
- "The AI has long context"
- "The AI uses RAG"
- "The AI stores embeddings"

Those are components. None create continuity alone.

**What continuity IS:**
The logic that answers:
- What from this interaction should persist?
- In what form should it persist?
- What is still active versus resolved?
- What changed since last time?
- What matters now versus later?
- When should something come back?
- How should it come back?

That is why continuity is a layer, not a feature.

---

## THE 7 PROPERTIES OF CONTINUITY

### 1. Persistence Beyond Session
If the model shuts down — continuity survives.
If the app closes — continuity survives.
If the device restarts — continuity survives.
If the user comes back tomorrow — continuity exists.

Without this, you only have temporary context.

### 2. Update Handling
Real life changes. Appointments move. Moods shift. Relationships evolve. Plans change.

Continuity preserves both historical state and current state without mixing them. "I was nervous before, I feel better now" becomes ordered reality, not noise.

### 3. Temporal Ordering
Not just what happened, but when, in what sequence, with what current status.

"Tomorrow" resolves to an actual date. "Moved to Wednesday" overwrites timing while preserving the fact of change. "Last time this failed" remains distinct from "this is what we do now."

### 4. Disambiguation
Two people can have similar events. Two companies can be involved in similar processes. Two feelings can exist around two different situations.

Continuity keeps subjects, entities, and emotional attachments correctly separated.

### 5. Reconstruction
A continuity system answers not only direct factual questions, but situation-level questions.

Not just: "When is my interview?"
But: "Summarize my current situation." "Why am I anxious?" "What am I preparing for?" "What changed since last time?"

That requires combining traces into a coherent state.

### 6. Model Independence
If continuity is real, it does not live inside one model session. One model can write the situation. Another model can read and reconstruct it later.

The continuity layer is below the intelligence layer, not trapped inside it.

### 7. Operational Usefulness
Continuity matters beyond personal chat. In a clinic, library, robot, service desk, or workflow — continuity means the system does not restart from zero. It carries forward repeated context, user needs, prior failures, preferences, and unresolved tasks.

---

## RETRIEVAL VS RECONSTRUCTION

**Retrieval says:**
Here are some related past things.

**Continuity says:**
Here is the current living state of the situation, including what changed, what still matters, and what should happen next.

The difference between searching and understanding.

---

## USER FIRST — ALWAYS

The user's data never leaves their device. Privacy is physics, not policy.

The user's time is sacred. Latency under 1 second or it's broken.

The user's trust is earned. No hallucination. No fabrication. If we don't know, we say so.

The user's context is continuous. What they said yesterday matters today. What they'll need tomorrow is anticipated now.

---

## HARD LIMITS

| Rule | Why |
|------|-----|
| **MAX 5 sub-agents** | Too many parallel sub-agents caused stack overflow |
| **MAX 3 test processes** | Shared SQLite corrupts with parallel runs |
| **50 tool calls → handoff.md → STOP** | Node crashes before token limits |
| **NEVER ask permission** | User is often asleep. Execute. |
| **NEVER read entire large files** | Use line ranges |
| **MONOTONIC IMPROVEMENT** | Pass count never decreases |

---

## GPU CONSTRAINTS — 8GB HARD LIMIT
```
ALWAYS RESIDENT:
└── 8B LLM (4-bit): ~4.5GB

ASYNC LOAD/OFFLOAD (one at a time):
├── STT: ~500MB
├── TTS: ~300MB
└── T5 SRL (220M): ~900MB
```

Load → Use → Offload → Load next. Never stack.

---

## ARCHITECTURE — DTCM

**Decomposed Trace Convergence Memory**

**Write path:** Don't dump raw text. Interpret structurally.

"I'm nervous because I have a Google interview next Tuesday at 3 PM and I need to leave by 1:30 because the drive is long"

Contains: identity, event, time, emotional state, entity, intent, preparation target, logistics. Write it so the system can live with it later.

**Read path:** Don't retrieve similar chunks. Reconstruct the current situation.

If the interview moved, if the emotional state changed, if the logistics still matter — give the current picture, not a bag of old snippets.

**5 traces at write time:**
1. **Episodic** — What happened
2. **Emotional** — How it felt
3. **Temporal** — When it occurred
4. **Relational** — Who was involved
5. **Schematic** — What pattern it fits

**Deterministic reconstruction at read time.** No probabilistic retrieval. No hallucinated connections.

**The equation:**
```
Score = E × P × T × F × I × C × R

E = Embedding similarity
P = Predicate alignment
T = Temporal relevance
F = Frequency weight
I = Importance score
C = Confidence
R = Relational proximity
```

---

## CORE BRAIN — 7 FILES

| File | Purpose |
|------|---------|
| `backbone.py` | Orchestrates all engines |
| `memory_engine.py` | Write path |
| `trace_decomposer.py` | 5 traces + predicted queries |
| `equation_pipeline.py` | 7-component scoring |
| `convergence_gate.py` | DTCM convergence |
| `structural_matcher.py` | Read path |
| `entity_graph.py` | Relationship graph |

**Modify these → Run full ATANT before and after.**

---

## T5 SRL — INPUT LAYER

220M parameters. Replaces all regex extraction.
```
<cleanup>   messy STT → clean text
<triplets>  text → (subject, predicate, object)
<roles>     text → AGENT | ENTITY | ROLE | LOCATION | TEMPORAL
<dimensions> text → valence | affiliation | equality | engagement
```

**Replaces 6 files + 244KB data with 3 model calls.**

---

## PIPELINE
```
MICROPHONE → SILERO VAD → WHISPER STT → T5 SRL
                                           ↓
                                    ┌──────────────┐
                                    │  CORE BRAIN  │
                                    │  (7 files)   │
                                    └──────┬───────┘
                                           ↓
                              8B LLM → KOKORO TTS → SPEAKER
```

Latency budget: <1000ms. Currently ~800ms.

---

## VALIDATION — ATANT

**Automated Test for Acceptance of Narrative Truth**

**Canonical runner:** `run_atant_cumulative.py` — 500 stories in `tests/stories/cumulative/`, 20 checkpoints.

`run_atant_ground_truth.py` is a 5-story smoke test, not the truth signal. Do not cite it.

| CP | Tests |
|----|-------|
| CP1-CP4 | Write path (extraction, storage, predicted queries, type tagging) |
| CP5-CP8 | Read path (query classification, structural match, DTCM convergence, final answer) |
| CP9-CP10 | Traces + emotional valence |
| CP11-CP12 | Grammar reconstruction + answer similarity |
| CP13-CP17 | Emotion detection, relational, schematic, frequency, proactive |
| CP18-CP20 | Situation reconstruction, contradiction detection, is_current filtering |

**Truth signal:** `cp8_answer` on the full 500 stories via `run_atant_cumulative.py`.
**Current baseline (2026-04-10):** 22% cp8_answer on stories 51-55. This is the honest starting line. The old "94%" number was the 5-story toy runner.
**Target:** 100% all checkpoints. Monotonic rule applies to `cp8_answer` on the 500-story run.

### LOCOMO policy — reference metric only, NOT a target

LOCOMO is **not** our truth signal. Use it only for external credibility with investors / reviewers. Never optimize to it. Reasons:

1. **23% of LOCOMO (cat 5, 446 questions) is unscorable by design** — `answer_matches` returns False on empty gold. Fixed in P0.
2. **42% of LOCOMO (cat 4, 841 questions) is substring-matched narrative paraphrase.** Measures verbosity + phrasing luck, not continuity.
3. **LOCOMO category labels in the forwarding doc were backwards.** cat2 is temporal, cat4 is narrative comprehension. Don't trust the labels.
4. **LOCOMO tests ~2 of our 7 properties.** Zero tests for persistence, update handling, disambiguation, reconstruction, supersession, model independence, operational usefulness.
5. **ATANT is the thesis.** We wrote it for our properties. LOCOMO was written by someone else for someone else.

Build the *capabilities* LOCOMO targets (refusal, narrative reconstruction, temporal extraction). Measure them in ATANT. Every "LOCOMO improvement" proposal must answer: **"which continuity property does this serve?"** If "none", reject.

Full reasoning: `ReasoningATANT1.1.md` in the repo root.

---

## 6 FOUNDATIONAL PRINCIPLES

### 1. Single Source of Truth
Define once. Reference everywhere.

### 2. Scale-Invariant Design
Ratios, not absolutes. `LIMIT 500` breaks. `LIMIT 0.1 * total` scales.

### 3. Granularity Match
Compute and consume at same level.

### 4. Semantic Purity
Embeddings only. No keyword hacks.

### 5. Reachable Code
Every branch must execute. Dead paths are bugs.

### 6. Edge Inclusion
n=1 is valid data.

**Debug by principle, not symptom.**

---

## DEPLOYMENT CONTEXTS

Same infrastructure. Prioritized by readiness and regulatory complexity.

### 1. RAYA (Consumer) — NOW
Status: Almost done. Core brain intact. T5 SRL ready to integrate.
File: `raya_consumer/raya.py`
What: Individual continuity. MCP integration. Message import (iMessage, WhatsApp, etc.)
Why first: Proves the thesis. Demo-ready. Investor-facing.

### 2. SDK — NEXT
Status: Extract from Raya once stable.
Package: `kenotic-sdk`
What: Continuity layer as importable library. Write path + Read path + DTCM.
Why second: Easily releasable. Testable by developers. Revenue from licensing.
Users: Developers building AI apps that need continuity.

### 3. AI AGENTS — SAME AS SDK
Status: SDK with agent-specific wrapper.
What: Continuity for autonomous agents. State persistence across tasks.
Why third: Same architecture as SDK. Different go-to-market (agent builders vs app developers).
Users: Companies deploying agentic AI that keeps failing (40%+ failure rate).

### 4. LIBRARIES — STABLE
Status: Architecture ready. Needs book data ingestion.
File: `raya_library/raya_library.py`
What: Multi-patron continuity. Voice profiles. SIP2/Koha integration.
Why fourth: Calm deployment. No regulatory complexity. Just data ingestion of available books.
Users: Public libraries, university libraries.

### 5. CLINICS — LAST
Status: Architecture ready. Regulations need research.
File: `raya_clinic/raya_clinic.py`
What: HIPAA-compliant continuity. Encrypted storage. Audit trail.
Why last: Regulatory complexity. Need to understand compliance before deployment.
Users: Small clinics, patient tracking, care continuity.

---

## EXECUTION SEQUENCE
```
WEEK 1-2:  Raya consumer → Demo-ready
WEEK 3:    Extract SDK from Raya
WEEK 4:    SDK packaging + documentation
WEEK 5:    Agent wrapper on SDK
WEEK 6-7:  Library deployment (book ingestion)
WEEK 8+:   Clinic research + deployment
```

One file each. If it doesn't fit in one file, the architecture is wrong.

---

## DEBUGGING

**Before fix:** Run ATANT baseline. Record pass count. Find pattern. Map to principle.

**After fix:** Run regression. If pass count dropped → REVERT.

**Never:** Fix tests one at a time. Add special cases. Run parallel tests.

---

## TOKEN BUDGETS
```
System Prompt:        100
User Facts:           200
Memory Context:       500
Conversation History: 1000
User Input:           500
Response Space:       ~1400
```

---

## REMEMBER

Continuity is the layer. Raya is the proof.

The user comes first. Always.

When in doubt, run ATANT.

---

## THE STRONGEST VERSION

The continuity layer is the missing layer between AI interaction and AI relationship.

That is what we are building.
