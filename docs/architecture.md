# Kenotic Labs -- Architecture Overview

## Decomposed Trace Convergence Memory (DTCM)

**The continuity layer for AI systems.**

---

## The Problem

Every AI system today is intelligent per session and amnesic across sessions. Ask it something brilliant at 2 AM -- by morning, the context is gone. The industry's answer is RAG: retrieve chunks of old text and paste them into the prompt. But retrieval is not understanding. RAG returns fragments. It does not know what changed, what still matters, what resolved, or what should come back later.

Human relationships work because people carry forward a living model of each other -- not a search index of past conversations. AI has no equivalent layer. The result: every session starts from zero, and "memory" means keyword search over old messages.

Continuity is the missing layer between AI interaction and AI relationship.

---

## The Solution: DTCM

Decomposed Trace Convergence Memory is a deterministic reconstruction engine grounded in cognitive science. Every input is decomposed into five trace dimensions -- episodic, emotional, temporal, relational, and schematic -- mirroring how the human hippocampus encodes experience. At query time, the system does not search for similar text. It sends a multi-dimensional probe to all stored traces simultaneously, and the trace that resonates across the most dimensions surfaces as the answer. This is pattern completion, not retrieval. The architecture is based on Hintzman's MINERVA 2 model (1984) and Multiple Trace Theory (Nadel and Moscovitch, 1997), adapted for structured storage and deterministic verification.

---

## The 7 Properties of Continuity

1. **Persistence Beyond Session** -- Memory survives app closure, device restart, and model shutdown. If the user returns tomorrow, continuity exists.

2. **Update Handling** -- Real life changes. Appointments move. Moods shift. The system preserves both historical and current state without mixing them.

3. **Temporal Ordering** -- Not just what happened, but when, in what sequence, and with what current status. "Tomorrow" resolves to a real date. "Moved to Wednesday" updates timing while preserving the fact of change.

4. **Disambiguation** -- Two people can have similar events. Two companies can be involved in similar processes. The system keeps subjects, entities, and contexts correctly separated.

5. **Reconstruction** -- The system answers situation-level questions, not just factual lookups. "Summarize my current situation." "What changed since last time?" "Why am I anxious?"

6. **Model Independence** -- Continuity lives below the intelligence layer. One model can write. Another model can read and reconstruct later. The layer is not trapped inside any single model session.

7. **Operational Usefulness** -- Continuity applies beyond personal chat. Clinics, libraries, service desks, autonomous agents -- any system that should not restart from zero.

---

## How It Works

### Write Path: 5 Traces

Every input is decomposed into five simultaneous dimensions of meaning.

```
Input: "I'm nervous because I have a Google interview next Tuesday at 3 PM"

Episodic trace    -- what happened: interview scheduled at Google
Emotional trace   -- how it felt: nervous, anxious
Temporal trace    -- when: next Tuesday, 3 PM, future tense
Relational trace  -- who's involved: Google, the speaker
Schematic trace   -- what pattern: career event, job preparation
```

These five traces are stored as structured columns on a single row in SQLite. The row is the container. The traces are the unit of meaning.

### Read Path: Resonance

At query time, the question is decomposed into the same five probe dimensions. Each stored trace responds proportionally to how many dimensions overlap with the probe. The activation follows MINERVA 2 dynamics:

```
                        5
A(i) = product of [ sim(probe_d, trace_i_d) ^ 3 ]
                       d=1

Where:
  A(i)     = activation of stored trace i
  d        = trace dimension (episodic, emotional, temporal, relational, schematic)
  sim()    = similarity between probe and stored trace on dimension d
  ^3       = cubing amplifies multi-dimensional overlap
```

The cubing is critical. A trace matching on 4 dimensions produces activation 64x stronger than a trace matching on 1 dimension. Traces self-select. The answer emerges from convergence, not from ranked search results.

The highest-activation trace is the answer candidate. Its structured fields (object, date, entity) provide the precise response. If no trace achieves convergence across multiple dimensions, the system refuses rather than guessing.

### Verification

Once a candidate trace is identified, a deterministic verification step confirms the answer exists as a stored fact:

```
Subject (from query) + Predicate (from query) + Object (from candidate) = one row in DB?
  Yes -> return the answer
  No  -> reject candidate, try next, or refuse
```

No probabilistic scoring. No thresholds. The fact exists or it does not.

---

## What Makes This Different

**No LLM in the retrieval loop.** Parsing is spaCy. Storage is SQL. Matching is arithmetic. Reconstruction is deterministic grammar rules. The intelligence layer (LLM) sits above, not inside, the continuity layer.

**Deterministic, not probabilistic.** Given the same stored traces and the same query, the system always returns the same answer. No temperature. No sampling. No "close enough."

**Sub-second latency.** The full write-read cycle runs under 1 second on consumer hardware. No API calls in the critical path.

**On-device SQLite.** All data stays on the user's device. Privacy is physics, not policy. No cloud dependency for the core memory operations.

**Model-independent.** Store traces from Claude. Retrieve them with GPT. Switch models between sessions. The continuity layer persists beneath whatever intelligence layer is active.

**Grounded in cognitive science.** MINERVA 2, Multiple Trace Theory, Temporal Context Model, hippocampal pattern completion. Not a novel invention -- a rigorous adaptation of how biological memory actually works.

---

## Architecture Diagram

```
                         WRITE PATH
                         ----------

User Input -----> Grammar Engine (spaCy + T5 SRL)
                         |
           +-------------+-------------+
           |       |       |       |       |
        Episodic  Emot.  Temporal Relat. Schematic
         Trace    Trace   Trace   Trace   Trace
           |       |       |       |       |
           +-------+-------+-------+-------+
                         |
                    SQLite Store
                  (structured rows,
                   embedded columns)


                         READ PATH
                         ----------

Query ----------> Grammar Engine (spaCy)
                         |
                  Query Decomposition
                  (5 probe dimensions)
                         |
                    Resonance Match
                  (MINERVA 2 activation
                   across all stored traces)
                         |
                  Convergence Gate
                  (multi-dimensional overlap?)
                         |
                  +------+------+
                  |             |
              Converged    No Convergence
                  |             |
            Verification     Refuse
            (SQL fact check)  ("Not mentioned
                  |            in conversation")
                  |
               Answer
```

---

## Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Parsing | spaCy (en_core_web_sm) | Dependency parse, NER, query decomposition |
| Semantic Roles | T5 SRL (220M params) | Triplet extraction, cleanup, role labeling |
| Embeddings | sentence-transformers | Trace embeddings for resonance matching |
| Storage | SQLite + FTS5 | On-device, zero-config, ACID-compliant |
| Transport | FastAPI + MCP | HTTP API and Model Context Protocol |
| Voice (optional) | Silero VAD + Whisper + Kokoro TTS | Full voice pipeline |

**Hardware requirements:** Runs on consumer hardware. 8GB GPU budget. The LLM (8B, 4-bit quantized) takes ~4.5GB. Remaining budget is shared across STT, TTS, and SRL via async load/offload -- one model resident at a time.

---

## Deployment Targets

DTCM is infrastructure, not a product. The same continuity layer serves multiple deployment contexts:

- **Consumer AI** -- Personal continuity across conversations. Message import from iMessage, WhatsApp, etc.
- **Developer SDK** -- Importable library. Write path + Read path + DTCM as a package.
- **AI Agents** -- State persistence for autonomous agents across tasks and sessions.
- **Libraries** -- Multi-patron continuity with voice profiles.
- **Clinics** -- HIPAA-compliant continuity with encrypted storage and audit trails.

---

## Further Reading

- Hintzman, D.L. (1984). MINERVA 2: A simulation model of human memory. *Behavior Research Methods, Instruments, & Computers.*
- Nadel, L. & Moscovitch, M. (1997). Memory consolidation, retrograde amnesia and the hippocampal complex. *Current Opinion in Neurobiology.*
- Howard, M.W. & Kahana, M.J. (2002). A distributed representation of temporal context. *Journal of Mathematical Psychology.*

---

*Kenotic Labs -- Building the continuity layer for AI systems.*
