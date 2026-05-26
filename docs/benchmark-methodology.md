# Benchmark Methodology: ATANT

How Kenotic Labs measures continuity -- and why the standard benchmarks are not enough.

---

## Why Not Just LOCOMO?

LOCOMO (Long Conversation Memory) is the most cited benchmark for conversational memory systems. We use it as a reference metric for external credibility, but we do not optimize to it. Here is why:

**It tests roughly 2 of our 7 continuity properties.** LOCOMO covers basic retrieval and some temporal reasoning. It has no tests for persistence beyond session, update handling, disambiguation, reconstruction, supersession, model independence, or operational usefulness.

**23% of LOCOMO questions are unscorable by design.** Category 5 (446 questions) has empty gold-standard answers. The scoring function returns False on empty gold regardless of the system's response. Any benchmark number that includes these questions is inflated by noise.

**42% of LOCOMO is substring-matched narrative paraphrase.** Category 4 (841 questions) uses substring matching against long gold answers. This measures verbosity and phrasing luck, not whether the system actually understood the situation. A system that returns the entire conversation transcript would score well on Category 4.

**The category labels in the original paper were misordered.** Category 2 is temporal, Category 4 is narrative comprehension. Any analysis built on the original label ordering is structurally wrong.

LOCOMO answers the question: "Can your system retrieve text that matches a gold answer?" ATANT answers the question: "Does your system understand and maintain continuity?"

---

## What Is ATANT?

**Automated Test for Acceptance of Narrative Truth.**

ATANT is Kenotic Labs' internal benchmark, purpose-built to test the 7 properties of continuity. It consists of 500 narrative stories with 20 checkpoints per story, evaluated at cumulative intervals.

The canonical runner is `run_atant_cumulative.py`. It processes all 500 stories from `tests/stories/cumulative/` and reports scores at 20 checkpoint intervals.

A separate 5-story smoke test (`run_atant_ground_truth.py`) exists for fast iteration but is not the truth signal.

---

## The 20 Checkpoints

The checkpoints are grouped by capability:

### Write Path (CP1-CP4)
| CP | What It Tests |
|----|---------------|
| CP1 | Extraction -- structured traces are produced from raw input |
| CP2 | Storage -- traces persist correctly in the database |
| CP3 | Predicted queries -- anticipated questions are generated at write time |
| CP4 | Type tagging -- traces are assigned correct schematic categories |

### Read Path (CP5-CP8)
| CP | What It Tests |
|----|---------------|
| CP5 | Query classification -- incoming questions are parsed into the right category |
| CP6 | Structural match -- the right candidate traces are surfaced |
| CP7 | DTCM convergence -- multi-dimensional trace convergence produces the correct answer |
| CP8 | Final answer -- the system returns a correct, grounded answer to the question |

### Traces and Emotion (CP9-CP10)
| CP | What It Tests |
|----|---------------|
| CP9 | Trace completeness -- all 5 trace dimensions are populated |
| CP10 | Emotional valence -- emotional traces are correctly extracted and scored |

### Reconstruction Quality (CP11-CP12)
| CP | What It Tests |
|----|---------------|
| CP11 | Grammar reconstruction -- answers are grammatically well-formed |
| CP12 | Answer similarity -- reconstructed answers are semantically close to ground truth |

### Advanced Properties (CP13-CP17)
| CP | What It Tests |
|----|---------------|
| CP13 | Emotion detection -- the system identifies emotional states from context |
| CP14 | Relational accuracy -- relationships between entities are correctly tracked |
| CP15 | Schematic patterns -- recurring life patterns are recognized |
| CP16 | Frequency weighting -- repeated information is weighted appropriately |
| CP17 | Proactive surfacing -- the system anticipates what the user needs |

### Continuity-Specific (CP18-CP20)
| CP | What It Tests |
|----|---------------|
| CP18 | Situation reconstruction -- the system produces a coherent current-state summary |
| CP19 | Contradiction detection -- conflicting information is identified, not merged |
| CP20 | is_current filtering -- superseded information is excluded from current-state answers |

---

## Truth Signal

The primary truth signal is **cp8_answer on the full 500 stories** via `run_atant_cumulative.py`.

CP8 represents end-to-end correctness: a question goes in, a verified answer comes out. It exercises the entire pipeline -- write path, trace decomposition, query classification, structural matching, DTCM convergence, and final answer generation.

**The monotonic improvement rule applies:** the cp8_answer score on the 500-story run must never decrease between changes. If a code change causes cp8_answer to drop, it is reverted regardless of improvements elsewhere.

---

## Current Baseline

As of 2026-04-10:

- **cp8_answer on stories 51-55:** 22%

This is the honest starting line. An earlier figure of 94% was measured on the 5-story smoke test runner, which is not representative of real-world performance. We do not cite it.

The target is 100% across all 20 checkpoints. Progress is measured incrementally and reported transparently.

---

## LOCOMO as Reference Only

LOCOMO scores are tracked for external credibility with investors and reviewers. They are never used as an optimization target.

Every proposed improvement must answer the question: **"Which continuity property does this serve?"** If the answer is "it improves LOCOMO but does not serve a continuity property," the proposal is rejected.

The 7 continuity properties that ATANT measures:

1. **Persistence beyond session** -- memory survives shutdown, restart, device change
2. **Update handling** -- changed facts are tracked as both historical and current
3. **Temporal ordering** -- events are sequenced, dates are resolved, status is tracked
4. **Disambiguation** -- similar entities and events are kept structurally separate
5. **Reconstruction** -- the system produces coherent situation-level answers, not fragments
6. **Model independence** -- one model writes, another reads; continuity lives below the LLM
7. **Operational usefulness** -- the system works in clinics, libraries, agents, not just chat

LOCOMO tests approximately properties 3 and 5 in limited form. ATANT tests all seven.
