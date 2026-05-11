# Nura Extractor v7 PRD

Canonical English Fact, Temporal, and Affective Extractor

## 1. Purpose

`Nura Extractor v7` is the write-path model responsible for converting messy real-world English into faithful, canonical memory facts for Nura.

It replaces and extends the current extractor role centered in:

- [app/engines/memory.py](../app/engines/memory.py)

It must remain compatible with the current write path, while materially improving:

- fact emission
- multi-fact extraction
- temporal extraction
- evidence preservation
- robustness to unseen English conversations
- affect extraction for downstream engines

The goal is not "better English triplets." The goal is a real-world extractor that can safely ingest unseen English conversations as they are actually spoken and written.

## 2. Context

The Nura architecture is now split into distinct engine responsibilities:

- `MemoryEngine`: canonical fact storage and write-path traces
- `RetrievalEngine`: query-time search, ranking, and answer assembly
- `TemporalEngine`: clustering, supersession, chronology, arcs
- `AdaptabilityEngine`: user style and relational adaptation
- `ProactiveEngine`: arc/timer/frequency-based surfacing

Recent structural fixes have moved the architecture from a low baseline to a much stronger state. The remaining ceiling is now mostly extractor quality:

- wrong relation semantics
- missing secondary facts
- missing duration facts
- malformed predicate serialization
- weak subject-centered fact formation
- weak affect/emotion structuring for downstream adaptation

The architecture is no longer the main bottleneck. The next major gain must come from the extractor.

## 3. Product Goal

Build a production extractor that maps unseen real-world English into stable Nura facts, temporal structures, and affective signals, while scaling across:

- direct English statements
- casual English conversation
- slangy English
- ASR-like noisy English
- typo-corrupted English
- self-corrected English
- multi-clause English
- short discourse-window English conversation

## 4. Non-Goals

- Not a response generator
- Not a retrieval reranker
- Not a grammar beautifier
- Not a regex/template system
- Not a multilingual parser
- Not a language-specific rules engine that grows by exceptions

## 5. User Stories

As a user:

- if I say `I work at Google as a software engineer`, both employer and role should be stored
- if I say `My girlfriend's name is Mika. We've been together for two years`, identity and duration should be stored correctly
- if I say `I graduated from MIT with a computer science degree`, both school and field of study should be stored
- if I say `I've been at Google for three years`, the duration should be attached to the employment fact
- if I say `I'm anxious about tomorrow`, the system should preserve emotional state as a first-class signal
- if I speak casually, indirectly, or with self-corrections, facts should still be recovered
- if I speak in unfamiliar but still English conversational phrasing, the system should still recover grounded facts or preserve evidence safely

## 6. Product Scope

`v7` is an English-only unified extractor layer with six responsibilities:

- canonical fact extraction
- temporal extraction
- local discourse resolution
- normalization into Nura's internal schema
- evidence preservation for out-of-distribution or low-confidence English cases
- explicit affect/emotion extraction for downstream adaptation and proactive behavior

## 7. Functional Requirements

### 7.1 Canonical Fact Extraction

The model must emit canonical memory facts usable by the current write path.

Minimum canonical fields:

- `subject`
- `predicate`
- `object`
- `is_historical`

Extended required fields:

- `subject_surface`
- `predicate_surface`
- `object_surface`
- `language`
- `script`
- `evidence_span`
- `confidence`
- `normalization_status`

`normalization_status` values:

- `canonical`
- `provisional`
- `surface_only`

### 7.2 Multi-Fact Extraction

The model must emit all materially distinct facts from one utterance.

Examples:

- `I work at Google as a software engineer`
  emits `works_at` and `occupation`
- `My favorite food is sushi, especially salmon nigiri`
  emits general and specific preference facts
- `I graduated from MIT with a computer science degree`
  emits `graduated_from` and `degree`
- `My girlfriend's name is Mika. We've been together for two years`
  emits partner identity and duration

No assumption is allowed that one utterance contains only one primary fact.

### 7.3 Temporal Extraction

The model must emit temporal structure explicitly, not bury time inside object text.

It must extract:

- absolute dates
- relative dates
- durations
- intervals
- recurrence
- tense/state timing
- historical/current/future/conditional status

Examples:

- `March 15th`
- `2022`
- `last month`
- `for three years`
- `30 miles per week`
- `twice a month`
- `used to`
- `planning to`
- `if the pain returns`

### 7.4 Temporal Representation

Each fact may carry a temporal payload.

Required temporal fields:

- `time_type`
- `surface`
- `normalized`
- `anchor`
- `recurrence`
- `state`

Example:

```json
{
  "subject": "user",
  "predicate": "works_at",
  "object": "Google",
  "is_historical": false,
  "temporal": {
    "time_type": "duration",
    "surface": "for three years",
    "normalized": "P3Y",
    "anchor": null,
    "recurrence": null,
    "state": "current"
  }
}
```

Rules:

- if normalization is uncertain, preserve `surface` and leave normalized time partial or null
- never fabricate calendar normalization for relative time without an anchor
- temporal control markers must never leak into predicate text

### 7.5 Event vs State Understanding

The extractor must distinguish:

- state facts
- events
- transitions
- plans
- conditions

Examples:

- `I live in San Francisco` is a state
- `I moved to San Francisco` is an event
- `I switched from accounting to nursing` is a transition
- `I'm thinking about switching roles` is a plan
- `If the pain returns, drop volume` is conditional

### 7.6 Local Discourse Resolution

The extractor must resolve short-range discourse within local windows.

It must support:

- pronoun resolution in adjacent sentences
- shared-subject continuation
- relation continuity across sentence boundaries
- duration attachment to the correct relation

Examples:

- `I have a dog named Luna. She's a golden retriever.`
- `My girlfriend's name is Mika. We've been together for two years.`
- `I graduated from MIT with a computer science degree. My thesis was on distributed systems.`

### 7.7 Role and Directionality Preservation

The extractor must preserve correct role direction.

Examples:

- `Mika is my girlfriend` must not become `groom`
- `My manager is Sarah Chen` must not invert manager relation
- `My best friend is Derek` must preserve relation direction and entity identity

### 7.8 Predicate Serialization Discipline

The extractor must never emit malformed internal predicates.

Forbidden outputs include:

- leaked training markers
- leaked historical markers
- duplicated suffixes
- invalid predicate concatenations

Example forbidden form:

- `historical_relationship_historical`

This is a hard model failure.

### 7.9 Canonical Predicate Mapping

The extractor must map English surface language into a controlled internal predicate inventory.

Core predicate families for `v7`:

- `works_at`
- `occupation`
- `manager_name`
- `lives_in`
- `rent_per_month`
- `birthday_date`
- `partner_name`
- `relationship_duration`
- `best_friend`
- `allergic_to`
- `has_pet`
- `pet_name`
- `pet_breed`
- `favorite_food`
- `training_for`
- `runs_per_week`
- `graduated_from`
- `degree`
- `thesis_topic`
- `worked_for_duration`
- `lives_since`
- `has_emotion`
- `feels_about`
- `emotional_state`

Mapping must be model-learned and ontology-backed, not maintained as phrase lists.

### 7.10 Unseen-English Graceful Degradation

The extractor must support unseen or weakly represented English conversation styles by degrading safely.

Support means:

- preserve source spans
- preserve relation surface text
- avoid false canonical mappings
- emit `surface_only` or `provisional` when needed

It does not mean forcing a canonical predicate for every unfamiliar English utterance.

### 7.11 Quantity and Unit Extraction

The extractor must represent quantities structurally.

Examples:

- `30 miles per week`
- `$2800 a month`
- `three years`
- `March 15th`

Quantities should not remain opaque text blobs when units matter for later QA.

### 7.12 Negation, Correction, and Supersession Intent

The extractor must recognize:

- negation
- correction
- replacement
- historical state
- supersession intent

Examples:

- `not X, Y`
- `actually`
- `I used to`
- `I switched from X to Y`
- `no longer`

This reduces downstream ambiguity and helps write-time temporal handling.

### 7.13 Type Hints

The extractor should emit soft type hints for subject and object:

- `PERSON`
- `ORG`
- `LOCATION`
- `TIME`
- `QUANTITY`
- `ROLE`
- `ANIMAL`
- `FOOD`
- `EMOTION`

These are hints, not hard truth, and should support downstream typing.

### 7.14 Emotion and Affect Extraction

The extractor must explicitly support affective content for downstream use.

It must extract:

- explicit self-reported emotions
- stable emotional states
- valenced reactions tied to an event or topic
- intensity when expressed
- temporal status of the emotion

Examples:

- `I feel anxious`
- `I've been overwhelmed lately`
- `I'm happy about the promotion`
- `I was really scared after the accident`

Required affect payload:

- `emotion_label`
- `valence_hint`
- `intensity_hint`
- `target_surface`
- `target_entity`
- `state`

Example:

```json
{
  "subject": "user",
  "predicate": "has_emotion",
  "object": "anxious",
  "is_historical": false,
  "emotion": {
    "emotion_label": "anxious",
    "valence_hint": "negative",
    "intensity_hint": "medium",
    "target_surface": "tomorrow",
    "target_entity": null,
    "state": "current"
  }
}
```

The extractor must not hallucinate hidden emotional states. It should emit affect only when there is linguistic evidence.

## 8. Engine Integration Requirements

### 8.1 Memory Engine Integration

The extractor is the producer for `MemoryEngine`.

It must emit facts that survive the current write-path contract in:

- [app/engines/memory.py](../app/engines/memory.py)

The extractor should reduce downstream normalization burden rather than depend on increasingly permissive validators.

### 8.2 Retrieval Engine Integration

The extractor must emit predicates and temporal structures that are retrieval-safe.

This means:

- stable predicate serialization
- factual duration predicates
- relation directionality
- type hints that help retrieval without becoming hard gates

### 8.3 Temporal Engine Integration

The extractor must produce facts that temporal logic can reason over cleanly.

This means:

- explicit historical status
- explicit temporal payloads
- explicit transition/change semantics
- no leaking of temporal flags into predicate text

### 8.4 Adaptability Engine Integration

Current `AdaptabilityEngine` in [app/engines/adaptability.py](../app/engines/adaptability.py) does not consume extractor facts directly. It updates user style from raw turn embeddings via `update_from_turn()`.

That means:

- the extractor is not currently required for baseline adaptability
- but `v7` should still emit affective signals that adaptability can consume later

Recommendation:

- keep `AdaptabilityEngine` turn-level for warmth, formality, initiative, and check-in preference
- add extractor-derived affect facts as additive signals, not replacements

Why:

- turn-level embeddings capture style better than explicit fact extraction alone
- explicit emotion extraction is still valuable for longitudinal adaptation, mood continuity, and crisis-sensitive behavior

### 8.5 Proactive Engine Integration

Current `ProactiveEngine` in [app/engines/proactive.py](../app/engines/proactive.py) is still minimal, but it is intended to use relational and emotional traces.

For proactive behavior, the extractor should support:

- emotionally salient events
- persistent concerns
- recurring stressors
- emotionally loaded future commitments

This enables later proactive behaviors such as:

- checking in after a stressful event
- following up on emotionally important plans
- surfacing unresolved topics

## 9. Output Schema

Required top-level output:

```json
{
  "facts": [
    {
      "subject": "user",
      "predicate": "occupation",
      "object": "software engineer",
      "is_historical": false,
      "subject_surface": "I",
      "predicate_surface": "work at ... as",
      "object_surface": "software engineer",
      "language": "en",
      "script": "Latn",
      "evidence_span": {"start": 20, "end": 39},
      "confidence": 0.94,
      "normalization_status": "canonical",
      "subject_type_hint": "PERSON",
      "object_type_hint": "ROLE",
      "temporal": null,
      "emotion": null
    }
  ]
}
```

Unknown or low-confidence canonicalization:

```json
{
  "facts": [
    {
      "subject": null,
      "predicate": null,
      "object": null,
      "is_historical": false,
      "subject_surface": "yo",
      "predicate_surface": "anda con",
      "object_surface": "Mika",
      "language": "es",
      "script": "Latn",
      "evidence_span": {"start": 0, "end": 15},
      "confidence": 0.41,
      "normalization_status": "surface_only",
      "subject_type_hint": "PERSON",
      "object_type_hint": "PERSON",
      "temporal": null,
      "emotion": null
    }
  ]
}
```

## 10. Recommended Model Architecture

Preferred architecture:

- English encoder-decoder base for canonical extraction
- subword or byte-aware robustness path for noisy English text

Recommended model direction:

- T5-family English seq2seq model as the core multitask extractor
- optional byte-aware denoising support only for noisy English robustness

Reason:

- the target is not multilingual generalization
- the target is strong generalization across unseen English conversations
- the current architecture already expects T5-style multitask prefixes for cleanup, triplets, and roles

`v7` should be designed as an English-only canonical extractor that generalizes across unseen English phrasing.

## 11. Training Data Strategy

### 11.1 Gold Data

Build a gold dataset around the exact English memory and affect relations Nura needs:

- employment
- occupation
- managers
- partner identity
- friendship
- pets
- education
- allergies
- preferences
- durations
- locations
- birthdays
- money/rates
- routines
- explicit emotions
- emotionally valenced reactions
- long-lived emotional states

Every example must follow the extractor schema:

- `source_text`
- `expected_facts`
- optional `negative_facts`
- optional `temporal`
- optional `emotion`
- language metadata

### 11.2 Coverage of Unseen English Conversation

For every major relation family, include:

- direct phrasing
- indirect/casual phrasing
- short discourse continuation
- typo/noisy phrasing
- ASR-style phrasing
- slang/informal phrasing
- self-corrected phrasing
- elliptical/conversational phrasing

### 11.3 Negative Data

Include negatives for:

- wrong role direction
- wrong subject anchoring
- wrong canonical predicate
- malformed serialization
- historical marker leakage
- over-extraction
- false duration attachment
- false emotion attribution

### 11.4 Weak Supervision

Large English chat corpora may only be used to mine candidate utterances.

They are not direct extractor training data unless converted into the canonical extractor schema.

That means large corpora should be used to:

- mine human-like phrasing
- mine English paraphrases
- mine casual English memory utterances
- mine English emotion-laden natural language
- then relabel into extractor supervision

### 11.5 Existing Local Data

Current aligned local data is under:

- [data/adversarial/README.md](../data/adversarial/README.md)

It is useful, but too small and too failure-taxonomy-oriented to be the full `v7` corpus by itself.

Existing local profile-style sources worth mining and relabeling:

- [locomo_bench/locomo/data/msc_personas_all.json](../locomo_bench/locomo/data/msc_personas_all.json)
- [locomo_bench/locomo/data/locomo10.json](../locomo_bench/locomo/data/locomo10.json)

Generic external corpora should mostly be treated as mining material, not direct supervision.

## 12. Evaluation Plan

### 12.1 Core Extractor Metrics

- Emission Recall
- Multi-Fact Recall
- Canonical Predicate Accuracy
- Subject/Object Accuracy
- Temporal Accuracy
- Historical Accuracy
- Emotion Accuracy
- Emotion Target Accuracy
- Serialization Validity
- Evidence Fidelity

### 12.2 Layered System Metrics

Retain the current layered evaluation:

- `L0` model emission
- `L1` stored extraction coverage
- `L2` type correctness
- `L3` top-1 edge retrieval
- `L4` final answer correctness

`v7` is primarily responsible for improving `L0` and indirectly raising the rest.

### 12.3 Unseen-English Robustness Metrics

- noisy-English robustness
- slang/informal-English robustness
- ASR-English robustness
- self-corrected-English robustness
- out-of-distribution English surface-preservation rate
- false-canonicalization rate on low-confidence English inputs

### 12.4 Hard Failure Budget

The model is not shippable if it frequently produces:

- hallucinated roles
- malformed predicates
- leaked control markers
- false canonicalization in weak-language cases
- systematic omission of durations
- single-fact collapse on multi-fact utterances
- hallucinated emotional state

## 13. Acceptance Criteria for v7

`v7` is successful when:

- it clears the current `v6` ceiling on the existing benchmark
- it eliminates malformed predicate serialization
- it materially improves multi-fact extraction
- it emits durations and historical state explicitly
- it emits explicit affective facts when grounded in text
- it preserves evidence safely on unseen-English or low-confidence inputs
- it requires no language-specific hand templates
- it remains compatible with the current write path

The aspiration target is `100%` on supported English relation families and unseen English conversations.

The expected practical operating target is:

- `90%+` on supported English relation families
- with strong performance on noisy, casual, and conversational English
- with low hallucination on unfamiliar English phrasing

## 14. Rollout Plan

### Phase 1: v7a

English and current failing relation families only.

Goal:

- fix current benchmark failures
- prove output schema and training setup

### Phase 2: v7b

Unseen-English robustness expansion.

Goal:

- support broader English conversational variation
- improve noisy/slang/ASR/generalization

### Phase 3: v7c

Hard-English edge cases.

Goal:

- stronger cleanup under noisy English
- `surface_only` safe mode for unfamiliar English phrasing
- better short-range discourse and temporal duration extraction

### Phase 4: v7d

Affect-aware integration.

Goal:

- explicit `has_emotion` and related affect facts
- adaptation/proactive integration without replacing existing turn-level style signals

## 15. Risks

- English canonical mapping can still drift into hallucination if evidence-preservation is weak
- Byte-aware robustness improves denoising but raises compute cost
- Temporal normalization can create false certainty if anchor logic is weak
- Local discourse resolution can over-link entities if not trained carefully
- Generic chat corpora can dilute extractor quality if used directly instead of relabeled
- Emotion extraction can over-ascribe hidden state if training data over-rewards inference

## 16. Open Product Decisions

- whether the model should emit both canonical facts and evidence facts in one pass, or canonical facts plus a separate evidence attachment layer
- whether historical duration should be stored as fact-local temporal metadata or as parallel temporal edges
- whether type hints should be extractor-owned or remain downstream-only
- whether byte-aware robustness remains a helper path for cleanup or becomes part of the main backbone
- whether affect should be emitted only for explicit self-report or also for strongly grounded implied affect

## 17. Implementation Guidance

The extractor should continue to feed the current `MemoryEngine`, but the architecture should treat canonical fact extraction, temporal extraction, affect extraction, and cleanup as first-class outputs, not as incidental text patterns.

The main product change is conceptual:

- `v6` was a triplet emitter
- `v7` should be an English canonical memory extractor with cleanup, temporal structure, affect structure, and evidence preservation

## 18. Success Definition

If `v7` ships correctly, Nura should stop failing mainly because the model never emitted the fact from unseen English conversation. At that point:

- `MemoryEngine` becomes a stable consumer of better facts
- `RetrievalEngine` stops compensating for malformed write output
- `TemporalEngine` gets cleaner state/time structure
- `AdaptabilityEngine` can consume affect facts additively
- `ProactiveEngine` can use emotionally salient arcs and concerns

## 19. Code-Rooted Note on Emotions

Today, Nura already computes emotional traces without T5:

- [app/engines/memory.py](../app/engines/memory.py) computes `edge_emotional_valence` and `edge_emotional_label`
- [app/engines/retrieval.py](../app/engines/retrieval.py) uses those in cluster fusion and mood summaries
- [app/engines/adaptability.py](../app/engines/adaptability.py) currently uses turn-level embeddings, not extractor facts
- [app/engines/proactive.py](../app/engines/proactive.py) is intended to use relational and emotional traces later

So the answer to "can T5 extract emotions?" is:

- yes, it can and should for explicit affective facts
- but it should complement, not replace, the existing turn-level adaptation signals

## 20. T5 Task Requirements

The current architecture already expects a T5-style multitask extractor with explicit task routing in [app/engines/memory.py](../app/engines/memory.py).

`v7` must support these tasks:

- `<cleanup>`: noisy English to semantically equivalent clean English
- `<triplets>`: clean English to canonical facts
- `<roles>`: temporal and role-bearing extraction
- optional `<affect>`: explicit affect extraction if split into a separate head/task

Cleanup is in scope and required.

Cleanup must:

- repair noisy English
- preserve semantics
- preserve relation direction
- preserve temporal cues
- preserve emotional cues

Cleanup must not:

- invent subjects
- invent predicates
- rewrite one relation into another
- erase duration or temporal markers
- convert weak implication into strong fact

## 21. Sources

- Universal Dependencies: https://universaldependencies.org/
- UD disfluency / reparandum: https://universaldependencies.org/u/dep/reparandum.html
