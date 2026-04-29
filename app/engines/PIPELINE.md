# NURA Ingest Pipeline — Model Roles and Rules

Every component in the ingest pipeline has a specific role, specific input expectations,
and specific failure modes. This document is the system prompt for the pipeline.

---

## Pipeline Flow

```
Raw dialogue turn (messy human speech)
  |
  v
[1] STRUCTURAL CLEANUP (spaCy dep labels — no model inference)
  |  Role: Strip conversational scaffolding. Normalize to clean English.
  |  Output: One or more clean sentences, each a grammatical clause.
  |
  v
[2] GRAMMAR POLISH (CoEdit coedit-small — T5 seq2seq)
  |  Role: Fix grammar errors ONLY. Do not change meaning.
  |  Output: Grammatically correct version of each sentence.
  |
  v
[3] GRAMMAR ENGINE (spaCy dep parse + WordNet — deterministic)
  |  Role: Decompose clean English into 5 traces.
  |  Output: TraceDecomposition (episodic, emotional, temporal, relational, schematic)
  |
  v
[4] MEMORY STORE (SQLite — deterministic)
     Role: Persist traces. Resolve speaker. Supersede stale facts.
```

---

## [1] STRUCTURAL CLEANUP

### Model
spaCy `en_core_web_md` — dep parse only. No model inference beyond spaCy's pipeline.

### Role
Convert messy human dialogue into clean, well-formed English sentences that
the grammar engine can parse. This is the ONLY place that handles conversational
noise. Everything downstream expects clean input.

### Input
Raw dialogue turns. May contain:
- Fillers: "um", "like", "you know", "I mean", "basically"
- False starts: "I was gonna say... actually"
- Self-corrections: "it was 2018, no wait, 2019"
- Retractions: "never mind", "forget it", "scratch that"
- Discourse frames: "the thing is", "long story short", "here's the deal"
- Rhetorical questions: "you know what gets me?"
- Run-on speech: no punctuation, multiple clauses
- Contractions: "gonna", "wanna", "kinda", "dunno"
- Backchannels: "yeah", "right", "totally", "hmm"

### Output
One or more clean English sentences. Each sentence:
- Has a subject and verb
- Has correct punctuation
- Contains ONLY factual content (no scaffolding)
- Preserves all named entities, dates, locations
- Preserves the speaker's intended meaning

### Detection Rules (all structural — dep labels + POS)

**1. INTJ tokens (POS=INTJ):** Remove entirely.
- "um", "oh", "wow", "haha", "hmm"
- spaCy tags these with POS=INTJ

**2. Parataxis clauses (dep=parataxis):** Remove entire subtree.
- "you know", "I mean", "you see"
- spaCy tags the verb as dep=parataxis

**3. Discourse frames:** Remove the frame, keep the content.
Detection: ROOT or ccomp verb whose lemma signals meta-speech,
with a subordinate clause that has its own subject+verb.

Patterns:
- ROOT is meta-verb + 1st/2nd person subject + ccomp/xcomp with own subject
  "I think [she moved to Portland]" → keep "she moved to Portland"
  But NOT "I think pineapple is good" — here "think" IS the content (opinion)
  Distinction: if the ccomp subject != ROOT subject → strip frame.
               if the ccomp subject == ROOT subject → keep (it's the same person's fact)

- ccomp is meta-frame (discourse in subordinate position)
  "[the thing is], I applied for this position" → ROOT=applied, ccomp=is
  Detection: ccomp whose subtree has no content nouns beyond filler ("thing", "point", "deal")

- Idiom adverbials (no verb):
  "long story short", "bottom line", "at the end of the day"
  Detection: adverbial phrase at sentence start before comma, no verb in the phrase

- Rhetorical question + answer:
  "You know what gets me? [People who leave carts]"
  Detection: sentence ends with "?", next sentence is a fragment (no main verb)
  Action: discard the question, keep the answer

**4. Self-corrections:** Keep the correction, discard the original.
- "it was 2018, no wait, 2019" → "it was 2019"
- Detection: "no" + meta-verb ("wait", "actually") + replacement value
- "I have two brothers, no, THREE brothers" → "I have three brothers"
- Detection: "no" or "actually" between two values

**5. Retractions:** Discard the entire utterance.
- "I was gonna say... actually never mind"
- "forget it", "scratch that", "never mind"
- Detection: retraction phrase ("never mind", "forget it", "scratch that")
  at end of utterance → return empty string

**6. Sentence splitting:**
- Split on periods, "!", "?"
- Split on ellipsis "..." and dashes "—" / "–"
- Split on semicolons (already in grammar engine, but should also happen here)
- Do NOT split on commas (comma splices are handled by grammar engine)

**7. Contraction normalization:**
- "gonna" → "going to"
- "wanna" → "want to"
- "kinda" → "kind of"
- "dunno" → "don't know"
- "gotta" → "got to"
- These are closed-class — finite list is appropriate (grammatical forms, not vocabulary)

### What this step does NOT do
- Does NOT fix grammar (that's CoEdit's job)
- Does NOT resolve pronouns (that's the grammar engine's job)
- Does NOT extract facts (that's the grammar engine's job)
- Does NOT interpret meaning (that's the retrieval engine's job)

### Failure modes
- Over-stripping: removing content verbs that look like discourse ("I think" when it IS the opinion)
- Under-stripping: leaving discourse frames that confuse the grammar engine
- Entity loss: removing tokens that contain named entities

### Guard
After cleanup, verify: does the output contain at least one VERB or NOUN?
If not, the cleanup stripped everything → return the original text.

---

## [2] GRAMMAR POLISH (CoEdit)

### Model
`jbochi/coedit-small` (77M params, flan-t5-small fine-tuned on CoEdit dataset)

### Role
Fix grammatical errors in structurally-cleaned sentences. This is a
NARROW role — grammar correction ONLY. The model does NOT do:
- Dialogue preprocessing
- Discourse stripping
- Self-correction detection
- Meaning changes of any kind

### Supported task prefixes (from EMNLP 2023 paper)
1. `Fix grammatical errors in this sentence:` — GEC (what we use)
2. `Simplify this sentence:` — Text simplification (NOT used)
3. `Make this text coherent:` — Coherence (NOT used)
4. `Paraphrase this sentence:` — Paraphrase (NOT used)
5. `Make this text formal:` — Formality transfer (NOT used)
6. `Make this text neutral:` — Neutralization (NOT used)

We use ONLY prefix #1. The other prefixes change meaning, which violates
the lossless contract.

### Input constraints
- Max input: 128 tokens (truncated by tokenizer)
- Max output: 96 tokens per sentence
- Multi-sentence input: split into individual sentences BEFORE calling CoEdit
- One sentence at a time — never a paragraph

### Output contract
- Output must preserve all named entities from input
- Output must preserve the ROOT verb lemma from input
- Output must not be more than 50% shorter than input
- If any of these are violated → reject the rewrite, use structural cleanup output

### Failure modes
- Hallucination: generates text not in input ("triple/tuple syntax")
- Meaning change: rewrites "You know how I mentioned" → "How did I mention"
- Content loss: truncates long sentences
- Hyphenated word breakage: "co-parent" → "co - parent"

### Guards (implemented in _rewrite_one)
1. ROOT verb lemma comparison (input vs output)
2. NER entity preservation check
3. Length ratio check (output >= 50% of input length)
4. Tuple/triple syntax rejection

---

## [3] GRAMMAR ENGINE

### Model
spaCy `en_core_web_md` (dep parse, POS, NER, lemmatization)
+ NLTK WordNet (hypernym closure for verb/noun classification)

### Role
Decompose one clean English sentence into 5 traces:
- Episodic: what happened (clause minus subject)
- Emotional: how it felt (ADJ in acomp/attr/oprd, animate subject only)
- Temporal: when (DATE/TIME NER, verb morphology, adverbial markers)
- Relational: who (PERSON/ORG/GPE NER, speaker resolution)
- Schematic: what domain (verb class → schema via 7-step derivation)

### Input expectations
- ONE well-formed English sentence
- Has a subject and verb
- Has correct punctuation
- Contains factual content (no discourse scaffolding)
- Named entities intact
- If input doesn't meet these → garbage in, garbage out

### Architecture
- Single extraction path for ALL sentence types
- Classification (statement/question/command) sets metadata, not code path
- Universal frame skipper: walks past framing verbs to content verb
- Imposed facts: subordinate clauses extracted via dep labels
- No LLM at any point — fully deterministic

### What this step does NOT do
- Does NOT clean dialogue (that's the cleanup step)
- Does NOT resolve 3rd-person coreference (known limitation)
- Does NOT interpret tense backshift in reported speech
- Does NOT detect free indirect speech (cross-sentence, needs discourse model)

---

## [4] MEMORY STORE

### Model
None — pure SQLite operations.

### Role
- Resolve "I"/"me"/"my" to speaker name
- Resolve "my X" to "Speaker's X" in objects
- Upsert relationship rows (dedup by source_text_hash)
- Write trace columns from grammar decomposition
- Resolve event dates (4-tier temporal waterfall)
- Detect supersession (new facts replace stale ones)

### Input expectations
- TraceDecomposition with all 5 traces populated
- Speaker name provided by caller
- Source text for dedup hashing

---

## Debugging checklist

When the pipeline produces wrong output:

1. **Check cleanup output first.** Did the messy input get cleaned properly?
   If cleanup output is wrong → fix Pass 1 detection rules.
   If cleanup output is clean but grammar engine output is wrong → grammar engine bug.

2. **Check CoEdit output.** Did it change meaning?
   If CoEdit corrupted the sentence → semantic guard should have caught it.
   If guard didn't catch it → tighten the guard.

3. **Check grammar engine output.** Are the 5 traces correct?
   If dep tree is right but trace assignment is wrong → mapping bug.
   If dep tree is wrong → spaCy limitation (fragment, informal speech).

4. **Never bypass a step.** Don't feed messy dialogue directly to the grammar engine.
   Don't skip CoEdit. Don't skip cleanup. The pipeline is sequential for a reason.
