"""
Pipeline configuration — runtime rules for each model in the ingest path.

These are the "system prompts" for off-the-shelf models. Each model is
designed for a general task. These configs tell it what to do in OUR pipeline.

Import and use these in the corresponding engine files.
"""

# =====================================================================
# [1] STRUCTURAL CLEANUP — spaCy dep-label rules
# =====================================================================
# spaCy doesn't take a "prompt". Its behavior is controlled by which
# dep labels we act on. These sets define what gets stripped.

# POS tags that are always conversational filler — remove entirely
FILLER_POS = frozenset({"INTJ"})

# dep labels whose entire subtree is conversational scaffolding
SCAFFOLDING_DEPS = frozenset({"parataxis"})

# Discourse frame verbs — when ROOT (or ccomp) is one of these with
# first/second person subject, the verb is meta-speech, not content.
# These are grammatical function words in dialogue, not vocabulary.
DISCOURSE_FRAME_LEMMAS = frozenset({
    "wait", "remind", "mean", "say", "tell", "know",
    "think", "wonder", "guess", "suppose", "remember",
    "hear", "listen", "look",  # attention-getters
})

# Discourse subjects — first AND second person trigger frame detection
DISCOURSE_SUBJECT_LEMMAS = frozenset({
    "i", "we", "you",
})

# Discourse filler nouns — in "the thing is", "the point is", "the deal is"
# these nouns signal a discourse frame when they're the subject of "be"
DISCOURSE_FILLER_NOUNS = frozenset({
    "thing", "point", "deal", "fact", "truth", "matter",
    "problem", "issue", "question",
})

# Idiom adverbial frames — sentence-initial phrases before comma
# that are conversational framing, not content.
# These are fixed multi-word expressions (MWEs), not vocabulary.
IDIOM_FRAMES = frozenset({
    "long story short",
    "bottom line",
    "at the end of the day",
    "truth be told",
    "between you and me",
    "to be honest",
    "to be fair",
    "for what it's worth",
    "if you ask me",
    "believe it or not",
    "here's the thing",
    "here's the deal",
})

# Retraction signals — when these appear at the end of an utterance,
# the entire utterance should be discarded (speaker withdrew the content)
RETRACTION_PHRASES = frozenset({
    "never mind",
    "nevermind",
    "forget it",
    "forget that",
    "scratch that",
    "disregard that",
    "ignore that",
})

# Self-correction signals — the word(s) that mark a correction boundary.
# Content AFTER the signal replaces content BEFORE it.
CORRECTION_MARKERS = frozenset({
    "actually",  # "it was 2018, actually 2019"
    "no wait",
    "wait no",
    "no",  # "I have two, no, three brothers" (only when between values)
    "I mean",
})

# Informal contractions → standard forms
# These are grammatical forms, not vocabulary — closed class.
CONTRACTION_MAP = {
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
    "ain't": "is not",
    "y'all": "you all",
    "ima": "I'm going to",
    "tryna": "trying to",
    "finna": "fixing to",
}


# =====================================================================
# [2] COEDIT GRAMMAR POLISH — task prefix and constraints
# =====================================================================
# CoEdit is a T5 seq2seq model. Its "system prompt" is the task prefix.

# The ONLY prefix we use. CoEdit was trained on 7 tasks but we use
# only GEC. The other tasks change meaning (simplify, paraphrase, etc.)
COEDIT_TASK_PREFIX = "Fix grammatical errors in this sentence:"

# Token limits (from model architecture)
COEDIT_MAX_INPUT_TOKENS = 128
COEDIT_MAX_OUTPUT_TOKENS = 96

# Generation config (deterministic, no sampling)
COEDIT_NUM_BEAMS = 2
COEDIT_DO_SAMPLE = False

# Semantic guard thresholds
COEDIT_MIN_LENGTH_RATIO = 0.5  # output must be >= 50% of input length


# =====================================================================
# [3] SPACY — pipeline configuration
# =====================================================================
# spaCy's "system prompt" is which components are active.

# Required model — no fallback to smaller models
SPACY_MODEL = "en_core_web_md"

# Full pipeline — for initial parse of complete sentences
SPACY_FULL_PIPELINE_DISABLE = []  # all components active

# Fragment pipeline — for re-parsing clause fragments after splitting
# NER and sentence segmentation need full-sentence context to work.
# Fragments don't have sentence boundaries and NER degrades without context.
SPACY_FRAGMENT_PIPELINE_DISABLE = ["ner", "senter"]

# Safety limit
SPACY_MAX_LENGTH = 100_000

# Known accuracy (en_core_web_md on web text):
#   POS tagging:  97.2%
#   Dep parsing:  91.7% UAS / 89.9% LAS
#   NER:          84.5% F1
#   Sent segment: 90.6% F1
# Accuracy degrades on: fragments, informal speech, non-web text
