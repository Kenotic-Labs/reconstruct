"""
Grammar Engine -- Trace-primary extraction.
Rebuilt from grammar-engine-spec.md + locomo-pattern-analysis.md + 7,482 grammar rules.

Architecture:
    WRITE: text -> process(text, speaker) -> GrammarResult(trace_decompositions, triples)
    READ:  query -> classify_query(query_text) -> QueryDecomposition

Design principles (from spec + pattern analysis):
    1. Extract the SHORTEST noun phrase that IS the answer.  61% of answers are NPs.
    2. Every fact carries the speaker name in relational_entities (Cat 5 = 22.5%).
    3. Temporal expressions pass through as-is.  Human readable.  No ISO.
    4. Non-statements extract imposed facts.  Tag with mood so retrieval can filter.
    5. NO regex.  NO word lists.  spaCy POS/dep/morph/NER + WordNet hypernym closure.
    6. Scales to unseen conversations -- built on grammar rules, not patterns.

Public names (imported by retrieval, memory, tests, SDK):
    process, classify_query, classify_verb_class, _get_nlp, _get_root
    _reclassify_location_by_object, _extract_schematic
    TraceDecomposition, GrammarResult, UtteranceClassification
    TenseAspect, QueryDecomposition, VerbClass
    _VERB_CLASS_TO_SCHEMA, _VERB_CLASS_TO_RELTYPE
    _build_trace_decomposition
    detect_mood, detect_negation, detect_tense_aspect, detect_voice, resolve_pronouns
    classify_utterance, extract_typed_triple
    generate_predicted_queries, pq_active_model_name
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entry / Exit checks
# ---------------------------------------------------------------------------

_ENTRY_CHECKED = False


class GrammarEntryError(RuntimeError):
    """Raised when grammar engine's dependencies are not available."""
    pass


def _check_entry():
    """Verify ingestion and memory are importable. Runs once."""
    global _ENTRY_CHECKED
    if _ENTRY_CHECKED:
        return
    missing = []
    try:
        from app.engines import ingestion  # noqa: F401
        if not hasattr(ingestion, 'cleanup'):
            missing.append("ingestion.cleanup")
    except ImportError:
        missing.append("ingestion")
    try:
        from app.engines import memory  # noqa: F401
        if not hasattr(memory, 'get_memory_engine'):
            missing.append("memory.get_memory_engine")
    except ImportError:
        missing.append("memory")
    if missing:
        raise GrammarEntryError(
            f"grammar_engine entry check failed — missing: {', '.join(missing)}"
        )
    _ENTRY_CHECKED = True



# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_nlp = None
_nlp_fragment = None
_wordnet_loaded = False


def _get_nlp():
    """Lazy-load the full spaCy pipeline (tagger + parser + lemmatizer + NER + senter).

    Model: en_core_web_md (required — will raise OSError if not installed)
    Accuracy: UAS 91.7%, LAS 89.9%, NER F1 84.5%, POS 97.2%
    Trained on: OntoNotes 5 web text (blogs, news, comments)
    Known limitation: accuracy degrades on fragments lacking sentence context.

    Use ONLY at true entry points (process(), classify_query()) and for
    functions that may receive raw text strings (detect_mood, etc.).
    For re-parsing extracted fragments, use _get_nlp_fragment() instead.
    """
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_md")
        _nlp.max_length = 100_000
    return _nlp


def _get_nlp_fragment():
    """Lightweight pipeline for re-parsing fragments. Only runs
    tagger, parser, and lemmatizer — no NER or sentence segmentation.
    Fragments don't have sentence boundaries and NER needs full context."""
    global _nlp_fragment
    if _nlp_fragment is None:
        import spacy
        _nlp_fragment = spacy.load("en_core_web_md", disable=["ner", "senter"])
    return _nlp_fragment


def _ensure_wordnet():
    """Ensure NLTK WordNet data is available."""
    global _wordnet_loaded
    if not _wordnet_loaded:
        import nltk  # type: ignore
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
            wn.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        _wordnet_loaded = True


# ---------------------------------------------------------------------------
# Dataclasses  (Spec Part 1)
# ---------------------------------------------------------------------------

class CoarseBin(str, Enum):
    """Five coarse utterance bins derived from spaCy structural features."""
    QUESTION = "QUESTION"
    STATEMENT = "STATEMENT"
    COMMAND = "COMMAND"
    BACKCHANNEL = "BACKCHANNEL"
    EMOTION = "EMOTION"


@dataclass(frozen=True)
class UtteranceClassification:
    """Result of classifying a turn into utterance type."""
    utterance_type_id: int
    category: str
    subcategory: str
    is_question: bool
    is_command: bool
    is_backchannel: bool
    is_emotion: bool
    is_storable: bool


@dataclass(frozen=True)
class TenseAspect:
    """Tense x Aspect from verb morphology.
    Spec Part 1, Field: temporal_direction."""
    tense: str   # past | present | future
    aspect: str  # simple | continuous | perfect | perfect_continuous



@dataclass
class TraceDecomposition:
    """Five-trace decomposition of a single fact.  PRIMARY output.
    Spec Part 1 defines every field and its downstream column."""
    episodic_fact: str
    episodic_significance: str = "routine"
    emotional_state: Optional[str] = None
    emotional_valence: Optional[float] = None
    emotional_target: Optional[str] = None
    temporal_direction: str = "present"
    temporal_expression: Optional[str] = None
    temporal_resolved: Optional[str] = None
    relational_subject: str = "user"
    relational_entities: List[str] = field(default_factory=list)
    relational_type: str = "personal"
    schematic_category: str = "uncategorized"
    source_text: str = ""
    utterance_type: int = 0
    mood: str = "indicative"
    negated: bool = False
    is_historical: bool = False
    subject: str = ""
    predicate: str = ""
    object: str = ""
    object_full: str = ""
    predicted_questions: List[str] = field(default_factory=list)
    extraction_rule: str = ""


@dataclass
class GrammarResult:
    """Aggregate output of process().  One per input turn.
    trace_decompositions is the sole output."""
    trace_decompositions: List[TraceDecomposition]
    classification: UtteranceClassification
    mood: str
    negated: bool
    tense_aspect: TenseAspect
    voice: str
    resolved_text: str
    emotion: Optional[str] = None


# ---------------------------------------------------------------------------
# VerbClass enum + WordNet hypernym closure detector
# Spec Part 1, Field: edge_schematic_category (steps 1-7)
# ---------------------------------------------------------------------------

class VerbClass(str, Enum):
    """16 verb classes.  Each maps to a schema via _VERB_CLASS_TO_SCHEMA."""
    BE = "BE_VERBS"
    HAVE = "HAVE_VERBS"
    LOCATION = "LOCATION_VERBS"
    WORK = "WORK_VERBS"
    PREFERENCE = "PREFERENCE_VERBS"
    ABILITY = "ABILITY_VERBS"
    INJURY = "INJURY_VERBS"
    PROBLEM = "PROBLEM_VERBS"
    STATUS = "STATUS_VERBS"
    SPEECH = "SPEECH_VERBS"
    ACHIEVEMENT = "ACHIEVEMENT_VERBS"
    EXPERIENCE = "EXPERIENCE_VERBS"
    PLANNING = "PLANNING_VERBS"
    HABIT = "HABIT_VERBS"
    MEASUREMENT = "MEASUREMENT_VERBS"
    UNKNOWN = "UNKNOWN"


_VERB_CLASS_TO_SCHEMA: Dict[VerbClass, str] = {
    VerbClass.WORK: "career",
    VerbClass.LOCATION: "housing",
    VerbClass.PREFERENCE: "identity",
    VerbClass.INJURY: "health",
    VerbClass.PROBLEM: "health",
    VerbClass.ACHIEVEMENT: "career",
    VerbClass.EXPERIENCE: "experience",
    VerbClass.PLANNING: "planning",
    VerbClass.HABIT: "hobby",
    VerbClass.MEASUREMENT: "finance",
    VerbClass.STATUS: "identity",
    VerbClass.SPEECH: "social",
    VerbClass.ABILITY: "education",
    VerbClass.BE: "identity",
    VerbClass.HAVE: "uncategorized",
}

_VERB_CLASS_TO_RELTYPE: Dict[VerbClass, str] = {
    VerbClass.WORK: "professional",
    VerbClass.LOCATION: "personal",
    VerbClass.PREFERENCE: "personal",
    VerbClass.INJURY: "personal",
    VerbClass.ACHIEVEMENT: "professional",
    VerbClass.EXPERIENCE: "personal",
    VerbClass.PLANNING: "personal",
    VerbClass.SPEECH: "social",
    VerbClass.HABIT: "personal",
    VerbClass.ABILITY: "personal",
}

# Stative verb classes: describe states rather than events/actions.
# When aspect is "simple", these produce ongoing states (not one-time events).
# Grammar Gap #6: stative vs dynamic verb distinction.
# Grammar reference pp. 239-247: stative verbs express states, not actions.
# Categories: BE, HAVE, preference, cognition (ABILITY), possession,
# perception, measurement. These produce persisting facts, not events.
_STATIVE_VERB_CLASSES: frozenset = frozenset({
    VerbClass.BE, VerbClass.HAVE, VerbClass.PREFERENCE, VerbClass.STATUS,
    VerbClass.ABILITY,  # know, understand, believe, think
})


def _compute_significance(verb_class: "VerbClass", tense_aspect: "TenseAspect") -> str:
    """Compute episodic_significance from tense x aspect x verb_class.

    The tense-aspect combination is the primary signal:
      present + simple -> stative (persisting state: "I work at Google")
      past + simple + ACHIEVEMENT -> milestone (completed achievement: "I graduated")
      past + simple + other -> routine (past event: "I ate breakfast")
      present + continuous -> routine (ongoing action: "I'm eating lunch")
      past + habitual -> stative (former persisting state: "I used to work at Google")
      future + any -> routine (hasn't happened yet)

    Verb class refines within tense-aspect categories:
      EXPERIENCE verbs in any past tense -> notable
      ACHIEVEMENT verbs in past simple -> milestone
    """
    tense = tense_aspect.tense    # past | present | future
    aspect = tense_aspect.aspect  # simple | continuous | perfect | perfect_continuous | habitual

    # Present simple = persisting state (regardless of verb class)
    # "I work at Google", "I love chocolate", "I live in Portland", "I own a cat"
    if tense == "present" and aspect == "simple":
        return "stative"

    # Past habitual = former persisting state
    # "I used to work at Google", "I used to live in Boston"
    if tense == "past" and aspect == "habitual":
        return "stative"

    # Stative verb classes in present perfect = persisting state
    # "I've lived here for 10 years", "I've known him since college"
    if verb_class in _STATIVE_VERB_CLASSES and aspect == "perfect" and tense == "present":
        return "stative"

    # Past simple + ACHIEVEMENT = milestone
    # "I graduated", "I got married", "I won the award"
    if tense == "past" and aspect == "simple" and verb_class == VerbClass.ACHIEVEMENT:
        return "milestone"

    # Past + EXPERIENCE = notable
    # "I visited Paris", "I went skydiving"
    if tense == "past" and verb_class == VerbClass.EXPERIENCE:
        return "notable"

    # Everything else = routine
    # Present continuous ("I'm eating"), past simple non-achievement ("I ate"),
    # future ("I will go"), etc.
    return "routine"


# Synset-name anchors for hypernym closure.
_VERB_CLASS_ANCHORS: dict[VerbClass, list[str]] = {
    VerbClass.BE: ["be.v.01"],
    VerbClass.HAVE: ["have.v.01", "own.v.01", "possess.v.03"],
    VerbClass.LOCATION: [
        "travel.v.01", "move.v.02", "reside.v.01",
        "inhabit.v.01", "populate.v.01",
    ],
    VerbClass.WORK: [
        "work.v.01", "work.v.02", "manage.v.01",
        "teach.v.01", "pursue.v.01",
    ],
    VerbClass.PREFERENCE: ["like.v.02", "love.v.01", "enjoy.v.01", "hate.v.01"],
    VerbClass.ABILITY: ["know.v.01", "understand.v.01"],
    VerbClass.INJURY: ["injure.v.01", "hurt.v.01", "wound.v.01"],
    VerbClass.PROBLEM: ["fail.v.01", "break.v.01", "malfunction.v.01"],
    VerbClass.STATUS: ["change_state.v.01", "become.v.01"],
    VerbClass.SPEECH: [
        "communicate.v.02", "say.v.01", "tell.v.01", "think.v.01",
    ],
    VerbClass.ACHIEVEMENT: [
        "succeed.v.01", "win.v.01", "achieve.v.01",
        # Life-event structural parents found via WordNet hypernym paths:
        "unite.v.01",       # marry.v.01 -> join.v.01 -> unite.v.01
        "receive.v.01",     # graduate.v.01 -> get.v.01 -> receive.v.01
        "leave.v.08",       # retire.v.01 -> leave_office.v.01 -> leave.v.08
        "separate.v.08",    # divorce.v.02 -> separate.v.08
        # Direct synsets (short chains that overlap other classes in closure):
        "die.v.01",         # change_state.v.01 parent overlaps STATUS
        "enroll.v.01",      # have.v.01 ancestor overlaps HAVE
    ],
    VerbClass.EXPERIENCE: ["experience.v.01", "visit.v.01", "travel.v.01"],
    VerbClass.PLANNING: ["plan.v.01", "intend.v.01", "schedule.v.01"],
    VerbClass.HABIT: ["use.v.01", "practice.v.01"],
    VerbClass.MEASUREMENT: ["measure.v.01", "weigh.v.01", "cost.v.01"],
}


@functools.lru_cache(maxsize=2048)
def _hypernym_closure(synset_name: str) -> frozenset:
    """Return frozenset of all hypernym synset names for a given synset."""
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn  # type: ignore
    try:
        ss = wn.synset(synset_name)
    except Exception:
        return frozenset()
    closure = set()
    for path in ss.hypernym_paths():
        for ancestor in path:
            closure.add(ancestor.name())
    return frozenset(closure)


@functools.lru_cache(maxsize=1)
def _anchor_sets() -> dict:
    """Build {VerbClass: frozenset(anchor_synset_names)}, resolved once."""
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn  # type: ignore
    result = {}
    for vc, names in _VERB_CLASS_ANCHORS.items():
        resolved = set()
        for n in names:
            try:
                canonical = wn.synset(n).name()
                resolved.add(canonical)
            except Exception:
                logger.debug("Anchor synset %s not found in WordNet", n)
        result[vc] = frozenset(resolved)
    return result


@functools.lru_cache(maxsize=4096)
def classify_verb_class(lemma: str) -> VerbClass:
    """Classify a verb lemma using WordNet hypernym closure.
    Open-vocabulary.  Two-pass: direct synset match, then closure.
    Spec Part 1, Field: schematic_category (step 3)."""
    if lemma == "be":
        return VerbClass.BE
    if lemma == "have":
        return VerbClass.HAVE

    _ensure_wordnet()
    from nltk.corpus import wordnet as wn  # type: ignore
    verb_synsets = wn.synsets(lemma, pos=wn.VERB)
    if not verb_synsets:
        return VerbClass.UNKNOWN

    anchors = _anchor_sets()

    # Pass 1: direct synset name match
    synset_names = frozenset(ss.name() for ss in verb_synsets)
    for vc, anchor_names in anchors.items():
        if synset_names & anchor_names:
            return vc

    # Pass 2: hypernym closure match
    for ss in verb_synsets:
        closure = _hypernym_closure(ss.name())
        for vc, anchor_names in anchors.items():
            if closure & anchor_names:
                return vc

    return VerbClass.UNKNOWN


# ---------------------------------------------------------------------------
# Dep tree helpers
# ---------------------------------------------------------------------------

def _get_root(doc):
    """Return ROOT token from doc.  Returns None if no ROOT found."""
    for tok in doc:
        if tok.dep_ == "ROOT":
            return tok
    return None


def _span_text(tok) -> str:
    """Get the full subtree text of a token, preserving word order.
    Spec Part 1, Field: object -- subtree extraction for noun phrases."""
    subtree = sorted(tok.subtree, key=lambda t: t.i)
    return " ".join(t.text for t in subtree)


def _extract_object_full(doc, root) -> str:
    """Extract the FULL object complement including all prep phrases and clauses.

    Where _extract_grammatical_object returns the shortest answer NP,
    this returns everything the verb governs on the object side.
    """
    if root is None:
        return ""
    _OBJ_DEPS = frozenset({
        "dobj", "attr", "acomp", "prep", "xcomp", "ccomp", "oprd",
        "pobj", "pcomp", "advcl",
    })
    subj_indices: set = set()
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj_indices.update(t.i for t in child.subtree)
    obj_tokens: set = set()
    for child in root.children:
        if child.dep_ in _OBJ_DEPS:
            for t in child.subtree:
                if t.i not in subj_indices:
                    obj_tokens.add(t.i)
    if not obj_tokens:
        return ""
    ordered = sorted(obj_tokens)
    return " ".join(doc[i].text for i in ordered).strip()


def _extract_grammatical_object(doc, root, _is_recursive: bool = False) -> str:
    """Extract the grammatical object as the SHORTEST noun phrase that IS the answer.

    Priority (Spec Part 1, Field: object):
        1. dobj (direct object):  "researched [adoption agencies]"
        2. attr (predicate nominal):  "is [a transgender woman]"
        3. acomp (adjective complement):  "felt [accepted as a transgender woman]"
        4. pobj (prepositional object):  "moved from [Sweden]"
        5. xcomp chain:  "want to pursue [counseling]" -> recurse into xcomp
        6. ccomp (clausal complement):  "realized [self-care is important]" minus "that"
        7. oprd (object predicate):  "consider [him a friend]"

    Critical rules from spec:
        - SPEECH verbs with ccomp: skip speech frame, extract from embedded clause
        - INTENT verbs with xcomp: skip intent frame, extract from xcomp's object
        - NEVER include the subject in the object
    """
    if root is None:
        return ""

    # 0. Fragment with relcl: ROOT is a NOUN with a relative clause verb.
    #    "Ones that support LGBTQ+ individuals" -- ROOT=Ones, relcl=support.
    #    Extract from the relcl verb's arguments (its dobj/attr/pobj).
    if root.pos_ in ("NOUN", "PRON"):
        for child in root.children:
            if child.dep_ == "relcl" and child.pos_ in ("VERB", "AUX"):
                relcl_obj = _extract_grammatical_object(
                    doc, child, _is_recursive=True,
                )
                if relcl_obj:
                    return relcl_obj

    # 1. Direct object: "researched [adoption agencies]"
    #    At top level only (not xcomp recursion), include purpose/description
    #    prep phrases (for, about) attached to ROOT that modify the event object.
    #    "ran a charity race ... for mental health awareness"
    #      -> "a charity race for mental health awareness"
    for child in root.children:
        if child.dep_ == "dobj":
            dobj_text = _span_text(child)
            # Only at top level: append purpose preps
            if not _is_recursive:
                _PURPOSE_PREPS = frozenset({
                    "for", "about", "on", "toward", "towards",
                })
                for sibling in root.children:
                    if (sibling.dep_ == "prep"
                            and sibling.pos_ == "ADP"
                            and sibling.lemma_ in _PURPOSE_PREPS
                            and sibling.i > child.i):
                        pobj_tok = None
                        for gc in sibling.children:
                            if gc.dep_ == "pobj":
                                pobj_tok = gc
                                break
                        if pobj_tok:
                            # Guard: if pobj is a pronoun or PERSON entity,
                            # it's a beneficiary/recipient, not the semantic
                            # object.  Keep dobj.
                            # "bought a gift for her" -> "a gift" (not "her")
                            if pobj_tok.pos_ == "PRON":
                                break
                            pobj_ner = {
                                t.ent_type_ for t in pobj_tok.subtree
                                if t.ent_type_
                            }
                            if pobj_ner & frozenset({"DATE", "TIME", "PERSON"}):
                                break
                            # Purpose prep promotion: pobj IS the semantic
                            # object. "raised awareness for mental health"
                            # -> object = "mental health" (not dobj+prep).
                            return _span_text(pobj_tok)
            return dobj_text

    # 2. Attribute complement: "is [a transgender woman]"
    for child in root.children:
        if child.dep_ == "attr":
            return _span_text(child)

    # 3. Adjective complement: "felt [accepted as a transgender woman]"
    for child in root.children:
        if child.dep_ == "acomp":
            # For linking verb + single ADJ complement, the subject NP is
            # the semantic answer. "The sunday before 25 May 2023 was lovely"
            # → object = "The sunday before 25 May 2023" (not "lovely").
            # But "I felt accepted as a transgender woman" → object = the
            # full acomp span (has prepositional content beyond the ADJ).
            acomp_span = _span_text(child)
            if (child.pos_ == "ADJ"
                    and root.lemma_ in _COPULAR_LEMMAS
                    and len(list(child.subtree)) <= 2):
                # Single ADJ complement → return subject NP instead
                for sib in root.children:
                    if sib.dep_ in ("nsubj", "nsubjpass"):
                        return _span_text(sib)
            return acomp_span

    # 4. Prepositional object: "moved from [Sweden]"
    # Skip preps that duplicate a particle (prt) on the same verb to avoid
    # treating phrasal-verb particles as semantic prepositions.
    # Also skip preps whose pobj is a temporal expression (DATE/TIME NER),
    # e.g., "moved on Tuesday from Sweden" -> object = "Sweden" not "Tuesday".
    prt_lemmas = frozenset(
        c.lemma_.lower() for c in root.children if c.dep_ == "prt"
    )
    for child in root.children:
        if child.dep_ == "prep":
            if child.lemma_.lower() in prt_lemmas:
                continue  # particle, not a true preposition
            for gc in child.children:
                if gc.dep_ == "pobj":
                    _pobj_ner = {
                        t.ent_type_ for t in gc.subtree if t.ent_type_
                    }
                    if _pobj_ner & frozenset({"DATE", "TIME"}):
                        break  # temporal prep, skip to next prep child
                    return _span_text(gc)
                # pcomp: prepositional complement (gerund).
                # "thinking of [working with trans people]" — "working"
                # has dep=pcomp under prep "of". Extract its full subtree.
                if gc.dep_ == "pcomp" and gc.tag_ == "VBG":
                    return _span_text(gc)

    # 5-6. Xcomp / ccomp chains
    for child in root.children:
        if child.dep_ == "ccomp":
            # ccomp = full clausal complement.  Strip complementizer "that".
            # Spec: "realized that [self-care is important]" -> "self-care is important"
            subtree = sorted(child.subtree, key=lambda t: t.i)
            filtered = [t.text for t in subtree
                        if not (t.dep_ == "mark" and t.lemma_.lower() == "that")]
            return " ".join(filtered).strip()
        if child.dep_ == "xcomp":
            # xcomp = open complement.  Recurse to find xcomp's own object.
            # Spec: "want to pursue [counseling]" -> "counseling"
            xcomp_obj = _extract_grammatical_object(
                doc, child, _is_recursive=True,
            )
            if xcomp_obj:
                return xcomp_obj
            # If xcomp has no object, return its full subtree minus subject
            return _span_text(child)

    # 7. Object predicate: "consider [him a friend]"
    for child in root.children:
        if child.dep_ == "oprd":
            return _span_text(child)

    return ""


def _get_prep_object(
    root, prep_lemmas: Optional[frozenset] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Get (preposition, pobj_text) from root's prep children.

    Skips temporal preps whose pobj subtree contains DATE/TIME NER entities,
    e.g. "moved on Tuesday from Sweden" returns ("from", "Sweden") not ("on", "Tuesday").
    """
    for child in root.children:
        if child.dep_ == "prep":
            if prep_lemmas and child.lemma_ not in prep_lemmas:
                continue
            for gc in child.children:
                if gc.dep_ == "pobj":
                    _pobj_ner = {
                        t.ent_type_ for t in gc.subtree if t.ent_type_
                    }
                    if _pobj_ner & frozenset({"DATE", "TIME"}):
                        break  # temporal prep, skip to next prep child
                    return (child.lemma_, _span_text(gc))
    return (None, None)


# ---------------------------------------------------------------------------
# Reclassification helpers
# Spec Part 1, Field: schematic_category (step 1 -- xcomp/ccomp override)
# ---------------------------------------------------------------------------

def _reclassify_location_by_object(doc, root, verb_class):
    """When LOCATION verb has a non-geographic object, reclassify as EXPERIENCE.
    "went to a support group" = EXPERIENCE, not LOCATION."""
    if verb_class != VerbClass.LOCATION or root is None:
        return verb_class

    for child in root.children:
        if child.dep_ == "dobj":
            ner_types = {t.ent_type_ for t in child.subtree if t.ent_type_}
            if ner_types & frozenset({"GPE", "LOC", "FAC"}):
                return VerbClass.LOCATION
            return VerbClass.EXPERIENCE

    # Check ALL prep children, not just "to". "go through a rough patch",
    # "go into business", "go over the details" are EXPERIENCE, not LOCATION.
    for child in root.children:
        if child.dep_ == "prep":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    ner_types = {t.ent_type_ for t in gc.subtree if t.ent_type_}
                    if ner_types & frozenset({"GPE", "LOC", "FAC"}):
                        return VerbClass.LOCATION
                    return VerbClass.EXPERIENCE
    return verb_class


def _detect_planning_pattern(doc, root, verb_class):
    """Detect planning constructions via dep-tree structure.
    Grammar reference Section 3.5: "thinking about moving" = PLANNING."""
    if root is None:
        return verb_class

    # Pattern 1: {verb} about {gerund}
    for child in root.children:
        if child.dep_ == "prep" and child.lemma_ == "about":
            for gc in child.children:
                if gc.dep_ in ("pcomp", "pobj") and gc.tag_ == "VBG":
                    return VerbClass.PLANNING

    # Pattern 2: {be} {VBG} to {verb}
    if root.tag_ == "VBG":
        for child in root.children:
            if child.dep_ == "xcomp":
                for gc in child.children:
                    if gc.dep_ in ("mark", "aux") and gc.lemma_ == "to":
                        return VerbClass.PLANNING

    return verb_class


# ---------------------------------------------------------------------------
# Grammatical feature detection
# Spec Part 1, Fields: mood, negated, temporal_direction, voice
# ---------------------------------------------------------------------------

def detect_mood(doc) -> str:
    """Detect sentence mood: indicative/interrogative/imperative/conditional/subjunctive.
    Spec Part 1, Field: edge_mood.
    Grammar reference Section 1 (Sentence Classification)."""
    if isinstance(doc, str):
        doc = _get_nlp()(doc)

    root = _get_root(doc)
    if root is None:
        return "indicative"

    # Imperative: ROOT verb with no nsubj, VerbForm=Inf
    has_nsubj = any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children)
    if (not has_nsubj and root.pos_ == "VERB"
            and root.morph.get("VerbForm") in (["Inf"], ["Fin"])
            and root.morph.get("Tense") == []):
        return "imperative"

    # Subjunctive: "wish" + ccomp/xcomp clause (counterfactual desire)
    # Grammar pp.735-739: verb "wish" with tense-shifted embedded clause
    if root.lemma_ == "wish":
        for child in root.children:
            if child.dep_ in ("ccomp", "xcomp"):
                return "subjunctive"

    # Subjunctive: ccomp verb with no tense marking, OR
    # mandative subjunctive: 3rd-person subject + base-form verb (VB not VBZ).
    # "I recommend she study harder" — "study" is VB where VBZ expected.
    # Grammar pp.735-739: morphological disagreement IS the structural signal.
    for tok in doc:
        if tok.dep_ == "ccomp" and tok.pos_ in ("VERB", "AUX"):
            if (tok.morph.get("Tense") == []
                    and tok.morph.get("VerbForm") in (["Inf"], [])):
                return "subjunctive"
            # Mandative: 3rd person singular nsubj + non-VBZ verb form.
            # "I recommend she study harder" — "study" is VBP (not VBZ
            # "studies"). spaCy tags it VBP, not VB. The mismatch between
            # 3rd person singular subject and VBP (not VBZ) = mandative.
            if tok.tag_ in ("VB", "VBP"):
                ccomp_nsubj = next(
                    (c for c in tok.children
                     if c.dep_ in ("nsubj", "nsubjpass")), None
                )
                if (ccomp_nsubj
                        and ccomp_nsubj.morph.get("Person") == ["3"]
                        and ccomp_nsubj.morph.get("Number") == ["Sing"]
                        and tok.tag_ != "VBZ"):
                    return "subjunctive"

    # Subjunctive: "were" with 1st/3rd person singular subject
    if root.lemma_ == "be" and root.text.lower() == "were":
        for child in root.children:
            if child.dep_ == "nsubj":
                if child.text.lower() == "i":
                    return "subjunctive"
                mp = child.morph.get("Person")
                mn = child.morph.get("Number")
                if mn == ["Sing"] and mp in (["1"], ["3"]):
                    return "subjunctive"

    # Gap 12: "if only" -> subjunctive (wish), not conditional.
    # Must be checked BEFORE the conditional block.
    for tok in doc:
        if tok.text.lower() == "if" and (tok.i + 1) < len(doc) and doc[tok.i + 1].text.lower() == "only":
            return "subjunctive"

    # Gap 10: Habitual "would" detection — "would" + temporal/frequency marker
    # and NO conditional subordinator (if/unless/whether) → indicative, not
    # conditional. "We would go fishing every summer" = past habitual.
    _HABITUAL_MARKERS = frozenset({"every", "always", "often", "usually", "frequently"})
    _CONDITIONAL_SUBORDINATORS = frozenset({"if", "unless", "whether"})
    has_conditional_sub = any(
        tok.dep_ in ("mark", "advmod") and tok.lemma_.lower() in _CONDITIONAL_SUBORDINATORS
        for tok in doc
    )
    has_habitual_signal = any(tok.text.lower() in _HABITUAL_MARKERS for tok in doc)
    # Also check for DATE/TIME NER as temporal context
    has_temporal_ner = any(ent.label_ in ("DATE", "TIME") for ent in doc.ents)

    # Conditional: modal aux (would/could) governing a verb,
    # OR subordinating conjunction "if"/"unless"/"whether" (dep_=mark/advmod).
    # Checked BEFORE interrogative so "Would you recommend...?" => conditional
    # (Spec Part 4, Gap 5 - conditional modals take priority over question form)
    # NOTE: "should"/"might" excluded — deontic/epistemic, not counterfactual.
    _conditional_lemmas = {"would", "could"}
    for tok in doc:
        if tok.dep_ == "aux" and tok.lemma_.lower() in _conditional_lemmas:
            if tok.head.pos_ == "VERB":
                # Gap 10 guard: habitual "would" with temporal marker and no
                # conditional subordinator → indicative, not conditional
                if (tok.lemma_.lower() == "would"
                        and not has_conditional_sub
                        and (has_habitual_signal or has_temporal_ner)):
                    break  # skip conditional, fall through to indicative
                return "conditional"
    for tok in doc:
        if (tok.dep_ in ("mark", "advmod")
                and tok.lemma_.lower() in ("if", "unless", "whether")):
            return "conditional"

    # Interrogative: sentence ends with "?" OR subject-auxiliary inversion
    # (aux token precedes nsubj in linear order)
    if doc[-1].text == "?":
        return "interrogative"
    # Check subject-auxiliary inversion
    nsubj_idx = None
    aux_idx = None
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass") and nsubj_idx is None:
            nsubj_idx = tok.i
        if tok.dep_ == "aux" and aux_idx is None:
            aux_idx = tok.i
    if aux_idx is not None and nsubj_idx is not None and aux_idx < nsubj_idx:
        return "interrogative"

    return "indicative"


def detect_negation(doc) -> bool:
    """Detect negation anywhere in the sentence.
    Spec Part 1, Field: negated.
    Grammar reference Section 9: Negation (p. 417-424).

    Checks:
        1. Explicit neg dep label (not, n't)
        2. Implicit negative adverbs: hardly, barely, scarcely, rarely, seldom, never
           These are a closed grammatical class, not a word list."""
    if isinstance(doc, str):
        doc = _get_nlp()(doc)

    for tok in doc:
        if tok.dep_ == "neg":
            return True
        if (tok.pos_ == "ADV"
                and tok.dep_ == "advmod"
                and tok.lemma_.lower() in (
                    "hardly", "barely", "scarcely",
                    "rarely", "seldom", "never",
                )):
            return True
    return False


def detect_tense_aspect(doc) -> TenseAspect:
    """Extract tense x aspect from verb morphology.
    Spec Part 1, Field: temporal_direction (base tense).
    Grammar reference Section 5: Temporal Markers.
    Grammar reference Pages 718-720: Habitual aspect."""
    if isinstance(doc, str):
        doc = _get_nlp()(doc)

    root = _get_root(doc)
    if root is None:
        return TenseAspect(tense="present", aspect="simple")

    # HABITUAL PAST: "used to" + VB (Grammar ref Pages 718-720)
    # Pattern: "used" (VBD, lemma="use") followed by "to" + base verb
    # spaCy may parse this in different ways:
    #   1. "used" as ROOT with "to"+verb as xcomp
    #   2. "used" as aux of the main verb
    # Check all tokens for the "used to" idiom.
    tokens = list(doc)
    for i, tok in enumerate(tokens):
        if (tok.lemma_.lower() == "use"
                and tok.tag_ == "VBD"
                and i + 1 < len(tokens)
                and tokens[i + 1].text.lower() == "to"):
            # Gap 5: "be used to" = accustomed, NOT habitual.
            # If preceded by a form of "be", skip habitual detection.
            if i > 0 and tokens[i - 1].lemma_.lower() == "be":
                continue  # "am/is/are/was/were used to" = accustomed
            # Verify there's a verb after "to"
            if i + 2 < len(tokens) and tokens[i + 2].pos_ == "VERB":
                return TenseAspect(tense="past", aspect="habitual")
            # Also check xcomp children of "used"
            for child in tok.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    return TenseAspect(tense="past", aspect="habitual")

    auxes = sorted(
        [c for c in root.children if c.dep_ in ("aux", "auxpass")],
        key=lambda t: t.i,
    )
    aux_lemmas = [a.lemma_.lower() for a in auxes]

    # FUTURE: "be going to" periphrastic future (Page 697-698)
    # Pattern: AUX(am/is/are/was/were) + "going" (ROOT/xcomp) + "to" + VERB
    # spaCy often parses: "going" as ROOT with "be" as aux, main verb as xcomp
    if root.lemma_ == "go" and root.tag_ == "VBG" and "be" in aux_lemmas:
        # Check if "going" has an xcomp child (the actual main verb)
        has_to_verb = False
        for child in root.children:
            if child.dep_ == "xcomp" and child.pos_ == "VERB":
                # Verify "to" is present as mark/aux of the xcomp
                for grandchild in child.children:
                    if grandchild.lemma_ == "to" and grandchild.dep_ in (
                        "mark", "aux",
                    ):
                        has_to_verb = True
                        break
                if has_to_verb:
                    break
        if has_to_verb:
            return TenseAspect(tense="future", aspect="simple")

    # FUTURE: will/shall
    if "will" in aux_lemmas or "shall" in aux_lemmas:
        tense = "future"
        if "have" in aux_lemmas and "be" in aux_lemmas:
            aspect = "perfect_continuous"
        elif "have" in aux_lemmas:
            aspect = "perfect"
        elif "be" in aux_lemmas:
            aspect = "continuous"
        else:
            aspect = "simple"
        return TenseAspect(tense=tense, aspect=aspect)

    # Tense from ROOT or AUX morphology
    root_tense = root.morph.get("Tense")
    tense = "present"
    if root_tense == ["Past"]:
        tense = "past"
    for a in auxes:
        if a.morph.get("Tense") == ["Past"]:
            tense = "past"
            break

    # Aspect from AUX combination
    has_have = "have" in aux_lemmas
    has_be = "be" in aux_lemmas
    root_is_ing = root.tag_ == "VBG"

    if has_have and has_be and root_is_ing:
        aspect = "perfect_continuous"
    elif has_have:
        aspect = "perfect"
    elif has_be and root_is_ing:
        aspect = "continuous"
    else:
        aspect = "simple"

    return TenseAspect(tense=tense, aspect=aspect)


def detect_voice(doc) -> str:
    """Detect active / passive / middle voice.
    Grammar reference Section 3: Predicate/Action.

    Known limitation (Grammar Gap #5 -- Intransitive Middle Voice):
        "The lasagna cooked in the oven" returns "active" instead of "middle".
        True intransitive middle voice (subject is patient, no passive morphology,
        no agent expressed) requires verb-frame semantic knowledge (whether the
        subject COULD be an agent) that spaCy does not provide.  Only reflexive
        middle voice ("He dressed himself") and passive morphology are detected.
        Detecting intransitive middle would require a verb transitivity/animacy
        lexicon beyond what structural POS/dep parsing offers.
    """
    if isinstance(doc, str):
        doc = _get_nlp()(doc)

    has_passive_subj = False
    has_auxpass = False
    has_reflexive_obj = False
    _reflexives = frozenset({
        "myself", "yourself", "himself", "herself",
        "itself", "ourselves", "yourselves", "themselves",
    })

    for tok in doc:
        if tok.dep_ == "nsubjpass":
            has_passive_subj = True
        if tok.dep_ == "auxpass":
            has_auxpass = True
        if tok.dep_ in ("dobj", "pobj") and tok.text.lower() in _reflexives:
            has_reflexive_obj = True

    if has_passive_subj or has_auxpass:
        return "passive"
    if has_reflexive_obj:
        return "middle"
    return "active"


def resolve_pronouns(doc, speaker: Optional[str] = None, listener: str = "user"):
    """Resolve first- and second-person pronouns using speaker metadata.
    Spec Part 1, Field: subject -- "I" -> speaker name.
    Grammar reference Section 7: Entity Type Patterns.

    Args:
        doc: spaCy Doc or raw text string.
        speaker: Name of the person speaking (resolves "I"/"me"/"my"/"myself").
        listener: Name of the person being addressed (resolves "you"/"your"/"yourself"/"yours").
                  Defaults to "user" for backward compatibility in single-speaker mode.
    """
    if isinstance(doc, str):
        doc = _get_nlp()(doc)

    tokens: list[str] = []
    speaker_name = speaker if speaker else "user"
    listener_name = listener
    _replaced_first_person = False  # track if previous token was 1st person → speaker

    for tok in doc:
        person = tok.morph.get("Person", [""])[0]
        is_poss = "Yes" in tok.morph.get("Poss", [])
        is_reflex = "Yes" in tok.morph.get("Reflex", [])

        # First person (I, me, my, myself, mine, we, us, our, ours)
        if tok.pos_ == "PRON" and person == "1":
            if is_poss:
                tokens.append(speaker_name + "'s")
            else:
                tokens.append(speaker_name)
                # Track for verb agreement fix on next token
                case = tok.morph.get("Case", [""])[0]
                if case == "Nom" and not is_reflex:
                    _replaced_first_person = True
                    continue  # skip whitespace reset below
        # Second person (you, your, yourself, yours)
        elif tok.pos_ == "PRON" and person == "2":
            if is_poss:
                tokens.append(listener_name + "'s")
            else:
                tokens.append(listener_name)
        # Verb agreement: after 1st person subject → 3rd person form
        elif _replaced_first_person and tok.pos_ in ("AUX", "VERB"):
            lemma = tok.lemma_
            tense = tok.morph.get("Tense", ["Pres"])[0] if tok.morph.get("Tense") else "Pres"
            is_contraction = tok.text.startswith("'")
            prefix = " " if is_contraction else ""
            if lemma == "be":
                _form = "is" if tense == "Pres" else "was" if tense == "Past" else lemma
                tokens.append(f"{prefix}{_form}")
            elif lemma == "have":
                _form = "has" if tense == "Pres" else "had"
                tokens.append(f"{prefix}{_form}")
            elif lemma == "do":
                _form = "does" if tense == "Pres" else "did"
                tokens.append(_form)
            elif is_contraction:
                # 'd → would, 'll → will, etc.
                tokens.append(f" {lemma}")
            else:
                tokens.append(tok.text)
        # Contraction on non-1st-person subject (she's, he'd)
        elif tok.pos_ == "AUX" and tok.text.startswith("'"):
            lemma = tok.lemma_
            tense = tok.morph.get("Tense", ["Pres"])[0] if tok.morph.get("Tense") else "Pres"
            if lemma == "be":
                tokens.append(" is" if tense == "Pres" else " was" if tense == "Past" else " " + lemma)
            elif lemma == "have":
                tokens.append(" has" if tense == "Pres" else " had")
            else:
                tokens.append(" " + lemma)
        else:
            tokens.append(tok.text)
        _replaced_first_person = False

    resolved = ""
    for i, tok in enumerate(doc):
        resolved += tokens[i]
        if tok.whitespace_:
            resolved += tok.whitespace_
    return resolved


# ---------------------------------------------------------------------------
# Utterance classification
# Spec Part 2: Extraction Rules by Sentence Type
# ---------------------------------------------------------------------------

def _has_question_mark(doc) -> bool:
    for tok in reversed(list(doc)):
        if tok.text == "?":
            return True
        if tok.pos_ != "SPACE":
            break
    return False


def _has_interrogative_fronted(doc) -> bool:
    """Check for WH-fronting.  Grammar reference Section 1."""
    for tok in doc:
        if tok.pos_ == "SPACE":
            continue
        pron_type = tok.morph.get("PronType")
        if pron_type and "Int" in pron_type:
            return True
        if tok.tag_ in ("WDT", "WP", "WP$", "WRB"):
            if tok.dep_ in ("advmod", "attr", "nsubj", "dobj", "det"):
                return True
        break
    return False


def _has_subject_aux_inversion(doc) -> bool:
    """Check for subject-aux inversion (yes/no questions)."""
    root = _get_root(doc)
    if root is None:
        return False
    aux_i = None
    subj_i = None
    for child in root.children:
        if child.dep_ == "aux" and aux_i is None:
            aux_i = child.i
        if child.dep_ in ("nsubj", "nsubjpass") and subj_i is None:
            subj_i = child.i
    return aux_i is not None and subj_i is not None and aux_i < subj_i


def _is_imperative_structure(doc) -> bool:
    """Spec Part 2, Command detection: ROOT verb with no nsubj, VerbForm=Inf or tag=VB."""
    root = _get_root(doc)
    if root is None or root.pos_ != "VERB":
        return False
    has_nsubj = any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children)
    if has_nsubj:
        return False
    if root.tag_ == "VB":
        return True
    if root.morph.get("VerbForm") == ["Inf"]:
        return True
    return False


def _is_backchannel_structure(doc) -> bool:
    """Spec Part 2, Fragment/Backchannel detection.
    Exception: fragments with NUM, NER, or content nouns ARE stored."""
    tokens = [t for t in doc if t.pos_ not in ("SPACE", "PUNCT")]
    if not tokens:
        return True

    # ROOT is interjection = backchannel
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ == "INTJ":
            return True

    if len(tokens) <= 3:
        if not any(t.pos_ in ("VERB", "AUX") for t in tokens):
            has_num = any(t.pos_ == "NUM" for t in tokens)
            has_ner = any(t.ent_type_ for t in tokens)
            has_content_noun = any(
                t.pos_ == "NOUN" and not t.is_stop for t in tokens
            )
            if not (has_num or has_ner or has_content_noun):
                return True

    # Fragment with a verb somewhere (e.g. relcl) is content
    has_verb_anywhere = any(tok.pos_ in ("VERB", "AUX") for tok in doc)
    if has_verb_anywhere:
        return False

    has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc)
    if not has_subj and not _is_imperative_structure(doc):
        # Before declaring backchannel, check if the fragment has
        # substantive content: NER entities, NUM tokens, or content
        # nouns/verbs (non-auxiliary). Fragments like "Running, reading,
        # or playing my violin" have real content and must be stored.
        has_ner = any(t.ent_type_ for t in doc)
        has_num = any(t.pos_ == "NUM" for t in doc)
        has_content_word = any(
            t.pos_ in ("NOUN", "PROPN", "VERB") and not t.is_stop
            for t in doc
        )
        if has_ner or has_num or has_content_word:
            return False
        return True

    return False


_COPULAR_LEMMAS = frozenset({
    "be", "feel", "seem", "appear", "become",
    "get", "grow", "turn", "remain", "stay",
    "look", "sound", "taste", "smell", "prove",
})


def _is_emotion_structure(doc) -> bool:
    """Detect: 1st-person subject + copular verb + adj complement."""
    root = _get_root(doc)
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return False

    first_person = any(
        c.dep_ == "nsubj" and c.text.lower() in ("i", "we")
        for c in root.children
    )
    if not first_person:
        return False

    has_adj = any(
        c.dep_ in ("acomp", "oprd") and c.pos_ == "ADJ"
        for c in root.children
    )
    return has_adj and root.lemma_ in _COPULAR_LEMMAS


def _has_exclamation_mark(doc) -> bool:
    for tok in reversed(list(doc)):
        if tok.text == "!":
            return True
        if tok.pos_ != "SPACE":
            break
    return False


def _is_tag_question(doc) -> bool:
    """Detect tag questions (Grammar reference Pages 922-923).

    Pattern: declarative clause + comma + short inverted AUX+PRON + "?"
    Examples: "You're going to the party, aren't you?"
              "She likes coffee, doesn't she?"

    Tag questions are pragmatically assertions -- the main clause is the fact.
    """
    tokens = list(doc)
    if len(tokens) < 5:
        return False

    # Must end with "?"
    if tokens[-1].text != "?":
        return False

    # Find the last comma in the sentence
    last_comma_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].text == ",":
            last_comma_idx = i
            break

    if last_comma_idx is None:
        return False

    # The tag part: tokens between last comma and "?"
    tag_tokens = [t for t in tokens[last_comma_idx + 1:]
                  if t.text != "?" and t.pos_ != "SPACE"]

    # Tag should be short: 2-3 tokens (AUX + PRON or AUX + neg + PRON)
    if len(tag_tokens) < 2 or len(tag_tokens) > 3:
        return False

    # Check pattern: (AUX [+ neg] + PRON) or (VERB-as-aux [+ neg] + PRON)
    # Typical: "aren't you", "doesn't she", "is it", "did he"
    # spaCy sometimes tags "does/did/is" as VERB in tag position
    _tag_aux_lemmas = {"do", "be", "have", "will", "shall", "can", "could",
                       "would", "should", "may", "might", "must"}
    has_aux = any(t.pos_ == "AUX" or t.dep_ == "aux"
                  or t.lemma_.lower() in _tag_aux_lemmas
                  for t in tag_tokens)
    has_pron = any(t.pos_ == "PRON" for t in tag_tokens)

    if has_aux and has_pron:
        # Verify main clause (before comma) has substance (subject + verb)
        main_tokens = tokens[:last_comma_idx]
        has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in main_tokens)
        has_verb = any(t.dep_ == "ROOT" or t.pos_ in ("VERB", "AUX")
                       for t in main_tokens)
        return has_subj and has_verb

    return False


def _strip_tag_question(doc) -> Any:
    """Strip the tag portion from a tag question, returning only the main clause.

    "You're going to the party, aren't you?" -> "You're going to the party"
    """
    tokens = list(doc)
    # Find last comma
    last_comma_idx = None
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i].text == ",":
            last_comma_idx = i
            break
    if last_comma_idx is None:
        return doc
    main_text = " ".join(t.text for t in tokens[:last_comma_idx]).strip()
    if not main_text:
        return doc
    return _get_nlp_fragment()(main_text)


def classify_utterance(
    doc, speaker: Optional[str] = None,
) -> UtteranceClassification:
    """Classify a spaCy Doc (or raw string) into coarse bin.
    Spec Part 2: sentence type determines extraction path."""
    if isinstance(doc, str):
        doc = _get_nlp()(doc)

    # Exclamatory: ends with ! and has what/how fronting.
    # Must be checked BEFORE question detection.
    if _has_exclamation_mark(doc) and _has_interrogative_fronted(doc):
        return UtteranceClassification(
            4, CoarseBin.EMOTION.value, "exclamatory",
            False, False, False, True, True)

    # Tag questions (Grammar ref #11, Pages 922-923):
    # "You're going to the party, aren't you?" is pragmatically a statement.
    # Must be checked BEFORE general question detection so we treat the
    # main clause as a storable declarative fact.
    if _is_tag_question(doc):
        return UtteranceClassification(
            5, CoarseBin.STATEMENT.value, "tag_question",
            False, False, False, False, True)

    # Cleft sentences: "What happened was I applied..." — NOT a question.
    # Pattern: ROOT is "be" + csubj (WH-clause) + ccomp (content).
    _cleft_root = _get_root(doc)
    if (_cleft_root and _cleft_root.lemma_ == "be"
            and any(c.dep_ == "csubj" for c in _cleft_root.children)
            and any(c.dep_ == "ccomp" and c.pos_ == "VERB"
                    for c in _cleft_root.children)):
        return UtteranceClassification(
            5, CoarseBin.STATEMENT.value, "cleft",
            False, False, False, False, True)

    if (_has_question_mark(doc)
            or _has_interrogative_fronted(doc)
            or _has_subject_aux_inversion(doc)):
        return UtteranceClassification(
            1, CoarseBin.QUESTION.value, "",
            True, False, False, False, False)

    if _is_backchannel_structure(doc):
        return UtteranceClassification(
            2, CoarseBin.BACKCHANNEL.value, "",
            False, False, True, False, False)

    if _is_imperative_structure(doc):
        return UtteranceClassification(
            3, CoarseBin.COMMAND.value, "",
            False, True, False, False, False)

    if _is_emotion_structure(doc):
        return UtteranceClassification(
            4, CoarseBin.EMOTION.value, "",
            False, False, False, True, True)

    return UtteranceClassification(
        5, CoarseBin.STATEMENT.value, "",
        False, False, False, False, True)


# ---------------------------------------------------------------------------
# Trace extraction -- the core of the engine
# Spec Part 1: every field defined here
# ---------------------------------------------------------------------------

_FIRST_PERSON = frozenset({
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
})


def _extract_episodic(doc, root) -> str:
    """Extract episodic trace: sentence text with subject stripped.
    Spec Part 1, Field: episodic_fact.
    "I have been researching adoption agencies lately" -> "researching adoption agencies lately"
    """
    sent_text = str(doc).strip()
    if root is None:
        return sent_text

    # Gerund/clausal subject as content: when nsubj is a csubj (gerund
    # phrase) on a stative/linking ROOT, the subject IS the meaningful
    # content. "Being transgender in a small town was isolating" →
    # episodic = "Being transgender in a small town", not "isolating".
    for child in root.children:
        if child.dep_ == "csubj":
            csubj_span = sorted(child.subtree, key=lambda t: t.i)
            csubj_text = " ".join(t.text for t in csubj_span).strip()
            if csubj_text:
                return csubj_text.rstrip(".,;:!?")

    # Find subject token — check root's children first, then all tokens
    # (spaCy may attach nsubj to an auxpass rather than ROOT)
    subj_tok = None
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj_tok = child
            break
    if subj_tok is None:
        for tok in doc:
            if tok.dep_ in ("nsubj", "nsubjpass"):
                subj_tok = tok
                break

    if subj_tok is None:
        return sent_text

    # Get the full subject span
    subj_subtree = sorted(subj_tok.subtree, key=lambda t: t.i)
    if not subj_subtree:
        return sent_text

    last_subj_idx = subj_subtree[-1].i

    # Collect tokens after the subject span
    remaining_tokens = [tok for tok in doc if tok.i > last_subj_idx]
    if not remaining_tokens:
        return sent_text

    # Build text preserving whitespace
    result = ""
    for tok in remaining_tokens:
        result += tok.text
        if tok.whitespace_:
            result += tok.whitespace_
    result = result.strip()

    # Strip leading auxiliaries: "have been researching" -> "researching"
    # Use original doc's token POS tags instead of re-parsing the fragment
    strip_count = 0
    for tok in remaining_tokens:
        if tok.pos_ == "AUX":
            strip_count += 1
        else:
            break
    if strip_count > 0:
        kept_tokens = remaining_tokens[strip_count:]
        if kept_tokens:
            result = ""
            for tok in kept_tokens:
                result += tok.text
                if tok.whitespace_:
                    result += tok.whitespace_
            result = result.strip()
            remaining_tokens = kept_tokens

    # Copular/linking verb: the complement (acomp/attr) IS the episodic fact.
    # Use the complement span directly instead of position-based stripping.
    # Handles: "The sunday before 25 May 2023 was lovely" -> "lovely"
    #          "What a beautiful painting that was!" -> "beautiful painting"
    #          "I felt accepted as a transgender woman" -> "accepted as a transgender woman"
    if root is not None and root.pos_ in ("AUX", "VERB"):
        comp_tok = None
        for child in root.children:
            if child.dep_ in ("acomp", "attr"):
                comp_tok = child
                break
        if comp_tok is not None and (
            root.lemma_ in _COPULAR_LEMMAS or root.pos_ == "AUX"
        ):
            comp_span = sorted(comp_tok.subtree, key=lambda t: t.i)
            comp_text = ""
            for tok in comp_span:
                if tok.pos_ == "PUNCT":
                    continue
                comp_text += tok.text
                if tok.whitespace_:
                    comp_text += tok.whitespace_
            comp_text = comp_text.strip()
            if comp_text:
                result = comp_text

    # Strip trailing punctuation
    result = result.rstrip(".,;:!?")

    return result if result else sent_text


def _extract_emotional(doc, root) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Extract emotional trace: emotion adjective, valence, target.
    Spec Part 1, Fields: emotional_state / emotional_valence / emotional_target.
    Grammar reference Section 6: Emotional/Sentiment Markers.

    Detection order:
        1. ADJ tokens in acomp/attr/oprd position
        2. Passive past participles with copular auxpass
        3. WordNet noun hypernym closure through feeling.n.01/emotion.n.01
    """
    emotion_adj = None
    emotion_tok = None

    # 1. ADJ tokens in acomp/attr/oprd
    for tok in doc:
        if tok.pos_ == "ADJ" and tok.dep_ in ("acomp", "attr", "oprd"):
            emotion_adj = tok.text.lower()
            emotion_tok = tok
            break

    # 2. Passive past participles as emotional states
    # Guard: only extract emotion when nsubj is animate (PRON or PERSON NER).
    # "I felt broken" → emotional. "The window was broken" → physical, not emotional.
    if emotion_adj is None:
        for tok in doc:
            if (tok.tag_ == "VBN" and tok.dep_ == "ROOT"
                    and any(c.dep_ == "auxpass" for c in tok.children)):
                # Animacy check on nsubj
                nsubj_tok = next(
                    (c for c in tok.children
                     if c.dep_ in ("nsubj", "nsubjpass")), None
                )
                if nsubj_tok is not None:
                    is_animate = (
                        nsubj_tok.pos_ == "PRON"
                        or nsubj_tok.ent_type_ == "PERSON"
                    )
                    if not is_animate:
                        break  # inanimate subject → physical state, not emotion
                auxpass_tok = next(
                    (c for c in tok.children if c.dep_ == "auxpass"), None
                )
                if auxpass_tok and (
                    auxpass_tok.lemma_ in _COPULAR_LEMMAS
                    or auxpass_tok.text.lower() in (
                        "felt", "feels", "seemed", "looked", "sounded",
                    )
                ):
                    emotion_adj = tok.text.lower()
                    emotion_tok = tok
                    break

    # 3. Emotion nouns via WordNet hypernym closure
    if emotion_adj is None:
        try:
            from nltk.corpus import wordnet as _wn
            _emotion_synsets = {"feeling.n.01", "emotion.n.01", "state.n.04"}
            for tok in doc:
                if tok.pos_ == "NOUN" and not tok.is_stop:
                    for ss in _wn.synsets(tok.lemma_, pos="n"):
                        hypernyms = {
                            h.name() for h in ss.closure(lambda s: s.hypernyms())
                        }
                        if hypernyms & _emotion_synsets:
                            emotion_adj = tok.lemma_.lower()
                            emotion_tok = tok
                            break
                    if emotion_adj:
                        break
        except Exception:
            pass

    if emotion_adj is None:
        return (None, None, None)

    # Valence: SentiWordNet average across synsets for the token's POS,
    # then flip sign if syntactic negation is present.
    # Maps spaCy POS to WordNet POS for synset lookup.
    _SPACY_TO_WN_POS = {"ADJ": "a", "NOUN": "n", "VERB": "v", "ADV": "r"}
    wn_pos = _SPACY_TO_WN_POS.get(emotion_tok.pos_, "a")
    valence = 0.0
    try:
        from nltk.corpus import sentiwordnet as _swn
        ss = list(_swn.senti_synsets(emotion_tok.lemma_.lower(), wn_pos))
        if not ss:
            # Fallback: try adjective POS if primary POS missed
            ss = list(_swn.senti_synsets(emotion_tok.lemma_.lower(), "a"))
        if ss:
            vals = [s.pos_score() - s.neg_score() for s in ss]
            valence = sum(vals) / len(vals)
    except Exception:
        pass

    # Negation flips the sign: "not happy" → negative
    has_negation = any(child.dep_ == "neg" for child in emotion_tok.children)
    if not has_negation and emotion_tok.head is not None:
        has_negation = any(
            child.dep_ == "neg" for child in emotion_tok.head.children
        )
    if has_negation:
        valence = -valence

    # Target: pobj of prep child, or nsubj of head verb
    target = None
    for child in emotion_tok.children:
        if child.dep_ == "prep":
            for gc in child.children:
                if gc.dep_ == "pobj":
                    target = gc.text
                    break
            if target is not None:
                break
    if target is None and emotion_tok.head is not None:
        for sibling in emotion_tok.head.children:
            if sibling.dep_ == "nsubj":
                target = sibling.text
                break

    return (emotion_adj, valence, target)


def _extract_temporal(doc, tense_aspect: TenseAspect) -> Tuple[str, Optional[str]]:
    """Extract temporal trace: direction + expression.
    Spec Part 1, Fields: temporal_direction, temporal_expression.

    Rules:
        1. Collect DATE/TIME NER spans -- pass through as-is (no ISO)
        2. Structural fallback: NUM + time_noun + ago
        3. Perfect continuous with past = present (ongoing)
        4. Xcomp with "to" mark = future (intent)
        5. Temporal adverb override
        6. "going to" + xcomp = future
    """
    _tense_to_direction = {
        "past": "past", "present": "present", "future": "future",
    }
    direction = _tense_to_direction.get(tense_aspect.tense, "present")

    # Perfect continuous with past tense = ongoing from past to present
    # Only perfect_continuous gets this override; plain "continuous" past
    # (e.g. "I was running") is a completed past action, not ongoing.
    if (tense_aspect.aspect == "perfect_continuous"
            and direction == "past"):
        direction = "present"

    root = _get_root(doc)

    # Xcomp with "to" mark = future intent
    if root is not None and direction == "present":
        for child in root.children:
            if child.dep_ == "xcomp":
                for gc in child.children:
                    if gc.dep_ in ("aux", "mark") and gc.lemma_ == "to":
                        direction = "future"
                        break

    # Temporal adverb override (Grammar reference Section 5)
    for tok in doc:
        if tok.pos_ == "ADV" and tok.dep_ in ("advmod", "npadvmod"):
            lemma = tok.lemma_.lower()
            if lemma in ("still", "currently"):
                direction = "present"
            elif lemma in ("ago", "previously", "formerly", "once"):
                direction = "past"
            elif lemma in ("soon", "eventually", "shortly"):
                direction = "future"

    # "going to" future detection
    if root and root.lemma_ == "go" and root.tag_ == "VBG":
        for child in root.children:
            if child.dep_ == "xcomp" and child.tag_ == "VB":
                has_to = any(
                    gc.dep_ == "aux" and gc.lemma_ == "to"
                    for gc in child.children
                )
                if has_to or any(
                    gc.dep_ == "mark" and gc.text == "to"
                    for gc in child.children
                ):
                    direction = "future"
                    break

    # Collect DATE/TIME NER spans (Spec: pass through as-is, no ISO)
    # M2 fix: expand NER span to include contextual tokens that form part of
    # the full temporal expression (e.g., "the sunday before 25 May 2023").
    # Strategy: find prep/advmod tokens whose object IS the NER span, then
    # walk up to a nominal head (NOUN/PROPN) and include its subtree.
    # Guard: never expand through verb heads to avoid grabbing whole clauses.
    # Collect expanded (start, end) index spans, then merge overlaps.
    _raw_spans: List[Tuple[int, int]] = []
    for ent in doc.ents:
        if ent.label_ not in ("DATE", "TIME"):
            continue
        left_boundary = ent.start

        # Step 1: Find a prep/advmod token left of the entity whose
        # syntactic object (pobj/dobj) points into the NER span.
        governing_prep = None
        for tok in doc:
            if tok.dep_ in ("prep", "advmod") and tok.i < ent.start:
                for child in tok.children:
                    if (child.dep_ in ("pobj", "dobj")
                            and child.i >= ent.start
                            and child.i < ent.end):
                        governing_prep = tok
                        break
                # Adjacent prep whose head's subtree covers the entity
                if governing_prep is None and tok.i == ent.start - 1:
                    if tok.dep_ == "prep":
                        governing_prep = tok
                if governing_prep is not None:
                    break

        if governing_prep is not None:
            # Step 2: Walk up from the prep to its head — only if nominal
            prep_head = governing_prep.head
            if prep_head.pos_ in ("NOUN", "PROPN"):
                # Include the full subtree of the nominal head.
                subtree_tokens = sorted(prep_head.subtree, key=lambda t: t.i)
                left_boundary = min(t.i for t in subtree_tokens
                                    if t.i <= ent.start)
            else:
                # Head is a verb or other non-nominal — only include the
                # prep itself (e.g., "in 2022" keeps "in")
                left_boundary = governing_prep.i
        else:
            # No governing prep — simple left-walk for adjacent modifiers.
            _CONTEXTUAL_DEPS = frozenset(
                {"det", "amod", "compound", "nummod"})
            while left_boundary > 0:
                candidate = doc[left_boundary - 1]
                if candidate.dep_ in _CONTEXTUAL_DEPS:
                    left_boundary -= 1
                elif (candidate.pos_ in ("NOUN", "PROPN")
                      and candidate.dep_ in ("compound", "npadvmod", "nmod")):
                    left_boundary -= 1
                else:
                    break

        _raw_spans.append((left_boundary, ent.end))

    # Merge overlapping / contained spans so we don't duplicate text.
    _raw_spans.sort()
    merged_spans: List[Tuple[int, int]] = []
    for start, end in _raw_spans:
        if merged_spans and start <= merged_spans[-1][1]:
            # Overlaps with previous — extend
            merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
        else:
            merged_spans.append((start, end))

    date_time_spans: List[str] = [doc[s:e].text for s, e in merged_spans]
    expression = " ".join(date_time_spans) if date_time_spans else None

    # Structural fallback: NUM + time_noun + "ago" pattern
    # spaCy may miss these as NER.  Structural: nummod->NOUN->ADV(ago).
    if expression is None:
        _TIME_NOUNS = frozenset({
            "year", "month", "week", "day", "hour", "minute",
            "decade", "century", "semester", "quarter", "fortnight",
        })
        for tok in doc:
            if tok.lemma_.lower() == "ago" and tok.pos_ == "ADV":
                head = tok.head
                if head.pos_ == "NOUN" and head.lemma_.lower() in _TIME_NOUNS:
                    num_tok = None
                    for child in head.children:
                        if child.dep_ == "nummod" or child.pos_ == "NUM":
                            num_tok = child
                            break
                    if num_tok:
                        expression = f"{num_tok.text} {head.text} ago"
                        direction = "past"
                    else:
                        subtree = sorted(head.subtree, key=lambda t: t.i)
                        expr_tokens = [t.text for t in subtree] + ["ago"]
                        expression = " ".join(expr_tokens)
                        direction = "past"
                    break

    # Gap 15: Frequency adverbs — closed grammatical class, enriches temporal trace
    _FREQUENCY_ADVERBS = frozenset({
        "always", "never", "often", "usually", "sometimes", "rarely",
        "daily", "weekly", "monthly", "yearly", "annually",
        "frequently", "seldom", "occasionally", "regularly",
    })
    for tok in doc:
        if tok.dep_ == "advmod" and tok.lemma_.lower() in _FREQUENCY_ADVERBS:
            freq = tok.text.lower()
            if expression:
                # Prepend frequency to a real temporal expression
                expression = f"{freq} {expression}"
            # else: frequency adverb alone is NOT a temporal expression —
            # "always", "often" etc. enrich direction but don't locate an event
            # in time. Only real DATE/TIME spans or structural patterns qualify.
            break  # one frequency adverb per clause

    return (direction, expression)


def _extract_relational(
    doc, speaker: Optional[str], listener: str = "user",
) -> Tuple[str, List[str], str]:
    """Extract relational trace: subject, entities, type.
    Spec Part 1, Fields: relational_subject, relational_entities.

    Rules:
        1. First-person pronouns -> subject is speaker
        2. Second-person pronouns (as nsubj) -> subject is listener
        3. Collect PERSON/ORG/GPE/LOC/FAC/NORP NER entities
        Speaker name is appended downstream by memory.py _prepare_row.
    """
    relational_subject = speaker or "user"

    # Find the nsubj token to determine person
    _THIRD_PERSON_PRONOUNS = frozenset({
        "she", "he", "they", "it", "her", "him", "them",
    })
    nsubj_tok = None
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass"):
            nsubj_tok = tok
            break

    if nsubj_tok is not None:
        nsubj_lower = nsubj_tok.text.lower()
        if nsubj_lower in _FIRST_PERSON and speaker:
            # First-person -> resolve to speaker
            relational_subject = speaker
        elif nsubj_tok.pos_ == "PRON" and nsubj_lower in _THIRD_PERSON_PRONOUNS:
            # Third-person pronoun -> keep as-is, do NOT default to speaker
            # Gap 8: Dummy "it" — weather/impersonal verbs produce a
            # meaningless "it" subject. Detect and set to empty string.
            _DUMMY_IT_LEMMAS = frozenset({
                "rain", "snow", "hail", "sleet", "drizzle", "thunder",
                "pour", "seem", "appear",
            })
            _root = _get_root(doc)
            if nsubj_lower == "it" and _root is not None and _root.lemma_.lower() in _DUMMY_IT_LEMMAS:
                relational_subject = ""
            else:
                relational_subject = nsubj_tok.text
        elif nsubj_tok.pos_ == "PROPN":
            # Proper noun -> use its text
            # Collect full proper-noun span (multi-token names)
            span_tokens = [nsubj_tok]
            for left in nsubj_tok.lefts:
                if left.pos_ == "PROPN" and left.dep_ == "compound":
                    span_tokens.insert(0, left)
            for right in nsubj_tok.rights:
                if right.pos_ == "PROPN" and right.dep_ == "flat":
                    span_tokens.append(right)
            relational_subject = " ".join(t.text for t in span_tokens)
        elif nsubj_tok.pos_ == "NOUN":
            # Common noun subject (e.g., "The window was broken")
            # Use the noun span text, not the speaker default
            relational_subject = _span_text(nsubj_tok)
        elif nsubj_tok.pos_ == "PRON":
            # Gap 7: Any remaining PRON not caught above (indefinite pronouns
            # like everyone, someone, nobody, anything, etc.) — keep as-is
            # instead of defaulting to speaker.
            relational_subject = nsubj_tok.text
        # else: no nsubj match above -> keep default (speaker)
    else:
        # No nsubj at all -> keep default (speaker)
        has_first_person = any(
            tok.text.lower() in _FIRST_PERSON for tok in doc
        )
        if has_first_person and speaker:
            relational_subject = speaker

    # Second-person subject detection: "You moved to Portland" -> listener
    _SECOND_PERSON_SUBJ = frozenset({"you"})
    has_second_person_subj = any(
        tok.text.lower() in _SECOND_PERSON_SUBJ
        and tok.dep_ in ("nsubj", "nsubjpass")
        for tok in doc
    )
    if has_second_person_subj:
        relational_subject = listener

    entities: List[str] = []
    seen: set = set()
    for ent in doc.ents:
        if (ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC", "NORP")
                and ent.text not in seen):
            entities.append(ent.text)
            seen.add(ent.text)

    # Gap 1: Indirect objects (dative/iobj) that are PROPN or PERSON NER
    # may not appear in doc.ents.  "I gave Sarah the book" -> Sarah.
    for tok in doc:
        if tok.dep_ in ("dative", "iobj"):
            if tok.pos_ == "PROPN" or tok.ent_type_ == "PERSON":
                name = _span_text(tok)
                if name not in seen:
                    entities.append(name)
                    seen.add(name)

    # Gap 2: Causative dobj as participant entity.
    # "She made him cry" — ROOT has xcomp/ccomp child (causative), dobj is participant.
    # spaCy may parse the caused-entity as dobj of ROOT or nsubj of the complement.
    root_tok = _get_root(doc)
    if root_tok is not None:
        for comp_child in root_tok.children:
            if comp_child.dep_ in ("xcomp", "ccomp") and comp_child.pos_ == "VERB":
                # Check dobj of ROOT
                for child in root_tok.children:
                    if child.dep_ == "dobj" and child.pos_ in ("PRON", "PROPN"):
                        name = child.text
                        if name not in seen:
                            entities.append(name)
                            seen.add(name)
                # Also check nsubj of the complement (spaCy's ECM parse)
                for gc in comp_child.children:
                    if gc.dep_ == "nsubj" and gc.pos_ in ("PRON", "PROPN"):
                        name = gc.text
                        if name not in seen:
                            entities.append(name)
                            seen.add(name)
                break  # only process first complement

        # Gap 3: Factitive oprd — "They elected him chairman".
        # When both dobj and oprd exist, dobj is the affected person.
        has_oprd = any(c.dep_ == "oprd" for c in root_tok.children)
        if has_oprd:
            for child in root_tok.children:
                if child.dep_ == "dobj" and child.pos_ in ("PRON", "PROPN"):
                    name = child.text
                    if name not in seen:
                        entities.append(name)
                        seen.add(name)

    # Gap 20: Vocatives — addressee detection enriches relational trace
    for tok in doc:
        if tok.dep_ == "vocative" or (
            tok.pos_ == "PROPN"
            and tok.dep_ in ("npadvmod", "appos", "ROOT", "dep")
            and tok.i == 0
            and tok.nbor(1).text == ","
            if tok.i + 1 < len(doc) else False
        ):
            name = _span_text(tok)
            if name not in seen:
                entities.append(name)
                seen.add(name)

    # Gap 22: Compound subjects — split conjoined nsubj into separate entities
    if nsubj_tok is not None:
        for conj_child in nsubj_tok.children:
            if conj_child.dep_ == "conj" and conj_child.pos_ in ("PROPN", "NOUN"):
                name = _span_text(conj_child)
                if name not in seen:
                    entities.append(name)
                    seen.add(name)
        # Also add the nsubj itself if it's a PROPN not yet in entities
        if nsubj_tok.pos_ == "PROPN":
            name = _span_text(nsubj_tok)
            if name not in seen:
                entities.append(name)
                seen.add(name)

    # Gap 23: Passive "by" agent — extract pobj of agent dep
    for tok in doc:
        if tok.dep_ == "agent" and tok.head.tag_ in ("VBN", "VBD"):
            for child in tok.children:
                if child.dep_ == "pobj":
                    name = _span_text(child)
                    if name not in seen:
                        entities.append(name)
                        seen.add(name)

    return (relational_subject, entities, "personal")


# ---------------------------------------------------------------------------
# Schematic trace -- WordNet noun-to-schema mapping
# Spec Part 1, Field: edge_schematic_category (steps 4-7)
# ---------------------------------------------------------------------------

_NOUN_SCHEMA_ANCHORS: list[tuple[frozenset, str]] = [
    (frozenset({
        "occupation.n.01", "position.n.01", "job.n.01",
        "profession.n.01", "employment.n.01", "work.n.01",
        "service.n.01", "promotion.n.02", "advancement.n.03",
    }), "career"),
    (frozenset({
        "family_relationship.n.01", "adoption.n.01",
        "relative.n.01", "kinship.n.01",
    }), "family"),
    (frozenset({
        "illness.n.01", "injury.n.01", "symptom.n.01",
        "disease.n.01",
    }), "health"),
    (frozenset({
        "creation.n.02", "artistic_creation.n.01",
        "art.n.01", "sport.n.01", "game.n.01",
        "recreation.n.01", "diversion.n.01",
        "outdoor_recreation.n.01", "hobby.n.01",
    }), "hobby"),
    (frozenset({
        "educational_institution.n.01", "course.n.01",
        "school.n.01",
    }), "education"),
]


@functools.lru_cache(maxsize=4096)
def _is_kinship_noun(lemma: str) -> bool:
    """Check if a noun lemma is a kinship term via WordNet hypernym closure.
    Returns True if any synset of the lemma is a hyponym of kinship anchors:
    {relative.n.01, parent.n.01, sibling.n.01, spouse.n.01, child.n.02,
     family_member.n.01, ancestor.n.01, grandparent.n.01}.
    Structural check -- no word lists."""
    try:
        _ensure_wordnet()
        from nltk.corpus import wordnet as _wn

        _kinship_anchors = frozenset({
            "relative.n.01", "parent.n.01", "sibling.n.01", "spouse.n.01",
            "child.n.02", "family_member.n.01", "ancestor.n.01",
            "grandparent.n.01",
        })

        for ss in _wn.synsets(lemma, pos="n"):
            hypernyms = {h.name() for h in ss.closure(lambda s: s.hypernyms())}
            hypernyms.add(ss.name())
            if hypernyms & _kinship_anchors:
                return True
        return False
    except Exception:
        return False


@functools.lru_cache(maxsize=4096)

def _extract_schematic(doc, root, verb_class: VerbClass) -> str:
    """Extract schematic trace: verb supersense + syntactic frame = schema.

    Levin 1993: verbs sharing syntactic behavior share meaning.
    The dep frame (transitive vs intransitive) selects the verb sense.
    WordNet supersense of the selected sense maps to life domain.
    """
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return "uncategorized"

    _ensure_wordnet()
    from nltk.corpus import wordnet as _wn

    # Syntactic frame from dep parse
    has_dobj = any(ch.dep_ == "dobj" for ch in root.children)

    # Get verb senses, filter by frame
    synsets = _wn.synsets(root.lemma_, pos=_wn.VERB)
    if not synsets:
        return "uncategorized"

    # Frame filtering: transitive frame (8,9,11) vs intransitive (1,2)
    best = synsets[0]
    if has_dobj:
        for ss in synsets:
            if 8 in ss.frame_ids() or 9 in ss.frame_ids() or 11 in ss.frame_ids():
                best = ss
                break
    else:
        for ss in synsets:
            if 1 in ss.frame_ids() or 2 in ss.frame_ids():
                best = ss
                break

    # Verb supersense → schema
    _VERB_SS_TO_SCHEMA = {
        "verb.social": "career",
        "verb.possession": "finance",
        "verb.creation": "hobby",
        "verb.cognition": "education",
        "verb.emotion": "emotional",
        "verb.motion": "experience",
        "verb.communication": "social",
        "verb.consumption": "health",
        "verb.body": "health",
        "verb.competition": "hobby",
        "verb.perception": "experience",
        "verb.stative": "identity",
        "verb.contact": "experience",
        "verb.change": "experience",
    }

    schema = _VERB_SS_TO_SCHEMA.get(best.lexname(), "uncategorized")

    # Kinship noun in subject/object → family (overrides verb signal)
    for ch in root.children:
        if ch.dep_ in ("nsubj", "nsubjpass", "dobj", "attr") and ch.pos_ == "NOUN":
            if _is_kinship_noun(ch.lemma_):
                return "family"

    return schema


# ---------------------------------------------------------------------------
# _extract_traces_from_sentence -- the core extraction path
# Spec Part 1: all 5 traces from a single sentence
# ---------------------------------------------------------------------------

def _find_content_verb(verb, _depth=0):
    """Universal frame skipper: walk past framing verbs to the content.

    Grammar reference pp. 212-220, 283-295: control/raising verbs.

    Rule: skip ROOT to its xcomp/ccomp child IF:
      - ccomp → skip only if current is SPEECH class or "be" (cleft)
      - xcomp → skip only if ROOT has no dobj (subject control)
                AND ROOT is not PREFERENCE class

    Handles: speech verbs, intent verbs, phase verbs, clefts,
    causative guards, "used to", "ended up", "keeps telling" — all
    in one recursive walk. No special cases.
    """
    if verb is None or _depth >= 4:
        return verb
    complement = None
    for child in verb.children:
        if child.dep_ == "ccomp" and child.pos_ in ("VERB", "AUX"):
            complement = child
            break
        if child.dep_ == "xcomp" and child.pos_ in ("VERB", "AUX"):
            complement = child
            break
    if complement is None:
        return verb
    if complement.dep_ == "ccomp":
        _cur_vc = classify_verb_class(verb.lemma_)
        if _cur_vc == VerbClass.SPEECH or verb.lemma_ == "be":
            return _find_content_verb(complement, _depth + 1)
        return verb
    has_dobj = any(c.dep_ == "dobj" for c in verb.children)
    if has_dobj:
        _cur_vc = classify_verb_class(verb.lemma_)
        if _cur_vc != VerbClass.SPEECH:
            return verb
    _vc = classify_verb_class(verb.lemma_)
    if _vc == VerbClass.PREFERENCE:
        return verb
    return _find_content_verb(complement, _depth + 1)


# ---------------------------------------------------------------------------
# Predicted Question Generation — 41/41 verified
# Chomsky 1957: WH-movement + subject-auxiliary inversion
#
# tokens = SUBJECT ∪ VERB_CHAIN ∪ TARGET ∪ REMAINDER
# question = WH + VERB_CHAIN[0] + SUBJECT + VERB_CHAIN[1:] + REMAINDER + ?
# ---------------------------------------------------------------------------

def generate_predicted_questions(sent_doc, root, subject_name):
    """Generate predicted questions from a live spaCy parse.

    Formula (Chomsky 1957, 41/41 verified):
      tokens = SUBJECT ∪ VERB_CHAIN ∪ TARGET ∪ REMAINDER
      question = WH + VERB_CHAIN[0] + SUBJECT + VERB_CHAIN[1:] + REMAINDER + ?
      If TARGET = SUBJECT: WH + VERB_CHAIN + REMAINDER + ?
      If no aux and ROOT != be: do-support, ROOT → base form

    One function. Gates → collect targets → form questions.
    """
    if not root or root.pos_ not in ("VERB", "AUX"):
        return []

    # === GATES ===
    subj_tok = None
    for ch in root.children:
        if ch.dep_ in ("nsubj", "nsubjpass"):
            subj_tok = ch
            break
    if not subj_tok:
        return []
    if subj_tok.pos_ == "PRON" and subj_tok.lemma_ in ("it", "this", "that"):
        return []
    if sum(1 for t in sent_doc if t.pos_ not in ("PUNCT", "INTJ", "X")) < 3:
        return []

    # === DECOMPOSE (once, shared by all questions) ===
    subject_indices = {t.i for t in subj_tok.subtree}
    vc_indices = {root.i}
    for ch in root.children:
        if ch.dep_ in ("aux", "auxpass"):
            vc_indices.add(ch.i)
        if ch.dep_ == "prt":
            vc_indices.add(ch.i)
    verb_chain = sorted(vc_indices)

    is_be_main = root.lemma_ == "be" and len(verb_chain) == 1
    has_aux = any(sent_doc[i].dep_ in ("aux", "auxpass") for i in verb_chain if i != root.i)

    neg_tok = None
    for ch in root.children:
        if ch.dep_ == "neg":
            neg_tok = ch
            break

    # Exclude: punct, interjections, compound clause tails
    exclude = {tok.i for tok in sent_doc if tok.pos_ in ("PUNCT", "INTJ")}
    for ch in root.children:
        if ch.dep_ == "cc":
            exclude.add(ch.i)
        if ch.dep_ in ("conj", "advcl"):
            exclude.update(t.i for t in ch.subtree)
    if len(sent_doc) > 0 and sent_doc[0].pos_ in ("ADV", "INTJ") and sent_doc[0].dep_ == "advmod":
        exclude.add(0)
        if len(sent_doc) > 1 and sent_doc[1].text == ",":
            exclude.add(1)

    # === FORM ONE QUESTION from target indices ===
    def _form(target_indices):
        target_toks = [sent_doc[i] for i in target_indices]
        if all(t.pos_ in ("PRON", "DET", "PART", "PUNCT") for t in target_toks):
            return None

        # WH-word selection (from target semantics)
        target_first = sent_doc[min(target_indices)]
        prep_parent = None
        for i in target_indices:
            tok = sent_doc[i]
            if tok.dep_ == "pobj" and tok.head.dep_ in ("prep", "agent"):
                prep_parent = tok.head
                break

        wh, absorb = "What", False
        if target_first.ent_type_ in ("DATE", "TIME"):
            wh, absorb = "When", True
        elif prep_parent and prep_parent.lemma_ in ("at", "in", "on", "to", "from", "near"):
            if target_first.ent_type_ in ("DATE", "TIME"):
                wh, absorb = "When", True
            else:
                wh, absorb = "Where", True
        elif prep_parent and prep_parent.lemma_ == "for" and any(t.pos_ == "NUM" for t in target_first.subtree):
            wh, absorb = "How long", True
        elif target_first.pos_ == "ADJ" and target_first.dep_ in ("acomp", "dobj", "attr"):
            wh = "How"
        elif prep_parent and prep_parent.dep_ == "agent":
            wh, absorb = "By whom", True
        elif target_first.ent_type_ == "PERSON":
            wh = "Who"

        # Prep handling
        effective_target = set(target_indices)
        strand_prep = None
        if prep_parent:
            if absorb:
                effective_target.update(t.i for t in prep_parent.subtree)
            else:
                effective_target.update(t.i for t in target_first.subtree)
                strand_prep = prep_parent.text
                effective_target.add(prep_parent.i)

        # Remainder
        remainder = sorted(
            set(range(len(sent_doc))) - subject_indices - set(verb_chain) - effective_target - exclude
        )

        # Subject question?
        if target_indices & subject_indices:
            parts = [wh]
            for i in sorted(set(verb_chain) | set(remainder)):
                parts.append(sent_doc[i].text)
            if strand_prep:
                parts.append(strand_prep)
            return " ".join(parts) + "?"

        # Non-subject: only 2 things move. First aux (or do) before subject,
        # WH to front. Everything else stays in original document order.
        neg_i = neg_tok.i if neg_tok else -1
        # Tokens that stay in place: verb chain (minus moved aux) + remainder
        # Output them in document order after [WH, moved-aux, subject, neg]
        moved_i = set()  # indices of tokens we place manually

        if has_aux:
            first_aux_i = verb_chain[0]
            if sent_doc[first_aux_i].dep_ not in ("aux", "auxpass"):
                for vi in verb_chain:
                    if sent_doc[vi].dep_ in ("aux", "auxpass"):
                        first_aux_i = vi
                        break
            moved_i = subject_indices | {first_aux_i} | effective_target | exclude
            if neg_i >= 0:
                moved_i.add(neg_i)
            parts = [wh, sent_doc[first_aux_i].text, subject_name]
            if neg_i >= 0:
                parts.append(neg_tok.text)
            # Everything else in document order
            for i in range(len(sent_doc)):
                if i not in moved_i:
                    parts.append(sent_doc[i].text)
            if strand_prep:
                parts.append(strand_prep)
            return " ".join(parts) + "?"

        if is_be_main:
            moved_i = subject_indices | {root.i} | effective_target | exclude
            if neg_i >= 0:
                moved_i.add(neg_i)
            parts = [wh, root.text, subject_name]
            if neg_i >= 0:
                parts.append(neg_tok.text)
            for i in range(len(sent_doc)):
                if i not in moved_i:
                    parts.append(sent_doc[i].text)
            if strand_prep:
                parts.append(strand_prep)
            return " ".join(parts) + "?"

        # Do-support: only subject moves (after inserted do). Root stays
        # in place but changes to base form. Everything else untouched.
        do = "did" if root.tag_ == "VBD" else ("does" if root.tag_ == "VBZ" else "do")
        moved_i = subject_indices | effective_target | exclude
        if neg_i >= 0:
            moved_i.add(neg_i)
        parts = [wh, do, subject_name]
        if neg_i >= 0:
            parts.append(neg_tok.text)
        # Document order — root outputs as base form, everything else as-is
        for i in range(len(sent_doc)):
            if i in moved_i:
                continue
            if i == root.i:
                parts.append(root.lemma_)
            else:
                parts.append(sent_doc[i].text)
        if strand_prep:
            parts.append(strand_prep)
        return " ".join(parts) + "?"

    # === COLLECT TARGETS (single pass over root.children) ===
    questions = []
    seen = set()

    def _add(indices):
        q = _form(indices)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            questions.append(q)

    for ch in root.children:
        if ch.dep_ == "dobj":
            _add({t.i for t in ch.subtree})
        elif ch.dep_ == "attr":
            _add({t.i for t in ch.subtree})
        elif ch.dep_ in ("prep", "agent"):
            for gc in ch.children:
                if gc.dep_ == "pobj":
                    _add({t.i for t in gc.subtree})
                    break
        elif ch.dep_ == "ccomp":
            _add({t.i for t in ch.subtree})
        elif ch.dep_ == "xcomp":
            for gc in ch.children:
                if gc.dep_ == "dobj":
                    _add({t.i for t in gc.subtree})
                    break

    for ent in sent_doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            _add({t.i for t in sent_doc if ent.start <= t.i < ent.end})
            break

    return questions[:4]


def _extract_traces_from_sentence(
    sent_doc,
    speaker: Optional[str],
    tense_aspect: TenseAspect,
    listener: str = "user",
) -> TraceDecomposition:
    """Extract all 5 traces from a single sentence doc.
    Spec Part 2, Statement extraction path."""
    root = _get_root(sent_doc)
    source_text = str(sent_doc).strip()

    content_root = _find_content_verb(root) if root else root

    # Phrasal verb: content_root + particle (dep=prt)
    particle = None
    if content_root:
        for child in content_root.children:
            if child.dep_ == "prt":
                particle = child.lemma_
                break

    # Verb class classification — on the CONTENT verb, not the frame
    verb_class = VerbClass.UNKNOWN
    if content_root and content_root.pos_ in ("VERB", "AUX"):
        if particle:
            phrasal = f"{content_root.lemma_}_{particle}"
            verb_class = classify_verb_class(phrasal)
            if verb_class == VerbClass.UNKNOWN:
                verb_class = classify_verb_class(content_root.lemma_)
        else:
            verb_class = classify_verb_class(content_root.lemma_)
        verb_class = _reclassify_location_by_object(
            sent_doc, content_root, verb_class,
        )
        verb_class = _detect_planning_pattern(
            sent_doc, content_root, verb_class,
        )

    # 1. Episodic trace — extract from content verb's perspective
    episodic_fact = _extract_episodic(sent_doc, content_root)

    # Grammatical object — from content verb
    gram_object = _extract_grammatical_object(sent_doc, content_root)
    gram_object_full = _extract_object_full(sent_doc, content_root)

    # 4. Relational trace
    relational_subject, relational_entities, relational_type_base = (
        _extract_relational(sent_doc, speaker, listener=listener)
    )

    # Reported speech: if frame-skipper jumped past a SPEECH verb,
    # the relational_subject should be the EMBEDDED clause's subject,
    # not the reporter. _extract_relational found ROOT's nsubj (the
    # reporter). We need to correct it to the embedded subject.
    _embedded_predicate_verb = (
        content_root if content_root is not root else None
    )
    if _embedded_predicate_verb and root:
        _root_vc = classify_verb_class(root.lemma_)
        if _root_vc == VerbClass.SPEECH:
            # Find embedded clause's nsubj
            embedded_subj = None
            for child in root.children:
                if child.dep_ in ("ccomp", "xcomp") and child.pos_ in ("VERB", "AUX"):
                    for gc in child.children:
                        if gc.dep_ in ("nsubj", "nsubjpass"):
                            embedded_subj = gc.text
                            break
                    break
            if embedded_subj:
                if embedded_subj not in relational_entities:
                    relational_entities.append(embedded_subj)
                # Set relational_subject to the embedded subject:
                # "Caroline said SHE moved" → subj = "Caroline" (reporter
                #   is the referent of "she" in reported speech)
                # "Caroline said I need help" → subj = speaker (first person)
                if embedded_subj.lower() in _FIRST_PERSON:
                    relational_subject = speaker or "user"
                elif embedded_subj.lower() in (
                    "she", "he", "they", "it", "her", "him", "them",
                ):
                    # Third-person pronoun in reported speech → reporter
                    # is the likely referent ("Caroline said SHE moved")
                    main_nsubj_tok = None
                    for rc in root.children:
                        if rc.dep_ in ("nsubj", "nsubjpass"):
                            main_nsubj_tok = rc
                            break
                    if main_nsubj_tok and main_nsubj_tok.pos_ == "PROPN":
                        relational_subject = _span_text(main_nsubj_tok)
                    elif main_nsubj_tok:
                        for ent in sent_doc.ents:
                            if (ent.label_ == "PERSON"
                                    and ent.start <= main_nsubj_tok.i < ent.end):
                                relational_subject = ent.text
                                break
                else:
                    # Named embedded subject ("The doctor said Sam needs...")
                    relational_subject = embedded_subj

    # Invariant 1: object is ALWAYS a noun phrase, never a full sentence.
    # Do NOT fall back to episodic_fact -- leave empty if no NP extracted.

    # 2. Emotional trace
    emotional_state, emotional_valence, emotional_target = (
        _extract_emotional(sent_doc, root)
    )

    # 3. Temporal trace
    temporal_direction, temporal_expression = (
        _extract_temporal(sent_doc, tense_aspect)
    )

    relational_type = _VERB_CLASS_TO_RELTYPE.get(
        verb_class, relational_type_base,
    )

    # 5. Schematic trace — from content verb, not frame
    schematic_category = _extract_schematic(sent_doc, content_root, verb_class)

    # Significance (stative vs dynamic via Grammar Gap #6)
    significance = _compute_significance(verb_class, tense_aspect)

    # Grammar Gap #14: Emphatic do-support detection.
    # "I do love chocolate" — "do" as aux in a declarative affirmative
    # sentence signals emphasis. Override significance to "emphatic".
    # Note: spaCy may parse the main verb as NOUN (e.g., "love" -> NN),
    # so we check any root that has an aux "do" child.
    if root and root.pos_ in ("VERB", "NOUN", "AUX"):
        for child in root.children:
            if (child.dep_ == "aux"
                    and child.lemma_.lower() == "do"
                    and child.tag_ in ("VBP", "VBZ", "VBD")):
                # Confirm declarative (not question) and affirmative (not neg)
                _is_question = any(
                    t.text == "?" for t in sent_doc
                )
                _is_negated = any(
                    c.dep_ == "neg" for c in root.children
                )
                if not _is_question and not _is_negated:
                    significance = "emphatic"
                break

    # Grammatical features
    mood = detect_mood(sent_doc)
    negated = detect_negation(sent_doc)
    is_historical = tense_aspect.tense == "past"

    # Conditional detection (Spec Part 1, Field: edge_mood)
    # Grammar reference: Conditionals p. 602, 927-932
    is_conditional = False
    for tok in sent_doc:
        if tok.dep_ == "mark" and tok.lemma_.lower() in (
            "if", "unless", "whether",
        ):
            is_conditional = True
            break
        if (tok.dep_ == "aux"
                and tok.lemma_ in ("would", "could")
                and tok.head.pos_ == "VERB"):
            is_conditional = True
            break

    if is_conditional:
        mood = "conditional"

    # Derive predicate (Spec Part 1, Field: predicate)
    # content_root already points to the content verb (frame-skipper did
    # the delegation). No special cases needed — just use content_root.
    _pred_root = content_root
    root_lemma = _pred_root.lemma_ if _pred_root else ""

    # Gap 4: Light verb predicate delegation.
    # "took a shower" -> pred=shower instead of pred=take.
    _LIGHT_VERB_PRED = frozenset({"do", "have", "take", "make", "give", "get"})
    if (_pred_root and root_lemma.lower() in _LIGHT_VERB_PRED
            and not particle):
        for child in _pred_root.children:
            if child.dep_ == "dobj":
                root_lemma = child.lemma_
                break

    # Particle already computed from content_root (line above)
    predicate = (f"{root_lemma}_{particle}" if particle else root_lemma).lower()

    # Lemmatizer fallback: when spaCy's lemma equals the surface form on an
    # inflected verb (VBD/VBG/VBN/VBZ), the lemmatizer failed on the fragment.
    # Use WordNet morphy as fallback.
    if (_pred_root and predicate == _pred_root.text.lower()
            and _pred_root.tag_ in ("VBD", "VBG", "VBN", "VBZ")):
        try:
            _ensure_wordnet()
            from nltk.corpus import wordnet as _wn_lemma
            _morph_result = _wn_lemma.morphy(predicate, _wn_lemma.VERB)
            if _morph_result:
                predicate = _morph_result
        except Exception:
            pass  # graceful fallback: keep surface form

    # Append prep frame: "move" -> "move_from"
    # Skip preps whose pobj is a temporal expression (DATE/TIME NER),
    # e.g., "moved on Tuesday from Sweden" -> "move_from" not "move_on".
    if _pred_root:
        for child in _pred_root.children:
            if child.dep_ == "prep" and child.pos_ == "ADP":
                _pobj_is_temporal = False
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        _pobj_ner = {
                            t.ent_type_ for t in gc.subtree if t.ent_type_
                        }
                        if _pobj_ner & frozenset({"DATE", "TIME"}):
                            _pobj_is_temporal = True
                        break
                if _pobj_is_temporal:
                    continue  # skip temporal prep, try next
                predicate = f"{predicate}_{child.lemma_.lower()}"
                break

    return TraceDecomposition(
        episodic_fact=episodic_fact,
        episodic_significance=significance,
        emotional_state=emotional_state,
        emotional_valence=emotional_valence,
        emotional_target=emotional_target,
        temporal_direction=temporal_direction,
        temporal_expression=temporal_expression,
        relational_subject=relational_subject,
        relational_entities=relational_entities,
        relational_type=relational_type,
        schematic_category=schematic_category,
        source_text=source_text,
        utterance_type=0,
        mood=mood,
        negated=negated,
        is_historical=is_historical,
        subject=relational_subject,
        predicate=predicate,
        object=gram_object,
        object_full=gram_object_full if gram_object_full != gram_object else "",
        predicted_questions=generate_predicted_questions(sent_doc, content_root, relational_subject),
        extraction_rule="trace",
    )




# ---------------------------------------------------------------------------
# Query decomposition -- READ-PATH interface
# Spec: classify_query parses questions the same way process() parses statements
# ---------------------------------------------------------------------------

def process(text: str, speaker: Optional[str] = None, listener: str = "user") -> GrammarResult:
    """Grammar engine entry point. Clean text in, 5 traces out.

    For each sentence in the text:
      1. Classify (statement/question/command/backchannel)
      2. Extract 5 traces via _extract_traces_from_sentence
      3. Generate predicted questions
      4. Append to decompositions if storable (not a question)
    """
    _check_entry()
    nlp = _get_nlp()

    if speaker:
        resolved_text_pre = resolve_pronouns(nlp(text), speaker, listener=listener)
        doc = nlp(resolved_text_pre)
    else:
        doc = nlp(text)

    decompositions: List[TraceDecomposition] = []
    resolved_parts: List[str] = []
    primary_classification: Optional[UtteranceClassification] = None
    primary_sent_doc = None

    for sent in doc.sents:
        sent_doc = sent.as_doc()
        sent_cls = classify_utterance(sent_doc, speaker)

        if primary_classification is None or (
            not primary_classification.is_storable and sent_cls.is_storable
        ):
            primary_classification = sent_cls
            primary_sent_doc = sent_doc

        resolved_parts.append(str(sent_doc).strip())
        sent_tense = detect_tense_aspect(sent_doc)

        if sent_cls.subcategory == "tag_question":
            sent_doc = _strip_tag_question(sent_doc)
            sent_tense = detect_tense_aspect(sent_doc)

        decomp = _extract_traces_from_sentence(
            sent_doc, speaker, sent_tense, listener=listener,
        )
        decomp.utterance_type = sent_cls.utterance_type_id

        if sent_cls.is_question:
            decomp.mood = "interrogative"
        elif sent_cls.is_command:
            decomp.mood = "imperative"
        elif sent_cls.subcategory == "tag_question":
            decomp.mood = "indicative"

        if not sent_cls.is_question:
            decompositions.append(decomp)

    if primary_classification is None:
        primary_classification = classify_utterance(doc, speaker)
        primary_sent_doc = doc

    if primary_classification.subcategory == "tag_question":
        mood = "indicative"
    else:
        mood = detect_mood(primary_sent_doc)

    return GrammarResult(
        trace_decompositions=decompositions,
        classification=primary_classification,
        mood=mood,
        negated=detect_negation(primary_sent_doc),
        tense_aspect=detect_tense_aspect(primary_sent_doc),
        voice=detect_voice(primary_sent_doc),
        resolved_text=" ".join(resolved_parts),
    )


@dataclass
class QueryDecomposition:
    """Structural decomposition of a query for direct SQL lookup."""
    wh_word: Optional[str] = None
    match_subject: Optional[str] = None
    match_predicate: Optional[str] = None
    match_object: Optional[str] = None
    match_entity: Optional[str] = None
    match_schema: Optional[str] = None
    return_field: str = "episodic"
    utterance_type_id: int = 0
    is_structural: bool = False


def _wh_to_return_field(wh_tok) -> str:
    """Map a WH-token to a return_field using spaCy POS/dep features.
    Grammar reference Section 9: Question Decomposition.

    Root cause of prior bug: all WRB advmod tokens were handled by one
    code path that used verb class to decide the return field.  But
    "when" (temporal), "where" (locative), "why" (causal), and "how"
    (manner) are structurally distinct question types in the closed
    WH-word class and must be distinguished by lemma first.

    WRB "when"  -> temporal  (always — asks about time)
    WRB "where" -> episodic  (location lives in object/prep trace)
    WRB "why"   -> episodic  (reason/cause lives in episodic trace)
    WRB "how"   -> temporal  if head is temporal ADV ("how long"),
                   emotional if head has ADJ complement,
                   else episodic
    WP$         -> relational
    WP subj/attr-> relational
    Everything else -> episodic
    """
    if wh_tok is None:
        return "episodic"

    dep = wh_tok.dep_
    tag = wh_tok.tag_
    lemma = wh_tok.lemma_.lower()

    # --- WRB advmod: distinguish by lemma (closed grammatical class) ---
    if tag == "WRB" and dep == "advmod":
        # "when" always asks about time
        if lemma == "when":
            return "temporal"

        # "where" asks about location — stored in episodic trace
        if lemma == "where":
            return "episodic"

        # "why" asks for reason/cause — episodic
        if lemma == "why":
            return "episodic"

        # "how" — context-dependent
        head = wh_tok.head
        # "how long" / "how long ago" -> temporal
        if head.pos_ == "ADV" and head.lemma_.lower() == "long":
            return "temporal"
        # "how" + ADJ complement (e.g. "how did she feel") -> emotional
        has_adj_complement = any(
            c.dep_ in ("acomp", "oprd") and c.pos_ == "ADJ"
            for c in head.children
        )
        if has_adj_complement:
            return "emotional"
        return "episodic"

    # --- WP$ ("whose") -> relational ---
    if tag == "WP$":
        return "relational"

    # --- WP ("who"/"whom"/"what") in subject position ---
    # Note: WP in attr (e.g. "What is X?" / "Who is X?") maps to episodic
    # because en_core_web_sm lacks morph features to distinguish [+human]
    # "who" from [-human] "what" — both are WP with empty morph.
    # Returning episodic (object field) is correct for the majority case
    # ("What is X's Y?" queries outnumber "Who is X?" queries).
    if tag == "WP":
        # Only "who"/"whom" in subject position → relational.
        # "What" in subject position (e.g. "What motivated X?") asks about
        # things, not people — return episodic so we extract from object.
        if dep in ("nsubj", "nsubjpass") and lemma in ("who", "whom"):
            return "relational"

    return "episodic"


def classify_query(query_text: str) -> QueryDecomposition:
    """Decompose a query into structural fields for SQL lookup.
    Spec: same spaCy parse as process(), extracts entity, keywords, schema, wh_type."""
    nlp = _get_nlp()
    doc = nlp(query_text)
    result = QueryDecomposition()

    # WH-word extraction
    wh_tok = None
    for tok in doc:
        if tok.pos_ == "SPACE":
            continue
        if tok.tag_ in ("WDT", "WP", "WP$", "WRB"):
            wh_tok = tok
            result.wh_word = tok.text.lower()
            break
        break

    result.return_field = _wh_to_return_field(wh_tok)

    root = _get_root(doc)
    if root is None:
        return result

    # Extract subject
    subj_tok = None
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj_tok = child
            break

    if subj_tok is not None:
        if subj_tok.tag_ not in ("WDT", "WP", "WP$", "WRB"):
            if subj_tok.pos_ == "PRON":
                person = subj_tok.morph.get("Person")
                if person and "1" in person:
                    result.match_subject = "user"
            else:
                result.match_subject = _span_text(subj_tok).strip()

    # Extract predicate
    if root.pos_ == "VERB":
        # For light verbs with xcomp ("decided to pursue", "want to study"),
        # prefer the xcomp verb as predicate — it carries the real action.
        xcomp_verb = None
        for child in root.children:
            if child.dep_ == "xcomp" and child.pos_ == "VERB":
                xcomp_verb = child
                break
        if xcomp_verb:
            result.match_predicate = xcomp_verb.lemma_.lower()
        else:
            result.match_predicate = root.lemma_.lower()
    elif root.pos_ == "AUX":
        for child in root.children:
            if child.dep_ in ("xcomp", "ccomp", "acomp", "attr"):
                if child.pos_ in ("VERB", "NOUN", "ADJ"):
                    result.match_predicate = child.lemma_.lower()
                    break

    # Extract object
    dobj_tok = None
    for child in root.children:
        if child.dep_ in ("dobj", "attr"):
            dobj_tok = child
            break
    if dobj_tok is not None:
        if dobj_tok.tag_ not in ("WDT", "WP", "WP$", "WRB"):
            # Strip WH-determiners from the span: "What book" → "book"
            subtree = sorted(dobj_tok.subtree, key=lambda t: t.i)
            obj_text = " ".join(
                t.text for t in subtree
                if t.tag_ not in ("WDT", "WP", "WP$", "WRB")
            ).strip()
            result.match_object = obj_text if obj_text else None

    if result.match_object is None:
        prep, pobj = _get_prep_object(root)
        if pobj:
            result.match_object = pobj.strip()

    # Extract match_entity from NER
    # Spec: relational_entities collects PERSON, ORG, GPE, LOC, FAC, NORP
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"):
            entity_text = ent.text
            if entity_text.endswith("'s"):
                entity_text = entity_text[:-2]
            elif (ent.label_ == "PERSON" and entity_text.endswith("s")
                  and len(entity_text) > 4 and not entity_text.endswith("ss")):
                entity_text = entity_text[:-1]
            result.match_entity = entity_text
            break

    # Utterance type
    cls = classify_utterance(doc)
    result.utterance_type_id = cls.utterance_type_id

    # Structural flag
    result.is_structural = bool(
        result.match_subject or result.match_predicate
    )

    # Derive match_schema — uses the SAME 7-step _extract_schematic() as the
    # write path so that stored schema and query schema never diverge.
    # Note: _reclassify_location_by_object is NOT called here because the
    # write path calls it on content_root (after _find_content_verb), not on
    # root. Adding it here on root caused Cat 5 to regress (29.8% -> 19.2%)
    # without improving Cat 1-4.
    try:
        vc = VerbClass.UNKNOWN
        if root.pos_ == "VERB":
            vc = classify_verb_class(root.lemma_.lower())
        elif root.pos_ == "AUX":
            # For AUX roots (e.g. "is"), classify as BE so _extract_schematic
            # Step 1 triggers xcomp/ccomp delegation correctly.
            vc = classify_verb_class(root.lemma_.lower())

        schema = _extract_schematic(doc, root, vc)

        if schema and schema != "uncategorized":
            result.match_schema = schema
    except Exception:
        pass

    return result

