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
    TraceDecomposition, GrammarResult, Triple, UtteranceClassification
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


def check_exit(result) -> bool:
    """Validate GrammarResult has valid trace decompositions."""
    if result is None:
        return False
    decomps = getattr(result, 'trace_decompositions', None)
    if not decomps:
        return True  # no decomps is valid (backchannel etc.)
    for td in decomps:
        # Every trace decomposition must have the 5 trace fields
        for field_name in ('episodic_fact', 'emotional_valence',
                           'temporal_direction', 'relational_subject',
                           'schematic_category'):
            if not hasattr(td, field_name):
                logger.error("grammar exit check: TraceDecomposition missing %s", field_name)
                return False
    return True


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


@dataclass(frozen=True)
class Triple:
    """Extracted (subject, predicate, object) -- DERIVED from traces.
    Exists for backward compatibility only."""
    subject: str
    predicate: str
    object: str
    is_historical: bool
    utterance_type: int
    negated: bool
    mood: str  # indicative | subjunctive | imperative | conditional
    extraction_rule: str = ""


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
    extraction_rule: str = ""


@dataclass
class GrammarResult:
    """Aggregate output of process().  One per input turn.
    trace_decompositions is PRIMARY; triples is DERIVED."""
    trace_decompositions: List[TraceDecomposition]
    triples: List[Triple]
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

    for tok in doc:
        lower = tok.text.lower()
        if lower == "i" and tok.dep_ in ("nsubj", "nsubjpass", "ROOT"):
            tokens.append(speaker_name)
        elif lower == "me" and tok.dep_ in ("dobj", "pobj", "dative"):
            tokens.append(speaker_name)
        elif lower == "my":
            tokens.append(speaker_name + "'s")
        elif lower == "myself":
            tokens.append(speaker_name)
        elif lower == "you":
            tokens.append(listener_name)
        elif lower == "your":
            tokens.append(listener_name + "'s")
        elif lower == "yourself":
            tokens.append(listener_name)
        elif lower == "yours":
            tokens.append(listener_name + "'s")
        # Gap 6: Standalone possessive pronouns
        elif lower == "mine":
            tokens.append(speaker_name + "'s")
        elif lower == "ours":
            tokens.append(speaker_name + "'s")
        # Contraction conjugation: after "I" -> speaker (3rd person),
        # AUX needs 3rd-person form.
        elif tok.pos_ == "AUX" and tok.text.startswith("'"):
            lemma = tok.lemma_
            morph = tok.morph
            if lemma == "be":
                t = morph.get("Tense", ["Pres"])[0] if morph.get("Tense") else "Pres"
                tokens.append(" is" if t == "Pres" else " was" if t == "Past" else " " + lemma)
            elif lemma == "have":
                t = morph.get("Tense", ["Pres"])[0] if morph.get("Tense") else "Pres"
                tokens.append(" has" if t == "Pres" else " had")
            else:
                tokens.append(" " + lemma)
        else:
            tokens.append(tok.text)

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
def _noun_to_schema_via_wordnet(lemma: str) -> Optional[str]:
    """Map a noun lemma to a schema via WordNet hypernym closure.
    Spec Part 1, Field: schematic_category (step 6).
    Grammar reference Section 8.1: structural noun detection."""
    try:
        _ensure_wordnet()
        from nltk.corpus import wordnet as _wn
        noun_synsets = _wn.synsets(lemma, pos="n")
        if not noun_synsets:
            return None

        for ss in noun_synsets:
            hypernyms = {h.name() for h in ss.closure(lambda s: s.hypernyms())}
            hypernyms.add(ss.name())
            for anchors, schema in _NOUN_SCHEMA_ANCHORS:
                if hypernyms & anchors:
                    return schema
        return None
    except Exception:
        return None


def _extract_schematic(doc, root, verb_class: VerbClass) -> str:
    """Extract schematic trace: category from verb class + NER + WordNet refinement.

    Spec Part 1, Field: edge_schematic_category.  Priority order:
        1. Xcomp/ccomp override for intent/preference/location verbs
        2. Light verb delegation (do/have/take/make/give/get -> use dobj)
        3. Verb class -> schema map
        4. Kinship noun override (WordNet hypernym closure)
        5. NER refinement (ORG->career, GPE->housing, etc.)
        6. Noun hypernym fallback for uncategorized/experience
        7. Creative/recreational verb check (WordNet)
    """
    # Step 1: xcomp/ccomp override for intent/preference/location verbs
    _LIGHT_VERB_LEMMAS = frozenset({"do", "have", "take", "make", "give", "get"})
    if (root is not None
            and verb_class in (
                VerbClass.PREFERENCE, VerbClass.UNKNOWN,
                VerbClass.BE, VerbClass.HAVE, VerbClass.LOCATION,
            )
            and root.pos_ in ("VERB", "AUX")):
        for child in root.children:
            if child.dep_ in ("xcomp", "ccomp") and child.pos_ == "VERB":
                # Step 2: light verb delegation
                if child.lemma_ in _LIGHT_VERB_LEMMAS:
                    for gc in child.children:
                        if gc.dep_ == "dobj":
                            dobj_schema = _noun_to_schema_via_wordnet(gc.lemma_)
                            if dobj_schema is not None:
                                return dobj_schema
                else:
                    # Non-light complement verb: trust its verb class
                    comp_vc = classify_verb_class(child.lemma_)
                    comp_schema = _VERB_CLASS_TO_SCHEMA.get(
                        comp_vc, "uncategorized",
                    )
                    if comp_schema not in ("uncategorized", "identity"):
                        return comp_schema
                    for gc in child.children:
                        if gc.dep_ == "dobj":
                            dobj_schema = _noun_to_schema_via_wordnet(gc.lemma_)
                            if dobj_schema is not None:
                                return dobj_schema

    # Step 2b: root-level light verb delegation
    # "She got a promotion" -> root=got, dobj=promotion -> delegate to "promotion"
    if (root is not None
            and root.pos_ in ("VERB", "AUX")
            and root.lemma_ in _LIGHT_VERB_LEMMAS):
        for child in root.children:
            if child.dep_ == "dobj":
                dobj_schema = _noun_to_schema_via_wordnet(child.lemma_)
                if dobj_schema is not None:
                    return dobj_schema

    # Step 3: verb class -> schema
    schema = _VERB_CLASS_TO_SCHEMA.get(verb_class, "uncategorized")

    # Step 4: kinship noun override (Grammar reference Section 2.1/7.1)
    # Do not override strong verb-class signals (career, health, finance, housing)
    if schema not in ("career", "health", "finance", "housing"):
        try:
            _ensure_wordnet()
            from nltk.corpus import wordnet as _wn
            _kinship_anchors = frozenset({
                "relative.n.01", "parent.n.01", "grandparent.n.01",
                "sibling.n.01", "child.n.02", "spouse.n.01",
                "kinsman.n.01", "ancestor.n.01",
            })
            # Only check kinship nouns in subject or direct object position,
            # not in prepositional phrases. "Celebrate with my family" is
            # about celebration, not family relationships.
            _kinship_deps = frozenset({"nsubj", "nsubjpass", "dobj", "attr"})
            for tok in doc:
                if (tok.pos_ == "NOUN" and not tok.is_stop
                        and tok.dep_ in _kinship_deps):
                    for ss in _wn.synsets(tok.lemma_, pos="n"):
                        hypernyms = {
                            h.name()
                            for h in ss.closure(lambda s: s.hypernyms())
                        }
                        if hypernyms & _kinship_anchors:
                            schema = "family"
                            break
                    if schema == "family":
                        break
        except Exception:
            pass

    if schema == "family":
        return schema

    # Step 5: NER refinement for generic schemas
    # Guard: only trust NER when entity root POS is PROPN. spaCy mis-tags
    # common nouns as GPE/ORG on re-parsed fragments (e.g., "interview" → GPE).
    if schema in ("uncategorized", "identity", "planning"):
        doc_ner_labels = frozenset(
            ent.label_ for ent in doc.ents
            if ent.root.pos_ == "PROPN"
        )
        if "ORG" in doc_ner_labels:
            schema = "career"
        elif doc_ner_labels & frozenset({"GPE", "FAC"}):
            schema = "housing"
        elif "EVENT" in doc_ner_labels:
            schema = "experience"
        elif "MONEY" in doc_ner_labels:
            schema = "finance"
        elif "NORP" in doc_ner_labels:
            schema = "social"
        elif "LAW" in doc_ner_labels:
            schema = "legal"
        elif "WORK_OF_ART" in doc_ner_labels:
            schema = "culture"
        elif "PRODUCT" in doc_ner_labels:
            schema = "commercial"
        elif "QUANTITY" in doc_ner_labels:
            schema = "measurement"

    # Step 6: noun hypernym fallback (Grammar reference Section 8.1/3.5)
    if schema in ("uncategorized", "experience"):
        for tok in doc:
            if tok.pos_ == "NOUN" and not tok.is_stop:
                noun_schema = _noun_to_schema_via_wordnet(tok.lemma_)
                if noun_schema is not None:
                    schema = noun_schema
                    break

        # Step 7: creative/recreational verb check
        if (schema in ("uncategorized", "experience")
                and root is not None
                and root.pos_ == "VERB"):
            try:
                _ensure_wordnet()
                from nltk.corpus import wordnet as _wn
                _creative_verb_anchors = frozenset({
                    "create.v.03", "create.v.05",
                })
                _recreational_verb_anchors = frozenset({
                    "play.v.01", "play.v.03",
                    "swim.v.01", "camp.v.01",
                })
                for ss in _wn.synsets(root.lemma_, pos=_wn.VERB):
                    hypernyms = {
                        h.name() for h in ss.closure(lambda s: s.hypernyms())
                    }
                    hypernyms.add(ss.name())
                    if hypernyms & _creative_verb_anchors:
                        schema = "hobby"
                        break
                    if hypernyms & _recreational_verb_anchors:
                        schema = "hobby"
                        break
            except Exception:
                pass

    # Possessive-subject family detection
    # Only triggers when the head noun of the subject is a kinship term
    # (validated via WordNet hypernym closure).
    if schema in ("uncategorized", "identity", "planning"):
        if root is not None:
            for child in root.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    for gc in child.children:
                        if gc.dep_ == "poss":
                            # Validate that the subject head noun is kinship
                            if _is_kinship_noun(child.lemma_):
                                schema = "family"
                            break

    return schema


# ---------------------------------------------------------------------------
# _build_trace_decomposition -- DEPRECATED (M5 audit 2026-04-29)
# Zero production callers. Only referenced by tests and scripts.
# Diverges from primary path (_extract_traces_from_sentence): no reported
# speech handling, no compound splitting, no inline conditional detection.
# Tests should migrate to _extract_traces_from_sentence. Do NOT add new
# callers -- use _extract_traces_from_sentence instead.
# ---------------------------------------------------------------------------

def _build_trace_decomposition(
    triple: Triple,
    doc,
    speaker: Optional[str],
    tense_aspect: TenseAspect,
    verb_class: VerbClass,
    emotion: Optional[str] = None,
    listener: str = "user",
) -> TraceDecomposition:
    """DEPRECATED: Build a TraceDecomposition from a Triple and its parse context.

    This function is kept for backward compatibility -- tests import it.
    It diverges from the primary extraction path (_extract_traces_from_sentence)
    and should NOT be used in new code. See VIOLATION M5 notes above.
    """
    root = _get_root(doc)

    pred_verb = triple.predicate.split("_")[0] if triple.predicate else ""
    if pred_verb:
        refined = classify_verb_class(pred_verb)
        if refined not in (VerbClass.BE, VerbClass.HAVE, VerbClass.UNKNOWN):
            verb_class = refined
        # If refinement returns UNKNOWN, keep the caller's verb_class

    episodic_fact = f"{triple.predicate} {triple.object}".strip()
    if not episodic_fact:
        episodic_fact = str(doc).strip()

    significance = _compute_significance(verb_class, tense_aspect)

    temporal_direction, temporal_expression = _extract_temporal(doc, tense_aspect)
    relational_subject, relational_entities, _ = _extract_relational(doc, speaker, listener=listener)
    relational_type = _VERB_CLASS_TO_RELTYPE.get(verb_class, "personal")
    schematic_category = _extract_schematic(doc, root, verb_class)

    emotional_state = emotion
    emotional_valence = None
    emotional_target = None
    if emotional_state is None:
        emotional_state, emotional_valence, emotional_target = (
            _extract_emotional(doc, root)
        )
    else:
        _, emotional_valence, emotional_target = _extract_emotional(doc, root)

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
        source_text=str(doc),
        utterance_type=triple.utterance_type,
        mood=triple.mood,
        negated=triple.negated,
        is_historical=triple.is_historical,
        subject=triple.subject,
        predicate=triple.predicate,
        object=triple.object,
        extraction_rule=triple.extraction_rule,
    )


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
        extraction_rule="trace",
    )


# ---------------------------------------------------------------------------
# Unified imposed-trace builder (module-level)
# All imposed-fact paths delegate here so every trace gets all 5 extractors.
# ---------------------------------------------------------------------------

def _build_imposed_trace(
    clause_text: str,
    imposed_subject: str,
    speaker: Optional[str],
    tense_aspect: "TenseAspect",
    listener: str = "user",
    mood: str = "indicative",
    extraction_rule: str = "imposed",
    schematic_hint: Optional[str] = None,
    predicate_override: Optional[str] = None,
    utterance_type: int = 0,
    source_text_override: Optional[str] = None,
) -> Optional["TraceDecomposition"]:
    """Build a fully-traced TraceDecomposition for an imposed fact.

    Calls all 5 extractors (episodic, emotional, temporal, relational, schematic).
    Used by _extract_imposed_facts and conditional clause handling.
    """
    clause_text = clause_text.strip()
    if not clause_text or len(clause_text) < 3:
        return None

    frag_nlp = _get_nlp_fragment()
    clause_doc = frag_nlp(clause_text)
    clause_root = _get_root(clause_doc)

    verb_class = VerbClass.UNKNOWN
    if clause_root and clause_root.pos_ in ("VERB", "AUX"):
        verb_class = classify_verb_class(clause_root.lemma_)
        verb_class = _reclassify_location_by_object(
            clause_doc, clause_root, verb_class,
        )

    # 1. Episodic trace
    episodic = _extract_episodic(clause_doc, clause_root)
    gram_obj = _extract_grammatical_object(clause_doc, clause_root)

    # 2. Emotional trace
    emotional_state, emotional_valence, emotional_target = (
        _extract_emotional(clause_doc, clause_root)
    )

    # 3. Temporal trace
    temporal_direction, temporal_expression = _extract_temporal(
        clause_doc, tense_aspect,
    )

    # 4. Relational trace
    rel_subject, rel_entities, rel_type_base = _extract_relational(
        clause_doc, speaker, listener=listener,
    )
    if imposed_subject and imposed_subject != "user":
        rel_subject = imposed_subject
    rel_type = _VERB_CLASS_TO_RELTYPE.get(verb_class, rel_type_base)

    # 5. Schematic trace
    schema = schematic_hint or _extract_schematic(
        clause_doc, clause_root, verb_class,
    )

    significance = _compute_significance(verb_class, tense_aspect)

    # Predicate derivation
    if predicate_override:
        root_lemma = predicate_override
    else:
        root_lemma = clause_root.lemma_ if clause_root else ""
        if clause_root and clause_root.pos_ == "AUX":
            for child in clause_root.children:
                if child.dep_ in ("xcomp", "ccomp", "acomp"):
                    if child.pos_ in ("VERB", "ADJ"):
                        root_lemma = child.lemma_
                        break

    return TraceDecomposition(
        episodic_fact=episodic,
        episodic_significance=significance,
        emotional_state=emotional_state,
        emotional_valence=emotional_valence,
        emotional_target=emotional_target,
        temporal_direction=temporal_direction,
        temporal_expression=temporal_expression,
        relational_subject=rel_subject,
        relational_entities=rel_entities,
        relational_type=rel_type,
        schematic_category=schema,
        source_text=source_text_override or clause_text,
        utterance_type=utterance_type,
        mood=mood,
        negated=detect_negation(clause_doc),
        is_historical=tense_aspect.tense == "past",
        subject=rel_subject,
        predicate=root_lemma.lower(),
        object=gram_obj,
        extraction_rule=extraction_rule,
    )


# ---------------------------------------------------------------------------
# Imposed facts extraction
# Spec Part 2, Multi-clause Sentences + imposed-facts-taxonomy.md
# ---------------------------------------------------------------------------

def _extract_imposed_facts(
    doc,
    speaker: Optional[str],
    tense_aspect: TenseAspect,
    existing_sources: Optional[frozenset] = None,
    listener: str = "user",
) -> List[TraceDecomposition]:
    """Extract imposed facts from subordinate constructions in a single pass.

    Scans every token once, dispatches on dep_/tag_ to detect
    construction types from the imposed-facts taxonomy:
        relcl (9,10), advcl (17), appos (19), acl (18),
        csubj (14), expl (24), gerund (21), ccomp/direct speech (12),
        correlative (26), comparative (27).
    """
    imposed: List[TraceDecomposition] = []
    _existing = existing_sources or frozenset()

    def _is_duplicate(clause_text: str) -> bool:
        ct = clause_text.strip().lower()
        if not ct:
            return True
        for existing in _existing:
            el = existing.strip().lower()
            if ct in el or el in ct:
                return True
        return False

    def _build_imposed(
        clause_text: str,
        imposed_subject: str,
        mood: str = "indicative",
        extraction_rule: str = "imposed",
        schematic_hint: Optional[str] = None,
        predicate_override: Optional[str] = None,
        skip_dedup: bool = False,
    ) -> Optional[TraceDecomposition]:
        clause_text = clause_text.strip()
        if not clause_text or len(clause_text) < 3:
            return None
        if not skip_dedup and _is_duplicate(clause_text):
            return None

        return _build_imposed_trace(
            clause_text=clause_text,
            imposed_subject=imposed_subject,
            speaker=speaker,
            tense_aspect=tense_aspect,
            listener=listener,
            mood=mood,
            extraction_rule=extraction_rule,
            schematic_hint=schematic_hint,
            predicate_override=predicate_override,
        )

    # Single pass over all tokens
    seen_subtree_starts: set = set()

    for tok in doc:
        if tok.i in seen_subtree_starts:
            continue

        dep = tok.dep_
        pos = tok.pos_
        tag = tok.tag_

        # --- Type 9/10: Relative clauses (dep=relcl) ---
        # Grammar Gap #13: Restrictive vs Non-restrictive relative clauses.
        # Non-restrictive are set off by commas ("My brother, who lives in
        # Paris") and carry stronger presupposition (background fact).
        # Restrictive have no commas ("The man who called") and are defining.
        if dep == "relcl" and pos in ("VERB", "AUX"):
            subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
            imposed_subject = (
                tok.head.text if tok.head else (speaker or "user")
            )
            _REL_PRONOUNS = frozenset({
                "that", "which", "who", "whom", "whose",
            })
            filtered = [
                t.text for t in subtree_sorted
                if t.text.lower() not in _REL_PRONOUNS
            ]
            clean_clause = " ".join(filtered).strip()
            imposed_text = f"{imposed_subject} {clean_clause}".strip()

            # Detect restrictive vs non-restrictive: check for comma
            # immediately before the relative clause span.
            _is_nonrestrictive = False
            relcl_start = min(t.i for t in tok.subtree)
            # Walk backwards to find the token just before the relcl span
            # (skipping relative pronouns that precede the verb).
            # A comma before the relcl region signals non-restrictive.
            if relcl_start > 0:
                prev_tok = doc[relcl_start - 1]
                if prev_tok.text == ",":
                    _is_nonrestrictive = True
                # Sometimes the relative pronoun is at relcl_start and the
                # comma is one more token back.
                elif (prev_tok.text.lower() in _REL_PRONOUNS
                        and relcl_start > 1
                        and doc[relcl_start - 2].text == ","):
                    _is_nonrestrictive = True
            # Also check if the head noun itself is followed by a comma
            if tok.head and tok.head.i + 1 < len(doc):
                next_after_head = doc[tok.head.i + 1]
                if next_after_head.text == ",":
                    _is_nonrestrictive = True

            relcl_rule = ("imposed_relcl_nonrestrictive" if _is_nonrestrictive
                          else "imposed_relcl_restrictive")

            decomp = _build_imposed(
                imposed_text, imposed_subject,
                extraction_rule=relcl_rule,
                predicate_override=tok.lemma_,
            )
            if decomp is not None:
                imposed.append(decomp)
                seen_subtree_starts.add(tok.i)

        # --- Type 17: Adverbial clauses (dep=advcl) ---
        elif dep == "advcl" and pos in ("VERB", "AUX"):
            mark_tok = None
            for child in tok.children:
                if child.dep_ == "mark":
                    mark_tok = child
                    break
            # spaCy sometimes labels temporal subordinators (when, where)
            # as advmod rather than mark — check advmod children too.
            if mark_tok is None:
                _SUBORDINATOR_ADVMODS = frozenset({
                    "when", "whenever", "where", "wherever", "while",
                    "once", "before", "after",
                })
                for child in tok.children:
                    if (child.dep_ == "advmod"
                            and child.lemma_.lower() in _SUBORDINATOR_ADVMODS):
                        mark_tok = child
                        break
            # spaCy sometimes assigns dep=aux to infinitive "to" particle
            # in purpose clauses (e.g., "to buy" -> "to" has dep=aux)
            if mark_tok is None:
                for child in tok.children:
                    if (child.dep_ == "aux" and child.lemma_ == "to"
                            and child.tag_ == "TO"):
                        mark_tok = child
                        break
            mark_lemma = mark_tok.lemma_.lower() if mark_tok else ""

            # --- Types 5-8: Conditional clauses (if/unless/whether/provided) ---
            if mark_lemma in ("if", "unless", "whether", "provided"):
                # Classify conditional type by tense pattern
                advcl_verb = tok
                advcl_tag = advcl_verb.tag_  # VBP, VBD, VBN, etc.

                # Check for "had + VBN" pattern (third conditional)
                has_had_vbn = False
                if advcl_tag == "VBN":
                    for child in advcl_verb.children:
                        if (child.dep_ == "aux" and
                                child.lemma_.lower() == "have" and
                                child.tag_ == "VBD"):
                            has_had_vbn = True
                            break

                # Check main clause for modal auxiliaries
                main_verb = tok.head
                main_modal = ""
                if main_verb:
                    for child in main_verb.children:
                        if child.dep_ == "aux" and child.tag_ == "MD":
                            main_modal = child.lemma_.lower()
                            break

                # Determine conditional type
                if has_had_vbn and main_modal in ("would", "could", "might"):
                    cond_type = "third"
                    cond_mood = "conditional"
                    cond_is_historical = True
                elif advcl_tag == "VBD" and main_modal in (
                    "would", "could", "might",
                ):
                    cond_type = "second"
                    cond_mood = "conditional"
                    cond_is_historical = False
                elif advcl_tag in ("VBP", "VBZ") and main_modal in (
                    "will", "shall",
                ):
                    cond_type = "first"
                    cond_mood = "indicative"
                    cond_is_historical = False
                elif advcl_tag in ("VBP", "VBZ") and main_modal == "":
                    cond_type = "zero"
                    cond_mood = "indicative"
                    cond_is_historical = False
                else:
                    # Default: first conditional (most common)
                    cond_type = "first"
                    cond_mood = "indicative"
                    cond_is_historical = False

                # Extract subject of the conditional clause
                cond_subj = speaker or "user"
                for child in tok.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        cond_subj = child.text
                        break

                # Build clause text without the mark token
                subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
                filtered = [
                    t.text for t in subtree_sorted if t.i != mark_tok.i
                ]
                clean_text = " ".join(filtered).strip()

                extraction_rule = f"imposed_conditional_{cond_type}"

                # Build the imposed fact via unified builder
                clause_text = clean_text.strip()
                if clause_text and len(clause_text) >= 3 and not _is_duplicate(clause_text):
                    decomp = _build_imposed_trace(
                        clause_text=clause_text,
                        imposed_subject=cond_subj,
                        speaker=speaker,
                        tense_aspect=tense_aspect,
                        listener=listener,
                        mood=cond_mood,
                        extraction_rule=extraction_rule,
                        predicate_override=tok.lemma_.lower(),
                    )
                    if decomp is not None:
                        # Override is_historical for conditional-specific semantics
                        decomp.is_historical = cond_is_historical
                        imposed.append(decomp)
                    seen_subtree_starts.add(tok.i)

            elif mark_lemma == "to":
                # --- Type 22: Infinitive purpose clauses ---
                # "I went to the store to buy groceries" -> purpose = "buy groceries"
                # The purpose clause presupposes the entities exist.

                # Skip "used to" habitual construction
                if (tok.head.lemma_ == "use" and tok.head.tag_ == "VBD"):
                    continue

                subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)

                # Determine the subject: inherit from main clause subject
                purpose_subj = speaker or "user"
                main_verb = tok.head
                if main_verb:
                    for child in main_verb.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            purpose_subj = child.text
                            break

                # Build clause text without the "to" mark
                filtered = [
                    t.text for t in subtree_sorted if t.i != mark_tok.i
                ]
                clean_text = " ".join(filtered).strip()

                # Extract grammatical object of the purpose verb directly
                # from the original parse (more reliable than re-parsing)
                purpose_obj = ""
                for child in tok.children:
                    if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
                        obj_subtree = sorted(child.subtree, key=lambda t: t.i)
                        purpose_obj = " ".join(t.text for t in obj_subtree)
                        break
                if not purpose_obj:
                    # Try prep object
                    for child in tok.children:
                        if child.dep_ == "prep":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    obj_subtree = sorted(
                                        grandchild.subtree, key=lambda t: t.i,
                                    )
                                    purpose_obj = " ".join(
                                        t.text for t in obj_subtree
                                    )
                                    break
                            break

                decomp = _build_imposed(
                    clean_text, purpose_subj,
                    extraction_rule="imposed_infinitive_purpose",
                    predicate_override=tok.lemma_,
                    skip_dedup=True,
                )
                if decomp is not None:
                    # Override object with what we extracted directly
                    if purpose_obj:
                        decomp.object = purpose_obj
                    imposed.append(decomp)
                    seen_subtree_starts.add(tok.i)

            # --- Type 20: Absolute phrases (noun + participle, no mark) ---
            # "The sun having set, we went home" — parsed as advcl with
            # VBG/VBN head, no subordinating conjunction, own nsubj that
            # differs from the main clause nsubj. Scene-setting presupposed.
            elif (not mark_tok and tag in ("VBG", "VBN")):
                # Check for own nsubj (absolute phrases have an independent
                # subject different from the main clause).
                abs_subj = None
                for child in tok.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        abs_subj = child.text
                        break
                # Find main clause nsubj for comparison
                main_subj = None
                main_verb = tok.head
                if main_verb:
                    for child in main_verb.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            main_subj = child.text
                            break
                # Absolute phrase: has own subject AND it differs from main
                if (abs_subj is not None
                        and (main_subj is None
                             or abs_subj.lower() != main_subj.lower())):
                    subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
                    clause_text = " ".join(t.text for t in subtree_sorted)
                    imposed_text = f"{abs_subj} {tok.lemma_}".strip()
                    # Use full clause for richer context
                    full_clause = f"{abs_subj} {clause_text}".strip()
                    decomp = _build_imposed(
                        full_clause, abs_subj,
                        extraction_rule="imposed_absolute_phrase",
                        predicate_override=tok.lemma_,
                    )
                    if decomp is not None:
                        imposed.append(decomp)
                        seen_subtree_starts.add(tok.i)
                else:
                    # VBG/VBN advcl without mark but no independent subject —
                    # treat as regular participial advcl.
                    subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
                    advcl_subj = speaker or "user"
                    for child in tok.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            advcl_subj = child.text
                            break
                    clean_text = " ".join(t.text for t in subtree_sorted)
                    decomp = _build_imposed(
                        clean_text, advcl_subj,
                        extraction_rule="imposed_advcl_general",
                    )
                    if decomp is not None:
                        imposed.append(decomp)
                        seen_subtree_starts.add(tok.i)

            elif not mark_tok:
                # advcl with no mark and not VBG/VBN — general advcl
                subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
                advcl_subj = speaker or "user"
                for child in tok.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        advcl_subj = child.text
                        break
                clean_text = " ".join(t.text for t in subtree_sorted)
                decomp = _build_imposed(
                    clean_text, advcl_subj,
                    extraction_rule="imposed_advcl_general",
                )
                if decomp is not None:
                    imposed.append(decomp)
                    seen_subtree_starts.add(tok.i)

            else:
                # Non-conditional, non-purpose advcl — classify by mark lemma.
                # Grammar pp.902-906: subordinating conjunctions are a closed
                # grammatical class grouped by function.
                _ADVCL_TIME = frozenset({
                    "when", "whenever", "while", "before", "after",
                    "since", "until", "once", "till", "as soon as",
                })
                _ADVCL_PLACE = frozenset({
                    "where", "wherever", "everywhere", "anywhere",
                })
                _ADVCL_REASON = frozenset({
                    "because", "as", "since", "so", "for",
                    "now that", "given that", "in that",
                })
                _ADVCL_MANNER = frozenset({
                    "like", "as if", "as though", "the way", "than",
                })
                _ADVCL_CONTRAST = frozenset({
                    "though", "although", "even though", "whereas",
                    "even if", "while", "whilst",
                })

                # Classify advcl type from the mark token's lemma
                if mark_lemma in _ADVCL_TIME:
                    advcl_type = "time"
                elif mark_lemma in _ADVCL_PLACE:
                    advcl_type = "place"
                elif mark_lemma in _ADVCL_REASON:
                    advcl_type = "reason"
                elif mark_lemma in _ADVCL_MANNER:
                    advcl_type = "manner"
                elif mark_lemma in _ADVCL_CONTRAST:
                    advcl_type = "contrast"
                else:
                    advcl_type = "general"

                subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
                advcl_subj = speaker or "user"
                for child in tok.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        advcl_subj = child.text
                        break

                if mark_tok:
                    filtered = [
                        t.text for t in subtree_sorted if t.i != mark_tok.i
                    ]
                    clean_text = " ".join(filtered).strip()
                else:
                    clean_text = " ".join(t.text for t in subtree_sorted)

                extraction_rule = f"imposed_advcl_{advcl_type}"

                # Classified advcl types carry semantic value beyond the
                # main trace (reason, contrast, time, etc.), so skip dedup
                # to ensure the classification is preserved.
                _skip = advcl_type != "general"

                decomp = _build_imposed(
                    clean_text, advcl_subj,
                    extraction_rule=extraction_rule,
                    skip_dedup=_skip,
                )
                if decomp is not None:
                    # Time advcl: promote clause content as temporal expression
                    # ONLY if the clause contains a real temporal signal (DATE/TIME
                    # NER or structural time noun). Without this guard, phrasal
                    # verbs like "look after" trigger false time classification and
                    # stuff entire clauses ("I look after myself") into the field.
                    if advcl_type == "time" and clean_text:
                        _advcl_doc = _get_nlp()(clean_text)
                        _has_date_ent = any(
                            e.label_ in ("DATE", "TIME") for e in _advcl_doc.ents
                        )
                        _TIME_NOUNS = frozenset({
                            "year", "month", "week", "day", "hour", "minute",
                            "decade", "century", "semester", "quarter",
                            "morning", "afternoon", "evening", "night",
                            "weekend", "tonight", "tomorrow", "yesterday",
                        })
                        _has_time_noun = any(
                            t.lemma_.lower() in _TIME_NOUNS for t in _advcl_doc
                        )
                        if _has_date_ent or _has_time_noun:
                            decomp.temporal_expression = clean_text
                    imposed.append(decomp)
                    seen_subtree_starts.add(tok.i)

        # --- Type 22: Infinitive purpose via xcomp (dep=xcomp with "to") ---
        # "She exercises to stay healthy" -> "stay" is xcomp, not advcl
        # Distinguish from subject-control: "I want to go" (control verb)
        elif (dep == "xcomp" and pos == "VERB" and tag == "VB"):
            # Check if this xcomp has a "to" particle child
            has_to = False
            to_tok = None
            for child in tok.children:
                if child.lemma_ == "to" and child.tag_ == "TO":
                    has_to = True
                    to_tok = child
                    break
            if has_to and tok.head and tok.head.pos_ == "VERB":
                # Skip "used to" habitual construction
                if (tok.head.lemma_ == "use" and tok.head.tag_ == "VBD"):
                    continue

                # Skip "be going to" periphrastic future — not a purpose clause
                _is_going_to_future = False
                if tok.head.lemma_ == "go" and tok.head.tag_ == "VBG":
                    for sibling in tok.head.children:
                        if sibling.dep_ == "aux" and sibling.lemma_ == "be":
                            _is_going_to_future = True
                            break
                # Check if head verb is a control/framing verb via the
                # universal frame skipper. If _find_content_verb skips
                # past the head to this xcomp, it's subject control
                # (not purpose). If it stays on the head, head IS
                # the content verb and this xcomp is purpose.
                _content = _find_content_verb(tok.head)
                _is_control = (_content != tok.head)
                if (not _is_going_to_future and not _is_control):
                    # This is a purpose xcomp — extract as imposed fact
                    subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)

                    purpose_subj = speaker or "user"
                    for child in tok.head.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            purpose_subj = child.text
                            break

                    # Build text without "to"
                    filtered = [
                        t.text for t in subtree_sorted
                        if not (t.i == to_tok.i)
                    ]
                    clean_text = " ".join(filtered).strip()

                    # Extract object from the purpose verb
                    purpose_obj = ""
                    for child in tok.children:
                        if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
                            obj_subtree = sorted(
                                child.subtree, key=lambda t: t.i,
                            )
                            purpose_obj = " ".join(
                                t.text for t in obj_subtree
                            )
                            break

                    decomp = _build_imposed(
                        clean_text, purpose_subj,
                        extraction_rule="imposed_infinitive_purpose",
                        predicate_override=tok.lemma_,
                        skip_dedup=True,
                    )
                    if decomp is not None:
                        if purpose_obj:
                            decomp.object = purpose_obj
                        imposed.append(decomp)
                        seen_subtree_starts.add(tok.i)

        # --- Type 19: Appositives (dep=appos) ---
        elif dep == "appos":
            entity = tok.head.text if tok.head else ""
            subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
            description = " ".join(t.text for t in subtree_sorted)
            if entity and description:
                imposed_text = f"{entity} is {description}"
                decomp = _build_imposed(
                    imposed_text, entity,
                    extraction_rule="imposed_appos",
                    schematic_hint="identity",
                    predicate_override="be",
                )
                if decomp is not None:
                    imposed.append(decomp)
                    seen_subtree_starts.add(tok.i)

        # --- Type 18: Participial phrases (dep=acl, VBG/VBN) ---
        elif dep == "acl" and tag in ("VBG", "VBN"):
            subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
            clause_text = " ".join(t.text for t in subtree_sorted)
            acl_subject = (
                tok.head.text if tok.head else (speaker or "user")
            )
            imposed_text = f"{acl_subject} {clause_text}".strip()
            decomp = _build_imposed(
                imposed_text, acl_subject,
                extraction_rule="imposed_acl",
                predicate_override=tok.lemma_,
            )
            if decomp is not None:
                imposed.append(decomp)
                seen_subtree_starts.add(tok.i)

        # --- Type 14: Noun clauses as subject (dep=csubj) ---
        elif dep in ("csubj", "csubjpass") and pos in ("VERB", "AUX"):
            head_vc = VerbClass.UNKNOWN
            if tok.head and tok.head.pos_ in ("VERB", "AUX"):
                head_vc = classify_verb_class(tok.head.lemma_)
            if head_vc == VerbClass.SPEECH:
                continue

            subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
            filtered = [
                t.text for t in subtree_sorted
                if not (t.dep_ == "mark" and t.lemma_.lower() == "that")
            ]
            clause_text = " ".join(filtered).strip()

            csubj_subject = speaker or "user"
            for child in tok.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    csubj_subject = child.text
                    break

            decomp = _build_imposed(
                clause_text, csubj_subject,
                extraction_rule="imposed_csubj",
            )
            if decomp is not None:
                imposed.append(decomp)
                seen_subtree_starts.add(tok.i)

        # --- Type 24: Existential there (dep=expl) ---
        elif dep == "expl" and tok.text.lower() == "there":
            be_verb = tok.head
            if be_verb and be_verb.lemma_ == "be":
                content_tok = None
                for child in be_verb.children:
                    if child.dep_ in ("attr", "nsubj"):
                        content_tok = child
                        break
                if content_tok is not None:
                    subtree_sorted = sorted(
                        content_tok.subtree, key=lambda t: t.i,
                    )
                    content_text = " ".join(
                        t.text for t in subtree_sorted
                    )
                    preps = []
                    for child in be_verb.children:
                        if child.dep_ == "prep":
                            prep_subtree = sorted(
                                child.subtree, key=lambda t: t.i,
                            )
                            preps.append(
                                " ".join(t.text for t in prep_subtree)
                            )
                    full_text = content_text
                    if preps:
                        full_text = f"{content_text} {' '.join(preps)}"

                    decomp = _build_imposed(
                        full_text, speaker or "user",
                        extraction_rule="imposed_expl",
                    )
                    if decomp is not None:
                        imposed.append(decomp)
                        seen_subtree_starts.add(tok.i)

        # --- Type 21: Gerund phrases (VBG as nsubj/dobj/pobj) ---
        elif tag == "VBG" and dep in ("nsubj", "dobj", "pobj"):
            subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
            clause_text = " ".join(t.text for t in subtree_sorted)
            gerund_subj = speaker or "user"
            for child in tok.children:
                if child.dep_ in ("poss", "nsubj"):
                    gerund_subj = child.text
                    break

            decomp = _build_imposed(
                clause_text, gerund_subj,
                extraction_rule="imposed_gerund",
            )
            if decomp is not None:
                imposed.append(decomp)
                seen_subtree_starts.add(tok.i)

        # --- Type 15: Subjunctive wish (Grammar pp.735-739) ---
        # "I wish I spoke French" -> wish clause is counterfactual
        elif dep == "ccomp" and pos in ("VERB", "AUX") and tok.head and tok.head.lemma_ == "wish":
            subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
            wish_subj = speaker or "user"
            for child in tok.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    wish_subj = child.text
                    break
            clause_text = " ".join(t.text for t in subtree_sorted).strip()

            decomp = _build_imposed(
                clause_text, wish_subj,
                mood="subjunctive",
                extraction_rule="imposed_subjunctive_wish",
                predicate_override=tok.lemma_,
                skip_dedup=True,
            )
            if decomp is not None:
                imposed.append(decomp)
                seen_subtree_starts.add(tok.i)

        # --- Type 12: Direct speech (ccomp with quotation marks) ---
        elif dep == "ccomp" and pos in ("VERB", "AUX"):
            has_quotes = any(
                t.text in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019")
                for t in doc
            )
            head_vc = VerbClass.UNKNOWN
            if tok.head and tok.head.pos_ in ("VERB", "AUX"):
                head_vc = classify_verb_class(tok.head.lemma_)
            if head_vc == VerbClass.SPEECH:
                continue

            if has_quotes:
                subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
                filtered = [
                    t.text for t in subtree_sorted
                    if t.text not in (
                        '"', "'", "\u201c", "\u201d", "\u2018", "\u2019",
                    )
                ]
                clause_text = " ".join(filtered).strip()
                ccomp_subj = speaker or "user"
                for child in tok.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        ccomp_subj = child.text
                        break

                decomp = _build_imposed(
                    clause_text, ccomp_subj,
                    extraction_rule="imposed_direct_speech",
                )
                if decomp is not None:
                    imposed.append(decomp)
                    seen_subtree_starts.add(tok.i)
            else:
                # --- Type 14: Noun clause as object ---
                # Non-speech, non-quote ccomp: the embedded clause is a
                # known proposition.  "I realized self-care is important"
                # -> imposed: "self-care is important" (its own fact).
                subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
                filtered = [
                    t.text for t in subtree_sorted
                    if not (t.dep_ == "mark"
                            and t.lemma_.lower() == "that")
                ]
                clause_text = " ".join(filtered).strip()
                ccomp_subj = speaker or "user"
                for child in tok.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        ccomp_subj = child.text
                        break

                decomp = _build_imposed(
                    clause_text, ccomp_subj,
                    extraction_rule="imposed_noun_clause",
                    predicate_override=tok.lemma_,
                )
                if decomp is not None:
                    imposed.append(decomp)
                    seen_subtree_starts.add(tok.i)

        # --- Type 26: Correlative conjunctions ---
        elif (dep == "preconj"
                and tok.lemma_.lower() in ("both", "either", "neither")):
            head = tok.head
            if head:
                conj_tok = None
                for child in head.children:
                    if child.dep_ == "conj":
                        conj_tok = child
                        break
                if conj_tok:
                    for element in (head, conj_tok):
                        el_subtree = sorted(
                            element.subtree, key=lambda t: t.i,
                        )
                        filtered = [
                            t.text for t in el_subtree
                            if t.dep_ not in ("preconj", "cc")
                        ]
                        el_text = " ".join(filtered).strip()
                        if el_text and not _is_duplicate(el_text):
                            decomp = _build_imposed(
                                f"{el_text} exists",
                                speaker or "user",
                                extraction_rule="imposed_correlative",
                            )
                            if decomp is not None:
                                imposed.append(decomp)

        # --- Type 27: Comparative/superlative ---
        elif pos == "ADJ" and tag in ("JJR", "JJS"):
            than_obj = None
            for child in tok.children:
                if child.dep_ == "prep" and child.lemma_ == "than":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            than_obj = _span_text(gc)
                            break
                    break
            if than_obj:
                base_adj = tok.lemma_ if tok.lemma_ else tok.text
                imposed_text = f"{than_obj} is {base_adj}"
                decomp = _build_imposed(
                    imposed_text, than_obj,
                    extraction_rule="imposed_comparative",
                )
                if decomp is not None:
                    imposed.append(decomp)

        # --- Type 2: Negative frame entity extraction (dep=neg) ---
        elif dep == "neg":
            # "I don't have a car" -> "car" exists as a concept
            # Walk to the negated verb, extract its dobj/pobj/attr
            head_verb = tok.head
            if head_verb and head_verb.pos_ in ("VERB", "AUX"):
                # Find extractable object from the negated verb
                neg_obj_tok = None
                for child in head_verb.children:
                    if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
                        neg_obj_tok = child
                        break
                # Try pobj via prep if no direct object
                if neg_obj_tok is None:
                    for child in head_verb.children:
                        if child.dep_ == "prep":
                            for gc in child.children:
                                if gc.dep_ == "pobj":
                                    neg_obj_tok = gc
                                    break
                            if neg_obj_tok is not None:
                                break
                if neg_obj_tok is not None:
                    obj_text = _span_text(neg_obj_tok)
                    if obj_text and len(obj_text.strip()) >= 2:
                        # Find subject of the negated verb
                        neg_subj = speaker or "user"
                        for child in head_verb.children:
                            if child.dep_ in ("nsubj", "nsubjpass"):
                                neg_subj = child.text
                                break
                        decomp = _build_imposed(
                            obj_text, neg_subj,
                            mood="indicative",
                            extraction_rule="imposed_negative_frame",
                            skip_dedup=True,
                        )
                        if decomp is not None:
                            decomp.predicate = head_verb.lemma_.lower()
                            decomp.object = obj_text
                            imposed.append(decomp)

        # --- Type 16: Passive agent presupposition (dep=nsubjpass) ---
        elif dep == "nsubjpass":
            # "The window was broken by the storm" -> agent = "the storm"
            head_verb = tok.head
            if head_verb and head_verb.pos_ in ("VERB", "AUX"):
                # Check for "by" prep to find agent
                agent_tok = None
                for child in head_verb.children:
                    if child.dep_ == "agent" or (
                        child.dep_ == "prep" and child.lemma_ == "by"
                    ):
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                agent_tok = gc
                                break
                        break

                # Guard: if no explicit agent AND nsubj is inanimate,
                # this is stative (not true passive). "The house was gone"
                # = state, not "someone made the house gone".
                if agent_tok is None:
                    nsubj_is_animate = (
                        tok.pos_ == "PRON"
                        or tok.ent_type_ == "PERSON"
                    )
                    if not nsubj_is_animate:
                        continue  # stative — skip passive agent extraction

                patient_text = _span_text(tok)
                verb_lemma = head_verb.lemma_.lower()

                if agent_tok is not None:
                    agent_text = _span_text(agent_tok)
                    imposed_text = (
                        f"{agent_text} {verb_lemma} {patient_text}"
                    )
                    decomp = _build_imposed(
                        imposed_text, agent_text,
                        mood="indicative",
                        extraction_rule="imposed_passive_agent",
                        predicate_override=verb_lemma,
                        skip_dedup=True,
                    )
                    if decomp is not None:
                        decomp.subject = agent_text
                        decomp.object = patient_text
                        imposed.append(decomp)
                else:
                    # No agent phrase — implied agent
                    imposed_text = f"{verb_lemma} {patient_text}"
                    decomp = _build_imposed(
                        imposed_text, "",
                        mood="indicative",
                        extraction_rule="imposed_passive_agent",
                        predicate_override=verb_lemma,
                        skip_dedup=True,
                    )
                    if decomp is not None:
                        decomp.subject = ""
                        decomp.object = patient_text
                        imposed.append(decomp)

    return imposed


# ---------------------------------------------------------------------------
# Triple derivation (backward compat)
# ---------------------------------------------------------------------------

def _derive_triple(decomp: TraceDecomposition) -> Triple:
    """Derive a backward-compatible Triple from a TraceDecomposition."""
    return Triple(
        subject=decomp.subject or decomp.relational_subject,
        predicate=decomp.predicate,
        object=decomp.object,  # Invariant 1: never fall back to episodic_fact (full clause)
        is_historical=decomp.is_historical,
        utterance_type=decomp.utterance_type,
        negated=decomp.negated,
        mood=decomp.mood,
        extraction_rule=decomp.extraction_rule,
    )


# ---------------------------------------------------------------------------
# Compound clause splitting
# Grammar reference: compound sentences have two independent clauses
# joined by a coordinating conjunction.
# ---------------------------------------------------------------------------

def _head_chain_reaches_root(tok, _max_depth: int = 5) -> bool:
    """Return True if following tok.head through conj deps reaches ROOT."""
    current = tok.head
    for _ in range(_max_depth):
        if current.dep_ == "ROOT":
            return True
        if current.dep_ != "conj":
            return False
        current = current.head
    return False


def _span_has_subject_and_verb(tokens) -> bool:
    """Return True if a list of tokens contains both a subject and a verb/root."""
    has_subj = False
    has_verb = False
    for tok in tokens:
        if tok.dep_ in ("nsubj", "nsubjpass"):
            has_subj = True
        if tok.dep_ == "ROOT" or tok.pos_ in ("VERB", "AUX"):
            has_verb = True
    return has_subj and has_verb


_CONJUNCTION_TYPE_MAP: Dict[str, str] = {
    "and": "additive",
    "but": "contrast",
    "yet": "contrast",
    "for": "reason",
    "so": "consequence",
    "or": "alternative",
    "nor": "alternative",
}


def _patch_fragment_lemmas(frag_doc, original_tokens) -> None:
    """Fix fragment re-parse degradation: when spaCy re-parses a clause
    fragment, POS/tag may change (e.g., VBD→JJ for 'preferred'), causing
    wrong lemmatization. Patch: for each token where the fragment's lemma
    equals the surface form but the original had a different (correct)
    lemma, copy the original's lemma and tag."""
    orig_by_text = {}
    for t in original_tokens:
        key = t.text.lower()
        if key not in orig_by_text:
            orig_by_text[key] = t

    for tok in frag_doc:
        key = tok.text.lower()
        orig = orig_by_text.get(key)
        if orig is None:
            continue
        # If fragment lemma equals surface form but original had a real lemma
        if (tok.lemma_.lower() == tok.text.lower()
                and orig.lemma_.lower() != orig.text.lower()):
            tok.lemma_ = orig.lemma_
        # If fragment tag changed from verb to adjective (common degradation)
        if (tok.tag_ in ("JJ", "NN") and orig.tag_ in ("VBD", "VBG", "VBN", "VBZ")):
            tok.tag_ = orig.tag_
            tok.pos_ = orig.pos_
            # Copy morph features (tense, aspect, etc.) from original
            tok.set_morph(str(orig.morph))


def _split_compound_clauses(doc) -> List:
    """Split compound sentences on coordinating conjunctions (FANBOYS)
    AND semicolons (Grammar reference Page 912).

    Only splits when BOTH sides have a subject + verb (independent clauses).
    "I like pottery and swimming" does NOT split.
    "I like pottery and I went swimming" DOES split.
    "She wanted tennis; he wanted basketball" DOES split.

    Returns a list of (doc, conjunction_type) tuples.  conjunction_type is
    None for the first clause in each split group and for unsplit sentences;
    subsequent clauses carry "additive", "contrast", "reason",
    "consequence", or "alternative" based on the conjunction that
    preceded them.
    """
    frag_nlp = _get_nlp_fragment()
    clauses: List = []

    for sent in doc.sents:
        sent_tokens = list(sent)

        # ---- Phase 1: split on semicolons (Grammar ref #8) ----
        semicolon_indices = [
            tok.i for tok in sent_tokens if tok.text == ";"
        ]

        # Build spans separated by semicolons
        semicolon_spans: List[List] = []
        if semicolon_indices:
            prev = sent.start
            valid_semicolon_split = True
            candidate_spans: List[List] = []
            for sc_i in semicolon_indices:
                span_tokens = [t for t in sent_tokens
                               if t.i >= prev and t.i < sc_i]
                candidate_spans.append(span_tokens)
                prev = sc_i + 1  # skip the semicolon itself
            # remaining tokens after last semicolon
            candidate_spans.append(
                [t for t in sent_tokens if t.i >= prev]
            )
            # Validate: every span must have subject + verb
            for span_toks in candidate_spans:
                if not span_toks or not _span_has_subject_and_verb(span_toks):
                    valid_semicolon_split = False
                    break
            if valid_semicolon_split:
                semicolon_spans = candidate_spans

        # If semicolons produced valid splits, re-parse each span
        if semicolon_spans:
            for idx, span_toks in enumerate(semicolon_spans):
                if span_toks:
                    clause_text = " ".join(
                        t.text for t in span_toks
                    ).strip().rstrip(" ,;")
                    if clause_text:
                        clause_doc = frag_nlp(clause_text)
                        _patch_fragment_lemmas(clause_doc, span_toks)
                        conj_type = "additive" if idx > 0 else None
                        clauses.append((clause_doc, conj_type))
            continue  # skip conjunction splitting for this sentence

        # ---- Phase 1.5: comma splice detection ----
        # Pattern: ccomp child with its own nsubj, preceded by comma,
        # no subordinating conjunction. "I have a cat, they are rescues"
        # spaCy treats first clause as ccomp of second. Split at comma.
        _comma_split_done = False
        root_tok = None
        for tok in sent_tokens:
            if tok.dep_ == "ROOT":
                root_tok = tok
                break
        if root_tok:
            for child in root_tok.children:
                if (child.dep_ == "ccomp" and child.pos_ in ("VERB", "AUX")
                        and any(gc.dep_ in ("nsubj", "nsubjpass")
                                for gc in child.children)):
                    # Check: no subordinating conjunction (mark) on ccomp
                    has_mark = any(
                        gc.dep_ == "mark" for gc in child.children
                    )
                    if has_mark:
                        continue
                    # Find comma between ccomp subtree and ROOT
                    ccomp_end = max(t.i for t in child.subtree)
                    comma_idx = None
                    for t in sent_tokens:
                        if (t.text == "," and t.i > ccomp_end
                                and t.i < root_tok.i):
                            comma_idx = t.i
                            break
                        elif (t.text == "," and t.i < root_tok.i
                              and t.i > min(tc.i for tc in child.subtree)):
                            comma_idx = t.i
                            break
                    if comma_idx is None:
                        # Comma might be between ccomp's last token and root's nsubj
                        for t in sent_tokens:
                            if t.text == ",":
                                comma_idx = t.i
                                break
                    if comma_idx is not None:
                        # Split: left = ccomp subtree, right = rest
                        left_toks = [t for t in sent_tokens if t.i <= ccomp_end]
                        right_toks = [t for t in sent_tokens
                                      if t.i > comma_idx and t.text != ","]
                        if (left_toks and right_toks
                                and _span_has_subject_and_verb(left_toks)
                                and _span_has_subject_and_verb(right_toks)):
                            left_text = " ".join(
                                t.text for t in left_toks
                            ).strip().rstrip(" ,")
                            right_text = " ".join(
                                t.text for t in right_toks
                            ).strip()
                            if left_text and right_text:
                                left_doc = frag_nlp(left_text)
                                _patch_fragment_lemmas(left_doc, left_toks)
                                right_doc = frag_nlp(right_text)
                                _patch_fragment_lemmas(right_doc, right_toks)
                                clauses.append((left_doc, None))
                                clauses.append((right_doc, "additive"))
                                _comma_split_done = True
                                break

        if _comma_split_done:
            continue

        # ---- Phase 2: split on coordinating conjunctions (FANBOYS) ----
        # Each entry is (split_index, conjunction_type_string).
        split_points: List[Tuple[int, Optional[str]]] = []
        for tok in sent_tokens:
            if (tok.dep_ == "conj"
                    and tok.pos_ in ("VERB", "AUX")
                    and _head_chain_reaches_root(tok)):
                has_own_subject = any(
                    child.dep_ in ("nsubj", "nsubjpass")
                    for child in tok.children
                )
                if has_own_subject:
                    cc_tok = None
                    for child in tok.head.children:
                        if (child.dep_ == "cc"
                                and child.i > tok.head.i
                                and child.i < tok.i):
                            cc_tok = child
                            break
                    if cc_tok is None:
                        for child in tok.children:
                            if child.dep_ == "cc":
                                cc_tok = child
                                break

                    split_idx = cc_tok.i if cc_tok else tok.i
                    conj_lemma = (
                        cc_tok.lemma_.lower() if cc_tok else None
                    )
                    conj_type = _CONJUNCTION_TYPE_MAP.get(
                        conj_lemma, "additive",
                    ) if conj_lemma else None
                    split_points.append((split_idx, conj_type))

        if not split_points:
            clauses.append((sent.as_doc(), None))
        else:
            prev_start = sent.start
            clause_idx = 0
            for split_i, conj_type in sorted(
                split_points, key=lambda x: x[0],
            ):
                clause_tokens = [
                    t for t in sent_tokens
                    if t.i >= prev_start and t.i < split_i
                ]
                if clause_tokens:
                    clause_text = " ".join(
                        t.text for t in clause_tokens
                    ).strip().rstrip(" ,")
                    if clause_text:
                        clause_doc = frag_nlp(clause_text)
                        _patch_fragment_lemmas(clause_doc, clause_tokens)
                        ct = None if clause_idx == 0 else conj_type
                        clauses.append((clause_doc, ct))
                        clause_idx += 1
                prev_start = split_i + 1

            remaining = [t for t in sent_tokens if t.i >= prev_start]
            if remaining:
                clause_text = " ".join(
                    t.text for t in remaining
                ).strip()
                if clause_text:
                    clause_doc = frag_nlp(clause_text)
                    _patch_fragment_lemmas(clause_doc, remaining)
                    last_conj_type = split_points[-1][1] if split_points else None
                    clauses.append((clause_doc, last_conj_type))

    return clauses


# ---------------------------------------------------------------------------
# process() -- the ONE entry point for the WRITE path
# Spec: Text in -> classify -> extract traces -> derive triples
# ---------------------------------------------------------------------------

def process(text: str, speaker: Optional[str] = None, listener: str = "user") -> GrammarResult:
    """Main entry point.  Classify, extract traces, derive triples.

    Multi-sentence aware: each sentence classified and extracted independently.
    Compound sentences split into independent clauses.

    Args:
        text: Raw input text to process.
        speaker: Name of the person speaking (resolves "I" -> speaker).
        listener: Name of the person being addressed (resolves "you" -> listener).
                  Defaults to "user" for backward compatibility.

    Spec Part 2: extraction rules by sentence type.
    """
    _check_entry()
    nlp = _get_nlp()
    doc = nlp(text)

    clause_tuples = _split_compound_clauses(doc)

    decompositions: List[TraceDecomposition] = []
    all_triples: List[Triple] = []
    resolved_parts: List[str] = []
    primary_classification: Optional[UtteranceClassification] = None
    primary_sent_doc = None

    # Type 13 (free indirect speech): track perspective subject across
    # sentences.  When sentence N has a PERSON/PROPN nsubj + mental verb,
    # the next sentence may be narrated from that person's perspective.
    _perspective_subject: Optional[str] = None
    _perspective_gap: int = 0  # reset after 1 sentence gap

    for sent_doc, _conj_type in clause_tuples:
        sent_cls = classify_utterance(sent_doc, speaker)

        if primary_classification is None or (
            not primary_classification.is_storable and sent_cls.is_storable
        ):
            primary_classification = sent_cls
            primary_sent_doc = sent_doc

        resolved_parts.append(resolve_pronouns(sent_doc, speaker, listener=listener))
        sent_tense = detect_tense_aspect(sent_doc)

        # ============================================================
        # SINGLE PATH: every clause goes through the same pipeline.
        # Classification sets metadata (mood, storable), not code path.
        # ============================================================

        # Step 1: Tag question pre-processing (strip the tag)
        extraction_doc = sent_doc
        if sent_cls.subcategory == "tag_question":
            extraction_doc = _strip_tag_question(sent_doc)
            sent_tense = detect_tense_aspect(extraction_doc)

        # Step 2: Extract traces — ALWAYS, for ALL sentence types
        decomp = _extract_traces_from_sentence(
            extraction_doc, speaker, sent_tense, listener=listener,
        )
        decomp.utterance_type = sent_cls.utterance_type_id

        # Step 3: Set mood from classification
        if sent_cls.is_question:
            decomp.mood = "interrogative"
        elif sent_cls.is_command:
            decomp.mood = "imperative"
        elif sent_cls.subcategory == "tag_question":
            decomp.mood = "indicative"
        # else: mood from detect_mood() inside _extract_traces (already set)

        # Step 4: Conjunction semantics (Type 25)
        if _conj_type:
            decomp.extraction_rule = (
                f"{decomp.extraction_rule or 'trace'}_conj_{_conj_type}"
            )

        # Step 5: Free indirect speech perspective (Type 13)
        if _perspective_subject and _perspective_gap == 0:
            root_tok = _get_root(sent_doc)
            clause_subj_tok = None
            if root_tok:
                for child in root_tok.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        clause_subj_tok = child
                        break
            _apply_fis = (
                clause_subj_tok is None
                or (clause_subj_tok.pos_ not in ("PROPN", "PRON")
                    and clause_subj_tok.ent_type_ != "PERSON")
            )
            if _apply_fis:
                decomp.relational_subject = _perspective_subject
                decomp.subject = _perspective_subject
                decomp.extraction_rule = (
                    f"{decomp.extraction_rule or 'trace'}"
                    f"_free_indirect_speech"
                )

        # Step 6: Store or skip based on storability
        # Questions are non-storable (the question itself isn't a fact).
        # Everything else is stored.
        if not sent_cls.is_question:
            decompositions.append(decomp)
            all_triples.append(_derive_triple(decomp))

        # Step 6b: Compound predicate (Grammar ref p.829, lines 5895-5911)
        # One subject + multiple conj verbs → each verb is a separate fact.
        # "moved to Chicago, worked three years, relocated to Portland"
        # → 3 traces, all sharing ROOT's subject.
        # Also handles gerund coordination: "Running, reading, playing"
        # Walk the full conj chain (conj of conj of conj...).
        root_tok = _get_root(sent_doc)
        if root_tok and root_tok.pos_ in ("VERB", "AUX"):
            conj_queue = [c for c in root_tok.children
                          if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX", "NOUN")]
            frag_nlp = _get_nlp_fragment()
            while conj_queue:
                conj_child = conj_queue.pop(0)
                # Skip conj verbs with their own nsubj — those are
                # compound SENTENCES, already split by _split_compound_clauses.
                # Compound predicates share ROOT's subject (no own nsubj).
                _has_own_subj = any(
                    c.dep_ in ("nsubj", "nsubjpass")
                    for c in conj_child.children
                )
                if _has_own_subj:
                    continue
                # Add this node's conj children to the queue (chain)
                conj_queue.extend(
                    c for c in conj_child.children
                    if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX", "NOUN")
                )
                conj_subtree = sorted(conj_child.subtree, key=lambda t: t.i)
                # Filter out conj children's subtrees (they get their own trace)
                conj_child_indices = set()
                for cc in conj_child.children:
                    if cc.dep_ == "conj":
                        conj_child_indices |= {t.i for t in cc.subtree}
                conj_tokens = [t for t in conj_subtree
                               if t.i not in conj_child_indices
                               and t.dep_ != "cc" and t.pos_ != "PUNCT"]
                conj_text = " ".join(t.text for t in conj_tokens).strip()
                if conj_text:
                    conj_doc = frag_nlp(conj_text)
                    _patch_fragment_lemmas(conj_doc, conj_tokens)
                    conj_decomp = _extract_traces_from_sentence(
                        conj_doc, speaker, sent_tense, listener=listener,
                    )
                    # Inherit subject from main trace (compound predicate)
                    conj_decomp.subject = decomp.subject
                    conj_decomp.relational_subject = decomp.relational_subject
                    conj_decomp.extraction_rule = "trace_compound_predicate"
                    decompositions.append(conj_decomp)
                    all_triples.append(_derive_triple(conj_decomp))

        # Step 7: Extract imposed facts from subordinate constructions
        # ALWAYS runs for ALL sentence types (questions, commands, statements).
        existing_episodics = frozenset(
            d.episodic_fact for d in decompositions if d.episodic_fact
        )
        sub_imposed = _extract_imposed_facts(
            sent_doc, speaker, sent_tense, existing_episodics, listener=listener,
        )
        for imp_decomp in sub_imposed:
            decompositions.append(imp_decomp)
            all_triples.append(_derive_triple(imp_decomp))

        # --- Type 13: update perspective subject for next clause ---
        # If this sentence has a PERSON/PROPN nsubj + mental/perception
        # verb, store the subject as the perspective holder.
        # Detect mental/perception verbs via VerbClass + clausal complement.
        # No word list — uses WordNet hypernym closure (VerbClass.ABILITY
        # covers know/understand/believe; EXPERIENCE covers feel/sense).
        # Structural fallback: any verb with ccomp/xcomp that isn't SPEECH
        # or PLANNING frames a proposition → mental/perception.
        _cur_root = _get_root(sent_doc)
        _found_perspective = False
        if _cur_root and _cur_root.pos_ in ("VERB", "AUX"):
            _root_lemma = _cur_root.lemma_.lower()
            _cur_vc = classify_verb_class(_root_lemma)
            _has_clausal = any(
                c.dep_ in ("ccomp", "xcomp") for c in _cur_root.children
            )
            _is_mental = _cur_vc in (VerbClass.ABILITY, VerbClass.EXPERIENCE)
            if not _is_mental and _has_clausal:
                if _cur_vc not in (VerbClass.PLANNING,):
                    # SPEECH verbs are mental when used without a recipient
                    # dobj. "She believed it" = mental. "She told me" = speech.
                    if _cur_vc == VerbClass.SPEECH:
                        _has_recipient = any(
                            c.dep_ == "dobj" and c.pos_ == "PRON"
                            for c in _cur_root.children
                        )
                        if not _has_recipient:
                            _is_mental = True
                    else:
                        _is_mental = True
            if _is_mental:
                for child in _cur_root.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        if (child.pos_ in ("PROPN", "PRON")
                                or child.ent_type_ == "PERSON"):
                            _perspective_subject = child.text
                            _perspective_gap = 0
                            _found_perspective = True
                        break
        if not _found_perspective:
            # Decay: if we had a perspective subject but this sentence
            # didn't renew it, increment the gap counter.  Reset after
            # 1 sentence gap (don't carry indefinitely).
            if _perspective_subject is not None:
                _perspective_gap += 1
                if _perspective_gap >= 1:
                    _perspective_subject = None
                    _perspective_gap = 0

    # Fallback classification
    if primary_classification is None:
        primary_classification = classify_utterance(doc, speaker)
        primary_sent_doc = doc

    # Derive grammatical features from primary sentence
    # Tag questions: override mood to indicative (pragmatically assertions)
    if (primary_classification is not None
            and primary_classification.subcategory == "tag_question"):
        mood = "indicative"
    else:
        mood = detect_mood(primary_sent_doc)
    negated = detect_negation(primary_sent_doc)
    tense_aspect = detect_tense_aspect(primary_sent_doc)
    voice = detect_voice(primary_sent_doc)
    resolved_text = " ".join(resolved_parts)

    emotion = None
    if primary_classification.is_emotion:
        root = _get_root(primary_sent_doc)
        if root:
            for child in root.children:
                if child.dep_ in ("acomp", "attr", "oprd") and child.pos_ == "ADJ":
                    emotion = child.text.lower()
                    break

    # Post-processing: fix common extraction errors
    speaker_name = speaker if speaker else "user"
    for decomp in decompositions:
        # Fix 1: Resolve possessive pronouns in subjects
        # "My son" → "Melanie's son", "My friend" → "Caroline's friend"
        if decomp.subject:
            subj = decomp.subject
            if subj.startswith("My ") or subj.startswith("my "):
                decomp.subject = f"{speaker_name}'s {subj[3:]}"
            # Handle spaCy tokenization: "My hand - painted bowl" (spaces around hyphen)
            elif "My " in subj or "my " in subj:
                decomp.subject = subj.replace("My ", f"{speaker_name}'s ").replace("my ", f"{speaker_name}'s ")
            # "His/Her X" → speaker's X (in first-person narrative)
            elif subj.startswith("His ") or subj.startswith("his "):
                decomp.subject = f"{speaker_name}'s {subj[4:]}"
            elif subj.startswith("Her ") or subj.startswith("her "):
                decomp.subject = f"{speaker_name}'s {subj[4:]}"
            elif subj == "I" or subj == "i":
                decomp.subject = speaker_name
            # "Our" → speaker's
            elif subj.startswith("Our ") or subj.startswith("our "):
                decomp.subject = f"{speaker_name}'s {subj[4:]}"

        # Fix 2: Filter garbage/pronoun subjects
        if decomp.subject:
            _sl = decomp.subject.lower()
            # Direct pronouns → speaker
            if _sl in ("it", "this", "that", "there", "here", "they", "them",
                        "something", "nothing", "everything",
                        "he", "she", "we", "the kids", "the children",
                        "children", "two", "three", "four", "five",
                        "seven years", "which", "pattern",
                        "running", "running and pottery"):
                decomp.subject = speaker_name
            # Hyphenated compound subjects: "Self - care", "Self - acceptance"
            elif _sl.startswith("self") or _sl.startswith("self -"):
                decomp.subject = speaker_name
            # Definite noun phrases ("The necklace", "The book") → speaker
            elif _sl.startswith("the ") and len(_sl) < 50:
                decomp.subject = speaker_name
            # Possessive noun phrases not caught by Fix 1
            # "My hand-painted bowl" → "Melanie's hand-painted bowl"
            elif _sl.startswith(f"{speaker_name.lower()}'s "):
                pass  # Already resolved
            # Non-person proper nouns in subject of "be" copula
            # "Bach are my favorites" → speaker = Melanie
            elif decomp.predicate == "be" and decomp.object:
                _obj_l = decomp.object.lower()
                if "my " in _obj_l or "favorite" in _obj_l:
                    decomp.subject = speaker_name

        # Fix 3: Predicate cleanup
        if decomp.predicate:
            # 3a: Strip preposition suffix first
            # "time_at" → "time", "cup_with" → "cup", "go_in" → "go"
            if "_" in decomp.predicate:
                parts = decomp.predicate.split("_")
                last = parts[-1].lower()
                if last in ("with", "after", "before", "in", "on", "at",
                            "from", "to", "for", "by", "about", "of"):
                    decomp.predicate = "_".join(parts[:-1])

            # 3b: If predicate is a noun (substring of object) or a bare
            # preposition → recover the actual ROOT verb from source text
            pred_lower = decomp.predicate.lower()
            _need_root = False
            if pred_lower in ("with", "after", "before", "in", "on",
                              "at", "from", "to", "for"):
                _need_root = True
            elif decomp.object:
                obj_lower = decomp.object.lower()
                if pred_lower in obj_lower and len(pred_lower) > 2:
                    _need_root = True
            if _need_root and decomp.source_text:
                src_doc = _get_nlp()(decomp.source_text)
                for tok in src_doc:
                    if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
                        decomp.predicate = tok.lemma_
                        break

        # Fix 4: Pronoun/garbage object cleanup
        # "I am lactose intolerant" → object="I" should be "lactose intolerant"
        if decomp.object and decomp.object.lower() in (
            "i", "me", "it", "this", "that", "them", "us",
            "my son", "him", "her", "he",
        ) and decomp.source_text:
            src_doc = _get_nlp()(decomp.source_text)
            root_tok = _get_root(src_doc)
            if root_tok:
                # Try acomp (adjective complement): "am lactose intolerant"
                for child in root_tok.children:
                    if child.dep_ in ("acomp", "oprd"):
                        span = sorted(child.subtree, key=lambda t: t.i)
                        decomp.object = " ".join(t.text for t in span).strip()
                        break

        # Fix 5: Empty object for passive + prep constructions
        # "married for 5 years" → object should be "5 years"
        if not decomp.object and decomp.source_text:
            src_doc = _get_nlp()(decomp.source_text)
            root_tok = _get_root(src_doc)
            if root_tok:
                for child in root_tok.children:
                    if child.dep_ == "prep":
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                pobj_span = sorted(gc.subtree, key=lambda t: t.i)
                                decomp.object = " ".join(
                                    t.text for t in pobj_span
                                ).strip()
                                break
                        if decomp.object:
                            break

    # Rebuild triples from fixed decompositions
    all_triples = [_derive_triple(d) for d in decompositions]

    return GrammarResult(
        trace_decompositions=decompositions,
        triples=all_triples,
        classification=primary_classification,
        mood=mood,
        negated=negated,
        tense_aspect=tense_aspect,
        voice=voice,
        resolved_text=resolved_text,
        emotion=emotion,
    )


# ---------------------------------------------------------------------------
# extract_typed_triple -- architecture verifier compatibility
# ---------------------------------------------------------------------------

def extract_typed_triple(
    text: str,
    speaker: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Backward-compatibility wrapper for architecture verifier."""
    result = process(text, speaker=speaker)
    return [(t.subject, t.predicate, t.object) for t in result.triples]


# ---------------------------------------------------------------------------
# Query decomposition -- READ-PATH interface
# Spec: classify_query parses questions the same way process() parses statements
# ---------------------------------------------------------------------------

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
            # Strip possessive suffixes: "Caroline's" → "Caroline",
            # "Carolines" → "Caroline" (informal possessive without apostrophe)
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


# ===========================================================================
# Predicted Queries — absorbed from predicted_queries.py
# ===========================================================================
# Question generator for the DTCM write path.
#
# For each stored triple (subject, predicate, object) with entity types
# (subject_type, object_type), emit one predicted question per WH-type the
# triple can answer:
#
#     WHO   -- fires if subject_type or object_type is PERSON
#     WHAT  -- always fires
#     WHEN  -- fires if object_type is TIME, or predicate carries a
#              temporal stem (WordNet closure over time_period.n.01)
#     WHERE -- fires if object_type is LOCATION
#
# Each question is embedded via the shared MiniLM embedder (same path the
# edge_embedding uses) so the retrieval layer can compare question vectors
# with one cosine op.

import os as _pq_os

import numpy as _pq_np

from app.vector.embedder import embed_text as _pq_embed_text

# Lazy-loaded spaCy model for POS-based verb lemmatization (en_core_web_sm).
# Separate from grammar_engine's _nlp which uses en_core_web_md.
_pq_nlp = None


def _pq_get_nlp():
    global _pq_nlp
    if _pq_nlp is None:
        import spacy as _spacy
        _pq_nlp = _spacy.load("en_core_web_sm")
    return _pq_nlp


def _pq_lemmatize_predicate(pred_clean: str) -> str:
    """Lemmatize the head verb of a predicate phrase using spaCy POS.

    'works at' -> 'work at', 'assigned to' -> 'assign to'.
    Only the first VERB token is lemmatized; everything else passes through.
    """
    doc = _pq_get_nlp()(pred_clean)
    tokens = []
    verb_done = False
    for tok in doc:
        if not verb_done and tok.pos_ == "VERB":
            tokens.append(tok.lemma_)
            verb_done = True
        else:
            tokens.append(tok.text)
    return " ".join(tokens)

# Closed WH-type set for this foundation layer.
_PQ_WH_WHO = "WHO"
_PQ_WH_WHAT = "WHAT"
_PQ_WH_WHEN = "WHEN"
_PQ_WH_WHERE = "WHERE"

# Type constants mirrored from type_resolver (avoid circular import).
_PQ_PERSON = "PERSON"
_PQ_LOCATION = "LOCATION"
_PQ_TIME = "TIME"

# ---- Model selection ---------------------------------------------------

_pq_model = None
_pq_tokenizer = None
_pq_device = "cuda"
_pq_model_name = None  # "raya" or "flan" after init
_pq_probed = False


def _pq_raya_path() -> str:
    from pathlib import Path
    here = Path(__file__).resolve()
    return str(here.parents[2] / "models" / "raya-srl-220m-v4" / "final")


def _pq_load_model(pref: str):
    try:
        import torch
        from transformers import T5ForConditionalGeneration, AutoTokenizer
    except Exception as e:
        print(f"[predicted_queries] transformers unavailable: {e}")
        return None

    if pref == "raya":
        path = _pq_os.environ.get("NURA_T5_MODEL_PATH", _pq_raya_path())
    else:
        _pq_os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
        _pq_os.environ.setdefault("HF_HUB_OFFLINE", "1")
        path = "google/flan-t5-base"

    try:
        device = "cuda"
        tok = AutoTokenizer.from_pretrained(path)
        mdl = T5ForConditionalGeneration.from_pretrained(path).to(device).eval()
        return (mdl, tok, device)
    except Exception as e:
        print(f"[predicted_queries] {pref} load failed: {e}")
        return None


def _pq_generate_with(model, tokenizer, device, prompt: str, max_new: int = 32) -> str:
    import torch
    try:
        inp = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=max_new, num_beams=4, do_sample=False
            )
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def _pq_build_prompt(model_name: str, s: str, p: str, o: str, wh: str) -> str:
    pred_clean = p.replace("_", " ")
    if model_name == "raya":
        return f"generate questions: {s}|{pred_clean}|{o} -> {wh}"
    # flan
    return (
        f"Generate a {wh} question whose answer is the fact that "
        f"{s} {pred_clean} {o}. Question:"
    )


def _pq_is_parseable(text: str) -> bool:
    """A generated string is parseable if it is non-empty and starts with
    a WH-word or an auxiliary-verb question stem."""
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    first = t.split()[0].lower().strip(".,?!:;")
    # WH words form a closed grammatical class; this is not a curated
    # topical list, it is the English interrogative paradigm.
    return first in {
        "who", "what", "when", "where", "why", "which", "whose", "how",
        "is", "are", "was", "were", "do", "does", "did", "can", "will",
    }


# 5-triple probe corpus used at first-call to pick the winner.
_PQ_PROBE_TRIPLES = [
    ("Maya", "works_at", "Google", "PERSON", "ORG", _PQ_WH_WHO),
    ("Sam", "lives_in", "Ann_Arbor", "PERSON", "LOCATION", _PQ_WH_WHERE),
    ("meeting", "scheduled_on", "Tuesday", "EVENT", "TIME", _PQ_WH_WHEN),
    ("Emily", "married_to", "Jake", "PERSON", "PERSON", _PQ_WH_WHO),
    ("Raya", "is", "assistant", "GENERIC", "GENERIC", _PQ_WH_WHAT),
]


def _pq_ensure_model() -> bool:
    """Load and probe models. Prefer raya; fall back to flan-t5-base."""
    global _pq_model, _pq_tokenizer, _pq_device, _pq_model_name, _pq_probed
    if _pq_probed:
        return _pq_model is not None

    _pq_probed = True

    # Probe raya first.
    attempt = _pq_load_model("raya")
    if attempt is not None:
        mdl, tok, dev = attempt
        passes = 0
        for (s, p, o, st, ot, wh) in _PQ_PROBE_TRIPLES:
            out = _pq_generate_with(mdl, tok, dev, _pq_build_prompt("raya", s, p, o, wh))
            if _pq_is_parseable(out):
                passes += 1
        if passes >= 4:
            _pq_model, _pq_tokenizer, _pq_device, _pq_model_name = mdl, tok, dev, "raya"
            print(f"[predicted_queries] model=raya probe_pass={passes}/5")
            return True
        else:
            print(f"[predicted_queries] raya probe_pass={passes}/5 -> falling back")

    # Fall back to flan.
    attempt = _pq_load_model("flan")
    if attempt is None:
        print("[predicted_queries] no QG model available")
        return False
    mdl, tok, dev = attempt
    passes = 0
    for (s, p, o, st, ot, wh) in _PQ_PROBE_TRIPLES:
        out = _pq_generate_with(mdl, tok, dev, _pq_build_prompt("flan", s, p, o, wh))
        if _pq_is_parseable(out):
            passes += 1
    _pq_model, _pq_tokenizer, _pq_device, _pq_model_name = mdl, tok, dev, "flan"
    print(f"[predicted_queries] model=flan probe_pass={passes}/5")
    return True


# ---- WH applicability --------------------------------------------------

_PQ_PREDICATE_TEMPORAL_CACHE: dict = {}


def _pq_predicate_is_temporal(predicate: str) -> bool:
    """Return True if the predicate's head lemma's WordNet closure
    contains time_period.n.01 OR if it contains event.n.01 with a
    temporal sense. Structural via WordNet; no curated stem list."""
    if not predicate:
        return False
    key = predicate.lower()
    if key in _PQ_PREDICATE_TEMPORAL_CACHE:
        return _PQ_PREDICATE_TEMPORAL_CACHE[key]

    head = predicate.replace("_", " ").strip().split()[0].lower() if predicate else ""
    result = False
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
        for pos in (wn.VERB, wn.NOUN):
            for syn in wn.synsets(head, pos=pos):
                for path in syn.hypernym_paths():
                    for anc in path:
                        if anc.name() in ("time_period.n.01", "time.n.05", "time.n.01"):
                            result = True
                            break
                    if result:
                        break
                if result:
                    break
            if result:
                break
    except Exception:
        result = False

    _PQ_PREDICATE_TEMPORAL_CACHE[key] = result
    return result


def _pq_applicable_wh_types(
    subject_type: str, object_type: str, predicate: str
) -> List[str]:
    """Return WH-types the triple can answer. Structural, not calibrated."""
    out: List[str] = [_PQ_WH_WHAT]  # always
    if subject_type == _PQ_PERSON or object_type == _PQ_PERSON:
        out.append(_PQ_WH_WHO)
    if object_type == _PQ_TIME or _pq_predicate_is_temporal(predicate):
        out.append(_PQ_WH_WHEN)
    if object_type in (_PQ_LOCATION, "ORG"):
        out.append(_PQ_WH_WHERE)
    # Dedup preserving order.
    seen = set()
    ordered = []
    for w in out:
        if w not in seen:
            ordered.append(w)
            seen.add(w)
    return ordered


# ---- public API --------------------------------------------------------

def _pq_is_canonical_user(subject: str) -> bool:
    """True when the subject is the first-person canonical placeholder."""
    return (subject or "").strip().lower() in ("user", "i", "me", "myself")


def generate_predicted_queries(
    subject: str,
    predicate: str,
    object: str,
    subject_type: str = "GENERIC",
    object_type: str = "GENERIC",
) -> List[Tuple[str, '_pq_np.ndarray']]:
    """Return a list of (question_text, embedding) tuples -- 2..5 rows.

    One row per WH-type the triple can answer. If the QG model is
    unavailable we fall back to a structural question skeleton built
    from the triple (still not a curated list -- it's the SPO surface
    re-ordered into an interrogative form).

    When subject is the canonical 'user' placeholder, we additionally
    generate entity-agnostic variants that drop the subject token.
    This is additive — all original PQs are kept, the agnostic variants
    are appended. The rationale: 'user' is a variable standing for the
    speaker's real name, so PQs that rely on predicate+object rather
    than subject will cosine-match third-person queries that use the
    speaker's name.
    """
    subject = (subject or "").strip()
    predicate = (predicate or "").strip()
    object = (object or "").strip()
    if not subject or not predicate or not object:
        return []

    # Structural identity: 'user' IS a person (the speaker). Promote
    # to PERSON so WHO templates fire for first-person edges.
    effective_subject_type = subject_type
    if _pq_is_canonical_user(subject) and subject_type == "GENERIC":
        effective_subject_type = _PQ_PERSON

    wh_types = _pq_applicable_wh_types(effective_subject_type, object_type, predicate)

    # ── Grammar-aware question generation ──────────────────────────
    pred_clean = predicate.replace("_", " ")
    pred_lemma = _pq_lemmatize_predicate(pred_clean)
    is_user = _pq_is_canonical_user(subject)

    # Parse object to extract prepositional frame and detect temporals.
    nlp = _pq_get_nlp()
    _obj_doc = nlp(object)

    # Rule 3: Extract prepositional frame from object.
    _obj_frame = ""
    _obj_answer_np = object
    try:
        for _tok in _obj_doc:
            if _tok.dep_ == "prep" or (_tok.pos_ == "ADP" and _tok.i < len(_obj_doc) - 1):
                _obj_frame = _obj_doc[:_tok.i + 1].text
                _obj_answer_np = _obj_doc[_tok.i + 1:].text
                break
    except Exception:
        pass

    # Extract verb and prep from predicate
    _pred_parts = pred_clean.split()
    _verb_base = _pred_parts[0] if _pred_parts else pred_lemma
    _pred_prep = " ".join(_pred_parts[1:]) if len(_pred_parts) > 1 else ""

    # Rule 4: Detect temporal expressions in the object via NER.
    _has_temporal_in_obj = False
    try:
        for ent in _obj_doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                _has_temporal_in_obj = True
                break
    except Exception:
        pass
    if not _has_temporal_in_obj:
        _temporal_deps = {"npadvmod", "advmod", "prep"}
        for tok in _obj_doc:
            if tok.dep_ in _temporal_deps and tok.ent_type_ in ("DATE", "TIME"):
                _has_temporal_in_obj = True
                break

    # Add WHEN to wh_types if temporal detected in object
    if _has_temporal_in_obj and _PQ_WH_WHEN not in wh_types:
        wh_types.append(_PQ_WH_WHEN)

    _is_be = pred_lemma == "be"

    results: List[Tuple[str, '_pq_np.ndarray']] = []
    for wh in wh_types:
        # ── Build the primary question ────────────────────────────
        if wh == _PQ_WH_WHO:
            if effective_subject_type == _PQ_PERSON:
                question = f"Who {pred_clean} {object}?"
            elif _is_be and _obj_frame:
                question = f"Who is {subject} {_obj_frame}?"
            elif _is_be:
                question = f"Who is {subject}?"
            else:
                question = f"Who does {subject} {pred_lemma}?"

        elif wh == _PQ_WH_WHEN:
            if _is_be:
                question = f"When is {subject} {_obj_frame}?".strip()
                if not question.endswith("?"):
                    question += "?"
            else:
                question = f"When did {subject} {_verb_base}?"

        elif wh == _PQ_WH_WHERE:
            if _is_be:
                question = f"Where is {subject}?"
            elif _pred_prep:
                question = f"Where does {subject} {_verb_base}?"
            else:
                question = f"Where does {subject} {pred_lemma}?"

        else:  # WH_WHAT
            if _is_be and _obj_frame:
                question = f"What is {subject} {_obj_frame}?"
            elif _is_be:
                question = f"What is {subject}?"
            elif _pred_prep:
                question = f"What does {subject} {_verb_base} {_pred_prep}?"
            else:
                question = f"What does {subject} {pred_lemma}?"

        try:
            emb = _pq_embed_text(question)
            results.append((question, emb))
        except Exception:
            pass

        # ── Object-foregrounded variant ───────────────────────────
        if wh == _PQ_WH_WHO:
            obj_q = f"Who is {object}?"
        elif wh == _PQ_WH_WHERE:
            obj_q = f"Where is {object}?"
        elif wh == _PQ_WH_WHEN:
            obj_q = f"When is {object}?"
        else:
            obj_q = f"What is {object}?"
        try:
            results.append((obj_q, _pq_embed_text(obj_q)))
        except Exception:
            pass

        # ── Entity-agnostic variant for canonical-user edges ──────
        if is_user and wh != _PQ_WH_WHO:
            if wh == _PQ_WH_WHEN:
                ag_q = f"When was {object}?"
            elif wh == _PQ_WH_WHERE:
                ag_q = f"Where is {pred_clean} {object}?"
            else:
                ag_q = f"What about {pred_clean} {object}?"
            try:
                results.append((ag_q, _pq_embed_text(ag_q)))
            except Exception:
                pass

    # ── Rule 5: Possessive/kinship subjects ───────────────────────
    if "'s" in subject or subject.lower().startswith("my "):
        poss_q = f"Who is {subject}?"
        try:
            results.append((poss_q, _pq_embed_text(poss_q)))
        except Exception:
            pass

    return results


def pq_active_model_name():
    """Return which QG model is currently active ('raya', 'flan', or None)."""
    return _pq_model_name
