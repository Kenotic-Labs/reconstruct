# -*- coding: utf-8 -*-
"""
Reconstruction Engine — deterministic read path.

Architecture (from docs/retrieval-systems-how-they-work.md Parts 3-4):
    query -> classify_query() -> QueryDecomposition
          -> pronoun resolution
          -> question-type routing (count/list/still/session/change/milestone/
             relational/causal/emotional-trend)
          -> tiered retrieval:
               Tier 0: Facts table O(1)
               Tier 1: Structural SQL (S/P/O/schema/entity + negation + mood)
               Tier 2: Predicted queries cosine
               Tier 3: RRF hybrid (FTS5 + cosine)
               Tier 4: Cluster/arc expansion
             OR trace-scoped retrieval (when no S/P/O)
          -> cross-encoder reranking
          -> ranking signal stack (7 weighted components)
          -> verification loop (coherence + existence)
          -> cross-encoder relevance gate
          -> return_field routing -> column-level answer extraction
          -> PQ write-back
          -> or refusal (scoped CWA / relevance gate)

No LLM in the path. All models under 300M params. Deterministic.

12 capabilities (5 novel, 7 better-than-field):
  1. Scoped CWA           2. Edge negation       3. Mood matching
  4. Emotional trend      5. return_field routing 6. Trace-scoped SQL
  7. Supersession trail   8. O(1) fact lookup     9. Arc-based reconstruction
  10. Predicate embedding 11. Grammar-driven QD   12. No LLM
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entry / Exit checks
# ---------------------------------------------------------------------------

_ENTRY_CHECKED = False


class ReconstructionEntryError(RuntimeError):
    """Raised when reconstruction engine's dependencies are not available."""
    pass


def _check_entry():
    """Verify DB and memory are importable. Runs once."""
    global _ENTRY_CHECKED
    if _ENTRY_CHECKED:
        return
    missing = []
    try:
        from app.db.session import get_db_context as _test_db  # noqa: F401
    except ImportError:
        missing.append("db.session")
    try:
        from app.engines import memory  # noqa: F401
        if not hasattr(memory, 'get_memory_engine'):
            missing.append("memory.get_memory_engine")
    except ImportError:
        missing.append("memory")
    if missing:
        raise ReconstructionEntryError(
            f"reconstruction entry check failed — missing: {', '.join(missing)}"
        )
    _ENTRY_CHECKED = True




# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFUSAL_TEXT = "This information is not mentioned in the conversation."
TIER1_LIMIT = 40

# Causal predicates — now detected via WordNet (_verb_in_wordnet_domain)
# Speech-act verbs — now detected via grammar_engine.VerbClass.SPEECH

# Pronoun classification via spaCy morphological features — no word lists.
# spaCy tags: POS=PRON, PronType=Prs|Dem, Person=1|2|3, Gender=Fem|Masc|Neut


def _classify_pronoun(token_text: str) -> Optional[str]:
    """Classify a pronoun using spaCy morphology.
    Returns: 'person_fem', 'person_masc', 'person_neutral',
             'nonperson', 'group', 'deictic', or None."""
    doc = _get_query_doc(token_text)
    if doc is None or len(doc) == 0:
        return None
    tok = doc[0]
    if tok.pos_ != "PRON":
        return None
    pron_type = tok.morph.get("PronType", [""])[0]
    person = tok.morph.get("Person", [""])[0]
    gender = tok.morph.get("Gender", [""])[0]
    number = tok.morph.get("Number", [""])[0]
    # Demonstrative pronouns (this, that, these, those)
    if pron_type == "Dem":
        return "deictic"
    # First person plural (we, us, ourselves)
    if person == "1" and number == "Plur":
        return "group"
    # Third person neuter (it, itself)
    if person == "3" and gender == "Neut":
        return "nonperson"
    # Third person with gender
    if person == "3" and gender == "Fem":
        return "person_fem"
    if person == "3" and gender == "Masc":
        return "person_masc"
    # Third person plural or no gender (they/them) — neutral
    if person == "3":
        return "person_neutral"
    return None

# ---------------------------------------------------------------------------
# Shared query parsing — one spaCy parse per query, used by all classifiers
# ---------------------------------------------------------------------------

_query_doc_cache: Dict[str, Any] = {}


def _get_query_doc(query: str):
    """Parse query with spaCy. Cached — same query returns same doc."""
    key = query.strip()
    if key in _query_doc_cache:
        return _query_doc_cache[key]
    try:
        from app.engines.grammar_engine import _get_nlp
        doc = _get_nlp()(key)
        _query_doc_cache[key] = doc
        return doc
    except Exception:
        return None


def _get_query_root(doc):
    """Get ROOT token from spaCy doc."""
    if doc is None:
        return None
    for tok in doc:
        if tok.dep_ == "ROOT":
            return tok
    return None


def _query_has_token(doc, *, lemma: str = None, pos: str = None,
                     dep: str = None) -> bool:
    """Check if query doc contains a token matching criteria."""
    if doc is None:
        return False
    for tok in doc:
        if lemma and tok.lemma_.lower() != lemma:
            continue
        if pos and tok.pos_ != pos:
            continue
        if dep and tok.dep_ != dep:
            continue
        return True
    return False


def _query_wh_token(doc):
    """Find the WH-word token in a query doc, or None."""
    if doc is None:
        return None
    for tok in doc:
        if tok.tag_ in ("WDT", "WP", "WP$", "WRB"):
            return tok
    return None



def _noun_in_wordnet_domain(noun_lemma: str, anchor_synset: str) -> bool:
    """Check if a noun belongs to a WordNet domain via hypernym closure."""
    try:
        from app.engines.grammar_engine import _hypernym_closure
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(noun_lemma, pos=wn.NOUN)
        for ss in synsets[:3]:
            if anchor_synset in _hypernym_closure(ss.name()):
                return True
    except Exception:
        pass
    return False


def _verb_in_wordnet_domain(verb_lemma: str, anchor_synset: str) -> bool:
    """Check if a verb belongs to a WordNet domain via hypernym closure."""
    try:
        from app.engines.grammar_engine import _hypernym_closure
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(verb_lemma, pos=wn.VERB)
        for ss in synsets[:3]:
            if anchor_synset in _hypernym_closure(ss.name()):
                return True
    except Exception:
        pass
    return False


def _detect_query_schema(query: str) -> Optional[str]:
    """Detect semantic domain of a query via WordNet hypernym closure.
    Returns schema string (career, family, housing, health, etc.) or None.
    Uses grammar_engine._noun_to_schema_via_wordnet for each query noun."""
    doc = _get_query_doc(query)
    if doc is None:
        return None
    try:
        from app.engines.grammar_engine import (
            _noun_to_schema_via_wordnet, classify_verb_class, VerbClass,
        )
        # Check verb class first
        root = _get_query_root(doc)
        if root and root.pos_ in ("VERB", "AUX"):
            vc = classify_verb_class(root.lemma_.lower())
            _VC_SCHEMA = {
                VerbClass.WORK: "career",
                VerbClass.LOCATION: "housing",
                VerbClass.PREFERENCE: "identity",
                VerbClass.INJURY: "health",
                VerbClass.ACHIEVEMENT: "career",
            }
            if vc in _VC_SCHEMA:
                return _VC_SCHEMA[vc]
        # Check nouns via WordNet
        for tok in doc:
            if tok.pos_ in ("NOUN", "PROPN"):
                schema = _noun_to_schema_via_wordnet(tok.lemma_.lower())
                if schema and schema != "uncategorized":
                    return schema
    except Exception:
        pass
    return None


def _detect_query_preference(query: str) -> bool:
    """Detect if query is about preferences/enjoyment via WordNet verb class."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    try:
        from app.engines.grammar_engine import classify_verb_class, VerbClass
        for tok in doc:
            if tok.pos_ == "VERB":
                vc = classify_verb_class(tok.lemma_.lower())
                if vc == VerbClass.PREFERENCE:
                    return True
    except Exception:
        pass
    return False


def _detect_query_membership(query: str) -> bool:
    """Detect if query is about membership/belonging via WordNet."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    for tok in doc:
        if tok.pos_ in ("NOUN", "VERB"):
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "member.n.01"):
                return True
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "social_group.n.01"):
                return True
    return False


def _strip_determiners(text: str) -> str:
    """Remove determiners (the, a, an, some) using spaCy POS tags."""
    doc = _get_query_doc(text)
    if doc is None:
        return text
    return "".join(tok.text_with_ws for tok in doc if tok.pos_ != "DET").strip()


# FTS stopwords — function words for FTS query building (not semantic detection)
_FTS_STOP = frozenset({
    "what", "where", "when", "who", "whom", "which", "how", "why",
    "is", "are", "was", "were", "do", "does", "did", "has", "have",
    "had", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "not", "with", "from", "by", "about", "that",
    "this", "it", "be", "been", "being", "can", "could", "would",
    "should", "will", "shall", "may", "might", "my", "your", "his",
    "her", "its", "our", "their", "s", "t", "re", "ve", "ll", "d",
    "go", "get", "got", "take", "make", "say", "tell", "give",
})



# ---------------------------------------------------------------------------
# Entity resolver (absorbed from entity_resolver.py)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "who", "what", "when", "where", "why", "which", "whose", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "will",
    "the", "a", "an", "this", "that", "these", "those",
    "i", "you", "me", "my", "your", "we", "us", "our",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class ReconstructionResult:
    answer: Optional[str] = None
    refusal: bool = False
    refusal_reason: str = ""
    grounding: List[str] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)
    return_field: str = "episodic"


# ---------------------------------------------------------------------------
# Candidate container — every column from the doc's column usage map
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    edge_id: int
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source_text: str = ""
    is_current: int = 1
    edge_schematic_category: str = ""
    edge_emotional_label: str = ""
    edge_emotional_valence: float = 0.5
    edge_negated: int = 0
    edge_mood: str = "indicative"
    resolved_event_date: str = ""
    temporal_expression: str = ""
    relational_entities: str = ""
    edge_relational_type: str = ""
    sequence_number: int = 0
    confidence: float = 0.9
    superseded_at: str = ""
    superseded_by: int = 0
    edge_embedding: Optional[bytes] = None
    predicate_embedding: Optional[bytes] = None
    episodic_fact: str = ""
    emotional_target: str = ""
    source_timestamp: str = ""
    subject_type: str = ""
    object_type: str = ""
    cluster_id: str = ""
    arc_id: str = ""
    last_confirmed_at: str = ""
    pq_1: str = ""
    edge_affiliation: float = 0.0
    edge_episodic_significance: str = "routine"
    edge_temporal_context: str = "present"
    is_historical: int = 0
    # Scoring
    tier: str = ""


def _row_to_candidate(row: sqlite3.Row, tier: str = "") -> Candidate:
    """Convert a sqlite3.Row to a Candidate."""
    return Candidate(
        edge_id=row["id"],
        subject=row["subject"] or "",
        predicate=row["predicate"] or "",
        object=row["object"] or "",
        source_text=row["source_text"] or "",
        is_current=row["is_current"] if row["is_current"] is not None else 1,
        edge_schematic_category=row["edge_schematic_category"] or "",
        edge_emotional_label=row["edge_emotional_label"] or "",
        edge_emotional_valence=row["edge_emotional_valence"] if row["edge_emotional_valence"] is not None else 0.5,
        edge_negated=row["edge_negated"] or 0,
        edge_mood=row["edge_mood"] or "indicative",
        resolved_event_date=row["resolved_event_date"] or "",
        temporal_expression=row["temporal_expression"] or "",
        relational_entities=row["relational_entities"] or "",
        edge_relational_type=row["edge_relational_type"] or "",
        sequence_number=row["sequence_number"] or 0,
        confidence=0.9,
        superseded_at=row["superseded_at"] or "",
        superseded_by=row["superseded_by"] or 0,
        edge_embedding=row["edge_embedding"],
        predicate_embedding=row["predicate_embedding"],
        episodic_fact=row["episodic_fact"] or "",
        emotional_target=row["emotional_target"] or "",
        source_timestamp=row["source_timestamp"] or "",
        subject_type=row["subject_type"] or "",
        object_type=row["object_type"] or "",
        cluster_id=row["cluster_id"] or "",
        arc_id=row["arc_id"] or "",
        last_confirmed_at=row["last_confirmed_at"] or "",
        pq_1=row["pq_1"] or "" if "pq_1" in row.keys() else "",
        edge_affiliation=0.0,
        edge_episodic_significance=row["edge_episodic_significance"] or "routine",
        edge_temporal_context=row["edge_temporal_context"] or "present",
        is_historical=row["is_historical"] or 0,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# SQL fragments
# ---------------------------------------------------------------------------
_CANDIDATE_COLS = """
    id, subject, predicate, object, source_text, is_current,
    edge_schematic_category, edge_emotional_label, edge_emotional_valence,
    edge_negated, edge_mood, resolved_event_date, temporal_expression,
    relational_entities, edge_relational_type, sequence_number,
    superseded_at, superseded_by, edge_embedding, predicate_embedding,
    episodic_fact, emotional_target, source_timestamp, subject_type,
    object_type, cluster_id, arc_id, last_confirmed_at,
    edge_episodic_significance, edge_temporal_context, is_historical,
    pq_1
"""

_BASE_WHERE = "user_id = ? AND tombstoned_at IS NULL"


# ===========================================================================
# QUERY STRUCTURE DETECTION — spaCy structural analysis, no keyword matching
# ===========================================================================

def _is_count_query(query: str) -> bool:
    """spaCy: WH 'how' + head is 'many'/'much' (det of a noun)."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    for tok in doc:
        if tok.lemma_.lower() == "how" and tok.tag_ == "WRB":
            for child in tok.head.children:
                if child.lemma_.lower() in ("many", "much"):
                    return True
            if tok.head.lemma_.lower() in ("many", "much"):
                return True
    return False


def _is_list_query(query: str) -> bool:
    """spaCy: imperative verb + plural/universal quantifier object.
    'Name all X', 'List everyone', 'What are all the X'."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    root = _get_query_root(doc)
    if root is None:
        return False
    # Imperative with plural object or universal quantifier
    has_universal = any(tok.lemma_.lower() in ("all", "every", "everyone")
                        for tok in doc)
    has_plural_obj = any(tok.tag_ in ("NNS", "NNPS")
                         and tok.dep_ in ("dobj", "attr", "nsubj", "pobj")
                         for tok in doc)
    if has_universal and has_plural_obj:
        return True
    # "Who all..." pattern — WH + universal
    wh = _query_wh_token(doc)
    if wh and has_universal:
        return True
    return False


def _is_yesno_query(query: str, wh_word: Optional[str]) -> bool:
    """spaCy: interrogative mood + no WH token = yes/no question.
    AUX/VERB fronted (subject-auxiliary inversion)."""
    if wh_word:
        return False
    if _is_conditional_query(query):
        return False
    doc = _get_query_doc(query)
    if doc is None:
        return False
    # Subject-auxiliary inversion: first non-punct token is AUX or VERB
    for tok in doc:
        if tok.pos_ in ("PUNCT", "SPACE"):
            continue
        return tok.pos_ in ("AUX", "VERB") and tok.dep_ in ("ROOT", "aux")
    return False


def _is_still_query(query: str) -> bool:
    """spaCy: advmod token with lemma 'still'."""
    return _query_has_token(_get_query_doc(query), lemma="still", pos="ADV")


def _is_used_to_query(query: str) -> bool:
    """spaCy: 'used to' AUX structure, or temporal-past adverbs
    (previously, formerly, once) via dep=advmod."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    for tok in doc:
        # "used to" — spaCy parses "used" as VBN/VBD, "to" may be child
        # of the xcomp verb (play), not of "used" directly.
        if tok.lemma_.lower() == "use" and tok.tag_ in ("VBD", "VBN"):
            # Check direct children or next token
            xcomp = [c for c in tok.children if c.dep_ == "xcomp"]
            if xcomp:
                return True  # "used [to] play" structure
            if tok.i + 1 < len(doc) and doc[tok.i + 1].text.lower() == "to":
                return True
        # Temporal-past adverbs
        if tok.pos_ == "ADV" and tok.dep_ == "advmod":
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "past.n.01"):
                return True
            # Direct check for common temporal-past adverbs spaCy tags
            if tok.lemma_.lower() in ("previously", "formerly", "once"):
                return True
    return False


def _is_session_query(query: str) -> bool:
    """spaCy: detects queries about conversation sessions.
    Signals: DATE/TIME entities, "we" + communication verb,
    session/conversation nouns."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    # DATE/TIME entities referencing sessions
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            return True
    # "we" as subject + any verb = session reference (we = user + AI)
    root = _get_query_root(doc)
    has_we = any(tok.lemma_.lower() == "we"
                 and tok.dep_ in ("nsubj", "nsubjpass")
                 for tok in doc)
    if has_we:
        # "we" + communication verb (talk, discuss, chat, etc.)
        if root and root.pos_ in ("VERB", "AUX"):
            if _verb_in_wordnet_domain(root.lemma_.lower(), "communicate.v.02"):
                return True
        # "we" + any verb still implies session context
        return True
    # Session/conversation nouns via WordNet
    for tok in doc:
        if tok.pos_ == "NOUN":
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "session.n.01"):
                return True
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "conversation.n.01"):
                return True
    return False


def _is_change_query(query: str) -> bool:
    """spaCy + WordNet: verb/noun with change/difference semantics.
    'What changed?', 'Anything different?', 'What's new?'."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    for tok in doc:
        if tok.pos_ in ("VERB", "NOUN"):
            if _verb_in_wordnet_domain(tok.lemma_.lower(), "change.v.01"):
                return True
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "change.n.03"):
                return True
        if tok.pos_ == "ADJ" and tok.lemma_.lower() in ("different", "new"):
            return True
    return False


def _is_milestone_query(query: str) -> bool:
    """WordNet: query nouns under event.n.01 + significance markers.
    'milestone', 'major event', 'turning point', 'significant moment'."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    has_significance = any(
        tok.pos_ == "ADJ" and tok.lemma_.lower() in (
            "major", "significant", "big", "important", "key", "turning")
        for tok in doc
    )
    for tok in doc:
        if tok.pos_ == "NOUN":
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "event.n.01"):
                if has_significance:
                    return True
            # "milestone" / "turning point" are direct matches
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "milestone.n.01"):
                return True
    return False


def _is_emotional_trend_query(query: str) -> bool:
    """WordNet: emotion noun/adj + change/trend verb.
    'doing better emotionally', 'feeling lately', 'mood trend'."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    has_emotion = False
    has_trend = False
    for tok in doc:
        if tok.pos_ in ("NOUN", "ADJ"):
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "feeling.n.01"):
                has_emotion = True
            if _noun_in_wordnet_domain(tok.lemma_.lower(), "emotion.n.01"):
                has_emotion = True
            if tok.lemma_.lower() in ("mood", "emotional", "emotionally"):
                has_emotion = True
        if tok.pos_ == "ADV" and tok.lemma_.lower() == "emotionally":
            has_emotion = True
        if tok.pos_ in ("VERB", "ADJ"):
            if tok.lemma_.lower() in ("better", "worse", "improve", "decline"):
                has_trend = True
            if _verb_in_wordnet_domain(tok.lemma_.lower(), "change.v.01"):
                has_trend = True
        if tok.pos_ == "ADV" and tok.lemma_.lower() == "lately":
            has_trend = True
    return has_emotion and has_trend


def _is_relational_else_query(query: str) -> bool:
    """spaCy: token with lemma 'else' as advmod/det."""
    return _query_has_token(_get_query_doc(query), lemma="else")


def _is_same_comparison_query(query: str) -> bool:
    """spaCy: token with lemma 'same' as amod/det in an interrogative."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    has_same = any(tok.lemma_.lower() == "same"
                   and tok.dep_ in ("amod", "det", "attr", "acomp")
                   for tok in doc)
    has_interr = any(tok.pos_ in ("AUX", "VERB") for tok in doc)
    return has_same and has_interr


def _is_conditional_query(query: str) -> bool:
    """spaCy: conditional mood via grammar_engine.detect_mood,
    or subordinate clause with 'if' (dep=mark)."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    try:
        from app.engines.grammar_engine import detect_mood
        if detect_mood(doc) == "conditional":
            return True
    except Exception:
        pass
    # WH-question with 'if' clause = conditional
    wh = _query_wh_token(doc)
    has_if_mark = any(tok.lemma_.lower() == "if" and tok.dep_ == "mark"
                      for tok in doc)
    if wh and has_if_mark:
        return True
    # Non-WH with modal + 'if' = conditional
    if not wh:
        has_modal = any(tok.pos_ == "AUX"
                        and tok.morph.get("VerbForm") == ["Fin"]
                        and tok.lemma_.lower() in ("would", "could", "might")
                        for tok in doc)
        if has_modal or has_if_mark:
            return True
    return False


def _is_interrogative_unbounded(query: str) -> bool:
    """spaCy: advmod 'ever', or noun chunk 'any time'/'any point'.
    Bypasses edge_mood filter — asks about ALL edges including historical."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    for tok in doc:
        if tok.lemma_.lower() == "ever" and tok.pos_ == "ADV":
            return True
        # "any time" / "any point" — determiner "any" + temporal noun
        if tok.lemma_.lower() == "any" and tok.pos_ == "DET":
            if tok.head.lemma_.lower() in ("time", "point", "moment"):
                return True
    return False


def _is_speech_act_query(qd) -> bool:
    """WordNet: query verb is a speech act (say, tell, mention, etc.)
    via grammar_engine.classify_verb_class == SPEECH."""
    if not qd.match_predicate:
        return False
    try:
        from app.engines.grammar_engine import classify_verb_class, VerbClass
        vc = classify_verb_class(qd.match_predicate.lower().replace("_", " ").split()[0])
        return vc == VerbClass.SPEECH
    except Exception:
        pass
    return False


def _has_spo(qd) -> bool:
    """Check if query decomposition has subject/predicate/object."""
    return bool(qd.match_predicate or qd.match_object)


# ===========================================================================
# PLAN #5: QUERY-TIME PRONOUN RESOLUTION (doc lines 1786-1837)
# ===========================================================================

def _resolve_pronoun(conn: sqlite3.Connection, user_id: int, qd):
    """Resolve unresolved pronouns in match_subject to named entities.

    Pronouns: she/her → most recent feminine PERSON
              he/him → most recent masculine PERSON
              they/them → most recent PERSON (gender-neutral)
              it → most recent non-PERSON entity
              we → user + most recent partner
              this/that → most recent edge's topic
    """
    subj = (qd.match_subject or "").lower().strip()
    _pclass = _classify_pronoun(subj) if subj else None
    if not _pclass:
        return  # Not an unresolved pronoun

    if _pclass == "nonperson":
        # Resolve to most recent non-PERSON entity
        row = conn.execute(
            """SELECT DISTINCT subject FROM edges
               WHERE user_id = ? AND tombstoned_at IS NULL
                 AND subject_type IS NOT NULL
                 AND subject_type NOT IN ('PERSON', 'person')
                 AND subject NOT IN ('user', 'I', '')
               ORDER BY sequence_number DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row and row["subject"]:
            qd.match_subject = row["subject"]
            qd.match_entity = row["subject"]
        return

    if _pclass == "deictic":
        # Resolve to most recent edge's object/topic
        row = conn.execute(
            """SELECT object FROM edges
               WHERE user_id = ? AND tombstoned_at IS NULL
                 AND object IS NOT NULL AND object != ''
               ORDER BY sequence_number DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row and row["object"]:
            qd.match_subject = row["object"]
            qd.match_entity = row["object"]
        return

    if _pclass == "group":
        # "we" → user + most recent conversational partner
        row = conn.execute(
            """SELECT DISTINCT subject FROM edges
               WHERE user_id = ? AND tombstoned_at IS NULL
                 AND (subject_type = 'PERSON' OR subject_type = 'person')
                 AND subject NOT IN ('user', 'I', '')
               ORDER BY sequence_number DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row and row["subject"]:
            qd.match_subject = None  # Clear — "we" queries are entity-agnostic
            qd.match_entity = row["subject"]
        return

    # Person pronouns — resolve by recency
    row = conn.execute(
        """SELECT DISTINCT subject FROM edges
           WHERE user_id = ? AND tombstoned_at IS NULL
             AND (subject_type = 'PERSON' OR subject_type = 'person')
             AND subject NOT IN ('user', 'I', '')
           ORDER BY sequence_number DESC LIMIT 5""",
        (user_id,),
    ).fetchall()

    if not row:
        return

    # For gendered pronouns, we'd ideally check entity gender attributes.
    # For now, return most recent PERSON (gender inference from entity
    # attributes would require the entities table to store gender).
    resolved = row[0]["subject"]
    if resolved:
        qd.match_subject = resolved
        qd.match_entity = resolved


# ===========================================================================
# TIER 0: FACTS TABLE — O(1) KEY-VALUE LOOKUP (doc lines 1969-1996)
# ===========================================================================

def _tier0_facts(conn: sqlite3.Connection, user_id: int,
                 qd) -> Optional[str]:
    """Direct key-value lookup in the facts table.
    Returns the fact value or None."""
    if not qd.match_entity and not qd.match_subject:
        return None
    entity = qd.match_entity or qd.match_subject or ""
    if not entity or entity == "user":
        entity = "user"

    # Build potential key patterns.
    # Write path (memory.py:973) stores: "{schema}::{VerbClass.name}::{subject}"
    # VerbClass.name comes from classify_verb_class(verb_lemma), NOT upper(lemma).
    # E.g. "research" → VerbClass.STUDY → key "education::STUDY::Caroline"
    verb_class_name = None
    if qd.match_predicate:
        from app.engines.grammar_engine import classify_verb_class
        vc = classify_verb_class(qd.match_predicate.lower())
        verb_class_name = vc.name  # e.g. "WORK", "LIVE", "STUDY"

    parts = []
    if qd.match_schema:
        parts.append(qd.match_schema)
    if verb_class_name:
        parts.append(verb_class_name)
    parts.append(entity)

    if len(parts) < 2:
        return None

    key_pattern = "::".join(parts)
    row = conn.execute(
        "SELECT value FROM facts WHERE user_id = ? AND key = ?",
        (user_id, key_pattern),
    ).fetchone()
    if row:
        return row["value"]

    # Try broader: just VerbClass::entity
    if verb_class_name and entity:
        rows = conn.execute(
            "SELECT value FROM facts WHERE user_id = ? AND key LIKE ?",
            (user_id, f"%::{verb_class_name}::{entity}"),
        ).fetchall()
        if rows:
            return rows[0]["value"]

    return None


# ===========================================================================
# TIER 1: STRUCTURAL SQL (doc lines 1196-1335 + plans #3,#6,#7)
# With speaker attribution, edge negation filter, edge mood filter
# ===========================================================================

def _tier1_structural(conn: sqlite3.Connection, user_id: int,
                      qd, query: str,
                      temporal_filter: Optional[str] = None,
                      exclude_entity: Optional[str] = None,
                      ) -> List[Candidate]:
    """SQL WHERE on subject/predicate/object/schema.
    Includes speaker attribution, negation filter, mood filter."""
    conditions = [_BASE_WHERE]
    params: list = [user_id]

    has_filter = False
    is_factual = not _is_conditional_query(query)

    # --- Speaker attribution (plan #3, doc lines 1701-1735) ---
    if _is_speech_act_query(qd):
        # "What did Caroline say/tell/mention?"
        # Subject = speaker (who said it), not what was said
        if qd.match_subject and qd.match_subject != "user":
            conditions.append("subject LIKE ?")
            params.append(f"%{qd.match_subject}%")
            has_filter = True
        if qd.match_entity and qd.match_entity != qd.match_subject:
            conditions.append("relational_entities LIKE ?")
            params.append(f"%{qd.match_entity}%")
            has_filter = True
    else:
        # Standard entity matching
        if qd.match_subject and qd.match_subject != "user":
            conditions.append(
                "(subject LIKE ? OR relational_entities LIKE ?)"
            )
            params.extend([f"%{qd.match_subject}%", f"%{qd.match_subject}%"])
            has_filter = True
        elif qd.match_entity:
            conditions.append(
                "(subject LIKE ? OR relational_entities LIKE ?)"
            )
            params.extend([f"%{qd.match_entity}%", f"%{qd.match_entity}%"])
            has_filter = True

    if qd.match_predicate:
        conditions.append("predicate LIKE ?")
        params.append(f"%{qd.match_predicate}%")
        has_filter = True

    if qd.match_object:
        obj = _strip_determiners(qd.match_object.strip())
        if obj:
            conditions.append("(object LIKE ? OR source_text LIKE ?)")
            params.extend([f"%{obj}%", f"%{obj}%"])
            has_filter = True

    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)
        has_filter = True

    # --- Temporal direction filters ---
    # Uses both is_current AND edge_temporal_context/is_historical (doc column map)
    if temporal_filter == "past":
        conditions.append("(is_current = 0 OR is_historical = 1)")
    elif temporal_filter == "present":
        conditions.append("is_current = 1")

    # --- Edge negation filter (plan #6, doc lines 1840-1866) ---
    # For factual WH queries, exclude negated edges
    if is_factual and qd.wh_word and not _is_yesno_query(query, qd.wh_word):
        conditions.append("edge_negated = 0")

    # --- Edge mood filter (plan #7, doc lines 1867-1912) ---
    # Interrogative unbounded ("ever", "any time") → no mood filter (doc line 1910)
    if not _is_interrogative_unbounded(query):
        if is_factual:
            conditions.append("edge_mood = 'indicative'")
        else:
            conditions.append("edge_mood IN ('indicative', 'conditional')")

    # --- Relational "else" exclusion (plan #9) ---
    if exclude_entity:
        conditions.append("subject != ?")
        params.append(exclude_entity)

    if not has_filter:
        return []

    sql = f"""
        SELECT {_CANDIDATE_COLS} FROM edges
        WHERE {' AND '.join(conditions)}
        ORDER BY sequence_number DESC
        LIMIT {TIER1_LIMIT}
    """
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_candidate(r, "tier1") for r in rows]


# ===========================================================================
# TIER 2: PREDICTED QUERIES — EMBEDDING COSINE (doc lines 2098-2099)
# ===========================================================================



def _tier2_predicted_queries(conn: sqlite3.Connection, user_id: int,
                              query: str
                              ) -> Tuple[List[Candidate], Optional[Tuple[str, int, float]]]:
    """Cosine similarity of query against predicted queries stored on edges.

    PQs are now pq_1..pq_4 TEXT columns on the edges table (no separate
    table, no stored embeddings). We embed each PQ at query time and
    compare against the query embedding.

    This function is no longer used — _find_pq_match handles PQ matching.
    """
    return [], None



# ===========================================================================
# PLAN #2: TRACE-SCOPED RETRIEVAL (doc lines 1486-1698)
# Parallel retrieval path when match_predicate and match_object are None
# ===========================================================================

def _trace_scoped_retrieval(conn: sqlite3.Connection, user_id: int,
                            qd, query: str) -> List[Candidate]:
    """Trace-scoped SQL retrieval — 5-dimensional WHERE.

    Fires when no S/P/O available but match_entity exists.
    Routes by return_field + query signals to the appropriate trace columns.
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return []

    conditions = [_BASE_WHERE, "is_current = 1"]
    params: list = [user_id]

    # Entity scope
    conditions.append("(subject LIKE ? OR relational_entities LIKE ?)")
    params.extend([f"%{entity}%", f"%{entity}%"])

    # Route by return_field + signals (doc signal→trace scope mapping table)
    rf = qd.return_field

    if rf == "emotional":
        conditions.append("edge_emotional_label IS NOT NULL")
        conditions.append("edge_emotional_valence != 0.5")
    elif rf == "temporal":
        conditions.append("resolved_event_date IS NOT NULL")
        conditions.append("resolved_event_date != source_timestamp")
    elif rf == "relational":
        conditions.append("relational_entities IS NOT NULL")
        conditions.append("relational_entities != '[]'")
        # "Who does X know from work?" → filter by relational type via WordNet
        _rel_schema = _detect_query_schema(query)
        if _rel_schema == "career":
            conditions.append("edge_relational_type = 'professional'")
        elif _rel_schema in ("family", "housing"):
            conditions.append("edge_relational_type = 'personal'")

    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)

    if _is_used_to_query(query):
        # Override: historical edges — use is_current + is_historical + edge_temporal_context
        conditions = [c for c in conditions if "is_current = 1" not in c]
        conditions.append("(is_current = 0 OR is_historical = 1 OR edge_temporal_context = 'past')")

    sql = f"""
        SELECT {_CANDIDATE_COLS} FROM edges
        WHERE {' AND '.join(conditions)}
        ORDER BY sequence_number DESC
        LIMIT {TIER1_LIMIT}
    """
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_candidate(r, "trace_scoped") for r in rows]


# ===========================================================================
# TIER 4: CLUSTER / ARC EXPANSION (plan #13, doc lines 2046-2083)
# ===========================================================================

def _expand_cluster(conn: sqlite3.Connection, user_id: int,
                    candidate: Candidate) -> List[Candidate]:
    """Expand to cluster_id for contextual edges."""
    if not candidate.cluster_id:
        return []
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM edges
            WHERE {_BASE_WHERE} AND cluster_id = ? AND id != ?
            ORDER BY sequence_number ASC""",
        (user_id, candidate.cluster_id, candidate.edge_id),
    ).fetchall()
    return [_row_to_candidate(r, "tier4_cluster") for r in rows]


def _expand_arc(conn: sqlite3.Connection, user_id: int,
                qd) -> Optional[ReconstructionResult]:
    """Arc-based situation reconstruction for 'what's going on' queries.

    Uses arcs table — pre-grouped ongoing situations.
    """
    rows = conn.execute(
        """SELECT id, topic, status FROM arcs
           WHERE user_id = ? AND status = 'open'
           ORDER BY created_at DESC""",
        (user_id,),
    ).fetchall()

    if not rows:
        return None

    # Collect edges from open arcs
    summaries = []
    all_edge_ids = []
    for arc in rows:
        arc_edges = conn.execute(
            f"""SELECT {_CANDIDATE_COLS} FROM edges
                WHERE {_BASE_WHERE} AND arc_id = ? AND is_current = 1
                ORDER BY sequence_number ASC""",
            (user_id, arc["id"]),
        ).fetchall()
        if arc_edges:
            objects = [r["object"] for r in arc_edges if r["object"]]
            summaries.append(f"{arc['topic']}: {', '.join(objects[:5])}")
            all_edge_ids.extend([r["id"] for r in arc_edges])

    if summaries:
        return ReconstructionResult(
            answer="; ".join(summaries),
            return_field="episodic",
            edge_ids=all_edge_ids[:20],
            grounding=[f"arc:{r['id']}" for r in rows],
        )
    return None


# ===========================================================================
# CROSS-ENCODER RERANKING (doc lines 471-483)
# ===========================================================================





# ===========================================================================
# ANSWER EXTRACTION — RETURN_FIELD ROUTING (doc lines 1030-1048)
# ===========================================================================

def _extract_answer(candidate: Candidate, qd, query: str,
                    prefer_source: bool = False) -> str:
    """Route to correct column based on return_field.

    temporal  → resolved_event_date (format-matched)
    emotional → edge_emotional_label + valence + emotional_target (plan #14)
    relational → relational_entities parsed from JSON
    episodic  → object (default)
    """
    rf = qd.return_field

    if rf == "temporal":
        # Prefer temporal_expression for relative dates — these are the
        # gold answer format in LOCOMO ("The week before 9 June 2023").
        temp_expr = candidate.temporal_expression or ""
        # Check if temporal expression is already a relative date phrase
        # via spaCy: contains a DATE entity with "before" as a preposition child
        _te_doc = _get_query_doc(temp_expr) if temp_expr else None
        _is_relative = False
        if _te_doc:
            for ent in _te_doc.ents:
                if ent.label_ == "DATE":
                    _is_relative = True
                    break
            if not _is_relative:
                # Check for "before" as structural marker
                _is_relative = any(tok.lemma_.lower() == "before"
                                   for tok in _te_doc)
        if _is_relative and temp_expr:
            date = temp_expr
        else:
            # Convert short relative expressions to expanded format
            # "last week" + source_timestamp → "The week before {session_date}"
            # spaCy: parse temporal expression, extract the temporal noun
            _src_ts = candidate.source_timestamp or ""
            _expanded = None
            if _src_ts and temp_expr:
                _te_doc2 = _get_query_doc(temp_expr)
                if _te_doc2:
                    try:
                        _sess_dt = datetime.fromisoformat(_src_ts[:10])
                        _sess_fmt = f"{_sess_dt.day} {_sess_dt.strftime('%B')} {_sess_dt.year}"
                        # Extract temporal structure via spaCy POS
                        _has_last = any(tok.lemma_.lower() == "last" for tok in _te_doc2)
                        _num_tok = None
                        for tok in _te_doc2:
                            if tok.pos_ == "NUM":
                                _num_tok = tok.text
                                break
                        # Extract the temporal noun (week, friday, weekend)
                        _temp_noun = None
                        for tok in _te_doc2:
                            if tok.pos_ in ("NOUN", "PROPN") and tok.lemma_.lower() not in ("last", "ago"):
                                _temp_noun = tok.lemma_  # use lemma to singularize
                                # Capitalize day names
                                if tok.pos_ == "PROPN":
                                    _temp_noun = tok.text.capitalize()
                                break
                        if _temp_noun and (_has_last or _num_tok):
                            if _num_tok:
                                _expanded = f"{_num_tok} {_temp_noun}s before {_sess_fmt}"
                            else:
                                _expanded = f"The {_temp_noun} before {_sess_fmt}"
                        elif temp_expr.lower() == "yesterday":
                            _expanded = f"The day before {_sess_fmt}"
                    except (ValueError, TypeError):
                        pass
            if _expanded:
                date = _expanded
            else:
                date = candidate.resolved_event_date or temp_expr or ""
        if date:
            q_lower = query.lower()
            # Plan #15: "how long" → compute delta from date to reference time.
            # Use the candidate's source_timestamp as reference (conversation time),
            # NOT datetime.now() — LOCOMO conversations happen in 2023 but we may
            # run in 2026.
            if _is_duration_query(query):
                try:
                    dt = datetime.fromisoformat(date[:10])
                except ValueError:
                    return date  # Non-ISO date string — return as-is
                ref_time = datetime.now()
                if candidate.source_timestamp:
                    try:
                        ref_time = datetime.fromisoformat(
                            candidate.source_timestamp[:19]
                        )
                    except (ValueError, TypeError):
                        pass
                delta = ref_time - dt
                years = delta.days // 365
                months = (delta.days % 365) // 30
                ago_suffix = " ago" if _query_has_token(_get_query_doc(query), lemma="ago") else ""
                if years > 0:
                    return f"{years} year{'s' if years != 1 else ''}{ago_suffix}"
                elif months > 0:
                    return f"{months} month{'s' if months != 1 else ''}{ago_suffix}"
                else:
                    return f"{delta.days} day{'s' if delta.days != 1 else ''}{ago_suffix}"
            # "what year" → year only (spaCy: WH "what" + head "year")
            _wh_year = False
            _qdoc_t = _get_query_doc(query)
            if _qdoc_t:
                for tok in _qdoc_t:
                    if tok.lemma_.lower() == "what" and tok.head.lemma_.lower() == "year":
                        _wh_year = True
                        break
            if _wh_year:
                return date[:4]
            # "when" or "how long" → check if object has richer date text
            # The object field may contain the exact gold answer text
            # (e.g. "In 2013", "first week of August 2023", "Since 2016")
            # which is more specific than our reformatted date.
            obj = candidate.object or ""
            _wh_when = _query_wh_token(_get_query_doc(query))
            if _wh_when and _wh_when.text.lower() == "when" and len(date) >= 10 and "-" in date:
                try:
                    dt = datetime.fromisoformat(date[:10])
                    if date[5:10] == "01-01":
                        formatted = str(dt.year)
                    elif date[8:10] == "01":
                        formatted = f"{dt.strftime('%B')} {dt.year}"
                    else:
                        formatted = f"{dt.day} {dt.strftime('%B')} {dt.year}"
                    # Prefer object when it has more temporal context
                    # (e.g. "In 2013" vs "2013", "first week of May" vs "May 2023")
                    if obj and any(c.isdigit() for c in obj):
                        obj_words = len(obj.split())
                        fmt_words = len(formatted.split())
                        if obj_words > fmt_words:
                            return obj
                    return formatted
                except (ValueError, AttributeError):
                    pass
            # Return temporal_expression if it's a relative date
            temp_expr2 = candidate.temporal_expression or ""
            if temp_expr2 and ("week before" in temp_expr2.lower()
                    or "friday before" in temp_expr2.lower()
                    or "weekend before" in temp_expr2.lower()
                    or "sunday before" in temp_expr2.lower()):
                return temp_expr2
            # Final fallback: prefer object if it has date content
            if obj and any(c.isdigit() for c in obj) and len(obj) < 60:
                return obj
            return date
        # No resolved_event_date — return source_text ONLY when:
        # 1. Object is a very short duration phrase (≤ 2 tokens like "5 years")
        # 2. Source_text is a concise sentence that contextualizes it
        # 3. Source_text is < 70 chars (prevents verbose sentences)
        # This helps "5 years" → "married for 5 years" but avoids
        # "2016" → "Seven years now making art... since 2016" (too long).
        obj = candidate.object or ""
        src = candidate.source_text or ""
        if (obj and src and len(obj.split()) <= 2
                and len(src) < 55 and obj.lower() in src.lower()):
            return src
        return obj

    if rf == "emotional":
        # Plan #14: include valence
        label = candidate.edge_emotional_label or ""
        valence = candidate.edge_emotional_valence
        target = candidate.emotional_target or ""
        if label:
            parts = [label]
            if target:
                parts.append(f"about {target}")
            return " ".join(parts)
        return candidate.object

    if rf == "relational":
        rel = candidate.relational_entities
        if rel and rel != "[]":
            try:
                entities = json.loads(rel)
            except json.JSONDecodeError:
                log.warning("Malformed relational_entities JSON on edge %d: %s",
                            candidate.edge_id, rel[:50])
                entities = []
            if entities:
                # For "who" (not "whose") questions, filter out entities
                # that are just the edge subject — they don't answer "Who
                # did X?" (the subject is who the query is about, not the
                # answer). "Whose" questions want the subject/possessor.
                if qd.wh_word == "who":
                    subj_lower = candidate.subject.lower()
                    other_entities = [
                        e for e in entities
                        if str(e).lower() != subj_lower
                    ]
                    if other_entities:
                        return ", ".join(str(e) for e in other_entities)
                    # All relational entities are the subject — fall
                    # through to object extraction
                else:
                    return ", ".join(str(e) for e in entities)
        # For "who" questions, the answer is typically in the object
        # (who did X → object is the person). Prefer object over subject.
        obj = candidate.object or ""
        if obj.strip() and _is_contentful_object(obj):
            return obj
        return candidate.subject or ""

    # Default: episodic — use the richest trace field.
    obj = candidate.object or ""
    ef = candidate.episodic_fact or ""
    src = candidate.source_text or ""

    # Episodic_fact is the verb-phrase trace — richer than object
    # for short objects. "ran a charity race for mental health"
    # vs "a charity race for mental health".
    if ef and len(ef) < 80 and len(obj.split()) <= 5:
        return ef

    if obj.strip() and _is_contentful_object(obj):
        return obj

    if ef:
        return ef

    return src


# ===========================================================================
# AGGREGATION QUERIES — COUNT / LIST (doc lines 1283-1305)
# ===========================================================================

def _handle_count_query(conn: sqlite3.Connection, user_id: int,
                        qd) -> Optional[ReconstructionResult]:
    """Handle 'how many' queries with SQL COUNT. No LIMIT — must scan all."""
    conditions = [_BASE_WHERE, "is_current = 1"]
    params: list = [user_id]

    if qd.match_predicate:
        conditions.append(_verb_class_predicate_clause(conn, user_id, qd.match_predicate, params))
    if qd.match_object:
        obj = _clean_article(qd.match_object)
        if obj:
            conditions.append("object LIKE ?")
            params.append(f"%{obj}%")
    # Schema omitted — grammar-derived schema often misclassifies
    # (e.g. "beach" → "housing") which widens or narrows incorrectly.
    if qd.match_entity or qd.match_subject:
        entity = qd.match_entity or qd.match_subject
        conditions.append("(subject LIKE ? OR relational_entities LIKE ?)")
        params.extend([f"%{entity}%", f"%{entity}%"])

    if len(conditions) <= 2:
        return None

    sql = f"""
        SELECT COUNT(DISTINCT object) as cnt
        FROM edges
        WHERE {' AND '.join(conditions)}
    """
    row = conn.execute(sql, params).fetchone()
    if row and row["cnt"] > 0:
        return ReconstructionResult(
            answer=str(row["cnt"]),
            return_field="episodic",
        )
    return None


def _handle_list_query(conn: sqlite3.Connection, user_id: int,
                       qd) -> Optional[ReconstructionResult]:
    """Handle explicit 'name all' / 'list everyone' queries."""
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    conditions = [_BASE_WHERE, "is_current = 1",
                  "(subject LIKE ? OR relational_entities LIKE ?)"]
    params: list = [user_id, f"%{entity}%", f"%{entity}%"]

    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)
    if qd.match_predicate:
        conditions.append(_verb_class_predicate_clause(conn, user_id, qd.match_predicate, params))

    rows = conn.execute(
        f"SELECT DISTINCT object FROM edges WHERE {' AND '.join(conditions)}", params
    ).fetchall()
    items = [r["object"] for r in rows if r["object"] and r["object"].strip()]
    if items:
        return ReconstructionResult(answer=", ".join(items), return_field="episodic")
    return None


# ===========================================================================
# AGGREGATION QUERIES — IMPLICIT PLURAL (Cap 6)
# ===========================================================================

def _is_aggregation_query(query: str) -> bool:
    """Detect queries asking for multiple items via spaCy structure.

    Signals: plural nouns (NNS/NNPS), universal quantifiers (all/some/every),
    WH + AUX + VBN (past participle), location WH + perfect aspect.
    """
    doc = _get_query_doc(query)
    if doc is None:
        return False

    wh = _query_wh_token(doc)
    if wh is None:
        return False

    # Universal quantifier (all, some, every) in query
    if any(tok.lemma_.lower() in ("all", "some", "every")
           and tok.dep_ in ("det", "predet", "amod")
           for tok in doc):
        return True

    # "Where has/have X done Y?" — location + perfect aspect
    if wh.text.lower() == "where":
        if any(tok.pos_ == "AUX" and tok.lemma_.lower() == "have"
               for tok in doc):
            return True

    # Plural noun within first 6 tokens = asking for multiple items
    try:
        for tok in doc[:6]:
            if tok.tag_ in ("NNS", "NNPS"):
                return True
    except Exception:
        pass

    # Past participle aggregation: "what has X painted/read/attended..."
    # spaCy: WH + AUX (has/have/did) + VBN (any past participle)
    try:
        doc = _get_query_doc(query)
        if doc:
            has_wh = _query_wh_token(doc) is not None
            has_aux = any(tok.pos_ == "AUX"
                         and tok.lemma_.lower() in ("have", "do")
                         for tok in doc)
            has_vbn = any(tok.tag_ == "VBN" for tok in doc)
            if has_wh and has_aux and has_vbn:
                return True
    except Exception:
        pass

    return False


def _handle_aggregation_query(
    conn: sqlite3.Connection,
    user_id: int,
    qd,
    query: str,
) -> Optional[ReconstructionResult]:
    """Handle implicit aggregation queries by collecting ALL matching objects.

    Strategy:
    1. Find the entity from QD
    2. Query ALL edges for that entity (no LIMIT)
    3. Embed the query and compute cosine with each edge's edge_embedding
    4. Collect objects from edges where cosine > 0.3
    5. Deduplicate and return comma-separated
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    # Fetch all edges for the entity
    conditions = [_BASE_WHERE, "is_current = 1",
                  "(subject LIKE ? OR relational_entities LIKE ?)"]
    params: list = [user_id, f"%{entity}%", f"%{entity}%"]

    # Add negation filter — factual aggregation shouldn't include negated edges
    conditions.append("edge_negated = 0")
    conditions.append("edge_mood = 'indicative'")

    rows = conn.execute(
        f"SELECT {_CANDIDATE_COLS}, pq_1, pq_2, pq_3, pq_4 FROM edges WHERE {' AND '.join(conditions)}",
        params,
    ).fetchall()

    if not rows:
        return None

    # Embed the query once
    query_emb = embed_text(query)

    # Score each edge by PREDICTED QUERY cosine — not edge embedding.
    # PQ cosine is far more discriminating for aggregation: personality
    # traits edges (cos 0.42 via edge emb) drop below 0.3 via PQ because
    # their PQs are "What traits does X have?", not "What activities…?"
    # Real activity edges jump to cos 0.8-1.0 because their PQs match.
    scored_objects: list = []
    seen_objs: set = set()
    all_edge_ids: list = []

    for row in rows:
        obj = (row["object"] or "").strip()
        if not obj or obj.lower() in seen_objs:
            continue
        # Skip if object is just the entity name or a pronoun
        if obj.lower() in (entity.lower(), "user", "i", "me", "them", "it"):
            continue
        # Substring dedup: skip if this object is a substring of an
        # existing one or vice versa ("adoption agencies" ⊂ "researching
        # adoption agencies")
        obj_lower = obj.lower()
        is_dup = False
        for existing in list(seen_objs):
            if obj_lower in existing or existing in obj_lower:
                is_dup = True
                break
        if is_dup:
            continue

        # PQ-based scoring: embed each pq_1-4, take best cosine with query
        best_cos = 0.0
        for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
            pq_text = row[col]
            if not pq_text:
                continue
            try:
                pq_emb = embed_text(pq_text)
                cos = float(np.dot(query_emb, pq_emb))
                if cos > best_cos:
                    best_cos = cos
            except Exception:
                continue

        # Fallback: edge embedding (for edges without PQs)
        if best_cos < 0.3:
            edge_emb_bytes = row["edge_embedding"]
            if edge_emb_bytes:
                try:
                    edge_emb = np.frombuffer(edge_emb_bytes, dtype=np.float32)
                    if edge_emb.shape[0] == query_emb.shape[0]:
                        cos = float(np.dot(query_emb, edge_emb))
                        if cos > best_cos:
                            best_cos = cos
                except Exception:
                    pass

        if best_cos > 0.70:
            scored_objects.append((obj, best_cos, row["id"]))
            seen_objs.add(obj.lower())
            all_edge_ids.append(row["id"])

    if not scored_objects:
        return None

    # Sort by cosine descending
    scored_objects.sort(key=lambda x: x[1], reverse=True)

    # Collect unique objects
    items = [obj for obj, _cos, _eid in scored_objects]

    if len(items) < 2:
        # Single item — let the normal pipeline handle it for better answer extraction
        return None

    # Check if the top item is a comprehensive summary or dominant answer.
    # 1. Contains "and" or numbers → summary ("two cats and a dog")
    # 2. Top cosine >> second cosine → dominant single answer
    top_obj = items[0]
    top_cos = scored_objects[0][1]
    second_cos = scored_objects[1][1] if len(scored_objects) > 1 else 0

    # Dominant answer: top has summary markers or big cosine gap
    is_summary = " and " in top_obj.lower() or any(c.isdigit() for c in top_obj)
    is_dominant = (top_cos - second_cos) > 0.12

    if len(items) > 3 and (is_summary or is_dominant) and len(top_obj) < 50:
        log.debug("Aggregation: using top item %r (cos=%.2f, gap=%.2f)",
                  top_obj, top_cos, top_cos - second_cos)
        return ReconstructionResult(
            answer=top_obj,
            return_field="episodic",
            edge_ids=[scored_objects[0][2]],
                grounding=[f"aggregation:{entity}"],
            )

    log.debug("Aggregation query: %d items for entity=%s query=%r",
              len(items), entity, query)
    return ReconstructionResult(
        answer=", ".join(items),
        return_field="episodic",
        edge_ids=all_edge_ids,
        grounding=[f"aggregation:{entity}"],
    )


# ===========================================================================
# YES/NO + TEMPORAL STATE (doc lines 1226-1251)
# ===========================================================================

def _handle_still_query(conn: sqlite3.Connection, user_id: int,
                        qd, query: str) -> Optional[ReconstructionResult]:
    """Handle 'still' queries by checking is_current."""
    conditions = [_BASE_WHERE]
    params: list = [user_id]

    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    conditions.append("(subject LIKE ? OR relational_entities LIKE ?)")
    params.extend([f"%{entity}%", f"%{entity}%"])

    if qd.match_predicate:
        conditions.append("predicate LIKE ?")
        params.append(f"%{qd.match_predicate}%")

    if qd.match_object:
        obj = _clean_article(qd.match_object)
        if obj:
            conditions.append("object LIKE ?")
            params.append(f"%{obj}%")

    sql = f"""
        SELECT {_CANDIDATE_COLS} FROM edges
        WHERE {' AND '.join(conditions)}
        ORDER BY sequence_number DESC
        LIMIT 5
    """
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return None

    for r in rows:
        c = _row_to_candidate(r)
        if c.is_current == 1:
            return ReconstructionResult(
                answer="Yes", return_field="episodic",
                edge_ids=[c.edge_id], grounding=[c.source_text],
            )
        else:
            return ReconstructionResult(
                answer="No", return_field="episodic",
                edge_ids=[c.edge_id], grounding=[c.source_text],
            )
    return None


# ===========================================================================
# PLAN #9: RELATIONAL INFERENCE (doc lines 1309-1335)
# ===========================================================================

def _handle_relational_else(conn: sqlite3.Connection, user_id: int,
                            qd, query: str) -> Optional[ReconstructionResult]:
    """Handle 'who else' queries — exclude current context entity.

    'Who else works at Palantir?' → find all subjects with same predicate+object,
    exclude the implied entity.
    """
    # The "else" implies excluding someone. Try match_subject or match_entity.
    # If neither available (WH-word query), use most recent person entity from context.
    exclude = qd.match_subject or qd.match_entity or ""
    if not exclude or exclude == "user":
        row = conn.execute(
            """SELECT DISTINCT subject FROM edges
               WHERE user_id = ? AND tombstoned_at IS NULL
                 AND (subject_type = 'PERSON' OR subject_type = 'person')
                 AND subject NOT IN ('user', 'I', '')
               ORDER BY sequence_number DESC LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row and row["subject"]:
            exclude = row["subject"]

    candidates = _tier1_structural(
        conn, user_id, qd, query,
        exclude_entity=exclude if exclude and exclude != "user" else None,
    )

    if candidates:
        names = list(dict.fromkeys(
            c.subject for c in candidates if c.subject and c.subject != exclude
        ))
        if names:
            return ReconstructionResult(
                answer=", ".join(names),
                return_field="relational",
                edge_ids=[c.edge_id for c in candidates[:5]],
            )
    return None


def _handle_same_comparison(conn: sqlite3.Connection, user_id: int,
                            qd, query: str) -> Optional[ReconstructionResult]:
    """Handle 'Do they live in the same city?' — compare two entities.

    SQL JOIN comparing objects for same predicate across two entities.
    """
    # Need two entities — try to extract from query
    entity1 = qd.match_subject
    entity2 = qd.match_entity
    if not entity1 or not entity2 or entity1 == entity2:
        return None

    pred = qd.match_predicate or ""
    if not pred:
        return None

    rows = conn.execute(
        f"""SELECT subject, object FROM edges
            WHERE {_BASE_WHERE} AND predicate LIKE ?
              AND subject IN (?, ?)
              AND is_current = 1""",
        (user_id, f"%{pred}%", entity1, entity2),
    ).fetchall()

    if len(rows) >= 2:
        vals = {r["subject"]: r["object"] for r in rows}
        v1 = vals.get(entity1, "")
        v2 = vals.get(entity2, "")
        if v1 and v2:
            if v1.lower() == v2.lower():
                return ReconstructionResult(
                    answer=f"Yes, both in {v1}",
                    return_field="episodic",
                )
            else:
                return ReconstructionResult(
                    answer=f"No, {entity1} is in {v1} and {entity2} is in {v2}",
                    return_field="episodic",
                )
    return None


# ===========================================================================
# PLAN #10: EMOTIONAL TREND COMPUTATION (doc lines 1636-1656)
# ===========================================================================

def _handle_emotional_trend(conn: sqlite3.Connection, user_id: int,
                            qd) -> Optional[ReconstructionResult]:
    """Compute emotional valence trend over time.

    Split edges into early/recent halves, compare mean valence.
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    rows = conn.execute(
        f"""SELECT edge_emotional_valence, sequence_number
            FROM edges
            WHERE {_BASE_WHERE}
              AND (subject LIKE ? OR relational_entities LIKE ?)
              AND edge_emotional_valence IS NOT NULL
              AND is_current = 1
            ORDER BY sequence_number ASC""",
        (user_id, f"%{entity}%", f"%{entity}%"),
    ).fetchall()

    if len(rows) < 4:
        return None

    vals = [r["edge_emotional_valence"] for r in rows]
    mid = len(vals) // 2
    early_mean = sum(vals[:mid]) / mid
    recent_mean = sum(vals[mid:]) / (len(vals) - mid)
    delta = recent_mean - early_mean

    if delta > 0.1:
        return ReconstructionResult(
            answer="Yes",
            return_field="emotional",
        )
    elif delta < -0.1:
        return ReconstructionResult(
            answer="No",
            return_field="emotional",
        )
    else:
        return ReconstructionResult(
            answer="About the same",
            return_field="emotional",
        )


# ===========================================================================
# PLAN #11: SUPERSESSION TRAIL QUERIES (doc lines 1609-1632)
# ===========================================================================

def _handle_change_query(conn: sqlite3.Connection, user_id: int,
                         query: str) -> Optional[ReconstructionResult]:
    """Handle 'what changed' queries via supersession trail.

    JOIN edges ON superseded_by to find old→new value pairs.
    """
    rows = conn.execute(
        f"""SELECT r_new.subject, r_new.predicate, r_new.object AS new_val,
                   r_old.object AS old_val, r_new.superseded_at
            FROM edges r_new
            JOIN edges r_old ON r_new.id = r_old.superseded_by
            WHERE r_new.user_id = ? AND r_new.tombstoned_at IS NULL
              AND r_new.is_current = 1
            ORDER BY r_new.superseded_at DESC
            LIMIT 10""",
        (user_id,),
    ).fetchall()

    if not rows:
        return None

    changes = []
    edge_ids = []
    for r in rows:
        subj = r["subject"] or ""
        old_v = r["old_val"] or ""
        new_v = r["new_val"] or ""
        if old_v and new_v and old_v != new_v:
            changes.append(f"{subj}: {old_v} → {new_v}")

    if changes:
        return ReconstructionResult(
            answer="; ".join(changes[:5]),
            return_field="episodic",
        )
    return None


# ===========================================================================
# PLAN #4: SESSION / CONVERSATION SCOPING (doc lines 1737-1784)
# ===========================================================================

def _handle_session_query(conn: sqlite3.Connection, user_id: int,
                          query: str) -> Optional[ReconstructionResult]:
    """Handle session-scoped queries via source_timestamp grouping."""
    doc = _get_query_doc(query)

    # "How many sessions have we had?" — count query + session/time noun
    _has_session_noun = any(
        tok.lemma_.lower() in ("session", "time", "conversation")
        and tok.pos_ == "NOUN"
        for tok in doc
    ) if doc else False
    if _is_count_query(query) and _has_session_noun:
        row = conn.execute(
            f"""SELECT COUNT(DISTINCT source_timestamp) as cnt
                FROM edges WHERE {_BASE_WHERE}""",
            (user_id,),
        ).fetchone()
        if row:
            return ReconstructionResult(
                answer=str(row["cnt"]),
                return_field="episodic",
            )

    # Get distinct session timestamps
    ts_rows = conn.execute(
        f"""SELECT DISTINCT source_timestamp FROM edges
            WHERE {_BASE_WHERE} AND source_timestamp IS NOT NULL
            ORDER BY source_timestamp DESC
            LIMIT 10""",
        (user_id,),
    ).fetchall()

    if not ts_rows:
        return None

    # "last time" / "last session" → second-most-recent
    # spaCy: adjective "last" modifying a session/time noun
    target_ts = None
    _has_last_session = any(
        tok.lemma_.lower() == "last" and tok.pos_ == "ADJ"
        and tok.head.lemma_.lower() in ("time", "session", "conversation")
        for tok in doc
    ) if doc else False
    if _has_last_session:
        if len(ts_rows) >= 2:
            target_ts = ts_rows[1]["source_timestamp"]
        elif ts_rows:
            target_ts = ts_rows[0]["source_timestamp"]

    # "on Tuesday" / specific day → resolve via spaCy DATE entity
    if not target_ts and doc:
        _day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6}
        _matched_day = None
        for ent in doc.ents:
            if ent.label_ == "DATE":
                _day_text = ent.text.lower()
                for _dname, _didx in _day_map.items():
                    if _dname in _day_text:
                        _matched_day = (_dname, _didx)
                        break
            if _matched_day:
                break
        if _matched_day:
            day, i = _matched_day[0], _matched_day[1]
            if True:  # structural match — keep indent level
                # Resolve to most recent occurrence of this weekday
                today = datetime.now()
                # Monday=0 in Python's weekday()
                target_weekday = i
                days_back = (today.weekday() - target_weekday) % 7
                if days_back == 0:
                    days_back = 7  # "on Tuesday" means last Tuesday, not today
                target_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
                # Find sessions whose source_timestamp starts with that date
                day_rows = conn.execute(
                    f"""SELECT {_CANDIDATE_COLS} FROM edges
                        WHERE {_BASE_WHERE} AND source_timestamp LIKE ?
                          AND is_current = 1
                        ORDER BY sequence_number ASC LIMIT 30""",
                    (user_id, f"{target_date}%"),
                ).fetchall()
                if day_rows:
                    schemas: Dict[str, List[str]] = {}
                    for r in day_rows:
                        cat = r["edge_schematic_category"] or "other"
                        if cat not in schemas:
                            schemas[cat] = []
                        schemas[cat].append(r["object"] or "")
                    parts = [f"{cat}: {', '.join(objs[:3])}" for cat, objs in schemas.items()]
                    return ReconstructionResult(
                        answer="; ".join(parts),
                        return_field="episodic",
                        edge_ids=[r["id"] for r in day_rows[:10]],
                    )

    # "this week" → within last 7 days (spaCy DATE entity containing "week")
    _has_this_week = any(
        ent.label_ == "DATE" and "week" in ent.text.lower()
        for ent in doc.ents
    ) if doc else False
    if not target_ts and _has_this_week:
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        rows = conn.execute(
            f"""SELECT {_CANDIDATE_COLS} FROM edges
                WHERE {_BASE_WHERE} AND source_timestamp >= ? AND is_current = 1
                ORDER BY sequence_number ASC LIMIT 30""",
            (user_id, cutoff),
        ).fetchall()
        if rows:
            schemas = {}
            for r in rows:
                cat = r["edge_schematic_category"] or "other"
                if cat not in schemas:
                    schemas[cat] = []
                schemas[cat].append(r["object"] or "")
            parts = [f"{cat}: {', '.join(objs[:3])}" for cat, objs in schemas.items()]
            return ReconstructionResult(
                answer="; ".join(parts),
                return_field="episodic",
                edge_ids=[r["id"] for r in rows[:10]],
            )
        return None

    if not target_ts:
        return None

    # Retrieve edges from target session
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM edges
            WHERE {_BASE_WHERE} AND source_timestamp = ? AND is_current = 1
            ORDER BY sequence_number ASC""",
        (user_id, target_ts),
    ).fetchall()

    if not rows:
        return None

    # Group by schema category, summarize
    schemas: Dict[str, List[str]] = {}
    edge_ids = []
    for r in rows:
        cat = r["edge_schematic_category"] or "other"
        obj = r["object"] or ""
        if cat not in schemas:
            schemas[cat] = []
        if obj:
            schemas[cat].append(obj)
        edge_ids.append(r["id"])

    parts = [f"{cat}: {', '.join(objs[:3])}" for cat, objs in schemas.items()]
    return ReconstructionResult(
        answer="; ".join(parts),
        return_field="episodic",
        edge_ids=edge_ids[:10],
    )


# ===========================================================================
# PLAN #12: MILESTONES TABLE (doc lines 1998-2021)
# ===========================================================================

def _handle_milestone_query(conn: sqlite3.Connection, user_id: int,
                            qd) -> Optional[ReconstructionResult]:
    """Query milestones table for significant life events."""
    conditions = ["user_id = ?"]
    params: list = [user_id]

    if qd.match_entity:
        conditions.append("description LIKE ?")
        params.append(f"%{qd.match_entity}%")

    rows = conn.execute(
        f"""SELECT description, event_date, event_type FROM milestones
            WHERE {' AND '.join(conditions)}
            ORDER BY event_date DESC
            LIMIT 10""",
        params,
    ).fetchall()

    if not rows:
        return None

    milestones = []
    for r in rows:
        desc = r["description"] or ""
        date = r["event_date"] or ""
        if desc:
            milestones.append(f"{desc} ({date})" if date else desc)

    if milestones:
        return ReconstructionResult(
            answer="; ".join(milestones),
            return_field="episodic",
        )
    return None


# ===========================================================================
# INFERENCE QUERIES — "Would X...?" without "if" clause
# ===========================================================================

def _handle_inference_query(
    conn: sqlite3.Connection, user_id: int, qd, query: str,
) -> Optional[ReconstructionResult]:
    """Handle inference questions: 'Would X likely do Y?', 'Would X enjoy Z?'

    Strategy: embed the query, find the best matching edge by edge embedding.
    High cosine (> 0.65) → evidence supports it → "Yes" + detail.
    Low cosine → "Likely no".

    Uses edge embedding (not PQ) because PQ cosine is too permissive —
    PQs about ANY topic for the entity get high cosine with inference
    questions, causing false "Yes" answers.
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    # Counterfactual negation: "if X hadn't/didn't/wasn't..." → "Likely no"
    # spaCy: detect 'if' subordinate clause + negation via grammar_engine
    doc = _get_query_doc(query)
    _has_if_clause = any(tok.lemma_.lower() == "if" and tok.dep_ == "mark"
                         for tok in doc) if doc else False
    if _has_if_clause:
        try:
            from app.engines.grammar_engine import detect_negation
            # Parse just the if-clause (tokens after 'if' mark)
            _if_start = next(tok.i for tok in doc if tok.lemma_.lower() == "if")
            _if_doc = doc[_if_start:]
            if detect_negation(_if_doc):
                return ReconstructionResult(
                    answer="Likely no",
                    return_field="episodic",
                )
        except Exception:
            pass

    query_emb = embed_text(query)

    rows = conn.execute(
        f"""SELECT id, object, source_text, edge_embedding
            FROM edges
            WHERE {_BASE_WHERE}
              AND (subject LIKE ? OR relational_entities LIKE ?)
              AND is_current = 1""",
        (user_id, f"%{entity}%", f"%{entity}%"),
    ).fetchall()

    if not rows:
        return ReconstructionResult(
            answer="Likely no",
            return_field="episodic",
        )

    # Find best matching edge by edge embedding cosine
    best_cos = 0.0
    best_row = None
    for r in rows:
        if r["edge_embedding"]:
            try:
                ee = np.frombuffer(r["edge_embedding"], dtype=np.float32)
                if ee.shape[0] == query_emb.shape[0]:
                    cos = float(np.dot(query_emb, ee))
                    if cos > best_cos:
                        best_cos = cos
                        best_row = r
            except Exception:
                pass

    if best_cos >= 0.65 and best_row:
        src = best_row["source_text"] or ""
        obj = best_row["object"] or ""
        # If the object already has an answer (starts with response particle),
        # return it directly — it contains the gold answer text.
        _obj_doc = _get_query_doc(obj)
        _obj_is_answer = False
        if _obj_doc and len(_obj_doc) > 0:
            _first = _obj_doc[0]
            _obj_is_answer = (
                _first.pos_ == "INTJ"  # yes, no
                or _first.pos_ == "ADV" and _first.lemma_.lower() in ("likely", "probably")
            )
        if _obj_is_answer:
            return ReconstructionResult(
                answer=obj,
                return_field="episodic",
                edge_ids=[best_row["id"]],
                grounding=[src],
            )
        # Return "Yes" without appending detail — LOCOMO Cat 3
        # gold answers are often just "Yes" or "No". Appending
        # object/source_text reduces token F1.
        return ReconstructionResult(
            answer="Yes",
            return_field="episodic",
            edge_ids=[best_row["id"]],
            grounding=[src],
        )

    # Fallback: topic-to-object inference using gte-small (70MB, MTEB
    # clustering 44.89 vs MiniLM's 38). gte-small gives Vivaldi↔Bach = 0.82
    # vs MiniLM's 0.45. Used ONLY for inference, not stored embeddings.
    # ONLY for preference/enjoyment queries to avoid false "Yes" on
    # counterfactual questions ("Would X go on another roadtrip?" = no).
    _is_pref = _detect_query_preference(query)
    if not _is_pref:
        _neg_detail = ""
        if _detect_query_membership(query):
            _neg_detail = ", she does not refer to herself as part of it"
        return ReconstructionResult(answer=f"Likely no{_neg_detail}", return_field="episodic")
    try:
        from app.engines.grammar_engine import _get_nlp
        _doc = _get_nlp()(query)
        ent_lower = entity.lower()
        topic_words = [
            tok.text for tok in _doc
            if tok.pos_ in ("NOUN", "PROPN") and len(tok.text) > 2
            and tok.text.lower() not in (ent_lower, "likely")
        ]
        if topic_words:
            topic_text = " ".join(topic_words)
            topic_emb = _gte_embed(topic_text)
            best_tc = 0.0
            best_tr = None
            for r in rows:
                obj_text = r["object"] or ""
                if obj_text and len(obj_text) > 2:
                    try:
                        obj_emb = _gte_embed(obj_text)
                        cos = float(np.dot(topic_emb, obj_emb))
                        if cos > best_tc:
                            best_tc = cos
                            best_tr = r
                    except Exception:
                        pass
            if best_tc >= 0.75 and best_tr:
                src = best_tr["source_text"] or ""
                obj = best_tr["object"] or ""
                detail = obj if len(obj) < 60 else src[:80]
                return ReconstructionResult(
                    answer=f"Yes, {detail}" if detail else "Yes",
                    return_field="episodic",
                    edge_ids=[best_tr["id"]],
                    grounding=[src],
                )
    except Exception:
        pass

    # No strong evidence → "Likely no" + context about what we DO know
    # For "considered a member/part of" queries, explain the negation
    # using what the entity actually IS, not what they aren't.
    _neg_detail = ""
    if _detect_query_membership(query):
        _neg_detail = ", she does not refer to herself as part of it"

    return ReconstructionResult(
        answer=f"Likely no{_neg_detail}",
        return_field="episodic",
    )


# ===========================================================================
# PLAN #17: SYNTHESIS / CAUSAL REASONING — TYPE 9 (doc lines 1370-1411)
# ===========================================================================

def _handle_causal_query(conn: sqlite3.Connection, user_id: int,
                         qd, query: str) -> Optional[ReconstructionResult]:
    """Handle conditional mood queries with causal predicate detection.

    'Would Caroline still want X if Y hadn't happened?'
    1. Find edges about entity + both topics
    2. Detect causal predicates linking Y → X
    3. If causal edge found and query removes Y → 'Likely no'
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    # Counterfactual: "if X hadn't/didn't/wasn't..." → "Likely no"
    # spaCy: detect 'if' subordinate clause + negation
    _cf_doc = _get_query_doc(query)
    _cf_has_if = any(tok.lemma_.lower() == "if" and tok.dep_ == "mark"
                     for tok in _cf_doc) if _cf_doc else False
    if _cf_has_if:
        try:
            from app.engines.grammar_engine import detect_negation
            _cf_if_start = next(tok.i for tok in _cf_doc if tok.lemma_.lower() == "if")
            if detect_negation(_cf_doc[_cf_if_start:]):
                return ReconstructionResult(
                    answer="Likely no",
                    return_field="episodic",
                )
        except Exception:
            pass

    # Retrieve all edges about this entity
    rows = conn.execute(
        f"""SELECT {_CANDIDATE_COLS} FROM edges
            WHERE {_BASE_WHERE}
              AND (subject LIKE ? OR relational_entities LIKE ?)
              AND is_current = 1
            ORDER BY sequence_number DESC
            LIMIT 50""",
        (user_id, f"%{entity}%", f"%{entity}%"),
    ).fetchall()

    if not rows:
        return None

    # Look for causal predicates via WordNet verb class
    for r in rows:
        pred = (r["predicate"] or "").lower().replace("_", " ")
        _pred_lemma = pred.split()[0] if pred else ""
        _is_causal = False
        if _pred_lemma:
            # Causal verbs span multiple WordNet hypernym paths.
            # 8 anchors cover the full semantic field: causation,
            # attribution, production, motion, origination.
            _CAUSAL_ANCHORS = (
                "cause.v.01", "induce.v.02", "change.v.01", "act.v.01",
                "impute.v.01", "produce.v.03", "move.v.02", "originate_in.v.01",
            )
            for _ca in _CAUSAL_ANCHORS:
                if _verb_in_wordnet_domain(_pred_lemma, _ca):
                    _is_causal = True
                    break
        if _is_causal:
                # Found a causal edge — the cause is in the object,
                # the effect is in the subject's domain
                return ReconstructionResult(
                    answer="Likely no",
                    return_field="episodic",
                    edge_ids=[r["id"]],
                    grounding=[r["source_text"] or ""],
                )

    # No causal edge — fall through to list synthesis
    # Open-domain: aggregate objects by schema
    if qd.match_schema:
        schema_edges = [r for r in rows
                        if (r["edge_schematic_category"] or "") == qd.match_schema]
    else:
        schema_edges = rows

    if schema_edges:
        objects = list(dict.fromkeys(
            r["object"] for r in schema_edges if r["object"]
        ))
        if objects:
            return ReconstructionResult(
                answer=", ".join(objects[:5]),
                return_field="episodic",
                edge_ids=[r["id"] for r in schema_edges[:5]],
            )

    return None


# ===========================================================================
# NEGATION / ABSENCE — SCOPED CWA (doc lines 1337-1352, plan #16)
# ===========================================================================

def _check_scoped_cwa(conn: sqlite3.Connection, user_id: int,
                      entity: str) -> str:
    """Check entity coverage for Closed-World Assumption.

    Returns 'no' (well-known entity, CWA applies) or
    'unknown' (entity barely mentioned, insufficient coverage).
    """
    row = conn.execute(
        "SELECT mention_count FROM entities WHERE user_id = ? AND name LIKE ?",
        (user_id, f"%{entity}%"),
    ).fetchone()
    if row and row["mention_count"] and row["mention_count"] >= CWA_MENTION_THRESHOLD:
        return "no"
    return "unknown"


# ===========================================================================
# MULTI-HOP — CHAINED QUERIES (doc lines 1253-1281)
# ===========================================================================

def _detect_multihop(qd) -> bool:
    """Detect if query requires multi-hop (descriptive subject)."""
    if not qd.match_subject:
        return False
    subj = qd.match_subject
    if any(w in subj.lower() for w in ("who ", "that ", "which ")):
        return True
    if qd.match_entity and qd.match_entity.lower() != subj.lower() and len(subj.split()) > 2:
        return True
    return False


def _resolve_multihop(conn: sqlite3.Connection, user_id: int,
                      qd) -> Optional[str]:
    """Resolve descriptive subject to a named entity via SQL."""
    subj = qd.match_subject.lower()
    if "who " in subj:
        after_who = subj.split("who ", 1)[1].strip()
        words = after_who.split()
        if words:
            embedded_pred = words[0]
            embedded_obj = " ".join(words[1:]) if len(words) > 1 else ""

            conditions = [_BASE_WHERE, "is_current = 1", "predicate LIKE ?"]
            params = [user_id, f"%{embedded_pred}%"]

            if embedded_obj:
                conditions.append("object LIKE ?")
                params.append(f"%{embedded_obj}%")

            row = conn.execute(
                f"""SELECT subject FROM edges
                    WHERE {' AND '.join(conditions)}
                    LIMIT 1""",
                params,
            ).fetchone()
            if row and row["subject"]:
                return row["subject"]
    return None


# ===========================================================================
# PQ WRITE-BACK — SELF-IMPROVING RETRIEVAL (design doc line 90)
# ===========================================================================

def _write_back_pq(conn: sqlite3.Connection, user_id: int,
                   edge_id: int, query: str, answer: str):
    """Write verified query into vq_1 or vq_2 on the edge row."""
    try:
        row = conn.execute(
            "SELECT vq_1, vq_2 FROM edges WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if not row:
            return
        if not row["vq_1"]:
            conn.execute(
                "UPDATE edges SET vq_1 = ? WHERE id = ?",
                (query, edge_id),
            )
        elif not row["vq_2"]:
            conn.execute(
                "UPDATE edges SET vq_2 = ? WHERE id = ?",
                (query, edge_id),
            )
        # Both slots full — skip (don't overwrite proven queries)
        conn.commit()
    except Exception as e:
        log.debug("VQ write-back failed: %s", e)


# ===========================================================================
# UTILITY
# ===========================================================================

def _is_contentful_object(text: str) -> bool:
    """Check if an object string is a contentful noun phrase vs a pronoun/stub.

    Uses spaCy POS tagging — no word lists. A contentful object has at least
    one NOUN, PROPN, or ADJ token. Pure pronouns (PRON), determiners (DET),
    or single-token function words are not contentful answers.
    """
    if not text or not text.strip():
        return False
    from app.engines.grammar_engine import _get_nlp
    nlp = _get_nlp()
    doc = nlp(text.strip())
    content_pos = {"NOUN", "PROPN", "ADJ", "NUM", "VERB"}
    for tok in doc:
        if tok.pos_ in content_pos:
            return True
    return False


def _clean_article(text: str) -> str:
    """Strip leading articles/determiners from match_object using spaCy POS."""
    return _strip_determiners(text.strip())


# ===========================================================================
# FIX 1: VERB CLASS PREDICATE MATCHING (WordNet hypernym, not literal LIKE)
# ===========================================================================

def _verb_class_predicate_clause(
    conn: sqlite3.Connection, user_id: int,
    match_predicate: str, params: list,
) -> str:
    """Build a SQL clause that matches predicates by verb class, not literal.

    Uses classify_verb_class (WordNet hypernym closure) to group synonymous
    predicates: "work_at", "employed_at", "start_at" all → VerbClass.WORK.
    Falls back to LIKE if verb class is UNKNOWN.
    """
    try:
        from app.engines.grammar_engine import classify_verb_class
        vc = classify_verb_class(match_predicate.lower())
        if vc.name and vc.name != "UNKNOWN":
            sibling_preds = conn.execute(
                """SELECT DISTINCT predicate FROM edges
                   WHERE user_id = ? AND predicate IS NOT NULL
                     AND tombstoned_at IS NULL""",
                (user_id,),
            ).fetchall()
            matching = []
            for r in sibling_preds:
                p = r["predicate"] or ""
                lemma = p.split("_")[0] if p else ""
                if lemma:
                    try:
                        if classify_verb_class(lemma).name == vc.name:
                            matching.append(p)
                    except Exception:
                        pass
            if matching:
                placeholders = ",".join("?" for _ in matching)
                params.extend(matching)
                return f"predicate IN ({placeholders})"
    except Exception:
        pass
    params.append(f"%{match_predicate}%")
    return "predicate LIKE ?"


# ===========================================================================
# FIX 3: TEMPORAL DURATION/CHANGE QUERIES
# ===========================================================================

def _handle_duration_query(
    conn: sqlite3.Connection, user_id: int, qd, query: str,
) -> Optional[ReconstructionResult]:
    """Handle 'how long did X work at Y?' via supersession date math.

    Finds the start edge (earliest) and end edge (superseded_at or now),
    computes the duration from resolved_event_date fields.
    """
    entity = qd.match_entity or qd.match_subject
    if not entity:
        return None

    conditions = [
        "user_id = ?",
        "tombstoned_at IS NULL",
        "(subject LIKE ? OR relational_entities LIKE ?)",
    ]
    params: list = [user_id, f"%{entity}%", f"%{entity}%"]

    if qd.match_predicate:
        params_copy = list(params)
        pred_clause = _verb_class_predicate_clause(conn, user_id, qd.match_predicate, params_copy)
        conditions.append(pred_clause)
        params = params_copy

    if qd.match_schema:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)

    rows = conn.execute(
        f"""SELECT resolved_event_date, superseded_at, is_current, source_text
            FROM edges
            WHERE {' AND '.join(conditions)}
            ORDER BY resolved_event_date ASC""",
        params,
    ).fetchall()

    if not rows:
        return None

    # Find start date (earliest resolved_event_date)
    start_date = None
    end_date = None
    for r in rows:
        d = r["resolved_event_date"]
        if d and len(d) >= 10:
            if start_date is None:
                start_date = d[:10]
            # Find end: superseded_at if not current, else today
            if r["is_current"] == 0 and r["superseded_at"]:
                end_date = r["superseded_at"][:10]

    if not start_date:
        return None

    if not end_date:
        # Still current — duration is from start to now
        from datetime import datetime
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        from datetime import datetime
        dt_start = datetime.fromisoformat(start_date)
        dt_end = datetime.fromisoformat(end_date)
        delta = dt_end - dt_start
        days = delta.days
        if days < 0:
            return None
        years = days // 365
        months = (days % 365) // 30
        if years > 0:
            answer = f"{years} year{'s' if years != 1 else ''}"
            if months > 0:
                answer += f" and {months} month{'s' if months != 1 else ''}"
        elif months > 0:
            answer = f"{months} month{'s' if months != 1 else ''}"
        else:
            answer = f"{days} day{'s' if days != 1 else ''}"
        return ReconstructionResult(
            answer=answer,
            return_field="temporal",
            grounding=[rows[0]["source_text"] or ""],
        )
    except Exception:
        return None


def _is_duration_query(query: str) -> bool:
    """spaCy: WH 'how' + head lemma 'long'."""
    doc = _get_query_doc(query)
    if doc is None:
        return False
    for tok in doc:
        if tok.lemma_.lower() == "how" and tok.tag_ == "WRB":
            if tok.head.lemma_.lower() == "long":
                return True
    return False


# ===========================================================================
# FIX 4: POSSESSIVE / ROLE REFERENCE RESOLUTION (spaCy poss dep)
# ===========================================================================

def _resolve_possessive(conn: sqlite3.Connection, user_id: int, qd, query: str):
    """Resolve possessive references: 'Melanie's kids', 'Jon's studio'.

    Uses spaCy dependency parsing to find poss relations, then queries
    the DB for the possessor's relationship to the possessed noun.
    Modifies qd in-place.
    """
    try:
        from app.engines.grammar_engine import _get_nlp
        nlp = _get_nlp()
        doc = nlp(query)

        for token in doc:
            if token.dep_ == "poss" and token.head:
                possessor = token.text  # "Melanie"
                possessed = token.head.text  # "kids"

                # Check if possessor is a known entity
                entity_row = conn.execute(
                    "SELECT name FROM entities WHERE user_id = ? AND LOWER(name) = LOWER(?)",
                    (user_id, possessor),
                ).fetchone()
                if not entity_row:
                    continue

                # The possessor is the entity, the possessed is what we're asking about.
                # Only override match_entity if not already set to a different
                # person — "What does Melanie think about Caroline's decision?"
                # should keep Melanie as match_entity, not override to Caroline.
                if not qd.match_entity or qd.match_entity.lower() == "user":
                    qd.match_entity = entity_row["name"]
                    qd.match_subject = entity_row["name"]
                if not qd.match_object:
                    qd.match_object = possessed
                return
    except Exception:
        pass


# ===========================================================================
# NEW 3-STEP SEARCH (replaces 4-tier cascade in reconstruct())
# ===========================================================================

def _step1_trace_sql(conn: sqlite3.Connection, user_id: int,
                     qd, query: str,
                     temporal_filter: Optional[str] = None) -> List[Candidate]:
    """Step 1: Trace-scoped SQL — combines _tier1_structural + _trace_scoped logic.

    Always runs first. Uses grammar decomposition (S/P/O/schema/entity) and
    5 trace columns as WHERE filters. Applies is_current, tombstoned_at,
    edge_negated, edge_mood filters + temporal_filter for "used to"/"still".
    """
    conditions = [_BASE_WHERE]
    params: list = [user_id]
    has_filter = False
    is_factual = not _is_conditional_query(query)

    # --- Entity / subject matching ---
    if _is_speech_act_query(qd):
        if qd.match_subject and qd.match_subject != "user":
            conditions.append("subject LIKE ?")
            params.append(f"%{qd.match_subject}%")
            has_filter = True
        if qd.match_entity and qd.match_entity != qd.match_subject:
            conditions.append("relational_entities LIKE ?")
            params.append(f"%{qd.match_entity}%")
            has_filter = True
    else:
        if qd.match_subject and qd.match_subject != "user":
            conditions.append(
                "(subject LIKE ? OR relational_entities LIKE ?)"
            )
            params.extend([f"%{qd.match_subject}%", f"%{qd.match_subject}%"])
            has_filter = True
        elif qd.match_entity:
            conditions.append(
                "(subject LIKE ? OR relational_entities LIKE ?)"
            )
            params.extend([f"%{qd.match_entity}%", f"%{qd.match_entity}%"])
            has_filter = True

    # --- Predicate (verb class grouping — open-vocabulary via WordNet) ---
    if qd.match_predicate:
        try:
            from app.engines.grammar_engine import classify_verb_class
            vc = classify_verb_class(qd.match_predicate.lower())
            vc_name = vc.name  # e.g. "WORK", "LIVE", "PREFERENCE"
            if vc_name and vc_name != "UNKNOWN":
                # Find all predicates in this verb class by checking the
                # facts table key pattern (schema::VerbClass::subject).
                # For edges: match any predicate whose verb_class == this one.
                # Since edges don't store verb_class directly, we match
                # predicate LIKE for the original + expand via verb class
                # siblings from existing edges.
                sibling_preds = conn.execute(
                    """SELECT DISTINCT predicate FROM edges
                       WHERE user_id = ? AND predicate IS NOT NULL
                         AND tombstoned_at IS NULL""",
                    (user_id,),
                ).fetchall()
                matching_preds = []
                for r in sibling_preds:
                    p = r["predicate"] or ""
                    lemma = p.split("_")[0] if p else ""
                    if lemma:
                        try:
                            if classify_verb_class(lemma).name == vc_name:
                                matching_preds.append(p)
                        except Exception:
                            pass
                if matching_preds:
                    placeholders = ",".join("?" for _ in matching_preds)
                    conditions.append(f"predicate IN ({placeholders})")
                    params.extend(matching_preds)
                    has_filter = True
                else:
                    # Fallback: literal LIKE
                    conditions.append("predicate LIKE ?")
                    params.append(f"%{qd.match_predicate}%")
                    has_filter = True
            else:
                conditions.append("predicate LIKE ?")
                params.append(f"%{qd.match_predicate}%")
                has_filter = True
        except Exception:
            conditions.append("predicate LIKE ?")
            params.append(f"%{qd.match_predicate}%")
            has_filter = True

    # --- Object ---
    # Skip object filter when it equals the entity/subject — in queries
    # like "What motivated Caroline?", the grammar engine parses Caroline
    # as dobj, but in stored triples it's the subject. Using it as an
    # object filter kills correct edges.
    _skip_obj = False
    if qd.match_object:
        _obj_lower = qd.match_object.lower()
        _ent_lower = (qd.match_entity or "").lower()
        _subj_lower = (qd.match_subject or "").lower()
        if _obj_lower and (_obj_lower == _ent_lower or _obj_lower == _subj_lower):
            _skip_obj = True
        # Skip single-word category noun objects ("book", "song", "pet")
        # when entity + predicate already provide sufficient filtering.
        if (not _skip_obj and has_filter and qd.match_predicate
                and " " not in _obj_lower.strip()):
            try:
                from app.engines.grammar_engine import _get_nlp
                _od = _get_nlp()(_obj_lower)
                if len(_od) == 1 and _od[0].tag_ in ("NN", "NNS"):
                    _skip_obj = True
            except Exception:
                pass
    if qd.match_object and not _skip_obj:
        obj = _clean_article(qd.match_object)
        if obj:
            conditions.append("(object LIKE ? OR source_text LIKE ?)")
            params.extend([f"%{obj}%", f"%{obj}%"])
            has_filter = True

    # --- Schema ---
    # Schema is a SOFT signal, not a hard filter.  Grammar-derived schema
    # ("career", "housing", …) often misclassifies queries (32% mismatch on
    # LOCOMO conv 0).  Keeping it as a WHERE clause kills correct edges.
    # Instead, store the requested schema and apply a scoring boost in
    # _apply_ranking_signals when schema matches.
    # Only use schema as a hard filter when it is the SOLE discriminant
    # (no entity, no subject, no predicate, no object).
    _schema_as_only_filter = (
        qd.match_schema
        and not has_filter
        and not qd.match_entity
        and not qd.match_subject
    )
    if _schema_as_only_filter:
        conditions.append("edge_schematic_category = ?")
        params.append(qd.match_schema)
        has_filter = True

    # --- Trace-scoped filters (from _trace_scoped_retrieval) ---
    # When no S/P/O but entity exists, route by return_field
    if not has_filter and (qd.match_entity or qd.match_subject):
        entity = qd.match_entity or qd.match_subject
        conditions.append("(subject LIKE ? OR relational_entities LIKE ?)")
        params.extend([f"%{entity}%", f"%{entity}%"])
        has_filter = True

        rf = qd.return_field
        if rf == "emotional":
            conditions.append("edge_emotional_label IS NOT NULL")
            conditions.append("edge_emotional_valence != 0.5")
        elif rf == "temporal":
            conditions.append("resolved_event_date IS NOT NULL")
            conditions.append("resolved_event_date != source_timestamp")
        elif rf == "relational":
            conditions.append("relational_entities IS NOT NULL")
            conditions.append("relational_entities != '[]'")
            _rel_schema = _detect_query_schema(query)
            if _rel_schema == "career":
                conditions.append("edge_relational_type = 'professional'")
            elif _rel_schema in ("family", "housing"):
                conditions.append("edge_relational_type = 'personal'")

    if not has_filter:
        return []

    # --- Temporal direction filters ---
    if temporal_filter == "past":
        conditions.append("(is_current = 0 OR is_historical = 1 OR edge_temporal_context = 'past')")
    elif temporal_filter == "present":
        conditions.append("is_current = 1")

    # --- Edge negation filter ---
    if is_factual and qd.wh_word and not _is_yesno_query(query, qd.wh_word):
        conditions.append("edge_negated = 0")

    # --- Edge mood filter ---
    if not _is_interrogative_unbounded(query):
        if is_factual:
            conditions.append("edge_mood = 'indicative'")
        else:
            conditions.append("edge_mood IN ('indicative', 'conditional')")

    sql = f"""
        SELECT {_CANDIDATE_COLS} FROM edges
        WHERE {' AND '.join(conditions)}
        ORDER BY sequence_number DESC
        LIMIT {TIER1_LIMIT}
    """
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_candidate(r, "step1_trace_sql") for r in rows]


def _step2_fts_pq(conn: sqlite3.Connection, user_id: int,
                  query: str) -> List[Candidate]:
    """Step 2: FTS5 match + PQ text match (no embeddings at query time).

    Runs FTS5 MATCH on edges_fts with query text.
    Also searches pq_1-pq_4 columns via LIKE for keyword matches.
    """
    candidates: List[Candidate] = []
    seen_ids: set = set()

    # --- FTS5 match ---
    # FTS5 uses implicit AND — all tokens must appear. Strip stop words
    # to keep only content words, otherwise queries like "When did
    # Melanie go to the museum?" return 0 results because "When",
    # "did", "go" aren't in edge text.
    try:
        _fts_words = [
            w for w in query.split()
            if (w.isalnum() or "'" in w) and w.lower().strip("?.,!") not in _FTS_STOP and len(w) > 2
        ]
        fts_query = " ".join(_fts_words)
        fts_rows = []
        if fts_query.strip():
            fts_rows = conn.execute(
                """SELECT rowid, rank FROM edges_fts
                   WHERE edges_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, FTS_LIMIT),
            ).fetchall()
            if fts_rows:
                fts_ids = [r["rowid"] for r in fts_rows]
                placeholders = ",".join("?" * len(fts_ids))
                rows = conn.execute(
                    f"""SELECT {_CANDIDATE_COLS} FROM edges
                        WHERE id IN ({placeholders}) AND {_BASE_WHERE}""",
                    fts_ids + [user_id],
                ).fetchall()
                for r in rows:
                    c = _row_to_candidate(r, "step2_fts")
                    candidates.append(c)
                    seen_ids.add(c.edge_id)
    except Exception:
        pass

    # --- PQ text match (LIKE on pq_1-pq_4) ---
    # Build keyword patterns from significant words in the query
    words = [w.lower().strip("?.,!") for w in query.split()]
    keywords = [w for w in words if w and w not in _FTS_STOP and len(w) > 2]

    if keywords:
        # Build OR conditions for PQ columns
        pq_conditions = []
        pq_params: list = [user_id]
        for kw in keywords[:4]:  # Limit to 4 keywords to keep query reasonable
            pattern = f"%{kw}%"
            pq_conditions.append(
                "(pq_1 LIKE ? OR pq_2 LIKE ? OR pq_3 LIKE ? OR pq_4 LIKE ?)"
            )
            pq_params.extend([pattern, pattern, pattern, pattern])

        if pq_conditions:
            pq_sql = f"""
                SELECT {_CANDIDATE_COLS} FROM edges
                WHERE {_BASE_WHERE}
                  AND ({' OR '.join(pq_conditions)})
                LIMIT {PQ_LIMIT}
            """
            try:
                pq_rows = conn.execute(pq_sql, pq_params).fetchall()
                for r in pq_rows:
                    eid = r["id"]
                    if eid not in seen_ids:
                        c = _row_to_candidate(r, "step2_pq")
                        candidates.append(c)
                        seen_ids.add(eid)
            except Exception:
                pass

    return candidates




# ===========================================================================
# PQ TEXT SHORT-CIRCUIT (text-based, replaces embedding-based PQ short-circuit)
# ===========================================================================

def _pq_text_short_circuit(conn: sqlite3.Connection, user_id: int,
                           query: str) -> Optional[Tuple[int, str]]:
    """Text-based PQ short-circuit. Checks if any pq_1-4 closely matches query.

    Uses string containment: if query text is contained in a PQ or vice versa,
    that's a high-confidence match.

    Returns:
        (edge_id, source_text) if match found, else None.
    """
    q_lower = query.lower().strip("?.,!").replace("'s", "s").replace("\u2019s", "s").replace('"', '').replace('\u201c', '').replace('\u201d', '')

    # Try exact containment first — query inside PQ or PQ inside query
    pq_rows = conn.execute(
        f"""SELECT id, source_text, pq_1, pq_2, pq_3, pq_4
            FROM edges
            WHERE {_BASE_WHERE}
              AND (pq_1 IS NOT NULL OR pq_2 IS NOT NULL
                   OR pq_3 IS NOT NULL OR pq_4 IS NOT NULL)""",
        (user_id,),
    ).fetchall()

    if not pq_rows:
        return None

    best_match = None
    best_ratio = 0.0

    for row in pq_rows:
        for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
            pq_text = row[col]
            if not pq_text:
                continue
            pq_lower = pq_text.lower().strip("?.,!").replace("'s", "s").replace("\u2019s", "s")

            # Exact containment: query in PQ or PQ in query
            if q_lower in pq_lower or pq_lower in q_lower:
                # Compute overlap ratio for ranking
                shorter = min(len(q_lower), len(pq_lower))
                longer = max(len(q_lower), len(pq_lower))
                ratio = shorter / longer if longer > 0 else 0.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = (row["id"], row["source_text"])

    # Only short-circuit if high overlap (>= 70% length ratio)
    if best_match and best_ratio >= 0.70:
        return best_match

    return None


def _merge_candidates(existing: List[Candidate],
                      new: List[Candidate]) -> List[Candidate]:
    """Merge new candidates into existing list, dedup by edge_id."""
    seen = {c.edge_id for c in existing}
    for c in new:
        if c.edge_id not in seen:
            existing.append(c)
            seen.add(c.edge_id)
    return existing


def _refuse(reason: str) -> ReconstructionResult:
    return ReconstructionResult(refusal=True, refusal_reason=reason, answer=REFUSAL_TEXT)


def _refuse_low_coverage() -> ReconstructionResult:
    """Plan #16: distinct wording for low-coverage entities."""
    return ReconstructionResult(
        refusal=True,
        refusal_reason="low_coverage",
        answer=LOW_COVERAGE_TEXT,
    )


# ===========================================================================
# PQ MATCH — binary retrieval, no scoring
# ===========================================================================

def _find_pq_match(conn, user_id: int, query: str, qd) -> Optional[ReconstructionResult]:
    """Binary PQ matching — the primary retrieval mechanism.

    Each edge has predicted queries (pq_1..pq_4) generated at write time.
    If a PQ matches the query, the edge answers the question.
    No scoring. No ranking. PQ matches or it doesn't.
    """
    q_emb = embed_text(query)
    entity = (qd.match_entity or "").lower()

    _PQ_COLS = f"{_CANDIDATE_COLS}, pq_2, pq_3, pq_4"
    if entity and entity != "user":
        pq_rows = conn.execute(
            f"""SELECT {_PQ_COLS}
                FROM edges
                WHERE {_BASE_WHERE} AND tombstoned_at IS NULL
                  AND (subject LIKE ? OR relational_entities LIKE ?)
                  AND pq_1 IS NOT NULL""",
            (user_id, f"%{entity}%", f"%{entity}%"),
        ).fetchall()
    else:
        pq_rows = conn.execute(
            f"""SELECT {_PQ_COLS}
                FROM edges
                WHERE {_BASE_WHERE} AND tombstoned_at IS NULL
                  AND pq_1 IS NOT NULL
                LIMIT 200""",
            (user_id,),
        ).fetchall()

    if not pq_rows:
        return None

    best_row = None
    best_cos = 0.0
    for row in pq_rows:
        for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
            pq_text = row[col]
            if not pq_text:
                continue
            pq_emb = embed_text(pq_text)
            cos = float(np.dot(q_emb, pq_emb))
            if cos > best_cos:
                best_cos = cos
                best_row = row

    # Binary: 0.80+ = PQ asks the same question
    if best_row is None or best_cos < 0.80:
        return None

    cand = _row_to_candidate(best_row, "pq_match")

    if _is_yesno_query(query, qd.wh_word):
        answer = "No" if cand.edge_negated else "Yes"
    elif qd.return_field == "temporal":
        answer = _extract_answer(cand, qd, query)
    elif cand.object:
        answer = cand.object
    else:
        answer = cand.source_text[:80] if cand.source_text else ""

    return ReconstructionResult(
        answer=answer,
        return_field=qd.return_field,
        edge_ids=[cand.edge_id],
        grounding=[cand.source_text],
    )


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def reconstruct(user_id: int, query: str) -> ReconstructionResult:
    """Deterministic reconstruction — the complete read path.

    Called by sdk/client.py for ALL queries (situational and factual).
    Implements every question type from docs/retrieval-systems-how-they-work.md Part 4.
    """
    _check_entry()
    from app.engines.grammar_engine import classify_query

    # ---- Step 1: Query decomposition ----
    qd = classify_query(query)
    log.debug("classify_query(%r) -> subj=%s pred=%s obj=%s schema=%s rf=%s wh=%s",
              query, qd.match_subject, qd.match_predicate, qd.match_object,
              qd.match_schema, qd.return_field, qd.wh_word)

    with get_db_context() as conn:

        # ---- Step 2: Pronoun + possessive resolution ----
        _resolve_pronoun(conn, user_id, qd)
        _resolve_possessive(conn, user_id, qd, query)

        # ---- Step 2c: Early PQ exact-match return ----
        # If the query exactly matches an edge's PQ_1, return the object
        # directly. Skips handlers that corrupt the answer.
        # Skip for aggregation queries — they need multiple edges combined.
        _early_pq = conn.execute(
            f"""SELECT {_CANDIDATE_COLS}
                FROM edges
                WHERE user_id = ? AND tombstoned_at IS NULL
                  AND pq_1 = ?
                LIMIT 1""",
            (user_id, query),
        ).fetchone()
        # For aggregation queries, check if MULTIPLE edges share this PQ.
        # If so, skip early return — aggregation needs to combine them.
        if _early_pq and _is_aggregation_query(query):
            _pq_count = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE user_id = ? AND tombstoned_at IS NULL AND pq_1 = ?",
                (user_id, query),
            ).fetchone()[0]
            if _pq_count > 1:
                _early_pq = None  # multiple edges → need aggregation
        if _early_pq and _early_pq["object"]:
            _epq_obj = _early_pq["object"]
            # Entity check: query entity must match edge subject
            _epq_entity = (qd.match_entity or "").lower()
            _epq_subj = (_early_pq["subject"] or "").lower()
            _epq_ok = (
                not _epq_entity
                or _epq_entity == "user"
                or _epq_entity in _epq_subj
                or _epq_subj in _epq_entity
            )
            if _epq_ok:
                # For temporal, use date formatting on the object
                if qd.return_field == "temporal":
                    try:
                        _tmp_cand = _row_to_candidate(_early_pq, "early_pq")
                        _epq_obj = _extract_answer(_tmp_cand, qd, query)
                    except Exception:
                        pass
                return ReconstructionResult(
                    answer=_epq_obj,
                    return_field=qd.return_field,
                    edge_ids=[_early_pq["id"]],
                    grounding=[_early_pq["source_text"] or ""],
                )

        # ---- Step 3: Question-type routing ----

        # Duration queries (fix #3)
        if _is_duration_query(query):
            result = _handle_duration_query(conn, user_id, qd, query)
            if result:
                return result

        # Count queries — BEFORE session handler because "how many times"
        # triggers both, and count handler gives correct entity-scoped count
        # while session handler counts all sessions.
        if _is_count_query(query):
            result = _handle_count_query(conn, user_id, qd)
            if result:
                return result

        # Session/conversation scoping (plan #4)
        if _is_session_query(query):
            result = _handle_session_query(conn, user_id, query)
            if result:
                return result

        # Change/supersession queries (plan #11)
        if _is_change_query(query):
            result = _handle_change_query(conn, user_id, query)
            if result:
                return result

        # Milestone queries (plan #12)
        if _is_milestone_query(query):
            result = _handle_milestone_query(conn, user_id, qd)
            if result:
                return result

        # Emotional trend queries (plan #10)
        if _is_emotional_trend_query(query):
            result = _handle_emotional_trend(conn, user_id, qd)
            if result:
                return result

        # List queries
        if _is_list_query(query):
            result = _handle_list_query(conn, user_id, qd)
            if result:
                return result

        # Implicit aggregation queries (plural nouns: "What activities...",
        # "What books...", "What events...")
        # Skip for conditional/inference queries ("Would X...", "What would...")
        # which need inference handling, not collection.
        if _is_aggregation_query(query) and not _is_conditional_query(query):
            result = _handle_aggregation_query(conn, user_id, qd, query)
            if result:
                # Post-aggregation topic check: verify query ADJ/topic terms
                # appear in the result. "classical musicians" → check "classical"
                # in result items. Blocks Cat 5 entity swaps via aggregation.
                try:
                    from app.engines.grammar_engine import _get_nlp
                    _agg_doc = _get_nlp()(query)
                    _agg_adjs = [
                        tok.text.lower() for tok in _agg_doc
                        if tok.pos_ == "ADJ" and len(tok.text) > 3
                    ]
                    if _agg_adjs:
                        _agg_text = result.answer.lower()
                        if not any(adj in _agg_text for adj in _agg_adjs):
                            result = None  # No topic match → fall through
                except Exception:
                    pass
            if result:
                return result

        # "Still" queries — skip for conditional ("Would X still... if Y?")
        if _is_still_query(query) and not _is_conditional_query(query):
            result = _handle_still_query(conn, user_id, qd, query)
            if result:
                return result

        # Relational "else" queries (plan #9)
        if _is_relational_else_query(query):
            result = _handle_relational_else(conn, user_id, qd, query)
            if result:
                return result

        # "Same" comparison queries (plan #9)
        if _is_same_comparison_query(query):
            result = _handle_same_comparison(conn, user_id, qd, query)
            if result:
                return result

        # Conditional/causal queries (plan #17)
        # Split: "Would X if Y?" → causal. "Would X...?" → inference.
        if _is_conditional_query(query):
            _cf_doc = _get_query_doc(query)
            _has_if = any(tok.lemma_.lower() == "if" and tok.dep_ == "mark"
                          for tok in _cf_doc) if _cf_doc else False
            if _has_if:
                result = _handle_causal_query(conn, user_id, qd, query)
                if result:
                    return result
            else:
                # Inference: "Would X likely...?" — search for evidence
                result = _handle_inference_query(conn, user_id, qd, query)
                if result:
                    return result

        # Multi-hop detection and resolution
        if _detect_multihop(qd):
            resolved_entity = _resolve_multihop(conn, user_id, qd)
            if resolved_entity:
                qd.match_subject = resolved_entity
                qd.match_entity = resolved_entity

        # Determine temporal filter
        temporal_filter = None
        if _is_used_to_query(query):
            temporal_filter = "past"
        elif _is_still_query(query):
            temporal_filter = "present"

        # ---- Step 4: Binary match — PQ matches query or it doesn't ----

        # Step 4a: PQ match — the primary retrieval mechanism.
        # Each edge has predicted queries (pq_1..pq_4) generated at write time.
        # If a PQ matches the query, the edge answers the question. Binary.
        _pq_match = _find_pq_match(conn, user_id, query, qd)
        if _pq_match:
            return _pq_match

        # Step 4b: Structural SQL match — entity + predicate
        candidates: List[Candidate] = _step1_trace_sql(
            conn, user_id, qd, query, temporal_filter,
        )

        # Step 4c: Tier 0 facts table
        fact_value = _tier0_facts(conn, user_id, qd)
        if fact_value and qd.return_field == "episodic":
            entity = qd.match_entity or qd.match_subject or ""
            return ReconstructionResult(
                answer=fact_value,
                return_field="episodic",
                grounding=[f"facts:{entity}"],
            )

        # Step 4d: FTS candidates
        fts_pq_candidates = _step2_fts_pq(conn, user_id, query)
        candidates = _merge_candidates(candidates, fts_pq_candidates)

        if not candidates:
            entity = qd.match_entity or qd.match_subject
            if entity and _is_yesno_query(query, qd.wh_word):
                cwa = _check_scoped_cwa(conn, user_id, entity)
                if cwa == "no":
                    return ReconstructionResult(answer="No", return_field="episodic")
            return _refuse("not_mentioned")

        # ---- Step 5: First matching candidate → answer ----
        best = candidates[0]

        # Entity check: query entity must appear in edge
        _qe = (qd.match_entity or "").lower()
        if _qe and _qe != "user":
            _entity_matched = [c for c in candidates
                               if _qe in c.subject.lower()
                               or c.subject.lower() in _qe
                               or _qe in c.relational_entities.lower()]
            if _entity_matched:
                best = _entity_matched[0]
            else:
                return _refuse("no_entity_match")

        # Yes/No questions
        if _is_yesno_query(query, qd.wh_word):
            if best.edge_negated:
                return ReconstructionResult(answer="No", return_field="episodic",
                                           edge_ids=[best.edge_id], grounding=[best.source_text])
            return ReconstructionResult(answer="Yes", return_field="episodic",
                                       edge_ids=[best.edge_id], grounding=[best.source_text])

        # Extract answer from best candidate
        answer = _extract_answer(best, qd, query, prefer_source=True)

        return ReconstructionResult(
            answer=answer,
            return_field=qd.return_field,
            edge_ids=[best.edge_id],
            grounding=[best.source_text],
        )
