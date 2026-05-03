# -*- coding: utf-8 -*-
"""
RetrievalEngine -- moat pipeline + verification loop.

Two layers:
  1. Moat pipeline (Stages 1-5): candidate generation and ordering.
     Does NOT answer questions. Produces ranked candidates only.
  2. Verification loop: anti-hallucination answer gate.
     Decides truth. First verified candidate wins; otherwise refuse.

Lookup pipeline:
    query -> embed + resolve entity
          -> Stage 1 ENTRY  (recall from PQ, edge, FTS)
          -> Stage 2 EXPAND (entity-union expansion)
          -> Stage 3 GROUP  (cluster coherence)
          -> Stage 4 RELATE (drop disconnected edges)
          -> Stage 5 EXIT   (final cosine ordering)
          -> verification loop
          -> answer / refusal
          -> PQ write-back

reconstruct() shares Stages 1-5, then reclusters and renders narrative.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text
from app.engines.retrieval_types import Candidate

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTRY_CAP = 80
RECONSTRUCT_TOP_N = 30
REFUSAL_TEXT = "This information is not mentioned in the conversation."


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    text: Optional[str]
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: float = 0.0
    source: str = "pq_cosine"
    survivors: int = 0
    convergence_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralRefusal:
    reason: str
    text: Optional[str] = None
    confidence: float = 0.0
    source: str = "structural_refusal"
    survivors: int = 0
    convergence_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Cluster:
    cluster_id: Optional[str]
    participants: List[str] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    narrative: str = ""
    schema_category: Optional[str] = None
    emotional_tone: Optional[str] = None


@dataclass
class Situation:
    clusters: List[Cluster] = field(default_factory=list)
    survivor_count: int = 0
    narrative: str = ""
    source_tag: str = "reconstruct"
    participants: List[str] = field(default_factory=list)


@dataclass
class IndirectResolution:
    """Typed result from _resolve_indirect_references.

    Root cause: the resolver mixes entity refs and comparative results
    in List[Dict], requiring string-tag routing in retrieve().
    Typed fields eliminate tag checking — same pattern as Answer and
    StructuralRefusal above.

    Fields:
    - entity_refs: resolved entity names from possessive/relcl/speaker
    - answer: pre-computed answer (e.g., comparative intersection "DC")
    - refusal: pre-computed refusal (e.g., empty comparative intersection)
    """
    entity_refs: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[Answer] = None
    refusal: Optional[StructuralRefusal] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    try:
        da = float(np.linalg.norm(a))
        db = float(np.linalg.norm(b))
        if da == 0.0 or db == 0.0:
            return 0.0
        return float(np.dot(a, b) / (da * db))
    except Exception:
        return 0.0


def _cosine_from_blob(q_emb: np.ndarray, blob) -> float:
    if blob is None:
        return 0.0
    try:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size != q_emb.size:
            return 0.0
        denom = float(np.linalg.norm(q_emb) * np.linalg.norm(v))
        if denom == 0.0:
            return 0.0
        return float(np.dot(q_emb, v) / denom)
    except Exception:
        return 0.0


def _deserialize_emb(blob) -> Optional[np.ndarray]:
    if blob is None:
        return None
    try:
        return np.frombuffer(blob, dtype=np.float32).copy()
    except Exception:
        return None


def _split_words(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    for ch in text or "":
        if ch.isalnum():
            buf.append(ch.lower())
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def _normalize_token(token: str) -> str:
    t = (token or "").strip().lower()
    if not t:
        return ""
    irregular = {
        "am": "be", "is": "be", "are": "be",
        "was": "be", "were": "be", "been": "be", "being": "be",
        "has": "have", "had": "have",
        "does": "do", "did": "do",
    }
    if t in irregular:
        return irregular[t]
    try:
        from nltk.corpus import wordnet as _wn
        lemma = _wn.morphy(t, _wn.VERB)
        if lemma:
            return lemma
        lemma = _wn.morphy(t, _wn.NOUN)
        if lemma:
            return lemma
    except Exception:
        pass
    if len(t) > 4 and t.endswith("ied"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("ed"):
        return t[:-2]
    if len(t) > 4 and t.endswith("es"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s"):
        return t[:-1]
    return t


def _normalized_words(text: str) -> Set[str]:
    words: Set[str] = set()
    for tok in _split_words(text):
        norm = _normalize_token(tok)
        if norm:
            words.add(norm)
    return words


def _extract_query_entity_fallback(query: str) -> List[str]:
    """Best-effort direct entity extraction from the raw query text.

    Retrieval cannot depend solely on the entities table because direct
    name queries like "Who is Tariq?" still need a query_entity even
    when entity indexing is sparse or delayed.
    """
    names: List[str] = []

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        for ent in doc.ents:
            text = (ent.text or "").strip()
            if text and text.lower() not in _FIRST_PERSON:
                names.append(text)
    except Exception:
        pass

    if not names:
        run: List[str] = []
        for raw in (query or "").split():
            cleaned = "".join(ch for ch in raw if ch.isalpha() or ch in ("'", "-"))
            if not cleaned:
                if run:
                    names.append(" ".join(run))
                    run = []
                continue
            if cleaned[:1].isupper() and cleaned.lower() not in _FIRST_PERSON:
                run.append(cleaned)
            else:
                if run:
                    names.append(" ".join(run))
                    run = []
        if run:
            names.append(" ".join(run))

    seen: Set[str] = set()
    deduped: List[str] = []
    for name in names:
        key = name.strip().lower()
        if key and key not in seen:
            deduped.append(name.strip())
            seen.add(key)
    return deduped


def _query_mentions_entity(query_text: str, query_entity: Optional[str]) -> bool:
    if not query_entity:
        return False
    q_words = _split_words(query_text)
    e_words = _split_words(query_entity)
    if not q_words or not e_words:
        return False
    width = len(e_words)
    for i in range(0, len(q_words) - width + 1):
        if q_words[i:i + width] == e_words:
            return True
    return False


def _is_direct_identity_query(
    query_text: str,
    query_entity: Optional[str],
) -> bool:
    """True for queries like "Who is Tariq?" / "Who is Mika?".

    These ask for a role/description of a named entity, not for any
    arbitrary relation whose predicate head happens to be "be".
    False for property queries like "What is Sam allergic to?" where
    the entity is followed by additional predicate content.
    """
    q_words = _split_words(query_text)
    if len(q_words) < 3:
        return False
    if q_words[0] not in {"who", "what"} or q_words[1] not in {"is", "are", "was", "were"}:
        return False
    if not query_entity or query_entity.lower() in _FIRST_PERSON:
        return False
    if not _query_mentions_entity(query_text, query_entity):
        return False
    # Identity only if entity is the LAST content in the query.
    # "Who is Sam?" → identity (Sam is last). "What is Sam allergic to?"
    # → NOT identity ("allergic to" follows Sam).
    ent_words = _split_words(query_entity)
    if not ent_words:
        return False
    last_ent_word = ent_words[-1].lower()
    try:
        idx = [w.lower() for w in q_words].index(last_ent_word)
        # Remaining words after entity (excluding closed-class)
        remaining = [w for w in q_words[idx + 1:]
                     if w.lower() not in {"s", "to", "the", "a", "an"}]
        return len(remaining) == 0
    except ValueError:
        return False


_FIRST_PERSON = frozenset({"i", "me", "my", "myself", "user"})


def _edge_mentions_entity(
    edge_subject: str, edge_object: str, query_entity: Optional[str],
    relational_entities: str = "",
) -> bool:
    """Does this edge mention the query entity in either position?

    'user' edges only match when the query entity is first-person
    (I/me/my/user) OR appears in relational_entities (the write path
    appends the speaker's real name to relational_entities on every edge).
    This prevents 'user work_at Google' from matching a query about
    Dad or Jake while still matching the speaker's own name."""
    if not query_entity:
        return True
    subj = " ".join(_split_words(edge_subject))
    obj = " ".join(_split_words(edge_object))
    ent = " ".join(_split_words(query_entity))
    if not ent:
        return True
    # Query entity must match edge subject
    if subj == ent:
        return True
    # Object match (e.g., "Linda be_mother_of user" with query about Linda)
    if obj == ent:
        return True
    # 'user' edges match first-person queries
    query_is_first_person = ent in _FIRST_PERSON
    if query_is_first_person and (subj == "user" or obj == "user"):
        return True
    # 'user' edges match when relational_entities contains the query entity
    # (speaker's real name appended by write path)
    if subj == "user" and relational_entities:
        rel_lower = relational_entities.lower()
        if query_entity.lower() in rel_lower:
            return True
    return False


def _edge_to_fact_statement(edge: Dict[str, Any]) -> str:
    s = (edge.get("subject") or "").replace("_", " ")
    p = (edge.get("predicate") or "").replace("_", " ")
    o = (edge.get("object") or "").replace("_", " ")
    # Replace structural "user" with "I" so the model sees
    # natural language, not an internal token
    if s.lower() == "user":
        s = "I"
    if o.lower() == "user":
        o = "me"
    return f"{s} {p} {o}".strip()


def _extract_query_verb(query: str) -> Optional[str]:
    """Extract the semantic predicate from a query via spaCy dep parse.

    Uses Universal Dependencies ROOT + complement extraction:
    - Content verbs: ROOT is the predicate. "What does Noor drive?" -> "drive"
    - Copula (be): semantic predicate is the attr/acomp child, not "be".
      "What is your job?" -> "job". "Where is Tariq based?" -> "base".
    - Light verb (do): semantic predicate is the pobj/dobj complement.
      "What does Suki do for exercise?" -> "exercise"."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return None

    doc = nlp(query)
    root = None
    for tok in doc:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if root is None:
        return None

    # Content verb: ROOT is VERB (not AUX) and not a light verb
    if root.pos_ == "VERB" and root.lemma_.lower() not in ("be", "do"):
        return root.lemma_.lower()

    # Copula "be" or light verb "do": extract semantic predicate
    # from complement children (attr, acomp, oprd, pobj, dobj)
    for child in root.children:
        if child.dep_ in ("attr", "acomp", "oprd"):
            if child.pos_ in ("NOUN", "ADJ", "VERB"):
                return child.lemma_.lower()
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return grandchild.lemma_.lower()
        if child.dep_ in ("dobj", "nsubj") and child.pos_ == "NOUN":
            # "What is your job?" -> nsubj=job
            if child.text.lower() not in ("what", "who", "where", "when", "which", "how"):
                return child.lemma_.lower()

    # Fallback: if ROOT is a real verb, return it
    if root.pos_ in ("VERB", "AUX"):
        return root.lemma_.lower()
    return None


def _extract_edge_predicate_lemma(predicate: str) -> Optional[str]:
    """Lemmatize head token of an edge predicate via _normalize_token."""
    if not predicate:
        return None
    head = predicate.replace("_", " ").strip().split()[0]
    return _normalize_token(head) or None


def _wordnet_verb_match(verb_a: str, verb_b: str) -> bool:
    """Check if two words are semantically related via WordNet.
    Walks hypernym chains and checks lemma name overlap.
    cuisine->cooking->cook, exercise->activity, job->work."""
    if not verb_a or not verb_b:
        return False
    if verb_a == verb_b:
        return True
    try:
        from nltk.corpus import wordnet as wn
        # Collect lemma names from synsets + derivationally related forms.
        # For each word: direct synset lemmas + derivational forms only.
        # NO hypernym traversal — too many false positives at 1 hop
        # because generic verbs (make, take, change) connect everything.
        def _collect(word):
            names = set()
            for pos in (wn.VERB, wn.NOUN):
                for s in wn.synsets(word, pos=pos):
                    names.update(s.lemma_names())
                    for lemma in s.lemmas():
                        for d in lemma.derivationally_related_forms():
                            names.add(d.name())
            return {n.lower().replace("_", "") for n in names}
        return bool(_collect(verb_a) & _collect(verb_b))

    except Exception:
        return False


def _wordnet_noun_in_predicate(query_word: str, predicate_text: str) -> bool:
    """Check if query_word appears in predicate_text via WordNet expansion.

    Expands query_word through synset lemmas, derivational forms, and
    1-hop hypernym lemmas. Then checks if ANY expanded term appears as
    a word in the predicate text. This catches:
      mom -> mother (hypernym) -> found in 'be mother of'
      exercise -> found in 'exercise' (direct)
      pet -> found in 'own a pet' (if source_text used)

    Uses WordNet's semantic graph — no hardcoded lists."""
    if not query_word or not predicate_text:
        return False
    pred_words = set(predicate_text.lower().split())
    # Direct word match (handles exact and substring)
    qw = query_word.lower()
    if qw in pred_words:
        return True
    if any(qw in pw for pw in pred_words):
        return True

    try:
        from nltk.corpus import wordnet as wn
        expanded: set = {qw}
        for pos in (wn.NOUN, wn.VERB, wn.ADJ):
            for syn in wn.synsets(query_word, pos=pos):
                for lemma in syn.lemmas():
                    expanded.add(lemma.name().lower().replace("_", " "))
                    for d in lemma.derivationally_related_forms():
                        expanded.add(d.name().lower().replace("_", " "))
                # 1-hop hypernyms
                for hyp in syn.hypernyms():
                    for lemma in hyp.lemmas():
                        expanded.add(lemma.name().lower().replace("_", " "))
        # Check expanded terms against predicate words.
        # For multi-word terms (e.g., "battle of marathon" from
        # hypernym expansion), only match content words (len > 2)
        # to avoid false positives from function words like "of"
        # matching "be roommate of user".
        for term in expanded:
            term_words = term.split()
            content_words = [tw for tw in term_words if len(tw) > 2]
            if content_words and any(tw in pred_words for tw in content_words):
                return True
    except Exception:
        pass
    return False


def _model_coherence_check(query: str, fact_statement: str) -> bool:
    """Coherence gate via coedit-small (already loaded for polish).

    Called only when predicate lemma match is structurally impossible
    (neither side extractable). Same model + inference pattern as
    sentence_model.polish()."""
    try:
        import app.engines.sentence_model as sm
        if not sm._load():
            return True
        import torch
        prompt = (
            f"{fact_statement} Question: "
            f"{query} Is the answer in the statement? yes or no?"
        )
        inp = sm._TOKENIZER(
            prompt, return_tensors="pt", max_length=128, truncation=True,
        ).to(sm._DEVICE)
        with torch.no_grad():
            out = sm._MODEL.generate(
                **inp, max_length=16, num_beams=2, do_sample=False,
            )
        result = sm._TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
        return result.lower().startswith("yes")
    except Exception:
        return True


def _extract_query_object_hint(query: str) -> Optional[str]:
    """Best-effort extraction of the query's target object phrase."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return None

    doc = nlp(query)

    def _span_text(tok) -> str:
        return " ".join(
            t.text for t in tok.subtree
            if t.dep_ != "det"
        ).strip()

    verb_candidates = [
        tok for tok in doc
        if tok.dep_ in ("relcl", "ROOT") and tok.pos_ in ("VERB", "AUX")
    ]
    for verb in verb_candidates:
        for child in verb.children:
            if child.dep_ in ("dobj", "attr", "oprd", "acomp"):
                text = _span_text(child)
                if text:
                    return text
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        text = _span_text(grandchild)
                        if text:
                            return text
    return None


def _query_requests_historical(query: str) -> bool:
    q = (query or "").lower()
    return (
        "used to" in q
        or " before " in f" {q} "
        or "previously" in q
        or "formerly" in q
        or "last worked" in q
        or "last lived" in q
    )


def _query_requests_current_state(query: str) -> bool:
    q_words = set(_split_words(query))
    return bool(q_words & {"still", "currently", "now", "current", "today"})


def _is_count_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return q.startswith("how many ")


def _is_list_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return (
        q.startswith("name everyone")
        or q.startswith("list all")
        or q.startswith("who all")
        or "everyone who" in q
    )


def _is_yes_no_query(query: str) -> bool:
    words = _split_words(query)
    if not words:
        return False
    return words[0] in {
        "does", "do", "did", "is", "are", "was", "were",
        "has", "have", "had", "can", "could", "will", "would",
    }


def _is_relational_peer_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return q.startswith("who else ")



# ---------------------------------------------------------------------------
# RetrievalEngine
# ---------------------------------------------------------------------------

class RetrievalEngine:
    def __init__(self, memory=None, temporal=None):
        self._memory = memory
        self._temporal = temporal

    # ===================================================================
    # INDIRECT REFERENCE RESOLUTION
    # ===================================================================

    # Speaker references and generic person nouns are closed grammatical
    # classes defined by Universal Dependencies — same exemption as
    # WH-words (who/what/where/when/why/how) already used in this codebase.
    _SPEAKER_REFS = frozenset({
        "speaker", "the speaker", "narrator", "the narrator",
        "talker", "the talker",
    })
    _GENERIC_PERSON_NOUNS = frozenset({
        "person", "one", "guy", "man", "woman", "boy", "girl",
        "friend", "someone", "somebody",
    })

    def _resolve_indirect_references(
        self, user_id: int, query: str,
    ) -> IndirectResolution:
        """Resolve possessive, relative-clause, speaker-reference,
        and comparative patterns against the SPO knowledge graph.

        Returns IndirectResolution (typed result):
        - entity_refs: resolved entity names for pipeline filtering
        - answer: pre-computed Answer (e.g., comparative intersection)
        - refusal: pre-computed StructuralRefusal (empty intersection)

        Root cause for IndirectResolution vs List[Dict]:
        Line 1476 assigns the return to indirect_refs. Pattern 4
        adds comparative results to the same list as entity refs.
        retrieve() at line 1478+ iterates and checks source tags
        to distinguish them — blocked as hardcoded branching.
        Typed fields let retrieve() check answer/refusal directly.

        The query string is NEVER rewritten.
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            return IndirectResolution()

        doc = nlp(query)

        resolved: List[Dict[str, Any]] = []

        # --- Pattern 1: Speaker references → resolve to "user" entity ---
        for tok in doc:
            if tok.text.lower() in self._SPEAKER_REFS:
                resolved.append({
                    "name": "user",
                    "source": "speaker",
                    "anchor": tok.text,
                })

            # "speakers" / "users" as compound modifier
            if tok.dep_ == "compound" and tok.text.lower().rstrip("s") in self._SPEAKER_REFS:
                resolved.append({
                    "name": "user",
                    "source": "speaker",
                    "anchor": tok.text,
                })

        # --- Pattern 2: Possessive → traverse graph ---
        # Multi-pass chain walk (left-to-right):
        #
        # Root cause: spaCy sometimes attaches the possessor to the
        # head noun, skipping intermediate compound nouns.
        #   "speaker's girlfriend study" parses as:
        #     poss(speaker, study), compound(girlfriend, study)
        #   The possessor skips "girlfriend" and attaches to "study".
        #   Code using tok.head.text directly gets "study" instead
        #   of the relational noun "girlfriend".
        #
        # Two sub-cases for compounds between possessor and head:
        #   "girlfriend study" — "girlfriend" IS relational (graph
        #     traversal succeeds: user → girlfriend → Suki)
        #   "college roommate" — "college" is NOT relational (just
        #     a modifier; graph traversal fails). In this case,
        #     combine compound+head: "college roommate" as a single
        #     relation phrase.
        for tok in doc:
            if tok.dep_ == "poss" and tok.head.pos_ == "NOUN":
                anchor = tok.text
                if anchor.lower() in _FIRST_PERSON or anchor.lower() in self._SPEAKER_REFS:
                    anchor = "user"

                head = tok.head
                # Collect compound siblings of head between possessor
                # and head in linear order.
                compounds = sorted(
                    [c for c in head.children
                     if c.dep_ == "compound"
                     and c.pos_ == "NOUN"
                     and c.i > tok.i
                     and c.i < head.i],
                    key=lambda c: c.i,
                )

                if compounds:
                    # Try combined phrase FIRST (compounds + head).
                    # "college roommate" as a phrase matches
                    # "be_roommate_of" better than "college" alone.
                    # This prevents "college" from falsely resolving
                    # to Georgetown via study_at.
                    #
                    # Only fall back to individual compound chain
                    # walk if combined phrase fails — the chain walk
                    # is for genuine multi-link chains like
                    # "speaker's girlfriend('s) study".
                    combined = " ".join(
                        [c.text for c in compounds] + [head.text],
                    )
                    entity_name = self._traverse_relation(
                        user_id, anchor, combined,
                    )
                    if entity_name:
                        resolved.append({
                            "name": entity_name,
                            "source": "possessive",
                            "anchor": anchor,
                        })
                    else:
                        # Combined phrase failed. Try chain walk:
                        # each compound as a relational link.
                        current_anchor = anchor
                        for comp in compounds:
                            entity_name = self._traverse_relation(
                                user_id, current_anchor, comp.text,
                            )
                            if entity_name:
                                resolved.append({
                                    "name": entity_name,
                                    "source": "possessive",
                                    "anchor": current_anchor,
                                })
                                current_anchor = entity_name
                            else:
                                break
                else:
                    # No intermediate compounds — direct resolution.
                    # "Tariq's wife" → resolve Tariq, wife
                    relation_word = head.text
                    entity_name = self._traverse_relation(
                        user_id, anchor, relation_word,
                    )
                    if entity_name:
                        resolved.append({
                            "name": entity_name,
                            "source": "possessive",
                            "anchor": anchor,
                        })

            # "Tariqs wife" -> compound(wife, Tariqs)
            if tok.dep_ == "compound" and tok.pos_ == "PROPN" and tok.head.pos_ == "NOUN":
                anchor = tok.text.rstrip("s")
                relation_word = tok.head.text
                entity_name = self._traverse_relation(user_id, anchor, relation_word)
                if entity_name:
                    resolved.append({
                        "name": entity_name,
                        "source": "possessive",
                        "anchor": anchor,
                    })

        # --- Pattern 3: Relative clause → find entity by edge ---
        for tok in doc:
            if tok.dep_ != "relcl" or tok.pos_ != "VERB":
                continue
            if tok.head.text.lower() not in self._GENERIC_PERSON_NOUNS:
                continue
            verb = tok.lemma_.lower()
            obj_toks = [c for c in tok.children if c.dep_ in ("dobj", "attr", "pobj", "compound", "oprd")]
            # Include compound modifiers of the extracted tokens
            # (spaCy may attach 'guitar' as compound of 'work' in
            # 'plays guitar work' instead of as dobj of 'plays')
            expanded = []
            for ot in obj_toks:
                for cc in ot.children:
                    if cc.dep_ == "compound":
                        expanded.append(cc)
                expanded.append(ot)
            obj_text = " ".join(c.text for c in expanded)
            entity_name = self._find_entity_by_edge(user_id, verb, obj_text)
            if entity_name:
                resolved.append({
                    "name": entity_name,
                    "source": "relcl",
                    "anchor": f"{verb} {obj_text}",
                })
            else:
                # Relcl describes an entity that doesn't exist in
                # the graph ("person who plays guitar" but no guitar
                # edge). Refuse — don't fall through to generic pipeline.
                return IndirectResolution(
                    refusal=StructuralRefusal(
                        reason="relcl_entity_not_found",
                        text=REFUSAL_TEXT,
                        confidence=0.0,
                    ),
                )

        # --- Pattern 4: Comparative "X and Y verb the same noun" ---
        #
        # Root cause: "Do the speaker and Suki live in the same city?"
        # Current code only resolves entity1 ("speaker" → "user") via
        # Pattern 1. Entity2 ("Suki") is left to the generic entity
        # resolver. The intersection check (do both share a value for
        # the predicate?) cannot happen because the resolver does not
        # detect the coordinated-subject + identity-comparison pattern.
        #
        # Detection: coordinated subjects (nsubj + conj) with an
        # identity determiner ("same"/"identical" — closed grammatical
        # class of identity/comparison modifiers per UD, same exemption
        # as WH-words and possessives already in this codebase) modifying
        # a noun via amod.
        _IDENTITY_DETERMINERS = frozenset({"same", "identical"})
        for tok in doc:
            if tok.dep_ not in ("nsubj", "nmod"):
                continue
            conj_children = [c for c in tok.children if c.dep_ == "conj"]
            if not conj_children:
                continue
            # Check for identity determiner (closed grammatical class)
            identity_noun = None
            for c in doc:
                if (c.dep_ == "amod"
                        and c.text.lower() in _IDENTITY_DETERMINERS):
                    identity_noun = c.head.text.lower()
                    break
            if not identity_noun:
                continue

            entity1 = tok.text
            if (entity1.lower() in _FIRST_PERSON
                    or entity1.lower() in self._SPEAKER_REFS):
                entity1 = "user"

            entity2 = conj_children[0].text
            if (entity2.lower() in _FIRST_PERSON
                    or entity2.lower() in self._SPEAKER_REFS):
                entity2 = "user"

            verb_tok = tok.head
            # Use head lemma regardless of POS tag. Root cause:
            # spaCy sometimes parses verbs as NOUN (e.g., "work"
            # in "Do X and Y work at..."). The lemma is still the
            # semantic verb for intersection matching.
            verb_lemma = verb_tok.lemma_.lower() if verb_tok.lemma_ else None

            if verb_lemma:
                common = self._find_edge_intersection(
                    user_id, entity1, entity2,
                    verb_lemma, identity_noun,
                )
                if common is not None:
                    log.debug(
                        "Comparative: %s and %s -> %s",
                        entity1, entity2, common,
                    )
                    return IndirectResolution(
                        answer=Answer(
                            text=common,
                            source="comparative_intersection",
                            confidence=1.0,
                            convergence_details={
                                "entity1": entity1,
                                "entity2": entity2,
                            },
                        ),
                    )
                else:
                    log.debug(
                        "Comparative empty: %s and %s",
                        entity1, entity2,
                    )
                    return IndirectResolution(
                        refusal=StructuralRefusal(
                            reason="comparative_no_intersection",
                            text=REFUSAL_TEXT,
                            confidence=0.0,
                        ),
                    )

        # --- Pattern 5: "Who else" peer query ---
        # "Who else works at Palantir?" → find all subjects with
        # same predicate+object, excluding the asking user.
        # SPARQL equivalent: SELECT ?s WHERE { ?s ?p ?o } minus user.
        if _is_relational_peer_query(query):
            obj_hint = _extract_query_object_hint(query)
            query_verb = _extract_query_verb(query)
            if obj_hint and query_verb:
                verb_lemmas = _normalized_words(query_verb)
                peers = self._find_peer_entities(
                    user_id, verb_lemmas, obj_hint,
                )
                if peers:
                    if len(peers) == 1:
                        answer_text = peers[0]
                    elif len(peers) == 2:
                        answer_text = f"{peers[0]} and {peers[1]}"
                    else:
                        answer_text = ", ".join(peers[:-1]) + f", and {peers[-1]}"
                    return IndirectResolution(
                        answer=Answer(
                            text=answer_text,
                            source="peer_query",
                            confidence=0.9,
                            convergence_details={
                                "peer_count": len(peers),
                                "peers": peers,
                            },
                        ),
                    )
                else:
                    return IndirectResolution(
                        refusal=StructuralRefusal(
                            reason="no_peers_found",
                            text=REFUSAL_TEXT,
                            confidence=0.0,
                        ),
                    )

        # Deduplicate entity refs by name
        seen: Set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for r in resolved:
            key = r["name"].lower()
            if key not in seen:
                deduped.append(r)
                seen.add(key)

        if deduped:
            log.debug(
                "Indirect ref resolution: %r -> %s",
                query, [r["name"] for r in deduped],
            )

        return IndirectResolution(entity_refs=deduped)


    def _traverse_relation(
        self, user_id: int, anchor: str, relation_word: str,
    ) -> Optional[str]:
        """Find the entity on the OTHER side of anchor's edge whose
        predicate best matches relation_word.

        Searches both subject and object positions for the anchor.
        "Wei be_manager_of user" with anchor="user" returns "Wei".
        "Tariq married_to Suki" with anchor="Tariq" returns "Suki".

        Two-phase ranking:
        1. Lemma overlap filter: if the relation_word's lemma appears
           in any predicate's tokens (via _normalized_words), restrict
           to those edges. "roommate" lemma-matches "be_roommate_of"
           but NOT "work_at". This prevents false positives where
           cosine picks a semantically adjacent but wrong predicate.
        2. Cosine ranking within the filtered set (or all edges if
           no lemma match): max(predicate_cosine, edge_cosine) —
           same dual-signal as Entry stage.

        Returns the resolved entity name (argmax, no threshold),
        or None if no edges exist for the anchor.
        """
        rel_emb = embed_text(relation_word)
        rel_lemmas = _normalized_words(relation_word)

        with get_db_context() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object, "
                "       predicate_embedding, edge_embedding, "
                "       edge_schematic_category "
                "FROM relationships "
                "WHERE user_id = ? "
                "  AND COALESCE(is_current, 1) = 1 "
                "  AND tombstoned_at IS NULL "
                "  AND (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))",
                (user_id, anchor, anchor),
            ).fetchall()

        if not rows:
            # Fallback: anchor may be the speaker's name. Check if it
            # appears in relational_entities of 'user' edges.
            with get_db_context() as conn:
                user_row = conn.execute(
                    "SELECT 1 FROM relationships "
                    "WHERE user_id = ? "
                    "  AND LOWER(subject) = 'user' "
                    "  AND relational_entities LIKE ? "
                    "  AND tombstoned_at IS NULL LIMIT 1",
                    (user_id, f'%"{anchor}"%'),
                ).fetchone()
            if user_row:
                # Re-query with 'user' as anchor
                with get_db_context() as conn:
                    rows = conn.execute(
                        "SELECT subject, predicate, object, "
                        "       predicate_embedding, edge_embedding, "
                        "       edge_schematic_category "
                        "FROM relationships "
                        "WHERE user_id = ? "
                        "  AND tombstoned_at IS NULL "
                        "  AND (LOWER(subject) = 'user' OR LOWER(object) = 'user')",
                        (user_id,),
                    ).fetchall()
                if rows:
                    anchor = "user"
                else:
                    return None
            else:
                return None

        # Three-tier partition: lemma > schema > all (cosine ranking).
        # Root cause: "colleague" has no lemma overlap with any predicate.
        # Cosine fallback picks "work_at Palantir" over "be_manager_of"
        # because "colleague" embeds closer to "work" than "manager".
        # Schema tier narrows to edges in the same semantic category
        # (career), so cosine picks among career edges only.
        lemma_matched = []
        schema_matched = []
        all_rows = []
        # Infer schema from relation_word for tier 2
        rel_schema = None
        try:
            from app.engines.grammar_engine import (
                _noun_to_schema_via_wordnet, _VERB_CLASS_TO_SCHEMA,
                classify_verb_class,
            )
            rel_schema = _noun_to_schema_via_wordnet(relation_word)
            if not rel_schema or rel_schema == "uncategorized":
                vc = classify_verb_class(relation_word)
                rel_schema = _VERB_CLASS_TO_SCHEMA.get(vc)
        except Exception:
            pass

        for row in rows:
            pred_text = (row["predicate"] or "").replace("_", " ")
            pred_lemmas = _normalized_words(pred_text)
            pred_cos = _cosine_from_blob(rel_emb, row["predicate_embedding"])
            edge_cos = _cosine_from_blob(rel_emb, row["edge_embedding"])
            score = max(pred_cos, edge_cos)
            entry = (row, score)
            all_rows.append(entry)
            if rel_lemmas & pred_lemmas:
                lemma_matched.append(entry)
            elif rel_schema:
                edge_schema = (row["edge_schematic_category"] or "").lower()
                if edge_schema == rel_schema.lower():
                    schema_matched.append(entry)

        # Use tightest non-empty tier
        if lemma_matched:
            pool = lemma_matched
        elif schema_matched:
            pool = schema_matched
        else:
            pool = all_rows

        best_row, _ = max(pool, key=lambda pair: pair[1])
        # Return the OTHER side of the edge — not the anchor
        anchor_lower = anchor.lower()
        if best_row["subject"].lower() == anchor_lower:
            return best_row["object"]
        return best_row["subject"]

    def _find_entity_by_edge(
        self, user_id: int, verb: str, obj: str,
    ) -> Optional[str]:
        """Find any entity (subject) that has an edge matching verb+obj.

        Embeds "verb obj" as a single phrase, cosine-matches against
        edge_embedding. Returns the subject of the best match (argmax,
        no threshold), or None if no edges exist.
        """
        phrase_emb = embed_text(f"{verb} {obj}")

        with get_db_context() as conn:
            rows = conn.execute(
                "SELECT subject, object, source_text, edge_embedding "
                "FROM relationships "
                "WHERE user_id = ? "
                "  AND COALESCE(is_current, 1) = 1 "
                "  AND tombstoned_at IS NULL "
                "  AND edge_embedding IS NOT NULL",
                (user_id,),
            ).fetchall()

        if not rows:
            return None

        scored = [
            (row, _cosine_from_blob(phrase_emb, row["edge_embedding"]))
            for row in rows
        ]
        best_row, best_score = max(scored, key=lambda pair: pair[1])

        # Structural validation: the object noun must appear in the
        # matched edge (source_text or object field). Without this,
        # 'play guitar' falsely matches 'Suki plays piano' because
        # piano is cosine-nearest to guitar. This is a string check,
        # not a threshold — the word must be present.
        obj_words = _normalized_words(obj)
        edge_text = " ".join([
            (best_row["object"] if best_row["object"] else ""),
            (best_row["source_text"] if best_row["source_text"] else ""),
        ]).lower()
        edge_words = _normalized_words(edge_text)
        if obj_words and not (obj_words & edge_words):
            # Object noun not found — check via WordNet expansion
            if not _wordnet_noun_in_predicate(obj, edge_text):
                return None
        return best_row["subject"]

    def _find_edge_intersection(
        self,
        user_id: int,
        entity1: str,
        entity2: str,
        verb_lemma: str,
        noun_category: str,
    ) -> Optional[str]:
        """Find a common object shared by entity1 and entity2 for edges
        whose predicate matches verb_lemma.

        Evidence:
        - user live_in DC, Suki live_in DC → common = {"dc"} → "DC"
        - user work_at Palantir, Tariq work_at Meta → common = {} → None

        Root cause: Pattern 4 detects the comparative structure but
        needs a method to check set intersection of edge objects.
        This is purely structural — no cosine, no thresholds.

        Uses predicate lemma overlap (same filter as _traverse_relation)
        to select relevant edges, then set intersection on objects.
        """
        verb_lemmas = _normalized_words(verb_lemma)

        def _objects_for_entity(entity: str) -> Dict[str, str]:
            """Return {lowered_obj: original_obj} for matching edges."""
            with get_db_context() as conn:
                rows = conn.execute(
                    "SELECT predicate, object "
                    "FROM relationships "
                    "WHERE user_id = ? "
                    "  AND COALESCE(is_current, 1) = 1 "
                    "  AND tombstoned_at IS NULL "
                    "  AND (LOWER(subject) = LOWER(?))",
                    (user_id, entity),
                ).fetchall()
            result: Dict[str, str] = {}
            for row in rows:
                pred_lemmas = _normalized_words(
                    (row["predicate"] or "").replace("_", " "),
                )
                if verb_lemmas & pred_lemmas:
                    obj = (row["object"] or "").strip()
                    if obj:
                        result[obj.lower()] = obj
            return result

        objs1 = _objects_for_entity(entity1)
        objs2 = _objects_for_entity(entity2)
        common_keys = set(objs1.keys()) & set(objs2.keys())

        if common_keys:
            # Return original-cased form of first common object
            first_key = sorted(common_keys)[0]
            return objs1[first_key]

        # Fallback: when verb lemma matching fails ("go" vs "study_at"),
        # try matching noun_category ("school") against edge schema
        # categories via WordNet noun expansion. "school" → "education"
        # via _wordnet_noun_in_predicate hypernym traversal.
        if noun_category:
            def _objects_by_schema(entity: str) -> Dict[str, str]:
                with get_db_context() as conn:
                    rows = conn.execute(
                        "SELECT predicate, object, edge_schematic_category "
                        "FROM relationships "
                        "WHERE user_id = ? "
                        "  AND COALESCE(is_current, 1) = 1 "
                        "  AND tombstoned_at IS NULL "
                        "  AND (LOWER(subject) = LOWER(?))",
                        (user_id, entity),
                    ).fetchall()
                result: Dict[str, str] = {}
                for row in rows:
                    schema = (row["edge_schematic_category"] or "").strip()
                    pred_text = (row["predicate"] or "").replace("_", " ")
                    # Check noun_category against schema or predicate
                    if (_wordnet_noun_in_predicate(noun_category, schema)
                            or _wordnet_noun_in_predicate(noun_category, pred_text)):
                        obj = (row["object"] or "").strip()
                        if obj:
                            result[obj.lower()] = obj
                return result

            objs1_s = _objects_by_schema(entity1)
            objs2_s = _objects_by_schema(entity2)
            common_s = set(objs1_s.keys()) & set(objs2_s.keys())
            if common_s:
                first_key = sorted(common_s)[0]
                return objs1_s[first_key]

        return None

    def _find_peer_entities(
        self,
        user_id: int,
        verb_lemmas: Set[str],
        obj_hint: str,
    ) -> List[str]:
        """Find all subjects that share a predicate+object, excluding user.

        SPARQL equivalent: SELECT DISTINCT ?s WHERE { ?s ?p ?o }
        where ?p matches verb_lemmas and ?o matches obj_hint.
        """
        obj_lower = obj_hint.strip().lower()
        obj_emb = embed_text(obj_hint)

        with get_db_context() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object, edge_embedding "
                "FROM relationships "
                "WHERE user_id = ? "
                "  AND tombstoned_at IS NULL "
                "  AND COALESCE(is_current, 1) = 1",
                (user_id,),
            ).fetchall()

        peers: List[str] = []
        seen: Set[str] = set()

        for row in rows:
            subj = (row["subject"] or "").strip()
            if subj.lower() == "user" or subj.lower() in seen:
                continue

            pred_text = (row["predicate"] or "").replace("_", " ")
            pred_lemmas = _normalized_words(pred_text)
            if not (verb_lemmas & pred_lemmas):
                continue

            edge_obj = (row["object"] or "").strip().lower()
            if obj_lower in edge_obj or edge_obj in obj_lower:
                peers.append(subj)
                seen.add(subj.lower())
            else:
                cos = _cosine_from_blob(obj_emb, row["edge_embedding"])
                if cos > 0.7:
                    peers.append(subj)
                    seen.add(subj.lower())

        return sorted(peers)

    # ===================================================================
    # MOAT PIPELINE — Stages 1-5
    # ===================================================================

    def _stage1_entry(
        self, user_id: int, q_emb: np.ndarray, query_text: str,
    ) -> List[Candidate]:
        """Recall candidates from PQ cosine, edge cosine, and FTS BM25.
        Union by relationship_id. Annotate each with signal cosines.
        Cap at ENTRY_CAP."""

        seen: Dict[int, Candidate] = {}

        with get_db_context() as conn:
            # --- PQ cosine recall ---
            pq_rows = conn.execute(
                "SELECT pq.relationship_id, pq.question_embedding, "
                "       r.id, r.subject, r.predicate, r.object, "
                "       r.source_text, r.confidence, r.is_current, "
                "       r.tombstoned_at, r.edge_embedding, "
                "       r.predicate_embedding, r.cluster_id, "
                "       r.sequence_number, r.edge_emotional_label, "
                "       r.edge_schematic_category, r.edge_emotional_valence, "
                "       r.source_tag, r.is_historical, "
                "       r.edge_temporal_context, r.temporal_expression, "
                "       r.relational_entities "
                "FROM predicted_queries pq "
                "JOIN relationships r ON r.id = pq.relationship_id "
                "WHERE pq.user_id = ? "
                "  AND r.tombstoned_at IS NULL",
                (user_id,),
            ).fetchall()

            for row in pq_rows:
                rid = row["relationship_id"]
                pq_cos = _cosine_from_blob(q_emb, row["question_embedding"])
                edge_cos = _cosine_from_blob(q_emb, row["edge_embedding"])
                if rid in seen:
                    c = seen[rid]
                    c.pq_cosine = max(c.pq_cosine, pq_cos)
                    c.edge_cosine = max(c.edge_cosine, edge_cos)
                    c.entry_cosine = max(c.pq_cosine, c.edge_cosine)
                    c.source_stages.add("pq")
                else:
                    edge = {
                        "id": row["id"],
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "source_text": row["source_text"],
                        "confidence": row["confidence"],
                        "is_current": row["is_current"],
                        "tombstoned_at": row["tombstoned_at"],
                        "cluster_id": row["cluster_id"],
                        "sequence_number": row["sequence_number"],
                        "edge_emotional_label": row["edge_emotional_label"],
                        "edge_schematic_category": row["edge_schematic_category"],
                        "edge_emotional_valence": row["edge_emotional_valence"],
                        "source_tag": row["source_tag"],
                        "is_historical": row["is_historical"],
                        "edge_temporal_context": row["edge_temporal_context"],
                        "temporal_expression": row["temporal_expression"],
                        "relational_entities": row["relational_entities"],
                        "edge_embedding": row["edge_embedding"],
                        "predicate_embedding": row["predicate_embedding"],
                    }
                    seen[rid] = Candidate(
                        relationship_id=rid,
                        edge=edge,
                        pq_cosine=pq_cos,
                        edge_cosine=edge_cos,
                        entry_cosine=max(pq_cos, edge_cos),
                        source_stages={"pq"},
                    )

            # --- Edge embedding cosine recall ---
            edge_rows = conn.execute(
                "SELECT id, subject, predicate, object, source_text, "
                "       confidence, is_current, tombstoned_at, "
                "       edge_embedding, predicate_embedding, cluster_id, "
                "       sequence_number, edge_emotional_label, "
                "       edge_schematic_category, edge_emotional_valence, "
                "       source_tag, is_historical, "
                "       edge_temporal_context, temporal_expression, "
                "       relational_entities "
                "FROM relationships "
                "WHERE user_id = ? "
                "  AND tombstoned_at IS NULL "
                "  AND edge_embedding IS NOT NULL",
                (user_id,),
            ).fetchall()

            for row in edge_rows:
                rid = row["id"]
                edge_cos = _cosine_from_blob(q_emb, row["edge_embedding"])
                if rid in seen:
                    c = seen[rid]
                    c.edge_cosine = max(c.edge_cosine, edge_cos)
                    c.entry_cosine = max(c.pq_cosine, c.edge_cosine)
                    c.source_stages.add("edge")
                else:
                    edge = {
                        "id": row["id"],
                        "subject": row["subject"],
                        "predicate": row["predicate"],
                        "object": row["object"],
                        "source_text": row["source_text"],
                        "confidence": row["confidence"],
                        "is_current": row["is_current"],
                        "tombstoned_at": row["tombstoned_at"],
                        "cluster_id": row["cluster_id"],
                        "sequence_number": row["sequence_number"],
                        "edge_emotional_label": row["edge_emotional_label"],
                        "edge_schematic_category": row["edge_schematic_category"],
                        "edge_emotional_valence": row["edge_emotional_valence"],
                        "source_tag": row["source_tag"],
                        "is_historical": row["is_historical"],
                        "edge_temporal_context": row["edge_temporal_context"],
                        "temporal_expression": row["temporal_expression"],
                        "relational_entities": row["relational_entities"],
                        "edge_embedding": row["edge_embedding"],
                        "predicate_embedding": row["predicate_embedding"],
                    }
                    seen[rid] = Candidate(
                        relationship_id=rid,
                        edge=edge,
                        edge_cosine=edge_cos,
                        entry_cosine=edge_cos,
                        source_stages={"edge"},
                    )

            # --- FTS BM25 recall ---
            fts_query = " OR ".join(_split_words(query_text))
            if fts_query:
                try:
                    fts_rows = conn.execute(
                        "SELECT r.id, r.subject, r.predicate, r.object, "
                        "       r.source_text, r.confidence, r.is_current, "
                        "       r.tombstoned_at, r.edge_embedding, "
                        "       r.predicate_embedding, r.cluster_id, "
                        "       r.sequence_number, r.edge_emotional_label, "
                        "       r.edge_schematic_category, "
                        "       r.edge_emotional_valence, r.source_tag, "
                        "       r.is_historical, r.edge_temporal_context, "
                        "       r.temporal_expression, "
                        "       r.relational_entities "
                        "FROM relationships_fts fts "
                        "JOIN relationships r ON r.rowid = fts.rowid "
                        "WHERE relationships_fts MATCH ? "
                        "  AND r.user_id = ? "
                        "  AND r.tombstoned_at IS NULL "
                        "LIMIT ?",
                        (fts_query, user_id, ENTRY_CAP),
                    ).fetchall()

                    for row in fts_rows:
                        rid = row["id"]
                        if rid in seen:
                            seen[rid].source_stages.add("fts")
                        else:
                            edge_cos = _cosine_from_blob(
                                q_emb, row["edge_embedding"],
                            )
                            edge = {
                                "id": row["id"],
                                "subject": row["subject"],
                                "predicate": row["predicate"],
                                "object": row["object"],
                                "source_text": row["source_text"],
                                "confidence": row["confidence"],
                                "is_current": row["is_current"],
                                "tombstoned_at": row["tombstoned_at"],
                                "cluster_id": row["cluster_id"],
                                "sequence_number": row["sequence_number"],
                                "edge_emotional_label": row["edge_emotional_label"],
                                "edge_schematic_category": row["edge_schematic_category"],
                                "edge_emotional_valence": row["edge_emotional_valence"],
                                "source_tag": row["source_tag"],
                                "is_historical": row["is_historical"],
                                "edge_temporal_context": row["edge_temporal_context"],
                                "temporal_expression": row["temporal_expression"],
                                "relational_entities": row["relational_entities"],
                                "edge_embedding": row["edge_embedding"],
                                "predicate_embedding": row["predicate_embedding"],
                            }
                            seen[rid] = Candidate(
                                relationship_id=rid,
                                edge=edge,
                                edge_cosine=edge_cos,
                                entry_cosine=edge_cos,
                                source_stages={"fts"},
                            )
                except Exception:
                    log.debug("FTS recall failed", exc_info=True)

        # Sort by entry_cosine desc, cap at ENTRY_CAP
        pool = sorted(seen.values(), key=lambda c: -c.entry_cosine)
        return pool[:ENTRY_CAP]

    def _stage2_expand(
        self,
        user_id: int,
        candidates: List[Candidate],
        resolved_entities: List[Dict[str, Any]],
    ) -> List[Candidate]:
        """Union in edges that touch resolved entities.
        Set entity_overlap and non_self_entity_overlap on each candidate."""

        if not resolved_entities:
            return candidates

        entity_names = {e["name"].lower() for e in resolved_entities}
        existing_ids = {c.relationship_id for c in candidates}

        # Fetch edges touching resolved entities
        with get_db_context() as conn:
            for ent in resolved_entities:
                ent_name = ent["name"]
                rows = conn.execute(
                    "SELECT id, subject, predicate, object, source_text, "
                    "       confidence, is_current, tombstoned_at, "
                    "       edge_embedding, predicate_embedding, cluster_id, "
                    "       sequence_number, edge_emotional_label, "
                    "       edge_schematic_category, edge_emotional_valence, "
                    "       source_tag, is_historical, "
                    "       edge_temporal_context, temporal_expression, "
                    "       relational_entities "
                    "FROM relationships "
                    "WHERE user_id = ? "
                    "  AND tombstoned_at IS NULL "
                    "  AND (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?)) "
                    "LIMIT ?",
                    (user_id, ent_name, ent_name, ENTRY_CAP),
                ).fetchall()

                for row in rows:
                    rid = row["id"]
                    if rid not in existing_ids:
                        edge = {
                            "id": row["id"],
                            "subject": row["subject"],
                            "predicate": row["predicate"],
                            "object": row["object"],
                            "source_text": row["source_text"],
                            "confidence": row["confidence"],
                            "is_current": row["is_current"],
                            "tombstoned_at": row["tombstoned_at"],
                            "cluster_id": row["cluster_id"],
                            "sequence_number": row["sequence_number"],
                            "edge_emotional_label": row["edge_emotional_label"],
                            "edge_schematic_category": row["edge_schematic_category"],
                            "edge_emotional_valence": row["edge_emotional_valence"],
                            "source_tag": row["source_tag"],
                            "is_historical": row["is_historical"],
                            "edge_temporal_context": row["edge_temporal_context"],
                            "temporal_expression": row["temporal_expression"],
                            "relational_entities": row["relational_entities"],
                            "edge_embedding": row["edge_embedding"],
                            "predicate_embedding": row["predicate_embedding"],
                        }
                        candidates.append(Candidate(
                            relationship_id=rid,
                            edge=edge,
                            source_stages={"expand"},
                        ))
                        existing_ids.add(rid)

        # Annotate entity overlap on all candidates
        for c in candidates:
            subj_lower = (c.edge.get("subject") or "").lower()
            obj_lower = (c.edge.get("object") or "").lower()
            overlap = 0
            non_self = 0
            for en in entity_names:
                if subj_lower == en or obj_lower == en:
                    overlap += 1
                    if en != "user":
                        non_self += 1
            c.entity_overlap = overlap
            c.non_self_entity_overlap = non_self

        return candidates

    def _stage3_group(
        self,
        candidates: List[Candidate],
        resolved_entities: List[Dict[str, Any]],
    ) -> List[Candidate]:
        """Cluster coherence stage. Find anchor cluster from best
        entity-overlap candidates, keep anchor + transitive neighbors."""

        if not resolved_entities or not candidates:
            return candidates

        # Find anchor: candidate with highest entity overlap
        best = max(candidates, key=lambda c: (c.entity_overlap, c.entry_cosine))
        anchor_cluster = best.edge.get("cluster_id")

        if not anchor_cluster:
            return candidates

        # Collect participant sets per cluster
        cluster_participants: Dict[str, Set[str]] = defaultdict(set)
        for c in candidates:
            cid = c.edge.get("cluster_id")
            if cid:
                s = (c.edge.get("subject") or "").lower()
                o = (c.edge.get("object") or "").lower()
                if s:
                    cluster_participants[cid].add(s)
                if o:
                    cluster_participants[cid].add(o)

        # Keep anchor cluster and neighbors that share participants
        anchor_parts = cluster_participants.get(anchor_cluster, set())
        keep_clusters = {anchor_cluster}
        for cid, parts in cluster_participants.items():
            if cid != anchor_cluster and parts & anchor_parts:
                keep_clusters.add(cid)

        # Filter: keep candidates in keep_clusters OR with no cluster
        survivors = []
        for c in candidates:
            cid = c.edge.get("cluster_id")
            if not cid or cid in keep_clusters:
                survivors.append(c)

        return survivors if survivors else candidates

    def _stage4_relate(
        self,
        candidates: List[Candidate],
        resolved_entities: List[Dict[str, Any]],
    ) -> List[Candidate]:
        """Drop disconnected edges. If no resolved entities, passthrough.
        Otherwise keep candidates with reachable relation to query entity."""

        if not resolved_entities:
            return candidates

        entity_names = {e["name"].lower() for e in resolved_entities}

        # Build adjacency from candidate edges
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for c in candidates:
            s = (c.edge.get("subject") or "").lower()
            o = (c.edge.get("object") or "").lower()
            if s and o:
                adjacency[s].add(o)
                adjacency[o].add(s)

        # BFS from entity names to find reachable nodes
        reachable: Set[str] = set()
        queue = list(entity_names)
        visited: Set[str] = set()
        max_hops = 3
        hop_map: Dict[str, int] = {e: 0 for e in entity_names}

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            reachable.add(node)
            current_hops = hop_map.get(node, 0)
            if current_hops < max_hops:
                for neighbor in adjacency.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
                        if neighbor not in hop_map:
                            hop_map[neighbor] = current_hops + 1

        # Filter candidates: keep if subject or object is reachable
        survivors = []
        for c in candidates:
            s = (c.edge.get("subject") or "").lower()
            o = (c.edge.get("object") or "").lower()
            if s in reachable or o in reachable:
                c.hops_to_entity = min(
                    hop_map.get(s, 999), hop_map.get(o, 999),
                )
                survivors.append(c)

        return survivors if survivors else candidates

    def _stage5_exit(
        self, q_emb: np.ndarray, candidates: List[Candidate],
    ) -> List[Candidate]:
        """Final ordering. Compute predicate_cosine, exit_cosine.
        Sort by exit_cosine desc, sequence_number desc."""

        for c in candidates:
            c.predicate_cosine = _cosine_from_blob(
                q_emb, c.edge.get("predicate_embedding"),
            )
            c.exit_cosine = max(
                c.pq_cosine, c.edge_cosine, c.predicate_cosine,
            )

        candidates.sort(key=lambda c: (
            -c.exit_cosine,
            -(c.edge.get("sequence_number") or 0),
        ))
        return candidates

    # ===================================================================
    # FULL MOAT PIPELINE
    # ===================================================================

    def _run_moat_pipeline(
        self, user_id: int, query_text: str,
    ) -> Tuple[List[Candidate], np.ndarray, List[Dict[str, Any]]]:
        """Run stages 1-5. Returns (survivors, q_emb, resolved_entities)."""

        q_emb = embed_text(query_text)

        # Stage 1: ENTRY
        candidates = self._stage1_entry(user_id, q_emb, query_text)
        log.debug("Stage 1 ENTRY: %d candidates", len(candidates))

        if not candidates:
            return [], q_emb, []

        # Resolve entities once for the query
        from app.engines.entity_resolver import resolve_query_entities
        resolved_entities = resolve_query_entities(user_id, query_text)
        seen_entities = {e["name"].lower() for e in resolved_entities}
        for name in _extract_query_entity_fallback(query_text):
            if name.lower() not in seen_entities:
                resolved_entities.insert(0, {
                    "id": None,
                    "name": name,
                    "entity_type": "UNKNOWN",
                    "score": 1.0,
                })
                seen_entities.add(name.lower())

        # Stage 2: EXPAND
        candidates = self._stage2_expand(user_id, candidates, resolved_entities)
        log.debug("Stage 2 EXPAND: %d candidates", len(candidates))

        # Stage 3: GROUP
        candidates = self._stage3_group(candidates, resolved_entities)
        log.debug("Stage 3 GROUP: %d candidates", len(candidates))

        # Stage 4: RELATE
        candidates = self._stage4_relate(candidates, resolved_entities)
        log.debug("Stage 4 RELATE: %d candidates", len(candidates))

        # Stage 5: EXIT
        candidates = self._stage5_exit(q_emb, candidates)
        log.debug("Stage 5 EXIT: %d candidates", len(candidates))

        return candidates, q_emb, resolved_entities

    # ===================================================================
    # FACTS LOOKUP
    # ===================================================================

    def _load_facts_for_entity(
        self, user_id: int, entity_name: str, query_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Load canonical fact for entity matching query intent.
        Returns {key, value, confidence} or None."""

        if not entity_name:
            return None

        q_words = _normalized_words(query_text)

        with get_db_context() as conn:
            rows = conn.execute(
                "SELECT key, value, confidence, embedding "
                "FROM facts WHERE user_id = ?",
                (user_id,),
            ).fetchall()

        if not rows:
            return None

        # Match by verb-class overlap with query words.
        # Entity name is excluded from overlap — only verb/schema words
        # count. This prevents "home::live_in::Dad" from matching a
        # "love" query just because "Dad" is in both.
        best_match = None
        best_overlap = 0
        entity_lower = entity_name.lower()
        entity_words = _normalized_words(entity_name)

        for row in rows:
            key = row["key"] or ""
            # Fact key format: schema::verb_class::subject
            parts = key.split("::")
            if len(parts) < 3:
                continue
            fact_subject = parts[-1].lower()
            # Only match facts whose subject IS the query entity.
            # 'user' facts match when query entity is first-person.
            if fact_subject == entity_lower:
                pass  # direct match
            elif fact_subject == "user" and entity_lower in _FIRST_PERSON:
                pass  # first-person match
            else:
                continue

            key_words = _normalized_words(key) - entity_words
            overlap = len((q_words - entity_words) & key_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = {
                    "key": key,
                    "value": row["value"],
                    "confidence": row["confidence"],
                }

        return best_match

    # ===================================================================
    # VERIFICATION LOOP
    # ===================================================================

    def _infer_query_schema(self, query_text: str) -> Optional[str]:
        """Infer schematic category from query using grammar engine.

        Strategy 1: content noun IS a schema name (hobby, career, health).
        Strategy 2: noun-to-schema via WordNet hypernym closure.
        Strategy 3: verb class → schema mapping."""
        try:
            from app.engines.grammar_engine import (
                _get_nlp, _get_root, classify_verb_class,
                _reclassify_location_by_object, _extract_schematic,
                _VERB_CLASS_TO_SCHEMA, _noun_to_schema_via_wordnet,
            )
            nlp = _get_nlp()
            doc = nlp(query_text)
            root = _get_root(doc)

            # Build known schema set from DB + verb class map
            known = set(_VERB_CLASS_TO_SCHEMA.values())
            try:
                with get_db_context() as conn:
                    for row in conn.execute(
                        "SELECT DISTINCT edge_schematic_category "
                        "FROM relationships WHERE edge_schematic_category IS NOT NULL"
                    ).fetchall():
                        if row[0]:
                            known.add(row[0].lower())
            except Exception:
                pass

            # Strategy 1: content noun IS a schema name
            for tok in doc:
                if tok.pos_ == "NOUN" and not tok.is_stop:
                    if tok.lemma_.lower() in known:
                        return tok.lemma_.lower()

            # Strategy 2: noun-to-schema via WordNet
            for tok in doc:
                if tok.pos_ == "NOUN" and not tok.is_stop:
                    ns = _noun_to_schema_via_wordnet(tok.lemma_)
                    if ns and ns != "uncategorized":
                        return ns

            # Strategy 3: verb class
            if root is not None:
                v = root
                if root.pos_ == "AUX":
                    for ch in root.children:
                        if ch.dep_ in ("xcomp", "ccomp") and ch.pos_ == "VERB":
                            v = ch
                            break
                vc = classify_verb_class(v.lemma_)
                vc = _reclassify_location_by_object(doc, v, vc)
                s = _extract_schematic(doc, root, vc)
                if s and s != "uncategorized":
                    return s
        except Exception:
            pass
        return None

    def _check_entity_match(
        self,
        candidate: Candidate,
        query_entity: Optional[str],
    ) -> bool:
        """Entity gate: does this edge mention the query entity?
        This is the ONLY hard filter in the verification loop.
        The moat pipeline's cosine ranking handles relevance;
        entity matching handles identity (who is this about?)."""
        edge = candidate.edge
        subj_text = edge.get("subject") or ""
        obj_text = edge.get("object") or ""
        rel_ents = edge.get("relational_entities") or ""
        return _edge_mentions_entity(subj_text, obj_text, query_entity, rel_ents)

    def _render_answer_text(
        self,
        edge: Dict[str, Any],
        query_entity: Optional[str],
        query_text: str,
    ) -> str:
        """WH-type-aware answer rendering (from committed 92% version).

        Uses parse_expected_answer_type to select the right field:
        - TIME  → temporal_expression or resolved_event_date or object
        - PERSON → the entity that ISN'T the query entity ('user' is never the answer)
        - LOCATION → object
        - default → object (short noun phrase)
        """
        from app.engines.wh_type import parse_expected_answer_type

        subj = (edge.get("subject") or "").strip()
        obj = (edge.get("object") or "").strip()
        source_text = (edge.get("source_text") or "").strip()

        expected_type = parse_expected_answer_type(query_text)

        if expected_type == "TIME":
            answer = (
                edge.get("temporal_expression")
                or edge.get("resolved_event_date")
                or obj
            )
        elif expected_type == "PERSON":
            # Return the entity that ISN'T the query entity.
            # 'user' is canonical first-person, never the answer.
            subj_lower = subj.lower()
            if subj_lower == "user":
                answer = obj
            elif query_entity and query_entity.lower() in subj_lower:
                answer = obj
            else:
                answer = subj
        elif expected_type == "LOCATION":
            answer = obj or subj
        else:
            answer = obj

        # Fallback to source_text if answer is empty
        if not answer or len(answer.strip()) <= 1:
            answer = source_text

        return (answer or "").strip()

    def _check_existence(
        self,
        candidate: Candidate,
        canonical_fact: Optional[Dict[str, Any]],
        temporal_mode: str = "current",
    ) -> bool:
        """Existence check: temporal-mode-aware edge filtering.

        temporal_mode:
          'current'    — default, accept only is_current=1 edges
          'historical' — accept only is_current=0 or is_historical=1 edges
          'any'        — accept both current and historical edges
        """

        edge = candidate.edge

        if edge.get("tombstoned_at") is not None:
            return False

        is_current = edge.get("is_current")
        is_hist = edge.get("is_historical")

        if temporal_mode == "historical":
            # Only accept historical edges
            if is_hist:
                pass  # explicitly historical — accept
            elif is_current is not None and is_current:
                return False  # explicitly current — reject
        elif temporal_mode == "current":
            # Only accept current edges (original behavior)
            if is_current is not None and not is_current:
                return False
        # temporal_mode == "any" — accept both

        # If canonical fact exists, candidate object must match fact value
        if canonical_fact:
            fact_value = (canonical_fact["value"] or "").strip().lower()
            edge_object = (edge.get("object") or "").strip().lower()
            if fact_value and edge_object != fact_value:
                return False

        return True

    def _verification_loop(
        self,
        candidates: List[Candidate],
        query_entity: Optional[str],
        canonical_fact: Optional[Dict[str, Any]],
        query_text: str,
        q_emb: np.ndarray,
        temporal_mode: str = "current",
    ) -> Union[Answer, StructuralRefusal]:
        """Schema-aware verification with entity filtering.

        Philosophy (from committed 92% version):
        1. Trust the moat pipeline's cosine ranking for relevance
        2. Schema re-ranking: infer schema from query, boost matching edges
        3. Entity match: hard gate (edge must mention query entity)
        4. Existence check: temporal-mode-aware
        5. First passing candidate wins

        Schema re-ranking prevents wrong-category answers (returning
        'Google' for a 'hobby' query) without needing strict predicate
        lemma gating that kills category queries."""

        # Infer query schema for re-ranking
        inferred_schema = self._infer_query_schema(query_text)

        # exit_cosine primary, schema secondary. PQ exact matches
        # (cosine=1.0) always win. Schema breaks ties among
        # similar-cosine candidates.
        schema_l = (inferred_schema or "").lower()
        candidates.sort(
            key=lambda c: (
                -c.exit_cosine,
                -(1 if schema_l and (c.edge.get("edge_schematic_category") or "").lower() == schema_l else 0),
                -(c.edge.get("sequence_number") or 0),
            ),
        )

        # Extract query verb for predicate alignment.
        #
        # Root cause: the old loop accepted the first entity+existence
        # passing candidate by exit_cosine. Exit cosine is dominated by
        # entity name overlap (proper nouns), so "Does Suki cook?" picked
        # the work_at edge (highest Suki cosine) instead of cook.
        #
        # Fix: two-phase approach.
        # Phase 1: collect all entity+existence-passing candidates.
        # Phase 2: partition into predicate-aligned (WordNet/lemma match
        #          to query verb) vs non-aligned. If any aligned exist,
        #          pick the best (by exit_cosine) among aligned. Otherwise
        #          pick the best among all. This is a structural filter
        #          (same pattern as _traverse_relation's lemma partition).
        query_verb = _extract_query_verb(query_text)

        # Phase 1: collect verified candidates
        verified: List[Candidate] = []
        rejected_ids: Set[int] = set()

        for c in candidates:
            rid = c.relationship_id
            if rid in rejected_ids:
                continue

            # Gate 1: Entity match (is this edge ABOUT the right entity?)
            if not self._check_entity_match(c, query_entity):
                rejected_ids.add(rid)
                continue

            # Gate 2: Existence check (temporal-aware)
            if not self._check_existence(c, canonical_fact, temporal_mode):
                rejected_ids.add(rid)
                continue

            verified.append(c)

        if not verified:
            return StructuralRefusal(
                reason="no_candidate_passed_verification",
                text=REFUSAL_TEXT,
                confidence=0.0,
                survivors=0,
                convergence_details={
                    "total_candidates": len(candidates),
                    "rejected": len(rejected_ids),
                },
            )

        # Phase 2: predicate alignment — three-tier filter cascade.
        #
        # Same partition pattern as _traverse_relation (lines 1000-1017):
        # successively broader structural filters until candidates found.
        #
        # Tier 1 (precise): lemma overlap — "cook" ↔ "cook", "study" ↔
        #   "study_at". Catches exact predicate matches.
        # Tier 2 (semantic): WordNet verb match + noun-in-predicate —
        #   "exercise" ↔ "run" (shared synsets), "allergy" ↔
        #   "be_allergic_to" (derivational forms). Catches paraphrases.
        # Tier 3 (fallback): all verified candidates, exit_cosine order.
        #
        # Root cause for tiers (traced above): when only Tier 3 is used,
        # exit_cosine is dominated by entity name overlap (proper nouns),
        # so "Does Suki cook?" picks work_at (highest Suki cosine)
        # instead of cook.
        lemma_aligned: List[Candidate] = []
        if query_verb:
            verb_lemmas = _normalized_words(query_verb)
            for c in verified:
                pred_text = (c.edge.get("predicate") or "").replace("_", " ")
                if verb_lemmas & _normalized_words(pred_text):
                    lemma_aligned.append(c)

        # Pool selection: lemma-aligned candidates preferred when
        # available (structural precision). Otherwise fall through
        # to all verified candidates in their original exit_cosine
        # order — PQ cosine is a reliable signal for WH queries.
        pool = lemma_aligned if lemma_aligned else verified
        best_c = pool[0]

        # Predicate absence for yes/no queries (CWA negation).
        # Same _is_yes_no_query check already used in retrieve() at
        # the post-loop CWA path (line ~2318).
        #
        # Root cause: WordNet verb match is too broad for absence —
        # "cook" matches "work" via shared synset lemma "make".
        # Tighter check: lemma overlap (lemma_aligned) plus noun-in-
        # predicate on the query object hint. _wordnet_noun_in_predicate
        # catches "allergy" ↔ "be_allergic_to" via derivational forms.
        if _is_yes_no_query(query_text) and query_verb and not lemma_aligned:
            obj_hint = _extract_query_object_hint(query_text)
            noun_match_found = False
            if obj_hint:
                for c in verified:
                    pred_text = (c.edge.get("predicate") or "").replace("_", " ")
                    obj_text = (c.edge.get("object") or "").strip()
                    combined = f"{pred_text} {obj_text}"
                    if _wordnet_noun_in_predicate(obj_hint, combined):
                        noun_match_found = True
                        break
            if not noun_match_found:
                return StructuralRefusal(
                    reason="predicate_absence_confirmed",
                    text=REFUSAL_TEXT,
                    confidence=0.0,
                    survivors=0,
                    convergence_details={
                        "query_verb": query_verb,
                        "object_hint": obj_hint,
                        "entity": query_entity,
                        "verified_count": len(verified),
                    },
                )

        # For yes/no queries where predicate matched but the query
        # specifies an object (e.g., "instruments" in "Does speaker
        # play any instruments?"), check if the edge object aligns.
        # "play soccer" does not answer "play instruments" — refuse.
        #
        # Root cause: lemma_aligned catches the play edge, but the
        # edge object "soccer" != query object hint "instruments".
        # Without this check, the system returns "soccer" for a
        # yes/no query about instruments.
        if _is_yes_no_query(query_text) and lemma_aligned:
            obj_hint = _extract_query_object_hint(query_text)
            if obj_hint:
                # Check if obj_hint matches best candidate's object
                best_obj = (best_c.edge.get("object") or "").strip()
                hint_words = _normalized_words(obj_hint)
                obj_words = _normalized_words(best_obj)
                if not (hint_words & obj_words):
                    # Lemma-level mismatch, try WordNet noun match
                    if not _wordnet_noun_in_predicate(obj_hint, best_obj):
                        return StructuralRefusal(
                            reason="predicate_absence_confirmed",
                            text=REFUSAL_TEXT,
                            confidence=0.0,
                            survivors=0,
                            convergence_details={
                                "query_verb": query_verb,
                                "object_hint": obj_hint,
                                "edge_object": best_obj,
                                "entity": query_entity,
                            },
                        )

        edge = best_c.edge
        answer_text = self._render_answer_text(
            edge, query_entity, query_text,
        )

        return Answer(
            text=answer_text,
            subject=edge.get("subject"),
            predicate=edge.get("predicate"),
            object=edge.get("object"),
            confidence=best_c.exit_cosine,
            source=self._best_source(best_c),
            survivors=len(pool),
            convergence_details={
                "exit_cosine": best_c.exit_cosine,
                "pq_cosine": best_c.pq_cosine,
                "edge_cosine": best_c.edge_cosine,
                "predicate_cosine": best_c.predicate_cosine,
                "entity_overlap": best_c.entity_overlap,
                "relationship_id": best_c.relationship_id,
                "source_stages": list(best_c.source_stages),
                "inferred_schema": inferred_schema,
            },
        )

    @staticmethod
    def _best_source(c: Candidate) -> str:
        if c.pq_cosine >= c.edge_cosine and c.pq_cosine >= c.predicate_cosine:
            return "pq_cosine"
        if c.edge_cosine >= c.predicate_cosine:
            return "edge_cosine"
        return "predicate_cosine"

    # ===================================================================
    # ENTITY EXISTENCE CHECK (for negation/absence confirmation)
    # ===================================================================

    def _entity_exists(self, user_id: int, entity_name: str) -> bool:
        """Check if entity has ANY edges in the graph (CWA check).
        Returns True if the entity appears as subject or object
        in at least one relationship for this user."""
        if not entity_name:
            return False
        with get_db_context() as conn:
            # Check entities table first
            row = conn.execute(
                "SELECT 1 FROM entities WHERE user_id = ? "
                "AND LOWER(name) = LOWER(?) LIMIT 1",
                (user_id, entity_name),
            ).fetchone()
            if row:
                return True
            # Fallback: check relationships
            ent = entity_name
            if ent.lower() in _FIRST_PERSON:
                ent = "user"
            row = conn.execute(
                "SELECT 1 FROM relationships WHERE user_id = ? "
                "AND (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?)) "
                "AND tombstoned_at IS NULL LIMIT 1",
                (user_id, ent, ent),
            ).fetchone()
            return row is not None

    # ===================================================================
    # AGGREGATION LOOP (counting / listing)
    # ===================================================================

    def _aggregate_loop(
        self,
        candidates: List[Candidate],
        query_entity: Optional[str],
        canonical_fact: Optional[Dict[str, Any]],
        query_text: str,
        q_emb: np.ndarray,
        temporal_mode: str = "current",
        aggregate_type: str = "count",
    ) -> Union[Answer, StructuralRefusal]:
        """Collect ALL verified candidates, deduplicate by object,
        render as count or comma-separated list.

        Same coherence + existence checks as _verification_loop,
        but collects instead of returning on first pass."""

        query_verb = _extract_query_verb(query_text)
        # For list queries with relative clauses (e.g., "Name everyone
        # who plays soccer"), the ROOT verb is "name" (command verb),
        # not the semantic predicate "play" (relcl verb). Extract the
        # relcl verb for predicate filtering.
        if query_verb and query_verb in ('name', 'list', 'tell'):
            try:
                import spacy
                _nlp = spacy.load('en_core_web_sm')
                _doc = _nlp(query_text)
                for _tok in _doc:
                    if _tok.dep_ == 'relcl' and _tok.pos_ == 'VERB':
                        query_verb = _tok.lemma_.lower()
                        break
            except Exception:
                pass
        verb_lemmas = _normalized_words(query_verb) if query_verb else set()
        results: List[str] = []
        seen_objects: set = set()

        for c in candidates:
            if not self._check_entity_match(c, query_entity):
                continue
            if not self._check_existence(c, canonical_fact, temporal_mode):
                continue
            # Predicate alignment: for aggregate queries, only include
            # candidates whose predicate matches the query verb.
            # Root cause: with entity_gate=None for list queries,
            # ALL edges pass; predicate filtering restricts to the
            # asked predicate (e.g., "play" for "who plays soccer").
            if verb_lemmas:
                pred_text = (c.edge.get("predicate") or "").replace("_", " ")
                if not (verb_lemmas & _normalized_words(pred_text)):
                    continue

            # Deduplicate by normalized object text
            obj = (c.edge.get("object") or "").strip().lower()
            subj = (c.edge.get("subject") or "").strip().lower()

            # For "who" queries, collect subjects; for others, collect objects
            if aggregate_type in ("count", "list") and query_text.lower().startswith(("how many", "name everyone", "who all")):
                # Collect subjects (entities matching the predicate+object)
                value = (c.edge.get("subject") or "").strip()
                if value.lower() == "user":
                    value = "I"
            else:
                value = self._render_answer_text(c.edge, query_entity, query_text)

            norm = value.strip().lower()
            if norm and norm not in seen_objects:
                seen_objects.add(norm)
                results.append(value)

        if not results:
            return StructuralRefusal(
                reason="no_candidate_passed_verification",
                text=REFUSAL_TEXT,
                confidence=0.0,
                survivors=0,
            )

        if aggregate_type == "count":
            # Include entity names alongside the count so the answer
            # is informative: "2 (I and Suki)" instead of just "2".
            if len(results) == 1:
                names = results[0]
            elif len(results) == 2:
                names = f"{results[0]} and {results[1]}"
            else:
                names = ", ".join(results[:-1]) + f", and {results[-1]}"
            answer_text = f"{len(results)} ({names})"
        else:
            # List: comma-separated with "and" for last item
            if len(results) == 1:
                answer_text = results[0]
            elif len(results) == 2:
                answer_text = f"{results[0]} and {results[1]}"
            else:
                answer_text = ", ".join(results[:-1]) + f", and {results[-1]}"

        return Answer(
            text=answer_text,
            confidence=0.9,
            source="aggregate",
            survivors=len(results),
            convergence_details={
                "aggregate_type": aggregate_type,
                "result_count": len(results),
                "results": results,
            },
        )

    # ===================================================================
    # PQ WRITE-BACK
    # ===================================================================

    def _writeback_pq(
        self, user_id: int, query_text: str, q_emb: np.ndarray,
        relationship_id: int, answer_text: str = "",
    ) -> None:
        """Insert the original user query into predicted_queries for the
        accepted edge. Avoid duplicates by checking question text."""

        try:
            with get_db_context() as conn:
                existing = conn.execute(
                    "SELECT id FROM predicted_queries "
                    "WHERE relationship_id = ? AND predicted_question = ?",
                    (relationship_id, query_text),
                ).fetchone()

                if existing:
                    return

                conn.execute(
                    "INSERT INTO predicted_queries "
                    "(relationship_id, user_id, predicted_question, "
                    " answer_text, question_embedding, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, 0.9, datetime('now'))",
                    (
                        relationship_id,
                        user_id,
                        query_text,
                        answer_text,
                        q_emb.tobytes(),
                    ),
                )
                conn.commit()
                log.debug(
                    "PQ write-back: query=%r -> relationship_id=%d",
                    query_text, relationship_id,
                )
        except Exception:
            log.debug("PQ write-back failed", exc_info=True)

    # ===================================================================
    # PUBLIC: retrieve()
    # ===================================================================

    def retrieve(
        self, user_id: int, query: str,
    ) -> Union[Answer, StructuralRefusal]:
        """Full read path: moat pipeline -> verification -> answer/refusal.

        On accepted answer, writes back the query as a new PQ row so
        repeated queries converge to near-direct hit behavior.
        """

        # Empty query → immediate refusal
        if not query or not query.strip():
            return StructuralRefusal(
                reason="empty_query",
                text=REFUSAL_TEXT,
                confidence=0.0,
            )

        resolution = self._resolve_indirect_references(user_id, query)

        # IndirectResolution typed fields: if the resolver produced
        # a pre-computed answer or refusal (comparative intersection),
        # return it directly. The moat pipeline cannot handle
        # dual-entity set intersection (traced: query_entity="DC"
        # fails _edge_mentions_entity for all candidates).
        if resolution.answer is not None:
            return resolution.answer
        if resolution.refusal is not None:
            return resolution.refusal

        indirect_refs = resolution.entity_refs

        candidates, q_emb, resolved_entities = self._run_moat_pipeline(
            user_id, query,
        )

        if not candidates:
            return StructuralRefusal(
                reason="no_candidates",
                text=REFUSAL_TEXT,
                confidence=0.0,
            )

        # Merge indirect-ref resolved entities into resolved_entities.
        existing_names = {e["name"].lower() for e in resolved_entities}
        for ref in indirect_refs:
            if ref["name"].lower() not in existing_names:
                resolved_entities.insert(0, {
                    "id": None,
                    "name": ref["name"],
                    "entity_type": "PERSON",
                    "score": 1.0,
                })
                existing_names.add(ref["name"].lower())

        # Extract query entity from indirect refs. Indirect resolution
        # produces refs in parse order: speaker refs first (intermediate),
        # then possessive/relcl refs (terminal — the entity the query
        # is actually about).
        #
        # "Where did the speaker's girlfriend study?"
        #   ref[0]: speaker → "user" (intermediate)
        #   ref[1]: possessive → "Suki" (terminal)
        #   query_entity = "Suki" (last ref wins — deepest resolution)
        #
        # "Does the speaker have food allergies?"
        #   ref[0]: speaker → "user" (only ref)
        #   query_entity = "user"
        #
        # Last ref is always the deepest resolution because
        # _resolve_indirect_references walks the parse left-to-right:
        # speaker refs fire on the anchor token, possessive refs fire
        # on the head noun (which is structurally deeper).
        query_entity = None
        if indirect_refs:
            # Last ref = deepest resolution in the possessive chain.
            query_entity = indirect_refs[-1]["name"]
        elif resolved_entities:
            # Prefer the grammatical subject of the query. Without
            # this, "Does Tariq still live in Seattle?" resolves to
            # entity "Seattle" (the pobj) instead of "Tariq" (nsubj).
            #
            # Root cause: resolved_entities order depends on the
            # entity resolver + fallback, not parse structure.
            # Extract nsubj via spaCy to pick the subject entity.
            resolved_names = {e["name"].lower() for e in resolved_entities}
            subj_entity = None
            try:
                import spacy
                _nlp = spacy.load("en_core_web_sm")
                _doc = _nlp(query)
                for _tok in _doc:
                    if _tok.dep_ == "nsubj" and _tok.pos_ == "PROPN":
                        if _tok.text.lower() in resolved_names:
                            subj_entity = _tok.text
                            break
            except Exception:
                pass
            query_entity = subj_entity or resolved_entities[0]["name"]

        # First-person override: if the query uses first-person
        # language, target is "user". Only apply when indirect
        # resolution did NOT produce a graph-traversal result —
        # if we resolved "speaker's girlfriend" to "Suki", the
        # terminal entity is "Suki", not "user".
        if len(indirect_refs) <= 1:
            query_words_lower = {w.lower() for w in query.split()}
            if query_words_lower & _FIRST_PERSON:
                query_entity = "user"
            elif query_entity and query_entity.lower() in _FIRST_PERSON:
                query_entity = "user"

        # No entity resolved — refuse. We don't know who the
        # query is about, so no candidate can be coherent.
        if not query_entity:
            return StructuralRefusal(
                reason="no_entity_resolved",
                text=REFUSAL_TEXT,
                confidence=0.0,
                survivors=len(candidates),
            )

        # Temporal mode detection (query-centric time filtering)
        if _query_requests_historical(query):
            temporal_mode = "historical"
        elif _query_requests_current_state(query):
            temporal_mode = "current"
        else:
            temporal_mode = "current"

        # Facts lookup against original query
        canonical_fact = self._load_facts_for_entity(
            user_id, query_entity, query,
        )

        # Aggregation routing: count/list queries collect all matches.
        # For list queries, use predicate-based filtering (via the
        # aggregate loop's verb check) instead of entity gating.
        # Root cause: "Name everyone who plays soccer" resolves
        # query_entity to "Name" (false parse artifact), rejecting
        # all candidates in entity gate.
        if _is_count_query(query):
            result = self._aggregate_loop(
                candidates, query_entity, canonical_fact, query, q_emb,
                temporal_mode, "count",
            )
        elif _is_list_query(query):
            result = self._aggregate_loop(
                candidates, None, canonical_fact, query, q_emb,
                temporal_mode, "list",
            )
        else:
            # Verification loop — uses original query for coherence check
            result = self._verification_loop(
                candidates, query_entity, canonical_fact, query, q_emb,
                temporal_mode,
            )

        # Negation / absence confirmation (CWA):
        # If yes/no query returned refusal AND the reason is NOT
        # predicate_absence_confirmed (which already checked entity
        # existence in the verification loop), check if entity exists
        # in the graph. If entity exists but NO candidates passed at
        # all (zero survivors), return "No".
        #
        # When predicate_absence_confirmed: the verification loop
        # already determined the entity has edges but none match the
        # asked predicate. The correct response is refusal ("not
        # mentioned"), not "No" — "No" implies contradictory evidence.
        if (isinstance(result, StructuralRefusal)
                and _is_yes_no_query(query)
                and query_entity
                and result.reason != "predicate_absence_confirmed"):
            entity_exists = self._entity_exists(user_id, query_entity)
            if entity_exists:
                return Answer(
                    text="No",
                    confidence=0.8,
                    source="confirmed_absence",
                    survivors=0,
                    convergence_details={
                        "reason": "entity_exists_predicate_absent",
                        "entity": query_entity,
                    },
                )

        # PQ write-back on accepted answer
        if isinstance(result, Answer) and result.convergence_details:
            rid = result.convergence_details.get("relationship_id")
            if rid is not None:
                self._writeback_pq(
                    user_id, query, q_emb, rid, result.text or "",
                )

        return result

    # ===================================================================
    # PUBLIC: reconstruct()
    # ===================================================================

    def reconstruct(self, user_id: int, query: str) -> Situation:
        """Reconstruction path: moat pipeline -> recluster -> render
        deterministic cluster narrative from source_text or SPO fallback.

        No verification loop needed — reconstruct selects a situation,
        not a single atomic truth claim.
        """

        if not query or not query.strip():
            return Situation(narrative="", survivor_count=0)

        candidates, q_emb, resolved_entities = self._run_moat_pipeline(
            user_id, query,
        )

        if not candidates:
            return Situation(narrative=REFUSAL_TEXT, survivor_count=0)

        # Take top N survivors
        top = candidates[:RECONSTRUCT_TOP_N]

        # Recluster by shared non-self entities
        entity_to_cluster: Dict[str, str] = {}
        cluster_edges: Dict[str, List[Candidate]] = defaultdict(list)
        cluster_counter = 0

        for c in top:
            s = (c.edge.get("subject") or "").lower()
            o = (c.edge.get("object") or "").lower()
            non_self = [
                x for x in (s, o)
                if x and x != "user"
            ]

            assigned = None
            for ent in non_self:
                if ent in entity_to_cluster:
                    assigned = entity_to_cluster[ent]
                    break

            if assigned is None:
                cluster_counter += 1
                assigned = f"rc_{cluster_counter}"

            for ent in non_self:
                entity_to_cluster[ent] = assigned

            cluster_edges[assigned].append(c)

        # Build Cluster objects
        clusters: List[Cluster] = []
        all_participants: Set[str] = set()
        narratives: List[str] = []

        for cid, cands in cluster_edges.items():
            # Sort by sequence_number for deterministic ordering
            cands.sort(key=lambda c: c.edge.get("sequence_number") or 0)

            participants: Set[str] = set()
            edges: List[Dict[str, Any]] = []
            tones: List[str] = []
            categories: List[str] = []

            for c in cands:
                e = c.edge
                participants.add(e.get("subject") or "")
                participants.add(e.get("object") or "")
                edges.append(e)
                t = e.get("edge_emotional_label")
                if t:
                    tones.append(t)
                cat = e.get("edge_schematic_category")
                if cat:
                    categories.append(cat)

            participants.discard("")
            all_participants.update(participants)

            # Render narrative from source_text or SPO fallback
            parts: List[str] = []
            for e in edges:
                st = (e.get("source_text") or "").strip()
                if st:
                    parts.append(st)
                else:
                    parts.append(_edge_to_fact_statement(e))

            narrative = " ".join(parts)
            narratives.append(narrative)

            # Dominant schema category
            cat_dominant = None
            if categories:
                from collections import Counter
                cat_dominant = Counter(categories).most_common(1)[0][0]

            # Dominant emotional tone
            tone_dominant = None
            if tones:
                from collections import Counter
                tone_dominant = Counter(tones).most_common(1)[0][0]

            clusters.append(Cluster(
                cluster_id=cid,
                participants=sorted(participants),
                edges=edges,
                narrative=narrative,
                schema_category=cat_dominant,
                emotional_tone=tone_dominant,
            ))

        full_narrative = " ".join(narratives)

        return Situation(
            clusters=clusters,
            survivor_count=len(top),
            narrative=full_narrative,
            source_tag="reconstruct",
            participants=sorted(all_participants),
        )


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_singleton: Optional[RetrievalEngine] = None


def get_retrieval_engine(memory, temporal) -> RetrievalEngine:
    global _singleton
    if _singleton is None:
        _singleton = RetrievalEngine(memory, temporal)
    return _singleton


__all__ = [
    "Answer",
    "Cluster",
    "Situation",
    "StructuralRefusal",
    "RetrievalEngine",
    "get_retrieval_engine",
]
