# -*- coding: utf-8 -*-
"""
RetrievalEngine v3 — SQL-first retrieval with cosine fallback.

Root cause for v3: the grammar engine's classify_query() already computes
match_entity, match_schema, match_predicate, and return_field — all the
structural information needed for direct SQL lookup. Using cosine as primary
(v2) causes wrong-entity edges to rank high, requiring a 300-line
verification loop to compensate. v3 replaces cosine-primary with SQL-primary
and keeps cosine only as fallback when SQL finds nothing.

Primary path: classify_query -> facts fast path -> SQL edge lookup ->
              speaker attribution -> return field extraction.
Fallback: cosine pipeline (only when SQL finds nothing).

reconstruct() uses temporal.recluster_for_reconstruction for situation queries.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
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

_FIRST_PERSON = frozenset({"i", "me", "my", "myself", "user"})

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
    """Typed result from _resolve_indirect_references."""
    entity_refs: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[Answer] = None
    refusal: Optional[StructuralRefusal] = None


# ---------------------------------------------------------------------------
# Utility functions (preserved from v2 for external imports)
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


def _extract_query_verb(query: str) -> Optional[str]:
    """Extract the semantic predicate from a query via spaCy dep parse."""
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

    if root.pos_ == "VERB" and root.lemma_.lower() not in ("be", "do"):
        return root.lemma_.lower()

    for child in root.children:
        if child.dep_ in ("attr", "acomp", "oprd"):
            if child.pos_ in ("NOUN", "ADJ", "VERB"):
                return child.lemma_.lower()
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    return grandchild.lemma_.lower()
        if child.dep_ in ("dobj", "nsubj") and child.pos_ == "NOUN":
            if child.text.lower() not in ("what", "who", "where", "when", "which", "how"):
                return child.lemma_.lower()

    if root.pos_ in ("VERB", "AUX"):
        return root.lemma_.lower()
    return None


def _extract_edge_predicate_lemma(predicate: str) -> Optional[str]:
    if not predicate:
        return None
    head = predicate.replace("_", " ").strip().split()[0]
    return _normalize_token(head) or None


def _wordnet_verb_match(verb_a: str, verb_b: str) -> bool:
    """Check if two words are semantically related via WordNet."""
    if not verb_a or not verb_b:
        return False
    if verb_a == verb_b:
        return True
    try:
        from nltk.corpus import wordnet as wn
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
    """Check if query_word appears in predicate_text via WordNet expansion."""
    if not query_word or not predicate_text:
        return False
    pred_words = set(predicate_text.lower().split())
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
                for hyp in syn.hypernyms():
                    for lemma in hyp.lemmas():
                        expanded.add(lemma.name().lower().replace("_", " "))
        for term in expanded:
            term_words = term.split()
            content_words = [tw for tw in term_words if len(tw) > 2]
            if content_words and any(tw in pred_words for tw in content_words):
                return True
    except Exception:
        pass
    return False


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
            t.text for t in tok.subtree if t.dep_ != "det"
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


def _edge_to_fact_statement(edge: Dict[str, Any]) -> str:
    s = (edge.get("subject") or "").replace("_", " ")
    p = (edge.get("predicate") or "").replace("_", " ")
    o = (edge.get("object") or "").replace("_", " ")
    if s.lower() == "user":
        s = "I"
    if o.lower() == "user":
        o = "me"
    return f"{s} {p} {o}".strip()


def _render_temporal(edge: Dict[str, Any]) -> str:
    """Extract best human-readable temporal answer from an edge."""
    te = (edge.get("temporal_expression") or "").strip()
    if te:
        import spacy as _sp_temp
        try:
            _nlp_temp = _sp_temp.load("en_core_web_sm")
            _doc_temp = _nlp_temp(te)
            start_idx = 0
            for _tok in _doc_temp:
                if _tok.pos_ in ("ADP", "DET", "PUNCT", "CCONJ", "SCONJ"):
                    start_idx = _tok.idx + len(_tok.text)
                else:
                    break
            stripped = te[start_idx:].strip()
            if stripped:
                return stripped
        except Exception:
            pass
        return te

    red = (edge.get("resolved_event_date") or "").strip()
    if red:
        try:
            from datetime import datetime as _dt_render
            dt = _dt_render.fromisoformat(red.replace("Z", "+00:00"))
            return dt.strftime("%-d %B %Y").lstrip("0")
        except Exception:
            try:
                from datetime import datetime as _dt_render2
                dt = _dt_render2.fromisoformat(red.replace("Z", "+00:00"))
                return dt.strftime("%d %B %Y").lstrip("0")
            except Exception:
                pass
    return ""


def _edge_speaker_matches_entity(
    edge: Dict[str, Any], query_entity: Optional[str],
) -> bool:
    """Check if the query entity is the speaker/subject of this edge.
    Cat 5 adversarial defense: speaker is last entry in relational_entities."""
    if not query_entity:
        return True

    q_lower = query_entity.strip().lower()
    if q_lower in _FIRST_PERSON:
        return True

    rel_raw = edge.get("relational_entities") or ""
    if not rel_raw:
        return True

    try:
        ents = json.loads(rel_raw) if rel_raw.startswith("[") else [rel_raw]
    except Exception:
        return True

    if not ents:
        return True

    speaker = ents[-1].strip().lower()

    if speaker == "user":
        for e in ents:
            if q_lower == e.strip().lower():
                return True
            if q_lower in e.strip().lower() or e.strip().lower() in q_lower:
                return True
        return True

    if q_lower == speaker:
        return True
    if q_lower in speaker or speaker in q_lower:
        return True

    return False


def _edge_mentions_entity(
    edge_subject: str, edge_object: str, query_entity: Optional[str],
    relational_entities: str = "",
) -> bool:
    """Does this edge mention the query entity in either position?"""
    if not query_entity:
        return True
    subj = " ".join(_split_words(edge_subject))
    obj = " ".join(_split_words(edge_object))
    ent = " ".join(_split_words(query_entity))
    if not ent:
        return True
    if subj == ent:
        return True
    if obj == ent:
        return True
    query_is_first_person = ent in _FIRST_PERSON
    if query_is_first_person and (subj == "user" or obj == "user"):
        return True
    if relational_entities:
        rel_lower = relational_entities.lower()
        if query_entity.lower() in rel_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# RetrievalEngine
# ---------------------------------------------------------------------------

class RetrievalEngine:
    def __init__(self, memory=None, temporal=None):
        self._memory = memory
        self._temporal = temporal

    # ===================================================================
    # INDIRECT REFERENCE RESOLUTION (preserved from v2)
    # ===================================================================

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
        and comparative patterns against the SPO knowledge graph."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            return IndirectResolution()

        doc = nlp(query)
        resolved: List[Dict[str, Any]] = []

        # --- Pattern 1: Speaker references ---
        for tok in doc:
            if tok.text.lower() in self._SPEAKER_REFS:
                resolved.append({
                    "name": "user", "source": "speaker", "anchor": tok.text,
                })
            if tok.dep_ == "compound" and tok.text.lower().rstrip("s") in self._SPEAKER_REFS:
                resolved.append({
                    "name": "user", "source": "speaker", "anchor": tok.text,
                })

        # --- Pattern 2: Possessive ---
        for tok in doc:
            if tok.dep_ == "poss" and tok.head.pos_ == "NOUN":
                anchor = tok.text
                if anchor.lower() in _FIRST_PERSON or anchor.lower() in self._SPEAKER_REFS:
                    anchor = "user"

                head = tok.head
                compounds = sorted(
                    [c for c in head.children
                     if c.dep_ == "compound"
                     and c.pos_ == "NOUN"
                     and c.i > tok.i
                     and c.i < head.i],
                    key=lambda c: c.i,
                )

                if compounds:
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

        # --- Pattern 3: Relative clause ---
        for tok in doc:
            if tok.dep_ != "relcl" or tok.pos_ != "VERB":
                continue
            if tok.head.text.lower() not in self._GENERIC_PERSON_NOUNS:
                continue
            verb = tok.lemma_.lower()
            obj_toks = [c for c in tok.children if c.dep_ in ("dobj", "attr", "pobj", "compound", "oprd")]
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
                return IndirectResolution(
                    refusal=StructuralRefusal(
                        reason="relcl_entity_not_found",
                        text=REFUSAL_TEXT,
                        confidence=0.0,
                    ),
                )

        # --- Pattern 4: Comparative ---
        _IDENTITY_DETERMINERS = frozenset({"same", "identical"})
        for tok in doc:
            if tok.dep_ not in ("nsubj", "nmod"):
                continue
            conj_children = [c for c in tok.children if c.dep_ == "conj"]
            if not conj_children:
                continue
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
            verb_lemma = verb_tok.lemma_.lower() if verb_tok.lemma_ else None

            if verb_lemma:
                common = self._find_edge_intersection(
                    user_id, entity1, entity2,
                    verb_lemma, identity_noun,
                )
                if common is not None:
                    return IndirectResolution(
                        answer=Answer(
                            text=common,
                            source="comparative_intersection",
                            confidence=1.0,
                            convergence_details={
                                "entity1": entity1, "entity2": entity2,
                            },
                        ),
                    )
                else:
                    return IndirectResolution(
                        refusal=StructuralRefusal(
                            reason="comparative_no_intersection",
                            text=REFUSAL_TEXT,
                            confidence=0.0,
                        ),
                    )

        # --- Pattern 5: "Who else" peer query ---
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
                                "peer_count": len(peers), "peers": peers,
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

        # Deduplicate
        seen: Set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for r in resolved:
            key = r["name"].lower()
            if key not in seen:
                deduped.append(r)
                seen.add(key)

        return IndirectResolution(entity_refs=deduped)

    # ===================================================================
    # GRAPH TRAVERSAL HELPERS (preserved from v2)
    # ===================================================================

    def _traverse_relation(
        self, user_id: int, anchor: str, relation_word: str,
    ) -> Optional[str]:
        """Find entity on OTHER side of anchor's edge matching relation_word."""
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

        # Three-tier partition: lemma > schema > all
        lemma_matched = []
        schema_matched = []
        all_rows = []
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

        if lemma_matched:
            pool = lemma_matched
        elif schema_matched:
            pool = schema_matched
        else:
            pool = all_rows

        best_row, _ = max(pool, key=lambda pair: pair[1])
        anchor_lower = anchor.lower()
        if best_row["subject"].lower() == anchor_lower:
            return best_row["object"]
        return best_row["subject"]

    def _find_entity_by_edge(
        self, user_id: int, verb: str, obj: str,
    ) -> Optional[str]:
        """Find entity (subject) with edge matching verb+obj."""
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

        obj_words = _normalized_words(obj)
        edge_text = " ".join([
            (best_row["object"] if best_row["object"] else ""),
            (best_row["source_text"] if best_row["source_text"] else ""),
        ]).lower()
        edge_words = _normalized_words(edge_text)
        if obj_words and not (obj_words & edge_words):
            if not _wordnet_noun_in_predicate(obj, edge_text):
                return None
        return best_row["subject"]

    def _find_edge_intersection(
        self, user_id: int, entity1: str, entity2: str,
        verb_lemma: str, noun_category: str,
    ) -> Optional[str]:
        """Find common object shared by entity1 and entity2."""
        verb_lemmas = _normalized_words(verb_lemma)

        def _objects_for_entity(entity: str) -> Dict[str, str]:
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
            first_key = sorted(common_keys)[0]
            return objs1[first_key]

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
        self, user_id: int, verb_lemmas: Set[str], obj_hint: str,
    ) -> List[str]:
        """Find all subjects sharing a predicate+object, excluding user.

        Uses structural matching: predicate lemma overlap + object text
        containment or WordNet semantic match. No cosine thresholds.
        """
        obj_lower = obj_hint.strip().lower()

        with get_db_context() as conn:
            rows = conn.execute(
                "SELECT subject, predicate, object "
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
            # Structural object match: substring containment or WordNet
            if obj_lower in edge_obj or edge_obj in obj_lower:
                peers.append(subj)
                seen.add(subj.lower())
            elif _wordnet_noun_in_predicate(obj_hint, edge_obj):
                peers.append(subj)
                seen.add(subj.lower())

        return sorted(peers)

    # ===================================================================
    # FACTS FAST PATH (Step 1)
    # ===================================================================

    def _try_facts_lookup(self, user_id: int, qd) -> Optional[Answer]:
        """O(1) fact lookup. key = schema::verb_class::subject."""
        try:
            from app.engines.grammar_engine import classify_verb_class

            subject = qd.match_subject or qd.match_entity
            schema = qd.match_schema
            if not subject or not schema:
                return None
            if qd.return_field == "temporal":
                return None
            if qd.wh_word in ("when",):
                return None

            verb_class = "UNKNOWN"
            if qd.match_predicate:
                vc = classify_verb_class(qd.match_predicate)
                verb_class = vc.name if hasattr(vc, 'name') else str(vc)

            with get_db_context() as conn:
                # Try exact key first
                key = f"{schema}::{verb_class}::{subject}"
                row = conn.execute(
                    "SELECT key, value, confidence FROM facts "
                    "WHERE user_id = ? AND key = ?",
                    (user_id, key),
                ).fetchone()

                # Fallback: match query content words against fact keys.
                # "What does Sam do for work?" → "work" matches "WORK" in key.
                # Extract content nouns from query via spaCy (not match_predicate
                # which is "do" — a light verb).
                if not row:
                    try:
                        import spacy as _sp_fact
                        _nlp_fact = _sp_fact.load("en_core_web_sm")
                        _doc_fact = _nlp_fact(qd.wh_word + " " if qd.wh_word else "" )
                        _doc_fact = _nlp_fact(query if hasattr(qd, '_raw_query') else
                                             (qd.match_predicate or "") + " " + (qd.match_object or ""))
                    except Exception:
                        _doc_fact = None

                    # Get ALL facts for this subject, score by content overlap
                    all_facts = conn.execute(
                        "SELECT key, value, confidence FROM facts "
                        "WHERE user_id = ? AND key LIKE ?",
                        (user_id, f"%::{subject}"),
                    ).fetchall()
                    if all_facts and len(all_facts) == 1:
                        row = all_facts[0]  # only one fact for this subject
                    elif all_facts:
                        # Multiple facts — use query content words to pick
                        q_words = _normalized_words(
                            (qd.match_predicate or "") + " " + (qd.match_object or "")
                        )
                        best = None
                        best_overlap = 0
                        for f_row in all_facts:
                            key_words = _normalized_words(f_row["key"].replace("::", " "))
                            overlap = len(q_words & key_words)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best = f_row
                        if best:
                            row = best

                if row and row["value"]:
                    return Answer(
                        text=row["value"],
                        confidence=row["confidence"] or 1.0,
                        source="fact_fast_path",
                        convergence_details={
                            "query": qd.wh_word,
                            "fact_key": row["key"],
                        },
                    )
        except Exception:
            log.debug("facts lookup failed", exc_info=True)
        return None

    # ===================================================================
    # SQL EDGE LOOKUP (Step 2)
    # ===================================================================

    def _sql_edge_lookup(
        self, user_id: int, qd,
        query_entity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """SQL filter on entity + predicate + schema. Returns edges
        ordered by sequence_number DESC (most recent first).

        Root cause for not putting schema in SQL WHERE: classify_query
        sometimes assigns wrong schema (e.g., 'marry' -> 'career').
        Predicate lemma match is more reliable. Schema is applied as
        a secondary filter only when predicate filtering finds nothing.
        """
        entity = query_entity or (qd.match_entity if qd else None)

        sql = """SELECT * FROM relationships
                 WHERE user_id = ?
                   AND (edge_mood IS NULL OR edge_mood = 'indicative')
                   AND COALESCE(is_current, 1) = 1
                   AND tombstoned_at IS NULL"""
        params: List[Any] = [user_id]

        # Entity scope
        if entity:
            entity_for_sql = entity
            if entity.lower() in _FIRST_PERSON:
                entity_for_sql = "user"
            sql += " AND (relational_entities LIKE ? OR LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))"
            params.extend([f'%{entity_for_sql}%', entity_for_sql, entity_for_sql])

        sql += " ORDER BY sequence_number DESC"

        with get_db_context() as conn:
            rows = conn.execute(sql, params).fetchall()

        edges = []
        for row in rows:
            edge = {key: row[key] for key in row.keys()}
            edges.append(edge)

        # Predicate scope (applied in Python for WordNet matching)
        # Takes priority over schema because classify_query's predicate
        # is extracted directly from the parse tree while schema uses
        # heuristic WordNet hypernym closure which can misclassify.
        #
        # Two-tier: lemma overlap first (precise), WordNet fallback
        # only when lemma finds nothing (WordNet is too broad for
        # generic verbs like be/have/live which connect to everything).
        # Light verbs (closed grammatical class from English grammar):
        # do/have/be/get/make/take — these are structural, not semantic.
        # "What does Sam DO for work?" — "do" is light, "work" is semantic.
        # Don't filter on light verbs — they match nothing useful.
        _LIGHT_VERBS = frozenset({"do", "have", "be", "get", "make", "take", "go"})

        if qd and qd.match_predicate and edges:
            if qd.match_predicate.lower() not in _LIGHT_VERBS:
                pred_lemmas = _normalized_words(qd.match_predicate)
                lemma_matched = []
                for edge in edges:
                    edge_pred = (edge.get("predicate") or "").replace("_", " ")
                    edge_pred_lemmas = _normalized_words(edge_pred)
                    if pred_lemmas & edge_pred_lemmas:
                        lemma_matched.append(edge)
                if lemma_matched:
                    edges = lemma_matched
                else:
                    edges = []

        # Schema scope (secondary filter — only when predicate filtering
        # did not narrow, to avoid over-filtering on misclassified schema)
        if qd and qd.match_schema and edges:
            schema_filtered = [
                e for e in edges
                if (e.get("edge_schematic_category") or "").lower() == qd.match_schema.lower()
            ]
            if schema_filtered:
                edges = schema_filtered

        return edges

    # ===================================================================
    # SPEAKER ATTRIBUTION (Step 3 — Cat 5)
    # ===================================================================

    def _filter_by_speaker(
        self, edges: List[Dict[str, Any]], query_entity: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Filter edges by speaker attribution.
        Returns (filtered_edges, was_speaker_filtered).
        If all edges rejected, returns ([], True) -> StructuralRefusal."""
        if not query_entity:
            return edges, False

        q_lower = query_entity.strip().lower()
        if q_lower in _FIRST_PERSON or q_lower == "user":
            return edges, False

        matched = [e for e in edges if _edge_speaker_matches_entity(e, query_entity)]
        if matched:
            return matched, True
        return [], True

    # ===================================================================
    # RETURN FIELD EXTRACTION (Step 4)
    # ===================================================================

    def _extract_answer_from_edge(
        self, edge: Dict[str, Any], qd, query_entity: Optional[str],
    ) -> str:
        """Extract the right field from the winning edge per spec Step 4."""
        return_field = qd.return_field if qd else "episodic"

        if return_field == "temporal":
            return _render_temporal(edge)

        if return_field == "emotional":
            label = (edge.get("edge_emotional_label") or "").strip()
            if label:
                return label

        if return_field == "relational":
            entity = query_entity or (qd.match_entity if qd else None)
            if entity:
                subj = (edge.get("subject") or "").strip()
                obj = (edge.get("object") or "").strip()
                if subj.lower() == entity.lower() or subj.lower() == "user":
                    return obj
                return subj
            return (edge.get("object") or "").strip()

        # episodic (default) -- return the object.
        # Bidirectional edge traversal: when the query entity appears
        # as the object (directed edge stored backwards), return the
        # subject instead. Same traversal as _traverse_relation.
        obj = (edge.get("object") or "").strip()
        entity = query_entity or (qd.match_entity if qd else None)
        if entity and obj.lower() == entity.lower():
            subj = (edge.get("subject") or "").strip()
            if subj:
                return subj
        return obj

    # ===================================================================
    # ENTITY EXISTENCE CHECK (for CWA negation)
    # ===================================================================

    def _entity_exists(self, user_id: int, entity_name: str) -> bool:
        """Check if entity has ANY edges in the graph."""
        if not entity_name:
            return False
        if self._memory:
            try:
                ent_obj = self._memory.get_entity(user_id, entity_name)
                if ent_obj is not None:
                    return True
            except Exception:
                pass
        ent = entity_name
        if ent.lower() in _FIRST_PERSON:
            ent = "user"
        if self._memory:
            try:
                rels = self._memory.get_relationships(user_id, subject=ent)
                if rels:
                    return True
                rels = self._memory.get_relationships(user_id, object=ent)
                if rels:
                    return True
            except Exception:
                pass
        with get_db_context() as conn:
            row = conn.execute(
                "SELECT 1 FROM relationships WHERE user_id = ? "
                "AND (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?)) "
                "AND tombstoned_at IS NULL LIMIT 1",
                (user_id, ent, ent),
            ).fetchone()
            return row is not None

    # ===================================================================
    # AGGREGATION (counting / listing)
    # ===================================================================

    def _aggregate(
        self, user_id: int, query: str, qd,
        query_entity: Optional[str],
        aggregate_type: str = "count",
    ) -> Union[Answer, StructuralRefusal]:
        """SQL-based aggregation for count/list queries."""
        query_verb = _extract_query_verb(query)
        if query_verb and query_verb in ('name', 'list', 'tell'):
            try:
                import spacy
                _nlp = spacy.load('en_core_web_sm')
                _doc = _nlp(query)
                for _tok in _doc:
                    if _tok.dep_ == 'relcl' and _tok.pos_ == 'VERB':
                        query_verb = _tok.lemma_.lower()
                        break
            except Exception:
                pass

        verb_lemmas = _normalized_words(query_verb) if query_verb else set()

        with get_db_context() as conn:
            rows = conn.execute(
                "SELECT * FROM relationships "
                "WHERE user_id = ? "
                "  AND (edge_mood IS NULL OR edge_mood = 'indicative') "
                "  AND COALESCE(is_current, 1) = 1 "
                "  AND tombstoned_at IS NULL "
                "ORDER BY sequence_number DESC",
                (user_id,),
            ).fetchall()

        results: List[str] = []
        seen: Set[str] = set()

        for row in rows:
            edge = {key: row[key] for key in row.keys()}
            if verb_lemmas:
                pred_text = (edge.get("predicate") or "").replace("_", " ")
                if not (verb_lemmas & _normalized_words(pred_text)):
                    continue

            if aggregate_type in ("count", "list") and query.lower().startswith(
                    ("how many", "name everyone", "who all")):
                value = (edge.get("subject") or "").strip()
                if value.lower() == "user":
                    value = "I"
            else:
                value = (edge.get("object") or "").strip()

            norm = value.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                results.append(value)

        if not results:
            return StructuralRefusal(
                reason="no_aggregate_results",
                text=REFUSAL_TEXT,
                confidence=0.0,
                survivors=0,
            )

        if aggregate_type == "count":
            if len(results) == 1:
                names = results[0]
            elif len(results) == 2:
                names = f"{results[0]} and {results[1]}"
            else:
                names = ", ".join(results[:-1]) + f", and {results[-1]}"
            answer_text = f"{len(results)} ({names})"
        else:
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
        self, user_id: int, query_text: str,
        relationship_id: int, answer_text: str = "",
    ) -> None:
        """Insert user query into predicted_queries for the winning edge."""
        try:
            q_emb = embed_text(query_text)
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
        except Exception:
            log.debug("PQ write-back failed", exc_info=True)

    # ===================================================================
    # COSINE FALLBACK (Step 5 — only when SQL found nothing)
    # ===================================================================

    def _cosine_fallback(
        self, user_id: int, query: str, q_emb: np.ndarray,
        query_entity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """PQ + edge cosine recall as fallback when SQL found nothing."""
        seen: Dict[int, Tuple[Dict[str, Any], float]] = {}

        with get_db_context() as conn:
            # PQ cosine
            pq_rows = conn.execute(
                "SELECT pq.relationship_id, pq.question_embedding, "
                "       r.* "
                "FROM predicted_queries pq "
                "JOIN relationships r ON r.id = pq.relationship_id "
                "WHERE pq.user_id = ? "
                "  AND r.tombstoned_at IS NULL "
                "  AND COALESCE(r.is_current, 1) = 1 "
                "  AND (r.edge_mood IS NULL OR r.edge_mood = 'indicative')",
                (user_id,),
            ).fetchall()

            for row in pq_rows:
                rid = row["relationship_id"]
                pq_cos = _cosine_from_blob(q_emb, row["question_embedding"])
                edge_cos = _cosine_from_blob(q_emb, row["edge_embedding"])
                score = max(pq_cos, edge_cos)
                edge = {key: row[key] for key in row.keys() if key != "question_embedding"}
                if rid in seen:
                    if score > seen[rid][1]:
                        seen[rid] = (edge, score)
                else:
                    seen[rid] = (edge, score)

            # Edge cosine
            edge_rows = conn.execute(
                "SELECT * FROM relationships "
                "WHERE user_id = ? "
                "  AND tombstoned_at IS NULL "
                "  AND COALESCE(is_current, 1) = 1 "
                "  AND edge_embedding IS NOT NULL "
                "  AND (edge_mood IS NULL OR edge_mood = 'indicative')",
                (user_id,),
            ).fetchall()

            for row in edge_rows:
                rid = row["id"]
                edge_cos = _cosine_from_blob(q_emb, row["edge_embedding"])
                edge = {key: row[key] for key in row.keys()}
                if rid in seen:
                    if edge_cos > seen[rid][1]:
                        seen[rid] = (edge, edge_cos)
                else:
                    seen[rid] = (edge, edge_cos)

        ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        edges = [e for e, s in ranked[:ENTRY_CAP]]

        # Entity filter
        if query_entity:
            entity_matched = [
                e for e in edges
                if _edge_mentions_entity(
                    e.get("subject", ""), e.get("object", ""),
                    query_entity, e.get("relational_entities", ""),
                )
            ]
            if entity_matched:
                edges = entity_matched

        return edges

    # ===================================================================
    # RESOLVE QUERY ENTITY
    # ===================================================================

    def _resolve_query_entity(
        self, query: str, qd, indirect_refs: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Determine the query entity from indirect refs or grammar parse."""
        if indirect_refs:
            return indirect_refs[-1]["name"]

        if qd and qd.match_entity:
            return qd.match_entity

        if qd and qd.match_subject:
            subj = qd.match_subject
            if subj.lower() in _FIRST_PERSON:
                return "user"
            return subj

        # First-person detection
        try:
            from app.engines.grammar_engine import resolve_pronouns, _get_nlp
            doc = _get_nlp()(query)
            resolved = resolve_pronouns(doc, speaker="user")
            if hasattr(resolved, '__iter__') and not isinstance(resolved, str):
                resolved_str = " ".join(t.text if hasattr(t, 'text') else str(t) for t in resolved)
            else:
                resolved_str = str(resolved)
            if resolved_str.lower() != query.lower():
                return "user"
        except Exception:
            pass

        return None

    # ===================================================================
    # PUBLIC: retrieve()
    # ===================================================================

    def retrieve(
        self, user_id: int, query: str,
    ) -> Union[Answer, StructuralRefusal]:
        """Full read path: SQL-first with cosine fallback.

        Steps:
          0. classify_query -> qd (structural decomposition)
          1. Facts fast path (O(1) for stative facts)
          2. SQL edge lookup (entity + schema + predicate)
          3. Speaker attribution (Cat 5 adversarial defense)
          4. Return the right field from winning edge
          5. Cosine fallback (only if SQL found nothing)
        """

        if not query or not query.strip():
            return StructuralRefusal(
                reason="empty_query", text=REFUSAL_TEXT, confidence=0.0,
            )

        # ── Step 0: Grammar engine query decomposition ──────────────
        qd = None
        try:
            from app.engines.grammar_engine import classify_query
            qd = classify_query(query)
        except Exception:
            pass

        # Backchannel detection
        try:
            from app.engines.grammar_engine import (
                classify_utterance as _classify_utt,
                _get_nlp as _grammar_nlp,
            )
            _qdoc = _grammar_nlp()(query)
            query_utterance = _classify_utt(_qdoc)
            if query_utterance and query_utterance.is_backchannel:
                return StructuralRefusal(
                    reason="backchannel_detected", text="", confidence=0.0,
                )
        except Exception:
            pass

        # ── Step 1: Facts fast path ──────────────────────────────────
        if qd and qd.is_structural and (qd.match_subject or qd.match_entity) and qd.match_schema:
            fact_answer = self._try_facts_lookup(user_id, qd)
            if fact_answer is not None:
                return fact_answer

        # ── Indirect reference resolution ────────────────────────────
        resolution = self._resolve_indirect_references(user_id, query)

        if resolution.answer is not None:
            return resolution.answer
        if resolution.refusal is not None:
            return resolution.refusal

        indirect_refs = resolution.entity_refs

        # ── Resolve query entity ─────────────────────────────────────
        query_entity = self._resolve_query_entity(query, qd, indirect_refs)

        if len(indirect_refs) <= 1:
            if query_entity and query_entity.lower() in _FIRST_PERSON:
                query_entity = "user"

        if not query_entity:
            query_entity = "user"

        # ── Aggregation routing ──────────────────────────────────────
        if _is_count_query(query):
            return self._aggregate(user_id, query, qd, query_entity, "count")
        if _is_list_query(query):
            return self._aggregate(user_id, query, qd, None, "list")

        # ── Step 2: SQL edge lookup ─��────────���───────────────────────
        # Two-pass: entity-only then full filtered.
        # Root cause: cosine fallback finds wrong edges when entity
        # exists but asked property does not (Jake/live, Dad/work).
        edges_entity_only = self._sql_edge_lookup(user_id, None, query_entity)
        edges = self._sql_edge_lookup(user_id, qd, query_entity)
        entity_found_but_predicate_absent = bool(edges_entity_only) and not edges

        # ── Step 3: Speaker attribution (Cat 5) ─────────────────────
        _query_person = None
        if query_entity and query_entity.lower() not in _FIRST_PERSON and query_entity.lower() != "user":
            _query_person = query_entity

        if _query_person and edges:
            filtered_edges, was_filtered = self._filter_by_speaker(edges, _query_person)
            if was_filtered and not filtered_edges:
                return StructuralRefusal(
                    reason="speaker_not_found",
                    text=REFUSAL_TEXT,
                    confidence=0.0,
                    survivors=0,
                    convergence_details={
                        "query_person": _query_person,
                        "total_candidates": len(edges),
                    },
                )
            if filtered_edges:
                edges = filtered_edges


        # Predicate absence: entity has edges but none match the asked
        # predicate. Refuse -- the entity exists but property does not.
        if entity_found_but_predicate_absent:
            return StructuralRefusal(
                reason="predicate_absence",
                text=REFUSAL_TEXT,
                confidence=0.0,
                survivors=0,
            )

        # ── Step 5: Cosine fallback (only if SQL found nothing) ──────
        if not edges:
            q_emb = embed_text(query)
            edges = self._cosine_fallback(user_id, query, q_emb, query_entity)

            if _query_person and edges:
                filtered_edges, was_filtered = self._filter_by_speaker(edges, _query_person)
                if was_filtered and not filtered_edges:
                    return StructuralRefusal(
                        reason="speaker_not_found",
                        text=REFUSAL_TEXT,
                        confidence=0.0,
                        survivors=0,
                    )
                if filtered_edges:
                    edges = filtered_edges

        # No edges -> refuse (with CWA check for yes/no)
        if not edges:
            if _is_yes_no_query(query) and query_entity:
                if self._entity_exists(user_id, query_entity):
                    return Answer(
                        text="No",
                        confidence=0.8,
                        source="confirmed_absence",
                        convergence_details={
                            "reason": "entity_exists_predicate_absent",
                            "entity": query_entity,
                        },
                    )
            return StructuralRefusal(
                reason="no_candidates",
                text=REFUSAL_TEXT,
                confidence=0.0,
            )

        # ─�� Step 4: Return the right field ───────────────────────────
        winning_edge = edges[0]
        answer_text = self._extract_answer_from_edge(
            winning_edge, qd, query_entity,
        )

        if not answer_text:
            answer_text = (winning_edge.get("source_text") or "").strip()

        if not answer_text:
            if _is_yes_no_query(query) and query_entity:
                if self._entity_exists(user_id, query_entity):
                    return Answer(
                        text="No",
                        confidence=0.8,
                        source="confirmed_absence",
                    )
            return StructuralRefusal(
                reason="empty_answer_field",
                text=REFUSAL_TEXT,
                confidence=0.0,
            )

        # Post-verification: winning edge must mention query person
        if _query_person:
            mentioned = _edge_mentions_entity(
                winning_edge.get("subject", ""),
                winning_edge.get("object", ""),
                _query_person,
                winning_edge.get("relational_entities", ""),
            )
            if not mentioned:
                return StructuralRefusal(
                    reason="speaker_mismatch",
                    text=REFUSAL_TEXT,
                    confidence=0.0,
                )

        # Yes/no: if we found a matching edge, answer "Yes"
        if _is_yes_no_query(query):
            answer_text = "Yes"

        result = Answer(
            text=answer_text,
            subject=winning_edge.get("subject"),
            predicate=winning_edge.get("predicate"),
            object=winning_edge.get("object"),
            confidence=0.9,
            source="sql_lookup",
            survivors=len(edges),
            convergence_details={
                "relationship_id": winning_edge.get("id"),
                "return_field": qd.return_field if qd else "episodic",
                "entity": query_entity,
            },
        )

        # PQ write-back
        rid = winning_edge.get("id")
        if rid is not None:
            self._writeback_pq(user_id, query, rid, answer_text)

        return result

    # ===================================================================
    # PUBLIC: reconstruct()
    # ===================================================================

    def reconstruct(self, user_id: int, query: str) -> Situation:
        """Reconstruction path: recluster and render narrative."""

        if not query or not query.strip():
            return Situation(narrative="", survivor_count=0)

        q_emb = embed_text(query)
        edges = self._cosine_fallback(user_id, query, q_emb)

        if not edges:
            return Situation(narrative=REFUSAL_TEXT, survivor_count=0)

        top = edges[:RECONSTRUCT_TOP_N]

        from app.engines.temporal import get_temporal_engine
        edge_ids = [e.get("id") for e in top if e.get("id")]
        temporal_clusters = get_temporal_engine().recluster_for_reconstruction(
            user_id, edge_ids,
        )

        id_to_edge = {e.get("id"): e for e in top}
        cluster_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for tc in temporal_clusters:
            cid = str(tc.get("cluster_id", 0))
            for eid in tc.get("edge_ids", []):
                if eid in id_to_edge:
                    cluster_edges[cid].append(id_to_edge[eid])

        # Enrich with temporal neighbors
        _te_recon = get_temporal_engine()
        existing_ids = {e.get("id") for e in top}
        for cid, cands in list(cluster_edges.items()):
            cluster_entities = set()
            for e in cands:
                cluster_entities.add((e.get("subject") or "").lower())
                cluster_entities.add((e.get("object") or "").lower())
            cluster_entities.discard("")
            cluster_entities.discard("user")

            for e in list(cands):
                for n in _te_recon.temporal_neighbors(user_id, e.get("id")):
                    nid = n.get("id")
                    if not nid or nid in existing_ids:
                        continue
                    n_subj = (n.get("subject") or "").lower()
                    n_obj = (n.get("object") or "").lower()
                    if n_subj in cluster_entities or n_obj in cluster_entities:
                        cands.append(n)
                        existing_ids.add(nid)

        # Build Cluster objects
        _te_humanize = _te_recon
        clusters: List[Cluster] = []
        all_participants: Set[str] = set()
        narratives: List[str] = []

        for cid, cands in cluster_edges.items():
            def _sort_key(e):
                date = e.get("resolved_event_date") or ""
                seq = e.get("sequence_number") or 0
                return (date, seq)

            if self._temporal and len(cands) > 1:
                import functools
                def _cmp(a, b):
                    da = a.get("resolved_event_date")
                    db = b.get("resolved_event_date")
                    if da and db:
                        try:
                            result = self._temporal.is_before(da, db)
                            if result is True:
                                return -1
                            elif result is False:
                                return 1
                        except Exception:
                            pass
                    ka = _sort_key(a)
                    kb = _sort_key(b)
                    return (ka > kb) - (ka < kb)
                cands.sort(key=functools.cmp_to_key(_cmp))
            else:
                cands.sort(key=_sort_key)

            participants: Set[str] = set()
            edges_list: List[Dict[str, Any]] = []
            tones: List[str] = []
            categories: List[str] = []

            for e in cands:
                participants.add(e.get("subject") or "")
                participants.add(e.get("object") or "")
                edges_list.append(e)
                t = e.get("edge_emotional_label")
                if t:
                    tones.append(t)
                cat = e.get("edge_schematic_category")
                if cat:
                    categories.append(cat)

            participants.discard("")
            all_participants.update(participants)

            dates = [e.get("resolved_event_date") for e in edges_list if e.get("resolved_event_date")]

            parts: List[str] = []
            for e in edges_list:
                st = (e.get("source_text") or "").strip()
                if st:
                    parts.append(st)
                else:
                    parts.append(_edge_to_fact_statement(e))

            narrative = " ".join(parts)

            if dates and self._temporal:
                try:
                    ds = self._temporal.days_since(dates[0])
                    if ds is not None:
                        prox = self._temporal.proximity(dates[0])
                        if prox:
                            narrative = f"({prox}) {narrative}"
                except Exception:
                    pass
            narratives.append(narrative)

            cat_dominant = None
            if categories:
                cat_dominant = Counter(categories).most_common(1)[0][0]
            tone_dominant = None
            if tones:
                tone_dominant = Counter(tones).most_common(1)[0][0]

            clusters.append(Cluster(
                cluster_id=cid,
                participants=sorted(participants),
                edges=edges_list,
                narrative=narrative,
                schema_category=cat_dominant,
                emotional_tone=tone_dominant,
            ))

        # Upcoming events
        if self._temporal:
            try:
                upcoming = self._temporal.upcoming_events(user_id)
                if upcoming:
                    ue_edges = []
                    ue_parts = []
                    for ev in upcoming:
                        prox = ev.get("proximity", "")
                        desc = "{} {} {}".format(ev["subject"], ev["predicate"], ev["object"])
                        if prox:
                            desc = "{} ({})".format(desc, prox)
                        ue_parts.append(desc)
                        ue_edges.append({
                            "id": ev["id"], "subject": ev["subject"],
                            "predicate": ev["predicate"], "object": ev["object"],
                            "source_text": desc,
                        })
                    if ue_parts:
                        ue_narrative = "Upcoming: " + ". ".join(ue_parts)
                        narratives.append(ue_narrative)
                        clusters.append(Cluster(
                            cluster_id="upcoming", participants=[],
                            edges=ue_edges, narrative=ue_narrative,
                            schema_category="temporal", emotional_tone=None,
                        ))
            except Exception:
                pass

        full_narrative = " ".join(narratives)

        if self._memory and all_participants:
            try:
                for p in sorted(all_participants):
                    if p.lower() == "user" or not p:
                        continue
                    summary = self._memory.summarize(user_id, entity=p)
                    if summary and summary not in full_narrative:
                        full_narrative = full_narrative + " " + summary
            except Exception:
                pass

        return Situation(
            clusters=clusters,
            survivor_count=len(top),
            narrative=full_narrative,
            source_tag="reconstruct",
            participants=sorted(all_participants),
        )

    # ===================================================================
    # BACKWARDS COMPAT (test harnesses call v2 internals for debug prints)
    #
    # Root cause: _test_retrieval_isolation.py lines 131-133 call
    # _run_moat_pipeline and _load_facts_for_entity for debug output
    # before the actual test loop. Without these, the script crashes
    # at line 131 with AttributeError and no test assertions execute.
    # ===================================================================

    def _run_moat_pipeline(
        self, user_id: int, query: str,
    ) -> Tuple[List[Candidate], np.ndarray, List[Dict[str, Any]]]:
        """Compat shim: returns (candidates, q_emb, resolved_entities).
        Wraps SQL + cosine paths into the old return signature."""
        q_emb = embed_text(query)
        edges = self._sql_edge_lookup(user_id, None)
        if not edges:
            edges = self._cosine_fallback(user_id, query, q_emb)
        candidates = [
            Candidate(
                relationship_id=e.get("id", 0),
                edge=e,
                exit_cosine=_cosine_from_blob(q_emb, e.get("edge_embedding")),
            )
            for e in edges
        ]
        candidates.sort(key=lambda c: c.exit_cosine, reverse=True)
        resolved_entities: List[Dict[str, Any]] = []
        try:
            from app.engines.entity_resolver import resolve_query_entities
            for ent in resolve_query_entities(user_id, query):
                resolved_entities.append(ent)
        except Exception:
            pass
        return candidates, q_emb, resolved_entities

    def _load_facts_for_entity(
        self, user_id: int, entity_name: str, query_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Compat shim: load canonical fact for entity matching query.
        Searches the facts table by key suffix using SQL LIKE,
        then ranks by query-word overlap."""
        if not entity_name:
            return None
        q_words = _normalized_words(query_text)
        entity_words = _normalized_words(entity_name)
        with get_db_context() as conn:
            rows = conn.execute(
                "SELECT key, value, confidence FROM facts "
                "WHERE user_id = ? AND LOWER(key) LIKE ?",
                (user_id, f'%::{entity_name.lower()}'),
            ).fetchall()
        if not rows:
            return None
        best_match = None
        best_overlap = 0
        for row in rows:
            key_words = _normalized_words(row["key"] or "") - entity_words
            overlap = len((q_words - entity_words) & key_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = {
                    "key": row["key"], "value": row["value"],
                    "confidence": row["confidence"],
                }
        return best_match


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
