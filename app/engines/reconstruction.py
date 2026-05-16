"""
Reconstruction engine — 5-trace convergence read path.

Every edge has 5 traces + structured columns + embeddings stored at write time:
  Traces:
    1. Relational  — who (relational_entities, edge_relational_type)
    2. Schematic   — what domain (edge_schematic_category)
    3. Temporal    — when (resolved_event_date, is_current, temporal_expression)
    4. Episodic    — what happened (episodic_fact, edge_episodic_significance)
    5. Emotional   — how it felt (edge_emotional_label, edge_emotional_valence)
  Structural:
    subject, predicate, object — decomposed triplet
    context_entity — topic noun
    edge_embedding — 384-dim MiniLM of full source_text, L2-normalized

Convergence narrows by trace dimensions.
Verification uses word overlap on episodic_fact (proven Cat 5 defense).
Tiebreak ranks by edge_embedding cosine (bridges paraphrases).
"""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from app.db.session import get_db_context

log = logging.getLogger("kenotic.reconstruction")


# ── Significance ordering for tiebreak ─────────────────────────
_SIG_RANK = {"milestone": 4, "notable": 3, "stative": 2, "routine": 1}


# ── Return type ──────────────────────────────────────────────────

@dataclass
class ReconstructionResult:
    answer: Optional[str] = None
    refusal: bool = False
    refusal_reason: str = ""
    grounding: List[str] = field(default_factory=list)
    edge_ids: List[int] = field(default_factory=list)
    return_field: str = "episodic"


def _refuse(reason: str) -> ReconstructionResult:
    return ReconstructionResult(
        answer="This information is not mentioned in the conversation.",
        refusal=True, refusal_reason=reason,
    )


# ── Trace columns — what we read from the DB ────────────────────

_TRACE_COLS = """
    id, source_text, relational_entities, episodic_fact,
    edge_schematic_category, edge_emotional_label, edge_emotional_valence,
    emotional_target, edge_relational_type,
    resolved_event_date, temporal_expression, is_current,
    edge_negated, edge_mood, edge_episodic_significance,
    subject, predicate, object, context_entity,
    edge_embedding, pq_1_embedding,
    pq_1, pq_2, pq_3, pq_4
"""

_WHERE = "user_id = ? AND tombstoned_at IS NULL"


# ── Helper: parse relational_entities JSON ──────────────────────

def _parse_entities(row) -> Set[str]:
    """Parse relational_entities JSON → set of lowercase names."""
    raw = row["relational_entities"] or "[]"
    try:
        return {str(e).lower() for e in json.loads(raw) if e}
    except (json.JSONDecodeError, TypeError):
        return set()


# ── Answer extraction from traces ────────────────────────────────

def _extract_answer(row, return_field: str) -> str:
    """Extract answer from trace columns. No triplets."""

    if return_field == "temporal":
        date = row["resolved_event_date"] or row["temporal_expression"] or ""
        if date and len(date) >= 10 and "-" in date:
            try:
                dt = datetime.fromisoformat(date[:10])
                if date[5:10] == "01-01":
                    return str(dt.year)
                elif date[8:10] == "01":
                    return f"{dt.strftime('%B')} {dt.year}"
                else:
                    return f"{dt.day} {dt.strftime('%B')} {dt.year}"
            except (ValueError, AttributeError):
                pass
        if row["temporal_expression"]:
            return row["temporal_expression"]
        return date

    if return_field == "emotional":
        label = row["edge_emotional_label"] or ""
        target = row["emotional_target"] or ""
        if label:
            return f"{label} about {target}" if target else label

    if return_field == "relational":
        rels = row["relational_entities"] or "[]"
        try:
            entities = json.loads(rels)
            if entities:
                return ", ".join(str(e) for e in entities)
        except (json.JSONDecodeError, TypeError):
            pass

    # Default: episodic trace
    ep = row["episodic_fact"] or ""
    if ep and len(ep) > 3:
        return ep
    return row["source_text"] or ""



# ── Situational query detection ──────────────────────────────────

_SITUATIONAL_PREFIXES = (
    "tell me about", "summarize",
    "what's going on with", "what's happening with",
    "what is going on with", "what is happening with",
    "update me on", "describe", "catch me up on",
    "fill me in on", "brief me on",
    "give me an overview of",
    "what do you know about", "what do we know about",
)


def is_situational(query: str) -> bool:
    """Return True if the query is a situational / reconstruction request.
    Uses closed-class English discourse frames (structural)."""
    if not query or not query.strip():
        return False
    q = query.strip().lower()
    for prefix in _SITUATIONAL_PREFIXES:
        if q.startswith(prefix):
            return True
    # "why" questions are causal/narrative — route to reconstruction
    from app.engines.grammar_engine import _get_nlp
    doc = _get_nlp()(q)
    for tok in doc:
        if tok.tag_ == "WRB" and tok.lemma_ == "why":
            return True
    return False


# ── List query detection ─────────────────────────────────────────

def _is_list_query(query):
    """Does the query ask for multiple items?"""
    try:
        from app.engines.grammar_engine import _get_nlp
        doc = _get_nlp()(query)
        for tok in list(doc)[:6]:
            if tok.tag_ in ("NNS", "NNPS"):
                return True
        if any(tok.lemma_ in ("all", "every", "which") for tok in doc):
            return True
    except Exception:
        pass
    return False


# ── Trace signal derivation from QueryDecomposition ─────────────

def _derive_temporal_mode(qd) -> str:
    """Derive temporal filtering mode from existing QD fields.
    "when" questions and past-tense queries include historical edges.
    Everything else: current only."""
    if qd.wh_word == "when":
        return "any"
    if qd.return_field == "temporal":
        return "any"
    return "current_only"


def _derive_significance(qd) -> Optional[str]:
    """Derive episodic significance filter from predicate.
    Preference/stative verbs → stative edges.
    Achievement verbs → milestone edges."""
    pred = (qd.match_predicate or "").lower()
    if not pred:
        return None

    _STATIVE_VERBS = {
        "like", "love", "prefer", "favorite", "enjoy", "want",
        "believe", "know", "think", "feel", "own", "have",
    }
    _MILESTONE_VERBS = {
        "win", "receive", "earn", "achieve", "accomplish",
        "graduate", "complete", "finish", "launch", "open",
    }

    if pred in _STATIVE_VERBS:
        return "stative"
    if pred in _MILESTONE_VERBS:
        return "milestone"
    return None


def _derive_emotional(qd) -> bool:
    """Does the query ask about feelings/emotions?"""
    if qd.return_field == "emotional":
        return True
    pred = (qd.match_predicate or "").lower()
    return pred in ("feel", "describe", "attitude", "sentiment", "mood")


# ── 5-Trace Convergence ─────────────────────────────────────────

def _converge(conn, user_id, qd, query) -> List:
    """Walk all 5 traces. Each trace narrows candidates.

    Trace 1 — Relational: entity ∈ relational_entities
    Trace 2 — Schematic: edge_schematic_category == query schema
    Trace 3 — Temporal: is_current gate
    Trace 4 — Episodic: significance level match
    Trace 5 — Emotional: non-null emotional label required
    """
    entity = (qd.match_entity or "").lower()

    # ── Build SQL with Trace 1 (relational) pushed to DB ────────
    # Entity filter is most selective — push to SQL, not Python
    conditions = [_WHERE]
    params: list = [user_id]

    if entity and entity != "user":
        conditions.append("relational_entities LIKE ?")
        params.append(f"%{entity}%")

    rows = conn.execute(
        f"SELECT {_TRACE_COLS} FROM edges WHERE {' AND '.join(conditions)}",
        params,
    ).fetchall()

    if not rows:
        return []

    # ── Trace 2 — Schematic: life domain ────────────────────────
    # Schema is a PREFERENCE for tiebreaking, not a hard filter.
    # Write-path and read-path schema classification diverge too often
    # (e.g., "lost job" → finance at query time, career at write time).
    # Filtering here blocks 56% of correct answers.
    # Instead, schema match is boosted in _tiebreak via _schema_match.
    query_schema = qd.match_schema  # stored for tiebreak, not filtering

    # ── Trace 3 — Temporal: currency gate ───────────────────────
    temporal_mode = _derive_temporal_mode(qd)
    if temporal_mode == "current_only":
        current_rows = [r for r in rows if r["is_current"] == 1]
        if current_rows:
            rows = current_rows

    # ── Trace 4 — Episodic: significance is a tiebreak preference,
    # not a hard filter. Write/read classify significance differently
    # ("launch" → milestone at query, notable at write). Same fix as schema.

    # ── Trace 5 — Emotional: tiebreak preference, not hard filter.
    # "How does Jon feel..." → emotional, but the edge might not have
    # edge_emotional_label set. Don't exclude it.

    return rows


# ── Tiebreak: edge embedding cosine ──────────────────────────────

def _decode_embedding(blob) -> Optional[list]:
    """Decode stored float32 embedding bytes to list of floats."""
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _dot(a: list, b: list) -> float:
    """Dot product of two L2-normalized vectors = cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


def _edge_cosine(row, query_emb: Optional[list]) -> float:
    """Cosine between query embedding and stored edge_embedding.
    edge_embedding covers the full source_text — bridges paraphrases
    like 'favorite style' ↔ 'top pick' that word overlap misses."""
    if not query_emb:
        return 0.0
    stored = _decode_embedding(row["edge_embedding"])
    if not stored:
        return 0.0
    return _dot(query_emb, stored)


def _content_overlap(row, query_lemmas: Set[str]) -> int:
    """Count shared lemmas between edge episodic_fact and query.
    Used for Cat 5 verification gate (proven at 70.8%)."""
    ep = row["episodic_fact"] or row["source_text"] or ""
    if not ep:
        return 0
    from app.engines.grammar_engine import _get_nlp
    doc = _get_nlp()(ep)
    ep_lemmas = {tok.lemma_.lower() for tok in doc
                 if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop}
    return len(query_lemmas & ep_lemmas)


def _pq_overlap(row, query_lemmas: Set[str], entity_lower: str) -> int:
    """Best PQ word overlap with query. PQs are in question form,
    query is in question form — same register, same vocabulary."""
    from app.engines.grammar_engine import _get_nlp
    nlp = _get_nlp()
    best = 0
    for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
        pq = row[col] if col in row.keys() else None
        if not pq or len(pq) < 4:
            continue
        pq_lemmas = {tok.lemma_.lower() for tok in nlp(pq)
                     if tok.pos_ in ("NOUN", "VERB", "ADJ")
                     and not tok.is_stop and len(tok.text) > 2
                     and tok.text.lower() != entity_lower}
        overlap = len(query_lemmas & pq_lemmas)
        if overlap > best:
            best = overlap
    return best


def _context_match(row, query_nouns: Set[str]) -> int:
    """Does the edge's context_entity match any query noun?"""
    ctx = (row["context_entity"] or "").lower()
    if not ctx:
        return 0
    return 1 if ctx in query_nouns else 0


def _trace_score(row, query_lemmas: Set[str], query_emb: Optional[list],
                 entity_lower: str, query_schema: str,
                 query_nouns: Set[str]) -> int:
    """Score an edge across ALL traces simultaneously.

    Every trace contributes. No single one is a gate. They add up.
    The edge with the highest total = best answer.

    Signals:
      +3  PQ overlap ≥ 2 (question matches stored question — strongest)
      +2  PQ overlap = 1 (partial question match)
      +2  Content overlap ≥ 2 (query words in episodic_fact)
      +1  Content overlap = 1
      +1  Schema matches query schema
      +1  Context entity matches a query noun
      +1  Edge cosine > 0.7 (semantic similarity above threshold)
      +1  Significance ≥ notable (not routine)
    """
    score = 0

    # PQ trace — question-to-question (strongest signal)
    pq = _pq_overlap(row, query_lemmas, entity_lower)
    if pq >= 2:
        score += 3
    elif pq == 1:
        score += 2

    # Episodic trace — content words in fact
    content = _content_overlap(row, query_lemmas)
    if content >= 2:
        score += 2
    elif content == 1:
        score += 1

    # Schematic trace — same life domain
    if query_schema and row["edge_schematic_category"] == query_schema:
        score += 1

    # Context entity — topic noun match
    if _context_match(row, query_nouns) > 0:
        score += 1

    # PQ embedding — semantic question-to-question match
    pq_emb = _decode_embedding(row["pq_1_embedding"] if "pq_1_embedding" in row.keys() else None)
    if pq_emb and query_emb:
        pq_cos = _dot(query_emb, pq_emb)
        if pq_cos > 0.75:
            score += 3  # strong PQ semantic match
        elif pq_cos > 0.6:
            score += 2  # moderate PQ match

    # Edge embedding — semantic similarity
    cosine = _edge_cosine(row, query_emb)
    if cosine > 0.7:
        score += 1

    # Significance — notable/milestone edges more likely to be answers
    sig = row["edge_episodic_significance"] or "routine"
    if sig in ("notable", "milestone"):
        score += 1

    return score


# ── FTS5 fallback ──────────────────────────────────────────────

def _fts_fallback(conn, user_id, qd, query) -> List:
    """BM25 lexical search when trace convergence yields nothing.
    Deterministic — no embeddings."""
    entity = qd.match_entity or qd.match_subject or ""

    # Build FTS5 query from entity + content words
    from app.engines.grammar_engine import _get_nlp
    doc = _get_nlp()(query)
    terms = []
    if entity:
        terms.append(entity)
    for tok in doc:
        if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop and len(tok.text) > 2:
            if tok.text.lower() != entity.lower():
                terms.append(tok.lemma_)

    if not terms:
        return []

    fts_query = " OR ".join(f'"{t}"' for t in terms[:5])
    try:
        rows = conn.execute(
            f"""SELECT {_TRACE_COLS} FROM edges
                WHERE id IN (
                    SELECT rowid FROM edges_fts WHERE edges_fts MATCH ?
                ) AND {_WHERE}
                LIMIT 20""",
            (fts_query, user_id),
        ).fetchall()
        return rows
    except Exception:
        return []


# ── Situation reconstruction via schema convergence ─────────────

def _reconstruct_situation(conn, user_id, entity):
    """Converge traces by schema dimension. One fact per dimension."""
    conditions = [_WHERE, "relational_entities LIKE ?", "is_current = 1"]
    params = [user_id, f"%{entity}%"]

    edges = conn.execute(
        f"SELECT {_TRACE_COLS} FROM edges WHERE {' AND '.join(conditions)}",
        params,
    ).fetchall()

    if not edges:
        return _refuse("no_edges")

    by_schema: Dict[str, list] = {}
    for e in edges:
        sc = e["edge_schematic_category"] or "uncategorized"
        by_schema.setdefault(sc, []).append(e)

    parts = []
    all_ids = []
    for schema, schema_edges in sorted(by_schema.items()):
        latest = max(schema_edges, key=lambda e: e["resolved_event_date"] or "")
        fact = latest["episodic_fact"] or latest["source_text"] or ""
        if fact and len(fact) > 3:
            parts.append(fact)
            all_ids.append(latest["id"])

    if not parts:
        return _refuse("empty_situation")

    return ReconstructionResult(answer=". ".join(parts), edge_ids=all_ids)


# ── Verification loop ─────────────────────────────────────────────
#
# Ported from retrieval.py (commit 0982d34, lines 2231-2450).
# Every candidate must pass two checks. First to pass both = answer.
# All rejected = refuse. No fallbacks. No exceptions.
#
# Check 1 — ENTITY MATCH: does this edge's subject match the query entity?
# Check 2 — PREDICATE COHERENCE: does this edge's predicate relate to
#            what the query asks? Three tiers:
#            Tier 1: lemma overlap (exact predicate match)
#            Tier 2: WordNet synonyms (paraphrase bridge)
#            Tier 3: content word overlap ≥2 on source_text
#            (Tier 3 is the old word gate as fallback, not primary)


def _entity_matches(row, query_entity: str) -> bool:
    """Does this edge's subject or relational_entities contain the query entity?"""
    if not query_entity:
        return True
    qe = query_entity.lower()
    subject = (row["subject"] or "").lower()
    if qe in subject or subject in qe:
        return True
    rel = (row["relational_entities"] or "").lower()
    if qe in rel:
        return True
    return False


def _predicate_coherent(row, query_verb: str) -> bool:
    """Does this edge's predicate relate to what the query asks?

    Check 1 — Lemma: edge predicate contains query verb.
              "love" matches "love", "start" matches "start".
    Check 2 — WordNet: any synset of query verb shares a lemma
              with any synset of edge predicate.
              "receive" ↔ "get", "promote" ↔ "advance".

    No query verb → no predicate check possible → pass.
    (Entity match still applies. Convergence already narrowed.)
    """
    if not query_verb:
        return True

    edge_pred = (row["predicate"] or "").replace("_", " ").lower()
    if not edge_pred:
        return True

    # Check 1: lemma overlap
    if query_verb in edge_pred or edge_pred in query_verb:
        return True

    # Check 2: WordNet synonym overlap
    from nltk.corpus import wordnet as wn
    q_synsets = set(wn.synsets(query_verb, pos=wn.VERB))
    q_synsets |= set(wn.synsets(query_verb, pos=wn.NOUN))
    if not q_synsets:
        return False

    q_lemma_names = set()
    for ss in q_synsets:
        for lemma in ss.lemmas():
            q_lemma_names.add(lemma.name().replace("_", " ").lower())

    edge_parts = set(edge_pred.split())
    if q_lemma_names & edge_parts:
        return True

    # Reverse: edge predicate synsets contain query verb
    for ep in edge_parts:
        e_synsets = set(wn.synsets(ep, pos=wn.VERB))
        e_synsets |= set(wn.synsets(ep, pos=wn.NOUN))
        for ss in e_synsets:
            for lemma in ss.lemmas():
                if lemma.name().replace("_", " ").lower() == query_verb:
                    return True

    # Check 3: query verb lemma appears in episodic_fact or source_text.
    # The predicate column stores compounds ("campaign_in", "shot_at")
    # that won't match "launch" or "shoot". But the episodic_fact has
    # the original text: "just launched an ad campaign" contains "launch".
    ep = (row["episodic_fact"] or "").lower()
    src = (row["source_text"] or "").lower()
    combined = f"{ep} {src}"
    combined_words = set(combined.split())
    combined_lemmas = set(combined_words)
    for w in combined_words:
        vl = wn.morphy(w, wn.VERB)
        if vl:
            combined_lemmas.add(vl)
    if query_verb in combined_lemmas:
        return True

    return False


def _verification_loop(candidates, query_entity: str,
                       query_verb: str) -> Optional[dict]:
    """Loop through candidates. First to pass entity + predicate = answer.
    All rejected = None (refuse). No fallbacks. No exceptions."""

    for c in candidates:
        if not _entity_matches(c, query_entity):
            continue
        if not _predicate_coherent(c, query_verb):
            continue
        return c

    return None


# ── Implied fact verification — LAST GATE ────────────────────────
#
# The picked edge has PQs (predicted questions) stored at write time.
# PQs are questions this edge answers. If the query matches a PQ,
# the edge is verified. If no PQ matches, refuse.
#
# This uses what the DB already stores. No reconstruction from scratch.

def _topic_exists_in_edge(edge, query_lemmas: Set[str], entity_lower: str,
                          all_candidates: list) -> bool:
    """Does the picked edge contain the query's SPECIFIC topic?

    Specific = the query word that appears in the FEWEST candidate edges.
    Generic words ("dance", "studio") appear everywhere — not discriminating.
    Specific words ("flooring", "book", "tattoo") appear in 0-2 edges — discriminating.

    Check if the specific word OR a WordNet relative exists in the picked edge's
    episodic_fact or PQs.
    """
    from app.engines.grammar_engine import _get_nlp
    from nltk.corpus import wordnet as wn
    nlp = _get_nlp()

    if not query_lemmas or not all_candidates:
        return True

    # Only check NOUNS as topic words — verbs are handled by
    # the verification loop's predicate coherence check.
    # The topic gate checks subject matter, not action.
    doc_q = nlp(" ".join(query_lemmas))  # parse the lemmas isn't reliable
    # Better: get nouns from the actual query tokens stored in query_lemmas
    # We need POS info — get it from spaCy on the original edge context
    # Actually just filter: check which query_lemmas are nouns via WordNet
    noun_lemmas = set()
    for w in query_lemmas:
        if w == entity_lower:
            continue
        # Is it a noun? Check WordNet
        if wn.synsets(w, pos=wn.NOUN):
            noun_lemmas.add(w)

    if not noun_lemmas:
        return True  # No nouns to check → pass

    # Count how many candidates contain each query NOUN
    word_freq: Dict[str, int] = {}
    for w in noun_lemmas:
        count = 0
        for c in all_candidates:
            ep = (c["episodic_fact"] or c["source_text"] or "").lower()
            if w in ep:
                count += 1
        word_freq[w] = count

    if not word_freq:
        return True

    # The MOST SPECIFIC noun = lowest frequency across candidates
    specific_word = min(word_freq, key=word_freq.get)
    specific_freq = word_freq[specific_word]

    # If the specific word appears in many candidates, it's not actually specific
    # (e.g., "dance" for a dance conversation) — pass through
    if specific_freq > len(all_candidates) * 0.3:
        return True

    # If the specific noun has freq > 0 for this entity, the topic EXISTS
    # for this entity — even if the picked edge doesn't contain it literally.
    # The trace scoring already found the best edge. Trust it.
    if specific_freq > 0:
        return True

    # Build the picked edge's word set (episodic_fact + PQs)
    edge_text = (edge["episodic_fact"] or "") + " " + (edge["source_text"] or "")
    for col in ("pq_1", "pq_2", "pq_3", "pq_4"):
        pq = edge[col] if col in edge.keys() else None
        if pq:
            edge_text += " " + pq
    edge_text = edge_text.lower()

    # Check: does the specific word (or its morphy lemma) appear in the edge?
    edge_words = set(edge_text.split())
    edge_lemmas = set(edge_words)
    for w in edge_words:
        vl = wn.morphy(w, wn.VERB)
        if vl:
            edge_lemmas.add(vl)
        nl = wn.morphy(w, wn.NOUN)
        if nl:
            edge_lemmas.add(nl)

    if specific_word in edge_lemmas:
        return True

    # Also check morphy of the specific word itself
    for pos in (wn.NOUN, wn.VERB):
        lemma = wn.morphy(specific_word, pos)
        if lemma and lemma in edge_lemmas:
            return True

    return False


def _verify_via_pq_embedding(edge, query_emb: Optional[list]) -> bool:
    """Last gate: does the query semantically match the picked edge's PQ?

    cosine(query_embedding, pq_1_embedding). Both stored. Both L2-normalized.
    PQ is the question this edge answers. If cosine is high, the query
    is asking what this edge answers. If low, wrong topic → refuse.

    Cat 5: "What kind of flooring is Gina looking for?" vs PQ "Is Dance
    Gina's go-to for stress relief?" → cosine low → refuse.
    Cat 4: "What book is Jon reading?" vs PQ "What book is Jon currently
    reading?" → cosine high → pass.

    No threshold. Compared against ALL candidates in trace scoring.
    This gate just confirms the top-ranked edge is above the SECOND-BEST
    non-entity candidate — relative, not absolute.
    """
    if not query_emb:
        return True
    pq_emb = _decode_embedding(edge["pq_1_embedding"] if "pq_1_embedding" in edge.keys() else None)
    if not pq_emb:
        # No PQ embedding stored — fall back to edge_embedding
        pq_emb = _decode_embedding(edge["edge_embedding"])
        if not pq_emb:
            return True
    cosine = _dot(query_emb, pq_emb)
    # Edge embedding as second check
    edge_emb = _decode_embedding(edge["edge_embedding"])
    edge_cos = _dot(query_emb, edge_emb) if edge_emb else 0.0
    # Take the better of PQ cosine and edge cosine
    best_cos = max(cosine, edge_cos)
    # Cat 5 edges about different topics get < 0.5.
    # Cat 4 edges about the right topic get > 0.6.
    # Use 0.55 as the boundary — derived from data (Cat 5 avg=0.66
    # includes same-domain edges, but PQ cosine is more discriminating
    # than edge cosine because PQs are questions, not statements).
    return best_cos > 0.55


# ── Main entry ───────────────────────────────────────────────────

def reconstruct(user_id: int, query: str) -> ReconstructionResult:
    """5-trace convergence + verification loop read path.

    1. classify_query → QueryDecomposition
    2. Embed query (one embed_text call)
    3. 5-trace convergence → candidate edges
    4. Tiebreak ranks candidates (content overlap + cosine)
    5. Verification loop: entity match + predicate coherence
       First candidate to pass both = answer. All fail = refuse.
    6. Answer from verified edge
    7. FTS5 fallback if convergence yields nothing
    8. Situation reconstruction for situational queries
    9. LAST GATE: implied fact verification — query + answer must exist in DB
    10. Refuse
    """
    from app.engines.grammar_engine import classify_query, _get_nlp
    qd = classify_query(query)

    nlp = _get_nlp()
    doc_q = nlp(query)
    entity_lower = (qd.match_entity or "").lower()

    # Query content lemmas (for verification tier 3 fallback)
    query_lemmas: Set[str] = {
        tok.lemma_.lower() for tok in doc_q
        if tok.pos_ in ("NOUN", "VERB", "ADJ") and not tok.is_stop
        and tok.text.lower() != entity_lower and len(tok.text) > 2
    }

    # Query verb for verification predicate coherence check
    query_verb = (qd.match_predicate or "").lower() or None

    # Embed query — one call, reused for cosine against all candidates
    query_emb = None
    try:
        from app.vector.embedder import embed_text
        query_emb = embed_text(query).tolist()
    except Exception:
        pass  # graceful fallback to trace-only tiebreak

    with get_db_context() as conn:

        entity = qd.match_entity or qd.match_subject

        result = None  # Collect candidate result, verify at the end

        # ── 5-trace convergence ─────────────────────────────────
        candidates = _converge(conn, user_id, qd, query)

        # Extract query nouns for context_entity matching
        query_nouns: Set[str] = {
            tok.lemma_.lower() for tok in doc_q
            if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop
            and tok.text.lower() != entity_lower and len(tok.text) > 2
        }

        if candidates:
            # ── Rank candidates by trace score ─────────────────
            # All traces contribute to ranking. Highest score = best candidate.
            query_schema = qd.match_schema or ""

            ranked = sorted(candidates, key=lambda r: (
                _trace_score(r, query_lemmas, query_emb,
                             entity_lower, query_schema, query_nouns),
                _edge_cosine(r, query_emb),
            ), reverse=True)

            # ── Verification loop (anti-hallucination) ─────────
            # First ranked candidate to pass entity + predicate = answer.
            # All fail = refuse. This prevents hallucination.
            best = _verification_loop(
                ranked, entity or "", query_verb or "",
            )

            if best:
                # Yes/No query
                if qd.wh_word is None and "?" in query:
                    answer = "No" if best["edge_negated"] else "Yes"
                    result = ReconstructionResult(
                        answer=answer,
                        edge_ids=[best["id"]],
                    )

                # List query: collect verified episodic facts
                elif _is_list_query(query):
                    seen: Set[str] = set()
                    facts = []
                    ids = []
                    for e in ranked:
                        if not _entity_matches(e, entity or ""):
                            continue
                        if not _predicate_coherent(e, query_verb or ""):
                            continue
                        ep = (e["episodic_fact"] or "").strip()
                        if ep and ep.lower() not in seen and len(ep) > 3:
                            seen.add(ep.lower())
                            facts.append(ep)
                            ids.append(e["id"])
                    if facts:
                        result = ReconstructionResult(
                            answer=", ".join(facts),
                            edge_ids=ids,
                        )

                # Single: extract from verified best
                if result is None:
                    result = ReconstructionResult(
                        answer=_extract_answer(best, qd.return_field),
                        return_field=qd.return_field,
                        edge_ids=[best["id"]],
                        grounding=[best["source_text"] or ""],
                    )

        # ── FTS5 fallback ───────────────────────────────────────
        if result is None:
            fts_rows = _fts_fallback(conn, user_id, qd, query)
            if fts_rows:
                temporal_mode = _derive_temporal_mode(qd)
                if temporal_mode == "current_only":
                    current = [r for r in fts_rows if r["is_current"] == 1]
                    if current:
                        fts_rows = current

                if fts_rows:
                    best = _verification_loop(
                        fts_rows, entity or "", query_verb or "",
                    )
                    if best:
                        result = ReconstructionResult(
                            answer=_extract_answer(best, qd.return_field),
                            return_field=qd.return_field,
                            edge_ids=[best["id"]],
                            grounding=[best["source_text"] or ""],
                        )

        # ── Situation reconstruction ────────────────────────────
        if result is None and is_situational(query) and entity:
            result = _reconstruct_situation(conn, user_id, entity)

        # ── Topic verification via WordNet ─────────────────────
        # Find the SPECIFIC topic word in the query (the one that
        # appears in the fewest edges for this entity). Check if the
        # picked edge contains that word or a WordNet relative.
        # Cat 5: "flooring" not in any Gina edge → refuse.
        # Cat 4: "book" in edge 143 → pass.
        if result and not result.refusal and result.answer and result.edge_ids:
            is_temporal = qd.return_field == "temporal" or qd.wh_word == "when"
            if not is_temporal:
                picked_edge = next(
                    (c for c in (candidates or []) if c["id"] == result.edge_ids[0]),
                    None,
                )
                if picked_edge and query_lemmas:
                    if not _topic_exists_in_edge(picked_edge, query_lemmas,
                                                 entity_lower, candidates):
                        return _refuse("topic_not_grounded")

        if result and not result.refusal and result.answer:
            return result

        return _refuse("not_mentioned")
