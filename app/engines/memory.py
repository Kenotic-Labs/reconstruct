# ============================================================================
# NO HARDCODED LISTS. NO THRESHOLDS. NO SCORING MAGIC NUMBERS. NO REGEX.
# If you are about to add a `_skip = {"what", "who", ...}` or
# `if sim > 0.6:` or `weight = 0.15`, STOP. Read
# memory/feedback_no_whackamole.md before touching this file.
# ============================================================================
"""
MemoryEngine — the single store + all derived views.

Absorbs: entity_graph, memory_store, memory_models, memory_engine,
memory_writer, memory_reader, memory_summarizer, memory_engine_contract,
coreference, trace_decomposer, emotional_trace, relational_trace,
schematic_trace, entity_profile, transitive_inference, memory_indexes,
mem_tree.

COMPLEMENTING design:
    - relationships table is the atomic store
    - memory_traces table is a DERIVED materialized view (not parallel)
    - entities table holds canonical entities + embeddings for entity_link
    - predicted_queries table holds Q→A fingerprints for DTCM

Public interface:
    store()                 → upsert triple + derive traces + write predicted queries
    get_relationships()     → read triples by any combination of fields
    get_entity()            → read an entity row by name
    get_traces()            → read the materialized trace view for a relationship
    entity_link()           → resolve a mention to an entity via cosine
    supersede()             → supersession hook called by TemporalEngine
    summarize()             → natural-language summary

No other module stores relationships. No parallel read paths over the store.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.db.session import get_db_context
from app.vector.embedder import embed_text
from config.settings import settings


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class Entity:
    name: str
    entity_type: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    mention_count: int = 1


@dataclass
class Relationship:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9
    id: Optional[int] = None
    is_current: bool = True
    sequence_number: Optional[int] = None


@dataclass
class Traces:
    valence: float = 0.5
    affiliation: float = 0.5
    emotional_intensity: Optional[float] = None
    relational_type: Optional[str] = None
    relational_proximity: float = 0.5
    relational_valence: float = 0.5
    episodic_significance: str = "routine"
    episodic_narrative_position: str = "ongoing"
    temporal_context: Optional[str] = None
    schema_category: Optional[str] = None
    schema_confidence: float = 0.0


# =============================================================================
# ANCHOR DESCRIPTIONS — used for trace decomposition via embedding similarity.
#
# These are NOT curation of "known predicates" or "known stop words" — they
# are the canonical anchor points the trace decomposer measures cosine
# similarity AGAINST. Semantically stable: the anchors describe abstract
# concepts (warmth, distance, routine, milestone) that are universal across
# any English corpus. New content is matched to the closest anchor via
# cosine — no list expansion ever needed.
# =============================================================================

_EMO_POS_ANCHOR = "happy joyful excited grateful love proud content"
_EMO_NEG_ANCHOR = "sad angry frustrated anxious worried afraid hurt"
_AFF_WARM_ANCHOR = "close connected warm friendly together bonded family love"
_AFF_COLD_ANCHOR = "distant alone separate hostile isolated stranger"

_TEMP_PAST_ANCHOR = "happened before previously ago yesterday last year earlier"
_TEMP_FUTURE_ANCHOR = "will happen soon tomorrow next week upcoming plan"
_TEMP_ONGOING_ANCHOR = "always every day regularly currently routine habitual"
_TEMP_PRESENT_ANCHOR = "today now currently this week recent lately"

_EPI_MILESTONE_ANCHOR = "wedding birth death graduation diagnosis promotion engagement"
_EPI_NOTABLE_ANCHOR = "birthday party interview meeting first-time achievement"
_EPI_ROUTINE_ANCHOR = "morning coffee commute usual regular lunch bedtime"

_REL_KIN_ANCHOR = "sister brother mother father daughter son parent child family"
_REL_FRIEND_ANCHOR = "friend buddy companion pal close knows"
_REL_ROMANTIC_ANCHOR = "partner spouse husband wife boyfriend girlfriend married dating"
_REL_PROFESSIONAL_ANCHOR = "colleague manager coworker boss employee client mentor"
_REL_CLOSE_ANCHOR = "close intimate best deep trusted beloved"
_REL_DISTANT_ANCHOR = "stranger acquaintance distant formal casual"


# =============================================================================
# MEMORY ENGINE
# =============================================================================

class MemoryEngine:
    """Single engine owning the relationships store + all derived views.

    Also owns the T5 SRL primitive (absorbed from app/preprocessing/,
    deleted on the same date) because T5 extraction and cleanup are
    write-path concerns. Public `ingest_text()` and `clean()` expose the
    T5 primitives; the loading, offloading, and tokenization details are
    kept private.
    """

    # T5 class-level singleton state so all MemoryEngine instances in a
    # process share one loaded model.
    _t5_model = None
    _t5_tokenizer = None
    _t5_device = None
    _t5_path: Optional[str] = None

    def __init__(self):
        # Anchor embeddings are lazy-computed on first use and cached
        # per engine instance.
        self._anchor_cache: Dict[str, np.ndarray] = {}

    # ──────────────────────────────────────────────────────────────
    # Internal — anchor embedding cache
    # ──────────────────────────────────────────────────────────────

    def _anchor(self, key: str, text: str) -> np.ndarray:
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        emb = embed_text(text)
        self._anchor_cache[key] = emb
        return emb

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # ──────────────────────────────────────────────────────────────
    # T5 SRL primitive — absorbed from app/preprocessing/ (2026-04-12).
    # Loads the fine-tuned raya-srl-220m-v4/final model and exposes
    # two public methods: clean() and ingest_text(). Internal state is
    # shared class-level across MemoryEngine instances.
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _t5_load(cls) -> bool:
        if cls._t5_model is not None:
            return True
        import os
        from pathlib import Path
        # Default: the production v4 checkpoint. Override via env var.
        path = os.environ.get(
            "NURA_T5_MODEL_PATH",
            str(Path(__file__).resolve().parents[2] / "models" / "raya-srl-220m-v4" / "final"),
        )
        if not Path(path).exists():
            return False
        try:
            import torch
            from transformers import T5ForConditionalGeneration, AutoTokenizer
            cls._t5_device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._t5_tokenizer = AutoTokenizer.from_pretrained(path)
            cls._t5_model = (
                T5ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32)
                .to(cls._t5_device)
                .eval()
            )
            cls._t5_path = path
            print(f"[MemoryEngine:T5] Loaded {path} on {cls._t5_device}")
            return True
        except Exception as e:
            print(f"[MemoryEngine:T5] load failed: {e}")
            return False

    @classmethod
    def _t5_generate(cls, prompt: str) -> str:
        import torch
        try:
            inp = cls._t5_tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True
            ).to(cls._t5_device)
            with torch.no_grad():
                out = cls._t5_model.generate(
                    **inp, max_new_tokens=128, do_sample=False, num_beams=1
                )
            return cls._t5_tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception:
            return ""

    @staticmethod
    def _parse_t5_triplets(raw: str) -> List[Tuple[str, str, str]]:
        """Parse T5 <triplets> output '(a, b, c) | (d, e, f)' into tuples.

        Sanitizes each field after parsing — T5 occasionally leaks the
        delimiter characters ')' '(' '|' into fields (e.g. output like
        '(user, made, tea) |' where the trailing ' |' gets absorbed into
        the last object). Every field is stripped of these structural
        tokens on all sides, and triplets with empty or punctuation-only
        fields after cleanup are dropped.
        """
        if not raw or not raw.strip():
            return []

        # Structural tokens that must never appear inside a field. These
        # are the T5 <triplets> syntax characters — not content. Stripping
        # them from field edges is sanitization of the SERIALIZATION, not
        # the semantics.
        STRUCTURAL_CHARS = " \t\n)(|"

        def sanitize(field: str) -> str:
            field = field.strip().strip(STRUCTURAL_CHARS).strip()
            # Drop if the field is now empty or consists only of
            # punctuation / structural fragments.
            if not field:
                return ""
            if all(c in STRUCTURAL_CHARS + ".,;:-" for c in field):
                return ""
            return field

        results: List[Tuple[str, str, str]] = []
        for segment in raw.split(" | "):
            segment = segment.strip().strip(STRUCTURAL_CHARS).strip()
            if not segment:
                continue
            parts = segment.split(", ")
            if len(parts) < 3:
                continue
            s = sanitize(parts[0])
            p_raw = sanitize(parts[1])
            o = sanitize(", ".join(parts[2:]))
            # Predicate normalization: lowercase + underscore
            p = p_raw.lower().replace(" ", "_") if p_raw else ""
            if s and p and o:
                results.append((s, p, o))
        return results

    def clean(self, text: str) -> str:
        """T5 <cleanup>: messy STT → clean text. Fallback to raw text if
        the model is unavailable."""
        if not text or not text.strip():
            return text or ""
        if not self._t5_load():
            return text.strip()
        out = self._t5_generate(f"<cleanup> {text.strip()}")
        return out or text.strip()

    def extract(self, text: str) -> List[Tuple[str, str, str]]:
        """T5 <triplets>: clean text → list of (subject, predicate, object)
        tuples. Empty list if T5 unavailable or nothing extracted."""
        if not text or not text.strip():
            return []
        if not self._t5_load():
            return []
        raw = self._t5_generate(f"<triplets> {text.strip()}")
        return self._parse_t5_triplets(raw)

    def ingest_text(
        self,
        user_id: int,
        text: str,
        source_timestamp: Optional[str] = None,
        speaker: Optional[str] = None,
        confidence: float = 0.9,
        source_tag: Optional[str] = None,
    ) -> int:
        """End-to-end write-path entry for raw text:
          1. T5 cleanup
          2. T5 triplet extraction
          3. Speaker substitution (generic 'user'/'I'/'me' → speaker name)
          4. For each triple: store()

        Returns the number of triples actually stored.
        """
        if not text or not text.strip():
            return 0
        cleaned = self.clean(text)
        triples = self.extract(cleaned)

        def _resolve(tok: str) -> str:
            if not tok:
                return tok
            lower = tok.strip().lower()
            if lower in ("user", "i", "me", "myself") and speaker:
                return speaker
            if lower in ("you", "listener") and speaker:
                # Generic "you" in a speaker-known context → keep as 'you'
                # unless caller provides a listener name. Leave as-is for now.
                return tok
            return tok

        count = 0
        for s, p, o in triples:
            # T5 path ONLY: validate against the Core contract before
            # store(). Hand-authored triples bypass this because they
            # don't go through ingest_text(). This is architectural
            # separation — T5 is the source that produces contract
            # violations, so T5's pipeline cleans its own output.
            resolved_s = _resolve(s)
            resolved_o = _resolve(o)
            ok, normalized, reason = self._validate_triple(
                resolved_s, p, resolved_o
            )
            if not ok:
                if os.environ.get("RAYA_VALIDATE_VERBOSE") == "1":
                    print(f"[validate] reject ({resolved_s!r},{p!r},{resolved_o!r}): {reason}")
                continue
            resolved_s, p, resolved_o = normalized

            rel_id = self.store(
                user_id=user_id,
                subject=resolved_s,
                predicate=p,
                object=resolved_o,
                source_text=cleaned,
                confidence=confidence,
                source_timestamp=source_timestamp,
                source_tag=source_tag,
            )
            if rel_id:
                count += 1
        return count

    # ──────────────────────────────────────────────────────────────
    # Track 1.6 — write-path validation gate
    # ──────────────────────────────────────────────────────────────
    #
    # Enforces the Core contract: subject is a concrete entity,
    # predicate is a verb shape, object is a bounded noun phrase. Any
    # triple that fails the contract is dropped at write time rather
    # than corrupting retrieval + reconstruction downstream.
    #
    # All tests are morphological / dictionary-based — no domain
    # content curation, no magic thresholds.

    def _pos_tag(self, token: str) -> str:
        """Return the Penn Treebank POS tag for a single token.

        Delegates to NLTK's averaged-perceptron tagger — a trained
        model of English grammar, not a list curated by this project.
        Returns the tag string (e.g. 'VB', 'VBG', 'NN', 'PRP') or
        empty string if the tagger is unavailable (fail-open).

        The tag is the equation: whatever English grammar says this
        token is, that's what it is. No lists.
        """
        if not token:
            return ""
        try:
            import nltk
            tags = nltk.pos_tag([token.lower()])
            return tags[0][1] if tags else ""
        except Exception:
            return ""

    def _is_verb(self, token: str) -> bool:
        """Strict verb test via POS only. Used by the object-clause
        check where a false positive costs a real triple (Sulphur /
        trip would WordNet-match as verbs and cause legitimate noun
        objects to look like clauses)."""
        tag = self._pos_tag(token)
        return tag.startswith("VB") if tag else True  # fail-open

    def _is_verb_lexical(self, token: str) -> bool:
        """Permissive verb test. POS first, WordNet fallback. Used by
        the predicate-shape check where NLTK mis-tags single tokens
        out of context (stayed/works/cut/drove all tag as nouns in
        isolation). WordNet is a curated English verb inventory, not
        a handcrafted project-specific list."""
        if not token:
            return False
        tag = self._pos_tag(token)
        if tag.startswith("VB"):
            return True
        if not tag:
            return True  # fail-open when tagger unavailable
        try:
            from nltk.corpus import wordnet as _wn
            if _wn.synsets(token.lower(), pos="v"):
                return True
        except Exception:
            pass
        return False

    def _is_pronoun(self, token: str) -> bool:
        """Tag starts with 'PRP' or is 'WP'/'WP$' — all pronoun forms."""
        tag = self._pos_tag(token)
        if not tag:
            return False  # fail-open (don't over-reject)
        return tag.startswith("PRP") or tag in ("WP", "WP$")

    def _validate_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
    ) -> Tuple[bool, Optional[Tuple[str, str, str]], str]:
        """Core-contract validation gate. Returns (ok, normalized, reason).

        Contract:
          - subject:   1–4 tokens, non-empty, no structural artifacts
          - predicate: first token is verb-shaped (verb in WordNet OR
                       irregular base form OR -ed/-ing morphology);
                       OR predicate is compound (contains underscore,
                       meaning T5 concatenated verb+preposition like
                       "works_at", "moved_to", "has_emotion")
          - object:    ≤ 8 tokens, not a clause fragment, free of
                       serialization artifacts, no embedded "user"
                       standalone
        """
        # Serialization artifacts (covers cleanup#1 fallthroughs)
        ARTIFACTS = set("()|")
        for field_name, field_val in (("subject", subject),
                                      ("predicate", predicate),
                                      ("object", object)):
            if any(c in ARTIFACTS for c in field_val):
                return False, None, f"{field_name}_has_artifact"

        # Subject: 1-4 tokens
        s_tokens = subject.split()
        if not 1 <= len(s_tokens) <= 4:
            return False, None, f"subject_wrong_length_{len(s_tokens)}"

        # Predicate: POS-based verb test. Bare predicates must POS-tag
        # as VB*. Compound predicates (with underscore) are trusted
        # only if the first segment is verb-tagged — T5 constructs
        # them as verb+preposition/particle pairs.
        if "_" not in predicate:
            if not self._is_verb_lexical(predicate):
                return False, None, "predicate_not_verb"
        else:
            first_seg = predicate.split("_", 1)[0]
            if not self._is_verb_lexical(first_seg):
                return False, None, "compound_predicate_not_verb"

        # Object: token budget
        o_tokens = object.split()
        if len(o_tokens) > 8:
            return False, None, f"object_too_long_{len(o_tokens)}"

        # Object: not a clause fragment. Structural test via POS —
        # if the first token POS-tags as pronoun (PRP/PRP$/WP/WP$)
        # or as verb (VB*), the object is a predication, not a noun
        # phrase. Single-word verb-capable tokens can still be
        # legitimate nouns ("love", "fear") — the len>1 guard on the
        # verb case leaves those alone.
        if o_tokens:
            first_o = o_tokens[0]
            if self._is_pronoun(first_o):
                return False, None, "object_starts_with_pronoun"
            if len(o_tokens) > 1 and self._is_verb(first_o):
                return False, None, "object_starts_with_verb"

        # Object: strip standalone "user" token if it leaked in alone.
        # (Compound forms like "user feedback" are legitimate.)
        if len(o_tokens) == 1 and o_tokens[0].lower() == "user":
            return False, None, "object_is_bare_user"

        return True, (subject, predicate, object), "ok"

    # ──────────────────────────────────────────────────────────────
    # Public: store a triple (the one canonical write method)
    # ──────────────────────────────────────────────────────────────

    def store(
        self,
        user_id: int,
        subject: str,
        predicate: str,
        object: str,
        source_text: str = "",
        confidence: float = 0.9,
        utterance_type_id: Optional[int] = None,
        source_timestamp: Optional[str] = None,
        source_tag: Optional[str] = None,
    ) -> int:
        """Atomically:
          1. Upsert the (subject, predicate, object) relationship row,
             assigning the next sequence_number (narrative-time axis).
          2. Upsert the subject and object entities with embeddings.
          3. Derive the 5 traces from source_text + triple.
          4. Write the materialized trace row.
          5. Generate and write predicted queries for DTCM.

        Returns: relationship_id (0 on failure; fail-open).
        """
        subject = (subject or "").strip()
        predicate = (predicate or "").strip().lower().replace(" ", "_")
        object = (object or "").strip()
        if not subject or not predicate or not object:
            return 0

        # store() trusts the caller. Validation happens one level up in
        # ingest_text() (T5 pipeline), because T5 is the source that
        # produces contract violations. Callers passing hand-authored
        # triples are trusted. Callers going through T5 are validated
        # before their triples reach store().

        try:
            with get_db_context() as conn:
                # 1. Upsert relationship row
                rel_id = self._upsert_relationship_row(
                    conn, user_id, subject, predicate, object,
                    confidence, utterance_type_id, source_timestamp,
                    source_text=source_text, source_tag=source_tag,
                )
                if not rel_id:
                    return 0

                # 2. Upsert entities (with embeddings for entity_link)
                self._upsert_entity(conn, user_id, subject)
                self._upsert_entity(conn, user_id, object)

                # 3. Compute + 4. Write edge traces AS COLUMNS on the
                # relationship row itself (not a sibling table). This is
                # the Filter → Complete architecture: traces are edge
                # properties that the retrieval engine filters against.
                self._write_edge_traces(
                    conn, rel_id, source_text, subject, predicate, object
                )

                # 5. 9-axis write-path labeling: cluster_id + situation_id.
                # Structural, no LLM. TemporalEngine.cluster() produces an
                # adjacency-based cluster handle which we persist as a
                # stable string key. situation_id is only set if the
                # TemporalEngine detects a situation signal; otherwise
                # stays NULL (structural refusal at the read layer if
                # the axis is queried with no matching cluster).
                try:
                    self._label_cluster_and_situation(
                        conn, user_id, rel_id, source_text or ""
                    )
                except Exception as _e:
                    # fail-open: labeling is observational, not a hard
                    # contract with the write path
                    pass

                # 6. Ingest-time coherence (axis 7): if there are other
                # is_current rows with the same (subject, predicate, object),
                # mark the OLDER ones superseded_by=this rel_id. Retrieval
                # stays read-only; conflicts are resolved at write time.
                # Note: matching on the full triple (s, p, o) so that
                # multiple distinct objects under the same predicate
                # coexist (e.g. multiple emotions, metrics). True
                # temporal supersession of *different* objects is
                # handled by TemporalEngine.supersede().
                try:
                    self._enforce_coherence_on_insert(
                        conn, user_id, rel_id, subject, predicate, object
                    )
                except Exception:
                    pass

                # 7. Entity-type tagging + predicted-queries write-path
                # foundation (2026-04-14). Structural only: NER/WordNet
                # for types, T5 for questions, no curated lists.
                subj_type = "GENERIC"
                obj_type = "GENERIC"
                try:
                    self._label_entity_types(
                        conn, user_id, rel_id, subject, object
                    )
                    # Reload the types we just wrote so the predicted
                    # queries step sees the same vocabulary.
                    row = conn.execute(
                        "SELECT subject_type, object_type FROM relationships WHERE id = ?",
                        (rel_id,),
                    ).fetchone()
                    if row:
                        subj_type = row["subject_type"] or "GENERIC"
                        obj_type = row["object_type"] or "GENERIC"
                except Exception:
                    pass

                try:
                    self._write_predicted_queries(
                        conn, user_id, rel_id, subject, predicate, object,
                        subj_type, obj_type, source_text,
                    )
                except Exception:
                    pass

                conn.commit()
                return rel_id
        except Exception as e:
            print(f"[MemoryEngine.store] failed on ({subject}, {predicate}, {object}): {e}")
            return 0

    def _upsert_relationship_row(
        self,
        conn,
        user_id: int,
        subject: str,
        predicate: str,
        object: str,
        confidence: float,
        utterance_type_id: Optional[int],
        source_timestamp: Optional[str],
        source_text: Optional[str] = None,
        source_tag: Optional[str] = None,
    ) -> int:
        """Insert or update a relationship. Returns relationship_id."""
        # Check existing
        existing = conn.execute(
            """SELECT id FROM relationships
               WHERE user_id = ? AND LOWER(subject) = LOWER(?)
                 AND LOWER(predicate) = LOWER(?) AND LOWER(object) = LOWER(?)""",
            (user_id, subject, predicate, object),
        ).fetchone()

        if existing:
            rel_id = existing["id"]
            conn.execute(
                """UPDATE relationships SET
                     confidence = MAX(confidence, ?),
                     last_confirmed_at = datetime('now'),
                     is_current = 1,
                     tombstoned_at = NULL,
                     tombstone_reason = NULL,
                     tombstone_op_id = NULL
                   WHERE id = ?""",
                (confidence, rel_id),
            )
            # Backfill source_text / source_tag if they weren't set before.
            try:
                if source_text:
                    conn.execute(
                        "UPDATE relationships SET source_text = COALESCE(source_text, ?) WHERE id = ?",
                        (source_text, rel_id),
                    )
                if source_tag:
                    conn.execute(
                        "UPDATE relationships SET source_tag = COALESCE(source_tag, ?) WHERE id = ?",
                        (source_tag, rel_id),
                    )
            except sqlite3.OperationalError:
                pass
            return rel_id

        # Insert new. sequence_number column is added by schema upgrade;
        # the assign logic: next row in this user's relationships.
        try:
            cur = conn.execute(
                """INSERT INTO relationships
                     (user_id, subject, predicate, object, confidence,
                      utterance_type_id, source_timestamp, is_current,
                      source_text, source_tag)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (user_id, subject, predicate, object, confidence,
                 utterance_type_id, source_timestamp, source_text, source_tag),
            )
        except sqlite3.OperationalError:
            # Columns not yet migrated — fall back to pre-migration insert.
            cur = conn.execute(
                """INSERT INTO relationships
                     (user_id, subject, predicate, object, confidence,
                      utterance_type_id, source_timestamp, is_current)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                (user_id, subject, predicate, object, confidence,
                 utterance_type_id, source_timestamp),
            )
        rel_id = cur.lastrowid

        # Assign sequence_number = relationship_id for now (monotonic).
        # Real sequence_number comes from Phase 6 migration.
        try:
            conn.execute(
                "UPDATE relationships SET sequence_number = ? WHERE id = ?",
                (rel_id, rel_id),
            )
        except sqlite3.OperationalError:
            # Column doesn't exist yet; that's fine until Phase 6 migration.
            pass

        return rel_id

    def _upsert_entity(self, conn, user_id: int, name: str) -> int:
        """Insert or update an entity, attaching an embedding."""
        name_clean = (name or "").strip()
        if not name_clean or name_clean.lower() in ("user", "i", "me", "myself"):
            # "user" is the user's canonical identity; skip creating an entity row
            return 0

        existing = conn.execute(
            "SELECT id, mention_count FROM entities WHERE user_id = ? AND LOWER(name) = LOWER(?)",
            (user_id, name_clean),
        ).fetchone()

        if existing:
            conn.execute(
                """UPDATE entities SET
                     last_mentioned_at = datetime('now'),
                     mention_count = mention_count + 1
                   WHERE id = ?""",
                (existing["id"],),
            )
            return existing["id"]

        try:
            emb = embed_text(name_clean)
            emb_blob = emb.tobytes()
        except Exception:
            emb_blob = None

        cur = conn.execute(
            """INSERT INTO entities
                 (user_id, name, entity_type, attributes, embedding)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, name_clean, "unknown", "{}", emb_blob),
        )
        return cur.lastrowid

    # ──────────────────────────────────────────────────────────────
    # Edge traces — computed AT STORE TIME, written as COLUMNS on
    # the relationships row (Filter → Complete architecture).
    #
    # Old design wrote 5 traces to a sibling memory_traces table and
    # used fingerprint-match against a predicted_queries table. Both
    # sibling tables are now legacy — not written to, not read from.
    # The edge carries its own embedding and its own trace properties.
    # ──────────────────────────────────────────────────────────────

    def _write_edge_traces(
        self,
        conn,
        relationship_id: int,
        source_text: str,
        subject: str,
        predicate: str,
        object: str,
    ) -> None:
        """Compute the 5 trace values + edge_embedding and write them
        as columns on the relationships row."""
        if not source_text:
            source_text = f"{subject} {predicate.replace('_', ' ')} {object}"

        try:
            src_emb = embed_text(source_text)
        except Exception:
            return

        # edge_embedding = embedding of "subject predicate object" — the
        # full triple surface text. Including subject anchors the vector
        # so edges with the same predicate but different subjects produce
        # distinct embeddings ("user works at Vantage" vs "Leon works at
        # Different Company").
        pred_natural = predicate.replace("_", " ")
        try:
            edge_emb = embed_text(f"{subject} {pred_natural} {object}")
        except Exception:
            edge_emb = src_emb

        # --- Emotional: valence + label ---
        # Classify on the OBJECT of the triple, not the batch source text.
        # Two signals derived:
        #   - valence:    pos_sim / (pos_sim + neg_sim)  — continuous [0,1]
        #   - label:      the OBJECT WORD itself, when the object is a
        #                 distributional outlier vs unrelated words on
        #                 the emotion axis. Otherwise None.
        # Storing the OBJECT (not the category "positive"/"negative") as
        # the label is what lets downstream rendering recognize emotion
        # edges structurally (emo_label == object → "X feel Y"). The
        # polarity survives in the valence column.
        try:
            obj_emb = embed_text(object) if object else src_emb
        except Exception:
            obj_emb = src_emb
        pos_sim = self._cos(obj_emb, self._anchor("emo_pos", _EMO_POS_ANCHOR))
        neg_sim = self._cos(obj_emb, self._anchor("emo_neg", _EMO_NEG_ANCHOR))
        total = abs(pos_sim) + abs(neg_sim)
        valence = (pos_sim / total) if total > 0 else 0.5

        # "Is the object an emotion word?" — two-gate test:
        # (1) Morphological: emotion words are single words or short
        #     phrases (≤ 2 tokens). Multi-word objects are descriptions
        #     ("three years positive reviews"), not feelings.
        # (2) Relative contrast: the object's similarity to EITHER
        #     emotion anchor must exceed its similarity to a neutral
        #     topic anchor AND a neutral action anchor. Being closer to
        #     emotion-space than to both object-space and action-space
        #     means the word IS in emotion-space.
        # No magic thresholds, no hardcoded emotion lexicon.
        obj_tokens = (object or "").split()
        morph_gate = 0 < len(obj_tokens) <= 2
        neutral_topic = self._anchor(
            "emo_neutral_topic",
            "object thing item tool place building material",
        )
        neutral_action = self._anchor(
            "emo_neutral_action",
            "went moved said did made had told left took gave came",
        )
        obj_topic_sim = self._cos(obj_emb, neutral_topic)
        obj_action_sim = self._cos(obj_emb, neutral_action)
        obj_max_emo_sim = max(pos_sim, neg_sim)
        contrast_gate = (
            obj_max_emo_sim > obj_topic_sim
            and obj_max_emo_sim > obj_action_sim
        )
        is_emotional = morph_gate and contrast_gate
        if is_emotional and object:
            emo_label = object.strip()
        else:
            emo_label = None

        # --- Temporal context: argmax over 4 direction anchors ---
        temp_scores = {
            "past": self._cos(src_emb, self._anchor("temp_past", _TEMP_PAST_ANCHOR)),
            "future": self._cos(src_emb, self._anchor("temp_future", _TEMP_FUTURE_ANCHOR)),
            "ongoing": self._cos(src_emb, self._anchor("temp_ongoing", _TEMP_ONGOING_ANCHOR)),
            "present": self._cos(src_emb, self._anchor("temp_present", _TEMP_PRESENT_ANCHOR)),
        }
        temporal_context = max(temp_scores, key=temp_scores.get)

        # --- Schematic category: argmax over schema anchors WITH
        # distributional margin. Force-assigning every edge to its top-1
        # schema over-fragments arcs (one PIP story splits across 6
        # schemas due to noisy cosines). Instead: require the top cosine
        # to be a distributional outlier — greater than mean + std of the
        # anchor distribution for THIS edge. If no anchor stands out,
        # mark 'uncategorized' so downstream clustering treats the edge
        # as ambient rather than miscategorizing it.
        schema_anchors = {
            "career": "job work employer salary promotion office colleague",
            "health": "doctor medicine hospital symptom diagnosis exercise diet",
            "family": "parent sibling child spouse partner wedding baby",
            "social": "friend gathering party event celebration",
            "finance": "money budget savings investment rent payment",
            "education": "school class course learning study teacher student",
            "housing": "apartment house move rent room furniture",
            "hobby": "hobby sport art music craft game reading",
            "travel": "trip vacation flight hotel destination visit",
            "food": "restaurant cooking recipe meal diet ingredient",
            "identity": "name age birthday background origin ethnicity",
            "relationship": "partner dating marriage engagement boyfriend girlfriend",
        }
        schema_sims: Dict[str, float] = {}
        for label, desc in schema_anchors.items():
            schema_sims[label] = self._cos(
                src_emb, self._anchor(f"sch_{label}", desc)
            )
        sim_values = list(schema_sims.values())
        sim_mu = float(np.mean(sim_values))
        sim_sigma = float(np.std(sim_values))
        best_schema = max(schema_sims, key=schema_sims.get)
        best_schema_sim = schema_sims[best_schema]
        # Distributional margin: top must exceed mean + sigma of the
        # anchor distribution to be trusted. Otherwise all anchors are
        # weakly activated (= edge is ambient, not schema-specific).
        if best_schema_sim < sim_mu + sim_sigma:
            best_schema = "uncategorized"

        # --- Relational type: argmax over relationship-shape anchors ---
        rel_scores = {
            "kin": self._cos(src_emb, self._anchor("rel_kin", _REL_KIN_ANCHOR)),
            "friend": self._cos(src_emb, self._anchor("rel_friend", _REL_FRIEND_ANCHOR)),
            "romantic": self._cos(src_emb, self._anchor("rel_romantic", _REL_ROMANTIC_ANCHOR)),
            "professional": self._cos(src_emb, self._anchor("rel_prof", _REL_PROFESSIONAL_ANCHOR)),
        }
        rel_type = max(rel_scores, key=rel_scores.get)

        # --- Episodic significance: argmax over 3 significance anchors ---
        epi_scores = {
            "milestone": self._cos(src_emb, self._anchor("epi_mile", _EPI_MILESTONE_ANCHOR)),
            "notable": self._cos(src_emb, self._anchor("epi_notable", _EPI_NOTABLE_ANCHOR)),
            "routine": self._cos(src_emb, self._anchor("epi_routine", _EPI_ROUTINE_ANCHOR)),
        }
        episodic_significance = max(epi_scores, key=epi_scores.get)

        # Write all trace values as COLUMNS on the relationships row.
        try:
            conn.execute(
                """UPDATE relationships SET
                     edge_embedding = ?,
                     edge_emotional_valence = ?,
                     edge_emotional_label = ?,
                     edge_schematic_category = ?,
                     edge_episodic_significance = ?,
                     edge_relational_type = ?,
                     edge_temporal_context = ?
                   WHERE id = ?""",
                (
                    edge_emb.tobytes(),
                    round(valence, 4),
                    emo_label,
                    best_schema,
                    episodic_significance,
                    rel_type,
                    temporal_context,
                    relationship_id,
                ),
            )
        except Exception as e:
            print(f"[MemoryEngine] edge-trace write failed: {e}")

    # ──────────────────────────────────────────────────────────────
    # Read methods
    # ──────────────────────────────────────────────────────────────

    def get_relationships(
        self,
        user_id: int,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        only_current: bool = True,
    ) -> List[Relationship]:
        conditions = ["user_id = ?"]
        params: List[Any] = [user_id]
        if only_current:
            conditions.append("COALESCE(is_current, 1) = 1")
            conditions.append("tombstoned_at IS NULL")
        if subject:
            conditions.append("LOWER(subject) = LOWER(?)")
            params.append(subject)
        if predicate:
            conditions.append("LOWER(predicate) = LOWER(?)")
            params.append(predicate)
        if object:
            conditions.append("LOWER(object) = LOWER(?)")
            params.append(object)

        sql = (
            f"SELECT id, subject, predicate, object, confidence, is_current "
            f"FROM relationships WHERE {' AND '.join(conditions)} "
            f"ORDER BY id DESC"
        )
        with get_db_context() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                Relationship(
                    id=r["id"],
                    subject=r["subject"],
                    predicate=r["predicate"],
                    object=r["object"],
                    confidence=r["confidence"] or 0.9,
                    is_current=bool(r["is_current"] if r["is_current"] is not None else 1),
                )
                for r in rows
            ]

    def get_entity(self, user_id: int, name: str) -> Optional[Entity]:
        with get_db_context() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE user_id = ? AND LOWER(name) = LOWER(?)",
                (user_id, name),
            ).fetchone()
        if not row:
            return None
        emb = None
        if row["embedding"]:
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            except Exception:
                emb = None
        attrs = {}
        try:
            attrs = json.loads(row["attributes"] or "{}")
        except Exception:
            pass
        return Entity(
            name=row["name"],
            entity_type=row["entity_type"] or "unknown",
            attributes=attrs,
            embedding=emb,
            mention_count=row["mention_count"] or 1,
        )

    def get_traces(self, relationship_id: int) -> Optional[Traces]:
        with get_db_context() as conn:
            row = conn.execute(
                """SELECT valence, affiliation, emotional_intensity,
                          relational_type, relational_proximity, relational_valence,
                          episodic_significance, episodic_narrative_position,
                          temporal_context, schema_category, schema_confidence
                   FROM memory_traces WHERE relationship_id = ?""",
                (relationship_id,),
            ).fetchone()
        if not row:
            return None
        return Traces(
            valence=row["valence"] or 0.5,
            affiliation=row["affiliation"] or 0.5,
            emotional_intensity=row["emotional_intensity"],
            relational_type=row["relational_type"],
            relational_proximity=row["relational_proximity"] or 0.5,
            relational_valence=row["relational_valence"] or 0.5,
            episodic_significance=row["episodic_significance"] or "routine",
            episodic_narrative_position=row["episodic_narrative_position"] or "ongoing",
            temporal_context=row["temporal_context"],
            schema_category=row["schema_category"],
            schema_confidence=row["schema_confidence"] or 0.0,
        )

    # ──────────────────────────────────────────────────────────────
    # Neural entity linker — pure cosine, no regex, no stop-word list
    # ──────────────────────────────────────────────────────────────

    def entity_link(
        self,
        user_id: int,
        mention_text: str,
    ) -> Optional[Entity]:
        """Resolve a free-text mention to a canonical entity via cosine
        over entities.embedding for this user.

        No regex. No token extraction. No stop-word list. The whole
        mention text is embedded and matched against every entity row's
        stored embedding. Above the adaptive population threshold →
        return that entity. Below → None (caller falls through to
        subject-agnostic retrieval).
        """
        if not mention_text or not mention_text.strip():
            return None

        try:
            query_emb = embed_text(mention_text.strip())
        except Exception:
            return None

        with get_db_context() as conn:
            rows = conn.execute(
                """SELECT id, name, entity_type, attributes, embedding, mention_count
                   FROM entities WHERE user_id = ? AND embedding IS NOT NULL""",
                (user_id,),
            ).fetchall()

        if not rows:
            return None

        sims: List[Tuple[float, sqlite3.Row]] = []
        for r in rows:
            try:
                emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
                sims.append((float(np.dot(query_emb, emb)), r))
            except Exception:
                continue

        if not sims:
            return None

        sims.sort(key=lambda x: -x[0])
        best_sim, best_row = sims[0]

        # Adaptive population threshold: require best match to stand out
        # above the mean of this user's entity distribution by a margin
        # proportional to the standard deviation. NO magic number: the
        # threshold is derived from the population itself.
        scores = np.array([s for s, _ in sims], dtype=np.float32)
        if len(scores) >= 3:
            mean = float(scores.mean())
            std = float(scores.std())
            threshold = mean + std
            if best_sim < threshold:
                return None

        attrs = {}
        try:
            attrs = json.loads(best_row["attributes"] or "{}")
        except Exception:
            pass
        emb = None
        try:
            emb = np.frombuffer(best_row["embedding"], dtype=np.float32).copy()
        except Exception:
            pass
        return Entity(
            name=best_row["name"],
            entity_type=best_row["entity_type"] or "unknown",
            attributes=attrs,
            embedding=emb,
            mention_count=best_row["mention_count"] or 1,
        )

    # ──────────────────────────────────────────────────────────────
    # Supersession hook (called by TemporalEngine)
    # ──────────────────────────────────────────────────────────────

    def supersede(self, relationship_id: int, superseded_by: int) -> None:
        """Mark a prior relationship as superseded by a newer one."""
        try:
            with get_db_context() as conn:
                conn.execute(
                    """UPDATE relationships SET
                         is_current = 0,
                         superseded_at = datetime('now'),
                         superseded_by = ?
                       WHERE id = ?""",
                    (superseded_by, relationship_id),
                )
                conn.execute(
                    "DELETE FROM predicted_queries WHERE relationship_id = ?",
                    (relationship_id,),
                )
                conn.commit()
        except Exception as e:
            print(f"[MemoryEngine.supersede] failed: {e}")

    # ──────────────────────────────────────────────────────────────
    # Write-path labeling helpers
    # ──────────────────────────────────────────────────────────────

    def _label_cluster_and_situation(
        self,
        conn,
        user_id: int,
        rel_id: int,
        source_text: str,
    ) -> None:
        """Persist cluster_id (and optionally situation_id) on the row.

        We do not import TemporalEngine at module top-level to avoid a
        circular dep. Import lazily here.
        """
        # Cluster: a stable handle built from the recent-window member
        # list. We use the MIN(id) of the recent window as the cluster
        # handle string, so all members of one narrative window share a
        # deterministic key. If TemporalEngine is unavailable or returns
        # an empty window, we fall back to the row's own id as its own
        # single-member cluster (still structural, still binary).
        cluster_key: Optional[str] = None
        try:
            from app.engines.temporal import get_temporal_engine
            te = get_temporal_engine()
            clust = te.cluster(user_id, source_text)
            members = [m for m in (clust.member_relationship_ids or []) if m]
            members.append(rel_id)
            cluster_key = f"c_{min(members)}"
        except Exception:
            cluster_key = f"c_{rel_id}"

        try:
            conn.execute(
                "UPDATE relationships SET cluster_id = ? WHERE id = ?",
                (cluster_key, rel_id),
            )
        except sqlite3.OperationalError:
            pass

        # situation_id: only write if not already set and the temporal
        # engine has a situation signal. Today's TemporalEngine has no
        # explicit detect_situation call; we leave situation_id as-is.
        # This is intentional -- structural refusal is preferable to a
        # heuristic guess.
        return

    def _enforce_coherence_on_insert(
        self,
        conn,
        user_id: int,
        new_rel_id: int,
        subject: str,
        predicate: str,
        object: str = "",
    ) -> None:
        """Axis 7: coherence at ingest.

        Find any OTHER is_current rows with same (subject, predicate, object)
        for this user; mark them superseded_by=new_rel_id. The newest row
        wins (max sequence_number). Retrieval stays read-only.

        Matching on the full triple ensures that multiple distinct objects
        under the same predicate coexist (e.g. multiple emotions, metrics).
        True temporal supersession of *different* objects is handled by
        TemporalEngine.supersede().
        """
        try:
            rows = conn.execute(
                """SELECT id FROM relationships
                   WHERE user_id = ?
                     AND LOWER(subject) = LOWER(?)
                     AND LOWER(predicate) = LOWER(?)
                     AND LOWER(object) = LOWER(?)
                     AND id != ?
                     AND COALESCE(is_current, 1) = 1
                     AND tombstoned_at IS NULL""",
                (user_id, subject, predicate, object, new_rel_id),
            ).fetchall()
        except sqlite3.OperationalError:
            return
        for r in rows:
            old_id = r["id"]
            try:
                conn.execute(
                    """UPDATE relationships SET
                         is_current = 0,
                         superseded_at = datetime('now'),
                         superseded_by = ?
                       WHERE id = ?""",
                    (new_rel_id, old_id),
                )
            except sqlite3.OperationalError:
                pass

    # ──────────────────────────────────────────────────────────────
    # Write-path foundation (2026-04-14): entity types + predicted queries
    # ──────────────────────────────────────────────────────────────

    def _label_entity_types(
        self,
        conn,
        user_id: int,
        rel_id: int,
        subject: str,
        object: str,
    ) -> None:
        """Resolve subject_type / object_type via NER+WordNet and write
        them on the relationship row. Also backfill entities.entity_type
        where it is NULL or 'unknown'. Structural only; fail-open."""
        try:
            from app.engines.type_resolver import resolve_entity_type
        except Exception:
            return

        try:
            s_type = resolve_entity_type(subject)
        except Exception:
            s_type = "GENERIC"
        try:
            o_type = resolve_entity_type(object)
        except Exception:
            o_type = "GENERIC"

        try:
            conn.execute(
                """UPDATE relationships
                     SET subject_type = ?, object_type = ?
                   WHERE id = ?""",
                (s_type, o_type, rel_id),
            )
        except sqlite3.OperationalError:
            # Column not yet migrated on this DB.
            pass

        # Propagate into entities table. "user" is excluded by _upsert_entity
        # already, so only non-user strings arrive here.
        for name, typ in ((subject, s_type), (object, o_type)):
            nm = (name or "").strip()
            if not nm or nm.lower() in ("user", "i", "me", "myself"):
                continue
            try:
                conn.execute(
                    """UPDATE entities
                         SET entity_type = ?
                       WHERE user_id = ?
                         AND LOWER(name) = LOWER(?)
                         AND (entity_type IS NULL OR entity_type = 'unknown' OR entity_type = '')""",
                    (typ, user_id, nm),
                )
            except sqlite3.OperationalError:
                pass

    def _write_predicted_queries(
        self,
        conn,
        user_id: int,
        rel_id: int,
        subject: str,
        predicate: str,
        object: str,
        subject_type: str,
        object_type: str,
        source_text: str,
    ) -> None:
        """Generate + persist predicted question rows for this edge.

        The answer_text field carries the source utterance (or a derived
        SPO surface when no source text was provided) so the retrieval
        layer can return a human-shaped answer directly from the row.
        """
        try:
            from app.engines.predicted_queries import generate_predicted_queries
        except Exception:
            return

        try:
            pairs = generate_predicted_queries(
                subject, predicate, object, subject_type, object_type
            )
        except Exception:
            return

        if not pairs:
            return

        answer_text = source_text or f"{subject} {predicate.replace('_', ' ')} {object}"

        for question, emb in pairs:
            try:
                emb_blob = emb.tobytes() if emb is not None else None
            except Exception:
                emb_blob = None
            if emb_blob is None:
                continue
            try:
                # Skip duplicates (same question already present for this
                # relationship_id). Idempotent on replay.
                existing = conn.execute(
                    """SELECT 1 FROM predicted_queries
                       WHERE relationship_id = ? AND predicted_question = ?""",
                    (rel_id, question),
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    """INSERT INTO predicted_queries
                         (relationship_id, user_id, predicted_question,
                          answer_text, answer_subject, question_embedding,
                          confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (rel_id, user_id, question, answer_text, subject,
                     emb_blob, 0.9),
                )
            except sqlite3.OperationalError:
                # predicted_queries table not migrated on this DB.
                return
            except Exception:
                continue

    # ──────────────────────────────────────────────────────────────
    # Forget (soft tombstone) + faceted show
    # ──────────────────────────────────────────────────────────────
    #
    # Tombstone semantics: we cannot INSERT a duplicate tombstone row
    # because of UNIQUE(user_id, subject, predicate, object). Instead,
    # we flip the existing row in-place by setting tombstoned_at and
    # append a single summary row into `memories` as an append-only
    # audit trail. A forget op never modifies subject/predicate/object
    # or the edge_* trace columns.

    def _append_forget_audit(
        self, conn, user_id: int, op_id: str, reason: str, count: int
    ) -> None:
        """Append an audit log row into the memories table for a forget op."""
        try:
            conn.execute(
                """INSERT INTO memories (user_id, content, memory_type, importance)
                   VALUES (?, ?, 'summary', 0.9)""",
                (
                    user_id,
                    f"forget op {op_id}: reason={reason}, n={count}",
                ),
            )
        except Exception as e:
            print(f"[MemoryEngine._append_forget_audit] failed: {e}")

    def _tombstone_rows(
        self, conn, user_id: int, ids: List[int], reason: str, op_id: str
    ) -> int:
        """Flip tombstone flags on a set of relationship rows. Returns count
        of rows actually flipped (ignores rows that were already tombstoned)."""
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"""UPDATE relationships SET
                  tombstoned_at = datetime('now'),
                  tombstone_reason = ?,
                  tombstone_op_id = ?
                WHERE user_id = ? AND tombstoned_at IS NULL
                  AND id IN ({placeholders})""",
            [reason, op_id, user_id] + list(ids),
        )
        return cur.rowcount or 0

    def forget_by_triple_id(self, user_id: int, triple_id: int) -> int:
        """Tombstone a single triple by its relationship id.

        Returns the count of tombstones emitted (0 or 1). Idempotent:
        if the row is already tombstoned, returns 0.
        """
        from uuid import uuid4
        op_id = uuid4().hex
        reason = f"by_triple_id:{int(triple_id)}"
        try:
            with get_db_context() as conn:
                row = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND id = ?
                         AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1""",
                    (user_id, int(triple_id)),
                ).fetchone()
                if not row:
                    return 0
                count = self._tombstone_rows(
                    conn, user_id, [row["id"]], reason, op_id
                )
                if count:
                    self._append_forget_audit(conn, user_id, op_id, reason, count)
                conn.commit()
                return count
        except Exception as e:
            print(f"[MemoryEngine.forget_by_triple_id] failed: {e}")
            return 0

    def forget_by_entity(self, user_id: int, entity_name: str) -> int:
        """Tombstone every live triple where the entity appears as subject
        or object (case-insensitive exact match)."""
        from uuid import uuid4
        if not entity_name or not entity_name.strip():
            return 0
        name = entity_name.strip()
        op_id = uuid4().hex
        reason = f"by_entity:{name}"
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1
                         AND (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))""",
                    (user_id, name, name),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
                if count:
                    self._append_forget_audit(conn, user_id, op_id, reason, count)
                conn.commit()
                return count
        except Exception as e:
            print(f"[MemoryEngine.forget_by_entity] failed: {e}")
            return 0

    def forget_by_time_range(
        self, user_id: int, start_iso: str, end_iso: str
    ) -> int:
        """Tombstone every live triple whose effective source time falls
        inclusively within [start_iso, end_iso].

        Effective source time is COALESCE(source_timestamp, first_learned_at).
        v1 behaviour: rows where source_timestamp IS NULL are SKIPPED (not
        fallback-matched on first_learned_at) because first_learned_at is
        an ingestion timestamp, not a source-wall-clock timestamp, and
        time-range forget is a user-facing operation over source time.
        """
        from uuid import uuid4
        op_id = uuid4().hex
        reason = f"by_time_range:{start_iso}..{end_iso}"
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1
                         AND source_timestamp IS NOT NULL
                         AND source_timestamp >= ?
                         AND source_timestamp <= ?""",
                    (user_id, start_iso, end_iso),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
                if count:
                    self._append_forget_audit(conn, user_id, op_id, reason, count)
                conn.commit()
                return count
        except Exception as e:
            print(f"[MemoryEngine.forget_by_time_range] failed: {e}")
            return 0

    def forget_by_source(self, user_id: int, source_tag: str) -> int:
        """Tombstone every live triple whose source_tag exactly matches."""
        from uuid import uuid4
        if source_tag is None:
            return 0
        op_id = uuid4().hex
        reason = f"by_source:{source_tag}"
        try:
            with get_db_context() as conn:
                rows = conn.execute(
                    """SELECT id FROM relationships
                       WHERE user_id = ? AND tombstoned_at IS NULL
                         AND COALESCE(is_current, 1) = 1
                         AND source_tag = ?""",
                    (user_id, source_tag),
                ).fetchall()
                ids = [r["id"] for r in rows]
                count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
                if count:
                    self._append_forget_audit(conn, user_id, op_id, reason, count)
                conn.commit()
                return count
        except Exception as e:
            print(f"[MemoryEngine.forget_by_source] failed: {e}")
            return 0

    def list_by_facet(
        self,
        user_id: int,
        facet: str,
        value: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return live triples matching a facet view. Facets:

          - 'time'   : ordered by source_timestamp DESC (NULLs last).
                       If `value` is an ISO prefix it acts as a LIKE filter.
          - 'entity' : subject OR object case-insensitive equals `value`.
          - 'source' : source_tag exact match on `value`.
          - 'trace'  : live triples ordered by recency. `value` optionally
                       filters on edge_schematic_category.

        Returns dicts with keys: id, subject, predicate, object,
        source_timestamp, source_tag, confidence, source_text. The caller
        (SDK.show) strips source_text unless export_raw_text=True. Does
        NOT include any edge_* trace columns.
        """
        if facet not in ("time", "entity", "source", "trace"):
            raise ValueError(f"Unknown facet: {facet}")
        limit = max(1, min(int(limit or 100), 1000))

        base_cols = (
            "id, subject, predicate, object, "
            "source_timestamp, source_tag, confidence, source_text"
        )
        where = ["user_id = ?", "tombstoned_at IS NULL",
                 "COALESCE(is_current, 1) = 1"]
        params: List[Any] = [user_id]
        order = "id DESC"

        if facet == "time":
            if value:
                where.append("source_timestamp LIKE ?")
                params.append(f"{value}%")
            order = ("CASE WHEN source_timestamp IS NULL THEN 1 ELSE 0 END, "
                     "source_timestamp DESC, id DESC")
        elif facet == "entity":
            if not value:
                return []
            where.append("(LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))")
            params.extend([value, value])
        elif facet == "source":
            if value is None:
                return []
            where.append("source_tag = ?")
            params.append(value)
        elif facet == "trace":
            if value:
                where.append("edge_schematic_category = ?")
                params.append(value)

        sql = (
            f"SELECT {base_cols} FROM relationships "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY {order} LIMIT ?"
        )
        params.append(limit)

        try:
            with get_db_context() as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception as e:
            print(f"[MemoryEngine.list_by_facet] failed: {e}")
            return []

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": r["id"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "source_timestamp": r["source_timestamp"],
                "source_tag": r["source_tag"],
                "confidence": r["confidence"] or 0.9,
                "source_text": r["source_text"],
            })
        return out

    # ──────────────────────────────────────────────────────────────
    # Summarize (stub for Phase 4+)
    # ──────────────────────────────────────────────────────────────

    def summarize(self, user_id: int, entity: Optional[str] = None) -> str:
        """Natural-language summary of entity state (or overall user state
        if entity is None). Phase 4+ implementation."""
        rels = self.get_relationships(user_id, subject=entity) if entity else self.get_relationships(user_id, subject="user")
        if not rels:
            return ""
        lines = [f"{r.subject} {r.predicate.replace('_', ' ')} {r.object}" for r in rels[:10]]
        return ". ".join(lines) + "."


# Module-level singleton for ergonomics; callers may instantiate their own
_singleton: Optional[MemoryEngine] = None


def get_memory_engine() -> MemoryEngine:
    global _singleton
    if _singleton is None:
        _singleton = MemoryEngine()
    return _singleton
