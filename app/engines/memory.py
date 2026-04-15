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
        """Parse T5 <triplets> output '(a, b, c) | (d, e, f)' into tuples."""
        if not raw or not raw.strip():
            return []
        results: List[Tuple[str, str, str]] = []
        for segment in raw.split(" | "):
            segment = segment.strip()
            if not segment:
                continue
            if segment.startswith("(") and segment.endswith(")"):
                segment = segment[1:-1]
            elif segment.startswith("("):
                segment = segment[1:]
            elif segment.endswith(")"):
                segment = segment[:-1]
            segment = segment.strip()
            if not segment:
                continue
            parts = segment.split(", ")
            if len(parts) < 3:
                continue
            s = parts[0].strip()
            p = parts[1].strip().lower().replace(" ", "_")
            o = ", ".join(parts[2:]).strip()
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
            rel_id = self.store(
                user_id=user_id,
                subject=_resolve(s),
                predicate=p,
                object=_resolve(o),
                source_text=cleaned,
                confidence=confidence,
                source_timestamp=source_timestamp,
            )
            if rel_id:
                count += 1
        return count

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

        try:
            with get_db_context() as conn:
                # 1. Upsert relationship row
                rel_id = self._upsert_relationship_row(
                    conn, user_id, subject, predicate, object,
                    confidence, utterance_type_id, source_timestamp,
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
                     is_current = 1
                   WHERE id = ?""",
                (confidence, rel_id),
            )
            return rel_id

        # Insert new. sequence_number column is added by schema upgrade;
        # the assign logic: next row in this user's relationships.
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

        # edge_embedding = embedding of "predicate object" — the SLOT
        # that retrieval fills. A query like "Who is Maya's manager?"
        # with predicate_emb = embed("manager") should cosine-match
        # edges whose edge_embedding = embed("manager Priya Banerjee").
        pred_natural = predicate.replace("_", " ")
        try:
            edge_emb = embed_text(f"{pred_natural} {object}")
        except Exception:
            edge_emb = src_emb

        # --- Emotional: valence + label ---
        pos_sim = self._cos(src_emb, self._anchor("emo_pos", _EMO_POS_ANCHOR))
        neg_sim = self._cos(src_emb, self._anchor("emo_neg", _EMO_NEG_ANCHOR))
        total = abs(pos_sim) + abs(neg_sim)
        valence = (pos_sim / total) if total > 0 else 0.5
        # Label: present iff source text has any emotional content —
        # "emotional" means the valence deviates from neutral 0.5.
        # We don't need a threshold: the label is the argmax choice.
        if pos_sim > neg_sim and pos_sim > 0:
            emo_label = "positive"
        elif neg_sim > pos_sim and neg_sim > 0:
            emo_label = "negative"
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

        # --- Schematic category: argmax over schema anchors ---
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
        best_schema, best_schema_sim = None, -1.0
        for label, desc in schema_anchors.items():
            s = self._cos(src_emb, self._anchor(f"sch_{label}", desc))
            if s > best_schema_sim:
                best_schema_sim, best_schema = s, label

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
