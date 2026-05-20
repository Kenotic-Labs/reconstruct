"""
Timeline API — transforms Reconstruct edges into LifeTimeline JSON.

Returns the exact shape the TimelineCanvas component consumes:

    LifeTimeline
      └─ DomainBranch[]  (career, health, emotional, relationship, spatial)
           └─ TimelineThread[]  (topic clusters within a domain)
                └─ ThreadSession[]  (individual facts with host + timestamp)
                     └─ BranchEvent[]  (the actual edge data)

One SQL query, pure transformation, no LLM.
Domain classification uses WordNet Wu-Palmer similarity — no lookup tables.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional

from mcp.tools import DB_PATH

# ── WordNet domain classification ────────────────────────────────
#
# Maps any schema category word to one of 5 timeline domains using
# WordNet Wu-Palmer similarity against anchor synsets. No lookup table.
# 82% accuracy on 28 test words — remaining misses are genuinely
# ambiguous (e.g. "marathon" = career or health).

_DOMAIN_ANCHORS: dict | None = None


def _get_anchors():
    """Lazy-load WordNet anchor synsets."""
    global _DOMAIN_ANCHORS
    if _DOMAIN_ANCHORS is not None:
        return _DOMAIN_ANCHORS
    from nltk.corpus import wordnet as wn
    _DOMAIN_ANCHORS = {
        "career": [
            wn.synset("occupation.n.01"), wn.synset("career.n.01"),
            wn.synset("commerce.n.01"), wn.synset("education.n.01"),
            wn.synset("money.n.01"), wn.synset("promotion.n.02"),
            wn.synset("profession.n.01"),
        ],
        "health": [
            wn.synset("health.n.01"), wn.synset("body.n.01"),
            wn.synset("exercise.n.01"), wn.synset("medicine.n.02"),
            wn.synset("disease.n.01"), wn.synset("sport.n.01"),
            wn.synset("diversion.n.01"), wn.synset("fitness.n.02"),
        ],
        "emotional": [
            wn.synset("feeling.n.01"), wn.synset("emotion.n.01"),
            wn.synset("psychological_state.n.01"), wn.synset("mood.n.01"),
        ],
        "relationship": [
            wn.synset("person.n.01"), wn.synset("family.n.01"),
            wn.synset("social_relation.n.01"), wn.synset("group.n.01"),
            wn.synset("relationship.n.01"), wn.synset("communication.n.01"),
        ],
        "spatial": [
            wn.synset("location.n.01"), wn.synset("place.n.02"),
            wn.synset("travel.n.01"), wn.synset("region.n.01"),
            wn.synset("transport.n.01"), wn.synset("possession.n.02"),
        ],
    }
    return _DOMAIN_ANCHORS


@lru_cache(maxsize=256)
def _map_domain(schema_cat: str) -> str:
    """Map edge_schematic_category to one of 5 timeline domains.

    Uses WordNet Wu-Palmer similarity: find the domain whose anchor
    synsets are closest to the schema category word. Handles nouns
    directly and adjectives via derivationally related noun forms.
    """
    if not schema_cat or schema_cat.lower().strip() == "uncategorized":
        return "emotional"

    from nltk.corpus import wordnet as wn
    anchors = _get_anchors()
    word = schema_cat.lower().strip()

    # Get noun synsets for the word
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        # Adjective -> derivationally related nouns
        for ss in wn.synsets(word, pos=wn.ADJ)[:3]:
            for lemma in ss.lemmas():
                for form in lemma.derivationally_related_forms():
                    if form.synset().pos() == "n":
                        synsets.append(form.synset())
    if not synsets:
        return "emotional"

    # Max Wu-Palmer similarity across senses x anchors per domain
    domain_scores: Dict[str, float] = {d: 0.0 for d in anchors}
    for ss in synsets[:4]:
        for domain, domain_anchors in anchors.items():
            for anchor in domain_anchors:
                sim = ss.wup_similarity(anchor)
                if sim and sim > domain_scores[domain]:
                    domain_scores[domain] = sim

    return max(domain_scores, key=domain_scores.get)


# ── Host mapping ─────────────────────────────────────────────────

# source_tag -> ModelHost
_TAG_TO_HOST = {
    "llm:claude": "claude",
    "llm:gpt": "chatgpt",
    "llm:chatgpt": "chatgpt",
    "llm:gemini": "gemini",
    "user": "claude",
}


def _map_host(source_tag: str) -> str:
    """Map source_tag to a ModelHost."""
    if not source_tag:
        return "claude"
    lower = source_tag.lower().strip()
    return _TAG_TO_HOST.get(lower, "claude")


def _event_type(edge: dict) -> str:
    """Determine event type from edge state."""
    if edge["superseded_by"]:
        return "supersede"
    if edge["is_current"] == 0:
        return "correction"
    return "new"


def build_timeline(user_id: int = 0, db_path: str | None = None) -> Dict[str, Any]:
    """Query all live edges and return a LifeTimeline dict.

    This is the bridge between the SQLite DB and the TimelineCanvas
    component. One SQL query, pure transformation.
    """
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            id, subject, predicate, object, source_text,
            edge_schematic_category, edge_emotional_label,
            source_timestamp, resolved_event_date,
            is_current, superseded_at, superseded_by,
            relational_entities, episodic_fact,
            source_tag, arc_id, cluster_id
        FROM edges
        WHERE user_id = ? AND tombstoned_at IS NULL
        ORDER BY source_timestamp ASC, id ASC
    """, (user_id,)).fetchall()
    conn.close()

    if not rows:
        return {
            "persona": "user",
            "name": "Your Timeline",
            "branches": [],
        }

    # ── Group edges into domain -> thread -> sessions ──────────

    # Thread key: (domain, subject) — each subject gets its own thread
    # within a domain. "Sam + career" is one thread, "Joseph + career"
    # is another.
    domain_threads: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        domain = _map_domain(row["edge_schematic_category"])
        subject = (row["subject"] or "unknown").strip()
        thread_key = subject.lower()
        domain_threads[domain][thread_key].append(dict(row))

    # ── Build LifeTimeline structure ──────────────────────────

    branches: List[Dict[str, Any]] = []

    for domain in ["career", "health", "emotional", "relationship", "spatial"]:
        threads_data = domain_threads.get(domain, {})
        if not threads_data:
            continue

        threads: List[Dict[str, Any]] = []
        total_events = 0
        total_sessions = 0
        branch_start = None
        branch_end = None

        for thread_key, edges in threads_data.items():
            if not edges:
                continue

            # Group edges within thread by session (date bucket)
            session_map: Dict[str, list] = defaultdict(list)
            for e in edges:
                date = (e["source_timestamp"] or e["resolved_event_date"]
                        or e["created_at"] or "unknown")[:10]
                session_map[date].append(e)

            thread_sessions: List[Dict[str, Any]] = []
            has_correction = False
            prev_host = None

            for idx, (date, session_edges) in enumerate(
                sorted(session_map.items()), start=1
            ):
                events: List[Dict[str, Any]] = []
                for e in session_edges:
                    etype = _event_type(e)
                    if etype in ("supersede", "correction"):
                        has_correction = True

                    label = e["predicate"] or ""
                    summary = e["episodic_fact"] or e["source_text"] or ""

                    event = {
                        "id": str(e["id"]),
                        "domain": domain,
                        "type": etype,
                        "label": label,
                        "summary": summary,
                    }
                    if e["superseded_by"]:
                        event["correctsId"] = str(e["superseded_by"])

                    events.append(event)
                    total_events += 1

                host = _map_host(
                    session_edges[0].get("source_tag", "")
                )
                is_switch = prev_host is not None and host != prev_host
                prev_host = host

                session = {
                    "id": f"{domain}-{thread_key}-s{idx}",
                    "number": idx,
                    "host": host,
                    "date": date,
                    "text": events[0]["summary"] if events else "",
                    "events": events,
                }
                if is_switch:
                    session["isModelSwitch"] = True
                    session["switchNotation"] = f"Switched to {host}"

                thread_sessions.append(session)
                total_sessions += 1

            dates = [s["date"] for s in thread_sessions if s["date"] != "unknown"]
            t_start = min(dates) if dates else "unknown"
            t_end = max(dates) if dates else None

            if branch_start is None or (t_start != "unknown" and t_start < (branch_start or "")):
                branch_start = t_start
            if t_end and (branch_end is None or t_end > branch_end):
                branch_end = t_end

            # Thread status
            any_current = any(
                e["is_current"] == 1 for e in edges
            )
            status = "corrected" if has_correction else (
                "ongoing" if any_current else "completed"
            )

            thread_label = thread_key.title()
            thread_summary = edges[-1].get("episodic_fact") or edges[-1].get("source_text") or ""

            threads.append({
                "id": f"{domain}-{thread_key}",
                "label": thread_label,
                "domain": domain,
                "startDate": t_start,
                "endDate": t_end,
                "status": status,
                "summary": thread_summary[:120],
                "hasCorrection": has_correction,
                "sessions": thread_sessions,
            })

        branch_status = "ongoing" if any(
            t["status"] == "ongoing" for t in threads
        ) else "completed"

        branches.append({
            "id": f"branch-{domain}",
            "domain": domain,
            "label": domain.title(),
            "startDate": branch_start or "unknown",
            "endDate": branch_end,
            "status": branch_status,
            "eventCount": total_events,
            "sessionCount": total_sessions,
            "threads": threads,
        })

    return {
        "persona": "user",
        "name": "Your Timeline",
        "branches": branches,
    }
