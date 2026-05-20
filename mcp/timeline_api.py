"""
Timeline API — transforms Reconstruct edges into LifeTimeline JSON.

Returns the exact shape the TimelineCanvas component consumes:

    LifeTimeline
      └─ DomainBranch[]  (career, health, emotional, relationship, spatial)
           └─ TimelineThread[]  (topic clusters within a domain)
                └─ ThreadSession[]  (individual facts with host + timestamp)
                     └─ BranchEvent[]  (the actual edge data)

One SQL query, pure transformation, no LLM.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional

from mcp.tools import DB_PATH

# Schema category -> timeline domain mapping
_SCHEMA_TO_DOMAIN = {
    # career
    "career": "career",
    "education": "career",
    "professional": "career",
    "work": "career",
    "financial": "career",
    "achievement": "career",
    # health
    "health": "health",
    "medical": "health",
    "fitness": "health",
    "body": "health",
    "hobby": "health",
    "sport": "health",
    "recreation": "health",
    # emotional
    "emotional": "emotional",
    "mood": "emotional",
    "mental": "emotional",
    "cognitive": "emotional",
    "uncategorized": "emotional",
    # relationship
    "relationship": "relationship",
    "family": "relationship",
    "social": "relationship",
    "personal": "relationship",
    "communication": "relationship",
    # spatial
    "spatial": "spatial",
    "location": "spatial",
    "travel": "spatial",
    "logistics": "spatial",
    "possession": "spatial",
    "consumption": "spatial",
}

# source_tag -> ModelHost
_TAG_TO_HOST = {
    "llm:claude": "claude",
    "llm:gpt": "chatgpt",
    "llm:chatgpt": "chatgpt",
    "llm:gemini": "gemini",
    "user": "claude",  # default: user input shown as claude-colored
}

_VALID_DOMAINS = {"career", "health", "emotional", "relationship", "spatial"}


def _map_domain(schema_cat: str) -> str:
    """Map edge_schematic_category to one of 5 timeline domains."""
    if not schema_cat:
        return "emotional"
    lower = schema_cat.lower().strip()
    return _SCHEMA_TO_DOMAIN.get(lower, "emotional")


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
