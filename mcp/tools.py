"""
Reconstruct MCP shared tool registry + JSON-RPC dispatch.

This module is transport-agnostic. Both the stdio server (mcp.server)
and the HTTP/Streamable-HTTP server (mcp.http_server) import from here.

Design:
  - Tool handlers live here.
  - `dispatch(method, params)` is a pure function mapping JSON-RPC
    method names to results. It raises ToolError for protocol-level
    failures and returns None for notifications (callers are expected
    to have already identified notifications by absence of `id`, but
    dispatch will still produce a result dict for any valid method).

Async ingest architecture
-------------------------
reconstruct.ingest queues work and returns immediately. A single
background worker thread drains the queue and performs spaCy extraction
+ SQLite writes serially. Serial writes are correct for SQLite — only
one writer is ever active at a time, even in WAL mode.

Flush-before-read guarantee
---------------------------
reconstruct and reconstruct.benchmark call _flush_for_user() before
executing. _flush_for_user() collects every pending job Event for the
target user and blocks until each one signals done. This is
deterministic — no timeouts, no magic numbers. Queries always see the
results of all prior ingests for the same user.

The queue, worker, and per-job Event objects are module-level singletons,
shared by both the stdio and HTTP transports (one process = one queue).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


log = logging.getLogger("reconstruct.tools")


# Default DB location — override via env
DB_PATH = os.environ.get(
    "KENOTIC_DB_PATH",
    os.path.expanduser("~/.kenotic/memory.db"),
)


# Single-user mode: the Bridge serves one person per install.
# user_id is hardcoded to 0 server-side and is NOT accepted from tool args.
# This prevents spoofing once the server is exposed via Tailscale Funnel.
_SOLO_USER_ID = 0


# MCP protocol version this server targets
PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "reconstruct"
SERVER_VERSION = "0.1.0"


# ── Async ingest queue ───────────────────────────────────────────
#
# Architecture:
#   _ingest_queue   — unbounded FIFO of _IngestJob dataclass instances.
#   _ingest_worker  — single daemon thread that drains the queue in order.
#                     Single-threaded → SQLite writes are serialized;
#                     no "database is locked" races, no WAL contention.
#   _pending_lock   — guards _pending_by_user during enqueue + flush.
#   _pending_by_user — maps user_id → list of threading.Event, one per
#                      job not yet signalled done.  Allows flush to wait
#                      only on jobs belonging to the requesting user.

class _IngestJob:
    """One queued ingest call. done is set() when the worker finishes."""

    __slots__ = ("job_id", "user_id", "args", "done", "error")

    def __init__(self, job_id: str, user_id: int, args: Dict[str, Any]):
        self.job_id: str = job_id
        self.user_id: int = user_id
        self.args: Dict[str, Any] = args
        self.done: threading.Event = threading.Event()
        self.error: Optional[Exception] = None


_ingest_queue: queue.Queue[_IngestJob] = queue.Queue()
_pending_lock: threading.Lock = threading.Lock()
_pending_by_user: Dict[int, List[threading.Event]] = {}


def _register_pending(job: _IngestJob) -> None:
    """Add job.done to the per-user pending list (called at enqueue time)."""
    with _pending_lock:
        _pending_by_user.setdefault(job.user_id, []).append(job.done)


def _deregister_pending(job: _IngestJob) -> None:
    """Remove job.done from the per-user pending list (called when done)."""
    with _pending_lock:
        lst = _pending_by_user.get(job.user_id)
        if lst:
            try:
                lst.remove(job.done)
            except ValueError:
                pass
            if not lst:
                _pending_by_user.pop(job.user_id, None)


def _flush_for_user(user_id: int) -> None:
    """Block until every pending ingest for user_id has completed.

    Called by retrieve / reconstruct / show before executing — guarantees
    that a retrieve immediately after an ingest always sees the new data.
    The caller cannot observe this wait; it is internal latency only.
    """
    with _pending_lock:
        # Snapshot the list so the worker can deregister freely
        events: List[threading.Event] = list(_pending_by_user.get(user_id, []))
    for ev in events:
        ev.wait()  # blocks until the worker sets it; no timeout, no polling


def _run_ingest_job(job: _IngestJob) -> None:
    """Execute one ingest job. Called by the single background worker."""
    try:
        from sdk import Kenotic
        k = Kenotic(user_id=job.user_id, db_path=DB_PATH)
        k.ingest(
            text=job.args["text"],
            source_timestamp=job.args.get("source_timestamp"),
            speaker=job.args.get("speaker"),
            confidence=job.args.get("confidence", 0.9),
            model_response=job.args.get("model_response"),
            llm_id=job.args.get("llm_id"),
        )
        log.debug("async ingest done job_id=%s user_id=%d", job.job_id, job.user_id)
    except Exception as exc:
        job.error = exc
        log.exception("async ingest failed job_id=%s", job.job_id)
    finally:
        _deregister_pending(job)
        job.done.set()


def _ingest_worker_loop() -> None:
    """Drain the ingest queue forever. Runs in a single daemon thread."""
    while True:
        job = _ingest_queue.get()
        try:
            _run_ingest_job(job)
        finally:
            _ingest_queue.task_done()


# Start the single background worker once at import time.
# daemon=True means it won't block process exit.
_worker_thread = threading.Thread(
    target=_ingest_worker_loop,
    name="reconstruct-ingest-worker",
    daemon=True,
)
_worker_thread.start()


class ToolError(Exception):
    """JSON-RPC protocol-level error. Carries a JSON-RPC error code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _json_default(o: Any) -> Any:
    """Fallback encoder for json.dumps — coerces numpy / torch / set /
    dataclass leaves the walker missed. Conservative: for anything
    unknown, fall back to its string repr rather than crashing the
    tool response."""
    # numpy scalars / arrays
    try:
        import numpy as _np
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, _np.generic):
            return o.item()
    except Exception:
        pass
    # torch tensors
    try:
        import torch as _torch
        if isinstance(o, _torch.Tensor):
            return o.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(o, (set, frozenset)):
        return list(o)
    if is_dataclass(o):
        return asdict(o)
    return str(o)


def _serialize(obj: Any) -> Any:
    """Turn SDK return shapes (dataclasses) into JSON-safe dicts."""
    if is_dataclass(obj):
        d = asdict(obj)
        # grounding_map keys may be ints — JSON keys must be strings
        if "grounding_map" in d and isinstance(d["grounding_map"], dict):
            d["grounding_map"] = {str(k): v for k, v in d["grounding_map"].items()}
        return d
    return obj


# ── Tool handlers ────────────────────────────────────────────────

def tool_ingest(args: Dict[str, Any]) -> Dict[str, Any]:
    """Queue the ingest and return immediately.

    Returns {"job_id": "...", "status": "queued"} — the caller does not
    need to poll; retrieve/reconstruct/show will flush automatically before
    executing, so follow-up reads always see the ingested data.
    """
    job = _IngestJob(
        job_id=uuid.uuid4().hex,
        user_id=_SOLO_USER_ID,
        args=args,
    )
    _register_pending(job)
    _ingest_queue.put(job)
    log.debug("ingest queued job_id=%s", job.job_id)
    return {"job_id": job.job_id, "status": "queued"}


def tool_retrieve(args: Dict[str, Any]) -> Dict[str, Any]:
    # Flush-before-read: wait for all pending ingests for this user to
    # complete before executing the retrieve. Deterministic, no timeouts.
    _flush_for_user(_SOLO_USER_ID)
    from sdk import Kenotic
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    result = k.retrieve(query=args["query"])
    return {
        "result_type": type(result).__name__,
        "data": _serialize(result),
    }


def tool_reconstruct(args: Dict[str, Any]) -> Dict[str, Any]:
    _flush_for_user(_SOLO_USER_ID)
    from sdk import Kenotic
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    result = k.retrieve(query=args["query"])
    return {"data": _serialize(result)}


def tool_benchmark(args: Dict[str, Any]) -> Dict[str, Any]:
    """LOCOMO-precise lookup. Forces short factual answers."""
    _flush_for_user(_SOLO_USER_ID)
    from sdk import Reconstruct
    result = Reconstruct(
        args["query"],
        db_path=DB_PATH,
        locomo_mode=True,
    )
    return {"data": _serialize(result)}


def tool_forget(args: Dict[str, Any]) -> Dict[str, Any]:
    from sdk import Kenotic
    by = args["by"]
    scope = args["scope"]
    # JSON has no tuple type; time_range arrives as a 2-element list.
    if by == "time_range" and isinstance(scope, list):
        if len(scope) != 2:
            raise ToolError(-32602, "time_range scope must be a 2-element array")
        scope = (scope[0], scope[1])
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    count = k.forget(by=by, scope=scope)
    # Forensic audit line — non-sensitive: the `by` and scope shape
    # (not memory content) plus the tombstone count. Useful when the
    # server is publicly reachable via Tailscale Funnel.
    log.info("reconstruct.forget by=%s scope=%r tombstones=%d", by, scope, count)
    return {"tombstones_emitted": count}


def tool_show(args: Dict[str, Any]) -> Dict[str, Any]:
    _flush_for_user(_SOLO_USER_ID)
    from sdk import Kenotic
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    rows = k.show(
        facet=args["facet"],
        value=args.get("value"),
        limit=int(args.get("limit", 100)),
        export_raw_text=bool(args.get("export_raw_text", False)),
    )
    return {"rows": rows, "count": len(rows)}


def tool_trace(args: Dict[str, Any]) -> Dict[str, Any]:
    _flush_for_user(_SOLO_USER_ID)
    from sdk import Kenotic
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    history = k.trace(subject=args["subject"], predicate=args["predicate"])
    return {"history": history, "count": len(history)}


def tool_check_proactive(args: Dict[str, Any]) -> Dict[str, Any]:
    _flush_for_user(_SOLO_USER_ID)
    from sdk import Kenotic
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    insights = k.check_proactive()
    return {"insights": [_serialize(i) for i in insights], "count": len(insights)}


def tool_profile(args: Dict[str, Any]) -> Dict[str, Any]:
    from sdk import Kenotic
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    prof = k.profile()
    return {"profile": _serialize(prof)}




def tool_process(args: Dict[str, Any]) -> Dict[str, Any]:
    """Unified entry point. Classifies intent and routes internally."""
    # Flush pending ingests so classification sees all prior data.
    _flush_for_user(_SOLO_USER_ID)
    from sdk import Kenotic
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    result = k.process(
        text=args.get("text", ""),
        speaker=args.get("speaker", "user"),
        source_timestamp=args.get("source_timestamp"),
        model_response=args.get("model_response"),
        check_proactive=bool(args.get("check_proactive", False)),
    )
    response = {
        "action": result.action,
        "result": _serialize(result.result),
        "proactive": [_serialize(i) for i in result.proactive],
        "triples_stored": result.triples_stored,
    }
    if result.ambiguities:
        response["ambiguities"] = [_serialize(a) for a in result.ambiguities]
    return response


def tool_architecture_status(args: Dict[str, Any]) -> Dict[str, Any]:
    from sdk import Kenotic
    k = Kenotic(user_id=_SOLO_USER_ID, db_path=DB_PATH)
    results = k.architecture_status()
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    return {
        "results": results,
        "summary": {"passed": passed, "failed": failed, "total": len(results)},
    }


TOOLS: Dict[str, Dict[str, Any]] = {
    "reconstruct.ingest": {
        "handler": tool_ingest,
        "description": (
            "Store text into continuity memory. Extracts structured traces "
            "and writes them locally. Returns immediately — follow-up queries "
            "automatically wait for pending writes to complete."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
                "source_timestamp": {"type": "string"},
                "speaker": {"type": "string"},
                "model_response": {"type": "string"},
                "llm_id": {
                    "type": "string",
                    "description": "Which LLM is calling (e.g. 'claude', 'gpt', 'cursor', 'grok').",
                },
            },
            "additionalProperties": False,
        },
    },
    "reconstruct": {
        "handler": tool_reconstruct,
        "description": (
            "Reconstruct a situation or answer a question from continuity memory. "
            "Ask anything — 'reconstruct the situation with my mom', "
            "'where do I work?', 'how was I feeling last week?'. Returns a "
            "verified answer grounded in what was stored, or refuses if the "
            "information doesn't exist. Never guesses."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "reconstruct.benchmark": {
        "handler": tool_benchmark,
        "description": (
            "LOCOMO-precise lookup mode. Short factual answers optimized for "
            "benchmark scoring. Same engine, forces the factual retrieval path."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
}


def _tools_list_payload() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": name,
                "description": spec["description"],
                "inputSchema": spec["inputSchema"],
            }
            for name, spec in TOOLS.items()
        ]
    }


def dispatch(method: str, params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pure JSON-RPC method dispatch. Returns the `result` payload for
    a valid request, or raises ToolError for protocol failures.

    Callers decide whether to wrap this in a JSON-RPC envelope and how
    to handle notifications (requests with no `id`).
    """
    params = params or {}

    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    if method == "ping":
        return {}

    # Lifecycle notifications from the client — accept silently.
    if method in ("notifications/initialized", "initialized"):
        return None

    if method == "tools/list":
        return _tools_list_payload()

    if method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments", {}) or {}
        spec = TOOLS.get(tool_name)
        if spec is None:
            raise ToolError(-32601, f"Unknown tool: {tool_name}")
        try:
            result = spec["handler"](tool_args)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(-32000, f"Tool error: {e}")
        return {
            "content": [{"type": "text", "text": json.dumps(result, default=_json_default)}],
        }

    raise ToolError(-32601, f"Unknown method: {method}")
