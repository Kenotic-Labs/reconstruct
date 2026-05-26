"""
REST API for Kenotic Continuity Memory.

Thin wrappers over the same tool handlers that MCP uses.
Auto-generates an OpenAPI spec suitable for GPT Actions, Gemini tool schemas,
or any platform that consumes OpenAPI/function definitions.

Mounted at /api/v1 inside the existing FastAPI app.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from mcp.tools import (
    ToolError,
    tool_check_proactive,
    tool_forget,
    tool_ingest,
    tool_profile,
    tool_reconstruct,
)


# ── Request models ──────────────────────────────────────────────

class StoreRequest(BaseModel):
    """Store a message into continuity memory."""
    text: str = Field(..., description="The user's message to store.")
    speaker: Optional[str] = Field(None, description="The user's name. 'I' resolves to this.")
    source_timestamp: Optional[str] = Field(None, description="ISO 8601 datetime of the message.")
    model_response: Optional[str] = Field(None, description="The AI's response text.")
    llm_id: Optional[str] = Field(None, description="Which LLM is calling (e.g. 'chatgpt').")
    confidence: Optional[float] = Field(0.9, description="STT confidence, 0-1.")


class RetrieveRequest(BaseModel):
    """Answer a question from continuity memory."""
    query: str = Field(..., description="The question to answer from stored memory.")


class ReconstructRequest(BaseModel):
    """Reconstruct the current living state of a situation or person."""
    query: str = Field(..., description="What to reconstruct, e.g. 'What's going on with Sam?'")


class ForgetRequest(BaseModel):
    """Delete memories by entity, time range, source, or triple ID."""
    by: Literal["entity", "time_range", "source", "triple_id"] = Field(
        ..., description="How to select what to forget."
    )
    scope: Union[str, int, List[str]] = Field(
        ..., description="The target: entity name, [start, end] timestamps, source string, or triple ID."
    )


# ── Response models ─────────────────────────────────────────────

class StoreResponse(BaseModel):
    job_id: str
    status: str


class RetrieveResponse(BaseModel):
    text: str = Field(..., description="The answer, or a refusal message.")
    refusal: bool = Field(False, description="True if the information was not found.")
    grounding: List[str] = Field(default_factory=list, description="Source texts that ground the answer.")
    edge_ids: List[int] = Field(default_factory=list)
    return_field: str = "episodic"


class ReconstructResponse(BaseModel):
    text: str = Field(..., description="Narrative reconstruction or refusal.")
    refusal: bool = False
    grounding: List[str] = Field(default_factory=list)
    edge_ids: List[int] = Field(default_factory=list)


class ProactiveResponse(BaseModel):
    insights: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0


class ForgetResponse(BaseModel):
    tombstones_emitted: int


class ProfileResponse(BaseModel):
    profile: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ── Helper ──────────────────────────────────────────────────────

def _tool_error_to_http(e: ToolError) -> HTTPException:
    if e.code == -32602:
        return HTTPException(status_code=400, detail={"error": "invalid_params", "detail": e.message})
    if e.code == -32601:
        return HTTPException(status_code=404, detail={"error": "not_found", "detail": e.message})
    return HTTPException(status_code=500, detail={"error": "internal", "detail": e.message})


# ── Router ──────────────────────────────────────────────────────

rest_router = APIRouter(tags=["memory"])


@rest_router.post(
    "/store",
    response_model=StoreResponse,
    operation_id="store_memory",
    summary="Store a message into continuity memory",
    description=(
        "Ingests text into the continuity memory system. The system decomposes "
        "it into 5 structured traces (episodic, emotional, temporal, relational, "
        "schematic) and stores them locally. Returns immediately — subsequent "
        "retrieves automatically wait for the write to complete."
    ),
)
async def store(req: StoreRequest):
    try:
        result = await run_in_threadpool(tool_ingest, req.model_dump(exclude_none=True))
        return StoreResponse(**result)
    except ToolError as e:
        raise _tool_error_to_http(e)


@rest_router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    operation_id="retrieve_memory",
    summary="Answer a question from continuity memory",
    description=(
        "Queries the continuity memory for a verified answer. Returns the answer "
        "grounded in what was actually stored, or refuses if the information was "
        "never stored. Refusals are authoritative — do not override them."
    ),
)
async def retrieve(req: RetrieveRequest):
    try:
        result = await run_in_threadpool(tool_reconstruct, {"query": req.query})
        data = result.get("data", {})
        if isinstance(data, dict):
            return RetrieveResponse(
                text=data.get("text", data.get("answer", "")),
                refusal=data.get("refusal", False),
                grounding=data.get("grounding", []),
                edge_ids=data.get("edge_ids", []),
                return_field=data.get("return_field", "episodic"),
            )
        return RetrieveResponse(text=str(data), refusal=False)
    except ToolError as e:
        raise _tool_error_to_http(e)


@rest_router.post(
    "/reconstruct",
    response_model=ReconstructResponse,
    operation_id="reconstruct_situation",
    summary="Reconstruct the living state of a situation",
    description=(
        "Reconstructs the current state of a person or situation from memory. "
        "Combines multiple traces into a coherent narrative. Use for open-ended "
        "questions like 'What's going on with Sam?' or 'Summarize my situation.'"
    ),
)
async def reconstruct(req: ReconstructRequest):
    try:
        result = await run_in_threadpool(tool_reconstruct, {"query": req.query})
        data = result.get("data", {})
        if isinstance(data, dict):
            return ReconstructResponse(
                text=data.get("text", data.get("narrative", data.get("answer", ""))),
                refusal=data.get("refusal", False),
                grounding=data.get("grounding", []),
                edge_ids=data.get("edge_ids", []),
            )
        return ReconstructResponse(text=str(data), refusal=False)
    except ToolError as e:
        raise _tool_error_to_http(e)


@rest_router.post(
    "/check_proactive",
    response_model=ProactiveResponse,
    operation_id="check_proactive",
    summary="Get proactive memory insights",
    description=(
        "Checks if there are any proactive insights the system should surface — "
        "upcoming events, unresolved situations, recurring patterns."
    ),
)
async def check_proactive():
    try:
        result = await run_in_threadpool(tool_check_proactive, {})
        return ProactiveResponse(
            insights=result.get("insights", []),
            count=result.get("count", 0),
        )
    except ToolError as e:
        raise _tool_error_to_http(e)


@rest_router.post(
    "/forget",
    response_model=ForgetResponse,
    operation_id="forget_memory",
    summary="Delete memories",
    description=(
        "Tombstones memories matching the given criteria. Supports deletion by "
        "entity name, time range, source, or specific triple ID."
    ),
)
async def forget(req: ForgetRequest):
    try:
        result = await run_in_threadpool(tool_forget, {"by": req.by, "scope": req.scope})
        return ForgetResponse(tombstones_emitted=result["tombstones_emitted"])
    except ToolError as e:
        raise _tool_error_to_http(e)


@rest_router.get(
    "/profile",
    response_model=ProfileResponse,
    operation_id="get_profile",
    summary="Get user memory profile",
    description="Returns the user's adaptation profile derived from stored memory patterns.",
)
async def profile():
    try:
        result = await run_in_threadpool(tool_profile, {})
        return ProfileResponse(profile=result.get("profile", {}))
    except ToolError as e:
        raise _tool_error_to_http(e)
