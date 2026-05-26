# API Quickstart

A step-by-step walkthrough of the Kenotic Continuity Memory REST API. All endpoints live under `/api/v1` and accept/return JSON.

The example data follows Maya, who works at a startup, has a dog named Bowie, and is planning a trip to Japan.

---

## 1. Start the Server

```bash
python -m mcp.http_server --host 127.0.0.1 --port 7130
```

Without a `--token` flag, authentication is disabled (localhost only). See [Section 8](#8-with-authentication) for enabling auth.

---

## 2. Store Messages

Store three messages from Maya across different topics.

**Message 1 -- Work context:**

```bash
curl -X POST http://localhost:7130/api/v1/store \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I just got promoted to engineering lead at Dawnlight. The team is growing fast and I am hiring two backend engineers this quarter.",
    "speaker": "Maya",
    "source_timestamp": "2026-05-20T09:15:00Z"
  }'
```

```json
{
  "job_id": "a1b2c3d4",
  "status": "accepted"
}
```

**Message 2 -- Personal life:**

```bash
curl -X POST http://localhost:7130/api/v1/store \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bowie has been limping since yesterday. I scheduled a vet appointment for Thursday at 10am.",
    "speaker": "Maya",
    "source_timestamp": "2026-05-21T18:30:00Z"
  }'
```

```json
{
  "job_id": "e5f6g7h8",
  "status": "accepted"
}
```

**Message 3 -- Trip planning:**

```bash
curl -X POST http://localhost:7130/api/v1/store \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I booked flights to Tokyo for July 10 through July 24. I want to visit Kyoto and Osaka too. My friend Priya is joining for the first week.",
    "speaker": "Maya",
    "source_timestamp": "2026-05-22T20:00:00Z"
  }'
```

```json
{
  "job_id": "i9j0k1l2",
  "status": "accepted"
}
```

The store endpoint returns immediately. The system decomposes each message into five structured traces (episodic, emotional, temporal, relational, schematic) in the background. Subsequent retrieves automatically wait for pending writes to complete.

---

## 3. Retrieve a Fact

Ask a direct factual question.

```bash
curl -X POST http://localhost:7130/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When is Maya's vet appointment?"
  }'
```

```json
{
  "text": "Thursday at 10am",
  "refusal": false,
  "grounding": [
    "Bowie has been limping since yesterday. I scheduled a vet appointment for Thursday at 10am."
  ],
  "edge_ids": [42],
  "return_field": "episodic"
}
```

If the information was never stored, the response refuses authoritatively:

```json
{
  "text": "This information is not mentioned in the conversation.",
  "refusal": true,
  "grounding": [],
  "edge_ids": [],
  "return_field": "episodic"
}
```

Refusals are authoritative. The system verified that the answer does not exist in stored memory -- do not override them.

---

## 4. Reconstruct a Situation

Ask an open-ended question that requires combining multiple traces.

```bash
curl -X POST http://localhost:7130/api/v1/reconstruct \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is going on with Maya right now?"
  }'
```

```json
{
  "text": "Maya was recently promoted to engineering lead at Dawnlight and is hiring two backend engineers this quarter. Her dog Bowie has been limping, and she has a vet appointment scheduled for Thursday at 10am. She is also planning a trip to Tokyo from July 10 to July 24, with stops in Kyoto and Osaka. Her friend Priya is joining for the first week.",
  "refusal": false,
  "grounding": [
    "I just got promoted to engineering lead at Dawnlight. The team is growing fast and I am hiring two backend engineers this quarter.",
    "Bowie has been limping since yesterday. I scheduled a vet appointment for Thursday at 10am.",
    "I booked flights to Tokyo for July 10 through July 24. I want to visit Kyoto and Osaka too. My friend Priya is joining for the first week."
  ],
  "edge_ids": [37, 42, 48]
}
```

---

## 5. Check Proactive Insights

Surface upcoming events, unresolved situations, or recurring patterns without asking a specific question.

```bash
curl -X POST http://localhost:7130/api/v1/check_proactive \
  -H "Content-Type: application/json" \
  -d '{}'
```

```json
{
  "insights": [
    {
      "type": "upcoming_event",
      "text": "Maya has a vet appointment for Bowie on Thursday at 10am.",
      "urgency": "high",
      "edge_ids": [42]
    },
    {
      "type": "upcoming_event",
      "text": "Maya's trip to Tokyo starts July 10.",
      "urgency": "low",
      "edge_ids": [48]
    }
  ],
  "count": 2
}
```

---

## 6. Get Profile

Retrieve the user's adaptation profile derived from stored memory patterns.

```bash
curl http://localhost:7130/api/v1/profile
```

```json
{
  "profile": {
    "top_entities": ["Dawnlight", "Bowie", "Priya", "Tokyo"],
    "active_topics": ["career", "pet_health", "travel"],
    "edge_count": 12
  }
}
```

---

## 7. Forget an Entity

Delete all memories related to a specific entity.

```bash
curl -X POST http://localhost:7130/api/v1/forget \
  -H "Content-Type: application/json" \
  -d '{
    "by": "entity",
    "scope": "Priya"
  }'
```

```json
{
  "tombstones_emitted": 3
}
```

Other deletion modes:

```bash
# By time range
curl -X POST http://localhost:7130/api/v1/forget \
  -H "Content-Type: application/json" \
  -d '{"by": "time_range", "scope": ["2026-05-21T00:00:00Z", "2026-05-21T23:59:59Z"]}'

# By specific triple ID
curl -X POST http://localhost:7130/api/v1/forget \
  -H "Content-Type: application/json" \
  -d '{"by": "triple_id", "scope": 42}'
```

Tombstoned memories are excluded from all future retrieval and reconstruction.

---

## 8. With Authentication

When deploying beyond localhost, start the server with a bearer token:

```bash
python -m mcp.http_server --host 0.0.0.0 --port 7130 --token YOUR_SECRET
```

Or set the environment variable:

```bash
export KENOTIC_MCP_TOKEN=YOUR_SECRET
python -m mcp.http_server --host 0.0.0.0 --port 7130
```

Then include the `Authorization` header on every request:

```bash
curl -X POST http://localhost:7130/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SECRET" \
  -d '{"query": "When is Maya's vet appointment?"}'
```

Without a valid token, all authenticated endpoints return `401 Unauthorized`. The `/healthz` endpoint is always unauthenticated.

---

## Error Handling

All errors return a JSON body:

```json
{
  "error": "invalid_params",
  "detail": "Field 'text' is required."
}
```

| Status | Meaning |
|--------|---------|
| 400 | Invalid parameters |
| 401 | Missing or invalid bearer token |
| 404 | Resource not found |
| 500 | Internal server error |
