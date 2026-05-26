# Kenotic SDK -- API Reference

Full reference for the `Kenotic` class and the `Reconstruct` convenience function.

```python
from sdk import Kenotic, Reconstruct
from sdk import Answer, Situation, ProcessResult, Cluster
```

---

## Kenotic

```python
class Kenotic(user_id=0, db_path="~/.kenotic/memory.db", *, embed_device="cuda")
```

Continuity-layer client. One instance per user/database.

### Parameters

| Parameter      | Type         | Default                  | Description |
|----------------|--------------|--------------------------|-------------|
| `user_id`      | `int`        | `0`                      | Partition key. Edges are stored and retrieved under this ID. Use 0 for single-user mode. |
| `db_path`      | `str | Path` | `~/.kenotic/memory.db`   | Path to the SQLite file. Created if missing, along with parent directories. |
| `embed_device` | `str`        | `"cuda"`                 | Device for MiniLM embedding inference. Use `"cpu"` if no GPU is available. |

---

## Methods

### ingest

```python
k.ingest(
    text: str,
    *,
    source_timestamp: str | None = None,
    speaker: str | None = None,
    listener: str | None = None,
    speaker_is_user: bool = True,
    confidence: float = 0.9,
    model_response: str | None = None,
    llm_id: str | None = None,
) -> int
```

Extract triples from raw text and store them. Runs the full write path: ingestion cleanup, grammar engine extraction, temporal engine, memory store.

**Parameters**

| Parameter          | Type          | Default | Description |
|--------------------|---------------|---------|-------------|
| `text`             | `str`         | --      | Raw user utterance or conversation turn. |
| `source_timestamp` | `str \| None` | `None`  | ISO datetime of the utterance. |
| `speaker`          | `str \| None` | `None`  | Who said it. `"I"` in the text resolves to this name. |
| `listener`         | `str \| None` | `None`  | Who is being spoken to. |
| `speaker_is_user`  | `bool`        | `True`  | Whether the speaker is the end user (affects pronoun resolution). |
| `confidence`       | `float`       | `0.9`   | Confidence score (0.0--1.0) assigned to every resulting triple. |
| `model_response`   | `str \| None` | `None`  | The AI's response text. Ingested separately with `source_tag="llm:{llm_id}"`. |
| `llm_id`           | `str \| None` | `None`  | Which LLM generated `model_response` (e.g. `"claude"`, `"gpt"`). |

**Returns:** `int` -- combined number of triples stored from user text and model response.

```python
count = k.ingest("I got promoted to senior engineer.", speaker="Maya")
```

---

### retrieve

```python
k.retrieve(query: str) -> Answer | Situation
```

Query the continuity layer. Routing is automatic:

- Situational queries (`"summarize"`, `"what's going on"`, `"tell me about"`) return a `Situation` via the reconstruction path.
- Factual/lookup queries return an `Answer` via the reconstruction engine's factual path.

**Returns:** `Answer` for factual queries, `Situation` for situational queries.

```python
answer = k.retrieve("Where does Maya work?")
print(answer.text)
```

---

### reconstruct

```python
k.reconstruct(query: str) -> Situation
```

Force the reconstruction path regardless of query type. Returns a narrative summary built from converged traces.

**Returns:** `Situation` with `narrative`, `grounding`, `edge_ids`, and `clusters`.

```python
situation = k.reconstruct("What's going on in Maya's life?")
print(situation.narrative)
```

---

### forget

```python
k.forget(by: str, scope) -> int
```

Tombstone (soft-delete) memory. Idempotent -- calling again with the same scope returns 0.

**Parameters**

| Parameter | Type                       | Description |
|-----------|----------------------------|-------------|
| `by`      | `str`                      | One of: `"entity"`, `"time_range"`, `"source"`, `"triple_id"`. |
| `scope`   | `str \| int \| tuple[str, str]` | The target. Entity name, triple ID, source tag, or `(start, end)` ISO datetime tuple. |

**Returns:** `int` -- number of tombstones emitted.

```python
k.forget(by="entity", scope="Google")
k.forget(by="triple_id", scope=42)
k.forget(by="time_range", scope=("2024-01-01", "2024-06-30"))
k.forget(by="source", scope="llm:gpt")
```

---

### show

```python
k.show(
    facet: str,
    value: str | None = None,
    limit: int = 100,
    export_raw_text: bool = False,
) -> list
```

Return a browsable view of stored memory. Never exposes trace internals (embeddings, convergence state, salience).

**Parameters**

| Parameter         | Type          | Default | Description |
|-------------------|---------------|---------|-------------|
| `facet`           | `str`         | --      | One of: `"time"`, `"entity"`, `"source"`, `"trace"`. |
| `value`           | `str \| None` | `None`  | Filter value (e.g. entity name, source tag). |
| `limit`           | `int`         | `100`   | Max rows to return. |
| `export_raw_text` | `bool`        | `False` | If True, includes the original `source_text` field. |

**Returns:** `list[dict]` -- fact summaries, optionally with source text.

```python
edges = k.show("entity", value="Maya", limit=10)
```

---

### trace

```python
k.trace(subject: str, predicate: str) -> list
```

Return the full supersession history for a subject+predicate pair. Surfaces every object value ever stored -- both active and superseded -- ordered oldest to newest.

Each entry is a dict with keys: `id`, `object`, `is_active`, `first_learned_at`, `source_timestamp`, `source_tag`, `superseded_by`, `superseded_at`, `sequence_number`.

**Returns:** `list[dict]`

```python
history = k.trace("Maya", "work_at")
# [{"object": "Startup Inc", "is_active": False, ...},
#  {"object": "Vantage Systems", "is_active": True, ...}]
```

---

### check_proactive

```python
k.check_proactive() -> list
```

Return proactive insights -- open arcs that are due for surfacing. Returns an empty list if no arcs are due or the arcs table does not exist.

**Returns:** `list` -- ProactiveInsight dataclasses.

```python
insights = k.check_proactive()
```

---

### profile

```python
k.profile() -> AdaptationProfile
```

Return the current adaptation profile for this user. The profile has dimensions `warmth`, `formality`, `initiative`, and `check_in_frequency`, each ranging 0.0--1.0.

**Returns:** `AdaptationProfile` dataclass.

```python
p = k.profile()
print(p.warmth, p.formality)
```

---

### process

```python
k.process(
    text: str,
    *,
    speaker: str = "user",
    listener: str | None = None,
    speaker_is_user: bool = True,
    source_timestamp: str | None = None,
    model_response: str | None = None,
    check_proactive: bool = False,
    llm_id: str | None = None,
) -> ProcessResult
```

All-in-one entry point. Classifies intent (statement, question, command, backchannel) and routes to the appropriate engine internally. Returns a `ProcessResult` with an `action` discriminator.

**Parameters**

| Parameter          | Type          | Default   | Description |
|--------------------|---------------|-----------|-------------|
| `text`             | `str`         | --        | Any English text -- statement, question, command. |
| `speaker`          | `str`         | `"user"`  | Who said it. |
| `listener`         | `str \| None` | `None`    | Who is being spoken to. |
| `speaker_is_user`  | `bool`        | `True`    | Whether the speaker is the end user. |
| `source_timestamp` | `str \| None` | `None`    | ISO datetime of the utterance. |
| `model_response`   | `str \| None` | `None`    | The AI's response text (ingested separately). |
| `check_proactive`  | `bool`        | `False`   | If True with empty text, returns due arcs. If True after a store, checks for new insights. |
| `llm_id`           | `str \| None` | `None`    | Which LLM generated `model_response`. |

**Returns:** `ProcessResult`

```python
result = k.process("Maya got promoted to VP.", speaker="Maya")
print(result.action)          # "stored"
print(result.triples_stored)  # 2
```

---

### architecture_status

```python
k.architecture_status() -> list[dict[str, str]]
```

Return the cached kenoticArchitectureV1 verifier results from initialization. Each entry has keys: `check`, `status`, `detail`.

```python
status = k.architecture_status()
for s in status:
    print(s["check"], s["status"])
```

---

## Reconstruct (function)

```python
Reconstruct(
    text: str,
    *,
    speaker: str = "user",
    listener: str | None = None,
    speaker_is_user: bool = True,
    source_timestamp: str | None = None,
    model_response: str | None = None,
    llm_id: str | None = None,
    check_proactive: bool = False,
    user_id: int = 0,
    db_path: str | Path = "~/.kenotic/memory.db",
    embed_device: str = "cuda",
    locomo_mode: bool = False,
) -> ProcessResult
```

Module-level convenience function. Creates a singleton `Kenotic` instance on first call and reuses it for subsequent calls with the same `db_path` and `user_id`. Routes everything through `process()`.

Set `locomo_mode=True` to force short factual answers (disables automatic situational routing).

```python
from sdk import Reconstruct

Reconstruct("I started a new job at Acme.", speaker="Sam")
result = Reconstruct("Where do I work?")
print(result.result.text)  # "Acme"
```

---

## Types

### Answer

```python
@dataclass
class Answer:
    text: str = ""
    grounding: list[str] = []
    edge_ids: list[int] = []
    return_field: str = "episodic"
    refusal: bool = False
    refusal_reason: str = ""
```

### Situation

```python
@dataclass
class Situation:
    narrative: str = ""
    clusters: list[Cluster] = []
    grounding: list[str] = []
    edge_ids: list[int] = []
```

### Cluster

```python
@dataclass
class Cluster:
    cluster_id: str = ""
    edge_ids: list[int] = []
    participants: list[str] = []
```

### ProcessResult

```python
@dataclass
class ProcessResult:
    action: str            # "stored", "answered", "reconstructed", "forgot", "skipped", "proactive"
    result: Any = None     # int (store count), Answer, Situation, list, or None
    proactive: list = []
    triples_stored: int = 0
    ambiguities: list = []
```
