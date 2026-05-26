# Kenotic SDK -- Quickstart

The Kenotic Python SDK provides the continuity layer as a library. Feed it raw text (statements, questions, commands) and it handles decomposition, storage, retrieval, and reconstruction internally.

## Installation

From PyPI:

```bash
pip install kenotic
```

From source:

```bash
git clone https://github.com/kenoticlabs/kenotic.git
cd kenotic
pip install -e .
```

Requires Python 3.10+.

## Basic Usage

```python
from sdk import Kenotic

k = Kenotic(user_id=1, db_path="my_memory.db")

# Ingest statements
k.ingest("Maya started at Vantage Systems in 2023.", speaker="Maya")
k.ingest("Maya's manager is Derek.", speaker="Maya")
k.ingest("Maya is preparing for her annual review next Friday.", speaker="Maya")

# Retrieve a factual answer
answer = k.retrieve("Who is Maya's manager?")
print(answer.text)       # "Derek"
print(answer.refusal)    # False

# Reconstruct a situation
situation = k.reconstruct("What's going on in Maya's life?")
print(situation.narrative)
```

### One-function shorthand

For scripts that don't need fine-grained control, `Reconstruct` wraps everything into a single call. It creates a singleton `Kenotic` instance internally and routes statements to storage, questions to retrieval.

```python
from sdk import Reconstruct

Reconstruct("Maya started at Vantage Systems in 2023.", speaker="Maya")
Reconstruct("Maya's manager is Derek.", speaker="Maya")

result = Reconstruct("Who is Maya's manager?")
print(result.action)        # "answered"
print(result.result.text)   # "Derek"
```

## Return Types

### Answer

Returned by `k.retrieve()` for factual queries.

| Field          | Type       | Description                                      |
|----------------|------------|--------------------------------------------------|
| `text`         | `str`      | The answer text, or a refusal message.           |
| `grounding`    | `list[str]`| Source texts that support the answer.             |
| `edge_ids`     | `list[int]`| Database row IDs of the supporting edges.        |
| `return_field` | `str`      | Which trace dimension provided the answer.       |
| `refusal`      | `bool`     | True if the system could not find an answer.      |
| `refusal_reason`| `str`     | Why the system refused (empty if not a refusal).  |

### Situation

Returned by `k.reconstruct()` for situational queries.

| Field       | Type            | Description                                 |
|-------------|-----------------|---------------------------------------------|
| `narrative` | `str`           | A reconstructed summary of the situation.   |
| `clusters`  | `list[Cluster]` | Groups of related edges.                    |
| `grounding` | `list[str]`     | Source texts that support the narrative.     |
| `edge_ids`  | `list[int]`     | Database row IDs of the contributing edges. |

### ProcessResult

Returned by `k.process()` and `Reconstruct()`.

| Field            | Type   | Description                                              |
|------------------|--------|----------------------------------------------------------|
| `action`         | `str`  | One of: `stored`, `answered`, `reconstructed`, `forgot`, `skipped`, `proactive`. |
| `result`         | `Any`  | Depends on action: int (store count), Answer, Situation, list, or None. |
| `proactive`      | `list` | Proactive insights surfaced after storage.               |
| `triples_stored` | `int`  | Number of triples written during this call.              |
| `ambiguities`    | `list` | Ambiguities detected during ingestion.                   |

## Where Data Lives

By default, the SQLite database is created at `~/.kenotic/memory.db`. Override this with the `db_path` parameter:

```python
k = Kenotic(user_id=1, db_path="/path/to/custom.db")
```

The directory is created automatically if it does not exist. Each `user_id` partitions data within the same database file -- edges for user 1 are never returned when querying as user 2.

## Dependencies

The SDK depends on spaCy, sentence-transformers, torch, numpy, and nltk (see `pyproject.toml` for the full list).

On first run, the following models are downloaded automatically:

- **spaCy**: `en_core_web_sm` language model
- **sentence-transformers**: `all-MiniLM-L6-v2` embedding model

These downloads happen once and are cached locally. First initialization may take 30-60 seconds depending on network speed. Subsequent starts are fast.

Set `embed_device="cpu"` if you do not have a CUDA-capable GPU:

```python
k = Kenotic(user_id=1, embed_device="cpu")
```
