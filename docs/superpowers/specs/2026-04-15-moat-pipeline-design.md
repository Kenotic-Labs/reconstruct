# Moat Pipeline — Design Spec

**Status:** approved by Sam 2026-04-15
**Author:** brainstorming session with Sam Tanguturi
**Implementation plan:** `docs/superpowers/plans/2026-04-15-moat-pipeline.md`

---

## Problem

Current `RetrievalEngine.retrieve()` runs only two of the five Moat layers (Entry + Validate). Expand, Group, and Relate exist as concepts in the pitch but are not wired into the read path. The Entry layer itself has a replacement bug — PQ cosine is used if any predicted_query row exists, otherwise edge_embedding. That's branching, not layering, and it threw away the conventional RAG signal that was hitting 54%.

Goal: wire all five layers as a **single pipeline** that runs on every query, with no weight tuning, no branching substitutions, and every property on the `relationships` / `entities` / `predicted_queries` tables actually *used* by the stage it belongs to.

---

## Non-goals

- No new tables. Schema is sufficient.
- No new ML models. Existing embedder + raya-srl-220m-v4 + spaCy are the stack.
- No attempt to answer subjective queries ("how is Maya feeling") beyond what the extraction layer stored. If the type tag doesn't exist, we don't invent one.
- No backward-compatibility shim for the old `_cosine_pool` path — it gets deleted.

---

## Architecture

```
                    ┌───────────────── shared pipeline ─────────────────┐
query_text ─────────▶ Entry Cosine → Expand(union) → Group(filter) →
                      Relate(filter) → Exit Cosine ──┐
                                                     │
                                        ranked survivors
                                                     │
                                      ┌──────────────┴──────────────┐
                                      ▼                             ▼
                                   Lookup                    Reconstruction
                                  (top 1)                    (top N → fuse)
                                      │                             │
                                      ▼                             ▼
                                 Validate                       Situation
                                 (warn only)                    narrative
                                      │
                                      ▼
                                   Answer
```

Two output modes, one shared pipeline. Mode is chosen from the query surface, not from user intent signals.

---

## Stage contracts

Every stage takes `(candidates: List[Candidate])` and returns `List[Candidate]`. No stage mutates DB. Each stage may ADD to a candidate's annotations (entity_overlap, cluster_members, hops_to_entity, exit_score) but stages DO NOT compute weighted combined scores — structural stages either keep or drop a candidate; cosine stages order them.

### `Candidate`

```python
@dataclass
class Candidate:
    relationship_id: int
    edge: Dict[str, Any]        # full relationships row (see property list)
    entry_cosine: float = 0.0   # stage 1 — max(pq_cos, edge_cos)
    entity_overlap: int = 0     # stage 2 — # of query entities touching this edge
    cluster_members: int = 0    # stage 3 — # of fellow survivors in same cluster
    hops_to_entity: int = -1    # stage 4 — BFS hops to nearest query entity
    exit_cosine: float = 0.0    # stage 5 — re-scored against edge_embedding
    source_stages: Set[str] = field(default_factory=set)  # {"entry","expand",...}
```

### Stage 1 — Entry Cosine

**Purpose:** semantic recall. Get every edge that might be relevant onto the table.

**Reads:**
- `predicted_queries.question_embedding` (join via `relationship_id`)
- `predicted_queries.user_id` (filter)
- `relationships.edge_embedding`
- `relationships.is_current`, `relationships.tombstoned_at` (must survive)
- Embedded query vector

**Logic:**
1. Fetch all live `predicted_queries` rows for user with joined relationship row. Compute `cosine(query_emb, pq.question_embedding)` per row.
2. Fetch all live `relationships` rows for user with non-null `edge_embedding`. Compute `cosine(query_emb, edge.edge_embedding)`.
3. Merge by `relationship_id`. `entry_cosine = max(pq_cos, edge_cos)` across all scores seen for that `relationship_id`.
4. Keep top **K=80** by `entry_cosine`.

**Output:** `Candidate` list with `source_stages = {"entry"}`, `entry_cosine` set.

**Inapplicability:** none — Entry always runs. If both pools are empty, return `[]`.

### Stage 2 — Expand (UNION)

**Purpose:** structural recall. Pull edges that Entry cosine missed but are *about* the things the query mentions.

**Reads:**
- Query noun-phrases (extracted via spaCy `en_core_web_sm`)
- `entities.embedding`, `entities.name`, `entities.entity_type`, `entities.user_id`
- `relationships.subject`, `relationships.object`, `relationships.user_id`, `relationships.is_current`, `relationships.tombstoned_at`

**Logic:**
1. Extract noun phrases + named entities from `query_text`. Filter the closed WH/auxiliary stopword set.
2. Embed each candidate phrase. For each user entity, compute `cosine(phrase_emb, entity.embedding)`. Keep entity rows with cosine ≥ **0.55**. Call this set `E_q` (resolved query entities by name).
3. Pull all live relationships where `subject ∈ E_q` OR `object ∈ E_q`. **UNION** into candidate set (not intersection). Each pulled edge gets `entity_overlap = |{ e ∈ E_q : e.name == subject or e.name == object }|`.
4. Also compute `entity_overlap` for pre-existing Entry candidates.

**Output:** expanded candidate list. Pre-existing candidates keep their `entry_cosine`; new candidates have `entry_cosine = 0.0` but `entity_overlap > 0`. `source_stages |= {"expand"}` for any candidate that got an overlap bump.

**Inapplicability:** if `E_q == ∅`, Stage 2 is a no-op. Candidates pass through unchanged.

### Stage 3 — Group (FILTER)

**Purpose:** keep only edges that sit in a narrative neighborhood containing a query-entity edge.

**Reads:**
- `relationships.cluster_id`
- `relationships.arc_id` (secondary key for cross-cluster grouping)
- `Candidate.entity_overlap` from stage 2

**Logic:**
1. Compute **anchor clusters**: set of `cluster_id` values held by any candidate with `entity_overlap > 0`.
2. Also compute **anchor arcs**: set of `arc_id` values held by any candidate with `entity_overlap > 0`.
3. Keep candidate if `cluster_id ∈ anchor_clusters` OR `arc_id ∈ anchor_arcs`.
4. Set `cluster_members = |{ c ∈ survivors : c.cluster_id == this.cluster_id }|` for each survivor (used by Reconstruction).

**Output:** narrowed candidate list. `source_stages |= {"group"}` for every survivor.

**Inapplicability:** if no candidate has `entity_overlap > 0` (no query entities, or none in DB), Stage 3 is a no-op.

### Stage 4 — Relate (FILTER)

**Purpose:** keep only edges reachable from a query entity within a short graph walk. Structural coherence — not semantic.

**Reads:**
- `relationships.subject`, `relationships.object` (full user graph, not just survivors)
- `relationships.edge_relational_type` (used to weight hops — future-proofing; v1 counts all types as 1 hop)
- `E_q` from stage 2

**Logic:**
1. Build adjacency `adj: name → set[name]` from ALL live relationships for user (not survivors only — we need the true graph to walk).
2. For each candidate, compute `hops_to_entity = min over (subject, object) of BFS distance to any name in E_q`. Cap at **MAX_HOPS=3**; unreachable → `hops_to_entity = -1`.
3. Drop candidates with `hops_to_entity < 0`.

**Output:** narrowed candidate list with `hops_to_entity` set.

**Inapplicability:** if `E_q == ∅`, Stage 4 is a no-op.

### Stage 5 — Exit Cosine

**Purpose:** precision. Re-rank every survivor against the query by the conventional edge_embedding (surface-text, includes entity names).

**Reads:**
- `relationships.edge_embedding` for each survivor (re-fetched or carried in `Candidate.edge`)
- Embedded query vector

**Logic:**
1. For each survivor, compute `exit_cosine = cosine(query_emb, edge.edge_embedding)`. If `edge_embedding` is null, `exit_cosine = 0.0`.
2. Sort survivors by `exit_cosine` DESC.

**Output:** same list, reordered. `source_stages |= {"exit"}`.

**Inapplicability:** none — always runs on whatever survived.

### Stage 6 — Validate (WARN only, Lookup mode only)

**Purpose:** last-line sanity check on the top answer. Not a filter on the pool.

**Reads:**
- `relationships.subject_type`, `relationships.object_type` on top-1 edge
- WH type parsed from query (`parse_expected_answer_type`)

**Logic:**
1. If `expected_wh_type is None` (query has no clean WH), skip — no warning.
2. If `top.subject_type == expected OR top.object_type == expected` → no warning.
3. Else → include `validate_warning = "type_mismatch"` in `convergence_details`, still return the answer.

**Output:** never drops anything.

---

## Output modes

### Lookup mode

Triggered when query starts with a WH word (`who|what|when|where|why|which|how`) or is otherwise a clear interrogative.

- Return top-1 from Exit Cosine sort.
- Run Validate as warning.
- Tie-break when `exit_cosine` ties: `sequence_number` DESC.
- Source recorded as `moat_pipeline_lookup`.

### Reconstruction mode

Triggered by non-WH or "tell me" / "what's going on" phrasings. Default: Lookup.

- Take top **N=20** from Exit Cosine sort.
- Bucket by `cluster_id`. Pick dominant cluster (most edges; tie → highest sum of `exit_cosine`).
- Fuse via existing `_fuse_cluster` / `_render_situation_deterministic`. These use:
  - `sequence_number` for timeline order
  - `subject`, `object` for participants
  - `predicate` for events
  - `edge_emotional_valence`, `edge_emotional_label` for dominant mood
  - `edge_episodic_significance` for pivotal events
  - `edge_temporal_context` for temporal anchor
  - `edge_schematic_category` for narrative role
- Return `Situation` object.
- Validate does not run in Reconstruction mode.

---

## Property-to-stage wiring map

Every column on `relationships` has a home. If it doesn't, it should not be on the table.

| Column | Read by | Role |
|---|---|---|
| `id`, `user_id` | all stages | filter / identity |
| `subject`, `object` | Expand, Relate, Lookup output, Reconstruction (participants) | entity resolution + graph |
| `predicate` | Lookup output, Reconstruction (events) | sentence render |
| `confidence` | — (future; currently unused at read time) | — |
| `object_type`, `subject_type` | Validate | type coherence warning |
| `is_current`, `tombstoned_at`, `superseded_at`, `superseded_by` | Entry, Expand, Relate | must survive |
| `sequence_number` | Lookup tie-break, Reconstruction timeline | recency |
| `edge_embedding` | Entry (via cosine), Exit Cosine | semantic |
| `question_embedding` (on `predicted_queries`) | Entry | semantic (question↔question) |
| `cluster_id` | Group, Reconstruction (grouping) | narrative neighborhood |
| `arc_id` | Group (secondary) | cross-cluster arc |
| `edge_emotional_valence`, `edge_emotional_label` | Reconstruction | mood fusion |
| `edge_episodic_significance` | Reconstruction | pivotal event selection |
| `edge_temporal_context` | Reconstruction | temporal anchor |
| `edge_schematic_category` | Reconstruction | narrative role |
| `edge_relational_type` | Relate (future hop weight), Reconstruction | relation kind |
| `source_text` | Exit Cosine (fallback embedding if edge_embedding null) | surface text |
| `source_tag`, `source_timestamp`, `provenance_memory_id` | provenance | — |
| `situation_id` | Reconstruction boundary | — |
| `canonical_fields`, `utterance_type_id`, `tombstone_reason`, `tombstone_op_id`, `first_learned_at`, `last_confirmed_at` | write-path audit | — |

`entities` table:

| Column | Read by | Role |
|---|---|---|
| `id`, `user_id` | all | identity |
| `name` | Expand (name match against subject/object), Relate (graph node key) | entity key |
| `entity_type` | (future Validate refinement) | — |
| `embedding` | Expand (query-entity resolution) | semantic |
| `mention_count`, `first_mentioned_at`, `last_mentioned_at`, `attributes` | write-path | — |

---

## Error handling

| Situation | Behavior |
|---|---|
| Empty query | `StructuralRefusal(reason="empty_query")` |
| Embed failure on query | `StructuralRefusal(reason="embed_failed")` |
| Entry returns `[]` AND Expand adds nothing | `StructuralRefusal(reason="no_candidates")` |
| Entry `[]` but Expand finds entity edges | continue — Expand is the source |
| Group / Relate filter everything | `StructuralRefusal(reason="no_structural_match")` |
| All survivors have `edge_embedding == null` | order by `sequence_number` DESC only; don't refuse |
| Validate top-1 type mismatch | return answer with `validate_warning` field; do not refuse |
| Reconstruction has survivors but no cluster_id on any | return Situation with `source="unclustered"` |

---

## Testing strategy

Unit tests per stage, property-focused:

- `test_entry_merges_pq_and_edge_by_max_relationship_id`
- `test_entry_filters_tombstoned`
- `test_expand_unions_entity_edges_not_in_entry_pool`
- `test_expand_passes_through_when_no_query_entities`
- `test_group_keeps_cluster_peers_of_entity_touching_edge`
- `test_group_keeps_arc_peers`
- `test_group_passes_through_when_no_overlap`
- `test_relate_drops_disconnected`
- `test_relate_passes_through_when_no_query_entities`
- `test_relate_uses_full_user_graph_not_survivors`
- `test_exit_reranks_by_edge_embedding_surface_text`
- `test_exit_handles_null_edge_embedding_without_refusing`
- `test_validate_warns_but_does_not_drop_on_type_mismatch`
- `test_validate_skips_when_wh_type_unresolved`

End-to-end (on seeded YAML stories 51-60):

- `test_where_maya_works_returns_vantage_via_full_pipeline`
- `test_tell_me_about_maya_returns_job_cluster_situation`
- `test_no_entity_query_uses_cosine_only_path`

Benchmark regression gates:

- YAML audit ≥ **54%** (was 54% pre-rewrite, 11.6% after rewrite, 29% after sort-key fix)
- ATANT Core 50 ≥ **20%**
- Stress 51-100 ≥ **5.4%**
- Banff continuity ≥ **5/8**
- DeLores long-convo ≥ **6/10**

Any benchmark below baseline → stop, diagnose, do not ship.

---

## Parameters (structural caps, not tuning knobs)

| Symbol | Value | Meaning |
|---|---|---|
| `K` | 80 | Entry top-K cap |
| `τ_entity` | 0.55 | Query-phrase → entity cosine threshold |
| `MAX_HOPS` | 3 | Relate BFS depth cap |
| `N_reconstruct` | 20 | Reconstruction top-N before cluster fuse |

These are structural caps — they bound pool sizes and search depth. They are not coefficients in a scored combination. Changing them changes pool sizes, not the relative importance of one signal vs another.

---

## What this spec intentionally does not specify

- How the embedder is invoked, parallelized, or cached. Existing `embed_text` is authoritative.
- How `parse_expected_answer_type` decides the WH type. Existing logic stands.
- How `_fuse_cluster` and `_render_situation_deterministic` work. Reconstruction reuses them unchanged.
- How the MCP HTTP server wires this. `retrieve()` is the single entry point; server integration is unaffected.

---

## Rollout

1. Write tests per the Testing section (TDD).
2. Implement stages in order: Entry → Expand → Group → Relate → Exit → Validate.
3. Wire into `retrieve()` and `reconstruct()`.
4. Run YAML audit first (fastest signal).
5. Only if YAML ≥ 54%: run Core 50, Stress, Banff, DeLores.
6. If any benchmark drops below baseline: stop, diagnose. Do not tune.

No coefficient tuning at any point. If benchmarks fail, the failure is structural — fix the structure, not a weight.
