"""
End-to-end retrieval accuracy test via SDK path.

Ingests real-world-style conversations, then queries and checks
if the retrieval engine returns the right answer.

Tests the FULL architecture: SDK → engines → DB → retrieval.
No product layer, no backbone, no LLM.

Reports 5 metrics separately:
  0. T5 emission — did T5 produce a triple containing the expected fact?
  1. Extraction coverage — was the expected fact stored as a triple?
  2. Type correctness — are subject_type/object_type correct?
  3. Top-1 edge retrieval — is the winning edge the right one?
  4. Final answer correctness — does the rendered text contain the keyword?
"""
import os
import sys
import tempfile
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RAYA_EMBED_DEVICE", "cpu")

from sdk.client import Kenotic
from app.engines.retrieval import Answer, StructuralRefusal
from app.engines.memory import MemoryEngine


# ── L0 emission cache: map conversation text → raw T5 triples ─
# Built once during ingest, used by L0 checks per query.
_l0_emission_cache: dict = {}  # conversation_text -> list of (s, p, o, is_hist)


def _build_l0_cache(conversations: list):
    """Run T5 extract() directly on each conversation (no validation)
    and cache the raw triples. This tells us what T5 saw vs what
    validation rejected."""
    global _l0_emission_cache
    me = MemoryEngine()
    for text in conversations:
        cleaned = me.clean(text)
        triples = me.extract(cleaned)
        _l0_emission_cache[text] = triples


def _check_l0_emission(conversation_texts: list, expected_triple: dict) -> bool:
    """Layer 0: Check if T5 emitted a triple containing the expected fact
    from any conversation. Runs extract() without validation.
    Returns True if any emitted triple matches the expected fields."""
    for text in conversation_texts:
        triples = _l0_emission_cache.get(text, [])
        for s, p, o, _hist in triples:
            match = True
            for field, value in expected_triple.items():
                if field == "subject":
                    if value.lower() not in s.lower():
                        match = False
                        break
                elif field == "predicate":
                    if value.lower() not in p.lower():
                        match = False
                        break
                elif field == "object":
                    if value.lower() not in o.lower():
                        match = False
                        break
            if match:
                return True
    return False


# ── Test data: real-world conversations ──────────────────────

CONVERSATIONS = [
    # Career / work
    "I work at Google as a software engineer.",
    "My manager's name is Sarah Chen.",
    "I've been at Google for three years now.",
    "I'm thinking about switching to a product management role.",

    # Relationships
    "My girlfriend's name is Mika. We've been together for two years.",
    "Mika works at a startup called Luminary.",
    "My best friend is Derek. We met in college.",
    "Derek just got promoted to VP at his company.",

    # Personal
    "My favorite food is sushi, especially salmon nigiri.",
    "I'm allergic to tree nuts.",
    "I have a dog named Luna. She's a golden retriever.",
    "My birthday is March 15th.",

    # Health
    "I started training for a marathon last month.",
    "I run about 30 miles per week right now.",
    "My doctor recommended I take vitamin D supplements.",

    # Living situation
    "I live in San Francisco, in the Mission district.",
    "My rent is $2800 a month.",
    "I've been living here since 2022.",

    # Education
    "I graduated from MIT with a computer science degree.",
    "My thesis was on distributed systems.",
]

# ── Queries and expected facts ───────────────────────────────
# Each tuple: (query, list_of_keywords_that_must_appear_in_answer)
# Keywords are case-insensitive substring matches.

# Each entry: (query, answer_keywords, expected_triple, expected_types)
#   answer_keywords: list of keywords that must appear in the final answer
#   expected_triple: dict with keys from {subject, predicate, object} —
#       each value is a case-insensitive substring that must appear in the
#       stored triple's field. Used for extraction coverage (layer 1) and
#       top-1 edge check (layer 3).
#   expected_types: dict with optional subject_type / object_type expected
#       values for type correctness (layer 2). None = don't check.

QUERIES = [
    # Direct factual
    ("Where do I work?", ["google"],
     {"object": "google"}, {"object_type": "ORG"}),
    ("What is my job?", ["software engineer"],
     {"object": "software engineer"}, {}),
    ("Who is my manager?", ["sarah"],
     {"object": "sarah"}, {"object_type": "PERSON"}),
    ("What is my favorite food?", ["sushi"],
     {"object": "sushi"}, {}),
    ("What am I allergic to?", ["tree nut", "nut"],
     {"object": "nut"}, {}),
    ("What is my dog's name?", ["luna"],
     {"object": "luna"}, {}),
    ("When is my birthday?", ["march", "15"],
     {"object": "march"}, {"object_type": "TIME"}),
    ("Where do I live?", ["san francisco"],
     {"object": "san francisco"}, {"object_type": "LOCATION"}),

    # Relationship queries
    ("Who is Mika?", ["girlfriend"],
     {"subject": "mika", "object": "girlfriend"}, {"subject_type": "PERSON"}),
    ("Where does Mika work?", ["luminary"],
     {"subject": "mika", "object": "luminary"}, {"object_type": "ORG"}),
    ("Who is Derek?", ["friend"],
     {"subject": "derek"}, {"subject_type": "PERSON"}),
    ("Who is my best friend?", ["derek"],
     {"object": "derek"}, {"object_type": "PERSON"}),

    # Activity queries
    ("What am I training for?", ["marathon"],
     {"object": "marathon"}, {}),
    ("How much do I run?", ["30", "miles"],
     {"object": "30"}, {}),

    # Education
    ("Where did I go to school?", ["mit"],
     {"object": "mit"}, {"object_type": "ORG"}),
    ("What did I study?", ["computer science"],
     {"object": "computer science"}, {}),

    # Temporal / status
    ("How long have I been at Google?", ["three years", "3 year"],
     {"object": "three year"}, {}),
    ("How long have Mika and I been together?", ["two years", "2 year"],
     {"object": "two year"}, {}),
]


def _check_extraction(db_path, expected_triple):
    """Layer 1: Check if the expected triple was stored in the DB.
    Returns (found: bool, matching_row: dict or None)."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Build WHERE clause from expected_triple fields
        where_parts = ["user_id = 1", "COALESCE(is_current, 1) = 1",
                       "tombstoned_at IS NULL"]
        params = []
        for field, value in expected_triple.items():
            where_parts.append(f"LOWER({field}) LIKE ?")
            params.append(f"%{value.lower()}%")
        sql = f"SELECT * FROM relationships WHERE {' AND '.join(where_parts)} LIMIT 1"
        row = conn.execute(sql, params).fetchone()
        conn.close()
        if row:
            return True, dict(row)
        return False, None
    except Exception as e:
        return False, None


def _check_types(stored_row, expected_types):
    """Layer 2: Check if subject_type/object_type are correct.
    Returns (checked: int, correct: int, details: list)."""
    if not stored_row or not expected_types:
        return 0, 0, []
    checked = 0
    correct = 0
    details = []
    for field, expected_val in expected_types.items():
        actual = (stored_row.get(field) or "").strip()
        checked += 1
        if actual.upper() == expected_val.upper():
            correct += 1
            details.append(f"{field}: {actual} (correct)")
        else:
            details.append(f"{field}: {actual} (expected {expected_val})")
    return checked, correct, details


def run_test():
    # Create a fresh temp DB for isolation
    tmp = tempfile.mkdtemp(prefix="nura_test_")
    db_path = os.path.join(tmp, "test.db")

    print("=" * 70)
    print("  RETRIEVAL E2E TEST -- SDK > Engines > DB")
    print(f"  DB: {db_path}")
    print("=" * 70)

    # Initialize SDK
    print("\n[1] Initializing SDK...")
    k = Kenotic(user_id=1, db_path=db_path, embed_device="cpu")

    # Build L0 emission cache BEFORE ingest (uses extract() directly)
    print("\n[1b] Building L0 emission cache (T5 extract without validation)...")
    _build_l0_cache(CONVERSATIONS)

    # Ingest all conversations
    print(f"\n[2] Ingesting {len(CONVERSATIONS)} utterances...")
    total_triples = 0
    for i, text in enumerate(CONVERSATIONS):
        n = k.ingest(text)
        total_triples += n
        print(f"  [{i+1:2d}] {n} triples <- \"{text[:60]}\"")

    print(f"\n  Total triples stored: {total_triples}")

    # ── Layer-by-layer evaluation ────────────────────────────────
    print(f"\n[3] Running {len(QUERIES)} queries with 5-layer evaluation...")

    # Counters for each layer
    l0_emitted = 0     # Layer 0: T5 emitted the triple
    l1_extracted = 0    # Layer 1: extraction coverage
    l2_checked = 0      # Layer 2: type checks attempted
    l2_correct = 0      # Layer 2: type checks passed
    l3_edge_correct = 0 # Layer 3: top-1 edge has right triple
    l4_answer_correct = 0  # Layer 4: final answer has keyword
    total = len(QUERIES)
    results = []

    for query, expected_keywords, expected_triple, expected_types in QUERIES:
        print(f"\n  Q: \"{query}\"")

        # Layer 0: T5 emission (before validation)
        emitted = _check_l0_emission(CONVERSATIONS, expected_triple)
        l0_emitted += int(emitted)
        l0_mark = "+" if emitted else "X"
        print(f"    L0 T5 emitted:  {l0_mark} {'yes' if emitted else 'NOT EMITTED by T5'}")

        # Layer 1: Extraction coverage
        extracted, stored_row = _check_extraction(db_path, expected_triple)
        l1_extracted += int(extracted)
        l1_mark = "+" if extracted else "X"
        print(f"    L1 extraction:  {l1_mark} {'found' if extracted else 'MISSING'}", end="")
        if stored_row:
            print(f"  ({stored_row.get('subject','?')}, {stored_row.get('predicate','?')}, {stored_row.get('object','?')})")
        else:
            print(f"  (expected {expected_triple})")

        # Layer 2: Type correctness
        t_checked, t_correct, t_details = _check_types(stored_row, expected_types)
        l2_checked += t_checked
        l2_correct += t_correct
        if t_details:
            l2_mark = "+" if t_checked == t_correct else "X"
            print(f"    L2 types:       {l2_mark} {', '.join(t_details)}")

        # Retrieve
        result = k.retrieve(query)

        # Layer 3: Top-1 edge retrieval
        edge_match = False
        if isinstance(result, Answer) and result.text:
            # Check if the winning edge's triple matches expected
            for field, value in expected_triple.items():
                actual = getattr(result, field, "") or ""
                if value.lower() in actual.lower():
                    edge_match = True
                    break
            # Also check object field specifically (most common)
            if not edge_match and "object" in expected_triple:
                if expected_triple["object"].lower() in (result.object or "").lower():
                    edge_match = True
        l3_edge_correct += int(edge_match)
        l3_mark = "+" if edge_match else "X"
        if isinstance(result, Answer):
            print(f"    L3 top-1 edge:  {l3_mark} ({result.subject}, {result.predicate}, {result.object})")
        else:
            print(f"    L3 top-1 edge:  X (refused: {getattr(result, 'reason', '?')})")

        # Layer 4: Final answer correctness
        answer_match = False
        answer_text = ""
        if isinstance(result, Answer) and result.text:
            answer_lower = result.text.lower()
            answer_match = any(kw.lower() in answer_lower for kw in expected_keywords)
            answer_text = result.text[:120]
        elif isinstance(result, StructuralRefusal):
            answer_text = f"[{result.reason}]"
        l4_answer_correct += int(answer_match)
        l4_mark = "+" if answer_match else "X"
        print(f"    L4 answer:      {l4_mark} \"{answer_text}\"")

        results.append({
            "query": query,
            "l0": emitted,
            "l1": extracted,
            "l2_checked": t_checked, "l2_correct": t_correct,
            "l3": edge_match,
            "l4": answer_match,
        })

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  LAYER-BY-LAYER RESULTS (5-LAYER)")
    print("=" * 70)

    l0_rate = (l0_emitted / total * 100) if total > 0 else 0
    l1_rate = (l1_extracted / total * 100) if total > 0 else 0
    l2_rate = (l2_correct / l2_checked * 100) if l2_checked > 0 else 0
    l3_rate = (l3_edge_correct / total * 100) if total > 0 else 0
    l4_rate = (l4_answer_correct / total * 100) if total > 0 else 0

    print(f"  L0 T5 emission:           {l0_emitted}/{total} ({l0_rate:.0f}%)")
    print(f"  L1 Extraction coverage:   {l1_extracted}/{total} ({l1_rate:.0f}%)")
    print(f"  L2 Type correctness:      {l2_correct}/{l2_checked} ({l2_rate:.0f}%)")
    print(f"  L3 Top-1 edge retrieval:  {l3_edge_correct}/{total} ({l3_rate:.0f}%)")
    print(f"  L4 Final answer correct:  {l4_answer_correct}/{total} ({l4_rate:.0f}%)")
    print("=" * 70)

    # Pinpoint: L0 pass but L1 fail = validation killed it
    l0_pass_l1_fail = [r for r in results if r["l0"] and not r["l1"]]
    if l0_pass_l1_fail:
        print(f"\n  L0->L1 DROP (T5 emitted but validation rejected):")
        for r in l0_pass_l1_fail:
            print(f"    - \"{r['query']}\"")

    l0_fail = [r for r in results if not r["l0"]]
    if l0_fail:
        print(f"\n  L0 FAIL (T5 never emitted the triple):")
        for r in l0_fail:
            print(f"    - \"{r['query']}\"")

    # Show which queries failed at each layer
    for layer_name, layer_key in [("L1 extraction", "l1"),
                                   ("L3 top-1 edge", "l3"),
                                   ("L4 answer", "l4")]:
        failures = [r for r in results if not r[layer_key]]
        if failures:
            print(f"\n  {layer_name} failures:")
            for r in failures:
                print(f"    - \"{r['query']}\"")

    # Cleanup
    try:
        os.remove(db_path)
        os.rmdir(tmp)
    except Exception:
        pass

    return l4_rate


if __name__ == "__main__":
    rate = run_test()
    sys.exit(0 if rate >= 90 else 1)
