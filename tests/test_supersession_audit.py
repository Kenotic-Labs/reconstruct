"""
Supersession audit diagnostic.

Ingests the 20 e2e test conversations, then queries the DB for ALL
relationships (including is_current=0) and reports which edges were
superseded, by whom, and whether the supersession was correct.

Specifically checks that complementary facts extracted from different
source sentences are NOT wrongly superseded by each other.
"""
import os
import sys
import tempfile
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RAYA_EMBED_DEVICE", "cpu")

from sdk.client import Kenotic

# Same conversations as the e2e test
CONVERSATIONS = [
    "I work at Google as a software engineer.",
    "My manager's name is Sarah Chen.",
    "I've been at Google for three years now.",
    "I'm thinking about switching to a product management role.",
    "My girlfriend's name is Mika. We've been together for two years.",
    "Mika works at a startup called Luminary.",
    "My best friend is Derek. We met in college.",
    "Derek just got promoted to VP at his company.",
    "My favorite food is sushi, especially salmon nigiri.",
    "I'm allergic to tree nuts.",
    "I have a dog named Luna. She's a golden retriever.",
    "My birthday is March 15th.",
    "I started training for a marathon last month.",
    "I run about 30 miles per week right now.",
    "My doctor recommended I take vitamin D supplements.",
    "I live in San Francisco, in the Mission district.",
    "My rent is $2800 a month.",
    "I've been living here since 2022.",
    "I graduated from MIT with a computer science degree.",
    "My thesis was on distributed systems.",
]

# Edges that must NOT be superseded. Each is a dict of field substrings
# that identify the edge. If any matching edge has is_current=0 AND
# superseded_by IS NOT NULL, the audit fails.
MUST_NOT_SUPERSEDE = [
    {"predicate": "works_at", "object": "google",
     "label": "(user, works_at, Google)"},
    {"object": "software engineer",
     "label": "(user, *, software engineer)"},
    {"object": "san francisco",
     "label": "(user, *, San Francisco)"},
    {"object": "marathon",
     "label": "(user, *, marathon)"},
    {"object": "30",
     "label": "(user, *, 30 miles)"},
]


def run_audit():
    tmp = tempfile.mkdtemp(prefix="nura_supersession_audit_")
    db_path = os.path.join(tmp, "audit.db")

    print("=" * 70)
    print("  SUPERSESSION AUDIT DIAGNOSTIC")
    print(f"  DB: {db_path}")
    print("=" * 70)

    k = Kenotic(user_id=1, db_path=db_path, embed_device="cpu")

    print(f"\n[1] Ingesting {len(CONVERSATIONS)} utterances...")
    total = 0
    for i, text in enumerate(CONVERSATIONS):
        n = k.ingest(text)
        total += n
        print(f"  [{i+1:2d}] {n} triples <- \"{text[:60]}\"")
    print(f"\n  Total triples stored: {total}")

    # Query ALL relationships including superseded
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, subject, predicate, object, is_current,
                  superseded_by, superseded_at, source_text,
                  edge_schematic_category
           FROM relationships
           WHERE user_id = 1
           ORDER BY id"""
    ).fetchall()

    print(f"\n[2] Total edges in DB: {len(rows)}")
    superseded = [r for r in rows if not r["is_current"]]
    current = [r for r in rows if r["is_current"]]
    print(f"    Current (is_current=1): {len(current)}")
    print(f"    Superseded (is_current=0): {len(superseded)}")

    if superseded:
        print(f"\n[3] Superseded edges:")
        for r in superseded:
            sup_by = r["superseded_by"]
            sup_detail = ""
            if sup_by:
                sup_row = conn.execute(
                    "SELECT subject, predicate, object FROM relationships WHERE id = ?",
                    (sup_by,),
                ).fetchone()
                if sup_row:
                    sup_detail = f" -> ({sup_row['subject']}, {sup_row['predicate']}, {sup_row['object']})"
            print(f"    rid={r['id']:3d} ({r['subject']}, {r['predicate']}, {r['object']})"
                  f"  superseded_by={sup_by}{sup_detail}"
                  f"  cat={r['edge_schematic_category']}")
    else:
        print(f"\n[3] No superseded edges found.")

    # Check MUST_NOT_SUPERSEDE edges
    print(f"\n[4] Checking protected edges...")
    violations = []
    for check in MUST_NOT_SUPERSEDE:
        label = check["label"]
        # Find matching edges
        matching = []
        for r in rows:
            match = True
            for field, val in check.items():
                if field == "label":
                    continue
                actual = (r[field] or "").lower()
                if val.lower() not in actual:
                    match = False
                    break
            if match:
                matching.append(r)

        if not matching:
            print(f"    {label}: NOT FOUND (extraction issue)")
            continue

        wrongly_superseded = [
            r for r in matching
            if not r["is_current"] and r["superseded_by"] is not None
        ]
        if wrongly_superseded:
            for r in wrongly_superseded:
                sup_by = r["superseded_by"]
                sup_row = conn.execute(
                    "SELECT subject, predicate, object FROM relationships WHERE id = ?",
                    (sup_by,),
                ).fetchone()
                sup_detail = ""
                if sup_row:
                    sup_detail = f" by ({sup_row['subject']}, {sup_row['predicate']}, {sup_row['object']})"
                print(f"    VIOLATION: {label} rid={r['id']} wrongly superseded{sup_detail}")
                violations.append({
                    "label": label,
                    "rid": r["id"],
                    "superseded_by": sup_by,
                    "superseder": dict(sup_row) if sup_row else None,
                })
        else:
            status = "CURRENT" if any(r["is_current"] for r in matching) else "MISSING"
            print(f"    OK: {label} ({status})")

    conn.close()

    # Summary
    print(f"\n{'=' * 70}")
    if violations:
        print(f"  FAILED: {len(violations)} wrongly superseded edge(s)")
        for v in violations:
            print(f"    - {v['label']}")
    else:
        print("  PASSED: No wrong supersessions detected")
    print("=" * 70)

    # Cleanup
    try:
        os.remove(db_path)
        os.rmdir(tmp)
    except Exception:
        pass

    return violations


if __name__ == "__main__":
    violations = run_audit()
    sys.exit(1 if violations else 0)
