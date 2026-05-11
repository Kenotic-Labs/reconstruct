"""
Backfill migration: re-embed all edge_embeddings and predicted_queries
using natural first-person surface forms.

Edge embeddings:  "user works at Google" → "I work at Google"
Predicted queries: "What does user work at?" → "Where do I work?"

Runs against the live SQLite DB. Idempotent (can re-run safely).
"""
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.db.session import init_db, get_db_context
from app.vector.embedder import embed_text
from config.settings import settings


def _lemmatize_verb(verb: str) -> str:
    """Lemmatize a verb for first-person agreement: works→work, has→have."""
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer().lemmatize(verb, pos="v")
    except Exception:
        return verb


def _build_edge_surface(subject: str, predicate: str, object_: str) -> str:
    """Build natural-language surface text for edge embedding.
    Mirrors the Fix 3 logic in memory.py _write_edge_traces()."""
    if subject.lower() == "user":
        segments = predicate.split("_")
        segments[0] = _lemmatize_verb(segments[0])
        pred_surface = " ".join(segments)
        return f"I {pred_surface} {object_}"
    else:
        pred_natural = predicate.replace("_", " ")
        return f"{subject} {pred_natural} {object_}"


def migrate_edge_embeddings():
    """Re-embed all relationships.edge_embedding with natural surface forms."""
    print("=" * 60)
    print("  Phase 1: Re-embedding edge_embedding")
    print("=" * 60)

    with get_db_context() as conn:
        rows = conn.execute(
            "SELECT id, subject, predicate, object FROM relationships"
        ).fetchall()

    total = len(rows)
    print(f"  Found {total} relationships to re-embed")

    done = 0
    errors = 0
    t0 = time.perf_counter()

    for row in rows:
        rid = row["id"]
        subject = row["subject"] or ""
        predicate = row["predicate"] or ""
        object_ = row["object"] or ""

        if not subject or not predicate:
            continue

        surface = _build_edge_surface(subject, predicate, object_)
        try:
            emb = embed_text(surface)
            emb_blob = emb.tobytes()
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR embedding rid={rid}: {e}")
            continue

        with get_db_context() as conn:
            conn.execute(
                "UPDATE relationships SET edge_embedding = ? WHERE id = ?",
                (emb_blob, rid),
            )

        done += 1
        if done % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{total} edges re-embedded ({rate:.0f}/sec)")

    elapsed = time.perf_counter() - t0
    print(f"  Done: {done} re-embedded, {errors} errors, {elapsed:.1f}s")
    return done


def migrate_predicted_queries():
    """Delete all existing PQs and regenerate with natural first-person forms."""
    print()
    print("=" * 60)
    print("  Phase 2: Regenerating predicted_queries")
    print("=" * 60)

    with get_db_context() as conn:
        rows = conn.execute(
            """SELECT id, user_id, subject, predicate, object,
                      subject_type, object_type, source_text
               FROM relationships"""
        ).fetchall()

    total = len(rows)
    print(f"  Found {total} relationships to regenerate PQs for")

    # Import the updated PQ generator
    from app.engines.predicted_queries import generate_predicted_queries

    # Clear all existing PQs
    with get_db_context() as conn:
        count = conn.execute("SELECT COUNT(*) FROM predicted_queries").fetchone()[0]
        conn.execute("DELETE FROM predicted_queries")
        print(f"  Deleted {count} old predicted queries")

    done = 0
    pqs_written = 0
    errors = 0
    t0 = time.perf_counter()

    for row in rows:
        rid = row["id"]
        user_id = row["user_id"]
        subject = row["subject"] or ""
        predicate = row["predicate"] or ""
        object_ = row["object"] or ""
        subject_type = row["subject_type"] or ""
        object_type = row["object_type"] or ""
        source_text = row["source_text"] or f"{subject} {predicate.replace('_', ' ')} {object_}"

        if not subject or not predicate:
            continue

        try:
            pairs = generate_predicted_queries(
                subject, predicate, object_, subject_type, object_type
            )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR generating PQs for rid={rid}: {e}")
            continue

        if not pairs:
            done += 1
            continue

        with get_db_context() as conn:
            for question, emb in pairs:
                try:
                    emb_blob = emb.tobytes() if emb is not None else None
                except Exception:
                    emb_blob = None
                if emb_blob is None:
                    continue
                conn.execute(
                    """INSERT INTO predicted_queries
                       (user_id, relationship_id, predicted_question,
                        question_embedding, answer_text)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, rid, question, emb_blob, source_text),
                )
                pqs_written += 1

        done += 1
        if done % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{total} relationships processed ({rate:.0f}/sec)")

    elapsed = time.perf_counter() - t0
    print(f"  Done: {done} processed, {pqs_written} PQs written, {errors} errors, {elapsed:.1f}s")
    return pqs_written


def main():
    print("Natural Embeddings Backfill Migration")
    print("=" * 60)

    init_db(settings.sqlite_path)

    edges = migrate_edge_embeddings()
    pqs = migrate_predicted_queries()

    print()
    print("=" * 60)
    print(f"  MIGRATION COMPLETE")
    print(f"  Edge embeddings re-embedded: {edges}")
    print(f"  Predicted queries regenerated: {pqs}")
    print("=" * 60)


if __name__ == "__main__":
    main()
