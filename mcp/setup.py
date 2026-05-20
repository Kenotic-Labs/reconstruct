"""
First-run setup for Reconstruct.

Downloads required models and creates the database on first
`reconstruct serve`. Runs once — subsequent starts skip everything
that's already present. All downloads are local to the venv/machine.

Called from mcp.cli before starting either transport.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_ready(db_path: str | None = None) -> None:
    """Ensure all dependencies are available before serving.

    Order matters: spaCy and NLTK are needed by the grammar engine,
    embeddings are needed by ingest, DB is needed by everything.
    """
    _ensure_spacy_model()
    _ensure_nltk_data()
    _ensure_embedding_model()
    _ensure_database(db_path)


def _ensure_spacy_model() -> None:
    """Download en_core_web_md if not installed."""
    import spacy
    model = "en_core_web_md"
    try:
        spacy.load(model)
    except OSError:
        print(f"[reconstruct] Downloading spaCy model '{model}'...", file=sys.stderr)
        from spacy.cli import download
        download(model)
        # Verify it loads after download
        spacy.load(model)
        print(f"[reconstruct] spaCy model '{model}' ready.", file=sys.stderr)


def _ensure_nltk_data() -> None:
    """Download WordNet + Open Multilingual Wordnet if not present."""
    import nltk
    for resource in ("wordnet", "omw-1.4"):
        found = False
        for suffix in (resource, f"{resource}.zip"):
            try:
                nltk.data.find(f"corpora/{suffix}")
                found = True
                break
            except LookupError:
                continue
        if not found:
            print(f"[reconstruct] Downloading NLTK '{resource}'...", file=sys.stderr)
            nltk.download(resource, quiet=True)
            print(f"[reconstruct] NLTK '{resource}' ready.", file=sys.stderr)


def _ensure_embedding_model() -> None:
    """Pre-download MiniLM so first ingest isn't blocked."""
    model_name = "all-MiniLM-L6-v2"
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check if already cached without loading the full model
        result = try_to_load_from_cache(f"sentence-transformers/{model_name}", "config.json")
        if result is not None:
            return  # already downloaded
    except Exception:
        pass

    try:
        print(f"[reconstruct] Downloading embedding model '{model_name}'...", file=sys.stderr)
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(model_name)
        print(f"[reconstruct] Embedding model '{model_name}' ready.", file=sys.stderr)
    except Exception as e:
        print(f"[reconstruct] Warning: could not pre-load embedding model: {e}", file=sys.stderr)


def _ensure_database(db_path: str | None = None) -> None:
    """Create ~/.kenotic/memory.db with schema if it doesn't exist."""
    db_path = db_path or os.environ.get(
        "KENOTIC_DB_PATH",
        os.path.expanduser("~/.kenotic/memory.db"),
    )
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    import sqlite3
    from app.db.models import MIGRATIONS, run_schema_upgrades
    conn = sqlite3.connect(db_path)
    conn.executescript(MIGRATIONS)
    run_schema_upgrades(conn)
    conn.commit()
    conn.close()
    print(f"[reconstruct] Database ready at {db_path}", file=sys.stderr)
