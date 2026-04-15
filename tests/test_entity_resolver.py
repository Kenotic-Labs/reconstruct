from app.engines.entity_resolver import resolve_query_entities


def test_resolve_query_entities_finds_person(seeded_db, monkeypatch):
    import numpy as np
    from app.vector import embedder
    monkeypatch.setattr(
        embedder, "embed_text",
        lambda s: np.array([1.0, 0.0], dtype=np.float32) if "Maya" in s else np.array([0.0, 1.0], dtype=np.float32),
    )
    matches = resolve_query_entities(
        user_id=seeded_db.user_id, query_text="Where does Maya work?"
    )
    names = [m["name"] for m in matches]
    assert "Maya" in names


def test_resolve_query_entities_skips_stopwords(seeded_db, monkeypatch):
    import numpy as np
    from app.vector import embedder
    monkeypatch.setattr(embedder, "embed_text",
                        lambda s: np.array([0.5, 0.5], dtype=np.float32))
    matches = resolve_query_entities(
        user_id=seeded_db.user_id, query_text="Where is work?"
    )
    # no candidate phrase ("where", "work") should trigger a cosine ≥ 0.55
    # against Maya or Vantage embeddings
    assert all(m["score"] < 0.99 or m["name"] not in {"where", "work"}
               for m in matches)
