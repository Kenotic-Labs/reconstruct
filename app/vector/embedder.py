import hashlib
import os
import numpy as np
from config.settings import settings

# Global model cache for sentence-transformers
_model_cache = None

def _get_sentence_transformer():
    """Load and cache sentence-transformers model (GPU-accelerated if available)."""
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = os.environ.get('RAYA_EMBED_DEVICE', 'cuda')
        _model_cache = SentenceTransformer(settings.embedding_model, device=device)
        print(f"[EmbeddingService] Loaded {settings.embedding_model} on GPU (CUDA)")

    return _model_cache

def _stable_hash(text: str) -> int:
    """Hash function for deterministic mock embeddings."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)

def _embed_text_hash(text: str, dim: int) -> np.ndarray:
    """V1: deterministic hash-based embedding (for tests/mock mode)."""
    rng = np.random.default_rng(_stable_hash(text))
    v = rng.normal(size=(dim,)).astype("float32")
    # normalize
    n = np.linalg.norm(v) + 1e-9
    return v / n

def _embed_text_real(text: str) -> np.ndarray:
    """V2: sentence-transformers embedding (production mode)."""
    model = _get_sentence_transformer()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype("float32")

def embed_text(text: str, dim: int | None = None) -> np.ndarray:
    """
    Embed text using either real embeddings (sentence-transformers) or mock (hash-based).

    Mode determined by settings.use_real_embeddings:
    - True: Use sentence-transformers "all-MiniLM-L6-v2"
    - False: Use hash-based embeddings (for tests)
    """
    if settings.use_real_embeddings:
        return _embed_text_real(text)
    else:
        dim = dim or settings.embedding_dim
        return _embed_text_hash(text, dim)
