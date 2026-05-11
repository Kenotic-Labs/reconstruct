from __future__ import annotations
import numpy as np
from app.vector.embedder import embed_text
from app.vector.vector_index import FAISSVectorIndex
from config.settings import settings

class EmbeddingService:
    """
    Embedding service with optional FAISS index integration.

    V1: Hash-based embedder (mock mode)
    V2: Sentence-transformers + FAISS (real mode)
    """

    def __init__(self):
        """Initialize embedding service with optional FAISS index."""
        self.index = None

        # Initialize FAISS index only in real embeddings mode
        if settings.use_real_embeddings:
            # Determine dimension from model (384 for all-MiniLM-L6-v2)
            self.index = FAISSVectorIndex(
                dimension=384,
                index_path=settings.vector_index_path
            )

    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        return embed_text(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Batch embed texts (V1: no optimization)."""
        return [self.embed(t) for t in texts]

    def add_to_index(self, vectors: list[np.ndarray]) -> None:
        """
        Add vectors to FAISS index (real mode only).

        Args:
            vectors: List of embedding vectors to add
        """
        if self.index is not None:
            vectors_array = np.array(vectors, dtype=np.float32)
            self.index.add_vectors(vectors_array)

    def search_index(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """
        Search FAISS index for top-k similar vectors (real mode only).

        Args:
            query_vector: Query embedding
            k: Number of results

        Returns:
            List of (index, score) tuples
        """
        if self.index is not None:
            return self.index.search(query_vector, k)
        else:
            return []


# Singleton instance
_embedding_service_instance = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the singleton EmbeddingService instance."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
