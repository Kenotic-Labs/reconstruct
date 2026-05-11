import numpy as np
import os
from pathlib import Path

# Backward compatibility functions (for mock mode)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fallback cosine similarity for mock mode."""
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))

def top_k_by_cosine(query_vec: np.ndarray, vectors: list[np.ndarray], k: int) -> list[tuple[int, float]]:
    """Fallback top-k search for mock mode."""
    scores = [(i, cosine_similarity(query_vec, v)) for i, v in enumerate(vectors)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


class FAISSVectorIndex:
    """FAISS-based vector index with disk persistence."""

    def __init__(self, dimension: int, index_path: str):
        """
        Initialize FAISS index.

        Args:
            dimension: Vector dimension (384 for all-MiniLM-L6-v2)
            index_path: Path to save/load index file
        """
        import faiss

        self.dimension = dimension
        self.index_path = index_path
        self.index = None

        # Try to load existing index
        if os.path.exists(index_path):
            try:
                self.load()
            except Exception:
                # If load fails, create fresh index
                self.index = faiss.IndexFlatIP(dimension)
        else:
            # Create fresh index (Inner Product for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)

    def add_vectors(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: numpy array of shape [n, dimension], dtype float32
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Ensure float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Add to index
        self.index.add(vectors)

        # Auto-save after adding
        self.save()

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """
        Search for top-k nearest neighbors.

        Args:
            query_vector: Query embedding (1D array)
            k: Number of results

        Returns:
            List of (index, score) tuples
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Ensure float32
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        # Search
        k = min(k, self.index.ntotal) if self.index.ntotal > 0 else 0
        if k == 0:
            return []

        distances, indices = self.index.search(query_vector, k)

        # Convert to list of tuples
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], distances[0])]
        return results

    def save(self) -> None:
        """Save index to disk."""
        import faiss

        # Ensure directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        # Write index
        faiss.write_index(self.index, self.index_path)

    def load(self) -> None:
        """Load index from disk."""
        import faiss

        self.index = faiss.read_index(self.index_path)
