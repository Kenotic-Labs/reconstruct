"""
Model Harness — binds model weights + role + preprocessing + postprocessing.

Every model in the continuity pipeline has a specific role.
The harness makes the role explicit and inseparable from the model.

Usage:
    from app.models.harness import get_embedder, get_reranker, get_corrector, get_topic_matcher

    # Each returns a harness that knows its role
    embedder = get_embedder()
    vector = embedder.encode("Melanie ran a charity race")

    reranker = get_reranker()
    score = reranker.score("When did Melanie run?", "Melanie ran a charity race last Saturday")
"""
import os
import logging
import numpy as np
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)

# ============================================================================
# BASE HARNESS
# ============================================================================

class ModelHarness:
    """Base class. Every model harness declares its role."""

    ROLE: str = ""          # What this model does in the pipeline
    MODEL_ID: str = ""      # HuggingFace model ID or local path
    DEVICE: str = "cuda:0"  # Where to run

    def __init__(self):
        self._model = None
        self._loaded = False

    def _load(self):
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return self._loaded


# ============================================================================
# 1. EMBEDDER — Encodes text into vectors for memory storage & retrieval
# ============================================================================

class EmbedderHarness(ModelHarness):
    """
    ROLE: Encode text into 384-dim vectors for semantic similarity search.

    Used at WRITE time to embed source_text and predicates into edges.
    Used at READ time to compare query embeddings against stored edges.

    This model encodes for MEMORY RETRIEVAL — not generic semantic search.
    The vectors must capture FACTUAL CONTENT (who did what, when, where)
    over stylistic similarity. Two sentences about the same fact should
    have high cosine even if phrased differently.

    Model: all-MiniLM-L6-v2 (22MB, 384-dim)
    Pre-processing: none (raw text)
    Post-processing: L2 normalize
    """

    ROLE = "memory_embedder"
    MODEL_ID = "all-MiniLM-L6-v2"

    def _load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            device = os.environ.get('RAYA_EMBED_DEVICE', 'cuda:0')
            self._model = SentenceTransformer(self.MODEL_ID, device=device)
            self._loaded = True
            log.info("[%s] Loaded %s on %s", self.ROLE, self.MODEL_ID, device)
        except Exception as e:
            log.warning("[%s] Failed to load: %s", self.ROLE, e)

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a normalized 384-dim vector."""
        self._load()
        if not self._model:
            # Fallback: deterministic hash embedding
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little", signed=False))
            v = rng.normal(size=(384,)).astype("float32")
            return v / (np.linalg.norm(v) + 1e-9)
        return self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encode — same preprocessing, more efficient."""
        self._load()
        if not self._model:
            return np.array([self.encode(t) for t in texts])
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
        ).astype("float32")


# ============================================================================
# 2. RERANKER — Scores (query, passage) pairs for answer relevance
# ============================================================================

class RerankerHarness(ModelHarness):
    """
    ROLE: Score how well an edge's source_text ANSWERS a query.

    This is NOT document retrieval relevance. This is ANSWER relevance.
    "Does this edge contain the fact that answers this question?"

    Input: (query, source_text) pairs
    Output: relevance score (higher = better answer)

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22MB)
    Pre-processing: pair as (query, passage) — order matters
    Post-processing: raw logits (not probabilities) — use for ranking only

    Note: ms-marco outputs raw logits, not 0-1 probabilities.
    A score of 5.0 is good, -5.0 is bad. Only use for relative ranking.
    """

    ROLE = "answer_reranker"
    MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.MODEL_ID, device="cuda:0")
            self._loaded = True
            log.info("[%s] Loaded %s", self.ROLE, self.MODEL_ID)
        except Exception as e:
            log.warning("[%s] Failed to load: %s", self.ROLE, e)

    def score(self, query: str, passage: str) -> float:
        """Score a single (query, passage) pair."""
        self._load()
        if not self._model:
            return 0.0
        return float(self._model.predict([(query, passage)])[0])

    def score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score multiple (query, passage) pairs in one forward pass."""
        self._load()
        if not self._model:
            return [0.0] * len(pairs)
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


# ============================================================================
# 3. TOPIC MATCHER — Category-aware semantic matching for inference
# ============================================================================

class TopicMatcherHarness(ModelHarness):
    """
    ROLE: Match query TOPICS to edge OBJECTS for inference questions.

    Used for Cat 3 open-domain inference: "Would Melanie enjoy Vivaldi?"
    Needs to know that Vivaldi ↔ Bach are in the same category (classical music).
    MiniLM gives Vivaldi↔Bach = 0.45. This model gives 0.82.

    ONLY used for inference queries. NOT for stored embeddings.
    Runs on CPU (small enough, keeps GPU for main embedder).

    Model: gte-small (70MB, MTEB clustering 44.89)
    Pre-processing: none
    Post-processing: L2 normalize
    """

    ROLE = "topic_matcher"
    MODEL_ID = "thenlper/gte-small"
    DEVICE = "cpu"

    def _load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_ID, device=self.DEVICE)
            self._loaded = True
            log.info("[%s] Loaded %s on %s", self.ROLE, self.MODEL_ID, self.DEVICE)
        except Exception as e:
            log.warning("[%s] Failed to load: %s", self.ROLE, e)

    def encode(self, text: str) -> np.ndarray:
        """Encode for topic-level similarity."""
        self._load()
        if not self._model:
            # Fallback to main embedder
            return get_embedder().encode(text)
        return self._model.encode(text, normalize_embeddings=True).astype("float32")

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts at topic level."""
        ea = self.encode(text_a)
        eb = self.encode(text_b)
        return float(np.dot(ea, eb))


# ============================================================================
# 4. GRAMMAR CORRECTOR — Fixes STT errors without changing meaning
# ============================================================================

class CorrectorHarness(ModelHarness):
    """
    ROLE: Fix grammar errors from speech-to-text output.

    This model CORRECTS, it does NOT rewrite or restructure.
    "I goed to store" → "I went to the store"
    "She dont like it" → "She doesn't like it"

    It must NOT change meaning. If uncertain, keep the original.

    Model: GECToR (gotutiyan/gector-roberta-base-5k, 128M params)
    Pre-processing: spaCy sentence splitting
    Post-processing: discourse frame stripping

    Config (bound to model — do not separate):
      n_iteration=5:      up to 5 correction passes
      min_error_prob=0.0:  apply all corrections
      keep_confidence=0.0: no bias toward original
      batch_size=128:      batch GPU forward passes
    """

    ROLE = "grammar_corrector"
    MODEL_ID = "models/gector-raya"

    # Bound config — inseparable from the model
    N_ITERATION = 5
    MIN_ERROR_PROB = 0.0
    KEEP_CONFIDENCE = 0.0
    BATCH_SIZE = 128

    def _load(self):
        if self._loaded:
            return
        try:
            import json as _json
            import torch
            from gector import GECToR
            from transformers import AutoTokenizer

            # Resolve vocab dir relative to project root
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            _vocab_dir = os.path.join(_project_root, self.MODEL_ID)

            # Load model + tokenizer (bound unit)
            model = GECToR.from_pretrained(self.MODEL_ID, local_files_only=True)
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, local_files_only=True)

            # Load encode/decode vocabs (bound to this specific model)
            with open(os.path.join(_vocab_dir, "encode_vocab.json")) as f:
                self._encode = _json.load(f)
            with open(os.path.join(_vocab_dir, "decode_vocab.json")) as f:
                self._decode = _json.load(f)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._model = model.to(device).eval()
            self._device = device
            self._loaded = True

            params = sum(p.numel() for p in model.parameters()) / 1e6
            log.info("[%s] Loaded %s (%.0fM params) on %s",
                     self.ROLE, self.MODEL_ID, params, device)
        except Exception as e:
            log.warning("[%s] Failed to load: %s", self.ROLE, e)

    def correct(self, sentences: List[str]) -> List[str]:
        """Correct grammar in a batch of sentences."""
        self._load()
        if not self._model:
            return sentences
        from gector import predict as gector_predict
        return gector_predict(
            self._model, self._tokenizer, sentences,
            self._encode, self._decode,
            keep_confidence=self.KEEP_CONFIDENCE,
            min_error_prob=self.MIN_ERROR_PROB,
            n_iteration=self.N_ITERATION,
            batch_size=self.BATCH_SIZE,
        )


# ============================================================================
# SINGLETON ACCESS — one harness per model, loaded once
# ============================================================================

_embedder: Optional[EmbedderHarness] = None
_reranker: Optional[RerankerHarness] = None
_topic_matcher: Optional[TopicMatcherHarness] = None
_corrector: Optional[CorrectorHarness] = None


def get_embedder() -> EmbedderHarness:
    global _embedder
    if _embedder is None:
        _embedder = EmbedderHarness()
    return _embedder


def get_reranker() -> RerankerHarness:
    global _reranker
    if _reranker is None:
        _reranker = RerankerHarness()
    return _reranker


def get_topic_matcher() -> TopicMatcherHarness:
    global _topic_matcher
    if _topic_matcher is None:
        _topic_matcher = TopicMatcherHarness()
    return _topic_matcher


def get_corrector() -> CorrectorHarness:
    global _corrector
    if _corrector is None:
        _corrector = CorrectorHarness()
    return _corrector
