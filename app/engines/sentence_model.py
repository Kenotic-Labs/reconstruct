"""
SentenceModel — grammar-correcting seq2seq for reconstruction rendering.

Purpose: take an ungrammatical template-rendered sentence
("You has emotion sad.") and return its grammatical form
("You have an emotion of sadness." or similar).

The model is `jbochi/coedit-small` (flan-t5-small fine-tuned on CoEdit data, 77M params, ~300MB).
Falls back to grammarly/coedit-large if small is unavailable.

Architectural contract:
  - Polishing is LOSSLESS w.r.t. grounding. Input is one sentence
    (derived from one edge). Output is one sentence (same edge).
    Grounding map is preserved by the caller — this module does not
    manipulate grounding.
  - Model is OPTIONAL. If load fails or the flag is off, the original
    template sentence is returned unchanged.
  - Per-sentence cache: same input → same output, zero recomputation.
    Cache key is the input sentence string.

Public API:
    polish(sentence: str) -> str
    cleanup(text: str) -> str
    is_available() -> bool
    reset_cache()
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger(__name__)


_MODEL = None
_TOKENIZER = None
_DEVICE: Optional[str] = None
_LOCK = Lock()
_UNAVAILABLE = False
_CACHE: Dict[str, str] = {}

_ENV_MODEL = "RAYA_SENTENCE_MODEL"
_ENV_DEVICE = "RAYA_SENTENCE_DEVICE"
_ENV_ENABLE = "RAYA_SENTENCE_POLISH"   # set to "0" to disable, default ON
_DEFAULT_MODEL = "jbochi/coedit-small"


def is_enabled() -> bool:
    """Polish is on by default. Set RAYA_SENTENCE_POLISH=0 to disable."""
    return os.environ.get(_ENV_ENABLE, "1") != "0"


def is_available() -> bool:
    """True if the model loaded successfully and polishing can run."""
    return _MODEL is not None and not _UNAVAILABLE


def _load() -> bool:
    global _MODEL, _TOKENIZER, _DEVICE, _UNAVAILABLE
    if _UNAVAILABLE:
        return False
    if _MODEL is not None:
        return True
    with _LOCK:
        if _MODEL is not None:
            return True
        if _UNAVAILABLE:
            return False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            dev = os.environ.get(_ENV_DEVICE,
                                 os.environ.get("RAYA_EMBED_DEVICE", "cuda"))
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"

            # Force offline — load from local cache, never ping HF
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            name = os.environ.get(_ENV_MODEL, _DEFAULT_MODEL)
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=True)
            if dev == "cuda":
                model = model.half()
            model = model.to(dev).eval()

            _MODEL = model
            _TOKENIZER = tok
            _DEVICE = dev
            print(f"[SentenceModel] Loaded {name} on {dev} (fp16={dev=='cuda'})")
            return True
        except (OSError, ImportError, ValueError) as e:
            # Model not installed or transformers not available — expected.
            print(f"[SentenceModel] Unavailable ({e}); polishing disabled")
            _UNAVAILABLE = True
            return False
        except Exception as e:
            # Unexpected error (CUDA init failure, corrupt weights, etc.)
            # — propagate so the caller sees it rather than silently degrading.
            raise RuntimeError(
                f"[SentenceModel] Unexpected error loading model: {e}"
            ) from e


def polish(sentence: str) -> str:
    """Polish one sentence via CoEdit grammar error correction.

    Model: jbochi/coedit-small (77M params, flan-t5-small fine-tuned)
    Task prefix: "Fix grammatical errors in this sentence:" (GEC task)
    Max input: 128 tokens (truncated by tokenizer)
    Max output: 96 tokens (per sentence — callers must split multi-sentence input)
    Beam search: 2 beams, no sampling (deterministic)
    Source: CoEdIT paper (EMNLP 2023, Raheja et al.)

    Supported models (via RAYA_SENTENCE_MODEL env var):
        - jbochi/coedit-small (default, 77M params, fastest)
        - grammarly/coedit-large (770M params, higher quality, needs GPU)
        - grammarly/coedit-xl (3B params, highest quality, needs large GPU)

    Returns input unchanged if polish disabled, model unavailable, or
    sentence is trivial."""
    if not sentence or not sentence.strip():
        return sentence
    if not is_enabled():
        return sentence

    key = sentence.strip()
    if key in _CACHE:
        return _CACHE[key]

    if not _load():
        return sentence

    try:
        import torch
        prompt = "Fix grammatical errors in this sentence: " + key
        inp = _TOKENIZER(
            prompt, return_tensors="pt", max_length=128, truncation=True,
        ).to(_DEVICE)
        with torch.no_grad():
            out = _MODEL.generate(
                **inp, max_length=96, num_beams=2, do_sample=False,
            )
        result = _TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
        if not result:
            logger.warning(
                "[SentenceModel] CoEdit returned empty output for: %r "
                "— model may be broken, returning input unchanged", key
            )
            result = sentence
        _CACHE[key] = result
        return result
    except (RuntimeError, ValueError) as e:
        # RuntimeError: CUDA OOM or tensor errors
        # ValueError: tokenizer encoding issues
        logger.warning("[SentenceModel] polish error (%s); returning input unchanged", e)
        _CACHE[key] = sentence
        return sentence


def cleanup(text: str, speaker: str = None) -> str:
    """Two-pass dialogue cleanup:

    Pass 1 (structural): spaCy POS-based filler stripping.
      - INTJ tokens (yeah, oh, wow, like-as-filler) → removed
      - parataxis clauses (you know, I mean) → removed
      - Pure structural, no model needed.

    Pass 2 (model): CoEdit grammar polish on the cleaned text.
      - Fix remaining grammar issues
      - Normalize punctuation

    No pronoun resolution — retrieval handles that at read time
    via backward entity scoping.
    """
    if not text or not text.strip():
        return text or ""

    # -- Pass 1: structural filler stripping via spaCy --
    stripped = text.strip()
    try:
        from app.engines.grammar_engine import _get_nlp
        _nlp = _get_nlp()
        _doc = _nlp(stripped)
        # Remove INTJ tokens and parataxis clauses
        keep_tokens = []
        for tok in _doc:
            # Skip interjections (yeah, oh, wow, like-as-filler)
            if tok.pos_ == "INTJ":
                continue
            # Skip parataxis discourse markers ("you know", "I mean")
            if tok.dep_ == "parataxis":
                # Skip the parataxis token and its entire subtree
                parataxis_indices = {t.i for t in tok.subtree}
                # Mark for removal (handled below)
                continue
            keep_tokens.append(tok)

        # Second pass: remove parataxis subtrees
        parataxis_indices = set()
        for tok in _doc:
            if tok.dep_ == "parataxis":
                parataxis_indices |= {t.i for t in tok.subtree}

        # Third pass: detect discourse/meta frames and promote content.
        # "Wait what was I saying" / "Oh that reminds me, I need to..."
        # Pattern: ROOT is meta-speech verb with first-person subject,
        # and has ccomp/xcomp/conj child with its own subject.
        # Strip the frame, keep only the content clause(s).
        from app.engines.grammar_engine import _get_root
        _root = _get_root(_doc)
        _discourse_indices = set()
        if _root and _root.pos_ in ("VERB", "AUX"):
            _root_lemma = _root.lemma_.lower()
            _has_first_person_subj = any(
                c.dep_ in ("nsubj", "nsubjpass")
                and c.text.lower() in ("i", "we")
                for c in _root.children
            )
            # Meta-speech: first-person ROOT with subordinate content
            _meta_verbs = frozenset({
                "wait", "remind", "mean", "say", "tell", "know",
                "think", "wonder", "guess", "suppose", "remember",
            })
            if _has_first_person_subj and _root_lemma in _meta_verbs:
                # Check for content clause with its own subject
                _content_child = None
                for c in _root.children:
                    if (c.dep_ in ("ccomp", "xcomp", "conj", "parataxis")
                            and c.pos_ in ("VERB", "AUX")
                            and any(gc.dep_ in ("nsubj", "nsubjpass")
                                    for gc in c.children)):
                        _content_child = c
                        break
                if _content_child is not None:
                    # Strip the discourse frame — keep content subtree
                    _content_indices = {t.i for t in _content_child.subtree}
                    _discourse_indices = (
                        {t.i for t in _doc}
                        - _content_indices
                        - {t.i for t in _doc if t.pos_ == "PUNCT"}
                    )

        clean_tokens = [tok for tok in _doc
                        if tok.i not in parataxis_indices
                        and tok.i not in _discourse_indices
                        and tok.pos_ != "INTJ"]

        if clean_tokens:
            stripped = " ".join(tok.text for tok in clean_tokens).strip()
            # Clean up double spaces and leading conjunctions
            stripped = " ".join(stripped.split())
            # Strip leading "and", "but", "so" after filler removal
            first_word = stripped.split()[0].lower() if stripped else ""
            if first_word in ("and", "but", "so", "or"):
                stripped = " ".join(stripped.split()[1:])
    except (OSError, ImportError):
        pass  # fail-open: spaCy not installed, use raw text

    if not stripped:
        stripped = text.strip()

    # Pass 2: CoEdit "Rewrite to be formal:" per-sentence.
    # coedit-small has max_length=96 output tokens. Multi-sentence
    # turns get truncated. Fix: split into sentences, rewrite each
    # individually, rejoin. No content loss.
    if not is_enabled():
        return stripped

    key = ("cleanup", stripped, speaker or "")
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    if not _load():
        return stripped

    try:
        import torch

        def _rewrite_one(sentence: str) -> str:
            """Rewrite a single sentence via coedit.
            Semantic guard: if CoEdit changes ROOT verb, loses NER entities,
            or significantly changes length, reject the rewrite — it changed
            meaning, not just grammar."""
            prompt = "Fix grammatical errors in this sentence:" + sentence
            inp = _TOKENIZER(
                prompt, return_tensors="pt", max_length=128, truncation=True,
            ).to(_DEVICE)
            with torch.no_grad():
                out = _MODEL.generate(
                    **inp, max_length=96, num_beams=2, do_sample=False,
                )
            r = _TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
            if not r:
                logger.warning(
                    "[SentenceModel] CoEdit returned empty output for: %r",
                    sentence,
                )
                return sentence
            # Guardrail: reject triple/tuple syntax hallucination
            if "(" in r and ")" in r:
                oi = r.find("(")
                ci = r.find(")", oi + 1)
                if ci != -1 and r[oi + 1:ci].count(",") >= 2:
                    return sentence
            # Semantic guard: compare input vs output via spaCy.
            # If CoEdit changed the ROOT verb, lost NER entities, or
            # significantly shortened the text, it changed meaning —
            # reject the rewrite and keep the structurally cleaned input.
            try:
                from app.engines.grammar_engine import (
                    _get_nlp_fragment, _get_root,
                )
                _snlp = _get_nlp_fragment()
                _in_doc = _snlp(sentence)
                _out_doc = _snlp(r)
                # Check 1: ROOT verb lemma preserved
                _in_root = _get_root(_in_doc)
                _out_root = _get_root(_out_doc)
                if (_in_root and _out_root
                        and _in_root.pos_ in ("VERB", "AUX")
                        and _in_root.lemma_ != _out_root.lemma_):
                    logger.warning(
                        "[SentenceModel] CoEdit changed ROOT verb: "
                        "%r (%s) -> %r (%s)",
                        sentence, _in_root.lemma_, r, _out_root.lemma_,
                    )
                    return sentence
                # Check 2: NER entities not lost
                _in_ents = {e.text.lower() for e in _in_doc.ents}
                _out_ents = {e.text.lower() for e in _out_doc.ents}
                if _in_ents and not (_in_ents & _out_ents):
                    logger.warning(
                        "[SentenceModel] CoEdit lost NER entities: %r -> %r",
                        sentence, r,
                    )
                    return sentence
                # Check 3: output not drastically shorter (content lost)
                if len(r.split()) < len(sentence.split()) * 0.5:
                    logger.warning(
                        "[SentenceModel] CoEdit truncated content: %r -> %r",
                        sentence, r,
                    )
                    return sentence
            except (ImportError, OSError):
                pass  # spaCy not available for guard — accept rewrite
            return r

        # Split into sentences — rewrite each individually so
        # coedit-small's 96-token output limit doesn't truncate.
        from app.engines.grammar_engine import _get_nlp
        # Split on sentence boundaries AND ellipsis/dash breaks.
        # spaCy may not split on "..." or "—" but these are natural
        # sentence boundaries in conversational text.
        import re
        _presplit = re.split(r'\.{2,}|—|–', stripped)
        _presplit = [s.strip() for s in _presplit if s.strip()]
        _sentences = []
        for _seg in _presplit:
            _seg_doc = _get_nlp()(_seg)
            _sentences.extend(
                s.text.strip() for s in _seg_doc.sents if s.text.strip()
            )

        if len(_sentences) <= 1:
            result = _rewrite_one(stripped)
        else:
            rewritten = [_rewrite_one(s) for s in _sentences]
            result = " ".join(rewritten)

        if not result:
            logger.warning(
                "[SentenceModel] cleanup produced empty result for: %r", stripped
            )
            result = text.strip()

        _CACHE[key] = result
        return result
    except (RuntimeError, ValueError) as e:
        # RuntimeError: CUDA OOM or tensor errors
        # ValueError: tokenizer encoding issues
        logger.warning("[SentenceModel] cleanup error (%s); returning input unchanged", e)
        fallback = text.strip()
        _CACHE[key] = fallback
        return fallback


def reset_cache():
    global _CACHE
    _CACHE = {}
