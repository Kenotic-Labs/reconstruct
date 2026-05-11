"""
Ingestion — messy text → clean structured English.

Two-pass pipeline:
  Pass 1 (spaCy): structural noise stripping
    - Strip interjections, parataxis, discourse frames, idiom frames
    - Detect retractions → discard
    - ~6ms per turn.

  Pass 2 (coedit-small): grammar correction
    - 77M param T5-small fine-tuned on CoEdit data
    - Task prefix: "Fix grammatical errors in this sentence:"
    - Fixes grammar without changing meaning. 97% fact retention.
    - ~340ms per turn on GPU. Runs on any device (GPU/CPU).

Pronoun resolution (I → speaker name) is NOT an ingestion concern.
It belongs in the write path (memory engine) where identity is stored.

Output goes to temporal + grammar engines.

Public API:
    cleanup(text: str) -> str
    is_available() -> bool
    reset_cache()
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Dict

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Entry check
# -----------------------------------------------------------------------

_ENTRY_CHECKED = False


class IngestionEntryError(RuntimeError):
    pass


def _check_entry():
    """Verify grammar_engine and temporal are importable. Runs once."""
    global _ENTRY_CHECKED
    if _ENTRY_CHECKED:
        return
    missing = []
    try:
        from app.engines import grammar_engine  # noqa: F401
        if not hasattr(grammar_engine, 'process'):
            missing.append("grammar_engine.process")
    except ImportError:
        missing.append("grammar_engine")
    try:
        from app.engines import temporal  # noqa: F401
        if not hasattr(temporal, 'get_temporal_engine'):
            missing.append("temporal.get_temporal_engine")
    except ImportError:
        missing.append("temporal")
    if missing:
        raise IngestionEntryError(
            f"ingestion entry check failed — missing: {', '.join(missing)}"
        )
    _ENTRY_CHECKED = True


# -----------------------------------------------------------------------
# coedit-small configuration
# -----------------------------------------------------------------------

_COEDIT_TASK_PREFIX = "Fix grammatical errors in this sentence: "
_COEDIT_MODEL_ID = "jbochi/coedit-small"

_MODEL = None
_TOKENIZER = None
_LOCK = Lock()
_UNAVAILABLE = False
_CACHE: Dict[str, str] = {}

# -----------------------------------------------------------------------
# spaCy noise-stripping rules (dep labels, POS tags, lemma sets)
# -----------------------------------------------------------------------

_FILLER_POS = frozenset({"INTJ"})
_SCAFFOLDING_DEPS = frozenset({"parataxis"})
_DISCOURSE_FRAME_LEMMAS = frozenset({
    "wait", "remind", "mean", "say", "tell", "know",
    "think", "wonder", "guess", "suppose", "remember",
    "hear", "listen", "look",
})
_DISCOURSE_SUBJECT_LEMMAS = frozenset({"i", "we", "you"})
_DISCOURSE_FILLER_NOUNS = frozenset({
    "thing", "point", "deal", "fact", "truth", "matter",
    "problem", "issue", "question",
})
_IDIOM_FRAMES = frozenset({
    "long story short", "bottom line", "at the end of the day",
    "truth be told", "between you and me", "to be honest",
    "to be fair", "for what it's worth", "if you ask me",
    "believe it or not", "here's the thing", "here's the deal",
})
_RETRACTION_PHRASES = frozenset({
    "never mind", "nevermind", "forget it", "forget that",
    "scratch that", "disregard that", "ignore that",
})


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------

def is_available() -> bool:
    """True if coedit-small loaded successfully."""
    return _MODEL is not None and not _UNAVAILABLE


def _load() -> bool:
    """Load coedit-small (77M T5-small seq2seq). GPU if available, CPU fallback."""
    global _MODEL, _TOKENIZER, _UNAVAILABLE
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
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            # Resolve to local snapshot path from HF cache.
            _hf_cache = os.environ.get(
                "HF_HUB_CACHE",
                os.path.join(os.environ.get("HF_HOME", ""), "hub"),
            )
            _model_dir = os.path.join(
                _hf_cache,
                "models--" + _COEDIT_MODEL_ID.replace("/", "--"),
            )
            _ref_path = os.path.join(_model_dir, "refs", "main")
            if os.path.exists(_ref_path):
                with open(_ref_path) as f:
                    _rev = f.read().strip()
                _local_path = os.path.join(_model_dir, "snapshots", _rev)
            else:
                _local_path = _COEDIT_MODEL_ID

            tok = AutoTokenizer.from_pretrained(_local_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(_local_path)
            model.eval()

            _dev = os.environ.get(
                "RAYA_SENTENCE_DEVICE",
                "cuda:0" if torch.cuda.is_available() else "cpu",
            )
            model = model.to(_dev)

            _MODEL = model
            _TOKENIZER = tok
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"[Ingestion] coedit-small loaded ({params:.0f}M params) on {_dev}")
            return True
        except (OSError, ImportError, ValueError) as e:
            print(f"[Ingestion] coedit-small unavailable ({e})")
            _UNAVAILABLE = True
            return False


# -----------------------------------------------------------------------
# Pass 1: spaCy noise stripping
# -----------------------------------------------------------------------

def _strip_noise(text: str) -> str:
    """Strip structural noise via spaCy dep labels.

    Removes: interjections, parataxis, discourse frames, idiom frames.
    Detects retractions → returns empty string.
    """
    stripped = text.strip()
    if not stripped:
        return ""

    try:
        from app.engines.grammar_engine import _get_nlp, _get_root
        _nlp = _get_nlp()

        # Retraction detection
        _lower = stripped.lower().rstrip(".,!? ")
        for phrase in _RETRACTION_PHRASES:
            if _lower.endswith(phrase):
                return ""

        _doc = _nlp(stripped)

        # Rhetorical/discourse question followed by factual answer
        _raw_segments = [seg.strip() for seg in stripped.split("?") if seg.strip()]
        if len(_raw_segments) >= 2:
            _first_doc = _nlp(_raw_segments[0] + "?")
            _first_root = _get_root(_first_doc)
            _first_text = _raw_segments[0].lower()
            _is_discourse_question = (
                _first_text.startswith("you know")
                or any(_first_text.startswith(frame) for frame in _IDIOM_FRAMES)
                or (
                    _first_root is not None
                    and _first_root.lemma_.lower() in _DISCOURSE_FRAME_LEMMAS
                )
            )
            if _is_discourse_question:
                _tail = "? ".join(_raw_segments[1:]).strip()
                if _tail:
                    stripped = _tail
                    _doc = _nlp(stripped)

        # Collect token indices to remove
        _remove = set()

        # INTJ tokens — all interjections are social noise, not facts.
        for tok in _doc:
            if tok.pos_ in _FILLER_POS:
                _remove.add(tok.i)

        # Parataxis subtrees
        for tok in _doc:
            if tok.dep_ in _SCAFFOLDING_DEPS:
                _remove |= {t.i for t in tok.subtree}

        # Discourse frame detection
        _root = _get_root(_doc)

        if _root and _root.pos_ in ("VERB", "AUX"):
            _frame_verb = None
            _content_verb = None

            # ROOT is discourse frame verb
            if _root.lemma_.lower() in _DISCOURSE_FRAME_LEMMAS:
                _subj_ok = any(
                    c.dep_ in ("nsubj", "nsubjpass")
                    and c.text.lower() in _DISCOURSE_SUBJECT_LEMMAS
                    for c in _root.children
                )
                if _subj_ok:
                    for c in _root.children:
                        if (c.dep_ in ("ccomp", "xcomp", "conj", "parataxis")
                                and c.pos_ in ("VERB", "AUX")
                                and any(gc.dep_ in ("nsubj", "nsubjpass")
                                        for gc in c.children)):
                            _root_subj = next(
                                (rc for rc in _root.children
                                 if rc.dep_ in ("nsubj", "nsubjpass")), None
                            )
                            _content_subj = next(
                                (gc for gc in c.children
                                 if gc.dep_ in ("nsubj", "nsubjpass")), None
                            )
                            if (_root_subj and _content_subj
                                    and _root_subj.text.lower()
                                    != _content_subj.text.lower()):
                                _frame_verb = _root
                                _content_verb = c
                            break

            # ROOT is filler noun frame ("The thing is, I applied")
            if _frame_verb is None and _root.lemma_.lower() == "be":
                _root_subj = next(
                    (c for c in _root.children
                     if c.dep_ in ("nsubj", "nsubjpass")), None
                )
                if (_root_subj
                        and _root_subj.lemma_.lower() in _DISCOURSE_FILLER_NOUNS):
                    _content_ccomp = next(
                        (c for c in _root.children
                         if c.dep_ == "ccomp" and c.pos_ in ("VERB", "AUX")),
                        None,
                    )
                    if _content_ccomp is not None:
                        _frame_verb = _root
                        _content_verb = _content_ccomp

            # ccomp as frame (discourse in subordinate position)
            if _frame_verb is None:
                for c in _root.children:
                    if c.dep_ == "ccomp" and c.pos_ in ("VERB", "AUX"):
                        if c.lemma_.lower() == "be":
                            _ccomp_subj = next(
                                (gc for gc in c.children
                                 if gc.dep_ in ("nsubj", "nsubjpass")), None
                            )
                            if (_ccomp_subj and _ccomp_subj.lemma_.lower()
                                    in _DISCOURSE_FILLER_NOUNS):
                                _frame_verb = c
                                _content_verb = _root
                                break

            if _frame_verb is not None and _content_verb is not None:
                _content_indices = {t.i for t in _content_verb.subtree}
                _frame_indices = {t.i for t in _doc} - _content_indices
                _frame_indices -= {t.i for t in _doc if t.pos_ == "PUNCT"}
                _remove |= _frame_indices

        # Idiom adverbial frames at sentence start
        _sent_lower = stripped.lower()
        for idiom in _IDIOM_FRAMES:
            if _sent_lower.startswith(idiom):
                _idiom_doc = _nlp(idiom)
                _idiom_tokens = [
                    t.text.lower() for t in _idiom_doc if not t.is_space
                ]
                _removed = 0
                for tok in _doc:
                    if _removed < len(_idiom_tokens):
                        _remove.add(tok.i)
                        if tok.text.lower() == _idiom_tokens[_removed]:
                            _removed += 1
                    elif tok.text == ",":
                        _remove.add(tok.i)
                    else:
                        break
                break

        # Strip rhetorical tail fragments left after idiom removal
        _remaining = [tok for tok in _doc if tok.i not in _remove]
        while _remaining:
            _first = _remaining[0].text.lower()
            if _first in _DISCOURSE_FILLER_NOUNS:
                _remove.add(_remaining[0].i)
                _remaining = _remaining[1:]
            elif _first in ("?", ",", ":", ";", "-", "\u2014", "\u2013"):
                _remove.add(_remaining[0].i)
                _remaining = _remaining[1:]
            elif len(_remaining) > 1 and _remaining[1].text == "?":
                _remove.add(_remaining[0].i)
                _remove.add(_remaining[1].i)
                _remaining = _remaining[2:]
            else:
                break

        # Build cleaned text — preserve original whitespace/contractions
        clean_tokens = [tok for tok in _doc if tok.i not in _remove]

        # Strip leading punctuation/conjunctions
        while clean_tokens and clean_tokens[0].pos_ in ("PUNCT", "SPACE"):
            clean_tokens = clean_tokens[1:]
        while (clean_tokens
               and clean_tokens[0].pos_ in ("CCONJ", "SCONJ")
               and clean_tokens[0].text.lower() in ("and", "but", "so", "or")):
            clean_tokens = clean_tokens[1:]

        # Rebuild with original whitespace
        if clean_tokens:
            parts = []
            for tok in clean_tokens:
                if tok.text == "," and (not parts or parts[-1].rstrip().endswith(",")):
                    continue
                parts.append(tok.text_with_ws)
            stripped = "".join(parts).strip()

        if not stripped or not stripped.strip():
            stripped = text.strip()

    except (OSError, ImportError):
        pass  # spaCy not available — return raw text

    return stripped if stripped else text.strip()


# -----------------------------------------------------------------------
# Pass 2: coedit-small simplify / rewrite
# -----------------------------------------------------------------------

def _simplify(text: str) -> str:
    """Simplify/rewrite one piece of text via coedit-small."""
    if not text or not text.strip():
        return text or ""

    if not _load():
        return text

    try:
        import torch
        inp = _COEDIT_TASK_PREFIX + text.strip()
        _dev = next(_MODEL.parameters()).device
        inputs = _TOKENIZER(inp, return_tensors="pt").to(_dev)
        with torch.no_grad():
            out = _MODEL.generate(**inputs, max_new_tokens=128)
        result = _TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
        return result if result else text
    except (RuntimeError, ValueError) as e:
        logger.warning("[Ingestion] simplify error (%s); returning input unchanged", e)
        return text


# -----------------------------------------------------------------------
# cleanup — the full pipeline
# -----------------------------------------------------------------------

def cleanup(text: str, speaker: str = None) -> str:
    """Ingestion pipeline: messy text → clean English.

    Pass 1: spaCy strips structural noise.
    Pass 2: coedit-small simplifies / rewrites.
    Output goes to temporal + grammar engines.
    """
    _check_entry()

    if not text or not text.strip():
        return text or ""

    key = text.strip()
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    # Pass 1: strip noise
    stripped = _strip_noise(text)
    if not stripped:
        _CACHE[key] = ""
        return ""

    # Pass 2: grammar correction — skip if text is already clean.
    # Clean text = spaCy didn't strip anything (no noise) and text has
    # proper capitalization + punctuation. No point running a 250ms model
    # on text that's already correct.
    _text_changed = (stripped != text.strip())
    _has_capitalization = stripped[0].isupper() if stripped else False
    _has_punctuation = stripped[-1] in ".!?" if stripped else False
    if not _text_changed and _has_capitalization and _has_punctuation:
        result = stripped  # already clean — skip coedit
    else:
        result = _simplify(stripped)
        if not result or not result.strip():
            result = text.strip()

    _CACHE[key] = result
    return result


def reset_cache():
    global _CACHE
    _CACHE = {}


__all__ = ["cleanup", "is_available", "reset_cache"]
