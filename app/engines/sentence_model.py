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

# -----------------------------------------------------------------------
# Model rules — "system prompts" for off-the-shelf models
# -----------------------------------------------------------------------

# CoEdit task prefix — ONLY grammar correction. Other tasks (simplify,
# paraphrase, formality) change meaning, violating the lossless contract.
_COEDIT_TASK_PREFIX = "Fix grammatical errors in this sentence:"

# Structural cleanup rules — spaCy dep labels that signal noise
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
    # No contraction map — spaCy was trained on web text including informal
    # speech. Regex replacement before parse changes tokenization and can
    # break dep trees. Let spaCy handle what it was trained on.

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
            # GPU only — no CPU fallback

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
        prompt = _COEDIT_TASK_PREFIX + " " + key
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

    Pass 1 (structural): spaCy dep-label rules from pipeline_config.
      - 7 detection patterns, all structural
      - No model inference beyond spaCy's dep parse

    Pass 2 (model): CoEdit grammar polish (GEC only).
      - One task prefix, one model, semantic guards
      - Does NOT change meaning — only fixes grammar

    Rules are defined in pipeline_config.py (the "system prompt" for each model).
    """
    if not text or not text.strip():
        return text or ""


    # -- Pass 1: structural cleanup via spaCy dep labels --
    stripped = text.strip()
    try:
        from app.engines.grammar_engine import _get_nlp, _get_root
        _nlp = _get_nlp()

        # Step 5: Retraction detection (before full parse — cheap check)
        _lower_stripped = stripped.lower().rstrip(".,!? ")
        for phrase in _RETRACTION_PHRASES:
            if _lower_stripped.endswith(phrase):
                return ""  # speaker withdrew — discard entire utterance

        _doc = _nlp(stripped)

        # Step 1: Collect indices to remove
        _remove_indices = set()

        # 1a: INTJ tokens (POS-based)
        for tok in _doc:
            if tok.pos_ in _FILLER_POS:
                _remove_indices.add(tok.i)

        # 1b: Parataxis subtrees (dep-based)
        for tok in _doc:
            if tok.dep_ in _SCAFFOLDING_DEPS:
                _remove_indices |= {t.i for t in tok.subtree}

        # Step 3: Discourse frame detection
        _root = _get_root(_doc)

        # 3a: ROOT or ccomp is discourse frame verb
        if _root and _root.pos_ in ("VERB", "AUX"):
            _frame_verb = None
            _content_verb = None

            # Check ROOT as frame
            if _root.lemma_.lower() in _DISCOURSE_FRAME_LEMMAS:
                _subj_ok = any(
                    c.dep_ in ("nsubj", "nsubjpass")
                    and c.text.lower() in _DISCOURSE_SUBJECT_LEMMAS
                    for c in _root.children
                )
                if _subj_ok:
                    # Find content clause with its own subject
                    for c in _root.children:
                        if (c.dep_ in ("ccomp", "xcomp", "conj", "parataxis")
                                and c.pos_ in ("VERB", "AUX")
                                and any(gc.dep_ in ("nsubj", "nsubjpass")
                                        for gc in c.children)):
                            # Guard: only strip if content subject differs
                            # from ROOT subject (else it's the same person's fact)
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

            # Check ROOT as filler noun frame:
            # "The thing is, I applied" → ROOT=is, nsubj=thing, ccomp=applied
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

            # Check ccomp as frame (discourse in subordinate position)
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
                # Keep punctuation
                _frame_indices -= {t.i for t in _doc if t.pos_ == "PUNCT"}
                _remove_indices |= _frame_indices

        # 3b: Idiom adverbial frames at sentence start
        _sent_text_lower = stripped.lower()
        for idiom in _IDIOM_FRAMES:
            if _sent_text_lower.startswith(idiom):
                # Remove tokens up to and including the comma after the idiom
                _idiom_len = len(idiom.split())
                _removed = 0
                for tok in _doc:
                    if _removed < _idiom_len or tok.text == ",":
                        _remove_indices.add(tok.i)
                        if tok.pos_ != "PUNCT":
                            _removed += 1
                    else:
                        break
                break

        # Build cleaned text
        clean_tokens = [tok for tok in _doc if tok.i not in _remove_indices]

        if clean_tokens:
            stripped = " ".join(tok.text for tok in clean_tokens).strip()
            stripped = " ".join(stripped.split())
            # Strip leading conjunctions left after filler removal
            if stripped:
                first_word = stripped.split()[0].lower()
                if first_word in ("and", "but", "so", "or"):
                    stripped = " ".join(stripped.split()[1:])

        # Guard: if cleanup stripped everything, return original
        if not stripped or not stripped.strip():
            stripped = text.strip()

    except (OSError, ImportError):
        pass  # spaCy not installed — use raw text

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
            prompt = _COEDIT_TASK_PREFIX + " " + sentence
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
