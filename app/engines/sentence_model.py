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
_ENCODE = None
_DECODE = None
_DEVICE: Optional[str] = None
_LOCK = Lock()
_UNAVAILABLE = False
_POLISH_CACHE: Dict[str, str] = {}
_CLEANUP_CACHE: Dict[tuple, str] = {}

_ENV_DEVICE = "RAYA_SENTENCE_DEVICE"
_ENV_ENABLE = "RAYA_SENTENCE_POLISH"   # set to "0" to disable, default ON
_GECTOR_MODEL_ID = "models/gector-raya"

# GECToR configuration — explicit instructions for the correction model:
#   n_iteration=5:     run up to 5 correction passes (complex errors need
#                      multiple passes: detect→fix→re-detect residual errors)
#   min_error_prob=0:  consider ALL predicted corrections regardless of
#                      confidence (let the model decide, not a threshold)
#   keep_confidence=0: no bias toward keeping original tokens — if the model
#                      predicts an edit, apply it
#   batch_size=128:    process up to 128 sentences in one GPU forward pass
_GECTOR_N_ITER = 5
_GECTOR_MIN_ERROR_PROB = 0.0
_GECTOR_KEEP_CONFIDENCE = 0.0
_GECTOR_BATCH_SIZE = 128


def is_enabled() -> bool:
    """Polish is on by default. Set RAYA_SENTENCE_POLISH=0 to disable."""
    return os.environ.get(_ENV_ENABLE, "1") != "0"


def is_available() -> bool:
    """True if the model loaded successfully and polishing can run."""
    return _MODEL is not None and not _UNAVAILABLE


def _load() -> bool:
    """Load GECToR (encoder-only grammar tagger) on GPU.

    GECToR = single forward pass, not autoregressive.
    ~10ms/sentence vs coedit's ~500ms/sentence.
    Corrections are applied as edit tags (keep/delete/replace/insert),
    not generated token by token.
    """
    global _MODEL, _TOKENIZER, _ENCODE, _DECODE, _DEVICE, _UNAVAILABLE
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
            import json as _json
            from transformers import AutoTokenizer
            from gector import GECToR

            dev = os.environ.get(_ENV_DEVICE,
                                 os.environ.get("RAYA_EMBED_DEVICE", "cuda:0"))
            # GPU only — RTX 4000 (cuda:0). No CPU fallback.

            # Temporarily allow local file access — app/__init__.py sets
            # HF_HUB_OFFLINE=1 globally, but we load from a local directory.
            _saved_offline = os.environ.pop("HF_HUB_OFFLINE", None)
            _saved_toffline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                model = GECToR.from_pretrained(_GECTOR_MODEL_ID, local_files_only=True)
                tok = AutoTokenizer.from_pretrained(_GECTOR_MODEL_ID, local_files_only=True)
            finally:
                if _saved_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = _saved_offline
                if _saved_toffline is not None:
                    os.environ["TRANSFORMERS_OFFLINE"] = _saved_toffline
            _vocab_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                _GECTOR_MODEL_ID,
            )
            with open(os.path.join(_vocab_dir, "encode_vocab.json")) as f:
                encode = _json.load(f)
            with open(os.path.join(_vocab_dir, "decode_vocab.json")) as f:
                decode = _json.load(f)
            model = model.to(dev).eval()

            _MODEL = model
            _TOKENIZER = tok
            _ENCODE = encode
            _DECODE = decode
            _DEVICE = dev
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"[SentenceModel] GECToR loaded ({params:.0f}M params) on {dev}")
            return True
        except (OSError, ImportError, ValueError) as e:
            print(f"[SentenceModel] Unavailable ({e}); polishing disabled")
            _UNAVAILABLE = True
            return False
        except Exception as e:
            raise RuntimeError(
                f"[SentenceModel] Unexpected error loading model: {e}"
            ) from e


def polish(sentence: str) -> str:
    """Polish one sentence via GECToR grammar error correction.

    Model: gotutiyan/gector-roberta-base-5k (124M params, encoder-only)
    Architecture: non-autoregressive tagger — single forward pass.
    Predicts edit tags (keep/delete/replace/insert) per token, then
    applies edits deterministically. ~10ms/sentence on GPU.

    GECToR configuration (explicit instructions):
      n_iteration=5:  up to 5 correction passes for complex errors
      min_error_prob=0: apply all predicted corrections
      keep_confidence=0: no bias toward keeping original tokens

    Returns input unchanged if polish disabled, model unavailable, or
    sentence is trivial."""
    if not sentence or not sentence.strip():
        return sentence
    if not is_enabled():
        return sentence

    key = sentence.strip()
    if key in _POLISH_CACHE:
        return _POLISH_CACHE[key]

    if not _load():
        return sentence

    try:
        from gector import predict as gector_predict
        corrected = gector_predict(
            _MODEL, _TOKENIZER, [key], _ENCODE, _DECODE,
            keep_confidence=_GECTOR_KEEP_CONFIDENCE,
            min_error_prob=_GECTOR_MIN_ERROR_PROB,
            n_iteration=_GECTOR_N_ITER,
            batch_size=_GECTOR_BATCH_SIZE,
        )
        result = corrected[0].strip() if corrected else key
        if not result:
            result = sentence
        _POLISH_CACHE[key] = result
        return result
    except (RuntimeError, ValueError) as e:
        logger.warning("[SentenceModel] polish error (%s); returning input unchanged", e)
        _POLISH_CACHE[key] = sentence
        return sentence


def cleanup(text: str, speaker: str = None) -> str:
    """Three-pass pipeline: STT → structured declarative sentences.

    HARNESS ROLE: Convert messy conversational text into clean,
    structured declarative sentences that the grammar engine can
    extract clean traces from.

    Pass 1 (PRE — structural normalization):
      - Resolve pronouns: I/me/my → speaker name
      - Strip backchannel/commentary: "Wow!", "That's great!"
      - spaCy dep-label rules for structural cleanup

    Pass 2 (MODEL — GECToR grammar correction):
      - Fix grammar: "goed" → "went", "dont" → "doesn't"
      - Insert missing words, fix agreement
      - Does NOT restructure — only corrects

    Pass 3 (POST — structural decomposition):
      - Split compound sentences into simple SVO clauses
      - Each clause = one extractable fact
      - "Melanie ran a race and it was rewarding" →
        "Melanie ran a race. The race was rewarding."
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

        # Step 0: rhetorical/discourse question followed by factual answer.
        # Keep the factual answer clause and drop the scaffolding question.
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
                # Remove the full idiom span as tokenized by spaCy, not
                # by naive whitespace. This handles "here's the deal"
                # -> ["here", "'s", "the", "deal"].
                _idiom_doc = _nlp(idiom)
                _idiom_tokens = [
                    t.text.lower() for t in _idiom_doc
                    if not t.is_space
                ]
                _removed = 0
                for tok in _doc:
                    if _removed < len(_idiom_tokens):
                        _remove_indices.add(tok.i)
                        if tok.text.lower() == _idiom_tokens[_removed]:
                            _removed += 1
                    elif tok.text == ",":
                        _remove_indices.add(tok.i)
                    else:
                        break
                break

        # Strip rhetorical tail fragments left after idiom removal:
        # "Deal? I work at Google..." -> "I work at Google..."
        _clean_probe = [tok for tok in _doc if tok.i not in _remove_indices]
        if _clean_probe:
            _probe_tokens = [tok.text for tok in _clean_probe]
            while _probe_tokens:
                if not _clean_probe:
                    break
                _first = _probe_tokens[0].lower()
                if _first in _DISCOURSE_FILLER_NOUNS:
                    _remove_indices.add(_clean_probe[0].i)
                    _clean_probe = _clean_probe[1:]
                    _probe_tokens = [tok.text for tok in _clean_probe]
                    continue
                if _first in ("?", ",", ":", ";", "-", "—", "–"):
                    _remove_indices.add(_clean_probe[0].i)
                    _clean_probe = _clean_probe[1:]
                    _probe_tokens = [tok.text for tok in _clean_probe]
                    continue
                if len(_clean_probe) > 1 and _clean_probe[1].text == "?":
                    _remove_indices.add(_clean_probe[0].i)
                    _remove_indices.add(_clean_probe[1].i)
                    _clean_probe = _clean_probe[2:]
                    _probe_tokens = [tok.text for tok in _clean_probe]
                else:
                    break

        # Build cleaned text
        clean_tokens = [tok for tok in _doc if tok.i not in _remove_indices]

        if clean_tokens:
            _normalized_tokens = []
            _last_text = None
            for tok in clean_tokens:
                _text = tok.text
                if (_text == "," and (_last_text is None or _last_text == ",")):
                    continue
                _normalized_tokens.append(_text)
                _last_text = _text
            stripped = " ".join(tok.text for tok in clean_tokens).strip()
            if _normalized_tokens:
                stripped = " ".join(_normalized_tokens).strip()
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

    # ── Pass 1b (PRE): Resolve pronouns BEFORE GECToR ──
    # "I ran a race" → "Melanie ran a race" BEFORE grammar correction.
    # This way GECToR corrects "Melanie ran" not "I ran", and the
    # grammar engine receives text with resolved entities.
    if speaker and stripped:
        try:
            from app.engines.grammar_engine import resolve_pronouns
            stripped = resolve_pronouns(stripped, speaker)
        except Exception:
            pass

    # Pass 2: GECToR grammar correction.
    # Non-autoregressive tagger — all sentences batched in ONE forward pass.
    # ~10ms total for a multi-sentence turn vs coedit's ~2000ms.
    if not is_enabled():
        return stripped

    key = ("cleanup", stripped, speaker or "")
    cached = _CLEANUP_CACHE.get(key)
    if cached is not None:
        return cached

    if not _load():
        return stripped

    try:
        from gector import predict as gector_predict

        # Split into sentences via spaCy
        from app.engines.grammar_engine import _get_nlp
        import re
        _presplit = re.split(r'\.{2,}|—|–', stripped)
        _presplit = [s.strip() for s in _presplit if s.strip()]
        _sentences = []
        for _seg in _presplit:
            _seg_doc = _get_nlp()(_seg)
            _sentences.extend(
                s.text.strip() for s in _seg_doc.sents if s.text.strip()
            )

        if not _sentences:
            _sentences = [stripped]

        # Batch ALL sentences through GECToR in one forward pass
        corrected = gector_predict(
            _MODEL, _TOKENIZER, _sentences, _ENCODE, _DECODE,
            keep_confidence=_GECTOR_KEEP_CONFIDENCE,
            min_error_prob=_GECTOR_MIN_ERROR_PROB,
            n_iteration=_GECTOR_N_ITER,
            batch_size=_GECTOR_BATCH_SIZE,
        )

        result = " ".join(s.strip() for s in corrected if s.strip())

        # ── Pass 3 (POST): structural decomposition ──
        # Split compound sentences into simple SVO clauses.
        # "Melanie ran a race and it was rewarding" →
        # "Melanie ran a race. It was rewarding."
        try:
            _post_doc = _get_nlp()(result)
            _decomposed = []
            for _sent in _post_doc.sents:
                _sent_root = None
                for _t in _sent:
                    if _t.dep_ == "ROOT":
                        _sent_root = _t
                        break
                if not _sent_root:
                    _decomposed.append(str(_sent).strip())
                    continue
                # Find conjunct verbs sharing the same subject
                _conj_verbs = [
                    c for c in _sent_root.children
                    if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")
                ]
                if _conj_verbs:
                    # Extract the main clause (up to the conjunction)
                    _conj_start = min(c.i for c in _conj_verbs)
                    # Find the "and"/"but" before the conj verb
                    _cc_idx = _conj_start
                    for _t in _sent:
                        if _t.dep_ == "cc" and _t.i < _conj_start:
                            _cc_idx = _t.i
                    _main = str(_sent[:_cc_idx]).strip().rstrip(",;")
                    if _main and len(_main) > 10:
                        _decomposed.append(_main + ".")
                    # Each conj verb becomes its own clause
                    for _cv in _conj_verbs:
                        # Get the subject — inherit from root if not explicit
                        _cv_subj = None
                        for _ch in _cv.children:
                            if _ch.dep_ in ("nsubj", "nsubjpass"):
                                _cv_subj = _ch
                                break
                        if _cv_subj:
                            _clause = str(_sent[_cv_subj.i:]).strip()
                        else:
                            # Inherit subject from main clause
                            _main_subj = None
                            for _ch in _sent_root.children:
                                if _ch.dep_ in ("nsubj", "nsubjpass"):
                                    _main_subj = str(_ch)
                                    break
                            if _main_subj:
                                _cv_span = str(_sent[_cv.i:]).strip()
                                _clause = f"{_main_subj} {_cv_span}"
                            else:
                                _clause = str(_sent[_cv.i:]).strip()
                        _clause = _clause.rstrip(",;").strip()
                        if _clause and len(_clause) > 10:
                            if not _clause.endswith("."):
                                _clause += "."
                            _decomposed.append(_clause)
                else:
                    _decomposed.append(str(_sent).strip())
            if _decomposed:
                result = " ".join(_decomposed)
        except Exception:
            pass

        # Post-cleanup: strip leading discourse frames left after correction
        try:
            _result_doc = _get_nlp()(result)
            _result_sents = [s for s in _result_doc.sents if s.text.strip()]
            if _result_sents:
                _lead_tokens = [
                    tok for tok in _result_sents[0]
                    if not tok.is_punct and not tok.is_space
                ]
                _lead_lemmas = {tok.lemma_.lower() for tok in _lead_tokens}
                if (
                    _lead_tokens
                    and len(_lead_tokens) <= 3
                    and _lead_lemmas <= (_DISCOURSE_FRAME_LEMMAS | {"actually"})
                ):
                    result = " ".join(
                        sent.text.strip() for sent in _result_sents[1:]
                    ).strip()
        except (OSError, ImportError):
            pass

        if not result:
            result = text.strip()

        _CLEANUP_CACHE[key] = result
        return result
    except (RuntimeError, ValueError) as e:
        logger.warning("[SentenceModel] cleanup error (%s); returning input unchanged", e)
        fallback = text.strip()
        _CLEANUP_CACHE[key] = fallback
        return fallback


def reset_cache():
    global _POLISH_CACHE, _CLEANUP_CACHE
    _POLISH_CACHE = {}
    _CLEANUP_CACHE = {}
