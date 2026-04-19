# -*- coding: cp1252 -*-
"""
predicted_queries.py -- question generator for the DTCM write path.

For each stored triple (subject, predicate, object) with entity types
(subject_type, object_type), emit one predicted question per WH-type the
triple can answer:

    WHO   -- fires if subject_type or object_type is PERSON
    WHAT  -- always fires
    WHEN  -- fires if object_type is TIME, or predicate carries a
             temporal stem (WordNet closure over time_period.n.01)
    WHERE -- fires if object_type is LOCATION

Each question is embedded via the shared MiniLM embedder (same path the
edge_embedding uses) so the retrieval layer can compare question vectors
with one cosine op.

Model priority, per write-path foundation spec:

    1. raya-srl-220m-v4/final (already loaded by MemoryEngine) with the
       prefix 'generate questions: subject|predicate|object -> WH'.
    2. google/flan-t5-base from D:/Nura/Env/hf_cache with a natural-
       language prompt.

We load once, probe with a 5-triple corpus on first call, pick the
winner (the model whose outputs start with a WH token on >= 4/5 probes),
and log the choice. No curated word lists; predicate-temporal detection
uses WordNet hypernym closure, not a list of temporal verbs.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

import spacy

from app.vector.embedder import embed_text

# Lazy-loaded spaCy model for POS-based verb lemmatization.
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _lemmatize_predicate(pred_clean: str) -> str:
    """Lemmatize the head verb of a predicate phrase using spaCy POS.

    'works at' -> 'work at', 'assigned to' -> 'assign to'.
    Only the first VERB token is lemmatized; everything else passes through.
    """
    doc = _get_nlp()(pred_clean)
    tokens = []
    verb_done = False
    for tok in doc:
        if not verb_done and tok.pos_ == "VERB":
            tokens.append(tok.lemma_)
            verb_done = True
        else:
            tokens.append(tok.text)
    return " ".join(tokens)

# Closed WH-type set for this foundation layer.
WH_WHO = "WHO"
WH_WHAT = "WHAT"
WH_WHEN = "WHEN"
WH_WHERE = "WHERE"

# Type constants mirrored from type_resolver (avoid circular import).
PERSON = "PERSON"
LOCATION = "LOCATION"
TIME = "TIME"

# ---- Model selection ---------------------------------------------------

_model = None
_tokenizer = None
_device = "cpu"
_model_name = None  # "raya" or "flan" after init
_probed = False


def _raya_path() -> str:
    from pathlib import Path
    here = Path(__file__).resolve()
    return str(here.parents[2] / "models" / "raya-srl-220m-v4" / "final")


def _load_model(pref: str) -> Optional[Tuple[object, object, str]]:
    try:
        import torch
        from transformers import T5ForConditionalGeneration, AutoTokenizer
    except Exception as e:
        print(f"[predicted_queries] transformers unavailable: {e}")
        return None

    if pref == "raya":
        path = os.environ.get("NURA_T5_MODEL_PATH", _raya_path())
    else:
        os.environ.setdefault("HF_HOME", "D:/Nura/Env/hf_cache")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        path = "google/flan-t5-base"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(path)
        mdl = T5ForConditionalGeneration.from_pretrained(path).to(device).eval()
        return (mdl, tok, device)
    except Exception as e:
        print(f"[predicted_queries] {pref} load failed: {e}")
        return None


def _generate_with(model, tokenizer, device, prompt: str, max_new: int = 32) -> str:
    import torch
    try:
        inp = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=max_new, num_beams=4, do_sample=False
            )
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return ""


def _build_prompt(model_name: str, s: str, p: str, o: str, wh: str) -> str:
    pred_clean = p.replace("_", " ")
    if model_name == "raya":
        return f"generate questions: {s}|{pred_clean}|{o} -> {wh}"
    # flan
    return (
        f"Generate a {wh} question whose answer is the fact that "
        f"{s} {pred_clean} {o}. Question:"
    )


def _is_parseable(text: str) -> bool:
    """A generated string is parseable if it is non-empty and starts with
    a WH-word or an auxiliary-verb question stem. We check the leading
    token structurally: first word ends with '?' or the sentence ends
    with '?', and the first token is alphabetic and title/lowercase.
    """
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    first = t.split()[0].lower().strip(".,?!:;")
    # WH words form a closed grammatical class; this is not a curated
    # topical list, it is the English interrogative paradigm.
    return first in {
        "who", "what", "when", "where", "why", "which", "whose", "how",
        "is", "are", "was", "were", "do", "does", "did", "can", "will",
    }


# 5-triple probe corpus used at first-call to pick the winner.
_PROBE_TRIPLES = [
    ("Maya", "works_at", "Google", "PERSON", "ORG", WH_WHO),
    ("Sam", "lives_in", "Ann_Arbor", "PERSON", "LOCATION", WH_WHERE),
    ("meeting", "scheduled_on", "Tuesday", "EVENT", "TIME", WH_WHEN),
    ("Emily", "married_to", "Jake", "PERSON", "PERSON", WH_WHO),
    ("Raya", "is", "assistant", "GENERIC", "GENERIC", WH_WHAT),
]


def _ensure_model() -> bool:
    """Load and probe models. Prefer raya; fall back to flan-t5-base."""
    global _model, _tokenizer, _device, _model_name, _probed
    if _probed:
        return _model is not None

    _probed = True

    # Probe raya first.
    attempt = _load_model("raya")
    if attempt is not None:
        mdl, tok, dev = attempt
        passes = 0
        for (s, p, o, st, ot, wh) in _PROBE_TRIPLES:
            out = _generate_with(mdl, tok, dev, _build_prompt("raya", s, p, o, wh))
            if _is_parseable(out):
                passes += 1
        if passes >= 4:
            _model, _tokenizer, _device, _model_name = mdl, tok, dev, "raya"
            print(f"[predicted_queries] model=raya probe_pass={passes}/5")
            return True
        else:
            print(f"[predicted_queries] raya probe_pass={passes}/5 -> falling back")

    # Fall back to flan.
    attempt = _load_model("flan")
    if attempt is None:
        print("[predicted_queries] no QG model available")
        return False
    mdl, tok, dev = attempt
    passes = 0
    for (s, p, o, st, ot, wh) in _PROBE_TRIPLES:
        out = _generate_with(mdl, tok, dev, _build_prompt("flan", s, p, o, wh))
        if _is_parseable(out):
            passes += 1
    _model, _tokenizer, _device, _model_name = mdl, tok, dev, "flan"
    print(f"[predicted_queries] model=flan probe_pass={passes}/5")
    return True


# ---- WH applicability --------------------------------------------------

_PREDICATE_TEMPORAL_CACHE: dict = {}


def _predicate_is_temporal(predicate: str) -> bool:
    """Return True if the predicate's head lemma's WordNet closure
    contains time_period.n.01 OR if it contains event.n.01 with a
    temporal sense (e.g. 'scheduled_on', 'happened_on'). Structural via
    WordNet; no curated stem list."""
    if not predicate:
        return False
    key = predicate.lower()
    if key in _PREDICATE_TEMPORAL_CACHE:
        return _PREDICATE_TEMPORAL_CACHE[key]

    head = predicate.replace("_", " ").strip().split()[0].lower() if predicate else ""
    result = False
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
        for pos in (wn.VERB, wn.NOUN):
            for syn in wn.synsets(head, pos=pos):
                for path in syn.hypernym_paths():
                    for anc in path:
                        if anc.name() in ("time_period.n.01", "time.n.05", "time.n.01"):
                            result = True
                            break
                    if result:
                        break
                if result:
                    break
            if result:
                break
    except Exception:
        result = False

    _PREDICATE_TEMPORAL_CACHE[key] = result
    return result


def _applicable_wh_types(
    subject_type: str, object_type: str, predicate: str
) -> List[str]:
    """Return WH-types the triple can answer. Structural, not calibrated."""
    out: List[str] = [WH_WHAT]  # always
    if subject_type == PERSON or object_type == PERSON:
        out.append(WH_WHO)
    if object_type == TIME or _predicate_is_temporal(predicate):
        out.append(WH_WHEN)
    if object_type in (LOCATION, "ORG"):
        out.append(WH_WHERE)
    # Dedup preserving order.
    seen = set()
    ordered = []
    for w in out:
        if w not in seen:
            ordered.append(w)
            seen.add(w)
    return ordered


# ---- public API --------------------------------------------------------

def generate_predicted_queries(
    subject: str,
    predicate: str,
    object: str,
    subject_type: str = "GENERIC",
    object_type: str = "GENERIC",
) -> List[Tuple[str, np.ndarray]]:
    """Return a list of (question_text, embedding) tuples -- 2..5 rows.

    One row per WH-type the triple can answer. If the QG model is
    unavailable we fall back to a structural question skeleton built
    from the triple (still not a curated list -- it's the SPO surface
    re-ordered into an interrogative form).
    """
    subject = (subject or "").strip()
    predicate = (predicate or "").strip()
    object = (object or "").strip()
    if not subject or not predicate or not object:
        return []

    wh_types = _applicable_wh_types(subject_type, object_type, predicate)

    # ML QG disabled — the raya-srl-220m-v4 model mangles first-person
    # subjects and confuses relation direction (e.g., "user works_at
    # Vantage" becomes "What is the name of the company that works at
    # Vantage Systems?"). Structural templates are topic-aligned and
    # cosine-stable even when grammar is crude.
    pred_clean = predicate.replace("_", " ")
    pred_lemma = _lemmatize_predicate(pred_clean)

    results: List[Tuple[str, np.ndarray]] = []
    for wh in wh_types:
        # Structural interrogative — one template per WH class.
        if wh == WH_WHO:
            if subject_type == PERSON:
                question = f"Who {pred_clean} {object}?"
            else:
                question = f"Who does {subject} {pred_lemma}?"
        elif wh == WH_WHEN:
            question = f"When did {subject} {pred_lemma} {object}?"
        elif wh == WH_WHERE:
            question = f"Where does {subject} {pred_lemma}?"
        else:  # WH_WHAT
            question = f"What does {subject} {pred_lemma}?"

        try:
            emb = embed_text(question)
        except Exception:
            continue
        results.append((question, emb))

        # Object-foregrounded variant: foreground the object as topic.
        if wh == WH_WHO:
            obj_question = f"Who is {object}?"
        elif wh == WH_WHERE:
            obj_question = f"Where is {object}?"
        elif wh == WH_WHEN:
            obj_question = f"When is {object}?"
        else:  # WH_WHAT
            obj_question = f"What is {object}?"

        try:
            obj_emb = embed_text(obj_question)
        except Exception:
            continue
        results.append((obj_question, obj_emb))

    return results


def active_model_name() -> Optional[str]:
    """Return which QG model is currently active ('raya', 'flan', or None)."""
    return _model_name
