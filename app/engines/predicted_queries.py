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
_device = "cuda"
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
        device = "cuda"
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

def _is_canonical_user(subject: str) -> bool:
    """True when the subject is the first-person canonical placeholder.

    Kenotic's decomposer maps I/me/myself to 'user'. This token is
    structurally a PERSON — every 'user' edge is a first-person
    statement about the speaker. Recognizing this lets PQ generation
    fire PERSON-applicable templates and produce entity-agnostic
    variants that cosine-match third-person queries ("Where does Maya
    work?" ↔ "Who works at Vantage Systems?").

    This is the English first-person pronoun paradigm — a closed
    grammatical class, not a curated word list.
    """
    return (subject or "").strip().lower() in ("user", "i", "me", "myself")


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

    When subject is the canonical 'user' placeholder, we additionally
    generate entity-agnostic variants that drop the subject token.
    This is additive — all original PQs are kept, the agnostic variants
    are appended. The rationale: 'user' is a variable standing for the
    speaker's real name, so PQs that rely on predicate+object rather
    than subject will cosine-match third-person queries that use the
    speaker's name.
    """
    subject = (subject or "").strip()
    predicate = (predicate or "").strip()
    object = (object or "").strip()
    if not subject or not predicate or not object:
        return []

    # Structural identity: 'user' IS a person (the speaker). Promote
    # to PERSON so WHO templates fire for first-person edges.
    effective_subject_type = subject_type
    if _is_canonical_user(subject) and subject_type == "GENERIC":
        effective_subject_type = PERSON

    wh_types = _applicable_wh_types(effective_subject_type, object_type, predicate)

    # ── Grammar-aware question generation ──────────────────────────
    # 5 rules from English grammar for question formation:
    #   1. "be" inverts directly (no do-support)
    #   2. Action verbs use do-support, verb → base form
    #   3. WH-word replaces the answer NP, prepositional frame stays
    #   4. Temporal expressions in object → generate WHEN questions
    #   5. Possessive/kinship subjects → generate questions about possessor
    pred_clean = predicate.replace("_", " ")
    pred_lemma = _lemmatize_predicate(pred_clean)
    is_user = _is_canonical_user(subject)

    # Parse object to extract prepositional frame and detect temporals.
    nlp = _get_nlp()
    _obj_doc = nlp(object)

    # Rule 3: Extract prepositional frame from object.
    # "allergic to peanuts" → frame="allergic to", answer_np="peanuts"
    # "married to Sarah in June 2024" → frame="married to", answer_np="Sarah in June 2024"
    _obj_frame = ""
    _obj_answer_np = object
    try:
        for _tok in _obj_doc:
            if _tok.dep_ == "prep" or (_tok.pos_ == "ADP" and _tok.i < len(_obj_doc) - 1):
                _obj_frame = _obj_doc[:_tok.i + 1].text
                _obj_answer_np = _obj_doc[_tok.i + 1:].text
                break
    except Exception:
        pass

    # Extract verb and prep from predicate ("work_at" → verb="work", prep="at")
    _pred_parts = pred_clean.split()
    _verb_base = _pred_parts[0] if _pred_parts else pred_lemma
    _pred_prep = " ".join(_pred_parts[1:]) if len(_pred_parts) > 1 else ""

    # Rule 4: Detect temporal expressions in the object via NER.
    _has_temporal_in_obj = False
    try:
        for ent in _obj_doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                _has_temporal_in_obj = True
                break
    except Exception:
        pass
    # Also detect temporal adverbs/phrases structurally
    if not _has_temporal_in_obj:
        _temporal_deps = {"npadvmod", "advmod", "prep"}
        for tok in _obj_doc:
            if tok.dep_ in _temporal_deps and tok.ent_type_ in ("DATE", "TIME"):
                _has_temporal_in_obj = True
                break

    # Add WHEN to wh_types if temporal detected in object
    if _has_temporal_in_obj and WH_WHEN not in wh_types:
        wh_types.append(WH_WHEN)

    _is_be = pred_lemma == "be"

    results: List[Tuple[str, np.ndarray]] = []
    for wh in wh_types:
        # ── Build the primary question ────────────────────────────
        if wh == WH_WHO:
            if effective_subject_type == PERSON:
                # Subject IS the answer: "Who works at Google?" → Sam
                question = f"Who {pred_clean} {object}?"
            elif _is_be and _obj_frame:
                question = f"Who is {subject} {_obj_frame}?"
            elif _is_be:
                question = f"Who is {subject}?"
            else:
                question = f"Who does {subject} {pred_lemma}?"

        elif wh == WH_WHEN:
            if _is_be:
                question = f"When is {subject} {_obj_frame}?".strip()
                if not question.endswith("?"):
                    question += "?"
            else:
                question = f"When did {subject} {_verb_base}?"

        elif wh == WH_WHERE:
            if _is_be:
                question = f"Where is {subject}?"
            elif _pred_prep:
                question = f"Where does {subject} {_verb_base}?"
            else:
                question = f"Where does {subject} {pred_lemma}?"

        else:  # WH_WHAT
            if _is_be and _obj_frame:
                question = f"What is {subject} {_obj_frame}?"
            elif _is_be:
                question = f"What is {subject}?"
            elif _pred_prep:
                question = f"What does {subject} {_verb_base} {_pred_prep}?"
            else:
                question = f"What does {subject} {pred_lemma}?"

        try:
            emb = embed_text(question)
            results.append((question, emb))
        except Exception:
            pass

        # ── Object-foregrounded variant ───────────────────────────
        # "What is allergic to peanuts?" / "Who is Google?"
        if wh == WH_WHO:
            obj_q = f"Who is {object}?"
        elif wh == WH_WHERE:
            obj_q = f"Where is {object}?"
        elif wh == WH_WHEN:
            obj_q = f"When is {object}?"
        else:
            obj_q = f"What is {object}?"
        try:
            results.append((obj_q, embed_text(obj_q)))
        except Exception:
            pass

        # ── Entity-agnostic variant for canonical-user edges ──────
        if is_user and wh != WH_WHO:
            if wh == WH_WHEN:
                ag_q = f"When was {object}?"
            elif wh == WH_WHERE:
                ag_q = f"Where is {pred_clean} {object}?"
            else:
                ag_q = f"What about {pred_clean} {object}?"
            try:
                results.append((ag_q, embed_text(ag_q)))
            except Exception:
                pass

    # ── Rule 5: Possessive/kinship subjects ───────────────────────
    # If subject looks like "My mom 's name" or "Sam 's brother",
    # generate a question about the possessor.
    if "'s" in subject or subject.lower().startswith("my "):
        poss_q = f"Who is {subject}?"
        try:
            results.append((poss_q, embed_text(poss_q)))
        except Exception:
            pass

    return results


def _syntactic_questions_from_source(source_text: str, subject: str) -> List[str]:
    """Generate WH questions from source_text via spaCy dep parse."""
    nlp = _get_nlp()
    doc = nlp(source_text)
    questions: List[str] = []

    root = None
    for tok in doc:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if root is None:
        return []

    content = root
    for _ in range(3):
        child_comp = None
        for ch in content.children:
            if ch.dep_ in ("xcomp", "ccomp", "pcomp") and ch.pos_ in ("VERB", "AUX"):
                child_comp = ch
                break
        if child_comp is None:
            break
        content = child_comp

    verb_lemma = content.lemma_

    for ch in content.children:
        if ch.dep_ == "dobj":
            questions.append(f"What does {subject} {verb_lemma}?")
            if ch.pos_ in ("NOUN", "PROPN"):
                questions.append(f"What {ch.lemma_} does {subject} {verb_lemma}?")
            break

    for ch in content.children:
        if ch.dep_ == "prep" and ch.pos_ == "ADP":
            prep = ch.lemma_
            for gc in ch.children:
                if gc.dep_ in ("pobj", "pcomp"):
                    if prep in ("about", "into", "of"):
                        questions.append(f"What is {subject} {verb_lemma} {prep}?")
                    elif prep in ("from", "to", "in", "at"):
                        questions.append(f"Where does {subject} {verb_lemma} {prep}?")
                    else:
                        questions.append(f"What does {subject} {verb_lemma} {prep}?")
                    break

    for ch in content.children:
        if ch.dep_ == "ccomp":
            questions.append(f"What did {subject} {verb_lemma}?")
            comp_text = " ".join(
                t.text for t in sorted(ch.subtree, key=lambda t: t.i)
                if not (t.dep_ == "mark" and t.lemma_ == "that")
            )
            if comp_text and len(comp_text) > 5:
                questions.append(comp_text)
            break

    for ch in content.children:
        if ch.dep_ in ("acomp", "attr"):
            questions.append(f"What is {subject}?")
            break

    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            questions.append(f"When did {subject} {verb_lemma}?")
            break

    return questions


def generate_predicted_queries_from_trace(
    td: object,
) -> List[Tuple[str, np.ndarray]]:
    """Source-text-driven PQ generation from TraceDecomposition."""
    subject = getattr(td, 'subject', '') or ''
    predicate = getattr(td, 'predicate', '') or ''
    obj = getattr(td, 'object', '') or ''
    if not subject or not predicate or not obj:
        return []

    results: List[Tuple[str, np.ndarray]] = []

    real_subject = getattr(td, 'relational_subject', '') or subject
    if real_subject.lower() in ("user", "i", "me", "myself"):
        real_subject = subject if subject.lower() not in ("user", "i", "me", "myself") else real_subject

    # Layer 1: Source-text syntactic questions
    source_text = getattr(td, 'source_text', '') or ''
    if source_text and len(source_text) > 10:
        for q in _syntactic_questions_from_source(source_text, real_subject):
            try:
                results.append((q, embed_text(q)))
            except Exception:
                pass

    # Layer 2: Episodic fact as cosine anchor
    episodic_fact = getattr(td, 'episodic_fact', '') or ''
    if episodic_fact and len(episodic_fact) > 5:
        try:
            results.append((episodic_fact, embed_text(episodic_fact)))
        except Exception:
            pass

    # Layer 3: Trace-aware supplements
    emotional_state = getattr(td, 'emotional_state', None)
    emotional_target = getattr(td, 'emotional_target', None)
    schematic_category = getattr(td, 'schematic_category', '') or ''

    if emotional_state:
        try:
            results.append((f"How is {real_subject} feeling?", embed_text(f"How is {real_subject} feeling?")))
        except Exception:
            pass

    if emotional_state and emotional_target:
        q = f"Why is {real_subject} {emotional_state} about {emotional_target}?"
        try:
            results.append((q, embed_text(q)))
        except Exception:
            pass

    if schematic_category and schematic_category != "uncategorized":
        q = f"How is {real_subject}'s {schematic_category} going?"
        try:
            results.append((q, embed_text(q)))
        except Exception:
            pass

    # Dedup
    seen: set = set()
    deduped: List[Tuple[str, np.ndarray]] = []
    for q_text, q_emb in results:
        key = q_text.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append((q_text, q_emb))
    return deduped


def active_model_name() -> Optional[str]:
    """Return which QG model is currently active ('raya', 'flan', or None)."""
    return _model_name
