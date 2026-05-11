"""
spaCy-based extraction: triples, NER entities, and temporal expressions.

Replaces T5 <triplets> and <roles> extraction with spaCy DependencyMatcher
patterns. The spaCy model is loaded once at module level; ``extract_all()``
runs a single ``nlp(text)`` call and returns triples, entities, and temporals
from the same Doc.

If spaCy or the model is not installed, ``extract_all()`` returns empty
results and ``is_available()`` returns False. There is no fallback --
T5 extraction was removed (2026-04-24).

Public API:
    extract_all(text: str) -> dict
    is_available() -> bool
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# spaCy singleton
# ---------------------------------------------------------------------------

_nlp = None
_UNAVAILABLE = False
_COREF_AVAILABLE = False
_COREF_CHECKED = False


def _load_nlp():
    """Load spaCy model once. Prefer en_core_web_trf, fall back to sm."""
    global _nlp, _UNAVAILABLE, _COREF_AVAILABLE, _COREF_CHECKED
    if _UNAVAILABLE:
        return None
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        for model_name in ("en_core_web_trf", "en_core_web_sm"):
            try:
                _nlp = spacy.load(model_name)
                print(f"[spacy_extractor] Loaded {model_name}")
                break
            except OSError:
                continue
        if _nlp is None:
            print("[spacy_extractor] No spaCy model available")
            _UNAVAILABLE = True
            return None

        # Try to add coreferee for coreference resolution
        if not _COREF_CHECKED:
            _COREF_CHECKED = True
            try:
                import coreferee  # noqa: F401
                _nlp.add_pipe("coreferee")
                _COREF_AVAILABLE = True
                print("[spacy_extractor] coreferee loaded — coref resolution enabled")
            except (ImportError, Exception) as exc:
                _COREF_AVAILABLE = False
                print(f"[spacy_extractor] coreferee not available ({exc}) — coref skipped")

        return _nlp
    except ImportError:
        print("[spacy_extractor] spaCy not installed")
        _UNAVAILABLE = True
        return None


def is_available() -> bool:
    """True if spaCy and a model are ready."""
    return _load_nlp() is not None


# ---------------------------------------------------------------------------
# NER label mapping
# ---------------------------------------------------------------------------

_LABEL_MAP: Dict[str, str] = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "TIME",
    "TIME": "TIME",
    "CARDINAL": "QUANTITY",
    "MONEY": "QUANTITY",
    "PERCENT": "QUANTITY",
    "QUANTITY": "QUANTITY",
    "EVENT": "EVENT",
}


def _map_label(spacy_label: str) -> str:
    return _LABEL_MAP.get(spacy_label, "GENERIC")


# ---------------------------------------------------------------------------
# Predicate normalization
# ---------------------------------------------------------------------------

def _normalize_predicate(tokens: list) -> str:
    """Build a lowercase_underscore predicate from a list of spaCy tokens.

    Uses each token's lemma. Compound predicates (verb + preposition,
    verb + particle) are joined with underscores.
    """
    parts: List[str] = []
    for tok in tokens:
        lemma = tok.lemma_.lower().strip()
        if lemma:
            parts.append(lemma)
    return "_".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Historical detection
# ---------------------------------------------------------------------------

def _is_historical(verb_token) -> bool:
    """Return True if the verb is past tense with no present-tense markers.

    Uses spaCy morphological analysis: Tense=Past without any
    present-tense auxiliaries in the same clause.
    """
    morph = verb_token.morph
    tense = morph.get("Tense")
    if not tense or "Past" not in tense:
        return False
    # Check for present-tense auxiliaries that would indicate
    # present-perfect ("have worked") rather than simple past.
    for child in verb_token.children:
        if child.dep_ == "aux":
            child_tense = child.morph.get("Tense")
            if child_tense and "Pres" in child_tense:
                return False
    return True


# ---------------------------------------------------------------------------
# Triple extraction via dependency patterns
# ---------------------------------------------------------------------------

def _collect_subtree_text(token, exclude_deps: Optional[set] = None) -> str:
    """Collect the text of a token's subtree, optionally excluding
    certain dependency relations. Returns the text span in document order."""
    exclude_deps = exclude_deps or set()
    tokens_in_subtree = []
    for t in token.subtree:
        if t == token:
            tokens_in_subtree.append(t)
        elif t.dep_ not in exclude_deps:
            tokens_in_subtree.append(t)
    tokens_in_subtree.sort(key=lambda t: t.i)
    return " ".join(t.text for t in tokens_in_subtree).strip()


def _get_subject_text(subj_token) -> str:
    """Get the full noun phrase text for a subject token."""
    # Collect compound modifiers and the token itself
    parts = []
    for child in subj_token.children:
        if child.dep_ in ("compound", "amod", "det", "poss"):
            parts.append(child)
    parts.append(subj_token)
    parts.sort(key=lambda t: t.i)
    return " ".join(t.text for t in parts).strip()


def _get_object_text(obj_token) -> str:
    """Get the full noun phrase text for an object token, including
    its subtree but excluding clausal dependents."""
    exclude = {"relcl", "advcl", "ccomp", "xcomp", "acl"}
    return _collect_subtree_text(obj_token, exclude_deps=exclude)


# ---------------------------------------------------------------------------
# Triple quality gate (LoCoMo + real dialogue flood protection)
# ---------------------------------------------------------------------------

def _morph_has(tok, feat: str, value: str) -> bool:
    try:
        return value in tok.morph.get(feat)
    except Exception:
        return False


def _is_effectively_empty(text: str) -> bool:
    return not text or not text.strip()


def _is_vague_subject_token(subj_tok) -> bool:
    if subj_tok is None:
        return True
    # Structural: pronoun subjects flood the graph unless they're first-person.
    # Use morph features rather than lexical lists.
    if subj_tok.pos_ == "PRON":
        # Drop interrogatives/relatives as "subjects" ("who/what/which ...")
        if _morph_has(subj_tok, "PronType", "Int") or _morph_has(subj_tok, "PronType", "Rel"):
            return True
        # Keep only 1st-person pronouns as stable subjects.
        if not _morph_has(subj_tok, "Person", "1"):
            return True
    return False


def _is_vague_object_token(obj_tok) -> bool:
    if obj_tok is None:
        return True
    # Copular complements that are pure adjectives ("is great") are not stable facts.
    if obj_tok.pos_ in ("ADJ", "ADV", "INTJ"):
        return True
    if obj_tok.pos_ == "PRON":
        # Pronoun objects are usually ungrounded unless coref rewrites them.
        # Keep only first-person reflexives ("myself/ourselves") via Person=1.
        if not _morph_has(obj_tok, "Person", "1"):
            return True
    return False


def _object_has_concrete_anchor(obj_tok) -> bool:
    """Heuristic: object subtree contains something that grounds the fact.

    This is structural (POS/NER/number), not a phrase list:
    - proper noun (PROPN)
    - named entity type (ent_type_)
    - number-like token
    """
    if obj_tok is None:
        return False
    try:
        for t in obj_tok.subtree:
            if t.pos_ == "PROPN":
                return True
            if getattr(t, "ent_type_", ""):
                return True
            if getattr(t, "like_num", False) or t.pos_ == "NUM":
                return True
    except Exception:
        return False
    return False


def _is_storeworthy_subject(subj_tok, subj_text: str, doc) -> bool:
    """Subject must be a named entity, self-reference, proper noun, or
    concrete common noun. Rejects pronouns (except 1st-person), determiners,
    adjectives, adverbs, interjections, and determiner-led vague phrases.

    Structural only — uses POS, NER, morphology. No word lists.
    """
    if subj_tok is None or _is_effectively_empty(subj_text):
        return False

    # Self-reference (1st person) is always valid — will be normalized to "user"
    if subj_tok.pos_ == "PRON" and _morph_has(subj_tok, "Person", "1"):
        return True

    # Named entity from NER is valid
    for ent in doc.ents:
        if subj_tok.text.strip() in ent.text:
            return True

    # Proper noun is valid
    if subj_tok.pos_ == "PROPN":
        return True

    # Structural POS rejection: pronouns (non-1st-person), determiners,
    # adjectives, adverbs, interjections are not storeworthy subjects
    if subj_tok.pos_ in ("PRON", "DET", "ADJ", "ADV", "INTJ"):
        return False

    # Determiner-led subjects where the head is still vague:
    # check if the leftmost token in the subject span is a DET
    # (e.g. "The those posters" — malformed NP with determiner lead)
    try:
        subtree_tokens = sorted(subj_tok.subtree, key=lambda t: t.i)
        if subtree_tokens and subtree_tokens[0].pos_ == "DET":
            # Allow if the head noun is a proper noun or named entity
            if subj_tok.pos_ not in ("NOUN", "PROPN"):
                return False
            # Allow "the company" but reject "the those posters" (DET DET NOUN)
            det_count = sum(1 for t in subtree_tokens if t.pos_ == "DET")
            if det_count > 1:
                return False
    except Exception:
        pass  # fail-open

    return True


def _is_asserted_fact(verb_token) -> bool:
    """Return True only if the verb expresses an asserted fact.

    Rejects:
    - Modal auxiliaries as children (should/would/could/might/may)
    - Conditional markers (if/unless/whether)
    - Copula "be" with adjective complement (backchannel: "that is great")

    Structural only — uses dep labels, POS tags, Penn tags. No word lists.
    """
    try:
        # Modal auxiliary child = hypothetical, not asserted
        for child in verb_token.children:
            if child.dep_ == "aux" and getattr(child, "tag_", "") == "MD":
                return False

        # Conditional subordinating marker = not asserted
        for child in verb_token.children:
            if child.dep_ == "mark" and child.pos_ == "SCONJ":
                # Structural: subordinating conjunctions that introduce
                # conditionality have dep_="mark". Check lemma for the
                # conditional subset via morph if possible; fall back to
                # checking if the clause is advcl (conditional clause).
                if verb_token.dep_ == "advcl":
                    return False

        # Copular "be" with BARE adjectival complement = evaluation/backchannel.
        # "That is great" (bare ADJ, no further content) = noise.
        # "I am allergic to tree nuts" (ADJ + prep attachment) = fact.
        # "Caroline is transgender" (ADJ used as identity) = fact.
        # The structural tell: if the ADJ has children (prep, conj, advcl),
        # it's a substantive fact. If it's bare (no children), it's noise.
        if verb_token.lemma_ == "be":
            for child in verb_token.children:
                if child.dep_ == "acomp" and child.pos_ == "ADJ":
                    # Check if the adjective has substantive children
                    adj_children = [c for c in child.children if c.dep_ in ("prep", "conj", "advcl", "xcomp", "ccomp")]
                    if not adj_children:
                        # Bare ADJ — but check the subject.
                        # PROPN or 1st-person subject = identity statement
                        # ("Caroline is transgender", "I am single")
                        # DET/demonstrative subject = backchannel
                        # ("That is great", "It is awesome")
                        subj_children = [c for c in verb_token.children if c.dep_ in ("nsubj", "nsubjpass")]
                        if subj_children:
                            subj = subj_children[0]
                            if subj.pos_ == "PROPN":
                                pass  # identity — keep
                            elif subj.pos_ == "PRON" and _morph_has(subj, "Person", "1"):
                                pass  # self-reported state — keep
                            else:
                                return False  # backchannel
                        else:
                            return False

    except Exception:
        pass  # fail-open: treat as asserted if analysis unavailable

    return True


def _is_storeworthy_object(obj_tok, obj_text: str, doc) -> bool:
    """Object must have substance. Bare pronouns, bare adjectives,
    bare adverbs, bare interjections, and bare determiners are not facts.

    Structural only — uses POS tags. No word lists.
    """
    if obj_tok is None or _is_effectively_empty(obj_text):
        return False

    # Single-token objects: check POS directly.
    # Reject bare pronouns, adverbs, interjections, determiners.
    # Allow ADJ — identity statements like "Caroline is transgender"
    # or "I am single" have ADJ objects that are real facts. The
    # copula factuality gate handles backchannel ("that is great")
    # at the verb level, so we don't need to double-reject here.
    tokens = obj_text.strip().split()
    if len(tokens) == 1:
        if obj_tok.pos_ in ("PRON", "ADV", "INTJ", "DET"):
            # Exception: 1st-person reflexive ("myself") is okay
            if obj_tok.pos_ == "PRON" and _morph_has(obj_tok, "Person", "1"):
                return True
            return False

    return True


def _normalize_subject(subj_text: str, subj_tok) -> str:
    """Map first-person pronouns to the canonical 'user' self-reference.

    This matches the contract from the old T5 extraction path where
    first-person subjects are stored as 'user' in the graph.
    """
    if subj_tok is not None and subj_tok.pos_ == "PRON":
        if _morph_has(subj_tok, "Person", "1"):
            return "user"
    return subj_text


def _should_keep_triple(subj_tok, pred_lemma: str, obj_tok, subj_text: str, obj_text: str, doc, verb_token=None) -> bool:
    # Basic empties
    if _is_effectively_empty(subj_text) or _is_effectively_empty(obj_text):
        return False

    # --- Filter 1: Subject must be storeworthy ---
    if not _is_storeworthy_subject(subj_tok, subj_text, doc):
        return False

    # --- Filter 2: Event must be asserted (not hypothetical/conditional/backchannel) ---
    if verb_token is not None and not _is_asserted_fact(verb_token):
        return False

    # --- Filter 3: Object must have substance ---
    if not _is_storeworthy_object(obj_tok, obj_text, doc):
        return False

    # Legacy flood protection: vague subjects (kept as secondary check)
    if _is_vague_subject_token(subj_tok):
        return False

    # Legacy flood protection: vague objects
    if _is_vague_object_token(obj_tok):
        return False

    # Reject copula-style "be" edges whose complement is adjective-like.
    # (We store affect elsewhere; these triples drown the graph.)
    if pred_lemma in ("be", "is", "are", "was", "were", "am"):
        # If the object head is ADJ/ADV, drop.
        if obj_tok is not None and obj_tok.pos_ in ("ADJ", "ADV", "INTJ"):
            return False

    # "We/us" subjects create a lot of conversational filler edges. Keep only
    # when the object is concretely grounded (named entity / proper noun / number).
    if subj_tok is not None and subj_tok.pos_ == "PRON":
        # Structural proxy for "we/us": 1st-person plural pronoun.
        if _morph_has(subj_tok, "Person", "1") and _morph_has(subj_tok, "Number", "Plur"):
            if not _object_has_concrete_anchor(obj_tok):
                return False

    # Hard cap: don't store huge clause objects as "facts"
    if len(obj_text) > 180:
        return False

    return True


def _extract_triples_from_doc(doc) -> List[Tuple[str, str, str, bool]]:
    """Extract (subject, predicate, object, is_historical) tuples
    from a spaCy Doc using dependency structure."""
    triples: List[Tuple[str, str, str, bool]] = []

    for sent in doc.sents:
        for token in sent:
            # Only process verbs and auxiliaries acting as roots
            if token.pos_ not in ("VERB", "AUX"):
                continue
            # Structural: skip modal auxiliary roots (Penn tag "MD") which
            # mostly express suggestions/possibility rather than stable facts.
            if token.pos_ == "AUX" and (getattr(token, "tag_", "") == "MD"):
                continue

            subjects = []
            objects = []
            pred_tokens = [token]
            historical = _is_historical(token)

            for child in token.children:
                dep = child.dep_

                # --- Subjects ---
                if dep in ("nsubj", "nsubjpass"):
                    subjects.append(child)

                # --- Direct objects, attributes, predicate nominals ---
                elif dep in ("dobj", "attr", "oprd"):
                    objects.append(child)

                # --- Prepositional attachment: "work at Google" ---
                elif dep == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            pred_tokens.append(child)  # the preposition
                            objects.append(pobj)

                # --- xcomp: "started training for a marathon" ---
                elif dep == "xcomp":
                    xcomp_text = _collect_subtree_text(
                        child, exclude_deps={"nsubj", "nsubjpass"}
                    )
                    if xcomp_text:
                        # Create a synthetic object from the xcomp subtree
                        objects.append(child)

                # --- Clausal complement as object ---
                elif dep == "acomp":
                    objects.append(child)

            # --- Copula/attr pattern: "I am a software engineer" ---
            # When a noun/adj is the ROOT with a copula child
            if token.pos_ in ("NOUN", "ADJ", "PROPN") and token.dep_ == "ROOT":
                cop_children = [c for c in token.children if c.dep_ == "cop"]
                subj_children = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                if cop_children and subj_children:
                    for subj in subj_children:
                        subj_text = _get_subject_text(subj)
                        # Predicate is "be" (the copula), object is the ROOT token
                        obj_text = _get_object_text(token)
                        if subj_text and obj_text:
                            cop_historical = _is_historical(cop_children[0])
                            if _should_keep_triple(subj, "be", token, subj_text, obj_text, doc, verb_token=cop_children[0]):
                                subj_text = _normalize_subject(subj_text, subj)
                                triples.append((subj_text, "is", obj_text, cop_historical))
                    continue  # Already handled this token

            # --- Apposition: "my sister Mika" ---
            for child in token.children:
                if child.dep_ == "appos":
                    head_text = _get_subject_text(token) if token.dep_ in ("nsubj", "nsubjpass", "pobj", "dobj") else token.text
                    appos_text = _get_object_text(child)
                    if head_text and appos_text:
                        if _should_keep_triple(token, "be", child, head_text, appos_text, doc):
                            head_text = _normalize_subject(head_text, token)
                            triples.append((head_text, "is", appos_text, False))

            # --- Possessive patterns: "my dog's name is Luna" ---
            for child in token.children:
                if child.dep_ == "poss":
                    poss_text = child.text
                    owned_text = _get_subject_text(token)
                    if poss_text and owned_text:
                        # This creates a possessive relationship but we
                        # let the SVO pattern handle the main predication
                        pass

            # Build triples from collected subjects and objects
            if not subjects or not objects:
                continue

            predicate = _normalize_predicate(pred_tokens)
            if not predicate:
                continue

            for subj in subjects:
                subj_text = _get_subject_text(subj)
                if not subj_text:
                    continue
                for obj in objects:
                    if obj.pos_ == "VERB":
                        # xcomp case: use the full subtree as object text
                        obj_text = _collect_subtree_text(
                            obj, exclude_deps={"nsubj", "nsubjpass"}
                        )
                    else:
                        obj_text = _get_object_text(obj)
                    if not obj_text:
                        continue
                    pred_lemma = (token.lemma_ or "").lower() or predicate.split("_", 1)[0]
                    if _should_keep_triple(subj, pred_lemma, obj, subj_text, obj_text, doc, verb_token=token):
                        subj_text = _normalize_subject(subj_text, subj)
                        triples.append((subj_text, predicate, obj_text, historical))

    # Also scan for appositions at the noun level across the whole doc
    for token in doc:
        if token.dep_ == "appos":
            head = token.head
            head_text = _get_subject_text(head)
            appos_text = _get_object_text(token)
            if head_text and appos_text:
                pred_lemma = "be"
                if _should_keep_triple(head, pred_lemma, token, head_text, appos_text, doc):
                    head_text = _normalize_subject(head_text, head)
                    triples.append((head_text, "is", appos_text, False))

    return triples


# ---------------------------------------------------------------------------
# Coreference resolution
# ---------------------------------------------------------------------------

def _resolve_coreferences(doc, nlp_fn) -> "Doc":
    """Replace pronoun mentions with their most recent named antecedent.

    Structural heuristic coref — no external coref library needed.
    Uses spaCy NER + POS to resolve:
      - she/her/hers → most recent PERSON entity (feminine or unknown)
      - he/him/his → most recent PERSON entity (masculine or unknown)
      - they/them/their → most recent PERSON entity (any)

    After substitution, re-parses so dep structure reflects resolved entities.
    Returns original doc unchanged if no resolutions are made.
    """

    try:
        # Collect PERSON entities from NER as candidate antecedents
        # ordered by position (most recent = later in text)
        person_entities = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                person_entities.append(ent.text)

        if not person_entities:
            return doc

        # Map: third-person pronouns → most recent PERSON entity
        # Structural: she/he/they are closed-class English pronouns.
        _THIRD_PERSON = {
            "she", "her", "hers", "herself",
            "he", "him", "his", "himself",
            "they", "them", "their", "theirs", "themselves",
        }

        replacements: Dict[int, str] = {}
        # Track the most recent PERSON entity seen so far (left-to-right)
        last_person: Optional[str] = None

        for token in doc:
            # Update last_person when we pass a PERSON entity
            if token.ent_type_ == "PERSON" and token.ent_iob_ == "B":
                # Find the full entity span
                for ent in doc.ents:
                    if ent.start == token.i and ent.label_ == "PERSON":
                        last_person = ent.text
                        break

            # Replace third-person pronouns with last_person
            if (token.pos_ == "PRON"
                    and token.text.lower() in _THIRD_PERSON
                    and last_person is not None):
                replacements[token.i] = last_person

        if not replacements:
            return doc

        # Rebuild text with replacements
        new_tokens: List[str] = []
        for token in doc:
            if token.i in replacements:
                new_tokens.append(replacements[token.i])
            else:
                new_tokens.append(token.text_with_ws if token.whitespace_ else token.text)

        resolved_text = " ".join(new_tokens)
        return nlp_fn(resolved_text)

    except Exception:
        return doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_all(text: str) -> Dict[str, list]:
    """Extract triples, NER entities, and temporal expressions from text.

    Returns::

        {
            "triples": [(subject, predicate, object, is_historical), ...],
            "entities": [(text, label), ...],
            "temporals": [str, ...],
        }

    If spaCy is unavailable, returns empty lists for all keys.
    """
    empty: Dict[str, list] = {"triples": [], "entities": [], "temporals": []}

    if not text or not text.strip():
        return empty

    nlp = _load_nlp()
    if nlp is None:
        return empty

    doc = nlp(text.strip())

    # --- Coreference resolution ---
    # If coreferee is available, resolve pronouns to antecedents and
    # re-parse so that triple extraction sees resolved entity names
    # instead of pronouns (e.g., "She" -> "Mika").
    doc = _resolve_coreferences(doc, nlp)

    # --- Triples ---
    triples = _extract_triples_from_doc(doc)

    # --- NER entities ---
    entities: List[Tuple[str, str]] = []
    seen_ents: set = set()
    for ent in doc.ents:
        ent_text = ent.text.strip()
        if not ent_text:
            continue
        key = (ent_text.lower(), ent.label_)
        if key not in seen_ents:
            entities.append((ent_text, _map_label(ent.label_)))
            seen_ents.add(key)

    # --- Temporals ---
    temporals: List[str] = []
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            temporal_text = ent.text.strip()
            if temporal_text:
                temporals.append(temporal_text)

    return {
        "triples": triples,
        "entities": entities,
        "temporals": temporals,
    }
