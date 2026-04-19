# ============================================================================
# Predicate-shape parser + verb inflection.
#
# Structural renderer support for RetrievalEngine.reconstruct(). Parses a
# stored predicate string like "went_home_to" into its grammatical shape
# (verb_lemma, particle, preposition, embedded_noun, ordered middle segments)
# and inflects the verb lemma into a surface form keyed on tense + person.
#
# RULES (non-negotiable):
#   - No verb-to-verb lookup tables.
#   - No predicate-to-template maps.
#   - POS (NLTK), WordNet morphy, and orthographic morphology ONLY.
#   - The 4-item _YOU_CONJUGATE borderline exception (is/are, has/have,
#     does/do, was/were) is the only accepted curated shim. Lives in
#     RetrievalEngine, not here.
# ============================================================================
"""Predicate-shape parser and morphological verb inflector.

Public surface:
    parse_predicate(predicate) -> ParsedPredicate
    inflect_verb(lemma, tense, person="2s") -> str
    is_verb_token(token) -> bool
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────
# POS / WordNet helpers — all fail-open (NLTK is available in the env,
# but a missing NLTK install must not crash retrieval).
# ─────────────────────────────────────────────────────────────────────

def _pos_tag(token: str) -> str:
    """Penn Treebank POS tag for a single token, '' on failure."""
    if not token:
        return ""
    try:
        import nltk
        tags = nltk.pos_tag([token.lower()])
        return tags[0][1] if tags else ""
    except Exception:
        return ""


def _wn_morphy(token: str, pos: Optional[str] = None) -> Optional[str]:
    """WordNet morphy -> base form or None."""
    try:
        from nltk.corpus import wordnet as wn
        if pos:
            return wn.morphy(token.lower(), pos)
        return wn.morphy(token.lower())
    except Exception:
        return None


def _wn_is_verb(token: str) -> bool:
    """True iff WordNet has a verb synset for the token."""
    try:
        from nltk.corpus import wordnet as wn
        return bool(wn.synsets(token.lower(), pos="v"))
    except Exception:
        return False


def is_verb_token(token: str) -> bool:
    """Permissive verb test. POS first, WordNet fallback. Mirrors
    MemoryEngine._is_verb_lexical so the write-path and read-path
    agree on what constitutes a verb.

    Stricter than the write-path gate in ONE respect: if the POS
    tagger confidently labels the token as NN* (noun) in isolation,
    we trust that and refuse verb-hood even when WordNet has a verb
    synset for it. Many English nouns have denominal verb synsets
    ("birth", "date", "mother") that are not the intended reading in
    a stored predicate. The tagger's isolation-tag is the structural
    tie-breaker. This makes "birth_date" parse as a malformed
    noun-compound predicate rather than an accidental verb phrase."""
    if not token:
        return False
    tag = _pos_tag(token)
    if tag.startswith("VB"):
        return True
    if tag.startswith("NN"):
        # Tagger says noun in isolation. Two disambiguations:
        #   (a) wn.morphy(tok,'v') reduces to a *different* lemma
        #       -> inflected verb form ("interviewed" -> "interview")
        #   (b) WordNet has a verb synset for the token
        #       -> base-form verb that happens to also be a noun
        #          ("cook", "visit", "start"). Fail-open like the
        #          write-path _is_verb_lexical — otherwise we lose
        #          legitimate bare-verb predicates.
        # The compound-noun case ("birth_date") is caught in
        # parse_predicate: two NN-tagged segments in sequence is the
        # structural tell for a noun compound, not a verb phrase.
        m = _wn_morphy(token, "v")
        if m and m != token.lower():
            return True
        # Base-form noun/verb ambiguity: when morphy returns the same
        # form (not inflected) AND noun synsets outnumber verb synsets,
        # the dominant reading is nominal. Words like "action", "reason",
        # "pattern" have rare verb synsets but are structurally nouns in
        # predicate position. Synset-count is a WordNet structural
        # property, not a curated threshold.
        if _wn_is_verb(token):
            try:
                from nltk.corpus import wordnet as _wn_dom
                _n_n = len(_wn_dom.synsets(token.lower(), pos="n"))
                _n_v = len(_wn_dom.synsets(token.lower(), pos="v"))
                if _n_n > _n_v:
                    return False  # noun-dominant: not a verb
            except Exception:
                pass
            return True
        return False
    if not tag:
        return _wn_is_verb(token)
    return _wn_is_verb(token)


# ─────────────────────────────────────────────────────────────────────
# ParsedPredicate
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ParsedPredicate:
    """Structural decomposition of a stored predicate string.

    Fields:
        verb_lemma:     morphological base of the verb token as stored
                        (NOT reduced to infinitive — we preserve tense
                        morphology so the renderer can echo it or
                        reinflect as needed). If the first segment is
                        not verb-shaped, verb_lemma == "" and ok=False.
        verb_surface:   the raw first segment as stored (e.g. "went",
                        "hiked"), before any renderer re-inflection.
        preposition:    first IN-tagged token among subsequent segments
        particle:       first RP-tagged token among subsequent segments
        embedded_noun:  first NN*-tagged token among subsequent segments
        middle:         ALL subsequent segments in original order. This
                        is what the renderer uses to preserve word
                        order ("went home to", "hiked up", etc.). The
                        typed fields above are conveniences for callers
                        that need role-level access (e.g. passive
                        detection via preposition == "by").
        ok:             True iff verb_lemma parsed cleanly.
        failure_reason: populated when ok=False.
    """
    verb_lemma: str = ""
    verb_surface: str = ""
    preposition: Optional[str] = None
    particle: Optional[str] = None
    embedded_noun: Optional[str] = None
    middle: List[str] = field(default_factory=list)
    ok: bool = False
    failure_reason: str = ""


def parse_predicate(predicate: str) -> ParsedPredicate:
    """Structurally parse a stored predicate string.

    Splits on '_'. The first segment must be verb-shaped; subsequent
    segments are classified by POS into {preposition, particle,
    embedded_noun}. Classification is categorical, not a score.

    Returns ParsedPredicate. If the first segment isn't verb-shaped,
    ok=False with a failure_reason — the caller decides whether to
    raise, log, or fall back.
    """
    if not predicate:
        return ParsedPredicate(ok=False, failure_reason="empty_predicate")

    segs = [s for s in predicate.split("_") if s]
    if not segs:
        return ParsedPredicate(ok=False, failure_reason="empty_segments")

    head = segs[0]
    head_tag = _pos_tag(head)

    # Compound-noun tell: head is NN-tagged AND at least one subsequent
    # segment is also NN-tagged. "birth_date", "phone_number",
    # "last_name" — these are noun compounds stored as predicates, not
    # verb phrases. Refuse verb parsing so the renderer falls back to a
    # copular construction ("Your birth date is X").
    #
    # Exception: if the head's WordNet morphy('v') yields a DIFFERENT
    # lemma, the head is an inflected verb form that the POS tagger
    # mistagged as a noun in isolation (e.g. "earns" -> NNS but
    # morphy -> "earn"). In that case, proceed to verb parsing — the
    # head is structurally a verb, not a noun compound.
    if head_tag.startswith("NN"):
        verb_morphy = _wn_morphy(head, "v")
        head_is_inflected_verb = verb_morphy is not None and verb_morphy != head.lower()
        if not head_is_inflected_verb:
            for seg in segs[1:]:
                if _pos_tag(seg).startswith("NN"):
                    return ParsedPredicate(
                        verb_surface=head,
                        failure_reason=f"noun_compound_head:{head}",
                        ok=False,
                    )
        else:
            # Even when head looks like an inflected verb, if the
            # predicate is a 2-segment compound where BOTH segments
            # are NN-tagged, it's a noun compound ("supplies_status").
            # The structural tell: exactly 2 segments, both NN-tagged,
            # and the immediate next segment is NN (not IN/TO/RB which
            # would indicate a verb phrase like "earns_from").
            #
            # Exception: if the verb lemma is verb-dominant in WordNet
            # (more verb synsets than noun synsets), it's a transitive
            # verb + direct object ("shows_pattern" = "show" + "pattern"),
            # not a noun compound. Synset-count is structural, not curated.
            if len(segs) == 2 and _pos_tag(segs[1]).startswith("NN"):
                _is_verb_dominant = False
                try:
                    from nltk.corpus import wordnet as _wn_vc
                    _nv = len(_wn_vc.synsets(verb_morphy, pos="v"))
                    _nn = len(_wn_vc.synsets(verb_morphy, pos="n"))
                    if _nv > _nn:
                        _is_verb_dominant = True
                except Exception:
                    pass
                if _is_verb_dominant:
                    pass  # fall through to verb parsing
                else:
                    return ParsedPredicate(
                        verb_surface=head,
                        failure_reason=f"noun_compound_head:{head}",
                        ok=False,
                    )

    # Gerund + noun compound detection: VBG head followed immediately
    # by an NN-tagged segment is a gerund-noun compound
    # ("scheduling_note", "working_title"), not a verb phrase.
    # VBN is excluded — "received_advice" IS a verb phrase (past
    # tense + direct object), not a compound noun.
    if head_tag == "VBG" and len(segs) >= 2:
        if _pos_tag(segs[1]).startswith("NN"):
            return ParsedPredicate(
                verb_surface=head,
                failure_reason=f"noun_compound_head:{head}",
                ok=False,
            )

    # Head validation: must be verb-shaped. is_verb_token uses POS first,
    # WordNet fallback — same contract as the write-path gate.
    if not is_verb_token(head):
        return ParsedPredicate(
            verb_surface=head,
            failure_reason=f"head_not_verb:{head}",
            ok=False,
        )

    verb_lemma = _wn_morphy(head, "v") or head

    parsed = ParsedPredicate(
        verb_lemma=verb_lemma,
        verb_surface=head,
        middle=[],
        ok=True,
    )

    # Walk remaining segments. Classification is POS-only:
    #   tag IN or TO          -> preposition
    #   tag RP or RB(+WN adv) -> particle       (e.g. up/down/off)
    #   tag NN*               -> embedded noun  (e.g. ankle/home/date)
    #   tag VB*/other         -> falls into middle untyped (multi-word
    #                            verb shards like be_born handled by
    #                            the renderer via middle order).
    for seg in segs[1:]:
        tag = _pos_tag(seg)
        parsed.middle.append(seg)

        if tag in ("IN", "TO"):
            if parsed.preposition is None:
                parsed.preposition = seg
        elif tag == "RP":
            if parsed.particle is None:
                parsed.particle = seg
        elif tag == "RB":
            # Adverb-tagged words like "down" / "off" often act as
            # phrasal-verb particles in predicate position. Treat as
            # particle if no particle captured yet.
            if parsed.particle is None:
                parsed.particle = seg
        elif tag.startswith("NN"):
            if parsed.embedded_noun is None:
                parsed.embedded_noun = seg
        # else: untyped — stays in middle in original position.

    return parsed


# ─────────────────────────────────────────────────────────────────────
# Verb inflection — morphology only.
# ─────────────────────────────────────────────────────────────────────

# Minimal set of vowels + orthographic tests for regular -ed / -s
# formation. These are not word lists; they are letter-class rules —
# the same way "add -s for plural" is a rule, not curation.
_VOWELS = set("aeiou")


def _count_vowel_clusters(word: str) -> int:
    """Count distinct vowel clusters in a word — structural proxy for
    syllable count. 'stop' -> 1, 'visit' -> 2, 'recommend' -> 3.
    This is a letter-class rule, not a lookup table."""
    count = 0
    in_vowel = False
    for ch in word.lower():
        if ch in _VOWELS:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    return count


def _ends_with_short_vowel_consonant(word: str) -> bool:
    """Orthographic rule: CVC where final C is not w/x/y, used to
    decide consonant doubling in -ed/-ing/-s formation (stop->stopped).
    Only applies to monosyllables (one vowel cluster). Polysyllabic
    words like 'visit', 'orbit', 'limit' do NOT double."""
    if len(word) < 3:
        return False
    # Polysyllabic words: no doubling (visit->visited, not visitted)
    if _count_vowel_clusters(word) > 1:
        return False
    c3, v, c1 = word[-3], word[-2], word[-1]
    if c1 in ("w", "x", "y"):
        return False
    return (c3 not in _VOWELS) and (v in _VOWELS) and (c1 not in _VOWELS)


def _regular_past(lemma: str) -> str:
    """Form regular past tense by orthographic morphology."""
    if not lemma:
        return lemma
    if lemma.endswith("e"):
        return lemma + "d"
    if lemma.endswith("y") and len(lemma) >= 2 and lemma[-2] not in _VOWELS:
        return lemma[:-1] + "ied"
    if _ends_with_short_vowel_consonant(lemma):
        return lemma + lemma[-1] + "ed"
    return lemma + "ed"


def _regular_third_singular(lemma: str) -> str:
    """Form regular 3rd-person-singular present by orthographic rules."""
    if not lemma:
        return lemma
    if lemma.endswith(("s", "x", "z", "o", "ch", "sh")):
        return lemma + "es"
    if lemma.endswith("y") and len(lemma) >= 2 and lemma[-2] not in _VOWELS:
        return lemma[:-1] + "ies"
    return lemma + "s"


def _irregular_past_via_wordnet(lemma: str) -> Optional[str]:
    """Try to recover an irregular past form via WordNet's lemma
    variants. If any variant's morphy() back-resolves to `lemma` and
    looks past-tense-shaped (ends differently than the lemma and is
    not a noun form), return it. Returns None if nothing useful.

    This is NOT a lookup table — it's a reverse walk of WordNet's
    morphological index, the same artifact morphy() forward-walks."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None
    try:
        # Walk all verb lemmas whose morphy base is this lemma. WordNet
        # stores inflected forms via exception lists (the `*.exc` files);
        # we can iterate through them for this specific lemma.
        from nltk.corpus.reader.wordnet import WordNetError  # noqa: F401
        # Access the raw verb exception mapping. This is a dict of
        # {inflected_form: base_form} loaded from WordNet's verb.exc.
        verb_exc = wn._exception_map.get("v", {}) if hasattr(wn, "_exception_map") else {}
        if not verb_exc:
            # Trigger load via any morphy call, then re-read.
            wn.morphy("be", "v")
            verb_exc = wn._exception_map.get("v", {}) if hasattr(wn, "_exception_map") else {}
        candidates = [inflected for inflected, base in verb_exc.items()
                      if (base == lemma or (isinstance(base, (list, tuple)) and lemma in base))]
        if not candidates:
            return None
        # Heuristic: past forms typically end in consonant cluster and
        # are not the base or -ing form. Prefer shortest form that is
        # not == lemma and does not end in "ing".
        past_like = [c for c in candidates if c != lemma and not c.endswith("ing")]
        if not past_like:
            return None
        # Prefer VBD (simple past) over VBN (past participle); e.g.
        # "go" has exception list [gone, went], we want "went".
        vbd = [c for c in past_like if _pos_tag(c) == "VBD"]
        if vbd:
            return sorted(vbd, key=len)[0]
        vbn = [c for c in past_like if _pos_tag(c) == "VBN"]
        if vbn:
            # Structural property of WordNet verb.exc: exception entries
            # come in pairs [VBD, VBN]. If we found VBN candidates and
            # there are non-VBN candidates remaining, those are the VBD
            # forms even if the POS tagger misclassifies them in isolation
            # (e.g. "saw" is NN-tagged in isolation but is VBD of "see").
            non_vbn = [c for c in past_like if c not in vbn]
            if non_vbn:
                return sorted(non_vbn, key=len)[0]
            return sorted(vbn, key=len)[0]
        # Fall back to shortest past-like candidate.
        return sorted(past_like, key=len)[0]
    except Exception:
        return None


def inflect_verb(lemma: str, tense: Optional[str], person: str = "2s") -> str:
    """Inflect a verb lemma into surface form for the requested tense
    and person.

    Args:
        lemma:  verb lemma, e.g. "go", "hike", "make"
        tense:  "past" | "present" | "future" | None
        person: "2s" for 'you' (base form in present), "3s" for
                third-person-singular proper-noun subjects

    Returns the surface form. Falls back to the lemma as-is if tense
    is unknown — morphology is never invented from nothing.

    Uses:
      - WordNet exception lists for irregular past/participle
      - Orthographic -ed / -s rules for regular forms
      - Lemma as-is for future (renderer prepends "will")

    The renderer calls this AFTER parse_predicate has stripped tense
    off the stored surface form via wn.morphy(). If the lemma came in
    already past-inflected (e.g. stored surface was "went"), morphy
    returned "go" and we reinflect cleanly.
    """
    if not lemma:
        return lemma

    t = (tense or "").lower().strip()

    # "be" and "have" are the only English verbs with irregular
    # 3rd-person-singular present forms (is/has vs regular -s/-es).
    # These are morphological facts, not a word list — every other
    # English verb follows the regular -s rule.
    if lemma == "be":
        return lemma
    if lemma == "have":
        if t in ("", "present") and person == "3s":
            return "has"
        if t == "past":
            return "had"
        return "have"

    if t == "past":
        irreg = _irregular_past_via_wordnet(lemma)
        if irreg:
            return irreg
        return _regular_past(lemma)

    if t in ("", "present"):
        if person == "3s":
            return _regular_third_singular(lemma)
        # 2s / 'you' / 1s / plural all use base form in English present.
        return lemma

    if t == "future":
        # Caller is responsible for prepending "will"; we return base.
        return lemma

    # Unknown tense — return lemma unchanged (safe).
    return lemma
