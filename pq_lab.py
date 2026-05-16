"""
PQ Lab — standalone test harness for predicted question generation.

Copy of grammar_engine.generate_predicted_questions (lines 2079-2324)
plus a new experimental version to iterate on.

Usage:
    python pq_lab.py
"""

import spacy
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")


def _has_kinds(lemma: str) -> bool:
    """WordNet reverse check: does this noun have hyponyms (sub-kinds)?
    If yes, 'What kind of X?' is a natural question."""
    synsets = wn.synsets(lemma, pos=wn.NOUN)
    if not synsets:
        return False
    return len(synsets[0].hyponyms()) > 0


_ORDINAL_HEAD = wn.synset('ordinal.a.02')


def _is_ordinal(lemma: str) -> bool:
    """WordNet check: is this adjective an ordinal (first, second, third...)?
    Ordinals → 'Which X?' not 'What X?'"""
    for s in wn.synsets(lemma, pos=wn.ADJ):
        if s == _ORDINAL_HEAD:
            return True
    for s in wn.synsets(lemma, pos=wn.ADJ_SAT):
        if _ORDINAL_HEAD in s.similar_tos():
            return True
    return False


# ---------------------------------------------------------------------------
# ORIGINAL — exact copy from grammar_engine.py lines 2079-2324
# ---------------------------------------------------------------------------

def generate_predicted_questions_ORIGINAL(sent_doc, root, subject_name):
    """Original Chomsky transformation — strips content from target slot."""
    if not root or root.pos_ not in ("VERB", "AUX"):
        return []

    subj_tok = None
    for ch in root.children:
        if ch.dep_ in ("nsubj", "nsubjpass"):
            subj_tok = ch
            break
    if not subj_tok:
        return []

    subj_is_real = (
        subj_tok.pos_ == "PROPN"
        or subj_tok.ent_type_ in ("PERSON", "ORG", "GPE")
        or any(t.pos_ == "PROPN" for t in subj_tok.subtree)
        or any(t.ent_type_ == "PERSON" for t in subj_tok.subtree)
    )
    if not subj_is_real:
        return []

    if sum(1 for t in sent_doc if t.pos_ not in ("PUNCT", "INTJ", "X")) < 4:
        return []

    has_answer = any(
        ch.dep_ in ("dobj", "attr", "acomp", "ccomp") for ch in root.children
    )
    if not has_answer:
        for ch in root.children:
            if ch.dep_ in ("prep", "agent"):
                if any(gc.dep_ == "pobj" for gc in ch.children):
                    has_answer = True
                    break
            if ch.dep_ == "xcomp":
                if any(gc.dep_ == "dobj" for gc in ch.children):
                    has_answer = True
                    break
        if not has_answer:
            has_answer = any(ent.label_ in ("DATE", "TIME") for ent in sent_doc.ents)
    if not has_answer:
        return []

    subject_indices = {t.i for t in subj_tok.subtree}
    vc_indices = {root.i}
    for ch in root.children:
        if ch.dep_ in ("aux", "auxpass"):
            vc_indices.add(ch.i)
        if ch.dep_ == "prt":
            vc_indices.add(ch.i)
    verb_chain = sorted(vc_indices)

    is_be_main = root.lemma_ == "be" and len(verb_chain) == 1
    has_aux = any(sent_doc[i].dep_ in ("aux", "auxpass") for i in verb_chain if i != root.i)

    neg_tok = None
    for ch in root.children:
        if ch.dep_ == "neg":
            neg_tok = ch
            break

    exclude = {tok.i for tok in sent_doc if tok.pos_ in ("PUNCT", "INTJ")}
    for ch in root.children:
        if ch.dep_ == "cc":
            exclude.add(ch.i)
        if ch.dep_ in ("conj", "advcl"):
            exclude.update(t.i for t in ch.subtree)
    if len(sent_doc) > 0 and sent_doc[0].pos_ in ("ADV", "INTJ") and sent_doc[0].dep_ == "advmod":
        exclude.add(0)
        if len(sent_doc) > 1 and sent_doc[1].text == ",":
            exclude.add(1)

    def _form(target_indices):
        target_toks = [sent_doc[i] for i in target_indices]
        if all(t.pos_ in ("PRON", "DET", "PART", "PUNCT") for t in target_toks):
            return None

        target_first = sent_doc[min(target_indices)]
        prep_parent = None
        for i in target_indices:
            tok = sent_doc[i]
            if tok.dep_ == "pobj" and tok.head.dep_ in ("prep", "agent"):
                prep_parent = tok.head
                break

        wh, absorb = "What", False
        if target_first.ent_type_ in ("DATE", "TIME"):
            wh, absorb = "When", True
        elif prep_parent and prep_parent.lemma_ in ("at", "in", "on", "to", "from", "near"):
            if target_first.ent_type_ in ("DATE", "TIME"):
                wh, absorb = "When", True
            else:
                wh, absorb = "Where", True
        elif prep_parent and prep_parent.lemma_ == "for" and any(t.pos_ == "NUM" for t in target_first.subtree):
            wh, absorb = "How long", True
        elif target_first.pos_ == "ADJ" and target_first.dep_ in ("acomp", "dobj", "attr"):
            wh = "How"
        elif prep_parent and prep_parent.dep_ == "agent":
            wh, absorb = "By whom", True
        elif target_first.ent_type_ == "PERSON":
            wh = "Who"

        effective_target = set(target_indices)
        strand_prep = None
        if prep_parent:
            if absorb:
                effective_target.update(t.i for t in prep_parent.subtree)
            else:
                effective_target.update(t.i for t in target_first.subtree)
                strand_prep = prep_parent.text
                effective_target.add(prep_parent.i)

        remainder = sorted(
            set(range(len(sent_doc))) - subject_indices - set(verb_chain) - effective_target - exclude
        )

        if target_indices & subject_indices:
            parts = [wh]
            for i in sorted(set(verb_chain) | set(remainder)):
                parts.append(sent_doc[i].text)
            if strand_prep:
                parts.append(strand_prep)
            return " ".join(parts) + "?"

        neg_i = neg_tok.i if neg_tok else -1
        moved_i = set()

        if has_aux:
            first_aux_i = verb_chain[0]
            if sent_doc[first_aux_i].dep_ not in ("aux", "auxpass"):
                for vi in verb_chain:
                    if sent_doc[vi].dep_ in ("aux", "auxpass"):
                        first_aux_i = vi
                        break
            moved_i = subject_indices | {first_aux_i} | effective_target | exclude
            if neg_i >= 0:
                moved_i.add(neg_i)
            parts = [wh, sent_doc[first_aux_i].text, subject_name]
            if neg_i >= 0:
                parts.append(neg_tok.text)
            for i in range(len(sent_doc)):
                if i not in moved_i:
                    parts.append(sent_doc[i].text)
            if strand_prep:
                parts.append(strand_prep)
            return " ".join(parts) + "?"

        if is_be_main:
            moved_i = subject_indices | {root.i} | effective_target | exclude
            if neg_i >= 0:
                moved_i.add(neg_i)
            parts = [wh, root.text, subject_name]
            if neg_i >= 0:
                parts.append(neg_tok.text)
            for i in range(len(sent_doc)):
                if i not in moved_i:
                    parts.append(sent_doc[i].text)
            if strand_prep:
                parts.append(strand_prep)
            return " ".join(parts) + "?"

        do = "did" if root.tag_ == "VBD" else ("does" if root.tag_ == "VBZ" else "do")
        moved_i = subject_indices | effective_target | exclude
        if neg_i >= 0:
            moved_i.add(neg_i)
        parts = [wh, do, subject_name]
        if neg_i >= 0:
            parts.append(neg_tok.text)
        for i in range(len(sent_doc)):
            if i in moved_i:
                continue
            if i == root.i:
                parts.append(root.lemma_)
            else:
                parts.append(sent_doc[i].text)
        if strand_prep:
            parts.append(strand_prep)
        return " ".join(parts) + "?"

    questions = []
    seen = set()

    def _add(indices):
        q = _form(indices)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            questions.append(q)

    for ch in root.children:
        if ch.dep_ == "dobj":
            _add({t.i for t in ch.subtree})
        elif ch.dep_ == "attr":
            _add({t.i for t in ch.subtree})
        elif ch.dep_ == "acomp":
            _add({t.i for t in ch.subtree})
        elif ch.dep_ in ("prep", "agent"):
            for gc in ch.children:
                if gc.dep_ == "pobj":
                    _add({t.i for t in gc.subtree})
                    break
        elif ch.dep_ == "ccomp":
            _add({t.i for t in ch.subtree})
        elif ch.dep_ == "xcomp":
            for gc in ch.children:
                if gc.dep_ == "dobj":
                    _add({t.i for t in gc.subtree})
                    break

    for ent in sent_doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            _add({t.i for t in sent_doc if ent.start <= t.i < ent.end})
            break

    return questions[:4]


# ---------------------------------------------------------------------------
# EXPERIMENTAL — iterate here
# ---------------------------------------------------------------------------

def generate_predicted_questions_NEW(sent_doc, root, subject_name):
    """Content-preserving PQ generator.

    English question formation gives us 4 independent transformations.
    Each preserves different content words:

    Rule 1 — Full WH-replacement (Chomsky standard):
        Replace target constituent entirely with WH-word.
        "Jon loves contemporary dance" → "What does Jon love?"
        Content: low (target stripped)

    Rule 2 — Yes/No inversion (Subject-Aux Inversion without WH):
        Move aux/do to front. Keep everything.
        "Jon loves contemporary dance" → "Does Jon love contemporary dance?"
        Content: maximum (nothing stripped)

    Rule 3 — Subject WH (WH replaces subject, object stays):
        "Jon loves contemporary dance" → "Who loves contemporary dance?"
        Content: high (only subject stripped, object preserved)

    Rule 4 — Partial WH (replace modifier, keep head noun):
        For modified NPs: WH + head_noun + do-support + subject + verb
        "Jon loves contemporary dance" → "What kind of dance does Jon love?"
        Content: good (head noun preserved, modifier → WH)
    """
    if not root or root.pos_ not in ("VERB", "AUX"):
        return []

    # === GATES (same as original) ===
    subj_tok = None
    for ch in root.children:
        if ch.dep_ in ("nsubj", "nsubjpass"):
            subj_tok = ch
            break
    if not subj_tok:
        return []

    subj_is_real = (
        subj_tok.pos_ == "PROPN"
        or subj_tok.ent_type_ in ("PERSON", "ORG", "GPE")
        or any(t.pos_ == "PROPN" for t in subj_tok.subtree)
        or any(t.ent_type_ == "PERSON" for t in subj_tok.subtree)
    )
    if not subj_is_real:
        return []

    if sum(1 for t in sent_doc if t.pos_ not in ("PUNCT", "INTJ", "X")) < 4:
        return []

    # === DECOMPOSE ===
    subject_indices = {t.i for t in subj_tok.subtree}
    vc_indices = {root.i}
    for ch in root.children:
        if ch.dep_ in ("aux", "auxpass"):
            vc_indices.add(ch.i)
        if ch.dep_ == "prt":
            vc_indices.add(ch.i)
    verb_chain = sorted(vc_indices)

    is_be_main = root.lemma_ == "be" and len(verb_chain) == 1
    has_aux = any(sent_doc[i].dep_ in ("aux", "auxpass") for i in verb_chain if i != root.i)

    neg_tok = None
    for ch in root.children:
        if ch.dep_ == "neg":
            neg_tok = ch
            break

    exclude = {tok.i for tok in sent_doc if tok.pos_ in ("PUNCT", "INTJ")}
    for ch in root.children:
        if ch.dep_ == "cc":
            exclude.add(ch.i)
        if ch.dep_ in ("conj", "advcl"):
            exclude.update(t.i for t in ch.subtree)
    if len(sent_doc) > 0 and sent_doc[0].pos_ in ("ADV", "INTJ") and sent_doc[0].dep_ == "advmod":
        exclude.add(0)
        if len(sent_doc) > 1 and sent_doc[1].text == ",":
            exclude.add(1)

    # Collect content tokens (everything except punct/intj)
    content_indices = sorted(set(range(len(sent_doc))) - exclude)

    questions = []
    seen = set()

    def _add(q):
        if q and q.lower() not in seen:
            seen.add(q.lower())
            questions.append(q)

    # ------------------------------------------------------------------
    # RULE 2: Yes/No inversion — maximum content preservation
    # Move aux/do before subject. Keep everything else in place.
    # "Jon loves contemporary dance" → "Does Jon love contemporary dance?"
    # ------------------------------------------------------------------
    def _yes_no():
        parts = []
        neg_i = neg_tok.i if neg_tok else -1

        if has_aux:
            # Find first aux
            first_aux_i = verb_chain[0]
            if sent_doc[first_aux_i].dep_ not in ("aux", "auxpass"):
                for vi in verb_chain:
                    if sent_doc[vi].dep_ in ("aux", "auxpass"):
                        first_aux_i = vi
                        break
            parts.append(sent_doc[first_aux_i].text.capitalize())
            # Subject
            for i in sorted(subject_indices):
                parts.append(sent_doc[i].text)
            # Neg if present
            if neg_i >= 0:
                parts.append(neg_tok.text)
            # Rest in document order, skip moved tokens
            moved = subject_indices | {first_aux_i} | exclude
            if neg_i >= 0:
                moved.add(neg_i)
            for i in content_indices:
                if i not in moved:
                    parts.append(sent_doc[i].text)

        elif is_be_main:
            parts.append(root.text.capitalize())
            for i in sorted(subject_indices):
                parts.append(sent_doc[i].text)
            if neg_i >= 0:
                parts.append(neg_tok.text)
            moved = subject_indices | {root.i} | exclude
            if neg_i >= 0:
                moved.add(neg_i)
            for i in content_indices:
                if i not in moved:
                    parts.append(sent_doc[i].text)

        else:
            # Do-support
            do = "Did" if root.tag_ == "VBD" else ("Does" if root.tag_ == "VBZ" else "Do")
            parts.append(do)
            for i in sorted(subject_indices):
                parts.append(sent_doc[i].text)
            if neg_i >= 0:
                parts.append(neg_tok.text)
            moved = subject_indices | exclude
            if neg_i >= 0:
                moved.add(neg_i)
            for i in content_indices:
                if i not in moved:
                    if i == root.i:
                        parts.append(root.lemma_)  # base form with do-support
                    else:
                        parts.append(sent_doc[i].text)

        return " ".join(parts) + "?" if parts else None

    _add(_yes_no())

    # ------------------------------------------------------------------
    # RULE 3: Subject WH — replace subject with Who/What, keep object
    # "Jon loves contemporary dance" → "Who loves contemporary dance?"
    # ------------------------------------------------------------------
    def _subject_wh():
        # Choose WH based on subject type
        wh = "Who" if (
            subj_tok.ent_type_ in ("PERSON",)
            or subj_tok.pos_ == "PROPN"
        ) else "What"

        parts = [wh]
        # Everything in document order except subject and excludes
        for i in content_indices:
            if i not in subject_indices and i not in exclude:
                parts.append(sent_doc[i].text)

        return " ".join(parts) + "?" if len(parts) > 1 else None

    _add(_subject_wh())

    # ------------------------------------------------------------------
    # RULE 4: Partial WH — for modified NPs, replace modifier, keep head
    # "Jon loves [contemporary dance]" → "What kind of dance does Jon love?"
    # "Jon visited [beautiful Paris]" → no modifier, skip
    # ------------------------------------------------------------------
    def _partial_wh():
        results = []
        for ch in root.children:
            if ch.dep_ not in ("dobj", "attr"):
                continue
            # Find the head noun and its modifiers
            head = ch
            children_deps = {t.dep_ for t in ch.children}

            # Modifier dep tag determines the WH-phrase:
            #   amod (adjective)  → "What X"       — "What dance does Jon love?"
            #   nummod (number)   → "How many X"    — "How many dollars did Chen donate?"
            #   compound (noun)   → "What kind of X" — "What kind of dog did Karen adopt?"
            has_amod = "amod" in children_deps
            has_nummod = "nummod" in children_deps
            has_compound = "compound" in children_deps
            has_poss = "poss" in children_deps

            if not (has_amod or has_nummod or has_compound or has_poss):
                continue

            head_text = head.text

            # Modifier dep tag → WH-phrase (English question formation)
            # Generate one question per modifier type present — fill all slots
            #   poss     → "Whose X"
            #   nummod   → "How many X"
            #   amod     → "Which X" (ordinal) or "What X" (quality)
            #   compound → "What kind of X"
            wh_phrases = []
            if has_poss:
                wh_phrases.append(f"Whose {head_text}")
            if has_nummod:
                wh_phrases.append(f"How many {head_text}")
            if has_amod:
                amod_toks = [t for t in ch.children if t.dep_ == "amod"]
                if any(_is_ordinal(t.lemma_) for t in amod_toks):
                    wh_phrases.append(f"Which {head_text}")
                else:
                    wh_phrases.append(f"What {head_text}")
            if has_compound:
                wh_phrases.append(f"What kind of {head_text}")

            neg_i = neg_tok.i if neg_tok else -1

            for wh_phrase in wh_phrases:
                if has_aux:
                    first_aux_i = verb_chain[0]
                    if sent_doc[first_aux_i].dep_ not in ("aux", "auxpass"):
                        for vi in verb_chain:
                            if sent_doc[vi].dep_ in ("aux", "auxpass"):
                                first_aux_i = vi
                                break
                    parts = [wh_phrase, sent_doc[first_aux_i].text, subject_name]
                    if neg_i >= 0:
                        parts.append(neg_tok.text)
                    target_all = {t.i for t in ch.subtree}
                    moved = subject_indices | {first_aux_i} | target_all | exclude
                    if neg_i >= 0:
                        moved.add(neg_i)
                    for i in range(len(sent_doc)):
                        if i not in moved:
                            parts.append(sent_doc[i].text)

                elif is_be_main:
                    parts = [wh_phrase, root.text, subject_name]
                    if neg_i >= 0:
                        parts.append(neg_tok.text)
                    target_all = {t.i for t in ch.subtree}
                    moved = subject_indices | {root.i} | target_all | exclude
                    if neg_i >= 0:
                        moved.add(neg_i)
                    for i in range(len(sent_doc)):
                        if i not in moved:
                            parts.append(sent_doc[i].text)

                else:
                    do = "did" if root.tag_ == "VBD" else ("does" if root.tag_ == "VBZ" else "do")
                    parts = [wh_phrase, do, subject_name]
                    if neg_i >= 0:
                        parts.append(neg_tok.text)
                    target_all = {t.i for t in ch.subtree}
                    moved = subject_indices | target_all | exclude
                    if neg_i >= 0:
                        moved.add(neg_i)
                    for i in range(len(sent_doc)):
                        if i in moved:
                            continue
                        if i == root.i:
                            parts.append(root.lemma_)
                        else:
                            parts.append(sent_doc[i].text)

                results.append(" ".join(parts) + "?")
        return results

    for q in _partial_wh():
        _add(q)

    # ------------------------------------------------------------------
    # RULE 5: Hypernym question — ask about the object's category
    # For named entities: NER label → category word
    # For bare common nouns: WordNet one-level-up
    # "Jon visited Paris" → "What city did Jon visit?"
    # "Jon is reading The Lean Startup" → "What work did Jon read?"
    # ------------------------------------------------------------------
    def _verb_definition_categories(verb_lemma):
        """Step 1: Extract object category nouns from verb's WordNet definitions.
        Tries first 3 senses. Returns all unique nouns found.
        visit -> {'place'}, cook -> {'meal'}, drive -> {'vehicle'}.
        """
        cats = []
        seen = set()
        for s in wn.synsets(verb_lemma, pos=wn.VERB)[:3]:
            defn_doc = nlp(s.definition())
            for tok in defn_doc:
                if tok.pos_ == "NOUN" and tok.dep_ in ("dobj", "pobj", "attr", "conj"):
                    w = tok.text.lower()
                    if w not in seen:
                        seen.add(w)
                        cats.append(tok.text)
                    break  # one per sense
        return cats

    def _verb_adj_hop_categories(verb_lemma):
        """Step 2: If verb definition uses adjective/participle instead of noun,
        follow it to its related verb, then extract noun from THAT definition.
        read -> 'something written' -> write -> 'work'.
        """
        cats = []
        seen = set()
        for s in wn.synsets(verb_lemma, pos=wn.VERB)[:2]:
            defn_doc = nlp(s.definition())
            for tok in defn_doc:
                if tok.dep_ in ("relcl", "acomp", "amod", "attr") and tok.pos_ in ("ADJ", "VERB"):
                    for vs in wn.synsets(tok.lemma_, pos=wn.VERB)[:2]:
                        inner_doc = nlp(vs.definition())
                        for itok in inner_doc:
                            if itok.pos_ == "NOUN" and itok.dep_ in ("dobj", "pobj", "attr", "conj"):
                                w = itok.text.lower()
                                if w not in seen:
                                    seen.add(w)
                                    cats.append(itok.text)
                                break
        return cats

    def _ner_explain_categories(ent_type):
        """Step 3: Parse spaCy's own NER label description to extract category.
        Not our hardcoding -- spaCy's metadata.
        WORK_OF_ART -> ['book'], GPE -> ['country'].
        """
        import spacy as _spacy_mod
        desc = _spacy_mod.explain(ent_type)
        if not desc:
            return []
        desc_doc = nlp(desc)
        cats = []
        seen = set()
        for tok in desc_doc:
            if tok.pos_ == "NOUN":
                w = tok.lemma_.lower()
                if w not in seen:
                    seen.add(w)
                    cats.append(w)
        return cats

    def _object_hypernym_categories(word, is_proper):
        """Step 4: WordNet hypernym of the object noun itself.
        Proper nouns use instance_hypernyms (Paris -> capital).
        Common nouns use regular hypernyms (biryani -> dish).
        """
        cats = []
        syns = wn.synsets(word.lower(), pos=wn.NOUN)
        if not syns:
            return cats
        if is_proper:
            for s in syns:
                ih = s.instance_hypernyms()
                if ih:
                    lemma = ih[0].lemmas()[0].name().replace("_", " ")
                    cats.append(lemma)
                    break
        if not cats:
            h = syns[0].hypernyms()
            if h:
                lemma = h[0].lemmas()[0].name().replace("_", " ")
                if lemma != word.lower() and len(lemma.split()) <= 2:
                    cats.append(lemma)
        return cats

    def _hypernym_wh():
        results = []
        for ch in root.children:
            if ch.dep_ not in ("dobj", "attr"):
                continue

            is_proper = ch.pos_ == "PROPN" or ch.ent_type_ != ""

            # Collect ALL candidates from all 4 steps. No priority, no threshold.
            # Reconstruction engine decides which PQ matches the query.
            all_cats = set()
            for c in _verb_definition_categories(root.lemma_):
                all_cats.add(c)
            for c in _verb_adj_hop_categories(root.lemma_):
                all_cats.add(c)
            if ch.ent_type_:
                for c in _ner_explain_categories(ch.ent_type_):
                    all_cats.add(c)
            for c in _object_hypernym_categories(ch.text, is_proper):
                all_cats.add(c)

            for category in all_cats:
                wh_phrase = f"What {category}"
                neg_i = neg_tok.i if neg_tok else -1

                if has_aux:
                    first_aux_i = verb_chain[0]
                    if sent_doc[first_aux_i].dep_ not in ("aux", "auxpass"):
                        for vi in verb_chain:
                            if sent_doc[vi].dep_ in ("aux", "auxpass"):
                                first_aux_i = vi
                                break
                    parts = [wh_phrase, sent_doc[first_aux_i].text, subject_name]
                    if neg_i >= 0:
                        parts.append(neg_tok.text)
                    target_all = {t.i for t in ch.subtree}
                    moved = subject_indices | {first_aux_i} | target_all | exclude
                    if neg_i >= 0:
                        moved.add(neg_i)
                    for i in range(len(sent_doc)):
                        if i not in moved:
                            parts.append(sent_doc[i].text)

                elif is_be_main:
                    parts = [wh_phrase, root.text, subject_name]
                    if neg_i >= 0:
                        parts.append(neg_tok.text)
                    target_all = {t.i for t in ch.subtree}
                    moved = subject_indices | {root.i} | target_all | exclude
                    if neg_i >= 0:
                        moved.add(neg_i)
                    for i in range(len(sent_doc)):
                        if i not in moved:
                            parts.append(sent_doc[i].text)

                else:
                    do = "did" if root.tag_ == "VBD" else ("does" if root.tag_ == "VBZ" else "do")
                    parts = [wh_phrase, do, subject_name]
                    if neg_i >= 0:
                        parts.append(neg_tok.text)
                    target_all = {t.i for t in ch.subtree}
                    moved = subject_indices | target_all | exclude
                    if neg_i >= 0:
                        moved.add(neg_i)
                    for i in range(len(sent_doc)):
                        if i in moved:
                            continue
                        if i == root.i:
                            parts.append(root.lemma_)
                        else:
                            parts.append(sent_doc[i].text)

                results.append(" ".join(parts) + "?")
        return results

    for q in _hypernym_wh():
        _add(q)

    # ------------------------------------------------------------------
    # RULE 1: Full WH-replacement (original) — still useful as one variant
    # ------------------------------------------------------------------
    orig = generate_predicted_questions_ORIGINAL(sent_doc, root, subject_name)
    for q in orig:
        _add(q)

    return questions[:8]


# ---------------------------------------------------------------------------
# Helper: parse a sentence and find root
# ---------------------------------------------------------------------------

def _get_root(doc):
    for tok in doc:
        if tok.dep_ == "ROOT":
            return tok
    return None


def run_test(text, speaker):
    """Run both generators on a sentence and compare."""
    doc = nlp(text)
    root = _get_root(doc)

    print(f"\n{'='*70}")
    print(f"INPUT: \"{text}\"")
    print(f"SPEAKER: {speaker}")
    print(f"ROOT: {root.text if root else 'None'} ({root.pos_ if root else ''})")

    # Show dep parse for debugging
    print(f"\nDEP PARSE:")
    for tok in doc:
        print(f"  {tok.i:2d} {tok.text:<15s} {tok.pos_:<6s} {tok.dep_:<10s} head={tok.head.text}")

    orig = generate_predicted_questions_ORIGINAL(doc, root, speaker)
    new = generate_predicted_questions_NEW(doc, root, speaker)

    print(f"\nORIGINAL PQs:")
    for i, q in enumerate(orig):
        print(f"  PQ{i+1}: {q}")
    if not orig:
        print(f"  (none)")

    print(f"\nNEW PQs:")
    for i, q in enumerate(new):
        print(f"  PQ{i+1}: {q}")
    if not new:
        print(f"  (none)")

    return orig, new


# ---------------------------------------------------------------------------
# Test cases from the user's diagnosis
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("PQ LAB — interactive mode")
    print("Type a sentence. Speaker is auto-detected from the subject.")
    print("Type 'q' to quit.\n")

    while True:
        text = input(">>> ").strip()
        if not text or text.lower() == "q":
            break
        doc = nlp(text)
        root = _get_root(doc)
        # Auto-detect speaker from nsubj
        speaker = "someone"
        if root:
            for ch in root.children:
                if ch.dep_ in ("nsubj", "nsubjpass"):
                    speaker = ch.text
                    break
        run_test(text, speaker)
