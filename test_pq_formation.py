"""Test: 4-branch English question formation on all 40 patterns."""
import spacy
nlp = spacy.load("en_core_web_md")


def _get_root(doc):
    for tok in doc:
        if tok.dep_ == "ROOT":
            return tok
    return None


def form_question(doc, target_dep, subject_name):
    root = _get_root(doc)
    if not root:
        return ""

    # === COLLECT STRUCTURE ===
    auxes = sorted(
        [ch for ch in root.children if ch.dep_ in ("aux", "auxpass")],
        key=lambda t: t.i,
    )
    subj_tok = None
    for ch in root.children:
        if ch.dep_ in ("nsubj", "nsubjpass"):
            subj_tok = ch
            break

    subject = subject_name
    is_be_main = root.lemma_ == "be" and not auxes
    prt_tok = None
    for ch in root.children:
        if ch.dep_ == "prt":
            prt_tok = ch
            break
    neg_tok = None
    for ch in root.children:
        if ch.dep_ == "neg":
            neg_tok = ch
            break

    # === FIND TARGET + PREP TO STRAND ===
    target_tok = None
    prep_to_strand = None

    if target_dep == "nsubj":
        target_tok = subj_tok
    elif target_dep == "dobj":
        for ch in root.children:
            if ch.dep_ == "dobj":
                target_tok = ch
                break
        if not target_tok:
            for ch in root.children:
                if ch.dep_ == "xcomp":
                    for gc in ch.children:
                        if gc.dep_ == "dobj":
                            target_tok = gc
                            break
                    break
    elif target_dep == "attr":
        for ch in root.children:
            if ch.dep_ == "attr":
                target_tok = ch
                break
    elif target_dep == "acomp":
        for ch in root.children:
            if ch.dep_ == "acomp":
                target_tok = ch
                break
    elif target_dep == "pobj":
        for ch in root.children:
            if ch.dep_ == "prep":
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        target_tok = gc
                        prep_to_strand = ch
                        break
                if target_tok:
                    break
        if not target_tok and prt_tok:
            for ch in root.children:
                if ch.dep_ == "prep" and ch.i > prt_tok.i:
                    for gc in ch.children:
                        if gc.dep_ == "pobj":
                            target_tok = gc
                            prep_to_strand = ch
                            break
                    if target_tok:
                        break
    elif target_dep == "ccomp":
        for ch in root.children:
            if ch.dep_ == "ccomp":
                target_tok = ch
                break
    elif target_dep == "xcomp":
        for ch in root.children:
            if ch.dep_ == "xcomp":
                target_tok = ch
                break
    elif target_dep == "advmod":
        for ch in root.children:
            if ch.dep_ in ("advmod", "npadvmod"):
                target_tok = ch
                break
    elif target_dep == "agent":
        for ch in root.children:
            if ch.dep_ == "agent":
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        target_tok = gc
                        prep_to_strand = ch
                        break
                break
    elif target_dep == "dative":
        for ch in root.children:
            if ch.dep_ == "dative":
                target_tok = ch
                break

    if not target_tok:
        return f"(no {target_dep} found)"

    # === SELECT WH-WORD ===
    wh = "What"
    if target_dep == "nsubj":
        wh = "Who" if target_tok.text[0].isupper() and target_tok.pos_ in ("PROPN", "PRON", "NOUN") else "What"
        if target_tok.ent_type_ == "PERSON" or target_tok.text.lower() in ("someone", "somebody"):
            wh = "Who"
    elif target_dep == "acomp":
        wh = "How"
    elif target_dep == "advmod":
        if target_tok.ent_type_ in ("DATE", "TIME") or target_tok.text.lower() in ("yesterday", "today", "tomorrow"):
            wh = "When"
        else:
            wh = "How"
    elif target_dep == "pobj" and prep_to_strand:
        p = prep_to_strand.lemma_
        if p in ("at", "in", "on", "to", "from", "near"):
            if target_tok.ent_type_ in ("DATE", "TIME"):
                wh = "When"
            else:
                wh = "Where"
        elif p == "for" and any(t.pos_ == "NUM" for t in target_tok.subtree):
            wh = "How long"
        if target_tok.ent_type_ == "PERSON":
            wh = "Whom"
    elif target_dep == "agent":
        wh = "By whom"
    elif target_dep == "dative":
        wh = "Whom"
    elif target_dep == "attr":
        if target_tok.ent_type_ == "PERSON":
            wh = "Who"

    absorb_prep = wh in ("Where", "When", "How long")

    # === SETS FOR EXCLUSION ===
    target_indices = {t.i for t in target_tok.subtree}
    subj_indices = {t.i for t in subj_tok.subtree} if subj_tok else set()
    first_aux_i = {auxes[0].i} if auxes else set()
    neg_i = {neg_tok.i} if neg_tok else set()
    prt_i = {prt_tok.i} if prt_tok else set()
    absorbed_i = set()
    if absorb_prep and prep_to_strand:
        absorbed_i = {t.i for t in prep_to_strand.subtree}
    strand_i = set()
    if prep_to_strand and not absorb_prep:
        strand_i = {t.i for t in target_tok.subtree}
    punct = {tok.i for tok in doc if tok.text in (".", "!", "?", ",")}

    is_subject_q = target_dep == "nsubj"

    # === BRANCH 1: SUBJECT QUESTION ===
    if is_subject_q:
        parts = []
        inserted_wh = False
        for tok in doc:
            if tok.i in target_indices:
                if not inserted_wh:
                    parts.append(wh)
                    inserted_wh = True
            elif tok.i in punct:
                continue
            else:
                parts.append(tok.text)
        return " ".join(parts) + "?"

    # Helper: remaining tokens after excluding sets
    def _remaining(extra_exclude=set()):
        exclude = target_indices | subj_indices | first_aux_i | absorbed_i | strand_i | neg_i | prt_i | punct | extra_exclude
        return [tok.text for tok in doc if tok.i not in exclude]

    neg_str = neg_tok.text if neg_tok else ""

    # === BRANCH 2: AUX EXISTS ===
    if auxes:
        first_aux = auxes[0].text
        remaining = _remaining({auxes[0].i})
        parts = [wh, first_aux]
        if neg_str:
            parts.append(neg_str)
        parts.append(subject)
        parts.extend(remaining)
        if prep_to_strand and not absorb_prep:
            parts.append(prep_to_strand.text)
        return " ".join(parts) + "?"

    # === BRANCH 3: BE AS MAIN VERB ===
    if is_be_main:
        be_form = root.text
        remaining = _remaining({root.i})
        parts = [wh, be_form]
        if neg_str:
            parts.append(neg_str)
        parts.append(subject)
        parts.extend(remaining)
        if prep_to_strand and not absorb_prep:
            parts.append(prep_to_strand.text)
        return " ".join(parts) + "?"

    # === BRANCH 4: DO-SUPPORT ===
    tag = root.tag_
    do = "did" if tag == "VBD" else ("does" if tag == "VBZ" else "do")
    verb_base = root.lemma_

    remaining = _remaining({root.i})
    parts = [wh, do]
    if neg_str:
        parts.append(neg_str)
    parts.append(subject)
    parts.append(verb_base)
    if prt_tok:
        parts.append(prt_tok.text)
    parts.extend(remaining)
    if prep_to_strand and not absorb_prep:
        parts.append(prep_to_strand.text)
    return " ".join(parts) + "?"


# ========================================================================
# ALL 40+ ROWS
# ========================================================================
tests = [
    (1, "She likes chocolate.", "She", "dobj", "What does She like?"),
    (2, "They play tennis.", "They", "dobj", "What do They play?"),
    (3, "He bought a car.", "He", "dobj", "What did He buy?"),
    (4, "John likes chocolate.", "John", "nsubj", "Who likes chocolate?"),
    (5, "Someone broke the window.", "Someone", "nsubj", "Who broke the window?"),
    (6, "She is reading a book.", "She", "dobj", "What is She reading?"),
    (7, "He was watching television.", "He", "dobj", "What was He watching?"),
    (8, "Someone is knocking on the door.", "Someone", "nsubj", "Who is knocking on the door?"),
    (9, "She has visited Paris.", "She", "dobj", "What has She visited?"),
    (10, "He had finished the report.", "He", "dobj", "What had He finished?"),
    (11, "Someone has eaten my sandwich.", "Someone", "nsubj", "Who has eaten my sandwich?"),
    (12, "She has been reading that book.", "She", "dobj", "What has She been reading?"),
    (13, "He had been watching the show.", "He", "dobj", "What had He been watching?"),
    (14, "They will build a bridge.", "They", "dobj", "What will They build?"),
    (15, "Someone will fix it.", "Someone", "nsubj", "Who will fix it?"),
    (16, "She will have finished the project.", "She", "dobj", "What will She have finished?"),
    (17, "He will be reading a book.", "He", "dobj", "What will He be reading?"),
    (18, "She can speak French.", "She", "dobj", "What can She speak?"),
    (19, "He should have done the homework.", "He", "dobj", "What should He have done?"),
    (20, "She might be hiding something.", "She", "dobj", "What might She be hiding?"),
    (21, "He could have been reading a novel.", "He", "dobj", "What could He have been reading?"),
    (22, "She is a doctor.", "She", "attr", "What is She?"),
    (23, "John is the teacher.", "John", "nsubj", "Who is the teacher?"),
    (24, "The window was broken by John.", "The window", "agent", "By whom was The window broken?"),
    (25, "The window was broken by John.", "The window", "nsubj", "What was broken by John?"),
    (26, "The car has been repaired.", "The car", "nsubj", "What has been repaired?"),
    (27, "The report should be submitted by Friday.", "The report", "pobj", "When should The report be submitted?"),
    (28, "She has a car.", "She", "dobj", "What does She have?"),
    (29, "She gave up smoking.", "She", "dobj", "What did She give up?"),
    (30, "He signed up for the course.", "He", "pobj", "What did He sign up for?"),
    (31, "She realized that he had left.", "She", "ccomp", "What did She realize?"),
    (32, "She wants to pursue counseling.", "She", "dobj", "What does She want to pursue?"),
    (33, "She enjoys playing tennis.", "She", "xcomp", "What does She enjoy?"),
    (34, "There is a problem.", "There", "attr", "What is There?"),
    (36, "She is not reading the book.", "She", "dobj", "What is She not reading?"),
    (37, "She does not like chocolate.", "She", "dobj", "What does She not like?"),
    (38, "She gave him a book.", "She", "dobj", "What did She give him?"),
    (39, "She became a nurse.", "She", "attr", "What did She become?"),
    (40, "The food tastes delicious.", "The food", "acomp", "How does The food taste?"),
    (41, "She works at the hospital.", "She", "pobj", "Where does She work?"),
    (42, "She left yesterday.", "She", "advmod", "When did She leave?"),
]

passed = 0
failed = 0
for row, statement, speaker, target, expected in tests:
    doc = nlp(statement)
    result = form_question(doc, target, speaker)
    ok = result.strip() == expected.strip()
    if ok:
        passed += 1
        print(f"  Row {row:2d} PASS: {result}")
    else:
        failed += 1
        print(f"  Row {row:2d} FAIL: {result}")
        print(f"         WANT: {expected}")

print(f"\n{passed}/{passed+failed} passed, {failed} failed")
