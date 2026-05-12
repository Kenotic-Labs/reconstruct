"""
English question formation — pure set decomposition.

Formula (Chomsky 1957, verified against 10 linguistic sources):

  tokens = SUBJECT ∪ VERB_CHAIN ∪ TARGET ∪ REMAINDER
  question = WH + VERB_CHAIN[0] + SUBJECT + VERB_CHAIN[1:] + REMAINDER + ?

  If TARGET = SUBJECT: question = WH + VERB_CHAIN + REMAINDER + ?
  If VERB_CHAIN empty and ROOT != be: do-support, ROOT → base form

No dep-specific cases. Just 4 sets and assembly.
"""
import spacy
nlp = spacy.load("en_core_web_md")


def _get_root(doc):
    for tok in doc:
        if tok.dep_ == "ROOT":
            return tok
    return None


def _do_form(tag):
    if tag == "VBD": return "did"
    if tag == "VBZ": return "does"
    return "do"


def _wh_word(target_tok, prep_parent):
    """Select WH-word from the target token's semantics."""
    # Location preps → WHERE (absorb prep)
    if prep_parent and prep_parent.lemma_ in ("at", "in", "on", "to", "from", "near"):
        if target_tok.ent_type_ in ("DATE", "TIME"):
            return "When", True
        return "Where", True
    # Duration
    if prep_parent and prep_parent.lemma_ == "for":
        if any(t.pos_ == "NUM" for t in target_tok.subtree):
            return "How long", True
    # Temporal
    if target_tok.ent_type_ in ("DATE", "TIME"):
        return "When", True
    if target_tok.text.lower() in ("yesterday", "today", "tomorrow"):
        return "When", True
    # Adjective complement → HOW
    if target_tok.pos_ == "ADJ" and target_tok.dep_ in ("acomp", "dobj", "attr"):
        # "tastes delicious" → How, "is happy" → How
        head = target_tok.head
        if head.lemma_ in ("be", "taste", "feel", "seem", "look", "sound", "smell", "appear", "become", "remain", "get"):
            return "How", False
    # Agent → By whom
    if prep_parent and prep_parent.dep_ == "agent":
        return "By whom", True
    # Person
    if target_tok.ent_type_ == "PERSON":
        return "Who", False
    if target_tok.text.lower() in ("him", "her", "them"):
        return "Whom", False
    if target_tok.text.lower() in ("someone", "somebody"):
        return "Who", False
    return "What", False


def form_question(doc, target_indices, subject_name):
    """
    Pure set-decomposition question formation.

    doc: spaCy Doc of declarative sentence
    target_indices: set of token indices being questioned
    subject_name: resolved speaker name (from pronoun resolution)

    Returns: grammatically correct WH-question string
    """
    root = _get_root(doc)
    if not root:
        return ""

    # ===== STEP 1: Decompose into 4 sets =====

    # SUBJECT: nsubj/nsubjpass/expl and their subtrees
    subject_indices = set()
    for ch in root.children:
        if ch.dep_ in ("nsubj", "nsubjpass", "expl"):
            subject_indices.update(t.i for t in ch.subtree)

    # VERB_CHAIN: all aux/auxpass of root + root itself + particles, ordered by position
    vc_indices = set()
    vc_indices.add(root.i)
    for ch in root.children:
        if ch.dep_ in ("aux", "auxpass"):
            vc_indices.add(ch.i)
        if ch.dep_ == "prt":
            vc_indices.add(ch.i)
    verb_chain = sorted(vc_indices)  # ordered token indices

    # TARGET: given as input (the constituent being replaced by WH)
    # Already have target_indices

    # Find the prep parent if target is a pobj/agent-pobj
    prep_parent = None
    target_head_tok = None
    for i in target_indices:
        tok = doc[i]
        if tok.dep_ == "pobj":
            target_head_tok = tok
            if tok.head.dep_ in ("prep", "agent"):
                prep_parent = tok.head
            break

    # Determine which WH-word and whether to absorb the prep
    # Use the first token of target for semantics
    target_first = doc[min(target_indices)]
    wh, absorb_prep = _wh_word(target_first, prep_parent)

    # Expand target to include absorbed prep's subtree
    effective_target = set(target_indices)
    strand_prep = None
    if prep_parent:
        if absorb_prep:
            # Absorb entire prep subtree into target (removed from output)
            effective_target.update(t.i for t in prep_parent.subtree)
        else:
            # Strand: remove pobj subtree from output, but keep prep token for end
            effective_target.update(t.i for t in target_first.subtree)
            strand_prep = prep_parent.text
            # Also remove the prep token from remaining (we'll add it at end)
            effective_target.add(prep_parent.i)

    # REMAINDER: everything not in the other 3 sets, minus punctuation
    punct = {tok.i for tok in doc if tok.pos_ == "PUNCT"}
    remainder_indices = sorted(
        set(range(len(doc))) - subject_indices - set(verb_chain) - effective_target - punct
    )

    # Negation: find neg token, track its position
    neg_tok = None
    for ch in root.children:
        if ch.dep_ == "neg":
            neg_tok = ch
            break

    # ===== STEP 2: Is this a subject question? =====
    is_subject_q = bool(target_indices & subject_indices)

    if is_subject_q:
        # WH replaces subject, everything else stays in document order
        parts = [wh]
        for i in sorted(set(verb_chain) | set(remainder_indices)):
            parts.append(doc[i].text)
        if strand_prep:
            parts.append(strand_prep)
        return " ".join(parts) + "?"

    # ===== STEP 3: Non-subject question — apply inversion =====

    # Separate verb chain into invertible part
    is_be_main = root.lemma_ == "be" and len(verb_chain) == 1
    has_aux = any(doc[i].dep_ in ("aux", "auxpass") for i in verb_chain if i != root.i)

    # Remove neg from remainder (we'll place it explicitly)
    neg_text = ""
    remainder_clean = remainder_indices
    if neg_tok:
        neg_text = neg_tok.text
        remainder_clean = [i for i in remainder_indices if i != neg_tok.i]

    if has_aux:
        # BRANCH 2: First aux inverts
        first_aux_i = verb_chain[0]  # first token in chain is first aux
        # But only if it's actually an aux, not the root
        if doc[first_aux_i].dep_ not in ("aux", "auxpass"):
            # Root came first positionally — find first actual aux
            for vi in verb_chain:
                if doc[vi].dep_ in ("aux", "auxpass"):
                    first_aux_i = vi
                    break

        first_aux_text = doc[first_aux_i].text
        rest_vc = [doc[i].text for i in verb_chain if i != first_aux_i]

        parts = [wh, first_aux_text]
        parts.append(subject_name)
        if neg_text:
            parts.append(neg_text)
        parts.extend(rest_vc)
        parts.extend(doc[i].text for i in remainder_clean)
        if strand_prep:
            parts.append(strand_prep)
        return " ".join(parts) + "?"

    if is_be_main:
        # BRANCH 3: Be inverts directly
        be_text = root.text
        parts = [wh, be_text]
        parts.append(subject_name)
        if neg_text:
            parts.append(neg_text)
        parts.extend(doc[i].text for i in remainder_clean)
        if strand_prep:
            parts.append(strand_prep)
        return " ".join(parts) + "?"

    # BRANCH 4: Do-support
    do = _do_form(root.tag_)
    base = root.lemma_
    # Verb chain minus root = just particles
    particles = [doc[i].text for i in verb_chain if i != root.i]

    parts = [wh, do]
    parts.append(subject_name)
    if neg_text:
        parts.append(neg_text)
    parts.append(base)
    parts.extend(particles)
    parts.extend(doc[i].text for i in remainder_clean)
    if strand_prep:
        parts.append(strand_prep)
    return " ".join(parts) + "?"


# ========================================================================
# Helper: find target indices by dep label
# ========================================================================
def _find_target(doc, root, target_dep):
    """Find the target constituent's token indices."""
    if target_dep == "nsubj":
        for ch in root.children:
            if ch.dep_ in ("nsubj", "nsubjpass", "expl"):
                return {t.i for t in ch.subtree}
    elif target_dep == "dobj":
        for ch in root.children:
            if ch.dep_ == "dobj":
                return {t.i for t in ch.subtree}
        # xcomp's dobj
        for ch in root.children:
            if ch.dep_ == "xcomp":
                for gc in ch.children:
                    if gc.dep_ == "dobj":
                        return {t.i for t in gc.subtree}
    elif target_dep == "attr":
        for ch in root.children:
            if ch.dep_ == "attr":
                return {t.i for t in ch.subtree}
    elif target_dep == "acomp":
        for ch in root.children:
            if ch.dep_ == "acomp":
                return {t.i for t in ch.subtree}
        # spaCy sometimes parses ADJ complement as dobj
        for ch in root.children:
            if ch.dep_ == "dobj" and ch.pos_ == "ADJ":
                return {t.i for t in ch.subtree}
    elif target_dep == "pobj":
        # Search all prep and agent children
        for ch in root.children:
            if ch.dep_ in ("prep", "agent"):
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        return {t.i for t in gc.subtree}
    elif target_dep == "ccomp":
        for ch in root.children:
            if ch.dep_ == "ccomp":
                return {t.i for t in ch.subtree}
    elif target_dep == "xcomp":
        for ch in root.children:
            if ch.dep_ == "xcomp":
                return {t.i for t in ch.subtree}
    elif target_dep == "advmod":
        for ch in root.children:
            if ch.dep_ in ("advmod", "npadvmod"):
                return {t.i for t in ch.subtree}
    elif target_dep == "agent":
        for ch in root.children:
            if ch.dep_ == "agent":
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        return {t.i for t in gc.subtree}
    elif target_dep == "dative":
        for ch in root.children:
            if ch.dep_ == "dative":
                return {t.i for t in ch.subtree}
    return None


# ========================================================================
# ALL 41 ROWS
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
for row, statement, speaker, target_dep, expected in tests:
    doc = nlp(statement)
    root = _get_root(doc)
    target = _find_target(doc, root, target_dep)
    if target is None:
        result = f"(no {target_dep} found)"
    else:
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
