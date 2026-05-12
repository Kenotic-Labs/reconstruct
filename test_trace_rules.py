"""Test: Extract 5 traces from structured English sentences using formal grammar rules."""
import spacy
nlp = spacy.load("en_core_web_md")


def extract_traces(text, speaker):
    """Extract 5 traces from one sentence using formal English grammar rules.

    Rules from Quirk et al., Cambridge Grammar, formal syntax.
    Each rule maps directly to spaCy dep/POS/morph tags.
    """
    doc = nlp(text)
    root = None
    for tok in doc:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if not root:
        return None

    # ================================================================
    # SENTENCE CLASSIFICATION (Rule 6)
    # 4 types, 4 syntactic markers
    # ================================================================
    has_question_mark = any(tok.text == "?" for tok in doc)
    has_nsubj = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in doc)
    root_is_base = root.tag_ == "VB"
    sent_type = "declarative"
    if has_question_mark:
        sent_type = "interrogative"
    elif root_is_base and not has_nsubj:
        sent_type = "imperative"

    # Only extract traces from declaratives (facts)
    if sent_type != "declarative":
        return None

    # ================================================================
    # TENSE-ASPECT (Rule 8)
    # Tense from first finite verb tag. Aspect from aux chain.
    # ================================================================
    auxes = [ch for ch in root.children if ch.dep_ in ("aux", "auxpass")]

    # Tense: from first finite element
    tense = "present"
    finite = auxes[0] if auxes else root
    if finite.tag_ == "VBD":
        tense = "past"
    elif finite.tag_ in ("VBZ", "VBP"):
        tense = "present"
    elif finite.tag_ == "MD" and finite.lemma_ in ("will", "shall"):
        tense = "future"
    elif finite.tag_ == "MD":
        tense = "present"  # modals like can/should are present

    # Aspect: from aux chain
    has_have_aux = any(a.lemma_ == "have" for a in auxes)
    has_be_aux = any(a.lemma_ == "be" for a in auxes)
    root_is_vbg = root.tag_ == "VBG"
    root_is_vbn = root.tag_ == "VBN"

    if has_have_aux and has_be_aux and root_is_vbg:
        aspect = "perfect_continuous"
    elif has_have_aux and root_is_vbn:
        aspect = "perfect"
    elif has_be_aux and root_is_vbg:
        aspect = "continuous"
    else:
        aspect = "simple"

    # ================================================================
    # VOICE (Rule 9)
    # Passive = auxpass or nsubjpass exists
    # ================================================================
    voice = "passive" if any(tok.dep_ in ("auxpass", "nsubjpass") for tok in doc) else "active"

    # ================================================================
    # MOOD (Rule 7)
    # ================================================================
    mood = "indicative"
    for ch in root.children:
        if ch.dep_ == "aux" and ch.lemma_ in ("would", "could", "should", "might"):
            mood = "conditional"
            break
        if ch.dep_ == "mark" and ch.lemma_ in ("if", "unless"):
            mood = "conditional"
            break

    # ================================================================
    # NEGATION (Rule: dep=neg exists)
    # ================================================================
    negated = any(tok.dep_ == "neg" for tok in doc)

    # ================================================================
    # RELATIONAL TRACE (Rule 4)
    # Subject = nsubj/nsubjpass. Entities = PERSON/PROPN/PRON NER.
    # ================================================================
    subject = speaker
    subj_tok = None
    for ch in root.children:
        if ch.dep_ in ("nsubj", "nsubjpass"):
            subj_tok = ch
            subject = " ".join(t.text for t in sorted(subj_tok.subtree, key=lambda t: t.i))
            break

    entities = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"):
            entities.append(ent.text)
    # Add speaker if not already there
    if speaker and speaker not in entities:
        entities.append(speaker)

    # Relational type from semantic role
    rel_type = "personal"
    for ch in root.children:
        if ch.dep_ == "dobj" and ch.ent_type_ == "PERSON":
            rel_type = "social"
            break
        if ch.dep_ == "dative":
            rel_type = "social"
            break

    # ================================================================
    # EPISODIC TRACE (Rule 1)
    # Predicate = everything that is NOT the subject subtree
    # ================================================================
    subj_indices = set()
    if subj_tok:
        subj_indices = {t.i for t in subj_tok.subtree}
    episodic = " ".join(
        tok.text for tok in doc
        if tok.i not in subj_indices and tok.pos_ != "PUNCT"
    ).strip()

    # ================================================================
    # OBJECT EXTRACTION
    # dobj > attr > acomp > pobj > xcomp-dobj > ccomp
    # ================================================================
    obj = ""
    for ch in root.children:
        if ch.dep_ == "dobj":
            obj = " ".join(t.text for t in sorted(ch.subtree, key=lambda t: t.i))
            break
    if not obj:
        for ch in root.children:
            if ch.dep_ == "attr":
                obj = " ".join(t.text for t in sorted(ch.subtree, key=lambda t: t.i))
                break
    if not obj:
        for ch in root.children:
            if ch.dep_ == "acomp":
                obj = ch.text
                break
    if not obj:
        for ch in root.children:
            if ch.dep_ == "prep":
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        obj = " ".join(t.text for t in sorted(gc.subtree, key=lambda t: t.i))
                        break
                if obj:
                    break
    if not obj:
        for ch in root.children:
            if ch.dep_ == "xcomp":
                for gc in ch.children:
                    if gc.dep_ == "dobj":
                        obj = " ".join(t.text for t in sorted(gc.subtree, key=lambda t: t.i))
                        break
                break
    if not obj:
        for ch in root.children:
            if ch.dep_ == "ccomp":
                tokens = sorted(ch.subtree, key=lambda t: t.i)
                obj = " ".join(t.text for t in tokens if not (t.dep_ == "mark" and t.lemma_ == "that"))
                break

    # ================================================================
    # TEMPORAL TRACE (Rule 3)
    # Direction from tense. Expression from NER DATE/TIME or advmod.
    # ================================================================
    temporal_direction = tense  # past/present/future from tense detection above
    temporal_expression = None

    # NER entities first
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            temporal_expression = ent.text
            break

    # Bare NP temporal (npadvmod)
    if not temporal_expression:
        for ch in root.children:
            if ch.dep_ == "npadvmod":
                temporal_expression = " ".join(t.text for t in sorted(ch.subtree, key=lambda t: t.i))
                break

    # ================================================================
    # EMOTIONAL TRACE (Rule 2)
    # ADJ in acomp/attr after copular verb = emotion
    # ================================================================
    emotional_state = None
    emotional_valence = None
    emotional_target = None

    for ch in root.children:
        if ch.dep_ == "acomp" and ch.pos_ == "ADJ":
            emotional_state = ch.text
            # Target from prep child
            for gc in ch.children:
                if gc.dep_ == "prep":
                    for ggc in gc.children:
                        if ggc.dep_ == "pobj":
                            emotional_target = " ".join(t.text for t in sorted(ggc.subtree, key=lambda t: t.i))
                            break
            break

    # Also check ROOT if it's an emotion verb
    if not emotional_state and root.pos_ == "VERB":
        try:
            from nltk.corpus import wordnet as wn
            synsets = wn.synsets(root.lemma_, pos=wn.VERB)
            for syn in synsets[:2]:
                if syn.lexname() == "verb.emotion":
                    emotional_state = root.lemma_
                    break
        except Exception:
            pass

    # ================================================================
    # SCHEMATIC TRACE (Rule 5)
    # WordNet supersense of verb or object noun
    # ================================================================
    schema = "uncategorized"
    try:
        from nltk.corpus import wordnet as wn
        # Try object noun first
        if obj:
            obj_head = obj.split()[-1]  # last word is usually the head noun
            for syn in wn.synsets(obj_head, pos=wn.NOUN)[:2]:
                lexname = syn.lexname()
                if "person" in lexname or "group" in lexname:
                    schema = "social"
                elif "location" in lexname:
                    schema = "housing"
                elif "food" in lexname:
                    schema = "health"
                elif "artifact" in lexname:
                    schema = "hobby"
                elif "possession" in lexname:
                    schema = "finance"
                elif "feeling" in lexname:
                    schema = "emotional"
                elif "communication" in lexname:
                    schema = "social"
                elif "act" in lexname or "cognition" in lexname:
                    schema = "career"
                elif "time" in lexname:
                    schema = "temporal"
                if schema != "uncategorized":
                    break
        # Fallback: verb supersense
        if schema == "uncategorized":
            for syn in wn.synsets(root.lemma_, pos=wn.VERB)[:2]:
                lexname = syn.lexname()
                if "emotion" in lexname:
                    schema = "emotional"
                elif "cognition" in lexname:
                    schema = "career"
                elif "creation" in lexname:
                    schema = "hobby"
                elif "motion" in lexname:
                    schema = "experience"
                elif "social" in lexname:
                    schema = "social"
                elif "communication" in lexname:
                    schema = "social"
                elif "consumption" in lexname:
                    schema = "health"
                elif "possession" in lexname:
                    schema = "finance"
                if schema != "uncategorized":
                    break
    except Exception:
        pass

    # ================================================================
    # PREDICATE (verb lemma + prep frame)
    # ================================================================
    predicate = root.lemma_
    for ch in root.children:
        if ch.dep_ == "prt":
            predicate = f"{predicate}_{ch.lemma_}"
            break
    for ch in root.children:
        if ch.dep_ == "prep" and ch.pos_ == "ADP":
            predicate = f"{predicate}_{ch.lemma_}"
            break

    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "episodic_fact": episodic,
        "tense": tense,
        "aspect": aspect,
        "temporal_direction": temporal_direction,
        "temporal_expression": temporal_expression,
        "voice": voice,
        "mood": mood,
        "negated": negated,
        "emotional_state": emotional_state,
        "emotional_target": emotional_target,
        "relational_subject": subject,
        "relational_entities": entities,
        "relational_type": rel_type,
        "schematic_category": schema,
        "sent_type": sent_type,
    }


# ================================================================
# TEST: Structured English conversation
# ================================================================
conversation = [
    ("Sarah", "I moved to Portland from Seattle three years ago."),
    ("Sarah", "I work as a software engineer at a startup."),
    ("Sarah", "My salary is about 120 thousand dollars."),
    ("Sarah", "I have been learning piano for six months."),
    ("Sarah", "I am really happy with my new apartment."),
    ("Sarah", "My brother David visited me last weekend."),
    ("Sarah", "We went hiking at Mount Hood on Saturday."),
    ("Sarah", "I adopted a dog named Biscuit two months ago."),
    ("Sarah", "David is getting married in June."),
    ("Sarah", "I was promoted to senior engineer last month."),
    ("Sarah", "I realized that I need more work life balance."),
    ("Sarah", "I signed up for a yoga class yesterday."),
    ("Sarah", "My manager gave me a great performance review."),
    ("Sarah", "I can not imagine life without Biscuit."),
    ("Sarah", "I will attend David's wedding in Seattle."),
    ("Tom", "I run a small coffee shop downtown."),
    ("Tom", "I have owned it for five years."),
    ("Tom", "I am passionate about specialty coffee."),
    ("Tom", "My wife and I bought a house near the coast last year."),
    ("Tom", "I was recently featured in a local magazine."),
]

for speaker, text in conversation:
    traces = extract_traces(text, speaker)
    if traces:
        print(f'[{speaker}] "{text}"')
        print(f'  subject:    {traces["subject"]}')
        print(f'  predicate:  {traces["predicate"]}')
        print(f'  object:     {traces["object"]}')
        print(f'  episodic:   {traces["episodic_fact"][:60]}')
        print(f'  tense:      {traces["tense"]} {traces["aspect"]}')
        print(f'  direction:  {traces["temporal_direction"]}')
        print(f'  temp_expr:  {traces["temporal_expression"]}')
        print(f'  voice:      {traces["voice"]}')
        print(f'  mood:       {traces["mood"]}')
        print(f'  negated:    {traces["negated"]}')
        print(f'  emotion:    {traces["emotional_state"]}')
        print(f'  emot_tgt:   {traces["emotional_target"]}')
        print(f'  entities:   {traces["relational_entities"]}')
        print(f'  rel_type:   {traces["relational_type"]}')
        print(f'  schema:     {traces["schematic_category"]}')
        print()
