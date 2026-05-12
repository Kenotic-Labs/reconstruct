"""Schema classification: NER > WordNet Domain > verb supersense. Priority chain."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacy_wordnet', after='tagger')

# WN Domain → life domain (only SPECIFIC domains, not generic ones)
DOMAIN_MAP = {
    'commerce': 'career', 'enterprise': 'career', 'industry': 'career',
    'administration': 'career',
    'medicine': 'health', 'surgery': 'health', 'pharmacy': 'health',
    'psychiatry': 'health', 'dentistry': 'health', 'health': 'health',
    'anatomy': 'health', 'body_care': 'health',
    'architecture': 'housing', 'town_planning': 'housing',
    'finance': 'finance', 'money': 'finance', 'exchange': 'finance',
    'insurance': 'finance', 'tax': 'finance',
    'pedagogy': 'education', 'school': 'education', 'university': 'education',
    'music': 'hobby', 'sport': 'hobby', 'dance': 'hobby',
    'photography': 'hobby', 'plastic_arts': 'hobby', 'artisanship': 'hobby',
    'drawing': 'hobby',
    'sociology': 'social', 'politics': 'social',
    'religion': 'identity', 'philosophy': 'identity', 'psychology': 'identity',
    'law': 'legal',
}
# Excluded: 'buildings' (too generic), 'person' (use NER instead),
# 'economy/banking/book_keeping' (ambiguous career vs finance — let verb decide),
# 'free_time/gastronomy/tourism' (ambiguous), 'time_period' (temporal not schema)

# Verb supersense → life domain
VERB_MAP = {
    'verb.social': 'social',
    'verb.possession': 'finance',
    'verb.creation': 'hobby',
    'verb.cognition': 'education',
    'verb.emotion': 'emotional',
    'verb.motion': 'experience',
    'verb.communication': 'social',
    'verb.consumption': 'health',
    'verb.body': 'health',
    'verb.competition': 'hobby',
    'verb.perception': 'experience',
    'verb.stative': 'identity',
    'verb.contact': 'experience',
    'verb.change': 'experience',
}


def _is_kinship(token):
    """Check if token is a kinship noun via WordNet hypernym."""
    for ss in wn.synsets(token.lemma_, pos=wn.NOUN)[:2]:
        for path in ss.hypernym_paths():
            for ancestor in path:
                if ancestor.name() in ('relative.n.01', 'parent.n.01', 'sibling.n.01',
                                        'spouse.n.01', 'child.n.02', 'family.n.01'):
                    return True
    return False


def classify_schema(text):
    doc = nlp(text)
    root = None
    for tok in doc:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if not root:
        return "uncategorized", "no root"

    # Find object token
    obj_tok = None
    for ch in root.children:
        if ch.dep_ == "dobj":
            obj_tok = ch
            break
    if not obj_tok:
        for ch in root.children:
            if ch.dep_ == "attr":
                obj_tok = ch
                break
    if not obj_tok:
        for ch in root.children:
            if ch.dep_ in ("prep", "agent"):
                for gc in ch.children:
                    if gc.dep_ == "pobj":
                        obj_tok = gc
                        break
                if obj_tok:
                    break
    if not obj_tok:
        for ch in root.children:
            if ch.dep_ == "xcomp":
                for gc in ch.children:
                    if gc.dep_ == "dobj":
                        obj_tok = gc
                        break
                break
    if not obj_tok:
        for ch in root.children:
            if ch.dep_ == "acomp":
                obj_tok = ch
                break

    # Get verb supersense
    verb_ss = None
    for ss in wn.synsets(root.lemma_, pos=wn.VERB)[:3]:
        verb_ss = ss.lexname()
        break

    # ================================================================
    # PRIORITY CHAIN: strongest signal wins
    # ================================================================

    # Priority 1: NER on object (most reliable)
    if obj_tok:
        ner = obj_tok.ent_type_
        if ner == "MONEY":
            return "finance", "NER:MONEY"
        if ner == "PERSON":
            if _is_kinship(obj_tok):
                return "family", "NER:PERSON+kinship"
            return "social", "NER:PERSON"
        if ner in ("GPE", "LOC", "FAC"):
            # Place: housing if moving there, experience if visiting
            if root.lemma_ in ("move", "relocate", "live", "settle", "rent"):
                return "housing", "NER:GPE+move"
            return "experience", "NER:GPE"
        if ner == "ORG":
            return "career", "NER:ORG"

    # Priority 2: Emotional (acomp = adjective complement)
    if obj_tok and obj_tok.pos_ == "ADJ" and obj_tok.dep_ in ("acomp", "attr"):
        return "emotional", "acomp:ADJ"

    # Priority 3: WordNet Domain on object (specific domains only)
    if obj_tok:
        domains = obj_tok._.wordnet.wordnet_domains()
        for d in domains:
            if d in DOMAIN_MAP:
                return DOMAIN_MAP[d], f"WND:{d}"

    # Priority 4: Kinship noun (family even without PERSON NER)
    if obj_tok and obj_tok.pos_ == "NOUN" and _is_kinship(obj_tok):
        return "family", "kinship"

    # Priority 5: Verb supersense (fallback)
    if verb_ss and verb_ss in VERB_MAP:
        return VERB_MAP[verb_ss], f"verb:{verb_ss}"

    return "uncategorized", "none"


# ================================================================
# TEST
# ================================================================
tests = [
    ("I run a coffee shop.", "career"),
    ("I adopted a dog.", "family"),
    ("I bought a house.", "housing"),
    ("I painted a sunset.", "hobby"),
    ("I took medication.", "health"),
    ("I read a textbook.", "education"),
    ("I met my neighbor.", "social"),
    ("I moved to Portland.", "housing"),
    ("I went to the doctor.", "health"),
    ("I traveled to Japan.", "experience"),
    ("I talked to my brother.", "family"),
    ("I started learning piano.", "hobby"),
    ("I signed up for yoga.", "identity"),
    ("My salary is 120 thousand dollars.", "finance"),
    ("David is getting married in June.", "social"),
    ("I was promoted to senior engineer.", "career"),
    ("I realized that I need more balance.", "education"),
    ("I signed up for a yoga class.", "hobby"),
    ("My manager gave me a performance review.", "career"),
    ("I am really happy with my apartment.", "emotional"),
    ("I have been learning guitar.", "hobby"),
    ("Tom runs a small restaurant.", "career"),
    ("I invested in the stock market.", "finance"),
    ("She works at a law firm.", "career"),
    ("I went hiking at Mount Hood.", "experience"),
]

correct = 0
for text, expected in tests:
    got, signal = classify_schema(text)
    ok = got == expected
    if ok:
        correct += 1
    mark = 'OK' if ok else 'XX'
    print(f'  {mark} {text[:50]:50s} exp={expected:12s} got={got:12s} [{signal}]')

print(f'\n{correct}/{len(tests)}')
