"""Test: verb supersense + object WordNet Domain = schema."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacy_wordnet', after='tagger')

# 165 WN domains → 11 life domains
DOMAIN_MAP = {
    'commerce': 'career', 'enterprise': 'career', 'industry': 'career',
    'administration': 'career', 'economy': 'career',
    'medicine': 'health', 'surgery': 'health', 'pharmacy': 'health',
    'psychiatry': 'health', 'dentistry': 'health', 'veterinary': 'health',
    'health': 'health', 'body_care': 'health', 'anatomy': 'health',
    'person': 'family', 'sexuality': 'family',
    'architecture': 'housing', 'furniture': 'housing', 'town_planning': 'housing',
    'buildings': 'housing',
    'finance': 'finance', 'money': 'finance', 'exchange': 'finance',
    'insurance': 'finance', 'tax': 'finance', 'banking': 'finance',
    'book_keeping': 'finance',
    'pedagogy': 'education', 'school': 'education', 'university': 'education',
    'grammar': 'education',
    'music': 'hobby', 'sport': 'hobby', 'play': 'hobby', 'dance': 'hobby',
    'photography': 'hobby', 'plastic_arts': 'hobby', 'artisanship': 'hobby',
    'free_time': 'hobby', 'drawing': 'hobby',
    'sociology': 'social', 'politics': 'social', 'fashion': 'social',
    'tourism': 'social', 'gastronomy': 'social',
    'religion': 'identity', 'philosophy': 'identity', 'psychology': 'identity',
    'transport': 'planning', 'time_period': 'planning',
    'law': 'finance',
}

# 15 verb supersenses → life domain (used when noun domain is ambiguous or generic)
VERB_SUPERSENSE_MAP = {
    'verb.social': 'career',        # work, hire, manage, promote, marry
    'verb.possession': 'finance',   # buy, sell, own, invest
    'verb.creation': 'hobby',       # paint, build, cook, write
    'verb.cognition': 'education',  # learn, study, think, realize
    'verb.emotion': 'emotional',    # love, fear, hate, enjoy
    'verb.motion': 'experience',    # go, move, travel, run
    'verb.communication': 'social', # talk, tell, say, discuss
    'verb.consumption': 'health',   # eat, drink, consume
    'verb.body': 'health',          # breathe, sleep, exercise
    'verb.competition': 'hobby',    # play, compete, race
    'verb.perception': 'experience', # see, hear, watch
    'verb.stative': 'identity',     # be, have, seem, exist
    'verb.contact': 'experience',   # touch, hit, meet
    'verb.change': 'experience',    # grow, develop, change
    'verb.weather': 'experience',   # rain, snow
}


def get_verb_supersense(lemma):
    """Get verb supersense from WordNet."""
    for ss in wn.synsets(lemma, pos=wn.VERB)[:3]:
        return ss.lexname()
    return None


def get_noun_domain(token):
    """Get life domain from WordNet Domains for a spaCy token."""
    domains = token._.wordnet.wordnet_domains()
    votes = {}
    for d in domains:
        if d == 'factotum':
            continue
        life = DOMAIN_MAP.get(d)
        if life:
            votes[life] = votes.get(life, 0) + 1
    if votes:
        return max(votes, key=votes.get)
    return None


def classify_schema(text):
    """verb supersense + object WordNet Domain = schema.

    Noun domain is primary signal.
    Verb supersense is secondary (for disambiguation and fallback).
    """
    doc = nlp(text)
    root = None
    for tok in doc:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if not root:
        return "uncategorized"

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

    # Signal 1: noun domain
    noun_domain = None
    if obj_tok:
        noun_domain = get_noun_domain(obj_tok)

    # Signal 2: verb supersense
    verb_ss = get_verb_supersense(root.lemma_)
    verb_domain = VERB_SUPERSENSE_MAP.get(verb_ss, 'uncategorized') if verb_ss else 'uncategorized'

    # Combine: noun domain wins if available, verb disambiguates ties
    if noun_domain and noun_domain != 'housing':
        # housing is over-triggered by 'buildings' — let verb override
        return noun_domain
    if noun_domain == 'housing' and verb_domain in ('career', 'experience', 'social'):
        # "I run a shop" — buildings=housing but verb says career
        return verb_domain
    if noun_domain:
        return noun_domain
    return verb_domain


# ================================================================
# TEST: Full sentences
# ================================================================
tests = [
    ("I run a coffee shop.", "career"),
    ("I adopted a dog.", "family"),
    ("I bought a house.", "finance"),
    ("I painted a sunset.", "hobby"),
    ("I took a medication.", "health"),
    ("I read a textbook.", "education"),
    ("I met my neighbor.", "family"),
    ("I moved to Portland.", "experience"),
    ("I went to the doctor.", "health"),
    ("I traveled to Japan.", "experience"),
    ("I talked to my brother.", "family"),
    ("I started learning piano.", "hobby"),
    ("I signed up for yoga.", "identity"),
    ("My salary is 120 thousand dollars.", "finance"),
    ("David is getting married in June.", "family"),
    ("I was promoted to senior engineer.", "career"),
    ("I realized that I need more balance.", "education"),
    ("I signed up for a yoga class.", "hobby"),
    ("My manager gave me a performance review.", "career"),
    ("I love my new apartment.", "emotional"),
    ("I have been learning guitar for six months.", "hobby"),
    ("Tom runs a small restaurant downtown.", "career"),
    ("I invested in the stock market.", "finance"),
    ("She works at a law firm.", "career"),
    ("I went hiking at Mount Hood.", "experience"),
]

correct = 0
for text, expected in tests:
    got = classify_schema(text)
    ok = got == expected
    if ok:
        correct += 1
    mark = 'OK' if ok else 'XX'
    print(f'  {mark:2s} {text[:50]:50s} exp={expected:12s} got={got}')

print(f'\n{correct}/{len(tests)}')
