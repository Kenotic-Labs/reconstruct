"""Test WordNet Domains for schema classification."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacy_wordnet', after='tagger')

# Map 165 WN domains to 11 life domains
DOMAIN_MAP = {
    # career
    'commerce': 'career', 'enterprise': 'career', 'industry': 'career',
    'administration': 'career', 'economy': 'career', 'banking': 'career',
    # health
    'medicine': 'health', 'surgery': 'health', 'pharmacy': 'health',
    'psychiatry': 'health', 'dentistry': 'health', 'veterinary': 'health',
    'health': 'health', 'body_care': 'health', 'anatomy': 'health',
    # family
    'person': 'family', 'sexuality': 'family',
    # housing
    'architecture': 'housing', 'furniture': 'housing',
    'town_planning': 'housing',
    # finance
    'finance': 'finance', 'money': 'finance', 'exchange': 'finance',
    'insurance': 'finance', 'tax': 'finance', 'banking': 'finance',
    'book_keeping': 'finance',
    # education
    'pedagogy': 'education', 'school': 'education', 'university': 'education',
    'grammar': 'education',
    # hobby
    'music': 'hobby', 'sport': 'hobby', 'play': 'hobby', 'dance': 'hobby',
    'photography': 'hobby', 'plastic_arts': 'hobby', 'artisanship': 'hobby',
    'free_time': 'hobby', 'drawing': 'hobby',
    # social
    'sociology': 'social', 'politics': 'social', 'fashion': 'social',
    'tourism': 'social', 'gastronomy': 'social',
    # identity
    'religion': 'identity', 'philosophy': 'identity', 'psychology': 'identity',
    # experience
    'astrology': 'experience', 'astronomy': 'experience', 'meteorology': 'experience',
    # planning
    'transport': 'planning', 'time_period': 'planning',
    # legal
    'law': 'finance',
}

# NOTE: 'buildings' deliberately excluded — too generic (shop, house, piano all get it)
# NOTE: 'publishing' excluded — ambiguous (career vs education)


def classify_domain(text):
    doc = nlp(text)
    tok = doc[-1]  # head noun
    domains = tok._.wordnet.wordnet_domains()

    votes = {}
    for d in domains:
        if d == 'factotum':
            continue
        life = DOMAIN_MAP.get(d)
        if life:
            votes[life] = votes.get(life, 0) + 1

    if votes:
        return max(votes, key=votes.get)
    return 'uncategorized'


tests = [
    ('coffee shop', 'career'),
    ('dog', 'family'),
    ('house', 'housing'),
    ('medication', 'health'),
    ('textbook', 'education'),
    ('piano', 'hobby'),
    ('yoga', 'health'),
    ('salary', 'finance'),
    ('guitar', 'hobby'),
    ('debt', 'finance'),
    ('business', 'career'),
    ('gym', 'hobby'),
    ('surgery', 'health'),
    ('therapy', 'health'),
    ('mortgage', 'finance'),
    ('painting', 'hobby'),
    ('wedding', 'social'),
    ('apartment', 'housing'),
    ('restaurant', 'career'),
    ('friend', 'family'),
    ('brother', 'family'),
    ('doctor', 'health'),
    ('neighbor', 'social'),
    ('promotion', 'career'),
    ('degree', 'education'),
]

correct = 0
for word, expected in tests:
    got = classify_domain(word)
    ok = got == expected
    if ok:
        correct += 1
    mark = 'OK' if ok else 'WRONG'
    print(f'  {mark:5s} {word:15s} expected={expected:12s} got={got}')

print(f'\n{correct}/{len(tests)}')
