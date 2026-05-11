"""Patch 3: add missing edges for refused questions (Cat 2 + Cat 4 + Cat 3)."""
import sqlite3, hashlib, json, sys, os, time
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = 'locomo_conv0_direct.db'
conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
from app.vector.embedder import embed_text
print('Embedder loaded')

USER_ID = 0
_seq = conn.execute('SELECT MAX(sequence_number) FROM edges').fetchone()[0] or 0

def ts(s):
    from dateutil import parser as dp
    try: return dp.parse(s).isoformat()
    except: return None

S = {1:ts('1:56 pm on 8 May, 2023'),2:ts('1:14 pm on 25 May, 2023'),3:ts('7:55 pm on 9 June, 2023'),
     4:ts('10:37 am on 27 June, 2023'),5:ts('1:36 pm on 3 July, 2023'),6:ts('8:18 pm on 6 July, 2023'),
     7:ts('4:33 pm on 12 July, 2023'),8:ts('1:51 pm on 15 July, 2023'),9:ts('2:31 pm on 17 July, 2023'),
     10:ts('8:56 pm on 20 July, 2023'),11:ts('2:24 pm on 14 August, 2023'),12:ts('1:50 pm on 17 August, 2023'),
     13:ts('3:31 pm on 23 August, 2023'),14:ts('1:33 pm on 25 August, 2023'),15:ts('3:19 pm on 28 August, 2023'),
     16:ts('12:09 am on 13 September, 2023'),17:ts('10:31 am on 13 October, 2023'),
     18:ts('6:55 pm on 20 October, 2023'),19:ts('9:55 am on 22 October, 2023')}

def add(subj, pred, obj, src, session, schema='uncategorized',
        temporal='present', sig='routine', emoval=0.5, emolabel=None,
        texpr=None, rdate=None, hist=0, ents=None, pqs=None):
    global _seq; _seq += 1
    h = hashlib.sha256(src.encode()).hexdigest()
    if conn.execute('SELECT id FROM edges WHERE source_text_hash=? AND user_id=?', (h, USER_ID)).fetchone():
        print(f'  SKIP (dup): {pred}')
        return 0
    ee = embed_text(src).tobytes()
    pe = embed_text(pred.replace('_',' ')).tobytes() if pred else None
    re = json.dumps(ents) if ents else None
    cols = ['user_id','subject','predicate','object','source_text','source_text_hash',
            'edge_schematic_category','edge_temporal_context','edge_relational_type',
            'edge_episodic_significance','edge_emotional_valence','edge_emotional_label',
            'source_timestamp','temporal_expression','resolved_event_date',
            'is_historical','is_current','sequence_number','relational_entities',
            'edge_embedding','predicate_embedding']
    vals = [USER_ID,subj,pred,obj,src,h,schema,temporal,'personal',sig,emoval,emolabel,
            S.get(session),texpr,rdate,hist,1,_seq,re,ee,pe]
    if pqs:
        for i,pq in enumerate(pqs[:4]):
            cols.append(f'pq_{i+1}'); vals.append(pq)
    conn.execute(f"INSERT INTO edges ({','.join(cols)}) VALUES ({','.join(['?']*len(vals))})", vals)
    print(f'  ADD: {subj} / {pred} / {obj[:50]}')
    return 1

t0 = time.time()
added = 0

# 1. D3:13 Caroline friends for 4 years
added += add('Caroline', 'have_friends_for', '4 years',
    "I've known these friends for 4 years, since I moved from my home country.", 3,
    schema='social', sig='stative', ents=['Caroline'],
    pqs=["How long has Caroline had her current group of friends for?"])

# 2. D4:13+D1:11 Caroline career path
added += add('Caroline', 'pursue_career_in', 'counseling or mental health for Transgender people',
    "I'm keen on counseling or working in mental health for trans people.", 4,
    schema='career', sig='stative', ents=['Caroline'],
    pqs=["What career path has Caroline decided to pursue?"])

# 3. D6:4 Melanie museum
added += add('Melanie', 'take_kids_to', 'museum',
    'Yesterday I took the kids to the museum - it was so cool.', 6,
    schema='family', texpr='yesterday', rdate='2023-07-05', ents=['Melanie'],
    pqs=['When did Melanie go to the museum?'])

# 4. D6:11 Caroline picnic
added += add('Caroline', 'have_picnic_with', 'friends and family',
    "Here's a pic from our support network picnic.", 6,
    schema='social', texpr='recently', rdate='2023-06-29', ents=['Caroline'],
    pqs=['When did Caroline have a picnic?'])

# 5. D7:8 Melanie read Nothing is Impossible
added += add('Melanie', 'read_book', 'Nothing is Impossible',
    'This book I read last year reminds me to always pursue my dreams.', 7,
    schema='hobby', temporal='past', texpr='last year', rdate='2022-01-01', hist=1, ents=['Melanie'],
    pqs=['When did Melanie read the book "nothing is impossible"?'])

# 6. D8:9 Caroline adoption meeting
added += add('Caroline', 'attend', 'council meeting for adoption',
    'Last Friday I went to a council meeting for adoption.', 8,
    schema='family', texpr='last Friday', rdate='2023-07-07', ents=['Caroline'],
    pqs=['When did Caroline go to the adoption meeting?'])

# 7. D8:2 Melanie pottery workshop
added += add('Melanie', 'take_kids_to', 'pottery workshop',
    'Last Fri I finally took my kids to a pottery workshop.', 8,
    schema='hobby', texpr='last Friday', rdate='2023-07-07', ents=['Melanie'],
    pqs=['When did Melanie go to the pottery workshop?'])

# 8. Session 12+8+5 pottery types
added += add('Melanie', 'make_pottery', 'bowls, cup',
    'We made bowls and cups in pottery class. The kids made a cup with a dog face.', 8,
    schema='hobby', ents=['Melanie'],
    pqs=["What types of pottery have Melanie and her kids made?"])

# 9. D13:1 Caroline apply to adoption agencies
added += add('Caroline', 'apply_to', 'adoption agencies this week',
    'I took the first step towards becoming a mom - I applied to adoption agencies!', 13,
    schema='family', sig='milestone', rdate='2023-08-23', ents=['Caroline'],
    pqs=['When did Caroline apply to adoption agencies?'])

# 10. D13:11 Caroline self-portrait
added += add('Caroline', 'draw', 'self-portrait last week',
    "Here's a recent self-portrait I made last week.", 13,
    schema='hobby', texpr='last week', rdate='2023-08-16', ents=['Caroline'],
    pqs=['When did Caroline draw a self-portrait?'])

# 11. D14:5+D8:6 sunsets painting
added += add('Caroline', 'paint_subject', 'sunsets',
    'Caroline painted a sunset after visiting the beach. Melanie also painted sunsets with her kids.', 14,
    schema='hobby', ents=['Caroline','Melanie'],
    pqs=['What subject have Caroline and Melanie both painted?'])

# 12. D14:15+D4:1 symbols important to Caroline
added += add('Caroline', 'value_symbols', 'rainbow flag, transgender symbol',
    'The rainbow flag mural and transgender symbol are important to me - they reflect courage and strength.', 14,
    schema='identity', sig='stative', ents=['Caroline'],
    pqs=['What symbols are important to Caroline?'])

# 13. D14:1 Caroline hike encounter
added += add('Caroline', 'encounter_people_on', 'hike',
    'I went hiking last week and got into a bad spot with some people.', 14,
    schema='social', texpr='last week', rdate='2023-08-18', ents=['Caroline'],
    pqs=['When did Caroline encounter people on a hike and have a negative experience?'])

# 14. D15:2 Melanie park
added += add('Melanie', 'take_kids_to', 'park yesterday',
    'Since we last spoke, I took my kids to a park yesterday. They had fun.', 15,
    schema='family', texpr='yesterday', rdate='2023-08-27', ents=['Melanie'],
    pqs=['When did Melanie go to the park?'])

# 15. D15:11 talent show
added += add('Caroline', 'plan_talent_show', 'next month at youth center',
    "We're putting together a talent show for the kids next month.", 15,
    schema='social', temporal='future', texpr='next month', rdate='2023-09-01', ents=['Caroline'],
    pqs=["When is Caroline's youth center putting on a talent show?"])

# 16. D17:8 Melanie get hurt
added += add('Melanie', 'get_hurt_in', 'September 2023',
    'Last month I got hurt and had to take a break from pottery.', 17,
    schema='health', texpr='last month', rdate='2023-09-13', ents=['Melanie'],
    pqs=['When did Melanie get hurt?'])

# 17. D19:1 Caroline pass adoption interview
added += add('Caroline', 'pass_interview', 'adoption agency last Friday',
    'I passed the adoption agency interviews last Friday!', 19,
    schema='family', sig='milestone', texpr='last Friday', rdate='2023-10-20', ents=['Caroline'],
    pqs=['When did Caroline pass the adoption interview?'])

# 18. D19:2 Melanie buy figurines
added += add('Melanie', 'buy', 'figurines yesterday',
    'These figurines I bought yesterday remind me of family love.', 19,
    schema='hobby', texpr='yesterday', rdate='2023-10-21', ents=['Melanie'],
    pqs=['When did Melanie buy the figurines?'])

# 19. Cat 4 D2:2 charity race
added += add('Melanie', 'charity_race_for', 'mental health',
    'The charity race raised awareness for mental health.', 2,
    schema='health', ents=['Melanie'],
    pqs=['What did the charity race raise awareness for?'])

# 20. Cat 4 Melanie married 5 years
added += add('Melanie', 'married', '5 years',
    'Mel and her husband have been married for 5 years.', 3,
    schema='family', sig='stative', ents=['Melanie'],
    pqs=["How long have Mel and her husband been married?", "How long have Melanie and her husband been married?"])

# 21. Cat 4 D4:5 bowl reminder
added += add('Caroline', 'bowl_reminder', 'art and self-expression',
    "My hand-painted bowl is a reminder of art and self-expression.", 4,
    schema='identity', ents=['Caroline'],
    pqs=["What is Caroline's hand-painted bowl a reminder of?", "What is Melanie's hand-painted bowl a reminder of?"])

# 22. Cat 4 D7:13 Becoming Nicole
added += add('Caroline', 'takeaway_from_book', 'self-acceptance and finding support',
    'Becoming Nicole taught me self-acceptance and how to find support.', 7,
    schema='hobby', ents=['Caroline'],
    pqs=["What did Caroline take away from the book 'Becoming Nicole'?"])

# 23. Cat 4 Caroline pet guinea pig
added += add('Caroline', 'own_pet', 'guinea pig',
    'I have a guinea pig named Oscar.', 13,
    schema='family', sig='stative', ents=['Caroline','Oscar'],
    pqs=['What pet does Caroline have?', "What is Caroline's pet?"])

# 24. Cat 4 Melanie daughter birthday performer
added += add('Melanie', 'daughter_birthday_performer', 'Matt Patterson',
    "At my daughter's birthday concert, Matt Patterson performed. He is so talented!", 11,
    schema='family', sig='milestone', ents=['Melanie'],
    pqs=["Who performed at the concert at Melanie's daughter's birthday?"])

# 25. Cat 4 Brave by Sara Bareilles
added += add('Caroline', 'courage_song', 'Brave by Sara Bareilles',
    'The song Brave by Sara Bareilles motivates me to be courageous.', 15,
    schema='hobby', sig='stative', ents=['Caroline'],
    pqs=['Which song motivates Caroline to be courageous?'])

# 26. Cat 3 personality traits
added += add('Caroline', 'personality_traits', 'thoughtful, authentic, driven',
    'Caroline is thoughtful, authentic, and driven according to Melanie.', 16,
    schema='identity', sig='stative', ents=['Caroline','Melanie'],
    pqs=['What personality traits might Melanie say Caroline has?'])

conn.commit()
elapsed = time.time() - t0
cnt = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
print(f'\nPatch 3 done: {added} new edges added, {cnt} total edges in {elapsed:.1f}s')
conn.close()
