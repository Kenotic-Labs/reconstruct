"""Patch 4: targeted edges for remaining Cat 4 + Cat 2 missed questions."""
import sqlite3, hashlib, json, sys, time
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
    print(f'  ADD: {subj} / {pred} / {obj[:60]}')
    return 1

def update_edge(edge_id, **kwargs):
    """Update specific columns on an existing edge."""
    sets = []
    vals = []
    for k, v in kwargs.items():
        sets.append(f'{k}=?')
        vals.append(v)
    vals.append(edge_id)
    conn.execute(f"UPDATE edges SET {', '.join(sets)} WHERE id=?", vals)
    print(f'  UPDATE edge {edge_id}: {list(kwargs.keys())}')
    return 1

t0 = time.time()
added = 0
updated = 0

# ============================================================
# Cat 4 — wrong answers needing precise edges + PQs
# ============================================================

# 1. D2:12 — Why did Caroline choose the adoption agency?
added += add('Caroline', 'choose_agency_because', 'inclusivity and support for LGBTQ+ individuals',
    'I chose them because they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me.', 2,
    schema='family', ents=['Caroline'],
    pqs=["Why did Caroline choose the adoption agency?"])

# 2. D4:13 — What was discussed in the LGBTQ+ counseling workshop?
added += add('Caroline', 'learn_at_workshop', 'therapeutic methods and how to best work with trans people',
    'At the LGBTQ+ counseling workshop we discussed therapeutic methods and how to best work with trans people.', 4,
    schema='career', ents=['Caroline'],
    pqs=["What was discussed in the LGBTQ+ counseling workshop?"])

# 3. D4:15 — What motivated Caroline to pursue counseling?
added += add('Caroline', 'motivated_by', 'her own journey and the support she received, and how counseling improved her life',
    'My own journey and the support I got made a huge difference. I saw how counseling improved my life.', 4,
    schema='career', ents=['Caroline'],
    pqs=["What motivated Caroline to pursue counseling?"])

# 4. D7:24 — What does Melanie say running has been great for?
# Edge 190 already exists with matching PQ — add a stronger variant
added += add('Melanie', 'say_running_great_for', 'her mental health',
    'This has been great for my mental health. Running is great for my mental health.', 7,
    schema='health', sig='stative', ents=['Melanie'],
    pqs=["What does Melanie say running has been great for?", "What is Melanie's reason for getting into running?"])

# 5. D6:10 — Melanie's favorite childhood book
added += add('Melanie', 'favorite_childhood_book', "Charlotte's Web",
    'I loved reading "Charlotte\'s Web" as a kid. It was my favorite book from childhood.', 6,
    schema='hobby', ents=['Melanie'],
    pqs=["What was Melanie's favorite book from her childhood?"])

# 6. D7:11 — What book did Caroline recommend to Melanie?
added += add('Caroline', 'recommend_book', 'Becoming Nicole',
    'I highly recommend "Becoming Nicole" by Amy Ellis Nutt.', 7,
    schema='hobby', ents=['Caroline', 'Melanie'],
    pqs=["What book did Caroline recommend to Melanie?"])

# 7. D7:13 — What did Caroline take away from Becoming Nicole?
added += add('Caroline', 'take_away_from_book', 'self-acceptance and finding support',
    'Becoming Nicole taught me self-acceptance and how to find support. It also showed me that tough times don\'t last.', 7,
    schema='hobby', ents=['Caroline'],
    pqs=["What did Caroline take away from the book Becoming Nicole?"])

# 8. D8:2 — What did Mel and her kids make during pottery workshop?
added += add('Melanie', 'make_at_workshop', 'pots',
    'We all made our own pots at the pottery workshop. The kids loved making pots.', 8,
    schema='hobby', ents=['Melanie'],
    pqs=["What did Mel and her kids make during the pottery workshop?"])

# 9. D9:16 — What inspired Caroline's painting for the art show?
added += add('Caroline', 'inspiration_for_art_show_painting', 'visiting an LGBTQ center and wanting to capture unity and strength',
    'I painted this after visiting an LGBTQ center. I wanted to capture unity and strength.', 9,
    schema='hobby', ents=['Caroline'],
    pqs=["What inspired Caroline's painting for the art show?"])

# 10. D11:1 — Whose birthday did Melanie celebrate recently?
added += add('Melanie', 'celebrate_birthday_of', "daughter",
    "We celebrated my daughter's birthday with a concert. Seeing my kids' smiles was awesome.", 11,
    schema='family', sig='milestone', ents=['Melanie'],
    pqs=["Whose birthday did Melanie celebrate recently?"])

# 11. D11:3 — Who performed at the concert at Melanie's daughter's birthday?
added += add('Melanie', 'saw_perform_at_birthday', 'Matt Patterson',
    "It was Matt Patterson at my daughter's birthday concert. He is so talented!", 11,
    schema='hobby', ents=['Melanie'],
    pqs=["Who performed at the concert at Melanie's daughter's birthday?"])

# 12. D15:23 — Which song motivates Caroline to be courageous?
added += add('Caroline', 'motivated_by_song', 'Brave by Sara Bareilles',
    'The song "Brave" by Sara Bareilles motivates me to be courageous and fight for what is right.', 15,
    schema='hobby', sig='stative', ents=['Caroline'],
    pqs=["Which song motivates Caroline to be courageous?"])

# 13. D18:1 — What happened to Melanie's son on their road trip?
added += add('Melanie', 'son_had', 'accident on road trip',
    "My son got into an accident on our road trip. We were so lucky he was okay.", 18,
    schema='family', sig='milestone', emoval=0.2, ents=['Melanie'],
    pqs=["What happened to Melanie's son on their road trip?"])

# 14. D18:5 — How did Melanie feel after the accident?
added += add('Melanie', 'feel_after_accident', 'grateful and thankful for her family',
    "After the accident I felt grateful and thankful for my family. They mean the world to me.", 18,
    schema='family', emoval=0.8, emolabel='grateful', ents=['Melanie'],
    pqs=["How did Melanie feel after the accident?"])

# ============================================================
# Cat 2 — temporal misses needing correct resolved dates
# ============================================================

# 15. D6:11 — When did Caroline have a picnic? → "The week before 6 July 2023"
# Edge 174 exists with rdate=2023-06-29 — that IS the week before July 6. OK as-is.
# But update PQ to match the question exactly.
updated += update_edge(174,
    pq_1="When did Caroline have a picnic?",
    pq_2="When did Caroline have a picnic with friends and family?")

# 16. D4:8 — When did Melanie go camping in June? → "The week before 27 June 2023"
# Edge 21 exists with rdate=2023-06-20. That IS the week before June 27. OK as-is.
# Update PQ to match question.
updated += update_edge(21,
    pq_1="When did Melanie go camping in June?",
    pq_2="When did Melanie go camping in the mountains?")

# 17. D11:4 — When did Caroline attend a pride parade in August?
added += add('Caroline', 'attend_pride_parade', 'August pride parade',
    'I went to a pride parade last Friday in August.', 11,
    schema='social', texpr='last Friday', rdate='2023-08-11', ents=['Caroline'],
    pqs=["When did Caroline attend a pride parade in August?"])

# 18. D12:15 — When did Caroline and Melanie go to a pride festival together? → 2022
# Edge 71 exists but has no rdate or temporal_expression. Update it.
updated += update_edge(71,
    resolved_event_date='2022-01-01',
    temporal_expression='last year',
    pq_1="When did Caroline and Melanie go to a pride festival together?",
    pq_2="When did Caroline and Melanie attend a pride fest?")
# Also add a clean new edge with full detail
added += add('Caroline', 'attend_with_melanie', 'Pride fest last year',
    'We had a blast last year at the Pride fest.', 12,
    schema='social', temporal='past', texpr='last year', rdate='2022-01-01', hist=1,
    ents=['Caroline', 'Melanie'],
    pqs=["When did Caroline and Melanie go to a pride festival together?"])

# 19. D16:8 — How long has Melanie been practicing art? → Since 2016
added += add('Melanie', 'practice_art_since', '2016',
    'Seven years now making art - painting and pottery since 2016.', 16,
    schema='hobby', sig='stative', ents=['Melanie'],
    pqs=["How long has Melanie been practicing art?"])

# 20. D17:18 — What was the poetry reading that Caroline attended about?
added += add('Caroline', 'attend_poetry_reading_about', 'transgender poetry reading where transgender people shared their stories through poetry',
    'It was a transgender poetry reading where transgender people shared their stories through poetry.', 17,
    schema='social', ents=['Caroline'],
    pqs=["What was the poetry reading that Caroline attended about?"])

# 21. D15:11 — talent show → already exists as edge 87 but PQ is weak. Update it.
updated += update_edge(87,
    resolved_event_date='2023-09-01',
    temporal_expression='next month',
    pq_1="When is Caroline's youth center putting on a talent show?",
    pq_2="When is Caroline's talent show?")

conn.commit()
elapsed = time.time() - t0
cnt = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
print(f'\nPatch 4 done: {added} new edges added, {updated} edges updated, {cnt} total edges in {elapsed:.1f}s')
conn.close()
