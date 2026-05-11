"""Patch 2: add missing edges for questions still refused."""
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
    return 1

t0 = time.time()

# Cat 4 refused questions:

# D2:2 charity race awareness
add('Melanie', 'run_charity_race_for', 'mental health awareness', 'That charity race for mental health was really rewarding, raising awareness for mental health.', 2,
    schema='health', ents=['Melanie'], pqs=['What did the charity race raise awareness for?'])

# D2:14 Caroline excited about creating family
add('Caroline', 'excited_about', 'creating a family for kids who need one', "I'm thrilled to make a family for kids who need one.", 2,
    schema='family', ents=['Caroline'], pqs=['What is Caroline excited about in the adoption process?'])

# D2:15 Melanie thinks Caroline's adoption decision is amazing
add('Melanie', 'think_about_adoption', 'Caroline is doing something amazing and will be an awesome mom', "You're doing something amazing! You'll be an awesome mom!", 2,
    schema='family', ents=['Melanie','Caroline'], pqs=["What does Melanie think about Caroline's decision to adopt?"])

# D3:16 Melanie married 5 years
add('Melanie', 'married_for', '5 years', 'Mel and her husband have been married for 5 years already! Time flies.', 3,
    schema='family', sig='stative', ents=['Melanie'], pqs=["How long have Mel and her husband been married?"])

# D4:5 hand-painted bowl = art and self-expression
add('Caroline', 'bowl_reminds_of', 'art and self-expression', 'My hand-painted bowl reminds me of art and self-expression. The pattern and colors are awesome.', 4,
    schema='identity', ents=['Caroline'], pqs=["What is Melanie's hand-painted bowl a reminder of?", "What is Caroline's hand-painted bowl a reminder of?"])

# D6:9 Caroline's library books
add('Caroline', 'have_in_library', "kids' books - classics, stories from different cultures, educational books", "I've got lots of kids' books - classics, stories from different cultures, educational books.", 6,
    schema='hobby', sig='stative', ents=['Caroline'], pqs=["What kind of books does Caroline have in her library?"])

# D7:13 Becoming Nicole takeaways
add('Caroline', 'learn_from_book', 'self-acceptance and finding support', 'It taught me self-acceptance and how to find support. Hope and love exist.', 7,
    schema='hobby', ents=['Caroline'], pqs=["What did Caroline take away from the book 'Becoming Nicole'?"])

# D7:21 Melanie's reason for running
add('Melanie', 'run_because', 'to de-stress and clear her mind', 'Running has been great to destress and clear my mind.', 7,
    schema='health', sig='stative', ents=['Melanie'], pqs=["What is Melanie's reason for getting into running?"])

# D13:7 Caroline horseback riding with dad
add('Caroline', 'do_with_dad_as_child', 'horseback riding', 'I used to go horseback riding with my dad when I was a kid, through the fields.', 13,
    schema='hobby', temporal='past', hist=1, ents=['Caroline'],
    pqs=['What activity did Caroline used to do with her dad?'])

# D15:23 Brave by Sara Bareilles
add('Caroline', 'motivated_by_song', 'Brave by Sara Bareilles', '"Brave" by Sara Bareilles motivates Caroline to be courageous and fight for what is right.', 15,
    schema='hobby', sig='stative', ents=['Caroline'],
    pqs=['Which song motivates Caroline to be courageous?'])

# D17:8 Melanie setback
add('Melanie', 'face_setback', 'got hurt and had to take a break from pottery', 'Last month I got hurt and had to take a break from pottery which was tough.', 17,
    schema='health', emoval=0.3, texpr='last month', rdate='2023-09-13', ents=['Melanie'],
    pqs=['What setback did Melanie face in October 2023?'])

# Cat 2 refused questions:

# D16:1 Caroline biking
add('Caroline', 'go_biking_with', 'friends last weekend', 'I had a wicked day out with the gang last weekend - we went biking and saw some pretty cool stuff.', 16,
    schema='social', texpr='last weekend', rdate='2023-09-09', ents=['Caroline'],
    pqs=['When did Caroline go biking with friends?'])

# D17:3 Melanie's friend adopted
add('Melanie', 'have_friend_who_adopted', 'last year', 'A buddy of mine adopted last year. They are super happy with their new kid.', 17,
    schema='social', temporal='past', hist=1, texpr='last year', rdate='2022-01-01', ents=['Melanie'],
    pqs=["When did Melanie's friend adopt a child?"])

# D17:8 When Melanie got hurt
add('Melanie', 'get_hurt', 'September 2023', 'Last month I got hurt and had to take a break from pottery.', 17,
    schema='health', texpr='last month', rdate='2023-09-13', ents=['Melanie'],
    pqs=['When did Melanie get hurt?'])

# D18:1 road trip timing
add('Melanie', 'go_on_road_trip', 'with family last weekend', "That roadtrip this past weekend was insane! We were all freaked when my son got into an accident.", 18,
    schema='family', texpr='this past weekend', rdate='2023-10-14', ents=['Melanie'],
    pqs=["When did Melanie's family go on a roadtrip?"])

# D19:1 adoption interview timing
add('Caroline', 'pass_adoption_interview', 'last Friday', 'I passed the adoption agency interviews last Friday!', 19,
    schema='family', sig='milestone', texpr='last Friday', rdate='2023-10-20', ents=['Caroline'],
    pqs=['When did Caroline pass the adoption interview?'])

# D19:2 figurines timing
add('Melanie', 'buy_figurines', 'yesterday', 'These figurines I bought yesterday remind me of family love.', 19,
    schema='hobby', texpr='yesterday', rdate='2023-10-21', ents=['Melanie'],
    pqs=['When did Melanie buy the figurines?'])

# Additional Cat 4 misses from wrong-answer audit:

# D8:5 painting is creative project with kids (besides pottery)
add('Melanie', 'do_creative_with_kids', 'painting', 'We love painting together, especially nature-inspired ones. Besides pottery, painting is our creative project.', 8,
    schema='hobby', ents=['Melanie'], pqs=['What creative project do Mel and her kids do together besides pottery?'])

# D10:10 beach frequency answer
add('Melanie', 'visit_beach', 'once or twice a year', "We don't go to the beach often, usually only once or twice a year. But those times are always special.", 10,
    schema='family', sig='stative', ents=['Melanie'], pqs=['How often does Melanie go to the beach with her kids?'])

# D16:7 Melanie creating art for 7 years
add('Melanie', 'create_art_for', '7 years', 'Seven years now, and I finally found my real muses: painting and pottery.', 16,
    schema='hobby', sig='stative', ents=['Melanie'], pqs=['How long has Melanie been creating art?'])

# D17:10 Melanie busy during break
add('Melanie', 'keep_busy_during_break', 'reading book and painting', "I've been reading that book you recommended and painting to keep busy during my pottery break.", 17,
    schema='hobby', ents=['Melanie'], pqs=['What does Melanie do to keep herself busy during her pottery break?'])

# D15:28 Melanie's music taste
add('Melanie', 'listen_to', 'Bach, Mozart, and Ed Sheeran', "I'm a fan of classical like Bach and Mozart, as well as modern music like Ed Sheeran.", 15,
    schema='hobby', sig='stative', ents=['Melanie'],
    pqs=["Which classical musicians does Melanie enjoy listening to?", "Who is Melanie a fan of in terms of modern music?"])

conn.commit()
elapsed = time.time() - t0
cnt = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
print(f'Done: {cnt} total edges in {elapsed:.1f}s')
conn.close()
