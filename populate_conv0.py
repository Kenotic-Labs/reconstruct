"""Populate conv 0 (Caroline & Melanie) directly into edges table.
No engines. Just facts extracted from reading the conversation."""

import sqlite3, hashlib, json, sys, os, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = 'locomo_conv0_direct.db'
for f in [DB_PATH, DB_PATH + '-shm', DB_PATH + '-wal']:
    if os.path.exists(f):
        os.remove(f)

from app.db.models import MIGRATIONS, run_schema_upgrades
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
conn.executescript(MIGRATIONS)
run_schema_upgrades(conn)
conn.commit()

from app.vector.embedder import embed_text
print('Embedder loaded')

USER_ID = 0
_seq = 0

def ts(s):
    from dateutil import parser as dp
    try: return dp.parse(s).isoformat()
    except: return None

S = {
    1: ts('1:56 pm on 8 May, 2023'),
    2: ts('1:14 pm on 25 May, 2023'),
    3: ts('7:55 pm on 9 June, 2023'),
    4: ts('10:37 am on 27 June, 2023'),
    5: ts('1:36 pm on 3 July, 2023'),
    6: ts('8:18 pm on 6 July, 2023'),
    7: ts('4:33 pm on 12 July, 2023'),
    8: ts('1:51 pm on 15 July, 2023'),
    9: ts('2:31 pm on 17 July, 2023'),
    10: ts('8:56 pm on 20 July, 2023'),
    11: ts('2:24 pm on 14 August, 2023'),
    12: ts('1:50 pm on 17 August, 2023'),
    13: ts('3:31 pm on 23 August, 2023'),
    14: ts('1:33 pm on 25 August, 2023'),
    15: ts('3:19 pm on 28 August, 2023'),
    16: ts('12:09 am on 13 September, 2023'),
    17: ts('10:31 am on 13 October, 2023'),
    18: ts('6:55 pm on 20 October, 2023'),
    19: ts('9:55 am on 22 October, 2023'),
}

def add(subj, pred, obj, src, session, schema='uncategorized',
        temporal='present', rel_type='personal', sig='routine',
        emoval=0.5, emolabel=None, texpr=None, rdate=None,
        hist=0, ents=None, pqs=None):
    global _seq
    _seq += 1
    h = hashlib.sha256(src.encode()).hexdigest()
    ee = embed_text(src).tobytes()
    pe = embed_text(pred.replace('_', ' ')).tobytes() if pred else None
    re = json.dumps(ents) if ents else None

    cols = ['user_id','subject','predicate','object','source_text','source_text_hash',
            'edge_schematic_category','edge_temporal_context','edge_relational_type',
            'edge_episodic_significance','edge_emotional_valence','edge_emotional_label',
            'source_timestamp','temporal_expression','resolved_event_date',
            'is_historical','is_current','sequence_number','relational_entities',
            'edge_embedding','predicate_embedding']
    vals = [USER_ID, subj, pred, obj, src, h,
            schema, temporal, rel_type, sig, emoval, emolabel,
            S.get(session), texpr, rdate, hist, 1, _seq, re, ee, pe]

    if pqs:
        for i, pq in enumerate(pqs[:4]):
            cols.append(f'pq_{i+1}')
            vals.append(pq)

    ph = ','.join(['?']*len(vals))
    cn = ','.join(cols)
    conn.execute(f'INSERT INTO edges ({cn}) VALUES ({ph})', vals)
    eid = conn.execute('SELECT last_insert_rowid()').fetchone()[0]

    # FTS
    try:
        fv = [eid, subj or '', (pred or '').replace('_',' '), obj or '', src]
        for i in range(4):
            fv.append(pqs[i] if pqs and i < len(pqs) else None)
        fv.extend([None, None])
        conn.execute('INSERT INTO edges_fts(rowid,subject,predicate,object,source_text,pq_1,pq_2,pq_3,pq_4,vq_1,vq_2) VALUES (?,?,?,?,?,?,?,?,?,?,?)', fv)
    except:
        pass
    return eid

def add_entity(name, etype='person'):
    existing = conn.execute('SELECT id FROM entities WHERE user_id=? AND name=?', (USER_ID, name)).fetchone()
    if existing:
        conn.execute('UPDATE entities SET mention_count=mention_count+1 WHERE id=?', (existing[0],))
    else:
        emb = embed_text(name).tobytes()
        conn.execute('INSERT INTO entities (user_id,name,entity_type,embedding,mention_count) VALUES (?,?,?,?,1)',
                     (USER_ID, name, etype, emb))

t0 = time.time()

# Add entities first
for name in ['Caroline', 'Melanie', 'Oscar', 'Luna', 'Oliver', 'Bailey']:
    add_entity(name, 'person' if name in ('Caroline','Melanie') else 'thing')
print('Entities added')

# =========================================================================
# SESSION 1: 8 May 2023
# =========================================================================
print('Session 1...')

add('Caroline', 'attend', 'LGBTQ support group', 'I went to a LGBTQ support group yesterday and it was so powerful.', 1,
    schema='social', sig='milestone', emoval=0.85, emolabel='empowered',
    texpr='yesterday', rdate='2023-05-07',
    ents=['Caroline'], pqs=['When did Caroline go to the LGBTQ support group?', 'What support group did Caroline attend?'])

add('Caroline', 'identify_as', 'transgender woman', 'The transgender stories were so inspiring! I was so happy and thankful for all the support.', 1,
    schema='identity', sig='stative',
    ents=['Caroline'], pqs=["What is Caroline's identity?", 'Is Caroline transgender?'])

add('Caroline', 'pursue', 'education and career options', 'Gonna continue my edu and check out career options, which is pretty exciting!', 1,
    schema='career', ents=['Caroline'])

add('Caroline', 'interested_in', 'counseling or mental health', "I'm keen on counseling or working in mental health - I'd love to support those with similar issues.", 1,
    schema='career', sig='stative',
    ents=['Caroline'], pqs=['What career path has Caroline decided to pursue?', 'What does Caroline want to do for work?'])

add('Melanie', 'paint', 'lake sunrise', "Yeah, I painted that lake sunrise last year! It's special to me.", 1,
    schema='hobby', temporal='past', texpr='last year', rdate='2022-01-01', hist=1,
    ents=['Melanie'], pqs=['When did Melanie paint a sunrise?', 'What has Melanie painted?'])

add('Melanie', 'enjoy', 'painting', "Painting's a fun way to express my feelings and get creative.", 1,
    schema='hobby', sig='stative',
    ents=['Melanie'], pqs=['What does Melanie do to relax?', "What are Melanie's hobbies?"])

add('Melanie', 'go_swimming_with', 'kids', "I'm off to go swimming with the kids.", 1,
    schema='family', ents=['Melanie'], pqs=['What activities has Melanie done with her family?'])

# =========================================================================
# SESSION 2: 25 May 2023
# =========================================================================
print('Session 2...')

add('Melanie', 'run', 'charity race for mental health', 'I ran a charity race for mental health last Saturday.', 2,
    schema='health', sig='milestone', texpr='last Saturday', rdate='2023-05-20',
    ents=['Melanie'], pqs=['When did Melanie run a charity race?', 'What did the charity race raise awareness for?'])

add('Melanie', 'realize', 'self-care is important', "I'm starting to realize that self-care is really important.", 2,
    schema='health', ents=['Melanie'], pqs=['What did Melanie realize after the charity race?'])

add('Melanie', 'do_for_self_care', 'running, reading, or playing violin', "I'm carving out some me-time each day - running, reading, or playing my violin", 2,
    schema='health', sig='stative',
    ents=['Melanie'], pqs=['How does Melanie prioritize self-care?'])

add('Melanie', 'plan', 'camping', "We're thinking about going camping next month.", 2,
    schema='family', temporal='future', texpr='next month', rdate='2023-06-01',
    ents=['Melanie'], pqs=['When is Melanie planning on going camping?'])

add('Caroline', 'research', 'adoption agencies', "Researching adoption agencies - it's been a dream to have a family.", 2,
    schema='family', sig='milestone',
    ents=['Caroline'], pqs=['What did Caroline research?', "What are Caroline's plans for the summer?"])

add('Caroline', 'choose_agency_supporting', 'LGBTQ+ individuals', 'I chose them cause they help LGBTQ+ folks with adoption.', 2,
    schema='family', ents=['Caroline'],
    pqs=['What type of individuals does the adoption agency Caroline is considering support?'])

add('Caroline', 'be', 'single', "It'll be tough as a single parent, but I'm up for the challenge!", 2,
    schema='identity', sig='stative',
    ents=['Caroline'], pqs=["What is Caroline's relationship status?"])

# =========================================================================
# SESSION 3: 9 June 2023
# =========================================================================
print('Session 3...')

add('Caroline', 'give_speech_at', 'school', 'I talked about my transgender journey and encouraged students to get involved in the LGBTQ community.', 3,
    schema='social', sig='milestone', texpr='last week', rdate='2023-06-02',
    ents=['Caroline'], pqs=['When did Caroline give a speech at a school?'])

add('Caroline', 'meet_with', 'friends, family and mentors', "My friends, family and mentors are my rocks. Here's a pic from when we met up last week!", 3,
    schema='social', texpr='last week', rdate='2023-06-02',
    ents=['Caroline'], pqs=['When did Caroline meet up with her friends, family, and mentors?'])

add('Caroline', 'move_from', 'Sweden', "I've known these friends for 4 years, since I moved from my home country.", 3,
    schema='identity', temporal='past', texpr='4 years ago', rdate='2019-06-01', hist=1,
    ents=['Caroline'], pqs=['Where did Caroline move from 4 years ago?', "What is Caroline's home country?"])

add('Melanie', 'have', 'husband and kids', "I'm lucky to have my husband and kids; they keep me motivated.", 3,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie'], pqs=['Does Melanie have a family?'])

add('Melanie', 'be_married_for', '5 years', '5 years already! Time flies.', 3,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie'], pqs=['How long has Melanie been married?'])

# =========================================================================
# SESSION 4: 27 June 2023
# =========================================================================
print('Session 4...')

add('Caroline', 'receive_from_grandma', 'necklace from Sweden', 'This necklace is super special to me - a gift from my grandma in my home country, Sweden.', 4,
    schema='identity', ents=['Caroline'])

add('Melanie', 'go_camping_in', 'mountains', 'I just took my fam camping in the mountains last week.', 4,
    schema='family', texpr='last week', rdate='2023-06-20',
    ents=['Melanie'], pqs=['Where has Melanie camped?'])

add('Melanie', 'have_kids_who_love', 'nature', 'The 2 younger kids love nature.', 4,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie'], pqs=["What do Melanie's kids like?"])

add('Caroline', 'attend', 'LGBTQ+ counseling workshop', 'Last Friday, I went to an LGBTQ+ counseling workshop.', 4,
    schema='career', texpr='last Friday', rdate='2023-06-23',
    ents=['Caroline'])

add('Caroline', 'want_to_work_with', 'trans people', "I'm thinking of working with trans people, helping them accept themselves and supporting their mental health.", 4,
    schema='career', sig='stative',
    ents=['Caroline'], pqs=['What kind of counseling does Caroline want to pursue?'])

add('Caroline', 'motivated_by', 'own journey and support received', 'My own journey and the support I got made a huge difference.', 4,
    schema='career', ents=['Caroline'])

# =========================================================================
# SESSION 5: 3 July 2023
# =========================================================================
print('Session 5...')

add('Caroline', 'attend', 'LGBTQ+ pride parade', 'Last week I went to an LGBTQ+ pride parade.', 5,
    schema='social', sig='milestone', texpr='last week', rdate='2023-06-26',
    ents=['Caroline'], pqs=['What LGBTQ+ events has Caroline participated in?'])

add('Melanie', 'sign_up_for', 'pottery class', 'I just signed up for a pottery class yesterday.', 5,
    schema='hobby', texpr='yesterday', rdate='2023-07-02',
    ents=['Melanie'], pqs=['What activities does Melanie partake in?'])

add('Caroline', 'learn', 'piano', "I'm getting creative too, just learning the piano.", 5,
    schema='hobby', ents=['Caroline'])

add('Caroline', 'plan_to_attend', 'transgender conference', "I'm going to a transgender conference this month.", 5,
    schema='social', temporal='future', texpr='this month', rdate='2023-07-01',
    ents=['Caroline'])

# =========================================================================
# SESSION 6: 6 July 2023
# =========================================================================
print('Session 6...')

add('Melanie', 'take_kids_to', 'museum', 'Yesterday I took the kids to the museum.', 6,
    schema='family', texpr='yesterday', rdate='2023-07-05',
    ents=['Melanie'], pqs=['What activities has Melanie done with her family?'])

add('Melanie', 'have_kids_who_love', 'dinosaurs', 'They were stoked for the dinosaur exhibit! They love learning about animals.', 6,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie'], pqs=["What do Melanie's kids like?"])

add('Caroline', 'create', 'library for future kids', "I'm creating a library for when I have kids.", 6,
    schema='family', ents=['Caroline'])

add('Caroline', 'collect', 'classic children\'s books', "I've got lots of kids' books- classics, stories from different cultures, educational books.", 6,
    schema='hobby', sig='stative',
    ents=['Caroline'], pqs=["Would Caroline likely have Dr. Seuss books on her bookshelf?"])

add('Melanie', 'read_as_child', "Charlotte's Web", 'I loved reading "Charlotte\'s Web" as a kid.', 6,
    schema='hobby', temporal='past', hist=1,
    ents=['Melanie'], pqs=['What books has Melanie read?'])

add('Melanie', 'go_camping_at', 'beach', "Here's a pic of my family camping at the beach.", 6,
    schema='family', ents=['Melanie'], pqs=['Where has Melanie camped?'])

# =========================================================================
# SESSION 7: 12 July 2023
# =========================================================================
print('Session 7...')

add('Caroline', 'attend', 'LGBTQ conference', 'I went to an LGBTQ conference two days ago and it was really special.', 7,
    schema='social', sig='milestone', texpr='two days ago', rdate='2023-07-10',
    ents=['Caroline'], pqs=['What LGBTQ+ events has Caroline participated in?'])

add('Caroline', 'interested_in', 'counseling and mental health jobs', "I'm still looking into counseling and mental health jobs.", 7,
    schema='career', sig='stative', ents=['Caroline'])

add('Melanie', 'read', 'Nothing is Impossible', 'This book I read last year reminds me to always pursue my dreams.', 7,
    schema='hobby', temporal='past', texpr='last year', rdate='2022-01-01', hist=1,
    ents=['Melanie'], pqs=['What books has Melanie read?'])

add('Caroline', 'love_book', 'Becoming Nicole', 'I loved "Becoming Nicole" by Amy Ellis Nutt.', 7,
    schema='hobby', ents=['Caroline'],
    pqs=['What book did Melanie read from Caroline\'s suggestion?'])

add('Melanie', 'have_pet', 'dog named Luna', 'Luna and Oliver! They are so sweet and playful.', 7,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie', 'Luna', 'Oliver'], pqs=["What are Melanie's pets' names?"])

add('Melanie', 'have_pet', 'dog named Oliver', "We've got a pup and a kitty.", 7,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie', 'Luna', 'Oliver'])

add('Melanie', 'buy', 'running shoes', 'Just got some new shoes, too!', 7,
    schema='hobby', ents=['Melanie'], pqs=['What items has Melanie bought?'])

add('Melanie', 'run_to', 'destress', "I've been running farther to de-stress, which has been great for my headspace.", 7,
    schema='health', sig='stative',
    ents=['Melanie'], pqs=['What does Melanie do to destress?'])

# =========================================================================
# SESSION 8: 15 July 2023
# =========================================================================
print('Session 8...')

add('Melanie', 'take_kids_to', 'pottery workshop', 'Last Fri I finally took my kids to a pottery workshop. We all made our own pots.', 8,
    schema='hobby', texpr='last Friday', rdate='2023-07-07',
    ents=['Melanie'], pqs=['What activities has Melanie done with her family?'])

add('Melanie', 'make_with_kids', 'cup', 'The kids loved it! They were so excited to get their hands dirty and make something with clay.', 8,
    schema='hobby', ents=['Melanie'], pqs=["What types of pottery have Melanie and her kids made?"])

add('Melanie', 'paint_with_family', 'sunset', "We love painting together lately, especially nature-inspired ones. Here's our latest work from last weekend.", 8,
    schema='hobby', texpr='last weekend', rdate='2023-07-08',
    ents=['Melanie'], pqs=['What has Melanie painted?'])

add('Caroline', 'attend', 'council meeting for adoption', 'Last Friday I went to a council meeting for adoption.', 8,
    schema='family', texpr='last Friday', rdate='2023-07-07',
    ents=['Caroline'])

add('Caroline', 'favorite_color', 'blue', "Blue's my fave, it makes me feel relaxed.", 8,
    schema='identity', sig='stative', ents=['Caroline'])

add('Caroline', 'attend', 'pride parade', 'One special memory for me was this pride parade I went to a few weeks ago.', 8,
    schema='social', texpr='a few weeks ago', rdate='2023-06-26',
    ents=['Caroline'], pqs=['What LGBTQ+ events has Caroline participated in?'])

add('Melanie', 'go_camping_in', 'forest', 'We even went on another camping trip in the forest.', 8,
    schema='family', ents=['Melanie'], pqs=['Where has Melanie camped?'])

add('Melanie', 'enjoy_with_family', 'hiking in mountains and exploring forests', 'We enjoy hiking in the mountains and exploring forests.', 8,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie'])

# =========================================================================
# SESSION 9: 17 July 2023
# =========================================================================
print('Session 9...')

add('Melanie', 'go_camping_with', 'family two weekends ago', 'I had a quiet weekend after we went camping with my fam two weekends ago.', 9,
    schema='family', texpr='two weekends ago', rdate='2023-07-03',
    ents=['Melanie'])

add('Caroline', 'join', 'mentorship program for LGBTQ youth', 'Last weekend I joined a mentorship program for LGBTQ youth.', 9,
    schema='social', sig='milestone', texpr='last weekend', rdate='2023-07-15',
    ents=['Caroline'], pqs=['What LGBTQ+ events has Caroline participated in?', 'What events has Caroline participated in to help children?'])

add('Caroline', 'mentor', 'transgender teen', 'I mentor a transgender teen just like me.', 9,
    schema='social', sig='stative',
    ents=['Caroline'])

add('Caroline', 'plan', 'LGBTQ art show', "Next month I'm having an LGBTQ art show with my paintings.", 9,
    schema='hobby', temporal='future', texpr='next month', rdate='2023-08-01',
    ents=['Caroline'])

add('Caroline', 'paint', 'abstract art inspired by LGBTQ center', 'I painted this after I visited a LGBTQ center. I wanted to capture everyone\'s unity and strength.', 9,
    schema='hobby', ents=['Caroline'], pqs=['What kind of art does Caroline make?'])

add('Melanie', 'paint_with_kids', 'sunset painting', 'My kids and I just finished another painting like our last one.', 9,
    schema='hobby', ents=['Melanie'], pqs=['What has Melanie painted?'])

# =========================================================================
# SESSION 10: 20 July 2023
# =========================================================================
print('Session 10...')

add('Caroline', 'join', 'Connected LGBTQ Activists', "I just joined a new LGBTQ activist group last Tues. Our group, 'Connected LGBTQ Activists'.", 10,
    schema='social', sig='milestone', texpr='last Tuesday', rdate='2023-07-18',
    ents=['Caroline'], pqs=['What LGBTQ+ events has Caroline participated in?'])

add('Melanie', 'go_to', 'beach with kids', 'We went to the beach recently. The kids had such a blast.', 10,
    schema='family', ents=['Melanie'],
    pqs=['How many times has Melanie gone to the beach in 2023?'])

add('Melanie', 'go_to_beach', 'once or twice a year', "We don't go often, usually only once or twice a year.", 10,
    schema='family', sig='stative', ents=['Melanie'])

add('Melanie', 'enjoy_camping', 'roast marshmallows and tell stories', 'We roast marshmallows, tell stories around the campfire.', 10,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie'], pqs=['What does Melanie do with her family on hikes?'])

add('Melanie', 'remember', 'Perseid meteor shower camping trip', 'Our camping trip last year when we saw the Perseid meteor shower.', 10,
    schema='family', temporal='past', hist=1,
    ents=['Melanie'])

add('Melanie', 'witness', 'youngest daughter first steps', 'The day my youngest took her first steps.', 10,
    schema='family', sig='milestone', rel_type='familial',
    ents=['Melanie'])

# =========================================================================
# SESSION 11: 14 August 2023
# =========================================================================
print('Session 11...')

add('Melanie', 'attend_concert', 'Matt Patterson', 'It was Matt Patterson, he is so talented!', 11,
    schema='hobby', texpr='last night', rdate='2023-08-13',
    ents=['Melanie'], pqs=['What musical artists/bands has Melanie seen?'])

add('Caroline', 'attend', 'pride parade', 'I went to a pride parade last Friday.', 11,
    schema='social', texpr='last Friday', rdate='2023-08-11',
    ents=['Caroline'])

add('Caroline', 'make', 'abstract art', "Representing inclusivity and diversity in my art is important to me. Here's a recent painting!", 11,
    schema='hobby', ents=['Caroline'], pqs=['What kind of art does Caroline make?'])

add('Caroline', 'paint', 'Embracing Identity', "'Embracing Identity' is all about finding comfort and love in being yourself.", 11,
    schema='hobby', ents=['Caroline'])

add('Caroline', 'experience', 'changes to body during transition', "Art's allowed me to explore my transition and my changing body.", 11,
    schema='identity', ents=['Caroline'],
    pqs=['What are some changes Caroline has faced during her transition journey?'])

# =========================================================================
# SESSION 12: 17 August 2023
# =========================================================================
print('Session 12...')

add('Caroline', 'have_negative_experience_on', 'hike', 'I had a not-so-great experience on a hike. I ran into a group of religious conservatives.', 12,
    schema='social', emoval=0.2, emolabel='upset',
    ents=['Caroline'], pqs=['Who supports Caroline when she has a negative experience?'])

add('Melanie', 'make', 'bowl', "Here it is. Pretty proud of it! It was a great experience.", 12,
    schema='hobby', ents=['Melanie'], pqs=["What types of pottery have Melanie and her kids made?"])

add('Caroline', 'attend', 'Pride fest last year', 'We had a blast last year at the Pride fest.', 12,
    schema='social', temporal='past', hist=1, ents=['Caroline'])

# =========================================================================
# SESSION 13: 23 August 2023
# =========================================================================
print('Session 13...')

add('Caroline', 'apply_to', 'adoption agencies', 'I applied to adoption agencies!', 13,
    schema='family', sig='milestone',
    ents=['Caroline'])

add('Caroline', 'have_pet', 'guinea pig named Oscar', 'Oscar, my guinea pig.', 13,
    schema='family', sig='stative',
    ents=['Caroline', 'Oscar'], pqs=["What is Caroline's pet's name?"])

add('Melanie', 'have_pet', 'cat named Bailey', 'We got another cat named Bailey too.', 13,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie', 'Bailey'], pqs=["What are Melanie's pets' names?"])

add('Caroline', 'used_to', 'horseback riding with dad', 'I used to go horseback riding with my dad when I was a kid.', 13,
    schema='hobby', temporal='past', hist=1,
    ents=['Caroline'])

add('Melanie', 'paint', 'horse', "Here's a photo of my horse painting I did recently.", 13,
    schema='hobby', ents=['Melanie'], pqs=['What has Melanie painted?'])

add('Caroline', 'paint', 'self-portrait', "Here's a recent self-portrait I made last week.", 13,
    schema='hobby', texpr='last week', rdate='2023-08-16',
    ents=['Caroline'])

# =========================================================================
# SESSION 14: 25 August 2023
# =========================================================================
print('Session 14...')

add('Melanie', 'make', 'plate', 'I made it in pottery class yesterday.', 14,
    schema='hobby', texpr='yesterday', rdate='2023-08-24',
    ents=['Melanie'])

add('Caroline', 'paint', 'sunset', "I painted it after I visited the beach last week.", 14,
    schema='hobby', texpr='last week', rdate='2023-08-18',
    ents=['Caroline'], pqs=['What subject have Caroline and Melanie both painted?'])

add('Melanie', 'volunteer_at', 'homeless shelter', 'Spending the day with my fam volunteering at a homeless shelter.', 14,
    schema='social', ents=['Melanie'])

add('Caroline', 'value', 'rainbow flag mural', 'The rainbow flag mural is important to me as it reflects the courage and strength of the trans community.', 14,
    schema='identity', ents=['Caroline'],
    pqs=['What symbols are important to Caroline?'])

add('Caroline', 'make', 'stained glass window', 'I made this stained glass window to remind myself and others.', 14,
    schema='hobby', ents=['Caroline'])

add('Caroline', 'plan', 'LGBTQ art show next month', "I'm putting together an LGBTQ art show next month.", 14,
    schema='hobby', temporal='future', texpr='next month', rdate='2023-09-01',
    ents=['Caroline'])

add('Caroline', 'lose', 'unsupportive friends during transition', "Some close friends kept supporting me, but a few weren't able to handle it.", 14,
    schema='identity', emoval=0.3, emolabel='sad',
    ents=['Caroline'], pqs=['What are some changes Caroline has faced during her transition journey?'])

add('Melanie', 'enjoy_painting', 'landscapes and still life', 'Painting landscapes and still life is my favorite!', 14,
    schema='hobby', sig='stative', ents=['Melanie'])

# =========================================================================
# SESSION 15: 28 August 2023
# =========================================================================
print('Session 15...')

add('Caroline', 'volunteer_at', 'LGBTQ+ youth center', 'I had the chance to volunteer at an LGBTQ+ youth center.', 15,
    schema='social', sig='milestone', ents=['Caroline'])

add('Caroline', 'plan', 'talent show for kids', "We're putting together a talent show for the kids next month.", 15,
    schema='social', temporal='future', texpr='next month',
    ents=['Caroline'])

add('Melanie', 'attend_concert', 'Summer Sounds', '"Summer Sounds"- they played an awesome pop song.', 15,
    schema='hobby', ents=['Melanie'], pqs=['What musical artists/bands has Melanie seen?'])

add('Caroline', 'play', 'acoustic guitar', 'I started playing acoustic guitar about five years ago.', 15,
    schema='hobby', sig='stative', texpr='five years ago', rdate='2018-08-01',
    ents=['Caroline'], pqs=['What instruments does Caroline play?'])

add('Caroline', 'love_song', 'Brave by Sara Bareilles', '"Brave" by Sara Bareilles has a lot of significance for me.', 15,
    schema='hobby', sig='stative', ents=['Caroline'])

add('Melanie', 'play', 'clarinet', 'Yeah, I play clarinet! Started when I was young.', 15,
    schema='hobby', sig='stative',
    ents=['Melanie'], pqs=['What instruments does Melanie play?'])

add('Melanie', 'also_play', 'violin', 'running, reading, or playing my violin', 15,
    schema='hobby', sig='stative',
    ents=['Melanie'], pqs=['What instruments does Melanie play?'])

# =========================================================================
# SESSION 16: 13 September 2023
# =========================================================================
print('Session 16...')

add('Caroline', 'go_biking_with', 'friends', 'I had a wicked day out with the gang last weekend - we went biking.', 16,
    schema='social', texpr='last weekend', rdate='2023-09-09',
    ents=['Caroline'])

add('Caroline', 'create_art_since', 'age 17', "Since I was 17 or so. I find it so empowering and cathartic.", 16,
    schema='hobby', sig='stative', ents=['Caroline'])

add('Melanie', 'do_art_for', '7 years', 'Seven years now, painting and pottery.', 16,
    schema='hobby', sig='stative', ents=['Melanie'])

add('Caroline', 'paint', 'red and blue painting about trans journey', 'I made this painting to show my path as a trans woman. The red and blue are for the binary gender system.', 16,
    schema='hobby', ents=['Caroline'])

add('Caroline', 'lose', 'unsupportive friends during transition', "It's definitely changed them. Some close friends kept supporting me, but a few weren't able to handle it.", 16,
    schema='identity', ents=['Caroline'])

add('Melanie', 'go_to', 'cafe', "Here's to a good time at the cafe last weekend.", 16,
    schema='social', texpr='last weekend', rdate='2023-09-09',
    ents=['Melanie'])

add('Melanie', 'go_camping', 'with kids a few weeks ago', 'We went camping with the kids a few weeks ago.', 16,
    schema='family', texpr='a few weeks ago',
    ents=['Melanie'])

add('Melanie', 'roast_marshmallows_and', 'tell stories', 'We roasted marshmallows and shared stories around the campfire.', 16,
    schema='family', ents=['Melanie'],
    pqs=['What does Melanie do with her family on hikes?'])

# =========================================================================
# SESSION 17: 13 October 2023
# =========================================================================
print('Session 17...')

add('Caroline', 'contact', 'mentor for adoption advice', 'I just contacted my mentor for adoption advice.', 17,
    schema='family', ents=['Caroline'])

add('Melanie', 'have_friend_who', 'adopted last year', 'A buddy of mine adopted last year.', 17,
    schema='social', temporal='past', hist=1, ents=['Melanie'])

add('Melanie', 'get_injured', 'had to take break from pottery', 'Last month I got hurt and had to take a break from pottery.', 17,
    schema='health', emoval=0.3, emolabel='frustrated', texpr='last month', rdate='2023-09-13',
    ents=['Melanie'])

add('Melanie', 'read', 'book Caroline recommended', "Been reading that book you recommended a while ago.", 17,
    schema='hobby', ents=['Melanie', 'Caroline'],
    pqs=['What book did Melanie read from Caroline\'s suggestion?'])

add('Melanie', 'paint', 'sunset', "Here's one I did last week. It's inspired by the sunsets.", 17,
    schema='hobby', texpr='last week', rdate='2023-10-06',
    ents=['Melanie'], pqs=['What has Melanie painted?'])

add('Caroline', 'attend', 'transgender poetry reading', 'I went to a poetry reading last Fri - it was really powerful!', 17,
    schema='social', texpr='last Friday', rdate='2023-10-06',
    ents=['Caroline'], pqs=['What transgender-specific events has Caroline attended?'])

add('Caroline', 'value', 'transgender symbol', 'It stands for freedom and being real.', 17,
    schema='identity', ents=['Caroline'],
    pqs=['What symbols are important to Caroline?'])

# =========================================================================
# SESSION 18: 20 October 2023
# =========================================================================
print('Session 18...')

add('Melanie', 'have_son_in', 'accident on road trip', "My son got into an accident. We were so lucky he was okay.", 18,
    schema='family', sig='milestone', emoval=0.2, emolabel='scared',
    texpr='this past weekend', rdate='2023-10-14',
    ents=['Melanie'])

add('Melanie', 'have', '3 kids', 'The kids look so cute.', 18,
    schema='family', sig='stative', rel_type='familial',
    ents=['Melanie'], pqs=['How many children does Melanie have?'])

add('Melanie', 'go_hiking', 'yesterday with kids', 'Yup, we just did it yesterday! The kids loved it.', 18,
    schema='family', texpr='yesterday', rdate='2023-10-19',
    ents=['Melanie'], pqs=['When did Melanie go on a hike after the roadtrip?'])

# =========================================================================
# SESSION 19: 22 October 2023
# =========================================================================
print('Session 19...')

add('Caroline', 'pass', 'adoption agency interviews', 'I passed the adoption agency interviews last Friday!', 19,
    schema='family', sig='milestone', texpr='last Friday', rdate='2023-10-20',
    ents=['Caroline'])

add('Melanie', 'buy', 'figurines', 'These figurines I bought yesterday remind me of family love.', 19,
    schema='hobby', texpr='yesterday', rdate='2023-10-21',
    ents=['Melanie'], pqs=['What items has Melanie bought?'])

conn.commit()
elapsed = time.time() - t0
cnt = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
ent_cnt = conn.execute('SELECT COUNT(*) FROM entities').fetchone()[0]
print(f'\nDone: {cnt} edges, {ent_cnt} entities in {elapsed:.1f}s')
conn.close()
