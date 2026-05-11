"""Add missing edges for conv 0 Cat 4 + Cat 2 + Cat 1 questions."""

import sqlite3, hashlib, json, sys, os, time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = 'locomo_conv0_direct.db'
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

from app.vector.embedder import embed_text
print('Embedder loaded')

USER_ID = 0
_seq = conn.execute('SELECT MAX(sequence_number) FROM edges').fetchone()[0] or 0

def ts(s):
    from dateutil import parser as dp
    try: return dp.parse(s).isoformat()
    except: return None

S = {
    1: ts('1:56 pm on 8 May, 2023'), 2: ts('1:14 pm on 25 May, 2023'),
    3: ts('7:55 pm on 9 June, 2023'), 4: ts('10:37 am on 27 June, 2023'),
    5: ts('1:36 pm on 3 July, 2023'), 6: ts('8:18 pm on 6 July, 2023'),
    7: ts('4:33 pm on 12 July, 2023'), 8: ts('1:51 pm on 15 July, 2023'),
    9: ts('2:31 pm on 17 July, 2023'), 10: ts('8:56 pm on 20 July, 2023'),
    11: ts('2:24 pm on 14 August, 2023'), 12: ts('1:50 pm on 17 August, 2023'),
    13: ts('3:31 pm on 23 August, 2023'), 14: ts('1:33 pm on 25 August, 2023'),
    15: ts('3:19 pm on 28 August, 2023'), 16: ts('12:09 am on 13 September, 2023'),
    17: ts('10:31 am on 13 October, 2023'), 18: ts('6:55 pm on 20 October, 2023'),
    19: ts('9:55 am on 22 October, 2023'),
}

def add(subj, pred, obj, src, session, schema='uncategorized',
        temporal='present', rel_type='personal', sig='routine',
        emoval=0.5, emolabel=None, texpr=None, rdate=None,
        hist=0, ents=None, pqs=None):
    global _seq
    _seq += 1
    h = hashlib.sha256(src.encode()).hexdigest()
    # Check for dups
    existing = conn.execute('SELECT id FROM edges WHERE source_text_hash=? AND user_id=?', (h, USER_ID)).fetchone()
    if existing:
        return existing[0]
    ee = embed_text(src).tobytes()
    pe = embed_text(pred.replace('_', ' ')).tobytes() if pred else None
    re = json.dumps(ents) if ents else None
    cols = ['user_id','subject','predicate','object','source_text','source_text_hash',
            'edge_schematic_category','edge_temporal_context','edge_relational_type',
            'edge_episodic_significance','edge_emotional_valence','edge_emotional_label',
            'source_timestamp','temporal_expression','resolved_event_date',
            'is_historical','is_current','sequence_number','relational_entities',
            'edge_embedding','predicate_embedding']
    vals = [USER_ID, subj, pred, obj, src, h, schema, temporal, rel_type, sig, emoval, emolabel,
            S.get(session), texpr, rdate, hist, 1, _seq, re, ee, pe]
    if pqs:
        for i, pq in enumerate(pqs[:4]):
            cols.append(f'pq_{i+1}')
            vals.append(pq)
    ph = ','.join(['?']*len(vals))
    cn = ','.join(cols)
    conn.execute(f'INSERT INTO edges ({cn}) VALUES ({ph})', vals)
    eid = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    try:
        fv = [eid, subj or '', (pred or '').replace('_',' '), obj or '', src]
        for i in range(4):
            fv.append(pqs[i] if pqs and i < len(pqs) else None)
        fv.extend([None, None])
        conn.execute('INSERT INTO edges_fts(rowid,subject,predicate,object,source_text,pq_1,pq_2,pq_3,pq_4,vq_1,vq_2) VALUES (?,?,?,?,?,?,?,?,?,?,?)', fv)
    except: pass
    return eid

t0 = time.time()

# ===== MISSING DATA FROM AUDIT =====

# D2:2 charity race raised awareness for mental health
add('Melanie', 'raise_awareness_for', 'mental health', 'Making a difference & raising awareness for mental health is super rewarding.', 2,
    schema='health', ents=['Melanie'], pqs=['What did the charity race raise awareness for?'])

# D2:8 Caroline's summer plans
add('Caroline', 'plan_for_summer', 'researching adoption agencies', "Researching adoption agencies - it's been a dream to have a family and give a loving home to kids who need it.", 2,
    schema='family', ents=['Caroline'], pqs=["What are Caroline's plans for the summer?"])

# D4:3 Caroline's necklace symbolizes love faith strength
add('Caroline', 'necklace_symbolize', 'love, faith and strength', 'She gave it to me when I was young, and it stands for love, faith and strength.', 4,
    schema='identity', ents=['Caroline'],
    pqs=["What does Caroline's necklace symbolize?", "What country is Caroline's grandma from?"])

# D4:3 Grandma's gift = necklace
add('Caroline', 'receive_gift_from', 'necklace from grandma', 'A gift from my grandma in my home country, Sweden. She gave it to me when I was young.', 4,
    schema='identity', ents=['Caroline'],
    pqs=["What was grandma's gift to Caroline?"])

# D4:5 hand-painted bowl reminder of art and self-expression
add('Caroline', 'hand_painted_bowl_reminds_of', 'art and self-expression', "The pattern and colors are awesome-- it reminds me of art and self-expression.", 4,
    schema='identity', ents=['Caroline'],
    pqs=["What is Caroline's hand-painted bowl a reminder of?"])

# D4:6-8 Melanie family camping activities
add('Melanie', 'do_while_camping', 'explored nature, roasted marshmallows, and went on a hike', 'We explored nature, roasted marshmallows around the campfire and even went on a hike.', 4,
    schema='family', ents=['Melanie'],
    pqs=['What did Melanie and her family do while camping?'])

# D4:11-13 Caroline interested in counseling for trans people
add('Caroline', 'want_to_provide', 'counseling for trans people', "I'm thinking of working with trans people, helping them accept themselves and supporting their mental health.", 4,
    schema='career', sig='stative', ents=['Caroline'],
    pqs=['What kind of counseling and mental health services is Caroline interested in?'])

# D4:13 Caroline attended LGBTQ+ counseling workshop
add('Caroline', 'attend', 'LGBTQ+ counseling workshop discussing therapeutic methods', 'Last Friday, I went to an LGBTQ+ counseling workshop where we discussed therapeutic methods and how to best work with trans people.', 4,
    schema='career', texpr='last Friday', rdate='2023-06-23', ents=['Caroline'],
    pqs=['What workshop did Caroline attend recently?', 'What was discussed in the LGBTQ+ counseling workshop?'])

# D4:15 motivation for counseling
add('Caroline', 'motivated_to_counsel_by', 'own journey and support received', 'My own journey and the support I got made a huge difference. I saw how counseling and support groups improved my life.', 4,
    schema='career', ents=['Caroline'],
    pqs=['What motivated Caroline to pursue counseling?'])

# D5:8 Melanie made black and white bowl
add('Melanie', 'make', 'black and white bowl', 'Yeah, I made this bowl in my class. It took some work, but I\'m pretty proud of it.', 5,
    schema='hobby', ents=['Melanie'],
    pqs=['Did Melanie make the black and white bowl in the photo?'])

# D8:5 creative projects besides pottery
add('Melanie', 'do_with_kids_besides_pottery', 'painting', 'We love painting together lately, especially nature-inspired ones.', 8,
    schema='hobby', ents=['Melanie'],
    pqs=['What creative project do Mel and her kids do together besides pottery?'])

# D8:6 Mel and kids painted sunset with palm tree
add('Melanie', 'paint_with_kids_recently', 'sunset with a palm tree', "We love painting together lately, especially nature-inspired ones. Here's our latest work from last weekend.", 8,
    schema='hobby', texpr='last weekend', rdate='2023-07-08', ents=['Melanie'],
    pqs=['What did Mel and her kids paint in their latest project in July 2023?'])

# D8:9 council meeting for adoption
add('Caroline', 'see_at_council_meeting', 'many people wanting to create loving homes for children', 'Last Friday I went to a council meeting for adoption. It was inspiring and emotional - so many people wanted to create loving homes for kids.', 8,
    schema='family', texpr='last Friday', rdate='2023-07-07', ents=['Caroline'],
    pqs=['What did Caroline see at the council meeting for adoption?'])

# D8:11 sunflowers represent warmth and happiness
add('Caroline', 'believe_sunflowers_represent', 'warmth and happiness', "Sunflowers mean warmth and happiness, right?", 8,
    schema='identity', ents=['Caroline'],
    pqs=['What do sunflowers represent according to Caroline?'])

# D8:12 flowers important to Melanie
add('Melanie', 'value_flowers_because', 'remind her to appreciate small moments and were part of wedding decor', 'Flowers bring joy. They represent growth, beauty and reminding us to appreciate the small moments. They were an important part of my wedding decor.', 8,
    schema='identity', ents=['Melanie'],
    pqs=['Why are flowers important to Melanie?'])

# D9:16 art show painting inspiration
add('Caroline', 'paint_art_show_piece_inspired_by', 'visiting LGBTQ center, capturing unity and strength', 'I painted this after I visited a LGBTQ center. I wanted to capture everyone\'s unity and strength.', 9,
    schema='hobby', ents=['Caroline'],
    pqs=["What inspired Caroline's painting for the art show?"])

# D10:10 beach frequency
add('Melanie', 'go_to_beach', 'once or twice a year', "We don't go often, usually only once or twice a year.", 10,
    schema='family', sig='stative', ents=['Melanie'],
    pqs=['How often does Melanie go to the beach with her kids?'])

# D10:14 Perseid meteor shower
add('Melanie', 'see_during_camping', 'Perseid meteor shower', 'Our camping trip last year when we saw the Perseid meteor shower. It was so amazing.', 10,
    schema='family', temporal='past', hist=1, ents=['Melanie'],
    pqs=['What did Melanie and her family see during their camping trip last year?'])

# D10:18 how Melanie felt during meteor shower
add('Melanie', 'feel_during_meteor_shower', 'in awe of the universe', 'It was one of those moments where I felt tiny and in awe of the universe.', 10,
    schema='family', emoval=0.9, emolabel='awe', ents=['Melanie'],
    pqs=['How did Melanie feel while watching the meteor shower?'])

# D11:1 daughter birthday concert
add('Melanie', 'celebrate', "daughter's birthday with concert", "We celebrated my daughter's birthday with a concert surrounded by music, joy and the warm summer breeze.", 11,
    schema='family', sig='milestone', texpr='last night', rdate='2023-08-13', ents=['Melanie'],
    pqs=["Whose birthday did Melanie celebrate recently?"])

# D11:3 Matt Patterson performed
add('Melanie', 'see_perform', 'Matt Patterson', 'It was Matt Patterson, he is so talented! His voice and songs were amazing.', 11,
    schema='hobby', ents=['Melanie'],
    pqs=["Who performed at the concert at Melanie's daughter's birthday?"])

# D12:6 pottery colors and patterns
add('Melanie', 'use_colors_in_pottery_to', 'catch the eye and make people smile', "I'm obsessed with those, so I made something to catch the eye and make people smile.", 12,
    schema='hobby', ents=['Melanie'],
    pqs=['Why did Melanie choose to use colors and patterns in her pottery project?'])

# D13:3 Caroline has guinea pig
add('Caroline', 'have_pet', 'guinea pig named Oscar', 'Oscar, my guinea pig. He\'s been great.', 13,
    schema='family', sig='stative', ents=['Caroline', 'Oscar'],
    pqs=['What pet does Caroline have?', "What is Caroline's pet's name?"])

# D13:4 Melanie's pets - two cats and a dog
add('Melanie', 'have_pets', 'two cats and a dog', "We got another cat named Bailey too. We've got a pup and a kitty.", 13,
    schema='family', sig='stative', ents=['Melanie', 'Oliver', 'Luna', 'Bailey'],
    pqs=['What pets does Melanie have?'])

# D13:6 Oliver hid bone in slipper
add('Melanie', 'funny_pet_story', 'Oliver hid bone in slipper', "Oliver's hilarious! He hid his bone in my slipper once!", 13,
    schema='family', ents=['Melanie', 'Oliver'],
    pqs=["Where did Oliver hide his bone once?"])

# D13:7 Caroline used to horseback ride with dad
add('Caroline', 'used_to_do_with_dad', 'horseback riding', 'I used to go horseback riding with my dad when I was a kid.', 13,
    schema='hobby', temporal='past', hist=1, ents=['Caroline'],
    pqs=['What activity did Caroline used to do with her dad?'])

# D14:17 stained glass window for church
add('Caroline', 'make_for_church', 'stained glass window', 'I made this stained glass window for a local church. It shows time changing our lives.', 14,
    schema='hobby', ents=['Caroline'],
    pqs=['What did Caroline make for a local church?'])

# D14:23 rainbow sidewalk in neighborhood
add('Caroline', 'find_in_neighborhood', 'rainbow sidewalk for Pride Month', 'I came across this cool rainbow sidewalk for Pride Month. It was so vibrant and welcoming.', 14,
    schema='social', ents=['Caroline'],
    pqs=['What did Caroline find in her neighborhood during her walk?'])

# D15:23 Brave by Sara Bareilles
add('Caroline', 'love_song', 'Brave by Sara Bareilles', '"Brave" by Sara Bareilles has a lot of significance for me. It\'s about being courageous and fighting for what\'s right.', 15,
    schema='hobby', sig='stative', ents=['Caroline'],
    pqs=['Which song motivates Caroline to be courageous?'])

# D15:28 Melanie enjoys Bach, Mozart, Ed Sheeran
add('Melanie', 'enjoy_listening_to', 'Bach, Mozart, and Ed Sheeran', "I'm a fan of both classical like Bach and Mozart, as well as modern music like Ed Sheeran's Perfect.", 15,
    schema='hobby', sig='stative', ents=['Melanie'],
    pqs=["Which classical musicians does Melanie enjoy listening to?", "Who is Melanie a fan of in terms of modern music?"])

# D16:7 Caroline creating art since age 17
add('Caroline', 'create_art_since', 'age 17', 'Since I was 17 or so. I find it so empowering and cathartic.', 16,
    schema='hobby', sig='stative', ents=['Caroline'],
    pqs=['How long has Caroline been creating art?'])

# D16:16-18 precautionary sign at cafe
add('Melanie', 'see_at_cafe', 'precautionary sign', 'The sign was just a precaution, I had a great time. But thank you for your concern.', 16,
    schema='social', ents=['Melanie'],
    pqs=['What precautionary sign did Melanie see at the cafe?'])

# D17:7 adoption advice
add('Caroline', 'give_adoption_advice', 'do research, find agency or lawyer, gather documents', "Do your research and find an adoption agency or lawyer. They'll help with the process. Gather documents like references, financial info and medical checks.", 17,
    schema='family', ents=['Caroline'],
    pqs=['What advice does Caroline give for getting started with adoption?'])

# D17:8 Melanie setback - injured
add('Melanie', 'get_injured_and', 'take break from pottery', 'Last month I got hurt and had to take a break from pottery.', 17,
    schema='health', emoval=0.3, emolabel='frustrated', texpr='last month', rdate='2023-09-13', ents=['Melanie'],
    pqs=['What setback did Melanie face in October 2023?'])

# D17:10 Melanie keeping busy during break
add('Melanie', 'do_during_pottery_break', 'read book and paint', "Been reading that book you recommended a while ago and painting to keep busy.", 17,
    schema='hobby', ents=['Melanie'],
    pqs=['What does Melanie do to keep herself busy during her pottery break?'])

# D17:19 poetry reading posters
add('Caroline', 'see_at_poetry_reading', 'Trans Lives Matter posters', 'The posters were amazing, so much pride and strength!', 17,
    schema='social', ents=['Caroline'],
    pqs=['What did the posters at the poetry reading say?'])

# D17:23 drawing symbolizes freedom
add('Caroline', 'draw_symbolizing', 'freedom and being true to herself', 'It stands for freedom and being real. It\'s like a nudge to always stay true to myself.', 17,
    schema='identity', ents=['Caroline'],
    pqs=["What does Caroline's drawing symbolize for her?"])

# D17:25 journey description
add('Caroline', 'describe_life_journey_as', 'ongoing adventure of learning and growing', "Being ourselves is such a great feeling. It's an ongoing adventure of learning and growing.", 17,
    schema='identity', ents=['Caroline', 'Melanie'],
    pqs=['How do Melanie and Caroline describe their journey through life together?'])

# D18:1 son accident on road trip
add('Melanie', 'experience', 'son got into accident on road trip', "My son got into an accident. We were so lucky he was okay. It was a real scary experience.", 18,
    schema='family', sig='milestone', emoval=0.2, emolabel='scared', texpr='this past weekend', rdate='2023-10-14', ents=['Melanie'],
    pqs=['What happened to Melanie\'s son on their road trip?'])

# D18:5 Melanie feelings after accident
add('Melanie', 'feel_after_accident', 'family is important and means the world', "Family's super important to me. Especially after the accident, I've thought a lot about how much I need them.", 18,
    schema='family', emoval=0.8, emolabel='grateful', ents=['Melanie'],
    pqs=['How did Melanie feel about her family after the accident?', 'How did Melanie feel after the accident?'])

# D18:7 children handled accident
add('Melanie', 'children_reaction_to_accident', 'scared but resilient', 'They were scared but we reassured them and explained their brother would be OK. They\'re tough kids.', 18,
    schema='family', ents=['Melanie'],
    pqs=["How did Melanie's children handle the accident?", "How did Melanie's son handle the accident?"])

# D18:9 family gives strength
add('Melanie', 'get_from_family', 'strength and motivation', "They give me the strength to keep going.", 18,
    schema='family', sig='stative', ents=['Melanie'],
    pqs=["What do Melanie's family give her?"])

# D18:13 Melanie appreciates family support
add('Melanie', 'feel_about_family_support', 'appreciated them a lot', "They're a real support. Appreciate them a lot.", 18,
    schema='family', emoval=0.8, emolabel='grateful', ents=['Melanie'],
    pqs=['How did Melanie feel about her family supporting her?'])

# D18:17 after road trip relaxation
add('Melanie', 'do_after_road_trip', 'went on hike with kids', 'We just did it yesterday! The kids loved it and it was a nice way to relax after the road trip.', 18,
    schema='family', texpr='yesterday', rdate='2023-10-19', ents=['Melanie'],
    pqs=['What did Melanie do after the road trip to relax?'])

# D4:13 Caroline wants to create safe inviting place
add('Caroline', 'want_to_create', 'safe and inviting place for people to grow', 'I want to create a safe space for people to grow and find support.', 4,
    schema='career', ents=['Caroline'],
    pqs=['What kind of place does Caroline want to create for people?'])

# D9:12 LGBTQ art show with paintings
add('Caroline', 'organize', 'LGBTQ art show', "Next month I'm having an LGBTQ art show with my paintings - can't wait!", 9,
    schema='hobby', temporal='future', texpr='next month', rdate='2023-08-01', ents=['Caroline'],
    pqs=['What LGBTQ+ events has Caroline participated in?'])

# D14:5 Caroline painted sunset
add('Caroline', 'paint', 'sunset painting', "I painted it after I visited the beach last week. The sun dipping below the horizon, all the amazing colors.", 14,
    schema='hobby', texpr='last week', rdate='2023-08-18', ents=['Caroline'],
    pqs=['What subject have Caroline and Melanie both painted?'])

conn.commit()
elapsed = time.time() - t0
cnt = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
print(f'Done: {cnt} edges total ({cnt - 112} new) in {elapsed:.1f}s')
conn.close()
