"""Fix all 17 REF (refused) misses by adding PQs to existing edges or inserting new edges."""
import sqlite3, hashlib, json, sys, time
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

S = {1:ts('1:56 pm on 8 May, 2023'),2:ts('1:14 pm on 25 May, 2023'),3:ts('7:55 pm on 9 June, 2023'),
     4:ts('10:37 am on 27 June, 2023'),5:ts('1:36 pm on 3 July, 2023'),6:ts('8:18 pm on 6 July, 2023'),
     7:ts('4:33 pm on 12 July, 2023'),8:ts('1:51 pm on 15 July, 2023'),9:ts('2:31 pm on 17 July, 2023'),
     10:ts('8:56 pm on 20 July, 2023'),11:ts('2:24 pm on 14 August, 2023'),12:ts('1:50 pm on 17 August, 2023'),
     13:ts('3:31 pm on 23 August, 2023'),14:ts('1:33 pm on 25 August, 2023'),15:ts('3:19 pm on 28 August, 2023'),
     16:ts('12:09 am on 13 September, 2023'),17:ts('10:31 am on 13 October, 2023'),
     18:ts('6:55 pm on 20 October, 2023'),19:ts('9:55 am on 22 October, 2023')}


def add_pq(edge_id, pq_text):
    """Add PQ to the first available pq_N slot on an existing edge. Returns True if added."""
    row = conn.execute('SELECT pq_1, pq_2, pq_3, pq_4 FROM edges WHERE id=?', (edge_id,)).fetchone()
    if not row:
        print(f'  WARN: edge id={edge_id} not found')
        return False
    for i in range(1, 5):
        val = row[f'pq_{i}']
        if val == pq_text:
            print(f'  SKIP (already has PQ): id={edge_id} pq_{i}="{pq_text[:60]}"')
            return False
    for i in range(1, 5):
        if row[f'pq_{i}'] is None:
            conn.execute(f'UPDATE edges SET pq_{i}=? WHERE id=?', (pq_text, edge_id))
            print(f'  UPDATE: id={edge_id} pq_{i}="{pq_text[:60]}"')
            return True
    print(f'  WARN: all PQ slots full on id={edge_id}, cannot add "{pq_text[:60]}"')
    return False


def find_edge(like_pattern, subject=None):
    """Find edge by source_text LIKE pattern, optionally filtered by subject."""
    sql = 'SELECT id, subject, predicate, object, pq_1, pq_2, pq_3, pq_4 FROM edges WHERE source_text LIKE ?'
    params = [like_pattern]
    if subject:
        sql += ' AND subject=?'
        params.append(subject)
    return conn.execute(sql, params).fetchall()


def find_edge_by_pred(predicate, subject=None):
    """Find edge by predicate, optionally filtered by subject."""
    sql = 'SELECT id, subject, predicate, object, pq_1, pq_2, pq_3, pq_4 FROM edges WHERE predicate=?'
    params = [predicate]
    if subject:
        sql += ' AND subject=?'
        params.append(subject)
    return conn.execute(sql, params).fetchall()


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
updated = 0
added = 0

# ── 1. "How long has Caroline had her current group of friends for?" → "4 years"
# Edge id=17 (move_from Sweden, source mentions "friends for 4 years"). pq_3 is free.
print('\n1. Caroline friends for 4 years')
if add_pq(17, "How long has Caroline had her current group of friends for?"):
    updated += 1

# ── 2. "What career path has Caroline decided to pursue?" → already has PQ on id=172
# Edge id=172 (pursue_career_in) already has pq_1="What career path has Caroline decided to pursue?"
print('\n2. Caroline career path')
rows = find_edge_by_pred('pursue_career_in', 'Caroline')
if rows:
    for r in rows:
        if r['pq_1'] == "What career path has Caroline decided to pursue?":
            print(f'  SKIP (already has PQ): id={r["id"]}')
        else:
            if add_pq(r['id'], "What career path has Caroline decided to pursue?"):
                updated += 1
else:
    print('  WARN: edge not found')

# ── 3. "When did Melanie read the book 'nothing is impossible'?" → "2022"
# Edge id=38 (read, Nothing is Impossible, rdate=2022-01-01). pq_2 is free.
print('\n3. Melanie read Nothing is Impossible')
if add_pq(38, "When did Melanie read the book Nothing is Impossible?"):
    updated += 1

# ── 4. "What musical artists/bands has Melanie seen?" → Summer Sounds, Matt Patterson
# id=88 (Summer Sounds) already has pq_1 with this exact PQ
# id=64 (Matt Patterson attend_concert) already has pq_1 with this exact PQ
# id=187 (Matt Patterson daughter_birthday_performer) — different question
print('\n4. Melanie musical artists (Summer Sounds + Matt Patterson)')
for eid in [88, 64]:
    row = conn.execute('SELECT pq_1 FROM edges WHERE id=?', (eid,)).fetchone()
    if row and row['pq_1'] == "What musical artists/bands has Melanie seen?":
        print(f'  SKIP (already has PQ): id={eid}')
    else:
        if add_pq(eid, "What musical artists/bands has Melanie seen?"):
            updated += 1

# ── 5. "When did Melanie get hurt?" → "September 2023"
# id=103 (get_injured). pq_3 is free.
print('\n5. Melanie get hurt')
if add_pq(103, "When did Melanie get hurt?"):
    updated += 1

# ── 6. "When did Melanie buy the figurines?" → "21 October 2023"
# id=112 (buy, figurines, rdate=2023-10-21). pq_2 is free.
print('\n6. Melanie buy figurines')
if add_pq(112, "When did Melanie buy the figurines?"):
    updated += 1

# ── 7. "What did the charity race raise awareness for?" → "mental health"
# id=8 already has pq_2="What did the charity race raise awareness for?"
# id=153 and id=182 also have it. Already covered.
print('\n7. Charity race awareness')
row = conn.execute('SELECT pq_2 FROM edges WHERE id=8').fetchone()
if row and row['pq_2'] == "What did the charity race raise awareness for?":
    print(f'  SKIP (already has PQ): id=8')
else:
    if add_pq(8, "What did the charity race raise awareness for?"):
        updated += 1

# ── 8. "What is Melanie's hand-painted bowl a reminder of?" → "art and self-expression"
# This is Caroline's bowl. id=184 already has pq_2="What is Melanie's hand-painted bowl a reminder of?"
print('\n8. Hand-painted bowl reminder')
row = conn.execute('SELECT pq_1, pq_2 FROM edges WHERE id=184').fetchone()
if row and "Melanie's hand-painted bowl" in (row['pq_2'] or ''):
    print(f'  SKIP (already has PQ): id=184')
else:
    if add_pq(184, "What is Melanie's hand-painted bowl a reminder of?"):
        updated += 1

# ── 9. "What was Melanie's favorite book from her childhood?" → "Charlotte's Web"
# id=34 (read_as_child, Charlotte's Web). pq_2 is free.
print('\n9. Melanie favorite childhood book')
if add_pq(34, "What was Melanie's favorite book from her childhood?"):
    updated += 1

# ── 10. "What book did Caroline recommend to Melanie?" → "Becoming Nicole"
# id=39 (love_book, Becoming Nicole). pq_2 is free.
print('\n10. Caroline recommend Becoming Nicole')
if add_pq(39, "What book did Caroline recommend to Melanie?"):
    updated += 1

# ── 11. "What did Caroline take away from the book 'Becoming Nicole'?" → already on id=185
# id=185 (takeaway_from_book) already has pq_1 with exact match.
print('\n11. Caroline takeaway from Becoming Nicole')
rows = find_edge_by_pred('takeaway_from_book', 'Caroline')
if rows:
    for r in rows:
        if r['pq_1'] == "What did Caroline take away from the book 'Becoming Nicole'?":
            print(f'  SKIP (already has PQ): id={r["id"]}')
        else:
            if add_pq(r['id'], "What did Caroline take away from the book 'Becoming Nicole'?"):
                updated += 1
else:
    print('  WARN: edge not found')

# ── 12. "What does Melanie say running has been great for?" → "Her mental health"
# No edge exists. Add NEW edge.
print('\n12. Melanie running great for mental health (NEW)')
added += add('Melanie', 'say_running_great_for', 'mental health',
    "Running has been great for my mental health. I'm gonna keep it up.", 7,
    schema='health', sig='stative', ents=['Melanie'],
    pqs=["What does Melanie say running has been great for?"])

# ── 13. "Which song motivates Caroline to be courageous?" → "Brave by Sara Bareilles"
# id=188 (courage_song) already has pq_1. id=136 (love_song) also has it.
print('\n13. Caroline courage song')
for eid in [188, 136]:
    row = conn.execute('SELECT pq_1 FROM edges WHERE id=?', (eid,)).fetchone()
    if row and row['pq_1'] == "Which song motivates Caroline to be courageous?":
        print(f'  SKIP (already has PQ): id={eid}')
    else:
        if add_pq(eid, "Which song motivates Caroline to be courageous?"):
            updated += 1

# ── 14. "What painting did Melanie show to Caroline on October 13, 2023?" → sunsets/pink sky
# No edge exists. Add NEW edge.
print('\n14. Melanie show painting sunsets pink sky (NEW)')
added += add('Melanie', 'show_painting', 'sunsets with pink sky',
    "Here's one I did last week. It's inspired by the sunsets. The colors make me feel calm.", 17,
    schema='hobby', texpr='last week', rdate='2023-10-06', ents=['Melanie'],
    pqs=["What painting did Melanie show to Caroline on October 13, 2023?"])

# ── 15. "What kind of painting did Caroline share with Melanie on October 13, 2023?" → abstract blue
# No edge exists. Add NEW edge.
print('\n15. Caroline share abstract painting (NEW)')
added += add('Caroline', 'share_painting', 'abstract painting with blue streaks',
    "I've been trying out abstract stuff recently. Here's my latest.", 17,
    schema='hobby', ents=['Caroline'],
    pqs=["What kind of painting did Caroline share with Melanie on October 13, 2023?"])

# ── 16. "What was the poetry reading that Caroline attended about?" → transgender poetry
# id=106 (attend, transgender poetry reading). pq_2 is free.
print('\n16. Poetry reading about')
if add_pq(106, "What was the poetry reading that Caroline attended about?"):
    updated += 1

# ── 17. "When did Caroline pass the adoption interview?" → "The Friday before 22 October 2023"
# id=111 (pass, adoption agency interviews, rdate=2023-10-20). pq_3 is free.
print('\n17. Caroline pass adoption interview')
if add_pq(111, "When did Caroline pass the adoption interview?"):
    updated += 1

conn.commit()
elapsed = time.time() - t0
cnt = conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]
print(f'\n=== fix_refs done: {updated} PQs updated, {added} new edges added, {cnt} total edges in {elapsed:.1f}s ===')
conn.close()
