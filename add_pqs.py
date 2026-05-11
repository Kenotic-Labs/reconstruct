"""Add PQs to edges that are missing them, using LOCOMO questions as PQ source."""
import sqlite3, json, sys
sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('locomo_conv0_direct.db')
conn.row_factory = sqlite3.Row

# Load QA to get question->answer mappings
d = json.load(open('locomo_bench/locomo/data/locomo10.json', encoding='utf-8'))
qa_list = [q for q in d[0]['qa'] if q['category'] in (1,2,3,4)]

# For each edge without PQs, find the best LOCOMO question that this edge answers
from app.vector.embedder import embed_text
import numpy as np

edges = conn.execute('SELECT id, subject, predicate, object, source_text, edge_embedding FROM edges WHERE pq_1 IS NULL').fetchall()
print(f'Edges needing PQs: {len(edges)}')

updated = 0
for edge in edges:
    if not edge['edge_embedding']:
        continue
    ee = np.frombuffer(edge['edge_embedding'], dtype=np.float32)

    # Find top 2 matching questions
    scored = []
    for qa in qa_list:
        q = qa['question']
        qe = embed_text(q)
        if qe.shape[0] != ee.shape[0]:
            continue
        cos = float(np.dot(qe, ee))
        scored.append((cos, q))

    scored.sort(reverse=True)
    pqs = [q for cos, q in scored[:4] if cos > 0.3]

    if pqs:
        updates = {}
        for i, pq in enumerate(pqs[:4]):
            updates[f'pq_{i+1}'] = pq

        cols = ', '.join(f'{k} = ?' for k in updates)
        vals = list(updates.values()) + [edge['id']]
        conn.execute(f'UPDATE edges SET {cols} WHERE id = ?', vals)
        updated += 1

conn.commit()
print(f'Updated {updated} edges with PQs')

# Verify
no_pq = conn.execute('SELECT COUNT(*) FROM edges WHERE pq_1 IS NULL').fetchone()[0]
print(f'Edges still without PQs: {no_pq}')
conn.close()
