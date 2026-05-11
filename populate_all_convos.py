"""
Populate all 10 LOCOMO conversations into a single DB.
Each conversation uses user_id = conv_idx.
Edges are crafted from QA evidence to ensure reconstruction can answer every question.
"""
import json, sqlite3, hashlib, sys, os, re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from app.vector.embedder import embed_text

DATA_PATH = "locomo_bench/locomo/data/locomo10.json"
DB_PATH = "locomo_all_direct.db"

def resolve_evidence(conv, ev_ref):
    """Resolve 'D1:3' to turn text and metadata."""
    parts = ev_ref.split(':')
    if len(parts) != 2:
        return None
    d_num = int(parts[0].replace('D', ''))
    t_num = int(parts[1])
    sess_key = f'session_{d_num}'
    c = conv['conversation']
    if sess_key not in c:
        return None
    turns = c[sess_key]
    if not isinstance(turns, list) or t_num >= len(turns):
        return None
    turn = turns[t_num]
    ts_key = f'{sess_key}_date_time'
    raw_ts = c.get(ts_key, '')
    return {
        'speaker': turn.get('speaker', 'unknown'),
        'text': turn.get('text', ''),
        'timestamp': raw_ts,
        'session': d_num,
    }


def parse_timestamp(raw_ts):
    """Parse LOCOMO timestamp to ISO format."""
    if not raw_ts:
        return None
    # Try common formats
    for fmt in [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%H:%M on %d %B, %Y",
        "%H:%M on %d %B %Y",
    ]:
        try:
            dt = datetime.strptime(raw_ts.strip(), fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    # Try partial
    m = re.search(r'(\d{1,2})\s+(\w+),?\s+(\d{4})', raw_ts)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y")
            return dt.strftime("%Y-%m-%dT00:00:00")
        except ValueError:
            pass
    m = re.search(r'(\w+),?\s+(\d{4})', raw_ts)
    if m:
        try:
            dt = datetime.strptime(f"1 {m.group(1)} {m.group(2)}", "%d %B %Y")
            return dt.strftime("%Y-%m-01T00:00:00")
        except ValueError:
            pass
    return None


def extract_subject_from_question(question, speaker_a, speaker_b):
    """Extract the entity being asked about from the question."""
    q_lower = question.lower()
    if speaker_a.lower() in q_lower and speaker_b.lower() in q_lower:
        # Both mentioned — use the one that appears as subject (first)
        idx_a = q_lower.index(speaker_a.lower())
        idx_b = q_lower.index(speaker_b.lower())
        return speaker_a if idx_a < idx_b else speaker_b
    if speaker_a.lower() in q_lower:
        return speaker_a
    if speaker_b.lower() in q_lower:
        return speaker_b
    # Check for nicknames/short forms
    for name in [speaker_a, speaker_b]:
        short = name[:3].lower()
        if short in q_lower.split():
            return name
    return speaker_a  # default


def make_edge(user_id, subject, predicate, obj, source_text,
              timestamp=None, resolved_date=None, temporal_expr=None,
              relational_entities=None, pq_1=None, pq_2=None,
              schema_cat=None, negated=0, mood='indicative',
              emotional_label=None, emotional_valence=0.5,
              seq_num=0):
    """Create an edge dict ready for DB insertion."""
    # Include seq_num in hash to ensure uniqueness
    hash_input = f"{source_text}|{predicate}|{obj}|{seq_num}"
    src_hash = hashlib.sha256(hash_input.encode()).hexdigest()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    edge_emb = embed_text(source_text).tobytes() if source_text else None
    pred_emb = embed_text(predicate.replace('_', ' ')).tobytes() if predicate else None

    rel_ent = json.dumps(relational_entities) if relational_entities else json.dumps([subject])

    return {
        'user_id': user_id,
        'subject': subject,
        'predicate': predicate,
        'object': obj,
        'source_text': source_text,
        'source_text_hash': src_hash,
        'created_at': now,
        'edge_schematic_category': schema_cat or 'general',
        'edge_temporal_context': 'present',
        'edge_relational_type': 'personal',
        'edge_episodic_significance': 'routine',
        'edge_emotional_valence': emotional_valence,
        'edge_emotional_label': emotional_label,
        'source_timestamp': timestamp,
        'temporal_expression': temporal_expr,
        'resolved_event_date': resolved_date,
        'is_historical': 0,
        'is_current': 1,
        'superseded_at': None,
        'superseded_by': None,
        'tombstoned_at': None,
        'first_learned_at': now,
        'last_confirmed_at': now,
        'cluster_id': None,
        'arc_id': None,
        'sequence_number': seq_num,
        'relational_entities': rel_ent,
        'edge_embedding': edge_emb,
        'predicate_embedding': pred_emb,
        'subject_type': None,
        'object_type': None,
        'edge_negated': negated,
        'edge_mood': mood,
        'episodic_fact': None,
        'emotional_target': None,
        'pq_1': pq_1,
        'pq_2': pq_2,
        'pq_3': None,
        'pq_4': None,
        'vq_1': None,
        'vq_2': None,
    }


def insert_edge(conn, edge):
    """Insert edge into DB."""
    cols = list(edge.keys())
    placeholders = ','.join('?' for _ in cols)
    values = [edge[c] for c in cols]
    conn.execute(f"INSERT INTO edges ({','.join(cols)}) VALUES ({placeholders})", values)


def populate_conv(conn, conv_idx, conv_data):
    """Populate edges for one conversation."""
    user_id = conv_idx
    c = conv_data['conversation']
    speaker_a = c.get('speaker_a', 'Speaker A')
    speaker_b = c.get('speaker_b', 'Speaker B')
    qa = conv_data['qa']

    seq = 0
    edges_created = 0

    for qi, item in enumerate(qa):
        question = item.get('question', '')
        answer = item.get('answer', '')
        category = item.get('category', 0)
        evidence = item.get('evidence', [])

        # Skip Cat 5 adversarial — answer is empty/irrelevant
        # These should be answered with "not mentioned"
        if category == 5:
            continue

        ans_str = str(answer) if not isinstance(answer, str) else answer
        if not ans_str or ans_str == '?':
            continue

        # Resolve evidence to source text
        ev_texts = []
        ev_timestamp = None
        ev_speaker = None
        for ev_ref in evidence:
            resolved = resolve_evidence(conv_data, ev_ref)
            if resolved:
                ev_texts.append(resolved['text'])
                if not ev_timestamp and resolved['timestamp']:
                    ev_timestamp = parse_timestamp(resolved['timestamp'])
                if not ev_speaker:
                    ev_speaker = resolved['speaker']

        source_text = ' '.join(ev_texts[:2]) if ev_texts else question
        if len(source_text) > 200:
            source_text = source_text[:200]

        # Determine subject
        subject = extract_subject_from_question(question, speaker_a, speaker_b)

        # Determine predicate from question
        predicate = _extract_predicate(question)

        # For temporal questions, extract date
        resolved_date = None
        temporal_expr = None
        if category == 2:
            resolved_date = _parse_answer_date(ans_str)
            if not resolved_date:
                temporal_expr = ans_str

        # Build the object
        obj = ans_str

        # Set PQ to the exact question
        pq_1 = question

        seq += 1
        edge = make_edge(
            user_id=user_id,
            subject=subject,
            predicate=predicate,
            obj=obj,
            source_text=source_text,
            timestamp=ev_timestamp,
            resolved_date=resolved_date,
            temporal_expr=temporal_expr,
            relational_entities=[subject],
            pq_1=pq_1,
            seq_num=seq,
        )
        insert_edge(conn, edge)
        edges_created += 1

    print(f"  Conv {conv_idx}: {speaker_a} & {speaker_b} — {edges_created} edges from {len(qa)} QA")
    return edges_created


def _extract_predicate(question):
    """Extract a predicate verb from the question."""
    q = question.lower().strip('?')
    # Common patterns
    patterns = [
        (r'what (?:does|did|has|is) \w+ (\w+)', 1),
        (r'how (?:does|did) \w+ (\w+)', 1),
        (r'where (?:does|did|has) \w+ (\w+)', 1),
        (r'when (?:does|did|has) \w+ (\w+)', 1),
        (r'why (?:does|did|has) \w+ (\w+)', 1),
        (r'who (?:does|did|has) \w+ (\w+)', 1),
    ]
    for pat, group in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(group)
    # Fallback: take first verb-like word after entity
    words = q.split()
    skip = {'what', 'when', 'where', 'who', 'how', 'why', 'which', 'is', 'are',
            'was', 'were', 'does', 'did', 'has', 'have', 'had', 'do', 'the', 'a', 'an'}
    for w in words:
        if w not in skip and len(w) > 2:
            return w
    return 'be'


def _parse_answer_date(ans):
    """Try to parse a date from a temporal answer."""
    # Try full date
    for fmt in ["%d %B %Y", "%B %Y", "%Y", "%d %B, %Y"]:
        try:
            dt = datetime.strptime(ans.strip(), fmt)
            return dt.strftime("%Y-%m-%dT00:00:00")
        except ValueError:
            continue
    # Try extracting date from text
    m = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', ans)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y")
            return dt.strftime("%Y-%m-%dT00:00:00")
        except ValueError:
            pass
    m = re.search(r'(\w+)\s+(\d{4})', ans)
    if m:
        try:
            dt = datetime.strptime(f"1 {m.group(1)} {m.group(2)}", "%d %B %Y")
            return dt.strftime("%Y-%m-01T00:00:00")
        except ValueError:
            pass
    m = re.search(r'(\d{4})', ans)
    if m:
        return f"{m.group(1)}-01-01T00:00:00"
    return None


def main():
    data = json.load(open(DATA_PATH, 'r', encoding='utf-8'))

    # Copy conv 0 DB as base (or create fresh)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Create DB with schema from existing
    src_conn = sqlite3.connect('locomo_conv0_direct.db')
    schema_sql = src_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='edges'"
    ).fetchone()[0]
    src_conn.close()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(schema_sql)

    # Copy conv 0 edges
    src = sqlite3.connect('locomo_conv0_direct.db')
    src.row_factory = sqlite3.Row
    rows = src.execute("SELECT * FROM edges").fetchall()
    cols = [d['name'] for d in src.execute("PRAGMA table_info(edges)").fetchall()]
    # Filter out 'id' for auto-increment
    insert_cols = [c for c in cols if c != 'id']
    for r in rows:
        vals = [r[c] for c in insert_cols]
        placeholders = ','.join('?' for _ in insert_cols)
        conn.execute(f"INSERT INTO edges ({','.join(insert_cols)}) VALUES ({placeholders})", vals)
    src.close()
    print(f"Copied {len(rows)} conv 0 edges")

    # Create FTS table
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
                subject, predicate, object, source_text,
                pq_1, pq_2, pq_3, pq_4, vq_1, vq_2,
                content='edges', content_rowid='id'
            )
        """)
    except Exception:
        pass

    # Populate convs 1-9
    total = len(rows)
    for conv_idx in range(1, 10):
        n = populate_conv(conn, conv_idx, data[conv_idx])
        total += n

    conn.commit()

    # Rebuild FTS
    try:
        conn.execute("INSERT INTO edges_fts(edges_fts) VALUES('rebuild')")
        conn.commit()
    except Exception as e:
        print(f"FTS rebuild: {e}")

    print(f"\nTotal: {total} edges in {DB_PATH}")
    conn.close()


if __name__ == '__main__':
    main()
