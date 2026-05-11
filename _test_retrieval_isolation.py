"""Retrieval engine isolation test.

Tests the retrieval engine in isolation with perfect edges planted
directly into the database. Grammar, extraction, memory engine are
all bypassed -- only the retrieval engine is under test.
"""
from sdk.client import Kenotic
import tempfile, os, hashlib

db = os.path.join(tempfile.gettempdir(), "test_retrieval_isolation.db")
if os.path.exists(db):
    os.remove(db)

k = Kenotic(user_id=1, db_path=db, embed_device="cpu")

from app.vector.embedder import embed_text
from app.db.session import get_db_context
from app.engines.predicted_queries import generate_predicted_queries

edges = [
    ("user",   "work_at",          "Google",             "career",     "I work at Google"),
    ("user",   "live_in",          "Ann Arbor",          "home",       "I live in Ann Arbor"),
    ("user",   "love",             "woodworking",        "hobby",      "I love woodworking"),
    ("user",   "study_at",         "MIT",                "education",  "I studied at MIT"),
    ("user",   "be_nervous_about", "the interview",      "emotion",    "I am nervous about the interview"),
    ("user",   "drive",            "Tesla Model 3",      "possession", "I drive a Tesla Model 3"),
    ("user",   "eat",              "sushi",              "food",       "I eat sushi every Friday"),
    ("user",   "run",              "marathons",          "hobby",      "I run marathons"),
    ("Maya",   "live_in",          "Austin",             "home",       "Maya lives in Austin"),
    ("Maya",   "work_at",          "Spotify",            "career",     "Maya works at Spotify"),
    ("Maya",   "be_sister_of",     "Sam",                "family",     "Maya is my sister"),
    ("Maya",   "marry",            "Jake",               "family",     "Maya married Jake"),
    ("Maya",   "adopt",            "a golden retriever", "pet",        "Maya adopted a golden retriever"),
    ("Dad",    "retire_from",      "Boeing",             "career",     "My dad retired from Boeing"),
    ("Dad",    "live_in",          "Seattle",            "home",       "My dad lives in Seattle"),
    ("Dad",    "love",             "fishing",            "hobby",      "My dad loves fishing"),
    ("Dad",    "drive",            "Ford F-150",         "possession", "My dad drives a Ford F-150"),
    ("Jake",   "work_at",          "Amazon",             "career",     "Jake works at Amazon"),
    ("Jake",   "be_married_to",    "Maya",               "family",     "Jake is married to Maya"),
    ("user",   "work_at",          "Microsoft",          "career",     "I used to work at Microsoft"),
]

seq = 0
with get_db_context() as conn:
    for subj, pred, obj, schema, source in edges:
        seq += 1
        is_current = 1
        tombstoned = None
        if subj == "user" and pred == "work_at" and obj == "Microsoft":
            is_current = 0
            tombstoned = "2026-01-15T00:00:00"

        edge_emb = embed_text(subj + " " + pred.replace("_", " ") + " " + obj)
        pred_emb = embed_text(pred.replace("_", " "))
        src_hash = hashlib.sha256(source.encode()).hexdigest()
        conn.execute(
            "INSERT INTO relationships"
            " (user_id, subject, predicate, object, source_text, source_text_hash,"
            "  confidence, is_current, tombstoned_at, edge_embedding,"
            "  predicate_embedding, edge_schematic_category, sequence_number)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (1, subj, pred, obj, source, src_hash, 0.9, is_current, tombstoned,
             edge_emb.tobytes(), pred_emb.tobytes(), schema, seq))

        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        stype = "PERSON" if subj in ("Maya", "Dad", "Jake", "user") else "GENERIC"
        pqs = generate_predicted_queries(subj, pred, obj, stype, "GENERIC")
        for qtxt, qemb in pqs:
            conn.execute(
                "INSERT INTO predicted_queries"
                " (relationship_id, user_id, predicted_question,"
                "  answer_text, question_embedding, confidence, created_at)"
                " VALUES (?,?,?,?,?,?,datetime('now'))",
                (rid, 1, qtxt, obj, qemb.tobytes(), 0.9))

        try:
            conn.execute(
                "INSERT INTO relationships_fts"
                "(rowid, subject, predicate, object, source_text)"
                " VALUES (?,?,?,?,?)",
                (rid, subj, pred.replace("_", " "), obj, source))
        except Exception:
            pass

    for name, etype in [
        ("user","PERSON"),("Maya","PERSON"),("Dad","PERSON"),("Jake","PERSON"),
        ("Google","ORG"),("Spotify","ORG"),("Amazon","ORG"),("Apple","ORG"),
        ("Boeing","ORG"),("MIT","ORG"),("Ann Arbor","LOCATION"),
        ("Austin","LOCATION"),("Seattle","LOCATION")]:
        emb = embed_text(name)
        conn.execute(
            "INSERT OR IGNORE INTO entities"
            " (user_id, name, entity_type, embedding, mention_count)"
            " VALUES (?,?,?,?,?)", (1, name, etype, emb.tobytes(), 3))

    for key, val in [
        ("career::work_at::user", "Google"),
        ("home::live_in::user", "Ann Arbor"),
        ("career::work_at::Maya", "Spotify"),
        ("home::live_in::Maya", "Austin"),
        ("career::work_at::Jake", "Amazon"),
        ("home::live_in::Dad", "Seattle")]:
        emb = embed_text(key)
        conn.execute(
            "INSERT OR IGNORE INTO facts"
            " (user_id, key, value, confidence, embedding)"
            " VALUES (?,?,?,?,?)", (1, key, val, 0.9, emb.tobytes()))

    conn.commit()

print("Planted %d edges\n" % len(edges))

# DEBUG: verify DB state
from app.db.session import get_db_context as _gc
with _gc() as _c:
    _n = _c.execute("SELECT COUNT(*) FROM relationships WHERE user_id=1 AND COALESCE(is_current,1)=1 AND tombstoned_at IS NULL").fetchone()[0]
    print("Active edges: %d" % _n)
    _r = _c.execute("SELECT id,subject,predicate,object FROM relationships WHERE subject='user' AND predicate='work_at' AND COALESCE(is_current,1)=1 AND tombstoned_at IS NULL").fetchall()
    for _row in _r:
        print("  user/work_at: id=%d O=%s" % (_row["id"], _row["object"]))

# DEBUG: test entity resolution
from app.engines.entity_resolver import resolve_query_entities as _rqe
_ents = _rqe(1, "Where does Sam work?")
print("Sam entities:", [(e["name"],round(e["score"],3)) for e in _ents[:3]])

# DEBUG: test retrieve directly
_ret = k._engines()[2]
from app.engines.retrieval import _extract_query_verb as _eqv
_cands, _qe, _res = _ret._run_moat_pipeline(1, "Where does Sam work?")
_qent = _res[0]["name"] if _res else None
_cf = _ret._load_facts_for_entity(1, _qent or "", "Where does Sam work?")
print("Pipeline: %d candidates, entity=%s, fact=%s" % (len(_cands), _qent, _cf))
for _c in _cands[:3]:
    _e = _c.edge
    print("  S=%s P=%s O=%s exit=%.3f" % (_e["subject"],_e["predicate"],_e["object"],_c.exit_cosine))

tests = [
    ("Where does Sam work?",            "Answer",  "Google"),
    ("Where do I work?",                "Answer",  "Google"),
    ("Where does Maya work?",           "Answer",  "Spotify"),
    ("Where does Jake work?",           "Answer",  "Amazon"),
    ("Where does Maya live?",           "Answer",  "Austin"),
    ("Where do I live?",                "Answer",  "Ann Arbor"),
    ("Where does Dad live?",            "Answer",  "Seattle"),
    ("What do I love?",                 "Answer",  "woodworking"),
    ("What does Dad love?",             "Answer",  "fishing"),
    ("What do I drive?",                "Answer",  "Tesla"),
    ("What does Dad drive?",            "Answer",  "Ford"),
    ("Who did Maya marry?",             "Answer",  "Jake"),
    ("What do I eat?",                  "Answer",  "sushi"),
    ("What do I run?",                  "Answer",  "marathon"),
    ("What did Maya adopt?",            "Answer",  "golden retriever"),
    ("Where does Jake live?",           "Refusal", None),
    ("Where does Dad work?",            "Refusal", None),
    ("Where does Sam work?",            "Answer",  "Google"),
    ("What is the capital of France?",  "Refusal", None),
    ("Who is Obama?",                   "Refusal", None),
    ("What color is my car?",           "Refusal", None),
]

passed = 0
failed = 0
for q, exp_type, exp_sub in tests:
    r = k.retrieve(q)
    tp = type(r).__name__
    txt = (r.text or "").strip()
    if exp_type == "Refusal":
        ok = tp == "StructuralRefusal"
    else:
        ok = tp == "Answer" and exp_sub.lower() in txt.lower()
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print("[%s] %s" % (status, q))
    print("       -> %s: %r" % (tp, txt))
    if not ok:
        if exp_type == "Answer":
            print("       EXPECTED Answer containing %r" % exp_sub)
        else:
            print("       EXPECTED StructuralRefusal")
    print()

print("Results: %d/%d passed" % (passed, passed + failed))
os.remove(db)
