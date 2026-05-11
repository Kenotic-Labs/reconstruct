import os, tempfile, hashlib, json
db = os.path.join(tempfile.gettempdir(), "test_locomo.db")
if os.path.exists(db): os.remove(db)
from sdk.client import Kenotic
k = Kenotic(user_id=1, db_path=db, embed_device="cpu")
from app.vector.embedder import embed_text
from app.db.session import get_db_context
from app.engines.predicted_queries import generate_predicted_queries
import app.engines.retrieval as _r
with open("_locomo_data.json") as f: D=json.load(f)
E=D["edges"]; ENTS=D["ents"]; FACTS=D["facts"]; T=D["tests"]
with get_db_context() as c:
 q=0
 for s,p,o,sc,sr,ic,tb in E:
  q+=1;ee=embed_text(s+" "+p.replace("_"," ")+" "+o);pe=embed_text(p.replace("_"," "));h=hashlib.sha256(sr.encode()).hexdigest()
  c.execute("INSERT INTO relationships (user_id,subject,predicate,object,source_text,source_text_hash,confidence,is_current,tombstoned_at,edge_embedding,predicate_embedding,edge_schematic_category,sequence_number) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",(1,s,p,o,sr,h,0.9,ic,tb,ee.tobytes(),pe.tobytes(),sc,q))
  rid=c.execute("SELECT last_insert_rowid()").fetchone()[0]
  st="PERSON" if s in ("Tariq","Suki","Wei","user") else "GENERIC"
  for qt,qe in generate_predicted_queries(s,p,o,st,"GENERIC"):
   c.execute("INSERT INTO predicted_queries (relationship_id,user_id,predicted_question,answer_text,question_embedding,confidence,created_at) VALUES (?,?,?,?,?,?,datetime('now'))",(rid,1,qt,o,qe.tobytes(),0.9))
  try: c.execute("INSERT INTO relationships_fts(rowid,subject,predicate,object,source_text) VALUES (?,?,?,?,?)",(rid,s,p.replace("_"," "),o,sr))
  except: pass
 for n,t in ENTS:
  c.execute("INSERT OR IGNORE INTO entities (user_id,name,entity_type,embedding,mention_count) VALUES (?,?,?,?,?)",(1,n,t,embed_text(n).tobytes(),3))
 for ky,vl in FACTS:
  c.execute("INSERT OR IGNORE INTO facts (user_id,key,value,confidence,embedding) VALUES (?,?,?,?,?)",(1,ky,vl,0.9,embed_text(ky).tobytes()))
 c.commit()
_r._singleton=None;k._retrieval=None;k._memory=None;k._temporal=None
print("Planted %d edges" % len(E))
p=0;f=0
for q,t,s in T:
 r=k.retrieve(q);tp=type(r).__name__;tx=(r.text or "").strip()
 if t=="R": ok=tp=="StructuralRefusal"
 else: ok=tp=="Answer" and s.lower() in tx.lower()
 st="PASS" if ok else "FAIL"
 if ok: p+=1
 else: f+=1
 print("[%s] %s"%(st,q))
 print("       -> %s: %r"%(tp,tx))
 if not ok:
  if t=="A": print("       EXPECTED Answer containing %r"%s)
  else: print("       EXPECTED StructuralRefusal")
 print()
print("Results: %d/%d passed"%(p,p+f))
os.remove(db)
