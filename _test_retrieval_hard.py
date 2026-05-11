import os, tempfile, hashlib
db = os.path.join(tempfile.gettempdir(), "test_hard.db")
if os.path.exists(db): os.remove(db)
from sdk.client import Kenotic
k = Kenotic(user_id=1, db_path=db, embed_device="cpu")
from app.vector.embedder import embed_text
from app.db.session import get_db_context
from app.engines.predicted_queries import generate_predicted_queries
import app.engines.retrieval as _r
E=[
("Priya","work_at","Netflix","career","Priya works at Netflix",1,None),
("Priya","live_in","San Francisco","home","Priya lives in San Francisco",1,None),
("Priya","study_at","Stanford","education","Priya studied at Stanford",1,None),
("Priya","play","violin","hobby","Priya plays violin",1,None),
("Priya","be_allergic_to","peanuts","health","Priya is allergic to peanuts",1,None),
("Omar","work_at","Tesla","career","Omar works at Tesla",1,None),
("Omar","live_in","Denver","home","Omar lives in Denver",1,None),
("Omar","study_at","CU Boulder","education","Omar studied at CU Boulder",1,None),
("Omar","play","guitar","hobby","Omar plays guitar",1,None),
("Omar","be_married_to","Priya","family","Omar is married to Priya",1,None),
("user","work_at","Stripe","career","I work at Stripe",1,None),
("user","live_in","Brooklyn","home","I live in Brooklyn",1,None),
("user","own","a cat named Miso","pet","I own a cat named Miso",1,None),
("user","study_at","NYU","education","I studied at NYU",1,None),
("user","be_afraid_of","flying","emotion","I am afraid of flying",1,None),
("user","teach","yoga on weekends","hobby","I teach yoga on weekends",1,None),
("Lena","live_in","Chicago","home","Lena lives in Chicago",1,None),
("Lena","work_at","Northwestern Hospital","career","Lena works at Northwestern Hospital",1,None),
("Lena","be_mother_of","user","family","Lena is my mother",1,None),
("user","live_in","Portland","home","I used to live in Portland",0,'2025-06-01'),
("Priya","work_at","Google","career","Priya used to work at Google",0,'2025-09-01'),
("Omar","live_in","Austin","home","Omar used to live in Austin",0,'2025-11-01'),
]

with get_db_context() as c:
 q=0
 for s,p,o,sc,sr,ic,tb in E:
  q+=1;ee=embed_text(s+" "+p.replace("_"," ")+" "+o);pe=embed_text(p.replace("_"," "));h=hashlib.sha256(sr.encode()).hexdigest()
  c.execute("INSERT INTO relationships (user_id,subject,predicate,object,source_text,source_text_hash,confidence,is_current,tombstoned_at,edge_embedding,predicate_embedding,edge_schematic_category,sequence_number) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",(1,s,p,o,sr,h,0.9,ic,tb,ee.tobytes(),pe.tobytes(),sc,q))
  rid=c.execute("SELECT last_insert_rowid()").fetchone()[0]
  st="PERSON" if s in ("Priya","Omar","Lena","user") else "GENERIC"
  for qt,qe in generate_predicted_queries(s,p,o,st,"GENERIC"):
   c.execute("INSERT INTO predicted_queries (relationship_id,user_id,predicted_question,answer_text,question_embedding,confidence,created_at) VALUES (?,?,?,?,?,?,datetime('now'))",(rid,1,qt,o,qe.tobytes(),0.9))
  try: c.execute("INSERT INTO relationships_fts(rowid,subject,predicate,object,source_text) VALUES (?,?,?,?,?)",(rid,s,p.replace("_"," "),o,sr))
  except: pass
 for n,t in [("user","PERSON"),("Priya","PERSON"),("Omar","PERSON"),("Lena","PERSON"),("Netflix","ORG"),("Tesla","ORG"),("Stripe","ORG"),("Google","ORG"),("Northwestern Hospital","ORG"),("Stanford","ORG"),("CU Boulder","ORG"),("NYU","ORG"),("San Francisco","LOCATION"),("Denver","LOCATION"),("Brooklyn","LOCATION"),("Chicago","LOCATION"),("Portland","LOCATION"),("Austin","LOCATION")]:
  c.execute("INSERT OR IGNORE INTO entities (user_id,name,entity_type,embedding,mention_count) VALUES (?,?,?,?,?)",(1,n,t,embed_text(n).tobytes(),3))
 for ky,vl in [("career::work_at::user","Stripe"),("home::live_in::user","Brooklyn"),("career::work_at::Priya","Netflix"),("home::live_in::Priya","San Francisco"),("career::work_at::Omar","Tesla"),("home::live_in::Omar","Denver"),("home::live_in::Lena","Chicago"),("career::work_at::Lena","Northwestern Hospital")]:
  c.execute("INSERT OR IGNORE INTO facts (user_id,key,value,confidence,embedding) VALUES (?,?,?,?,?)",(1,ky,vl,0.9,embed_text(ky).tobytes()))
 c.commit()
_r._singleton=None;k._retrieval=None;k._memory=None;k._temporal=None
print("Planted %d edges" % len(E))

T=[
("Where does Priya work?","A",'Netflix'),
("Where does Omar work?","A",'Tesla'),
("Where do I work?","A",'Stripe'),
("Where does Priya live?","A",'San Francisco'),
("Where does Omar live?","A",'Denver'),
("Where do I live?","A",'Brooklyn'),
("Where does Lena live?","A",'Chicago'),
("Where does Lena work?","A",'Northwestern'),
("What does Priya play?","A",'violin'),
("What does Omar play?","A",'guitar'),
("What do I own?","A",'Miso'),
("What do I teach?","A",'yoga'),
("Where did Priya study?","A",'Stanford'),
("Where did Omar study?","A",'CU Boulder'),
("Where did I study?","A",'NYU'),
("Who is Omar married to?","A",'Priya'),
("Who is Lena?","A",'mother'),
("What is Omar allergic to?","R",None),
("What does Priya own?","R",None),
("Where did Lena study?","R",None),
("What does Omar teach?","R",None),
("What is Priya afraid of?","R",None),
("Where do I live?","A",'Brooklyn'),
("Where does Priya work?","A",'Netflix'),
("Where does Omar live?","A",'Denver'),
("What is Priya's favorite color?","R",None),
("Does Omar have siblings?","R",None),
("What kind of car does Lena drive?","R",None),
("Who is Rachel?","R",None),
("Where does Elon Musk work?","R",None),
]

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
