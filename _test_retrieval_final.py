import os, tempfile, hashlib
db = os.path.join(tempfile.gettempdir(), "test_final.db")
if os.path.exists(db): os.remove(db)
from sdk.client import Kenotic
k = Kenotic(user_id=1, db_path=db, embed_device="cpu")
from app.vector.embedder import embed_text
from app.db.session import get_db_context
from app.engines.predicted_queries import generate_predicted_queries
import app.engines.retrieval as _r
E=[
("user","work_at","Shopify","career","I work at Shopify",1,None),
("user","live_in","Toronto","home","I live in Toronto",1,None),
("user","drive","Honda Civic","possession","I drive a Honda Civic",1,None),
("user","study_at","University of Waterloo","education","I studied at University of Waterloo",1,None),
("user","be_engaged_to","Anika","family","I am engaged to Anika",1,None),
("user","play","basketball","hobby","I play basketball on Thursdays",1,None),
("user","be_lactose_intolerant","dairy","health","I am lactose intolerant",1,None),
("Anika","work_at","TD Bank","career","Anika works at TD Bank",1,None),
("Anika","live_in","Toronto","home","Anika lives in Toronto",1,None),
("Anika","study_at","McGill","education","Anika studied at McGill",1,None),
("Anika","play","tennis","hobby","Anika plays tennis",1,None),
("Anika","cook","Thai food","hobby","Anika cooks Thai food",1,None),
("Ravi","work_at","Google","career","Ravi works at Google",1,None),
("Ravi","live_in","Mountain View","home","Ravi lives in Mountain View",1,None),
("Ravi","be_brother_of","user","family","Ravi is my brother",1,None),
("Ravi","drive","BMW X5","possession","Ravi drives a BMW X5",1,None),
("Ravi","play","chess","hobby","Ravi plays chess competitively",1,None),
("Noor","work_at","CBC","career","Noor works at CBC",1,None),
("Noor","live_in","Ottawa","home","Noor lives in Ottawa",1,None),
("Noor","be_best_friend_of","user","family","Noor is my best friend",1,None),
("Noor","study_at","Carleton","education","Noor studied at Carleton",1,None),
("Noor","run","half marathons","hobby","Noor runs half marathons",1,None),
("user","work_at","RBC","career","I used to work at RBC",0,'2025-03-01'),
("user","live_in","Vancouver","home","I used to live in Vancouver",0,'2024-08-01'),
("Ravi","live_in","Seattle","home","Ravi used to live in Seattle",0,'2025-01-01'),
]

with get_db_context() as c:
 q=0
 for s,p,o,sc,sr,ic,tb in E:
  q+=1;ee=embed_text(s+" "+p.replace("_"," ")+" "+o);pe=embed_text(p.replace("_"," "));h=hashlib.sha256(sr.encode()).hexdigest()
  c.execute("INSERT INTO relationships (user_id,subject,predicate,object,source_text,source_text_hash,confidence,is_current,tombstoned_at,edge_embedding,predicate_embedding,edge_schematic_category,sequence_number) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",(1,s,p,o,sr,h,0.9,ic,tb,ee.tobytes(),pe.tobytes(),sc,q))
  rid=c.execute("SELECT last_insert_rowid()").fetchone()[0]
  st="PERSON" if s in ("Anika","Ravi","Noor","user") else "GENERIC"
  for qt,qe in generate_predicted_queries(s,p,o,st,"GENERIC"):
   c.execute("INSERT INTO predicted_queries (relationship_id,user_id,predicted_question,answer_text,question_embedding,confidence,created_at) VALUES (?,?,?,?,?,?,datetime('now'))",(rid,1,qt,o,qe.tobytes(),0.9))
  try: c.execute("INSERT INTO relationships_fts(rowid,subject,predicate,object,source_text) VALUES (?,?,?,?,?)",(rid,s,p.replace("_"," "),o,sr))
  except: pass
 for n,t in [("user","PERSON"),("Anika","PERSON"),("Ravi","PERSON"),("Noor","PERSON"),("Shopify","ORG"),("TD Bank","ORG"),("Google","ORG"),("CBC","ORG"),("RBC","ORG"),("University of Waterloo","ORG"),("McGill","ORG"),("Carleton","ORG"),("Toronto","LOCATION"),("Mountain View","LOCATION"),("Ottawa","LOCATION"),("Vancouver","LOCATION"),("Seattle","LOCATION")]:
  c.execute("INSERT OR IGNORE INTO entities (user_id,name,entity_type,embedding,mention_count) VALUES (?,?,?,?,?)",(1,n,t,embed_text(n).tobytes(),3))
 for ky,vl in [("career::work_at::user","Shopify"),("home::live_in::user","Toronto"),("career::work_at::Anika","TD Bank"),("home::live_in::Anika","Toronto"),("career::work_at::Ravi","Google"),("home::live_in::Ravi","Mountain View"),("career::work_at::Noor","CBC"),("home::live_in::Noor","Ottawa")]:
  c.execute("INSERT OR IGNORE INTO facts (user_id,key,value,confidence,embedding) VALUES (?,?,?,?,?)",(1,ky,vl,0.9,embed_text(ky).tobytes()))
 c.commit()
_r._singleton=None;k._retrieval=None;k._memory=None;k._temporal=None
print("Planted %d edges" % len(E))

T=[
("Where do I work?","A",'Shopify'),
("Where do I live?","A",'Toronto'),
("What do I drive?","A",'Honda Civic'),
("Where did I study?","A",'Waterloo'),
("What do I play?","A",'basketball'),
("Where does Anika work?","A",'TD Bank'),
("Where does Anika live?","A",'Toronto'),
("Where did Anika study?","A",'McGill'),
("What does Anika play?","A",'tennis'),
("What does Anika cook?","A",'Thai'),
("Where does Ravi work?","A",'Google'),
("Where does Ravi live?","A",'Mountain View'),
("What does Ravi drive?","A",'BMW'),
("What does Ravi play?","A",'chess'),
("Who is Ravi?","A",'brother'),
("Where does Noor work?","A",'CBC'),
("Where does Noor live?","A",'Ottawa'),
("Where did Noor study?","A",'Carleton'),
("What does Noor run?","A",'marathon'),
("Who is Noor?","A",'best friend'),
("What does Ravi cook?","R",None),
("What does Noor drive?","R",None),
("What does Anika run?","R",None),
("What is Ravi allergic to?","R",None),
("Where did Ravi study?","R",None),
("What does Noor play?","R",None),
("Where do I work?","A",'Shopify'),
("Where do I live?","A",'Toronto'),
("Where does Ravi live?","A",'Mountain View'),
("Who is Marcus?","R",None),
("Where does Justin Trudeau live?","R",None),
("What is Anika afraid of?","R",None),
("Does Ravi have kids?","R",None),
("What is my favorite movie?","R",None),
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
