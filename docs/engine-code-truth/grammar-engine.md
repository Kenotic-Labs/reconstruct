# grammar-engine

Source: [app/engines/grammar_engine.py](/D:/Nura/Code/nura_living_memory_code/app/engines/grammar_engine.py)

```text
0001: """
0002: Grammar Engine -- Trace-primary extraction.
0003: Rebuilt from grammar-engine-spec.md + locomo-pattern-analysis.md + 7,482 grammar rules.
0004: 
0005: Architecture:
0006:     WRITE: text -> process(text, speaker) -> GrammarResult(trace_decompositions, triples)
0007:     READ:  query -> classify_query(query_text) -> QueryDecomposition
0008: 
0009: Design principles (from spec + pattern analysis):
0010:     1. Extract the SHORTEST noun phrase that IS the answer.  61% of answers are NPs.
0011:     2. Every fact carries the speaker name in relational_entities (Cat 5 = 22.5%).
0012:     3. Temporal expressions pass through as-is.  Human readable.  No ISO.
0013:     4. Non-statements extract imposed facts.  Tag with mood so retrieval can filter.
0014:     5. NO regex.  NO word lists.  spaCy POS/dep/morph/NER + WordNet hypernym closure.
0015:     6. Scales to unseen conversations -- built on grammar rules, not patterns.
0016: 
0017: Public names (imported by retrieval, memory, tests, SDK):
0018:     process, classify_query, classify_verb_class, _get_nlp, _get_root
0019:     _reclassify_location_by_object, _extract_schematic
0020:     TraceDecomposition, GrammarResult, Triple, UtteranceClassification
0021:     TenseAspect, QueryDecomposition, VerbClass
0022:     _VERB_CLASS_TO_SCHEMA, _VERB_CLASS_TO_RELTYPE
0023:     _build_trace_decomposition
0024:     detect_mood, detect_negation, detect_tense_aspect, detect_voice, resolve_pronouns
0025:     classify_utterance, extract_typed_triple
0026: """
0027: 
0028: from __future__ import annotations
0029: 
0030: import functools
0031: import logging
0032: from dataclasses import dataclass, field
0033: from enum import Enum
0034: from typing import Any, Dict, List, Optional, Tuple
0035: 
0036: logger = logging.getLogger(__name__)
0037: 
0038: # ---------------------------------------------------------------------------
0039: # Lazy-loaded singletons
0040: # ---------------------------------------------------------------------------
0041: 
0042: _nlp = None
0043: _nlp_fragment = None
0044: _wordnet_loaded = False
0045: 
0046: 
0047: def _get_nlp():
0048:     """Lazy-load the full spaCy pipeline (tagger + parser + lemmatizer + NER + senter).
0049: 
0050:     Model: en_core_web_md (required â€” will raise OSError if not installed)
0051:     Accuracy: UAS 91.7%, LAS 89.9%, NER F1 84.5%, POS 97.2%
0052:     Trained on: OntoNotes 5 web text (blogs, news, comments)
0053:     Known limitation: accuracy degrades on fragments lacking sentence context.
0054: 
0055:     Use ONLY at true entry points (process(), classify_query()) and for
0056:     functions that may receive raw text strings (detect_mood, etc.).
0057:     For re-parsing extracted fragments, use _get_nlp_fragment() instead.
0058:     """
0059:     global _nlp
0060:     if _nlp is None:
0061:         import spacy
0062:         _nlp = spacy.load("en_core_web_md")
0063:         _nlp.max_length = 100_000
0064:     return _nlp
0065: 
0066: 
0067: def _get_nlp_fragment():
0068:     """Lightweight pipeline for re-parsing fragments. Only runs
0069:     tagger, parser, and lemmatizer â€” no NER or sentence segmentation.
0070:     Fragments don't have sentence boundaries and NER needs full context."""
0071:     global _nlp_fragment
0072:     if _nlp_fragment is None:
0073:         import spacy
0074:         _nlp_fragment = spacy.load("en_core_web_md", disable=["ner", "senter"])
0075:     return _nlp_fragment
0076: 
0077: 
0078: def _ensure_wordnet():
0079:     """Ensure NLTK WordNet data is available."""
0080:     global _wordnet_loaded
0081:     if not _wordnet_loaded:
0082:         import nltk  # type: ignore
0083:         try:
0084:             from nltk.corpus import wordnet as wn  # type: ignore
0085:             wn.synsets("test")
0086:         except LookupError:
0087:             nltk.download("wordnet", quiet=True)
0088:             nltk.download("omw-1.4", quiet=True)
0089:         _wordnet_loaded = True
0090: 
0091: 
0092: # ---------------------------------------------------------------------------
0093: # Dataclasses  (Spec Part 1)
0094: # ---------------------------------------------------------------------------
0095: 
0096: class CoarseBin(str, Enum):
0097:     """Five coarse utterance bins derived from spaCy structural features."""
0098:     QUESTION = "QUESTION"
0099:     STATEMENT = "STATEMENT"
0100:     COMMAND = "COMMAND"
0101:     BACKCHANNEL = "BACKCHANNEL"
0102:     EMOTION = "EMOTION"
0103: 
0104: 
0105: @dataclass(frozen=True)
0106: class UtteranceClassification:
0107:     """Result of classifying a turn into utterance type."""
0108:     utterance_type_id: int
0109:     category: str
0110:     subcategory: str
0111:     is_question: bool
0112:     is_command: bool
0113:     is_backchannel: bool
0114:     is_emotion: bool
0115:     is_storable: bool
0116: 
0117: 
0118: @dataclass(frozen=True)
0119: class TenseAspect:
0120:     """Tense x Aspect from verb morphology.
0121:     Spec Part 1, Field: temporal_direction."""
0122:     tense: str   # past | present | future
0123:     aspect: str  # simple | continuous | perfect | perfect_continuous
0124: 
0125: 
0126: @dataclass(frozen=True)
0127: class Triple:
0128:     """Extracted (subject, predicate, object) -- DERIVED from traces.
0129:     Exists for backward compatibility only."""
0130:     subject: str
0131:     predicate: str
0132:     object: str
0133:     is_historical: bool
0134:     utterance_type: int
0135:     negated: bool
0136:     mood: str  # indicative | subjunctive | imperative | conditional
0137:     extraction_rule: str = ""
0138: 
0139: 
0140: @dataclass
0141: class TraceDecomposition:
0142:     """Five-trace decomposition of a single fact.  PRIMARY output.
0143:     Spec Part 1 defines every field and its downstream column."""
0144:     episodic_fact: str
0145:     episodic_significance: str = "routine"
0146:     emotional_state: Optional[str] = None
0147:     emotional_valence: Optional[float] = None
0148:     emotional_target: Optional[str] = None
0149:     temporal_direction: str = "present"
0150:     temporal_expression: Optional[str] = None
0151:     temporal_resolved: Optional[str] = None
0152:     relational_subject: str = "user"
0153:     relational_entities: List[str] = field(default_factory=list)
0154:     relational_type: str = "personal"
0155:     schematic_category: str = "uncategorized"
0156:     source_text: str = ""
0157:     utterance_type: int = 0
0158:     mood: str = "indicative"
0159:     negated: bool = False
0160:     is_historical: bool = False
0161:     subject: str = ""
0162:     predicate: str = ""
0163:     object: str = ""
0164:     extraction_rule: str = ""
0165: 
0166: 
0167: @dataclass
0168: class GrammarResult:
0169:     """Aggregate output of process().  One per input turn.
0170:     trace_decompositions is PRIMARY; triples is DERIVED."""
0171:     trace_decompositions: List[TraceDecomposition]
0172:     triples: List[Triple]
0173:     classification: UtteranceClassification
0174:     mood: str
0175:     negated: bool
0176:     tense_aspect: TenseAspect
0177:     voice: str
0178:     resolved_text: str
0179:     emotion: Optional[str] = None
0180: 
0181: 
0182: # ---------------------------------------------------------------------------
0183: # VerbClass enum + WordNet hypernym closure detector
0184: # Spec Part 1, Field: edge_schematic_category (steps 1-7)
0185: # ---------------------------------------------------------------------------
0186: 
0187: class VerbClass(str, Enum):
0188:     """16 verb classes.  Each maps to a schema via _VERB_CLASS_TO_SCHEMA."""
0189:     BE = "BE_VERBS"
0190:     HAVE = "HAVE_VERBS"
0191:     LOCATION = "LOCATION_VERBS"
0192:     WORK = "WORK_VERBS"
0193:     PREFERENCE = "PREFERENCE_VERBS"
0194:     ABILITY = "ABILITY_VERBS"
0195:     INJURY = "INJURY_VERBS"
0196:     PROBLEM = "PROBLEM_VERBS"
0197:     STATUS = "STATUS_VERBS"
0198:     SPEECH = "SPEECH_VERBS"
0199:     ACHIEVEMENT = "ACHIEVEMENT_VERBS"
0200:     EXPERIENCE = "EXPERIENCE_VERBS"
0201:     PLANNING = "PLANNING_VERBS"
0202:     HABIT = "HABIT_VERBS"
0203:     MEASUREMENT = "MEASUREMENT_VERBS"
0204:     UNKNOWN = "UNKNOWN"
0205: 
0206: 
0207: _VERB_CLASS_TO_SCHEMA: Dict[VerbClass, str] = {
0208:     VerbClass.WORK: "career",
0209:     VerbClass.LOCATION: "housing",
0210:     VerbClass.PREFERENCE: "identity",
0211:     VerbClass.INJURY: "health",
0212:     VerbClass.PROBLEM: "health",
0213:     VerbClass.ACHIEVEMENT: "career",
0214:     VerbClass.EXPERIENCE: "experience",
0215:     VerbClass.PLANNING: "planning",
0216:     VerbClass.HABIT: "hobby",
0217:     VerbClass.MEASUREMENT: "finance",
0218:     VerbClass.STATUS: "identity",
0219:     VerbClass.SPEECH: "social",
0220:     VerbClass.ABILITY: "education",
0221:     VerbClass.BE: "identity",
0222:     VerbClass.HAVE: "uncategorized",
0223: }
0224: 
0225: _VERB_CLASS_TO_RELTYPE: Dict[VerbClass, str] = {
0226:     VerbClass.WORK: "professional",
0227:     VerbClass.LOCATION: "personal",
0228:     VerbClass.PREFERENCE: "personal",
0229:     VerbClass.INJURY: "personal",
0230:     VerbClass.ACHIEVEMENT: "professional",
0231:     VerbClass.EXPERIENCE: "personal",
0232:     VerbClass.PLANNING: "personal",
0233:     VerbClass.SPEECH: "social",
0234:     VerbClass.HABIT: "personal",
0235:     VerbClass.ABILITY: "personal",
0236: }
0237: 
0238: # Stative verb classes: describe states rather than events/actions.
0239: # When aspect is "simple", these produce ongoing states (not one-time events).
0240: # Grammar Gap #6: stative vs dynamic verb distinction.
0241: # Grammar reference pp. 239-247: stative verbs express states, not actions.
0242: # Categories: BE, HAVE, preference, cognition (ABILITY), possession,
0243: # perception, measurement. These produce persisting facts, not events.
0244: _STATIVE_VERB_CLASSES: frozenset = frozenset({
0245:     VerbClass.BE, VerbClass.HAVE, VerbClass.PREFERENCE, VerbClass.STATUS,
0246:     VerbClass.ABILITY,  # know, understand, believe, think
0247: })
0248: 
0249: 
0250: def _compute_significance(verb_class: "VerbClass", tense_aspect: "TenseAspect") -> str:
0251:     """Compute episodic_significance from tense x aspect x verb_class.
0252: 
0253:     The tense-aspect combination is the primary signal:
0254:       present + simple -> stative (persisting state: "I work at Google")
0255:       past + simple + ACHIEVEMENT -> milestone (completed achievement: "I graduated")
0256:       past + simple + other -> routine (past event: "I ate breakfast")
0257:       present + continuous -> routine (ongoing action: "I'm eating lunch")
0258:       past + habitual -> stative (former persisting state: "I used to work at Google")
0259:       future + any -> routine (hasn't happened yet)
0260: 
0261:     Verb class refines within tense-aspect categories:
0262:       EXPERIENCE verbs in any past tense -> notable
0263:       ACHIEVEMENT verbs in past simple -> milestone
0264:     """
0265:     tense = tense_aspect.tense    # past | present | future
0266:     aspect = tense_aspect.aspect  # simple | continuous | perfect | perfect_continuous | habitual
0267: 
0268:     # Present simple = persisting state (regardless of verb class)
0269:     # "I work at Google", "I love chocolate", "I live in Portland", "I own a cat"
0270:     if tense == "present" and aspect == "simple":
0271:         return "stative"
0272: 
0273:     # Past habitual = former persisting state
0274:     # "I used to work at Google", "I used to live in Boston"
0275:     if tense == "past" and aspect == "habitual":
0276:         return "stative"
0277: 
0278:     # Stative verb classes in present perfect = persisting state
0279:     # "I've lived here for 10 years", "I've known him since college"
0280:     if verb_class in _STATIVE_VERB_CLASSES and aspect == "perfect" and tense == "present":
0281:         return "stative"
0282: 
0283:     # Past simple + ACHIEVEMENT = milestone
0284:     # "I graduated", "I got married", "I won the award"
0285:     if tense == "past" and aspect == "simple" and verb_class == VerbClass.ACHIEVEMENT:
0286:         return "milestone"
0287: 
0288:     # Past + EXPERIENCE = notable
0289:     # "I visited Paris", "I went skydiving"
0290:     if tense == "past" and verb_class == VerbClass.EXPERIENCE:
0291:         return "notable"
0292: 
0293:     # Everything else = routine
0294:     # Present continuous ("I'm eating"), past simple non-achievement ("I ate"),
0295:     # future ("I will go"), etc.
0296:     return "routine"
0297: 
0298: 
0299: # Synset-name anchors for hypernym closure.
0300: _VERB_CLASS_ANCHORS: dict[VerbClass, list[str]] = {
0301:     VerbClass.BE: ["be.v.01"],
0302:     VerbClass.HAVE: ["have.v.01", "own.v.01", "possess.v.03"],
0303:     VerbClass.LOCATION: [
0304:         "travel.v.01", "move.v.02", "reside.v.01",
0305:         "inhabit.v.01", "populate.v.01",
0306:     ],
0307:     VerbClass.WORK: [
0308:         "work.v.01", "work.v.02", "manage.v.01",
0309:         "teach.v.01", "pursue.v.01",
0310:     ],
0311:     VerbClass.PREFERENCE: ["like.v.02", "love.v.01", "enjoy.v.01", "hate.v.01"],
0312:     VerbClass.ABILITY: ["know.v.01", "understand.v.01"],
0313:     VerbClass.INJURY: ["injure.v.01", "hurt.v.01", "wound.v.01"],
0314:     VerbClass.PROBLEM: ["fail.v.01", "break.v.01", "malfunction.v.01"],
0315:     VerbClass.STATUS: ["change_state.v.01", "become.v.01"],
0316:     VerbClass.SPEECH: [
0317:         "communicate.v.02", "say.v.01", "tell.v.01", "think.v.01",
0318:     ],
0319:     VerbClass.ACHIEVEMENT: [
0320:         "succeed.v.01", "win.v.01", "achieve.v.01",
0321:         # Life-event structural parents found via WordNet hypernym paths:
0322:         "unite.v.01",       # marry.v.01 -> join.v.01 -> unite.v.01
0323:         "receive.v.01",     # graduate.v.01 -> get.v.01 -> receive.v.01
0324:         "leave.v.08",       # retire.v.01 -> leave_office.v.01 -> leave.v.08
0325:         "separate.v.08",    # divorce.v.02 -> separate.v.08
0326:         # Direct synsets (short chains that overlap other classes in closure):
0327:         "die.v.01",         # change_state.v.01 parent overlaps STATUS
0328:         "enroll.v.01",      # have.v.01 ancestor overlaps HAVE
0329:     ],
0330:     VerbClass.EXPERIENCE: ["experience.v.01", "visit.v.01", "travel.v.01"],
0331:     VerbClass.PLANNING: ["plan.v.01", "intend.v.01", "schedule.v.01"],
0332:     VerbClass.HABIT: ["use.v.01", "practice.v.01"],
0333:     VerbClass.MEASUREMENT: ["measure.v.01", "weigh.v.01", "cost.v.01"],
0334: }
0335: 
0336: 
0337: @functools.lru_cache(maxsize=2048)
0338: def _hypernym_closure(synset_name: str) -> frozenset:
0339:     """Return frozenset of all hypernym synset names for a given synset."""
0340:     _ensure_wordnet()
0341:     from nltk.corpus import wordnet as wn  # type: ignore
0342:     try:
0343:         ss = wn.synset(synset_name)
0344:     except Exception:
0345:         return frozenset()
0346:     closure = set()
0347:     for path in ss.hypernym_paths():
0348:         for ancestor in path:
0349:             closure.add(ancestor.name())
0350:     return frozenset(closure)
0351: 
0352: 
0353: @functools.lru_cache(maxsize=1)
0354: def _anchor_sets() -> dict:
0355:     """Build {VerbClass: frozenset(anchor_synset_names)}, resolved once."""
0356:     _ensure_wordnet()
0357:     from nltk.corpus import wordnet as wn  # type: ignore
0358:     result = {}
0359:     for vc, names in _VERB_CLASS_ANCHORS.items():
0360:         resolved = set()
0361:         for n in names:
0362:             try:
0363:                 canonical = wn.synset(n).name()
0364:                 resolved.add(canonical)
0365:             except Exception:
0366:                 logger.debug("Anchor synset %s not found in WordNet", n)
0367:         result[vc] = frozenset(resolved)
0368:     return result
0369: 
0370: 
0371: @functools.lru_cache(maxsize=4096)
0372: def classify_verb_class(lemma: str) -> VerbClass:
0373:     """Classify a verb lemma using WordNet hypernym closure.
0374:     Open-vocabulary.  Two-pass: direct synset match, then closure.
0375:     Spec Part 1, Field: schematic_category (step 3)."""
0376:     if lemma == "be":
0377:         return VerbClass.BE
0378:     if lemma == "have":
0379:         return VerbClass.HAVE
0380: 
0381:     _ensure_wordnet()
0382:     from nltk.corpus import wordnet as wn  # type: ignore
0383:     verb_synsets = wn.synsets(lemma, pos=wn.VERB)
0384:     if not verb_synsets:
0385:         return VerbClass.UNKNOWN
0386: 
0387:     anchors = _anchor_sets()
0388: 
0389:     # Pass 1: direct synset name match
0390:     synset_names = frozenset(ss.name() for ss in verb_synsets)
0391:     for vc, anchor_names in anchors.items():
0392:         if synset_names & anchor_names:
0393:             return vc
0394: 
0395:     # Pass 2: hypernym closure match
0396:     for ss in verb_synsets:
0397:         closure = _hypernym_closure(ss.name())
0398:         for vc, anchor_names in anchors.items():
0399:             if closure & anchor_names:
0400:                 return vc
0401: 
0402:     return VerbClass.UNKNOWN
0403: 
0404: 
0405: # ---------------------------------------------------------------------------
0406: # Dep tree helpers
0407: # ---------------------------------------------------------------------------
0408: 
0409: def _get_root(doc):
0410:     """Return ROOT token from doc.  Returns None if no ROOT found."""
0411:     for tok in doc:
0412:         if tok.dep_ == "ROOT":
0413:             return tok
0414:     return None
0415: 
0416: 
0417: def _span_text(tok) -> str:
0418:     """Get the full subtree text of a token, preserving word order.
0419:     Spec Part 1, Field: object -- subtree extraction for noun phrases."""
0420:     subtree = sorted(tok.subtree, key=lambda t: t.i)
0421:     return " ".join(t.text for t in subtree)
0422: 
0423: 
0424: def _extract_grammatical_object(doc, root, _is_recursive: bool = False) -> str:
0425:     """Extract the grammatical object as the SHORTEST noun phrase that IS the answer.
0426: 
0427:     Priority (Spec Part 1, Field: object):
0428:         1. dobj (direct object):  "researched [adoption agencies]"
0429:         2. attr (predicate nominal):  "is [a transgender woman]"
0430:         3. acomp (adjective complement):  "felt [accepted as a transgender woman]"
0431:         4. pobj (prepositional object):  "moved from [Sweden]"
0432:         5. xcomp chain:  "want to pursue [counseling]" -> recurse into xcomp
0433:         6. ccomp (clausal complement):  "realized [self-care is important]" minus "that"
0434:         7. oprd (object predicate):  "consider [him a friend]"
0435: 
0436:     Critical rules from spec:
0437:         - SPEECH verbs with ccomp: skip speech frame, extract from embedded clause
0438:         - INTENT verbs with xcomp: skip intent frame, extract from xcomp's object
0439:         - NEVER include the subject in the object
0440:     """
0441:     if root is None:
0442:         return ""
0443: 
0444:     # 0. Fragment with relcl: ROOT is a NOUN with a relative clause verb.
0445:     #    "Ones that support LGBTQ+ individuals" -- ROOT=Ones, relcl=support.
0446:     #    Extract from the relcl verb's arguments (its dobj/attr/pobj).
0447:     if root.pos_ in ("NOUN", "PRON"):
0448:         for child in root.children:
0449:             if child.dep_ == "relcl" and child.pos_ in ("VERB", "AUX"):
0450:                 relcl_obj = _extract_grammatical_object(
0451:                     doc, child, _is_recursive=True,
0452:                 )
0453:                 if relcl_obj:
0454:                     return relcl_obj
0455: 
0456:     # 1. Direct object: "researched [adoption agencies]"
0457:     #    At top level only (not xcomp recursion), include purpose/description
0458:     #    prep phrases (for, about) attached to ROOT that modify the event object.
0459:     #    "ran a charity race ... for mental health awareness"
0460:     #      -> "a charity race for mental health awareness"
0461:     for child in root.children:
0462:         if child.dep_ == "dobj":
0463:             dobj_text = _span_text(child)
0464:             # Only at top level: append purpose preps
0465:             if not _is_recursive:
0466:                 _PURPOSE_PREPS = frozenset({
0467:                     "for", "about", "on", "toward", "towards",
0468:                 })
0469:                 for sibling in root.children:
0470:                     if (sibling.dep_ == "prep"
0471:                             and sibling.pos_ == "ADP"
0472:                             and sibling.lemma_ in _PURPOSE_PREPS
0473:                             and sibling.i > child.i):
0474:                         pobj_tok = None
0475:                         for gc in sibling.children:
0476:                             if gc.dep_ == "pobj":
0477:                                 pobj_tok = gc
0478:                                 break
0479:                         if pobj_tok:
0480:                             # Guard: if pobj is a pronoun or PERSON entity,
0481:                             # it's a beneficiary/recipient, not the semantic
0482:                             # object.  Keep dobj.
0483:                             # "bought a gift for her" -> "a gift" (not "her")
0484:                             if pobj_tok.pos_ == "PRON":
0485:                                 break
0486:                             pobj_ner = {
0487:                                 t.ent_type_ for t in pobj_tok.subtree
0488:                                 if t.ent_type_
0489:                             }
0490:                             if pobj_ner & frozenset({"DATE", "TIME", "PERSON"}):
0491:                                 break
0492:                             # Purpose prep promotion: pobj IS the semantic
0493:                             # object. "raised awareness for mental health"
0494:                             # -> object = "mental health" (not dobj+prep).
0495:                             return _span_text(pobj_tok)
0496:             return dobj_text
0497: 
0498:     # 2. Attribute complement: "is [a transgender woman]"
0499:     for child in root.children:
0500:         if child.dep_ == "attr":
0501:             return _span_text(child)
0502: 
0503:     # 3. Adjective complement: "felt [accepted as a transgender woman]"
0504:     for child in root.children:
0505:         if child.dep_ == "acomp":
0506:             # For linking verb + single ADJ complement, the subject NP is
0507:             # the semantic answer. "The sunday before 25 May 2023 was lovely"
0508:             # â†’ object = "The sunday before 25 May 2023" (not "lovely").
0509:             # But "I felt accepted as a transgender woman" â†’ object = the
0510:             # full acomp span (has prepositional content beyond the ADJ).
0511:             acomp_span = _span_text(child)
0512:             if (child.pos_ == "ADJ"
0513:                     and root.lemma_ in _COPULAR_LEMMAS
0514:                     and len(list(child.subtree)) <= 2):
0515:                 # Single ADJ complement â†’ return subject NP instead
0516:                 for sib in root.children:
0517:                     if sib.dep_ in ("nsubj", "nsubjpass"):
0518:                         return _span_text(sib)
0519:             return acomp_span
0520: 
0521:     # 4. Prepositional object: "moved from [Sweden]"
0522:     # Skip preps that duplicate a particle (prt) on the same verb to avoid
0523:     # treating phrasal-verb particles as semantic prepositions.
0524:     # Also skip preps whose pobj is a temporal expression (DATE/TIME NER),
0525:     # e.g., "moved on Tuesday from Sweden" -> object = "Sweden" not "Tuesday".
0526:     prt_lemmas = frozenset(
0527:         c.lemma_.lower() for c in root.children if c.dep_ == "prt"
0528:     )
0529:     for child in root.children:
0530:         if child.dep_ == "prep":
0531:             if child.lemma_.lower() in prt_lemmas:
0532:                 continue  # particle, not a true preposition
0533:             for gc in child.children:
0534:                 if gc.dep_ == "pobj":
0535:                     _pobj_ner = {
0536:                         t.ent_type_ for t in gc.subtree if t.ent_type_
0537:                     }
0538:                     if _pobj_ner & frozenset({"DATE", "TIME"}):
0539:                         break  # temporal prep, skip to next prep child
0540:                     return _span_text(gc)
0541: 
0542:     # 5-6. Xcomp / ccomp chains
0543:     for child in root.children:
0544:         if child.dep_ == "ccomp":
0545:             # ccomp = full clausal complement.  Strip complementizer "that".
0546:             # Spec: "realized that [self-care is important]" -> "self-care is important"
0547:             subtree = sorted(child.subtree, key=lambda t: t.i)
0548:             filtered = [t.text for t in subtree
0549:                         if not (t.dep_ == "mark" and t.lemma_.lower() == "that")]
0550:             return " ".join(filtered).strip()
0551:         if child.dep_ == "xcomp":
0552:             # xcomp = open complement.  Recurse to find xcomp's own object.
0553:             # Spec: "want to pursue [counseling]" -> "counseling"
0554:             xcomp_obj = _extract_grammatical_object(
0555:                 doc, child, _is_recursive=True,
0556:             )
0557:             if xcomp_obj:
0558:                 return xcomp_obj
0559:             # If xcomp has no object, return its full subtree minus subject
0560:             return _span_text(child)
0561: 
0562:     # 7. Object predicate: "consider [him a friend]"
0563:     for child in root.children:
0564:         if child.dep_ == "oprd":
0565:             return _span_text(child)
0566: 
0567:     return ""
0568: 
0569: 
0570: def _get_prep_object(
0571:     root, prep_lemmas: Optional[frozenset] = None,
0572: ) -> Tuple[Optional[str], Optional[str]]:
0573:     """Get (preposition, pobj_text) from root's prep children.
0574: 
0575:     Skips temporal preps whose pobj subtree contains DATE/TIME NER entities,
0576:     e.g. "moved on Tuesday from Sweden" returns ("from", "Sweden") not ("on", "Tuesday").
0577:     """
0578:     for child in root.children:
0579:         if child.dep_ == "prep":
0580:             if prep_lemmas and child.lemma_ not in prep_lemmas:
0581:                 continue
0582:             for gc in child.children:
0583:                 if gc.dep_ == "pobj":
0584:                     _pobj_ner = {
0585:                         t.ent_type_ for t in gc.subtree if t.ent_type_
0586:                     }
0587:                     if _pobj_ner & frozenset({"DATE", "TIME"}):
0588:                         break  # temporal prep, skip to next prep child
0589:                     return (child.lemma_, _span_text(gc))
0590:     return (None, None)
0591: 
0592: 
0593: # ---------------------------------------------------------------------------
0594: # Reclassification helpers
0595: # Spec Part 1, Field: schematic_category (step 1 -- xcomp/ccomp override)
0596: # ---------------------------------------------------------------------------
0597: 
0598: def _reclassify_location_by_object(doc, root, verb_class):
0599:     """When LOCATION verb has a non-geographic object, reclassify as EXPERIENCE.
0600:     "went to a support group" = EXPERIENCE, not LOCATION."""
0601:     if verb_class != VerbClass.LOCATION or root is None:
0602:         return verb_class
0603: 
0604:     for child in root.children:
0605:         if child.dep_ == "dobj":
0606:             ner_types = {t.ent_type_ for t in child.subtree if t.ent_type_}
0607:             if ner_types & frozenset({"GPE", "LOC", "FAC"}):
0608:                 return VerbClass.LOCATION
0609:             return VerbClass.EXPERIENCE
0610: 
0611:     # Check ALL prep children, not just "to". "go through a rough patch",
0612:     # "go into business", "go over the details" are EXPERIENCE, not LOCATION.
0613:     for child in root.children:
0614:         if child.dep_ == "prep":
0615:             for gc in child.children:
0616:                 if gc.dep_ == "pobj":
0617:                     ner_types = {t.ent_type_ for t in gc.subtree if t.ent_type_}
0618:                     if ner_types & frozenset({"GPE", "LOC", "FAC"}):
0619:                         return VerbClass.LOCATION
0620:                     return VerbClass.EXPERIENCE
0621:     return verb_class
0622: 
0623: 
0624: def _detect_planning_pattern(doc, root, verb_class):
0625:     """Detect planning constructions via dep-tree structure.
0626:     Grammar reference Section 3.5: "thinking about moving" = PLANNING."""
0627:     if root is None:
0628:         return verb_class
0629: 
0630:     # Pattern 1: {verb} about {gerund}
0631:     for child in root.children:
0632:         if child.dep_ == "prep" and child.lemma_ == "about":
0633:             for gc in child.children:
0634:                 if gc.dep_ in ("pcomp", "pobj") and gc.tag_ == "VBG":
0635:                     return VerbClass.PLANNING
0636: 
0637:     # Pattern 2: {be} {VBG} to {verb}
0638:     if root.tag_ == "VBG":
0639:         for child in root.children:
0640:             if child.dep_ == "xcomp":
0641:                 for gc in child.children:
0642:                     if gc.dep_ in ("mark", "aux") and gc.lemma_ == "to":
0643:                         return VerbClass.PLANNING
0644: 
0645:     return verb_class
0646: 
0647: 
0648: # ---------------------------------------------------------------------------
0649: # Grammatical feature detection
0650: # Spec Part 1, Fields: mood, negated, temporal_direction, voice
0651: # ---------------------------------------------------------------------------
0652: 
0653: def detect_mood(doc) -> str:
0654:     """Detect sentence mood: indicative/interrogative/imperative/conditional/subjunctive.
0655:     Spec Part 1, Field: edge_mood.
0656:     Grammar reference Section 1 (Sentence Classification)."""
0657:     if isinstance(doc, str):
0658:         doc = _get_nlp()(doc)
0659: 
0660:     root = _get_root(doc)
0661:     if root is None:
0662:         return "indicative"
0663: 
0664:     # Imperative: ROOT verb with no nsubj, VerbForm=Inf
0665:     has_nsubj = any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children)
0666:     if (not has_nsubj and root.pos_ == "VERB"
0667:             and root.morph.get("VerbForm") in (["Inf"], ["Fin"])
0668:             and root.morph.get("Tense") == []):
0669:         return "imperative"
0670: 
0671:     # Subjunctive: "wish" + ccomp/xcomp clause (counterfactual desire)
0672:     # Grammar pp.735-739: verb "wish" with tense-shifted embedded clause
0673:     if root.lemma_ == "wish":
0674:         for child in root.children:
0675:             if child.dep_ in ("ccomp", "xcomp"):
0676:                 return "subjunctive"
0677: 
0678:     # Subjunctive: ccomp verb with no tense marking, OR
0679:     # mandative subjunctive: 3rd-person subject + base-form verb (VB not VBZ).
0680:     # "I recommend she study harder" â€” "study" is VB where VBZ expected.
0681:     # Grammar pp.735-739: morphological disagreement IS the structural signal.
0682:     for tok in doc:
0683:         if tok.dep_ == "ccomp" and tok.pos_ in ("VERB", "AUX"):
0684:             if (tok.morph.get("Tense") == []
0685:                     and tok.morph.get("VerbForm") in (["Inf"], [])):
0686:                 return "subjunctive"
0687:             # Mandative: 3rd person singular nsubj + non-VBZ verb form.
0688:             # "I recommend she study harder" â€” "study" is VBP (not VBZ
0689:             # "studies"). spaCy tags it VBP, not VB. The mismatch between
0690:             # 3rd person singular subject and VBP (not VBZ) = mandative.
0691:             if tok.tag_ in ("VB", "VBP"):
0692:                 ccomp_nsubj = next(
0693:                     (c for c in tok.children
0694:                      if c.dep_ in ("nsubj", "nsubjpass")), None
0695:                 )
0696:                 if (ccomp_nsubj
0697:                         and ccomp_nsubj.morph.get("Person") == ["3"]
0698:                         and ccomp_nsubj.morph.get("Number") == ["Sing"]
0699:                         and tok.tag_ != "VBZ"):
0700:                     return "subjunctive"
0701: 
0702:     # Subjunctive: "were" with 1st/3rd person singular subject
0703:     if root.lemma_ == "be" and root.text.lower() == "were":
0704:         for child in root.children:
0705:             if child.dep_ == "nsubj":
0706:                 if child.text.lower() == "i":
0707:                     return "subjunctive"
0708:                 mp = child.morph.get("Person")
0709:                 mn = child.morph.get("Number")
0710:                 if mn == ["Sing"] and mp in (["1"], ["3"]):
0711:                     return "subjunctive"
0712: 
0713:     # Gap 12: "if only" -> subjunctive (wish), not conditional.
0714:     # Must be checked BEFORE the conditional block.
0715:     for tok in doc:
0716:         if tok.text.lower() == "if" and (tok.i + 1) < len(doc) and doc[tok.i + 1].text.lower() == "only":
0717:             return "subjunctive"
0718: 
0719:     # Gap 10: Habitual "would" detection â€” "would" + temporal/frequency marker
0720:     # and NO conditional subordinator (if/unless/whether) â†’ indicative, not
0721:     # conditional. "We would go fishing every summer" = past habitual.
0722:     _HABITUAL_MARKERS = frozenset({"every", "always", "often", "usually", "frequently"})
0723:     _CONDITIONAL_SUBORDINATORS = frozenset({"if", "unless", "whether"})
0724:     has_conditional_sub = any(
0725:         tok.dep_ in ("mark", "advmod") and tok.lemma_.lower() in _CONDITIONAL_SUBORDINATORS
0726:         for tok in doc
0727:     )
0728:     has_habitual_signal = any(tok.text.lower() in _HABITUAL_MARKERS for tok in doc)
0729:     # Also check for DATE/TIME NER as temporal context
0730:     has_temporal_ner = any(ent.label_ in ("DATE", "TIME") for ent in doc.ents)
0731: 
0732:     # Conditional: modal aux (would/could) governing a verb,
0733:     # OR subordinating conjunction "if"/"unless"/"whether" (dep_=mark/advmod).
0734:     # Checked BEFORE interrogative so "Would you recommend...?" => conditional
0735:     # (Spec Part 4, Gap 5 - conditional modals take priority over question form)
0736:     # NOTE: "should"/"might" excluded â€” deontic/epistemic, not counterfactual.
0737:     _conditional_lemmas = {"would", "could"}
0738:     for tok in doc:
0739:         if tok.dep_ == "aux" and tok.lemma_.lower() in _conditional_lemmas:
0740:             if tok.head.pos_ == "VERB":
0741:                 # Gap 10 guard: habitual "would" with temporal marker and no
0742:                 # conditional subordinator â†’ indicative, not conditional
0743:                 if (tok.lemma_.lower() == "would"
0744:                         and not has_conditional_sub
0745:                         and (has_habitual_signal or has_temporal_ner)):
0746:                     break  # skip conditional, fall through to indicative
0747:                 return "conditional"
0748:     for tok in doc:
0749:         if (tok.dep_ in ("mark", "advmod")
0750:                 and tok.lemma_.lower() in ("if", "unless", "whether")):
0751:             return "conditional"
0752: 
0753:     # Interrogative: sentence ends with "?" OR subject-auxiliary inversion
0754:     # (aux token precedes nsubj in linear order)
0755:     if doc[-1].text == "?":
0756:         return "interrogative"
0757:     # Check subject-auxiliary inversion
0758:     nsubj_idx = None
0759:     aux_idx = None
0760:     for tok in doc:
0761:         if tok.dep_ in ("nsubj", "nsubjpass") and nsubj_idx is None:
0762:             nsubj_idx = tok.i
0763:         if tok.dep_ == "aux" and aux_idx is None:
0764:             aux_idx = tok.i
0765:     if aux_idx is not None and nsubj_idx is not None and aux_idx < nsubj_idx:
0766:         return "interrogative"
0767: 
0768:     return "indicative"
0769: 
0770: 
0771: def detect_negation(doc) -> bool:
0772:     """Detect negation anywhere in the sentence.
0773:     Spec Part 1, Field: negated.
0774:     Grammar reference Section 9: Negation (p. 417-424).
0775: 
0776:     Checks:
0777:         1. Explicit neg dep label (not, n't)
0778:         2. Implicit negative adverbs: hardly, barely, scarcely, rarely, seldom, never
0779:            These are a closed grammatical class, not a word list."""
0780:     if isinstance(doc, str):
0781:         doc = _get_nlp()(doc)
0782: 
0783:     for tok in doc:
0784:         if tok.dep_ == "neg":
0785:             return True
0786:         if (tok.pos_ == "ADV"
0787:                 and tok.dep_ == "advmod"
0788:                 and tok.lemma_.lower() in (
0789:                     "hardly", "barely", "scarcely",
0790:                     "rarely", "seldom", "never",
0791:                 )):
0792:             return True
0793:     return False
0794: 
0795: 
0796: def detect_tense_aspect(doc) -> TenseAspect:
0797:     """Extract tense x aspect from verb morphology.
0798:     Spec Part 1, Field: temporal_direction (base tense).
0799:     Grammar reference Section 5: Temporal Markers.
0800:     Grammar reference Pages 718-720: Habitual aspect."""
0801:     if isinstance(doc, str):
0802:         doc = _get_nlp()(doc)
0803: 
0804:     root = _get_root(doc)
0805:     if root is None:
0806:         return TenseAspect(tense="present", aspect="simple")
0807: 
0808:     # HABITUAL PAST: "used to" + VB (Grammar ref Pages 718-720)
0809:     # Pattern: "used" (VBD, lemma="use") followed by "to" + base verb
0810:     # spaCy may parse this in different ways:
0811:     #   1. "used" as ROOT with "to"+verb as xcomp
0812:     #   2. "used" as aux of the main verb
0813:     # Check all tokens for the "used to" idiom.
0814:     tokens = list(doc)
0815:     for i, tok in enumerate(tokens):
0816:         if (tok.lemma_.lower() == "use"
0817:                 and tok.tag_ == "VBD"
0818:                 and i + 1 < len(tokens)
0819:                 and tokens[i + 1].text.lower() == "to"):
0820:             # Gap 5: "be used to" = accustomed, NOT habitual.
0821:             # If preceded by a form of "be", skip habitual detection.
0822:             if i > 0 and tokens[i - 1].lemma_.lower() == "be":
0823:                 continue  # "am/is/are/was/were used to" = accustomed
0824:             # Verify there's a verb after "to"
0825:             if i + 2 < len(tokens) and tokens[i + 2].pos_ == "VERB":
0826:                 return TenseAspect(tense="past", aspect="habitual")
0827:             # Also check xcomp children of "used"
0828:             for child in tok.children:
0829:                 if child.dep_ == "xcomp" and child.pos_ == "VERB":
0830:                     return TenseAspect(tense="past", aspect="habitual")
0831: 
0832:     auxes = sorted(
0833:         [c for c in root.children if c.dep_ in ("aux", "auxpass")],
0834:         key=lambda t: t.i,
0835:     )
0836:     aux_lemmas = [a.lemma_.lower() for a in auxes]
0837: 
0838:     # FUTURE: "be going to" periphrastic future (Page 697-698)
0839:     # Pattern: AUX(am/is/are/was/were) + "going" (ROOT/xcomp) + "to" + VERB
0840:     # spaCy often parses: "going" as ROOT with "be" as aux, main verb as xcomp
0841:     if root.lemma_ == "go" and root.tag_ == "VBG" and "be" in aux_lemmas:
0842:         # Check if "going" has an xcomp child (the actual main verb)
0843:         has_to_verb = False
0844:         for child in root.children:
0845:             if child.dep_ == "xcomp" and child.pos_ == "VERB":
0846:                 # Verify "to" is present as mark/aux of the xcomp
0847:                 for grandchild in child.children:
0848:                     if grandchild.lemma_ == "to" and grandchild.dep_ in (
0849:                         "mark", "aux",
0850:                     ):
0851:                         has_to_verb = True
0852:                         break
0853:                 if has_to_verb:
0854:                     break
0855:         if has_to_verb:
0856:             return TenseAspect(tense="future", aspect="simple")
0857: 
0858:     # FUTURE: will/shall
0859:     if "will" in aux_lemmas or "shall" in aux_lemmas:
0860:         tense = "future"
0861:         if "have" in aux_lemmas and "be" in aux_lemmas:
0862:             aspect = "perfect_continuous"
0863:         elif "have" in aux_lemmas:
0864:             aspect = "perfect"
0865:         elif "be" in aux_lemmas:
0866:             aspect = "continuous"
0867:         else:
0868:             aspect = "simple"
0869:         return TenseAspect(tense=tense, aspect=aspect)
0870: 
0871:     # Tense from ROOT or AUX morphology
0872:     root_tense = root.morph.get("Tense")
0873:     tense = "present"
0874:     if root_tense == ["Past"]:
0875:         tense = "past"
0876:     for a in auxes:
0877:         if a.morph.get("Tense") == ["Past"]:
0878:             tense = "past"
0879:             break
0880: 
0881:     # Aspect from AUX combination
0882:     has_have = "have" in aux_lemmas
0883:     has_be = "be" in aux_lemmas
0884:     root_is_ing = root.tag_ == "VBG"
0885: 
0886:     if has_have and has_be and root_is_ing:
0887:         aspect = "perfect_continuous"
0888:     elif has_have:
0889:         aspect = "perfect"
0890:     elif has_be and root_is_ing:
0891:         aspect = "continuous"
0892:     else:
0893:         aspect = "simple"
0894: 
0895:     return TenseAspect(tense=tense, aspect=aspect)
0896: 
0897: 
0898: def detect_voice(doc) -> str:
0899:     """Detect active / passive / middle voice.
0900:     Grammar reference Section 3: Predicate/Action.
0901: 
0902:     Known limitation (Grammar Gap #5 -- Intransitive Middle Voice):
0903:         "The lasagna cooked in the oven" returns "active" instead of "middle".
0904:         True intransitive middle voice (subject is patient, no passive morphology,
0905:         no agent expressed) requires verb-frame semantic knowledge (whether the
0906:         subject COULD be an agent) that spaCy does not provide.  Only reflexive
0907:         middle voice ("He dressed himself") and passive morphology are detected.
0908:         Detecting intransitive middle would require a verb transitivity/animacy
0909:         lexicon beyond what structural POS/dep parsing offers.
0910:     """
0911:     if isinstance(doc, str):
0912:         doc = _get_nlp()(doc)
0913: 
0914:     has_passive_subj = False
0915:     has_auxpass = False
0916:     has_reflexive_obj = False
0917:     _reflexives = frozenset({
0918:         "myself", "yourself", "himself", "herself",
0919:         "itself", "ourselves", "yourselves", "themselves",
0920:     })
0921: 
0922:     for tok in doc:
0923:         if tok.dep_ == "nsubjpass":
0924:             has_passive_subj = True
0925:         if tok.dep_ == "auxpass":
0926:             has_auxpass = True
0927:         if tok.dep_ in ("dobj", "pobj") and tok.text.lower() in _reflexives:
0928:             has_reflexive_obj = True
0929: 
0930:     if has_passive_subj or has_auxpass:
0931:         return "passive"
0932:     if has_reflexive_obj:
0933:         return "middle"
0934:     return "active"
0935: 
0936: 
0937: def resolve_pronouns(doc, speaker: Optional[str] = None, listener: str = "user"):
0938:     """Resolve first- and second-person pronouns using speaker metadata.
0939:     Spec Part 1, Field: subject -- "I" -> speaker name.
0940:     Grammar reference Section 7: Entity Type Patterns.
0941: 
0942:     Args:
0943:         doc: spaCy Doc or raw text string.
0944:         speaker: Name of the person speaking (resolves "I"/"me"/"my"/"myself").
0945:         listener: Name of the person being addressed (resolves "you"/"your"/"yourself"/"yours").
0946:                   Defaults to "user" for backward compatibility in single-speaker mode.
0947:     """
0948:     if isinstance(doc, str):
0949:         doc = _get_nlp()(doc)
0950: 
0951:     tokens: list[str] = []
0952:     speaker_name = speaker if speaker else "user"
0953:     listener_name = listener
0954: 
0955:     for tok in doc:
0956:         lower = tok.text.lower()
0957:         if lower == "i" and tok.dep_ in ("nsubj", "nsubjpass", "ROOT"):
0958:             tokens.append(speaker_name)
0959:         elif lower == "me" and tok.dep_ in ("dobj", "pobj", "dative"):
0960:             tokens.append(speaker_name)
0961:         elif lower == "my":
0962:             tokens.append(speaker_name + "'s")
0963:         elif lower == "myself":
0964:             tokens.append(speaker_name)
0965:         elif lower == "you":
0966:             tokens.append(listener_name)
0967:         elif lower == "your":
0968:             tokens.append(listener_name + "'s")
0969:         elif lower == "yourself":
0970:             tokens.append(listener_name)
0971:         elif lower == "yours":
0972:             tokens.append(listener_name + "'s")
0973:         # Gap 6: Standalone possessive pronouns
0974:         elif lower == "mine":
0975:             tokens.append(speaker_name + "'s")
0976:         elif lower == "ours":
0977:             tokens.append(speaker_name + "'s")
0978:         # Contraction conjugation: after "I" -> speaker (3rd person),
0979:         # AUX needs 3rd-person form.
0980:         elif tok.pos_ == "AUX" and tok.text.startswith("'"):
0981:             lemma = tok.lemma_
0982:             morph = tok.morph
0983:             if lemma == "be":
0984:                 t = morph.get("Tense", ["Pres"])[0] if morph.get("Tense") else "Pres"
0985:                 tokens.append(" is" if t == "Pres" else " was" if t == "Past" else " " + lemma)
0986:             elif lemma == "have":
0987:                 t = morph.get("Tense", ["Pres"])[0] if morph.get("Tense") else "Pres"
0988:                 tokens.append(" has" if t == "Pres" else " had")
0989:             else:
0990:                 tokens.append(" " + lemma)
0991:         else:
0992:             tokens.append(tok.text)
0993: 
0994:     resolved = ""
0995:     for i, tok in enumerate(doc):
0996:         resolved += tokens[i]
0997:         if tok.whitespace_:
0998:             resolved += tok.whitespace_
0999:     return resolved
1000: 
1001: 
1002: # ---------------------------------------------------------------------------
1003: # Utterance classification
1004: # Spec Part 2: Extraction Rules by Sentence Type
1005: # ---------------------------------------------------------------------------
1006: 
1007: def _has_question_mark(doc) -> bool:
1008:     for tok in reversed(list(doc)):
1009:         if tok.text == "?":
1010:             return True
1011:         if tok.pos_ != "SPACE":
1012:             break
1013:     return False
1014: 
1015: 
1016: def _has_interrogative_fronted(doc) -> bool:
1017:     """Check for WH-fronting.  Grammar reference Section 1."""
1018:     for tok in doc:
1019:         if tok.pos_ == "SPACE":
1020:             continue
1021:         pron_type = tok.morph.get("PronType")
1022:         if pron_type and "Int" in pron_type:
1023:             return True
1024:         if tok.tag_ in ("WDT", "WP", "WP$", "WRB"):
1025:             if tok.dep_ in ("advmod", "attr", "nsubj", "dobj", "det"):
1026:                 return True
1027:         break
1028:     return False
1029: 
1030: 
1031: def _has_subject_aux_inversion(doc) -> bool:
1032:     """Check for subject-aux inversion (yes/no questions)."""
1033:     root = _get_root(doc)
1034:     if root is None:
1035:         return False
1036:     aux_i = None
1037:     subj_i = None
1038:     for child in root.children:
1039:         if child.dep_ == "aux" and aux_i is None:
1040:             aux_i = child.i
1041:         if child.dep_ in ("nsubj", "nsubjpass") and subj_i is None:
1042:             subj_i = child.i
1043:     return aux_i is not None and subj_i is not None and aux_i < subj_i
1044: 
1045: 
1046: def _is_imperative_structure(doc) -> bool:
1047:     """Spec Part 2, Command detection: ROOT verb with no nsubj, VerbForm=Inf or tag=VB."""
1048:     root = _get_root(doc)
1049:     if root is None or root.pos_ != "VERB":
1050:         return False
1051:     has_nsubj = any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children)
1052:     if has_nsubj:
1053:         return False
1054:     if root.tag_ == "VB":
1055:         return True
1056:     if root.morph.get("VerbForm") == ["Inf"]:
1057:         return True
1058:     return False
1059: 
1060: 
1061: def _is_backchannel_structure(doc) -> bool:
1062:     """Spec Part 2, Fragment/Backchannel detection.
1063:     Exception: fragments with NUM, NER, or content nouns ARE stored."""
1064:     tokens = [t for t in doc if t.pos_ not in ("SPACE", "PUNCT")]
1065:     if not tokens:
1066:         return True
1067: 
1068:     # ROOT is interjection = backchannel
1069:     for tok in doc:
1070:         if tok.dep_ == "ROOT" and tok.pos_ == "INTJ":
1071:             return True
1072: 
1073:     if len(tokens) <= 3:
1074:         if not any(t.pos_ in ("VERB", "AUX") for t in tokens):
1075:             has_num = any(t.pos_ == "NUM" for t in tokens)
1076:             has_ner = any(t.ent_type_ for t in tokens)
1077:             has_content_noun = any(
1078:                 t.pos_ == "NOUN" and not t.is_stop for t in tokens
1079:             )
1080:             if not (has_num or has_ner or has_content_noun):
1081:                 return True
1082: 
1083:     # Fragment with a verb somewhere (e.g. relcl) is content
1084:     has_verb_anywhere = any(tok.pos_ in ("VERB", "AUX") for tok in doc)
1085:     if has_verb_anywhere:
1086:         return False
1087: 
1088:     has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc)
1089:     if not has_subj and not _is_imperative_structure(doc):
1090:         # Before declaring backchannel, check if the fragment has
1091:         # substantive content: NER entities, NUM tokens, or content
1092:         # nouns/verbs (non-auxiliary). Fragments like "Running, reading,
1093:         # or playing my violin" have real content and must be stored.
1094:         has_ner = any(t.ent_type_ for t in doc)
1095:         has_num = any(t.pos_ == "NUM" for t in doc)
1096:         has_content_word = any(
1097:             t.pos_ in ("NOUN", "PROPN", "VERB") and not t.is_stop
1098:             for t in doc
1099:         )
1100:         if has_ner or has_num or has_content_word:
1101:             return False
1102:         return True
1103: 
1104:     return False
1105: 
1106: 
1107: _COPULAR_LEMMAS = frozenset({
1108:     "be", "feel", "seem", "appear", "become",
1109:     "get", "grow", "turn", "remain", "stay",
1110:     "look", "sound", "taste", "smell", "prove",
1111: })
1112: 
1113: 
1114: def _is_emotion_structure(doc) -> bool:
1115:     """Detect: 1st-person subject + copular verb + adj complement."""
1116:     root = _get_root(doc)
1117:     if root is None or root.pos_ not in ("VERB", "AUX"):
1118:         return False
1119: 
1120:     first_person = any(
1121:         c.dep_ == "nsubj" and c.text.lower() in ("i", "we")
1122:         for c in root.children
1123:     )
1124:     if not first_person:
1125:         return False
1126: 
1127:     has_adj = any(
1128:         c.dep_ in ("acomp", "oprd") and c.pos_ == "ADJ"
1129:         for c in root.children
1130:     )
1131:     return has_adj and root.lemma_ in _COPULAR_LEMMAS
1132: 
1133: 
1134: def _has_exclamation_mark(doc) -> bool:
1135:     for tok in reversed(list(doc)):
1136:         if tok.text == "!":
1137:             return True
1138:         if tok.pos_ != "SPACE":
1139:             break
1140:     return False
1141: 
1142: 
1143: def _is_tag_question(doc) -> bool:
1144:     """Detect tag questions (Grammar reference Pages 922-923).
1145: 
1146:     Pattern: declarative clause + comma + short inverted AUX+PRON + "?"
1147:     Examples: "You're going to the party, aren't you?"
1148:               "She likes coffee, doesn't she?"
1149: 
1150:     Tag questions are pragmatically assertions -- the main clause is the fact.
1151:     """
1152:     tokens = list(doc)
1153:     if len(tokens) < 5:
1154:         return False
1155: 
1156:     # Must end with "?"
1157:     if tokens[-1].text != "?":
1158:         return False
1159: 
1160:     # Find the last comma in the sentence
1161:     last_comma_idx = None
1162:     for i in range(len(tokens) - 1, -1, -1):
1163:         if tokens[i].text == ",":
1164:             last_comma_idx = i
1165:             break
1166: 
1167:     if last_comma_idx is None:
1168:         return False
1169: 
1170:     # The tag part: tokens between last comma and "?"
1171:     tag_tokens = [t for t in tokens[last_comma_idx + 1:]
1172:                   if t.text != "?" and t.pos_ != "SPACE"]
1173: 
1174:     # Tag should be short: 2-3 tokens (AUX + PRON or AUX + neg + PRON)
1175:     if len(tag_tokens) < 2 or len(tag_tokens) > 3:
1176:         return False
1177: 
1178:     # Check pattern: (AUX [+ neg] + PRON) or (VERB-as-aux [+ neg] + PRON)
1179:     # Typical: "aren't you", "doesn't she", "is it", "did he"
1180:     # spaCy sometimes tags "does/did/is" as VERB in tag position
1181:     _tag_aux_lemmas = {"do", "be", "have", "will", "shall", "can", "could",
1182:                        "would", "should", "may", "might", "must"}
1183:     has_aux = any(t.pos_ == "AUX" or t.dep_ == "aux"
1184:                   or t.lemma_.lower() in _tag_aux_lemmas
1185:                   for t in tag_tokens)
1186:     has_pron = any(t.pos_ == "PRON" for t in tag_tokens)
1187: 
1188:     if has_aux and has_pron:
1189:         # Verify main clause (before comma) has substance (subject + verb)
1190:         main_tokens = tokens[:last_comma_idx]
1191:         has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in main_tokens)
1192:         has_verb = any(t.dep_ == "ROOT" or t.pos_ in ("VERB", "AUX")
1193:                        for t in main_tokens)
1194:         return has_subj and has_verb
1195: 
1196:     return False
1197: 
1198: 
1199: def _strip_tag_question(doc) -> Any:
1200:     """Strip the tag portion from a tag question, returning only the main clause.
1201: 
1202:     "You're going to the party, aren't you?" -> "You're going to the party"
1203:     """
1204:     tokens = list(doc)
1205:     # Find last comma
1206:     last_comma_idx = None
1207:     for i in range(len(tokens) - 1, -1, -1):
1208:         if tokens[i].text == ",":
1209:             last_comma_idx = i
1210:             break
1211:     if last_comma_idx is None:
1212:         return doc
1213:     main_text = " ".join(t.text for t in tokens[:last_comma_idx]).strip()
1214:     if not main_text:
1215:         return doc
1216:     return _get_nlp_fragment()(main_text)
1217: 
1218: 
1219: def classify_utterance(
1220:     doc, speaker: Optional[str] = None,
1221: ) -> UtteranceClassification:
1222:     """Classify a spaCy Doc (or raw string) into coarse bin.
1223:     Spec Part 2: sentence type determines extraction path."""
1224:     if isinstance(doc, str):
1225:         doc = _get_nlp()(doc)
1226: 
1227:     # Exclamatory: ends with ! and has what/how fronting.
1228:     # Must be checked BEFORE question detection.
1229:     if _has_exclamation_mark(doc) and _has_interrogative_fronted(doc):
1230:         return UtteranceClassification(
1231:             4, CoarseBin.EMOTION.value, "exclamatory",
1232:             False, False, False, True, True)
1233: 
1234:     # Tag questions (Grammar ref #11, Pages 922-923):
1235:     # "You're going to the party, aren't you?" is pragmatically a statement.
1236:     # Must be checked BEFORE general question detection so we treat the
1237:     # main clause as a storable declarative fact.
1238:     if _is_tag_question(doc):
1239:         return UtteranceClassification(
1240:             5, CoarseBin.STATEMENT.value, "tag_question",
1241:             False, False, False, False, True)
1242: 
1243:     # Cleft sentences: "What happened was I applied..." â€” NOT a question.
1244:     # Pattern: ROOT is "be" + csubj (WH-clause) + ccomp (content).
1245:     _cleft_root = _get_root(doc)
1246:     if (_cleft_root and _cleft_root.lemma_ == "be"
1247:             and any(c.dep_ == "csubj" for c in _cleft_root.children)
1248:             and any(c.dep_ == "ccomp" and c.pos_ == "VERB"
1249:                     for c in _cleft_root.children)):
1250:         return UtteranceClassification(
1251:             5, CoarseBin.STATEMENT.value, "cleft",
1252:             False, False, False, False, True)
1253: 
1254:     if (_has_question_mark(doc)
1255:             or _has_interrogative_fronted(doc)
1256:             or _has_subject_aux_inversion(doc)):
1257:         return UtteranceClassification(
1258:             1, CoarseBin.QUESTION.value, "",
1259:             True, False, False, False, False)
1260: 
1261:     if _is_backchannel_structure(doc):
1262:         return UtteranceClassification(
1263:             2, CoarseBin.BACKCHANNEL.value, "",
1264:             False, False, True, False, False)
1265: 
1266:     if _is_imperative_structure(doc):
1267:         return UtteranceClassification(
1268:             3, CoarseBin.COMMAND.value, "",
1269:             False, True, False, False, False)
1270: 
1271:     if _is_emotion_structure(doc):
1272:         return UtteranceClassification(
1273:             4, CoarseBin.EMOTION.value, "",
1274:             False, False, False, True, True)
1275: 
1276:     return UtteranceClassification(
1277:         5, CoarseBin.STATEMENT.value, "",
1278:         False, False, False, False, True)
1279: 
1280: 
1281: # ---------------------------------------------------------------------------
1282: # Trace extraction -- the core of the engine
1283: # Spec Part 1: every field defined here
1284: # ---------------------------------------------------------------------------
1285: 
1286: _FIRST_PERSON = frozenset({
1287:     "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
1288: })
1289: 
1290: 
1291: def _extract_episodic(doc, root) -> str:
1292:     """Extract episodic trace: sentence text with subject stripped.
1293:     Spec Part 1, Field: episodic_fact.
1294:     "I have been researching adoption agencies lately" -> "researching adoption agencies lately"
1295:     """
1296:     sent_text = str(doc).strip()
1297:     if root is None:
1298:         return sent_text
1299: 
1300:     # Gerund/clausal subject as content: when nsubj is a csubj (gerund
1301:     # phrase) on a stative/linking ROOT, the subject IS the meaningful
1302:     # content. "Being transgender in a small town was isolating" â†’
1303:     # episodic = "Being transgender in a small town", not "isolating".
1304:     for child in root.children:
1305:         if child.dep_ == "csubj":
1306:             csubj_span = sorted(child.subtree, key=lambda t: t.i)
1307:             csubj_text = " ".join(t.text for t in csubj_span).strip()
1308:             if csubj_text:
1309:                 return csubj_text.rstrip(".,;:!?")
1310: 
1311:     # Find subject token â€” check root's children first, then all tokens
1312:     # (spaCy may attach nsubj to an auxpass rather than ROOT)
1313:     subj_tok = None
1314:     for child in root.children:
1315:         if child.dep_ in ("nsubj", "nsubjpass"):
1316:             subj_tok = child
1317:             break
1318:     if subj_tok is None:
1319:         for tok in doc:
1320:             if tok.dep_ in ("nsubj", "nsubjpass"):
1321:                 subj_tok = tok
1322:                 break
1323: 
1324:     if subj_tok is None:
1325:         return sent_text
1326: 
1327:     # Get the full subject span
1328:     subj_subtree = sorted(subj_tok.subtree, key=lambda t: t.i)
1329:     if not subj_subtree:
1330:         return sent_text
1331: 
1332:     last_subj_idx = subj_subtree[-1].i
1333: 
1334:     # Collect tokens after the subject span
1335:     remaining_tokens = [tok for tok in doc if tok.i > last_subj_idx]
1336:     if not remaining_tokens:
1337:         return sent_text
1338: 
1339:     # Build text preserving whitespace
1340:     result = ""
1341:     for tok in remaining_tokens:
1342:         result += tok.text
1343:         if tok.whitespace_:
1344:             result += tok.whitespace_
1345:     result = result.strip()
1346: 
1347:     # Strip leading auxiliaries: "have been researching" -> "researching"
1348:     # Use original doc's token POS tags instead of re-parsing the fragment
1349:     strip_count = 0
1350:     for tok in remaining_tokens:
1351:         if tok.pos_ == "AUX":
1352:             strip_count += 1
1353:         else:
1354:             break
1355:     if strip_count > 0:
1356:         kept_tokens = remaining_tokens[strip_count:]
1357:         if kept_tokens:
1358:             result = ""
1359:             for tok in kept_tokens:
1360:                 result += tok.text
1361:                 if tok.whitespace_:
1362:                     result += tok.whitespace_
1363:             result = result.strip()
1364:             remaining_tokens = kept_tokens
1365: 
1366:     # Copular/linking verb: the complement (acomp/attr) IS the episodic fact.
1367:     # Use the complement span directly instead of position-based stripping.
1368:     # Handles: "The sunday before 25 May 2023 was lovely" -> "lovely"
1369:     #          "What a beautiful painting that was!" -> "beautiful painting"
1370:     #          "I felt accepted as a transgender woman" -> "accepted as a transgender woman"
1371:     if root is not None and root.pos_ in ("AUX", "VERB"):
1372:         comp_tok = None
1373:         for child in root.children:
1374:             if child.dep_ in ("acomp", "attr"):
1375:                 comp_tok = child
1376:                 break
1377:         if comp_tok is not None and (
1378:             root.lemma_ in _COPULAR_LEMMAS or root.pos_ == "AUX"
1379:         ):
1380:             comp_span = sorted(comp_tok.subtree, key=lambda t: t.i)
1381:             comp_text = ""
1382:             for tok in comp_span:
1383:                 if tok.pos_ == "PUNCT":
1384:                     continue
1385:                 comp_text += tok.text
1386:                 if tok.whitespace_:
1387:                     comp_text += tok.whitespace_
1388:             comp_text = comp_text.strip()
1389:             if comp_text:
1390:                 result = comp_text
1391: 
1392:     # Strip trailing punctuation
1393:     result = result.rstrip(".,;:!?")
1394: 
1395:     return result if result else sent_text
1396: 
1397: 
1398: def _extract_emotional(doc, root) -> Tuple[Optional[str], Optional[float], Optional[str]]:
1399:     """Extract emotional trace: emotion adjective, valence, target.
1400:     Spec Part 1, Fields: emotional_state / emotional_valence / emotional_target.
1401:     Grammar reference Section 6: Emotional/Sentiment Markers.
1402: 
1403:     Detection order:
1404:         1. ADJ tokens in acomp/attr/oprd position
1405:         2. Passive past participles with copular auxpass
1406:         3. WordNet noun hypernym closure through feeling.n.01/emotion.n.01
1407:     """
1408:     emotion_adj = None
1409:     emotion_tok = None
1410: 
1411:     # 1. ADJ tokens in acomp/attr/oprd
1412:     for tok in doc:
1413:         if tok.pos_ == "ADJ" and tok.dep_ in ("acomp", "attr", "oprd"):
1414:             emotion_adj = tok.text.lower()
1415:             emotion_tok = tok
1416:             break
1417: 
1418:     # 2. Passive past participles as emotional states
1419:     # Guard: only extract emotion when nsubj is animate (PRON or PERSON NER).
1420:     # "I felt broken" â†’ emotional. "The window was broken" â†’ physical, not emotional.
1421:     if emotion_adj is None:
1422:         for tok in doc:
1423:             if (tok.tag_ == "VBN" and tok.dep_ == "ROOT"
1424:                     and any(c.dep_ == "auxpass" for c in tok.children)):
1425:                 # Animacy check on nsubj
1426:                 nsubj_tok = next(
1427:                     (c for c in tok.children
1428:                      if c.dep_ in ("nsubj", "nsubjpass")), None
1429:                 )
1430:                 if nsubj_tok is not None:
1431:                     is_animate = (
1432:                         nsubj_tok.pos_ == "PRON"
1433:                         or nsubj_tok.ent_type_ == "PERSON"
1434:                     )
1435:                     if not is_animate:
1436:                         break  # inanimate subject â†’ physical state, not emotion
1437:                 auxpass_tok = next(
1438:                     (c for c in tok.children if c.dep_ == "auxpass"), None
1439:                 )
1440:                 if auxpass_tok and (
1441:                     auxpass_tok.lemma_ in _COPULAR_LEMMAS
1442:                     or auxpass_tok.text.lower() in (
1443:                         "felt", "feels", "seemed", "looked", "sounded",
1444:                     )
1445:                 ):
1446:                     emotion_adj = tok.text.lower()
1447:                     emotion_tok = tok
1448:                     break
1449: 
1450:     # 3. Emotion nouns via WordNet hypernym closure
1451:     if emotion_adj is None:
1452:         try:
1453:             from nltk.corpus import wordnet as _wn
1454:             _emotion_synsets = {"feeling.n.01", "emotion.n.01", "state.n.04"}
1455:             for tok in doc:
1456:                 if tok.pos_ == "NOUN" and not tok.is_stop:
1457:                     for ss in _wn.synsets(tok.lemma_, pos="n"):
1458:                         hypernyms = {
1459:                             h.name() for h in ss.closure(lambda s: s.hypernyms())
1460:                         }
1461:                         if hypernyms & _emotion_synsets:
1462:                             emotion_adj = tok.lemma_.lower()
1463:                             emotion_tok = tok
1464:                             break
1465:                     if emotion_adj:
1466:                         break
1467:         except Exception:
1468:             pass
1469: 
1470:     if emotion_adj is None:
1471:         return (None, None, None)
1472: 
1473:     # Valence: check negation on the emotion token or its head
1474:     has_negation = any(child.dep_ == "neg" for child in emotion_tok.children)
1475:     if not has_negation and emotion_tok.head is not None:
1476:         has_negation = any(
1477:             child.dep_ == "neg" for child in emotion_tok.head.children
1478:         )
1479:     valence = float(not has_negation)
1480: 
1481:     # Target: pobj of prep child, or nsubj of head verb
1482:     target = None
1483:     for child in emotion_tok.children:
1484:         if child.dep_ == "prep":
1485:             for gc in child.children:
1486:                 if gc.dep_ == "pobj":
1487:                     target = gc.text
1488:                     break
1489:             if target is not None:
1490:                 break
1491:     if target is None and emotion_tok.head is not None:
1492:         for sibling in emotion_tok.head.children:
1493:             if sibling.dep_ == "nsubj":
1494:                 target = sibling.text
1495:                 break
1496: 
1497:     return (emotion_adj, valence, target)
1498: 
1499: 
1500: def _extract_temporal(doc, tense_aspect: TenseAspect) -> Tuple[str, Optional[str]]:
1501:     """Extract temporal trace: direction + expression.
1502:     Spec Part 1, Fields: temporal_direction, temporal_expression.
1503: 
1504:     Rules:
1505:         1. Collect DATE/TIME NER spans -- pass through as-is (no ISO)
1506:         2. Structural fallback: NUM + time_noun + ago
1507:         3. Perfect continuous with past = present (ongoing)
1508:         4. Xcomp with "to" mark = future (intent)
1509:         5. Temporal adverb override
1510:         6. "going to" + xcomp = future
1511:     """
1512:     _tense_to_direction = {
1513:         "past": "past", "present": "present", "future": "future",
1514:     }
1515:     direction = _tense_to_direction.get(tense_aspect.tense, "present")
1516: 
1517:     # Perfect continuous with past tense = ongoing from past to present
1518:     # Only perfect_continuous gets this override; plain "continuous" past
1519:     # (e.g. "I was running") is a completed past action, not ongoing.
1520:     if (tense_aspect.aspect == "perfect_continuous"
1521:             and direction == "past"):
1522:         direction = "present"
1523: 
1524:     root = _get_root(doc)
1525: 
1526:     # Xcomp with "to" mark = future intent
1527:     if root is not None and direction == "present":
1528:         for child in root.children:
1529:             if child.dep_ == "xcomp":
1530:                 for gc in child.children:
1531:                     if gc.dep_ in ("aux", "mark") and gc.lemma_ == "to":
1532:                         direction = "future"
1533:                         break
1534: 
1535:     # Temporal adverb override (Grammar reference Section 5)
1536:     for tok in doc:
1537:         if tok.pos_ == "ADV" and tok.dep_ in ("advmod", "npadvmod"):
1538:             lemma = tok.lemma_.lower()
1539:             if lemma in ("still", "currently"):
1540:                 direction = "present"
1541:             elif lemma in ("ago", "previously", "formerly", "once"):
1542:                 direction = "past"
1543:             elif lemma in ("soon", "eventually", "shortly"):
1544:                 direction = "future"
1545: 
1546:     # "going to" future detection
1547:     if root and root.lemma_ == "go" and root.tag_ == "VBG":
1548:         for child in root.children:
1549:             if child.dep_ == "xcomp" and child.tag_ == "VB":
1550:                 has_to = any(
1551:                     gc.dep_ == "aux" and gc.lemma_ == "to"
1552:                     for gc in child.children
1553:                 )
1554:                 if has_to or any(
1555:                     gc.dep_ == "mark" and gc.text == "to"
1556:                     for gc in child.children
1557:                 ):
1558:                     direction = "future"
1559:                     break
1560: 
1561:     # Collect DATE/TIME NER spans (Spec: pass through as-is, no ISO)
1562:     # M2 fix: expand NER span to include contextual tokens that form part of
1563:     # the full temporal expression (e.g., "the sunday before 25 May 2023").
1564:     # Strategy: find prep/advmod tokens whose object IS the NER span, then
1565:     # walk up to a nominal head (NOUN/PROPN) and include its subtree.
1566:     # Guard: never expand through verb heads to avoid grabbing whole clauses.
1567:     # Collect expanded (start, end) index spans, then merge overlaps.
1568:     _raw_spans: List[Tuple[int, int]] = []
1569:     for ent in doc.ents:
1570:         if ent.label_ not in ("DATE", "TIME"):
1571:             continue
1572:         left_boundary = ent.start
1573: 
1574:         # Step 1: Find a prep/advmod token left of the entity whose
1575:         # syntactic object (pobj/dobj) points into the NER span.
1576:         governing_prep = None
1577:         for tok in doc:
1578:             if tok.dep_ in ("prep", "advmod") and tok.i < ent.start:
1579:                 for child in tok.children:
1580:                     if (child.dep_ in ("pobj", "dobj")
1581:                             and child.i >= ent.start
1582:                             and child.i < ent.end):
1583:                         governing_prep = tok
1584:                         break
1585:                 # Adjacent prep whose head's subtree covers the entity
1586:                 if governing_prep is None and tok.i == ent.start - 1:
1587:                     if tok.dep_ == "prep":
1588:                         governing_prep = tok
1589:                 if governing_prep is not None:
1590:                     break
1591: 
1592:         if governing_prep is not None:
1593:             # Step 2: Walk up from the prep to its head â€” only if nominal
1594:             prep_head = governing_prep.head
1595:             if prep_head.pos_ in ("NOUN", "PROPN"):
1596:                 # Include the full subtree of the nominal head.
1597:                 subtree_tokens = sorted(prep_head.subtree, key=lambda t: t.i)
1598:                 left_boundary = min(t.i for t in subtree_tokens
1599:                                     if t.i <= ent.start)
1600:             else:
1601:                 # Head is a verb or other non-nominal â€” only include the
1602:                 # prep itself (e.g., "in 2022" keeps "in")
1603:                 left_boundary = governing_prep.i
1604:         else:
1605:             # No governing prep â€” simple left-walk for adjacent modifiers.
1606:             _CONTEXTUAL_DEPS = frozenset(
1607:                 {"det", "amod", "compound", "nummod"})
1608:             while left_boundary > 0:
1609:                 candidate = doc[left_boundary - 1]
1610:                 if candidate.dep_ in _CONTEXTUAL_DEPS:
1611:                     left_boundary -= 1
1612:                 elif (candidate.pos_ in ("NOUN", "PROPN")
1613:                       and candidate.dep_ in ("compound", "npadvmod", "nmod")):
1614:                     left_boundary -= 1
1615:                 else:
1616:                     break
1617: 
1618:         _raw_spans.append((left_boundary, ent.end))
1619: 
1620:     # Merge overlapping / contained spans so we don't duplicate text.
1621:     _raw_spans.sort()
1622:     merged_spans: List[Tuple[int, int]] = []
1623:     for start, end in _raw_spans:
1624:         if merged_spans and start <= merged_spans[-1][1]:
1625:             # Overlaps with previous â€” extend
1626:             merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
1627:         else:
1628:             merged_spans.append((start, end))
1629: 
1630:     date_time_spans: List[str] = [doc[s:e].text for s, e in merged_spans]
1631:     expression = " ".join(date_time_spans) if date_time_spans else None
1632: 
1633:     # Structural fallback: NUM + time_noun + "ago" pattern
1634:     # spaCy may miss these as NER.  Structural: nummod->NOUN->ADV(ago).
1635:     if expression is None:
1636:         _TIME_NOUNS = frozenset({
1637:             "year", "month", "week", "day", "hour", "minute",
1638:             "decade", "century", "semester", "quarter", "fortnight",
1639:         })
1640:         for tok in doc:
1641:             if tok.lemma_.lower() == "ago" and tok.pos_ == "ADV":
1642:                 head = tok.head
1643:                 if head.pos_ == "NOUN" and head.lemma_.lower() in _TIME_NOUNS:
1644:                     num_tok = None
1645:                     for child in head.children:
1646:                         if child.dep_ == "nummod" or child.pos_ == "NUM":
1647:                             num_tok = child
1648:                             break
1649:                     if num_tok:
1650:                         expression = f"{num_tok.text} {head.text} ago"
1651:                         direction = "past"
1652:                     else:
1653:                         subtree = sorted(head.subtree, key=lambda t: t.i)
1654:                         expr_tokens = [t.text for t in subtree] + ["ago"]
1655:                         expression = " ".join(expr_tokens)
1656:                         direction = "past"
1657:                     break
1658: 
1659:     # Gap 15: Frequency adverbs â€” closed grammatical class, enriches temporal trace
1660:     _FREQUENCY_ADVERBS = frozenset({
1661:         "always", "never", "often", "usually", "sometimes", "rarely",
1662:         "daily", "weekly", "monthly", "yearly", "annually",
1663:         "frequently", "seldom", "occasionally", "regularly",
1664:     })
1665:     for tok in doc:
1666:         if tok.dep_ == "advmod" and tok.lemma_.lower() in _FREQUENCY_ADVERBS:
1667:             freq = tok.text.lower()
1668:             if expression:
1669:                 expression = f"{freq} {expression}"
1670:             else:
1671:                 expression = freq
1672:             break  # one frequency adverb per clause
1673: 
1674:     return (direction, expression)
1675: 
1676: 
1677: def _extract_relational(
1678:     doc, speaker: Optional[str], listener: str = "user",
1679: ) -> Tuple[str, List[str], str]:
1680:     """Extract relational trace: subject, entities, type.
1681:     Spec Part 1, Fields: relational_subject, relational_entities.
1682: 
1683:     Rules:
1684:         1. First-person pronouns -> subject is speaker
1685:         2. Second-person pronouns (as nsubj) -> subject is listener
1686:         3. Collect PERSON/ORG/GPE/LOC/FAC/NORP NER entities
1687:         Speaker name is appended downstream by memory.py _prepare_row.
1688:     """
1689:     relational_subject = speaker or "user"
1690: 
1691:     # Find the nsubj token to determine person
1692:     _THIRD_PERSON_PRONOUNS = frozenset({
1693:         "she", "he", "they", "it", "her", "him", "them",
1694:     })
1695:     nsubj_tok = None
1696:     for tok in doc:
1697:         if tok.dep_ in ("nsubj", "nsubjpass"):
1698:             nsubj_tok = tok
1699:             break
1700: 
1701:     if nsubj_tok is not None:
1702:         nsubj_lower = nsubj_tok.text.lower()
1703:         if nsubj_lower in _FIRST_PERSON and speaker:
1704:             # First-person -> resolve to speaker
1705:             relational_subject = speaker
1706:         elif nsubj_tok.pos_ == "PRON" and nsubj_lower in _THIRD_PERSON_PRONOUNS:
1707:             # Third-person pronoun -> keep as-is, do NOT default to speaker
1708:             # Gap 8: Dummy "it" â€” weather/impersonal verbs produce a
1709:             # meaningless "it" subject. Detect and set to empty string.
1710:             _DUMMY_IT_LEMMAS = frozenset({
1711:                 "rain", "snow", "hail", "sleet", "drizzle", "thunder",
1712:                 "pour", "seem", "appear",
1713:             })
1714:             _root = _get_root(doc)
1715:             if nsubj_lower == "it" and _root is not None and _root.lemma_.lower() in _DUMMY_IT_LEMMAS:
1716:                 relational_subject = ""
1717:             else:
1718:                 relational_subject = nsubj_tok.text
1719:         elif nsubj_tok.pos_ == "PROPN":
1720:             # Proper noun -> use its text
1721:             # Collect full proper-noun span (multi-token names)
1722:             span_tokens = [nsubj_tok]
1723:             for left in nsubj_tok.lefts:
1724:                 if left.pos_ == "PROPN" and left.dep_ == "compound":
1725:                     span_tokens.insert(0, left)
1726:             for right in nsubj_tok.rights:
1727:                 if right.pos_ == "PROPN" and right.dep_ == "flat":
1728:                     span_tokens.append(right)
1729:             relational_subject = " ".join(t.text for t in span_tokens)
1730:         elif nsubj_tok.pos_ == "NOUN":
1731:             # Common noun subject (e.g., "The window was broken")
1732:             # Use the noun span text, not the speaker default
1733:             relational_subject = _span_text(nsubj_tok)
1734:         elif nsubj_tok.pos_ == "PRON":
1735:             # Gap 7: Any remaining PRON not caught above (indefinite pronouns
1736:             # like everyone, someone, nobody, anything, etc.) â€” keep as-is
1737:             # instead of defaulting to speaker.
1738:             relational_subject = nsubj_tok.text
1739:         # else: no nsubj match above -> keep default (speaker)
1740:     else:
1741:         # No nsubj at all -> keep default (speaker)
1742:         has_first_person = any(
1743:             tok.text.lower() in _FIRST_PERSON for tok in doc
1744:         )
1745:         if has_first_person and speaker:
1746:             relational_subject = speaker
1747: 
1748:     # Second-person subject detection: "You moved to Portland" -> listener
1749:     _SECOND_PERSON_SUBJ = frozenset({"you"})
1750:     has_second_person_subj = any(
1751:         tok.text.lower() in _SECOND_PERSON_SUBJ
1752:         and tok.dep_ in ("nsubj", "nsubjpass")
1753:         for tok in doc
1754:     )
1755:     if has_second_person_subj:
1756:         relational_subject = listener
1757: 
1758:     entities: List[str] = []
1759:     seen: set = set()
1760:     for ent in doc.ents:
1761:         if (ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC", "NORP")
1762:                 and ent.text not in seen):
1763:             entities.append(ent.text)
1764:             seen.add(ent.text)
1765: 
1766:     # Gap 1: Indirect objects (dative/iobj) that are PROPN or PERSON NER
1767:     # may not appear in doc.ents.  "I gave Sarah the book" -> Sarah.
1768:     for tok in doc:
1769:         if tok.dep_ in ("dative", "iobj"):
1770:             if tok.pos_ == "PROPN" or tok.ent_type_ == "PERSON":
1771:                 name = _span_text(tok)
1772:                 if name not in seen:
1773:                     entities.append(name)
1774:                     seen.add(name)
1775: 
1776:     # Gap 2: Causative dobj as participant entity.
1777:     # "She made him cry" â€” ROOT has xcomp/ccomp child (causative), dobj is participant.
1778:     # spaCy may parse the caused-entity as dobj of ROOT or nsubj of the complement.
1779:     root_tok = _get_root(doc)
1780:     if root_tok is not None:
1781:         for comp_child in root_tok.children:
1782:             if comp_child.dep_ in ("xcomp", "ccomp") and comp_child.pos_ == "VERB":
1783:                 # Check dobj of ROOT
1784:                 for child in root_tok.children:
1785:                     if child.dep_ == "dobj" and child.pos_ in ("PRON", "PROPN"):
1786:                         name = child.text
1787:                         if name not in seen:
1788:                             entities.append(name)
1789:                             seen.add(name)
1790:                 # Also check nsubj of the complement (spaCy's ECM parse)
1791:                 for gc in comp_child.children:
1792:                     if gc.dep_ == "nsubj" and gc.pos_ in ("PRON", "PROPN"):
1793:                         name = gc.text
1794:                         if name not in seen:
1795:                             entities.append(name)
1796:                             seen.add(name)
1797:                 break  # only process first complement
1798: 
1799:         # Gap 3: Factitive oprd â€” "They elected him chairman".
1800:         # When both dobj and oprd exist, dobj is the affected person.
1801:         has_oprd = any(c.dep_ == "oprd" for c in root_tok.children)
1802:         if has_oprd:
1803:             for child in root_tok.children:
1804:                 if child.dep_ == "dobj" and child.pos_ in ("PRON", "PROPN"):
1805:                     name = child.text
1806:                     if name not in seen:
1807:                         entities.append(name)
1808:                         seen.add(name)
1809: 
1810:     # Gap 20: Vocatives â€” addressee detection enriches relational trace
1811:     for tok in doc:
1812:         if tok.dep_ == "vocative" or (
1813:             tok.pos_ == "PROPN"
1814:             and tok.dep_ in ("npadvmod", "appos", "ROOT", "dep")
1815:             and tok.i == 0
1816:             and tok.nbor(1).text == ","
1817:             if tok.i + 1 < len(doc) else False
1818:         ):
1819:             name = _span_text(tok)
1820:             if name not in seen:
1821:                 entities.append(name)
1822:                 seen.add(name)
1823: 
1824:     # Gap 22: Compound subjects â€” split conjoined nsubj into separate entities
1825:     if nsubj_tok is not None:
1826:         for conj_child in nsubj_tok.children:
1827:             if conj_child.dep_ == "conj" and conj_child.pos_ in ("PROPN", "NOUN"):
1828:                 name = _span_text(conj_child)
1829:                 if name not in seen:
1830:                     entities.append(name)
1831:                     seen.add(name)
1832:         # Also add the nsubj itself if it's a PROPN not yet in entities
1833:         if nsubj_tok.pos_ == "PROPN":
1834:             name = _span_text(nsubj_tok)
1835:             if name not in seen:
1836:                 entities.append(name)
1837:                 seen.add(name)
1838: 
1839:     # Gap 23: Passive "by" agent â€” extract pobj of agent dep
1840:     for tok in doc:
1841:         if tok.dep_ == "agent" and tok.head.tag_ in ("VBN", "VBD"):
1842:             for child in tok.children:
1843:                 if child.dep_ == "pobj":
1844:                     name = _span_text(child)
1845:                     if name not in seen:
1846:                         entities.append(name)
1847:                         seen.add(name)
1848: 
1849:     return (relational_subject, entities, "personal")
1850: 
1851: 
1852: # ---------------------------------------------------------------------------
1853: # Schematic trace -- WordNet noun-to-schema mapping
1854: # Spec Part 1, Field: edge_schematic_category (steps 4-7)
1855: # ---------------------------------------------------------------------------
1856: 
1857: _NOUN_SCHEMA_ANCHORS: list[tuple[frozenset, str]] = [
1858:     (frozenset({
1859:         "occupation.n.01", "position.n.01", "job.n.01",
1860:         "profession.n.01", "employment.n.01", "work.n.01",
1861:         "service.n.01", "promotion.n.02", "advancement.n.03",
1862:     }), "career"),
1863:     (frozenset({
1864:         "family_relationship.n.01", "adoption.n.01",
1865:         "relative.n.01", "kinship.n.01",
1866:     }), "family"),
1867:     (frozenset({
1868:         "illness.n.01", "injury.n.01", "symptom.n.01",
1869:         "disease.n.01",
1870:     }), "health"),
1871:     (frozenset({
1872:         "creation.n.02", "artistic_creation.n.01",
1873:         "art.n.01", "sport.n.01", "game.n.01",
1874:         "recreation.n.01", "diversion.n.01",
1875:         "outdoor_recreation.n.01", "hobby.n.01",
1876:     }), "hobby"),
1877:     (frozenset({
1878:         "educational_institution.n.01", "course.n.01",
1879:         "school.n.01",
1880:     }), "education"),
1881: ]
1882: 
1883: 
1884: @functools.lru_cache(maxsize=4096)
1885: def _is_kinship_noun(lemma: str) -> bool:
1886:     """Check if a noun lemma is a kinship term via WordNet hypernym closure.
1887:     Returns True if any synset of the lemma is a hyponym of kinship anchors:
1888:     {relative.n.01, parent.n.01, sibling.n.01, spouse.n.01, child.n.02,
1889:      family_member.n.01, ancestor.n.01, grandparent.n.01}.
1890:     Structural check -- no word lists."""
1891:     try:
1892:         _ensure_wordnet()
1893:         from nltk.corpus import wordnet as _wn
1894: 
1895:         _kinship_anchors = frozenset({
1896:             "relative.n.01", "parent.n.01", "sibling.n.01", "spouse.n.01",
1897:             "child.n.02", "family_member.n.01", "ancestor.n.01",
1898:             "grandparent.n.01",
1899:         })
1900: 
1901:         for ss in _wn.synsets(lemma, pos="n"):
1902:             hypernyms = {h.name() for h in ss.closure(lambda s: s.hypernyms())}
1903:             hypernyms.add(ss.name())
1904:             if hypernyms & _kinship_anchors:
1905:                 return True
1906:         return False
1907:     except Exception:
1908:         return False
1909: 
1910: 
1911: @functools.lru_cache(maxsize=4096)
1912: def _noun_to_schema_via_wordnet(lemma: str) -> Optional[str]:
1913:     """Map a noun lemma to a schema via WordNet hypernym closure.
1914:     Spec Part 1, Field: schematic_category (step 6).
1915:     Grammar reference Section 8.1: structural noun detection."""
1916:     try:
1917:         _ensure_wordnet()
1918:         from nltk.corpus import wordnet as _wn
1919:         noun_synsets = _wn.synsets(lemma, pos="n")
1920:         if not noun_synsets:
1921:             return None
1922: 
1923:         for ss in noun_synsets:
1924:             hypernyms = {h.name() for h in ss.closure(lambda s: s.hypernyms())}
1925:             hypernyms.add(ss.name())
1926:             for anchors, schema in _NOUN_SCHEMA_ANCHORS:
1927:                 if hypernyms & anchors:
1928:                     return schema
1929:         return None
1930:     except Exception:
1931:         return None
1932: 
1933: 
1934: def _extract_schematic(doc, root, verb_class: VerbClass) -> str:
1935:     """Extract schematic trace: category from verb class + NER + WordNet refinement.
1936: 
1937:     Spec Part 1, Field: edge_schematic_category.  Priority order:
1938:         1. Xcomp/ccomp override for intent/preference/location verbs
1939:         2. Light verb delegation (do/have/take/make/give/get -> use dobj)
1940:         3. Verb class -> schema map
1941:         4. Kinship noun override (WordNet hypernym closure)
1942:         5. NER refinement (ORG->career, GPE->housing, etc.)
1943:         6. Noun hypernym fallback for uncategorized/experience
1944:         7. Creative/recreational verb check (WordNet)
1945:     """
1946:     # Step 1: xcomp/ccomp override for intent/preference/location verbs
1947:     _LIGHT_VERB_LEMMAS = frozenset({"do", "have", "take", "make", "give", "get"})
1948:     if (root is not None
1949:             and verb_class in (
1950:                 VerbClass.PREFERENCE, VerbClass.UNKNOWN,
1951:                 VerbClass.BE, VerbClass.HAVE, VerbClass.LOCATION,
1952:             )
1953:             and root.pos_ in ("VERB", "AUX")):
1954:         for child in root.children:
1955:             if child.dep_ in ("xcomp", "ccomp") and child.pos_ == "VERB":
1956:                 # Step 2: light verb delegation
1957:                 if child.lemma_ in _LIGHT_VERB_LEMMAS:
1958:                     for gc in child.children:
1959:                         if gc.dep_ == "dobj":
1960:                             dobj_schema = _noun_to_schema_via_wordnet(gc.lemma_)
1961:                             if dobj_schema is not None:
1962:                                 return dobj_schema
1963:                 else:
1964:                     # Non-light complement verb: trust its verb class
1965:                     comp_vc = classify_verb_class(child.lemma_)
1966:                     comp_schema = _VERB_CLASS_TO_SCHEMA.get(
1967:                         comp_vc, "uncategorized",
1968:                     )
1969:                     if comp_schema not in ("uncategorized", "identity"):
1970:                         return comp_schema
1971:                     for gc in child.children:
1972:                         if gc.dep_ == "dobj":
1973:                             dobj_schema = _noun_to_schema_via_wordnet(gc.lemma_)
1974:                             if dobj_schema is not None:
1975:                                 return dobj_schema
1976: 
1977:     # Step 2b: root-level light verb delegation
1978:     # "She got a promotion" -> root=got, dobj=promotion -> delegate to "promotion"
1979:     if (root is not None
1980:             and root.pos_ in ("VERB", "AUX")
1981:             and root.lemma_ in _LIGHT_VERB_LEMMAS):
1982:         for child in root.children:
1983:             if child.dep_ == "dobj":
1984:                 dobj_schema = _noun_to_schema_via_wordnet(child.lemma_)
1985:                 if dobj_schema is not None:
1986:                     return dobj_schema
1987: 
1988:     # Step 3: verb class -> schema
1989:     schema = _VERB_CLASS_TO_SCHEMA.get(verb_class, "uncategorized")
1990: 
1991:     # Step 4: kinship noun override (Grammar reference Section 2.1/7.1)
1992:     # Do not override strong verb-class signals (career, health, finance, housing)
1993:     if schema not in ("career", "health", "finance", "housing"):
1994:         try:
1995:             _ensure_wordnet()
1996:             from nltk.corpus import wordnet as _wn
1997:             _kinship_anchors = frozenset({
1998:                 "relative.n.01", "parent.n.01", "grandparent.n.01",
1999:                 "sibling.n.01", "child.n.02", "spouse.n.01",
2000:                 "kinsman.n.01", "ancestor.n.01",
2001:             })
2002:             # Only check kinship nouns in subject or direct object position,
2003:             # not in prepositional phrases. "Celebrate with my family" is
2004:             # about celebration, not family relationships.
2005:             _kinship_deps = frozenset({"nsubj", "nsubjpass", "dobj", "attr"})
2006:             for tok in doc:
2007:                 if (tok.pos_ == "NOUN" and not tok.is_stop
2008:                         and tok.dep_ in _kinship_deps):
2009:                     for ss in _wn.synsets(tok.lemma_, pos="n"):
2010:                         hypernyms = {
2011:                             h.name()
2012:                             for h in ss.closure(lambda s: s.hypernyms())
2013:                         }
2014:                         if hypernyms & _kinship_anchors:
2015:                             schema = "family"
2016:                             break
2017:                     if schema == "family":
2018:                         break
2019:         except Exception:
2020:             pass
2021: 
2022:     if schema == "family":
2023:         return schema
2024: 
2025:     # Step 5: NER refinement for generic schemas
2026:     # Guard: only trust NER when entity root POS is PROPN. spaCy mis-tags
2027:     # common nouns as GPE/ORG on re-parsed fragments (e.g., "interview" â†’ GPE).
2028:     if schema in ("uncategorized", "identity", "planning"):
2029:         doc_ner_labels = frozenset(
2030:             ent.label_ for ent in doc.ents
2031:             if ent.root.pos_ == "PROPN"
2032:         )
2033:         if "ORG" in doc_ner_labels:
2034:             schema = "career"
2035:         elif doc_ner_labels & frozenset({"GPE", "FAC"}):
2036:             schema = "housing"
2037:         elif "EVENT" in doc_ner_labels:
2038:             schema = "experience"
2039:         elif "MONEY" in doc_ner_labels:
2040:             schema = "finance"
2041:         elif "NORP" in doc_ner_labels:
2042:             schema = "social"
2043:         elif "LAW" in doc_ner_labels:
2044:             schema = "legal"
2045:         elif "WORK_OF_ART" in doc_ner_labels:
2046:             schema = "culture"
2047:         elif "PRODUCT" in doc_ner_labels:
2048:             schema = "commercial"
2049:         elif "QUANTITY" in doc_ner_labels:
2050:             schema = "measurement"
2051: 
2052:     # Step 6: noun hypernym fallback (Grammar reference Section 8.1/3.5)
2053:     if schema in ("uncategorized", "experience"):
2054:         for tok in doc:
2055:             if tok.pos_ == "NOUN" and not tok.is_stop:
2056:                 noun_schema = _noun_to_schema_via_wordnet(tok.lemma_)
2057:                 if noun_schema is not None:
2058:                     schema = noun_schema
2059:                     break
2060: 
2061:         # Step 7: creative/recreational verb check
2062:         if (schema in ("uncategorized", "experience")
2063:                 and root is not None
2064:                 and root.pos_ == "VERB"):
2065:             try:
2066:                 _ensure_wordnet()
2067:                 from nltk.corpus import wordnet as _wn
2068:                 _creative_verb_anchors = frozenset({
2069:                     "create.v.03", "create.v.05",
2070:                 })
2071:                 _recreational_verb_anchors = frozenset({
2072:                     "play.v.01", "play.v.03",
2073:                     "swim.v.01", "camp.v.01",
2074:                 })
2075:                 for ss in _wn.synsets(root.lemma_, pos=_wn.VERB):
2076:                     hypernyms = {
2077:                         h.name() for h in ss.closure(lambda s: s.hypernyms())
2078:                     }
2079:                     hypernyms.add(ss.name())
2080:                     if hypernyms & _creative_verb_anchors:
2081:                         schema = "hobby"
2082:                         break
2083:                     if hypernyms & _recreational_verb_anchors:
2084:                         schema = "hobby"
2085:                         break
2086:             except Exception:
2087:                 pass
2088: 
2089:     # Possessive-subject family detection
2090:     # Only triggers when the head noun of the subject is a kinship term
2091:     # (validated via WordNet hypernym closure).
2092:     if schema in ("uncategorized", "identity", "planning"):
2093:         if root is not None:
2094:             for child in root.children:
2095:                 if child.dep_ in ("nsubj", "nsubjpass"):
2096:                     for gc in child.children:
2097:                         if gc.dep_ == "poss":
2098:                             # Validate that the subject head noun is kinship
2099:                             if _is_kinship_noun(child.lemma_):
2100:                                 schema = "family"
2101:                             break
2102: 
2103:     return schema
2104: 
2105: 
2106: # ---------------------------------------------------------------------------
2107: # _build_trace_decomposition -- DEPRECATED (M5 audit 2026-04-29)
2108: # Zero production callers. Only referenced by tests and scripts.
2109: # Diverges from primary path (_extract_traces_from_sentence): no reported
2110: # speech handling, no compound splitting, no inline conditional detection.
2111: # Tests should migrate to _extract_traces_from_sentence. Do NOT add new
2112: # callers -- use _extract_traces_from_sentence instead.
2113: # ---------------------------------------------------------------------------
2114: 
2115: def _build_trace_decomposition(
2116:     triple: Triple,
2117:     doc,
2118:     speaker: Optional[str],
2119:     tense_aspect: TenseAspect,
2120:     verb_class: VerbClass,
2121:     emotion: Optional[str] = None,
2122:     listener: str = "user",
2123: ) -> TraceDecomposition:
2124:     """DEPRECATED: Build a TraceDecomposition from a Triple and its parse context.
2125: 
2126:     This function is kept for backward compatibility -- tests import it.
2127:     It diverges from the primary extraction path (_extract_traces_from_sentence)
2128:     and should NOT be used in new code. See VIOLATION M5 notes above.
2129:     """
2130:     root = _get_root(doc)
2131: 
2132:     pred_verb = triple.predicate.split("_")[0] if triple.predicate else ""
2133:     if pred_verb:
2134:         refined = classify_verb_class(pred_verb)
2135:         if refined not in (VerbClass.BE, VerbClass.HAVE, VerbClass.UNKNOWN):
2136:             verb_class = refined
2137:         # If refinement returns UNKNOWN, keep the caller's verb_class
2138: 
2139:     episodic_fact = f"{triple.predicate} {triple.object}".strip()
2140:     if not episodic_fact:
2141:         episodic_fact = str(doc).strip()
2142: 
2143:     significance = _compute_significance(verb_class, tense_aspect)
2144: 
2145:     temporal_direction, temporal_expression = _extract_temporal(doc, tense_aspect)
2146:     relational_subject, relational_entities, _ = _extract_relational(doc, speaker, listener=listener)
2147:     relational_type = _VERB_CLASS_TO_RELTYPE.get(verb_class, "personal")
2148:     schematic_category = _extract_schematic(doc, root, verb_class)
2149: 
2150:     emotional_state = emotion
2151:     emotional_valence = None
2152:     emotional_target = None
2153:     if emotional_state is None:
2154:         emotional_state, emotional_valence, emotional_target = (
2155:             _extract_emotional(doc, root)
2156:         )
2157:     else:
2158:         _, emotional_valence, emotional_target = _extract_emotional(doc, root)
2159: 
2160:     return TraceDecomposition(
2161:         episodic_fact=episodic_fact,
2162:         episodic_significance=significance,
2163:         emotional_state=emotional_state,
2164:         emotional_valence=emotional_valence,
2165:         emotional_target=emotional_target,
2166:         temporal_direction=temporal_direction,
2167:         temporal_expression=temporal_expression,
2168:         relational_subject=relational_subject,
2169:         relational_entities=relational_entities,
2170:         relational_type=relational_type,
2171:         schematic_category=schematic_category,
2172:         source_text=str(doc),
2173:         utterance_type=triple.utterance_type,
2174:         mood=triple.mood,
2175:         negated=triple.negated,
2176:         is_historical=triple.is_historical,
2177:         subject=triple.subject,
2178:         predicate=triple.predicate,
2179:         object=triple.object,
2180:         extraction_rule=triple.extraction_rule,
2181:     )
2182: 
2183: 
2184: # ---------------------------------------------------------------------------
2185: # _extract_traces_from_sentence -- the core extraction path
2186: # Spec Part 1: all 5 traces from a single sentence
2187: # ---------------------------------------------------------------------------
2188: 
2189: def _find_content_verb(verb, _depth=0):
2190:     """Universal frame skipper: walk past framing verbs to the content.
2191: 
2192:     Grammar reference pp. 212-220, 283-295: control/raising verbs.
2193: 
2194:     Rule: skip ROOT to its xcomp/ccomp child IF:
2195:       - ccomp â†’ skip only if current is SPEECH class or "be" (cleft)
2196:       - xcomp â†’ skip only if ROOT has no dobj (subject control)
2197:                 AND ROOT is not PREFERENCE class
2198: 
2199:     Handles: speech verbs, intent verbs, phase verbs, clefts,
2200:     causative guards, "used to", "ended up", "keeps telling" â€” all
2201:     in one recursive walk. No special cases.
2202:     """
2203:     if verb is None or _depth >= 4:
2204:         return verb
2205:     complement = None
2206:     for child in verb.children:
2207:         if child.dep_ == "ccomp" and child.pos_ in ("VERB", "AUX"):
2208:             complement = child
2209:             break
2210:         if child.dep_ == "xcomp" and child.pos_ in ("VERB", "AUX"):
2211:             complement = child
2212:             break
2213:     if complement is None:
2214:         return verb
2215:     if complement.dep_ == "ccomp":
2216:         _cur_vc = classify_verb_class(verb.lemma_)
2217:         if _cur_vc == VerbClass.SPEECH or verb.lemma_ == "be":
2218:             return _find_content_verb(complement, _depth + 1)
2219:         return verb
2220:     has_dobj = any(c.dep_ == "dobj" for c in verb.children)
2221:     if has_dobj:
2222:         _cur_vc = classify_verb_class(verb.lemma_)
2223:         if _cur_vc != VerbClass.SPEECH:
2224:             return verb
2225:     _vc = classify_verb_class(verb.lemma_)
2226:     if _vc == VerbClass.PREFERENCE:
2227:         return verb
2228:     return _find_content_verb(complement, _depth + 1)
2229: 
2230: 
2231: def _extract_traces_from_sentence(
2232:     sent_doc,
2233:     speaker: Optional[str],
2234:     tense_aspect: TenseAspect,
2235:     listener: str = "user",
2236: ) -> TraceDecomposition:
2237:     """Extract all 5 traces from a single sentence doc.
2238:     Spec Part 2, Statement extraction path."""
2239:     root = _get_root(sent_doc)
2240:     source_text = str(sent_doc).strip()
2241: 
2242:     content_root = _find_content_verb(root) if root else root
2243: 
2244:     # Phrasal verb: content_root + particle (dep=prt)
2245:     particle = None
2246:     if content_root:
2247:         for child in content_root.children:
2248:             if child.dep_ == "prt":
2249:                 particle = child.lemma_
2250:                 break
2251: 
2252:     # Verb class classification â€” on the CONTENT verb, not the frame
2253:     verb_class = VerbClass.UNKNOWN
2254:     if content_root and content_root.pos_ in ("VERB", "AUX"):
2255:         if particle:
2256:             phrasal = f"{content_root.lemma_}_{particle}"
2257:             verb_class = classify_verb_class(phrasal)
2258:             if verb_class == VerbClass.UNKNOWN:
2259:                 verb_class = classify_verb_class(content_root.lemma_)
2260:         else:
2261:             verb_class = classify_verb_class(content_root.lemma_)
2262:         verb_class = _reclassify_location_by_object(
2263:             sent_doc, content_root, verb_class,
2264:         )
2265:         verb_class = _detect_planning_pattern(
2266:             sent_doc, content_root, verb_class,
2267:         )
2268: 
2269:     # 1. Episodic trace â€” extract from content verb's perspective
2270:     episodic_fact = _extract_episodic(sent_doc, content_root)
2271: 
2272:     # Grammatical object â€” from content verb
2273:     gram_object = _extract_grammatical_object(sent_doc, content_root)
2274: 
2275:     # 4. Relational trace
2276:     relational_subject, relational_entities, relational_type_base = (
2277:         _extract_relational(sent_doc, speaker, listener=listener)
2278:     )
2279: 
2280:     # Reported speech: if frame-skipper jumped past a SPEECH verb,
2281:     # the relational_subject should be the EMBEDDED clause's subject,
2282:     # not the reporter. _extract_relational found ROOT's nsubj (the
2283:     # reporter). We need to correct it to the embedded subject.
2284:     _embedded_predicate_verb = (
2285:         content_root if content_root is not root else None
2286:     )
2287:     if _embedded_predicate_verb and root:
2288:         _root_vc = classify_verb_class(root.lemma_)
2289:         if _root_vc == VerbClass.SPEECH:
2290:             # Find embedded clause's nsubj
2291:             embedded_subj = None
2292:             for child in root.children:
2293:                 if child.dep_ in ("ccomp", "xcomp") and child.pos_ in ("VERB", "AUX"):
2294:                     for gc in child.children:
2295:                         if gc.dep_ in ("nsubj", "nsubjpass"):
2296:                             embedded_subj = gc.text
2297:                             break
2298:                     break
2299:             if embedded_subj:
2300:                 if embedded_subj not in relational_entities:
2301:                     relational_entities.append(embedded_subj)
2302:                 # Set relational_subject to the embedded subject:
2303:                 # "Caroline said SHE moved" â†’ subj = "Caroline" (reporter
2304:                 #   is the referent of "she" in reported speech)
2305:                 # "Caroline said I need help" â†’ subj = speaker (first person)
2306:                 if embedded_subj.lower() in _FIRST_PERSON:
2307:                     relational_subject = speaker or "user"
2308:                 elif embedded_subj.lower() in (
2309:                     "she", "he", "they", "it", "her", "him", "them",
2310:                 ):
2311:                     # Third-person pronoun in reported speech â†’ reporter
2312:                     # is the likely referent ("Caroline said SHE moved")
2313:                     main_nsubj_tok = None
2314:                     for rc in root.children:
2315:                         if rc.dep_ in ("nsubj", "nsubjpass"):
2316:                             main_nsubj_tok = rc
2317:                             break
2318:                     if main_nsubj_tok and main_nsubj_tok.pos_ == "PROPN":
2319:                         relational_subject = _span_text(main_nsubj_tok)
2320:                     elif main_nsubj_tok:
2321:                         for ent in sent_doc.ents:
2322:                             if (ent.label_ == "PERSON"
2323:                                     and ent.start <= main_nsubj_tok.i < ent.end):
2324:                                 relational_subject = ent.text
2325:                                 break
2326:                 else:
2327:                     # Named embedded subject ("The doctor said Sam needs...")
2328:                     relational_subject = embedded_subj
2329: 
2330:     # Invariant 1: object is ALWAYS a noun phrase, never a full sentence.
2331:     # Do NOT fall back to episodic_fact -- leave empty if no NP extracted.
2332: 
2333:     # 2. Emotional trace
2334:     emotional_state, emotional_valence, emotional_target = (
2335:         _extract_emotional(sent_doc, root)
2336:     )
2337: 
2338:     # 3. Temporal trace
2339:     temporal_direction, temporal_expression = (
2340:         _extract_temporal(sent_doc, tense_aspect)
2341:     )
2342: 
2343:     relational_type = _VERB_CLASS_TO_RELTYPE.get(
2344:         verb_class, relational_type_base,
2345:     )
2346: 
2347:     # 5. Schematic trace â€” from content verb, not frame
2348:     schematic_category = _extract_schematic(sent_doc, content_root, verb_class)
2349: 
2350:     # Significance (stative vs dynamic via Grammar Gap #6)
2351:     significance = _compute_significance(verb_class, tense_aspect)
2352: 
2353:     # Grammar Gap #14: Emphatic do-support detection.
2354:     # "I do love chocolate" â€” "do" as aux in a declarative affirmative
2355:     # sentence signals emphasis. Override significance to "emphatic".
2356:     # Note: spaCy may parse the main verb as NOUN (e.g., "love" -> NN),
2357:     # so we check any root that has an aux "do" child.
2358:     if root and root.pos_ in ("VERB", "NOUN", "AUX"):
2359:         for child in root.children:
2360:             if (child.dep_ == "aux"
2361:                     and child.lemma_.lower() == "do"
2362:                     and child.tag_ in ("VBP", "VBZ", "VBD")):
2363:                 # Confirm declarative (not question) and affirmative (not neg)
2364:                 _is_question = any(
2365:                     t.text == "?" for t in sent_doc
2366:                 )
2367:                 _is_negated = any(
2368:                     c.dep_ == "neg" for c in root.children
2369:                 )
2370:                 if not _is_question and not _is_negated:
2371:                     significance = "emphatic"
2372:                 break
2373: 
2374:     # Grammatical features
2375:     mood = detect_mood(sent_doc)
2376:     negated = detect_negation(sent_doc)
2377:     is_historical = tense_aspect.tense == "past"
2378: 
2379:     # Conditional detection (Spec Part 1, Field: edge_mood)
2380:     # Grammar reference: Conditionals p. 602, 927-932
2381:     is_conditional = False
2382:     for tok in sent_doc:
2383:         if tok.dep_ == "mark" and tok.lemma_.lower() in (
2384:             "if", "unless", "whether",
2385:         ):
2386:             is_conditional = True
2387:             break
2388:         if (tok.dep_ == "aux"
2389:                 and tok.lemma_ in ("would", "could")
2390:                 and tok.head.pos_ == "VERB"):
2391:             is_conditional = True
2392:             break
2393: 
2394:     if is_conditional:
2395:         mood = "conditional"
2396: 
2397:     # Derive predicate (Spec Part 1, Field: predicate)
2398:     # content_root already points to the content verb (frame-skipper did
2399:     # the delegation). No special cases needed â€” just use content_root.
2400:     _pred_root = content_root
2401:     root_lemma = _pred_root.lemma_ if _pred_root else ""
2402: 
2403:     # Gap 4: Light verb predicate delegation.
2404:     # "took a shower" -> pred=shower instead of pred=take.
2405:     _LIGHT_VERB_PRED = frozenset({"do", "have", "take", "make", "give", "get"})
2406:     if (_pred_root and root_lemma.lower() in _LIGHT_VERB_PRED
2407:             and not particle):
2408:         for child in _pred_root.children:
2409:             if child.dep_ == "dobj":
2410:                 root_lemma = child.lemma_
2411:                 break
2412: 
2413:     # Particle already computed from content_root (line above)
2414:     predicate = (f"{root_lemma}_{particle}" if particle else root_lemma).lower()
2415: 
2416:     # Lemmatizer fallback: when spaCy's lemma equals the surface form on an
2417:     # inflected verb (VBD/VBG/VBN/VBZ), the lemmatizer failed on the fragment.
2418:     # Use WordNet morphy as fallback.
2419:     if (_pred_root and predicate == _pred_root.text.lower()
2420:             and _pred_root.tag_ in ("VBD", "VBG", "VBN", "VBZ")):
2421:         try:
2422:             _ensure_wordnet()
2423:             from nltk.corpus import wordnet as _wn_lemma
2424:             _morph_result = _wn_lemma.morphy(predicate, _wn_lemma.VERB)
2425:             if _morph_result:
2426:                 predicate = _morph_result
2427:         except Exception:
2428:             pass  # graceful fallback: keep surface form
2429: 
2430:     # Append prep frame: "move" -> "move_from"
2431:     # Skip preps whose pobj is a temporal expression (DATE/TIME NER),
2432:     # e.g., "moved on Tuesday from Sweden" -> "move_from" not "move_on".
2433:     if _pred_root:
2434:         for child in _pred_root.children:
2435:             if child.dep_ == "prep" and child.pos_ == "ADP":
2436:                 _pobj_is_temporal = False
2437:                 for gc in child.children:
2438:                     if gc.dep_ == "pobj":
2439:                         _pobj_ner = {
2440:                             t.ent_type_ for t in gc.subtree if t.ent_type_
2441:                         }
2442:                         if _pobj_ner & frozenset({"DATE", "TIME"}):
2443:                             _pobj_is_temporal = True
2444:                         break
2445:                 if _pobj_is_temporal:
2446:                     continue  # skip temporal prep, try next
2447:                 predicate = f"{predicate}_{child.lemma_.lower()}"
2448:                 break
2449: 
2450:     return TraceDecomposition(
2451:         episodic_fact=episodic_fact,
2452:         episodic_significance=significance,
2453:         emotional_state=emotional_state,
2454:         emotional_valence=emotional_valence,
2455:         emotional_target=emotional_target,
2456:         temporal_direction=temporal_direction,
2457:         temporal_expression=temporal_expression,
2458:         relational_subject=relational_subject,
2459:         relational_entities=relational_entities,
2460:         relational_type=relational_type,
2461:         schematic_category=schematic_category,
2462:         source_text=source_text,
2463:         utterance_type=0,
2464:         mood=mood,
2465:         negated=negated,
2466:         is_historical=is_historical,
2467:         subject=relational_subject,
2468:         predicate=predicate,
2469:         object=gram_object,
2470:         extraction_rule="trace",
2471:     )
2472: 
2473: 
2474: # ---------------------------------------------------------------------------
2475: # Unified imposed-trace builder (module-level)
2476: # All imposed-fact paths delegate here so every trace gets all 5 extractors.
2477: # ---------------------------------------------------------------------------
2478: 
2479: def _build_imposed_trace(
2480:     clause_text: str,
2481:     imposed_subject: str,
2482:     speaker: Optional[str],
2483:     tense_aspect: "TenseAspect",
2484:     listener: str = "user",
2485:     mood: str = "indicative",
2486:     extraction_rule: str = "imposed",
2487:     schematic_hint: Optional[str] = None,
2488:     predicate_override: Optional[str] = None,
2489:     utterance_type: int = 0,
2490:     source_text_override: Optional[str] = None,
2491: ) -> Optional["TraceDecomposition"]:
2492:     """Build a fully-traced TraceDecomposition for an imposed fact.
2493: 
2494:     Calls all 5 extractors (episodic, emotional, temporal, relational, schematic).
2495:     Used by _extract_imposed_facts and conditional clause handling.
2496:     """
2497:     clause_text = clause_text.strip()
2498:     if not clause_text or len(clause_text) < 3:
2499:         return None
2500: 
2501:     frag_nlp = _get_nlp_fragment()
2502:     clause_doc = frag_nlp(clause_text)
2503:     clause_root = _get_root(clause_doc)
2504: 
2505:     verb_class = VerbClass.UNKNOWN
2506:     if clause_root and clause_root.pos_ in ("VERB", "AUX"):
2507:         verb_class = classify_verb_class(clause_root.lemma_)
2508:         verb_class = _reclassify_location_by_object(
2509:             clause_doc, clause_root, verb_class,
2510:         )
2511: 
2512:     # 1. Episodic trace
2513:     episodic = _extract_episodic(clause_doc, clause_root)
2514:     gram_obj = _extract_grammatical_object(clause_doc, clause_root)
2515: 
2516:     # 2. Emotional trace
2517:     emotional_state, emotional_valence, emotional_target = (
2518:         _extract_emotional(clause_doc, clause_root)
2519:     )
2520: 
2521:     # 3. Temporal trace
2522:     temporal_direction, temporal_expression = _extract_temporal(
2523:         clause_doc, tense_aspect,
2524:     )
2525: 
2526:     # 4. Relational trace
2527:     rel_subject, rel_entities, rel_type_base = _extract_relational(
2528:         clause_doc, speaker, listener=listener,
2529:     )
2530:     if imposed_subject and imposed_subject != "user":
2531:         rel_subject = imposed_subject
2532:     rel_type = _VERB_CLASS_TO_RELTYPE.get(verb_class, rel_type_base)
2533: 
2534:     # 5. Schematic trace
2535:     schema = schematic_hint or _extract_schematic(
2536:         clause_doc, clause_root, verb_class,
2537:     )
2538: 
2539:     significance = _compute_significance(verb_class, tense_aspect)
2540: 
2541:     # Predicate derivation
2542:     if predicate_override:
2543:         root_lemma = predicate_override
2544:     else:
2545:         root_lemma = clause_root.lemma_ if clause_root else ""
2546:         if clause_root and clause_root.pos_ == "AUX":
2547:             for child in clause_root.children:
2548:                 if child.dep_ in ("xcomp", "ccomp", "acomp"):
2549:                     if child.pos_ in ("VERB", "ADJ"):
2550:                         root_lemma = child.lemma_
2551:                         break
2552: 
2553:     return TraceDecomposition(
2554:         episodic_fact=episodic,
2555:         episodic_significance=significance,
2556:         emotional_state=emotional_state,
2557:         emotional_valence=emotional_valence,
2558:         emotional_target=emotional_target,
2559:         temporal_direction=temporal_direction,
2560:         temporal_expression=temporal_expression,
2561:         relational_subject=rel_subject,
2562:         relational_entities=rel_entities,
2563:         relational_type=rel_type,
2564:         schematic_category=schema,
2565:         source_text=source_text_override or clause_text,
2566:         utterance_type=utterance_type,
2567:         mood=mood,
2568:         negated=detect_negation(clause_doc),
2569:         is_historical=tense_aspect.tense == "past",
2570:         subject=rel_subject,
2571:         predicate=root_lemma.lower(),
2572:         object=gram_obj,
2573:         extraction_rule=extraction_rule,
2574:     )
2575: 
2576: 
2577: # ---------------------------------------------------------------------------
2578: # Imposed facts extraction
2579: # Spec Part 2, Multi-clause Sentences + imposed-facts-taxonomy.md
2580: # ---------------------------------------------------------------------------
2581: 
2582: def _extract_imposed_facts(
2583:     doc,
2584:     speaker: Optional[str],
2585:     tense_aspect: TenseAspect,
2586:     existing_sources: Optional[frozenset] = None,
2587:     listener: str = "user",
2588: ) -> List[TraceDecomposition]:
2589:     """Extract imposed facts from subordinate constructions in a single pass.
2590: 
2591:     Scans every token once, dispatches on dep_/tag_ to detect
2592:     construction types from the imposed-facts taxonomy:
2593:         relcl (9,10), advcl (17), appos (19), acl (18),
2594:         csubj (14), expl (24), gerund (21), ccomp/direct speech (12),
2595:         correlative (26), comparative (27).
2596:     """
2597:     imposed: List[TraceDecomposition] = []
2598:     _existing = existing_sources or frozenset()
2599: 
2600:     def _is_duplicate(clause_text: str) -> bool:
2601:         ct = clause_text.strip().lower()
2602:         if not ct:
2603:             return True
2604:         for existing in _existing:
2605:             el = existing.strip().lower()
2606:             if ct in el or el in ct:
2607:                 return True
2608:         return False
2609: 
2610:     def _build_imposed(
2611:         clause_text: str,
2612:         imposed_subject: str,
2613:         mood: str = "indicative",
2614:         extraction_rule: str = "imposed",
2615:         schematic_hint: Optional[str] = None,
2616:         predicate_override: Optional[str] = None,
2617:         skip_dedup: bool = False,
2618:     ) -> Optional[TraceDecomposition]:
2619:         clause_text = clause_text.strip()
2620:         if not clause_text or len(clause_text) < 3:
2621:             return None
2622:         if not skip_dedup and _is_duplicate(clause_text):
2623:             return None
2624: 
2625:         return _build_imposed_trace(
2626:             clause_text=clause_text,
2627:             imposed_subject=imposed_subject,
2628:             speaker=speaker,
2629:             tense_aspect=tense_aspect,
2630:             listener=listener,
2631:             mood=mood,
2632:             extraction_rule=extraction_rule,
2633:             schematic_hint=schematic_hint,
2634:             predicate_override=predicate_override,
2635:         )
2636: 
2637:     # Single pass over all tokens
2638:     seen_subtree_starts: set = set()
2639: 
2640:     for tok in doc:
2641:         if tok.i in seen_subtree_starts:
2642:             continue
2643: 
2644:         dep = tok.dep_
2645:         pos = tok.pos_
2646:         tag = tok.tag_
2647: 
2648:         # --- Type 9/10: Relative clauses (dep=relcl) ---
2649:         # Grammar Gap #13: Restrictive vs Non-restrictive relative clauses.
2650:         # Non-restrictive are set off by commas ("My brother, who lives in
2651:         # Paris") and carry stronger presupposition (background fact).
2652:         # Restrictive have no commas ("The man who called") and are defining.
2653:         if dep == "relcl" and pos in ("VERB", "AUX"):
2654:             subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
2655:             imposed_subject = (
2656:                 tok.head.text if tok.head else (speaker or "user")
2657:             )
2658:             _REL_PRONOUNS = frozenset({
2659:                 "that", "which", "who", "whom", "whose",
2660:             })
2661:             filtered = [
2662:                 t.text for t in subtree_sorted
2663:                 if t.text.lower() not in _REL_PRONOUNS
2664:             ]
2665:             clean_clause = " ".join(filtered).strip()
2666:             imposed_text = f"{imposed_subject} {clean_clause}".strip()
2667: 
2668:             # Detect restrictive vs non-restrictive: check for comma
2669:             # immediately before the relative clause span.
2670:             _is_nonrestrictive = False
2671:             relcl_start = min(t.i for t in tok.subtree)
2672:             # Walk backwards to find the token just before the relcl span
2673:             # (skipping relative pronouns that precede the verb).
2674:             # A comma before the relcl region signals non-restrictive.
2675:             if relcl_start > 0:
2676:                 prev_tok = doc[relcl_start - 1]
2677:                 if prev_tok.text == ",":
2678:                     _is_nonrestrictive = True
2679:                 # Sometimes the relative pronoun is at relcl_start and the
2680:                 # comma is one more token back.
2681:                 elif (prev_tok.text.lower() in _REL_PRONOUNS
2682:                         and relcl_start > 1
2683:                         and doc[relcl_start - 2].text == ","):
2684:                     _is_nonrestrictive = True
2685:             # Also check if the head noun itself is followed by a comma
2686:             if tok.head and tok.head.i + 1 < len(doc):
2687:                 next_after_head = doc[tok.head.i + 1]
2688:                 if next_after_head.text == ",":
2689:                     _is_nonrestrictive = True
2690: 
2691:             relcl_rule = ("imposed_relcl_nonrestrictive" if _is_nonrestrictive
2692:                           else "imposed_relcl_restrictive")
2693: 
2694:             decomp = _build_imposed(
2695:                 imposed_text, imposed_subject,
2696:                 extraction_rule=relcl_rule,
2697:                 predicate_override=tok.lemma_,
2698:             )
2699:             if decomp is not None:
2700:                 imposed.append(decomp)
2701:                 seen_subtree_starts.add(tok.i)
2702: 
2703:         # --- Type 17: Adverbial clauses (dep=advcl) ---
2704:         elif dep == "advcl" and pos in ("VERB", "AUX"):
2705:             mark_tok = None
2706:             for child in tok.children:
2707:                 if child.dep_ == "mark":
2708:                     mark_tok = child
2709:                     break
2710:             # spaCy sometimes labels temporal subordinators (when, where)
2711:             # as advmod rather than mark â€” check advmod children too.
2712:             if mark_tok is None:
2713:                 _SUBORDINATOR_ADVMODS = frozenset({
2714:                     "when", "whenever", "where", "wherever", "while",
2715:                     "once", "before", "after",
2716:                 })
2717:                 for child in tok.children:
2718:                     if (child.dep_ == "advmod"
2719:                             and child.lemma_.lower() in _SUBORDINATOR_ADVMODS):
2720:                         mark_tok = child
2721:                         break
2722:             # spaCy sometimes assigns dep=aux to infinitive "to" particle
2723:             # in purpose clauses (e.g., "to buy" -> "to" has dep=aux)
2724:             if mark_tok is None:
2725:                 for child in tok.children:
2726:                     if (child.dep_ == "aux" and child.lemma_ == "to"
2727:                             and child.tag_ == "TO"):
2728:                         mark_tok = child
2729:                         break
2730:             mark_lemma = mark_tok.lemma_.lower() if mark_tok else ""
2731: 
2732:             # --- Types 5-8: Conditional clauses (if/unless/whether/provided) ---
2733:             if mark_lemma in ("if", "unless", "whether", "provided"):
2734:                 # Classify conditional type by tense pattern
2735:                 advcl_verb = tok
2736:                 advcl_tag = advcl_verb.tag_  # VBP, VBD, VBN, etc.
2737: 
2738:                 # Check for "had + VBN" pattern (third conditional)
2739:                 has_had_vbn = False
2740:                 if advcl_tag == "VBN":
2741:                     for child in advcl_verb.children:
2742:                         if (child.dep_ == "aux" and
2743:                                 child.lemma_.lower() == "have" and
2744:                                 child.tag_ == "VBD"):
2745:                             has_had_vbn = True
2746:                             break
2747: 
2748:                 # Check main clause for modal auxiliaries
2749:                 main_verb = tok.head
2750:                 main_modal = ""
2751:                 if main_verb:
2752:                     for child in main_verb.children:
2753:                         if child.dep_ == "aux" and child.tag_ == "MD":
2754:                             main_modal = child.lemma_.lower()
2755:                             break
2756: 
2757:                 # Determine conditional type
2758:                 if has_had_vbn and main_modal in ("would", "could", "might"):
2759:                     cond_type = "third"
2760:                     cond_mood = "conditional"
2761:                     cond_is_historical = True
2762:                 elif advcl_tag == "VBD" and main_modal in (
2763:                     "would", "could", "might",
2764:                 ):
2765:                     cond_type = "second"
2766:                     cond_mood = "conditional"
2767:                     cond_is_historical = False
2768:                 elif advcl_tag in ("VBP", "VBZ") and main_modal in (
2769:                     "will", "shall",
2770:                 ):
2771:                     cond_type = "first"
2772:                     cond_mood = "indicative"
2773:                     cond_is_historical = False
2774:                 elif advcl_tag in ("VBP", "VBZ") and main_modal == "":
2775:                     cond_type = "zero"
2776:                     cond_mood = "indicative"
2777:                     cond_is_historical = False
2778:                 else:
2779:                     # Default: first conditional (most common)
2780:                     cond_type = "first"
2781:                     cond_mood = "indicative"
2782:                     cond_is_historical = False
2783: 
2784:                 # Extract subject of the conditional clause
2785:                 cond_subj = speaker or "user"
2786:                 for child in tok.children:
2787:                     if child.dep_ in ("nsubj", "nsubjpass"):
2788:                         cond_subj = child.text
2789:                         break
2790: 
2791:                 # Build clause text without the mark token
2792:                 subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
2793:                 filtered = [
2794:                     t.text for t in subtree_sorted if t.i != mark_tok.i
2795:                 ]
2796:                 clean_text = " ".join(filtered).strip()
2797: 
2798:                 extraction_rule = f"imposed_conditional_{cond_type}"
2799: 
2800:                 # Build the imposed fact via unified builder
2801:                 clause_text = clean_text.strip()
2802:                 if clause_text and len(clause_text) >= 3 and not _is_duplicate(clause_text):
2803:                     decomp = _build_imposed_trace(
2804:                         clause_text=clause_text,
2805:                         imposed_subject=cond_subj,
2806:                         speaker=speaker,
2807:                         tense_aspect=tense_aspect,
2808:                         listener=listener,
2809:                         mood=cond_mood,
2810:                         extraction_rule=extraction_rule,
2811:                         predicate_override=tok.lemma_.lower(),
2812:                     )
2813:                     if decomp is not None:
2814:                         # Override is_historical for conditional-specific semantics
2815:                         decomp.is_historical = cond_is_historical
2816:                         imposed.append(decomp)
2817:                     seen_subtree_starts.add(tok.i)
2818: 
2819:             elif mark_lemma == "to":
2820:                 # --- Type 22: Infinitive purpose clauses ---
2821:                 # "I went to the store to buy groceries" -> purpose = "buy groceries"
2822:                 # The purpose clause presupposes the entities exist.
2823: 
2824:                 # Skip "used to" habitual construction
2825:                 if (tok.head.lemma_ == "use" and tok.head.tag_ == "VBD"):
2826:                     continue
2827: 
2828:                 subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
2829: 
2830:                 # Determine the subject: inherit from main clause subject
2831:                 purpose_subj = speaker or "user"
2832:                 main_verb = tok.head
2833:                 if main_verb:
2834:                     for child in main_verb.children:
2835:                         if child.dep_ in ("nsubj", "nsubjpass"):
2836:                             purpose_subj = child.text
2837:                             break
2838: 
2839:                 # Build clause text without the "to" mark
2840:                 filtered = [
2841:                     t.text for t in subtree_sorted if t.i != mark_tok.i
2842:                 ]
2843:                 clean_text = " ".join(filtered).strip()
2844: 
2845:                 # Extract grammatical object of the purpose verb directly
2846:                 # from the original parse (more reliable than re-parsing)
2847:                 purpose_obj = ""
2848:                 for child in tok.children:
2849:                     if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
2850:                         obj_subtree = sorted(child.subtree, key=lambda t: t.i)
2851:                         purpose_obj = " ".join(t.text for t in obj_subtree)
2852:                         break
2853:                 if not purpose_obj:
2854:                     # Try prep object
2855:                     for child in tok.children:
2856:                         if child.dep_ == "prep":
2857:                             for grandchild in child.children:
2858:                                 if grandchild.dep_ == "pobj":
2859:                                     obj_subtree = sorted(
2860:                                         grandchild.subtree, key=lambda t: t.i,
2861:                                     )
2862:                                     purpose_obj = " ".join(
2863:                                         t.text for t in obj_subtree
2864:                                     )
2865:                                     break
2866:                             break
2867: 
2868:                 decomp = _build_imposed(
2869:                     clean_text, purpose_subj,
2870:                     extraction_rule="imposed_infinitive_purpose",
2871:                     predicate_override=tok.lemma_,
2872:                     skip_dedup=True,
2873:                 )
2874:                 if decomp is not None:
2875:                     # Override object with what we extracted directly
2876:                     if purpose_obj:
2877:                         decomp.object = purpose_obj
2878:                     imposed.append(decomp)
2879:                     seen_subtree_starts.add(tok.i)
2880: 
2881:             # --- Type 20: Absolute phrases (noun + participle, no mark) ---
2882:             # "The sun having set, we went home" â€” parsed as advcl with
2883:             # VBG/VBN head, no subordinating conjunction, own nsubj that
2884:             # differs from the main clause nsubj. Scene-setting presupposed.
2885:             elif (not mark_tok and tag in ("VBG", "VBN")):
2886:                 # Check for own nsubj (absolute phrases have an independent
2887:                 # subject different from the main clause).
2888:                 abs_subj = None
2889:                 for child in tok.children:
2890:                     if child.dep_ in ("nsubj", "nsubjpass"):
2891:                         abs_subj = child.text
2892:                         break
2893:                 # Find main clause nsubj for comparison
2894:                 main_subj = None
2895:                 main_verb = tok.head
2896:                 if main_verb:
2897:                     for child in main_verb.children:
2898:                         if child.dep_ in ("nsubj", "nsubjpass"):
2899:                             main_subj = child.text
2900:                             break
2901:                 # Absolute phrase: has own subject AND it differs from main
2902:                 if (abs_subj is not None
2903:                         and (main_subj is None
2904:                              or abs_subj.lower() != main_subj.lower())):
2905:                     subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
2906:                     clause_text = " ".join(t.text for t in subtree_sorted)
2907:                     imposed_text = f"{abs_subj} {tok.lemma_}".strip()
2908:                     # Use full clause for richer context
2909:                     full_clause = f"{abs_subj} {clause_text}".strip()
2910:                     decomp = _build_imposed(
2911:                         full_clause, abs_subj,
2912:                         extraction_rule="imposed_absolute_phrase",
2913:                         predicate_override=tok.lemma_,
2914:                     )
2915:                     if decomp is not None:
2916:                         imposed.append(decomp)
2917:                         seen_subtree_starts.add(tok.i)
2918:                 else:
2919:                     # VBG/VBN advcl without mark but no independent subject â€”
2920:                     # treat as regular participial advcl.
2921:                     subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
2922:                     advcl_subj = speaker or "user"
2923:                     for child in tok.children:
2924:                         if child.dep_ in ("nsubj", "nsubjpass"):
2925:                             advcl_subj = child.text
2926:                             break
2927:                     clean_text = " ".join(t.text for t in subtree_sorted)
2928:                     decomp = _build_imposed(
2929:                         clean_text, advcl_subj,
2930:                         extraction_rule="imposed_advcl_general",
2931:                     )
2932:                     if decomp is not None:
2933:                         imposed.append(decomp)
2934:                         seen_subtree_starts.add(tok.i)
2935: 
2936:             elif not mark_tok:
2937:                 # advcl with no mark and not VBG/VBN â€” general advcl
2938:                 subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
2939:                 advcl_subj = speaker or "user"
2940:                 for child in tok.children:
2941:                     if child.dep_ in ("nsubj", "nsubjpass"):
2942:                         advcl_subj = child.text
2943:                         break
2944:                 clean_text = " ".join(t.text for t in subtree_sorted)
2945:                 decomp = _build_imposed(
2946:                     clean_text, advcl_subj,
2947:                     extraction_rule="imposed_advcl_general",
2948:                 )
2949:                 if decomp is not None:
2950:                     imposed.append(decomp)
2951:                     seen_subtree_starts.add(tok.i)
2952: 
2953:             else:
2954:                 # Non-conditional, non-purpose advcl â€” classify by mark lemma.
2955:                 # Grammar pp.902-906: subordinating conjunctions are a closed
2956:                 # grammatical class grouped by function.
2957:                 _ADVCL_TIME = frozenset({
2958:                     "when", "whenever", "while", "before", "after",
2959:                     "since", "until", "once", "till", "as soon as",
2960:                 })
2961:                 _ADVCL_PLACE = frozenset({
2962:                     "where", "wherever", "everywhere", "anywhere",
2963:                 })
2964:                 _ADVCL_REASON = frozenset({
2965:                     "because", "as", "since", "so", "for",
2966:                     "now that", "given that", "in that",
2967:                 })
2968:                 _ADVCL_MANNER = frozenset({
2969:                     "like", "as if", "as though", "the way", "than",
2970:                 })
2971:                 _ADVCL_CONTRAST = frozenset({
2972:                     "though", "although", "even though", "whereas",
2973:                     "even if", "while", "whilst",
2974:                 })
2975: 
2976:                 # Classify advcl type from the mark token's lemma
2977:                 if mark_lemma in _ADVCL_TIME:
2978:                     advcl_type = "time"
2979:                 elif mark_lemma in _ADVCL_PLACE:
2980:                     advcl_type = "place"
2981:                 elif mark_lemma in _ADVCL_REASON:
2982:                     advcl_type = "reason"
2983:                 elif mark_lemma in _ADVCL_MANNER:
2984:                     advcl_type = "manner"
2985:                 elif mark_lemma in _ADVCL_CONTRAST:
2986:                     advcl_type = "contrast"
2987:                 else:
2988:                     advcl_type = "general"
2989: 
2990:                 subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
2991:                 advcl_subj = speaker or "user"
2992:                 for child in tok.children:
2993:                     if child.dep_ in ("nsubj", "nsubjpass"):
2994:                         advcl_subj = child.text
2995:                         break
2996: 
2997:                 if mark_tok:
2998:                     filtered = [
2999:                         t.text for t in subtree_sorted if t.i != mark_tok.i
3000:                     ]
3001:                     clean_text = " ".join(filtered).strip()
3002:                 else:
3003:                     clean_text = " ".join(t.text for t in subtree_sorted)
3004: 
3005:                 extraction_rule = f"imposed_advcl_{advcl_type}"
3006: 
3007:                 # Classified advcl types carry semantic value beyond the
3008:                 # main trace (reason, contrast, time, etc.), so skip dedup
3009:                 # to ensure the classification is preserved.
3010:                 _skip = advcl_type != "general"
3011: 
3012:                 decomp = _build_imposed(
3013:                     clean_text, advcl_subj,
3014:                     extraction_rule=extraction_rule,
3015:                     skip_dedup=_skip,
3016:                 )
3017:                 if decomp is not None:
3018:                     # Time advcl: promote clause content as temporal expression
3019:                     if advcl_type == "time" and clean_text:
3020:                         decomp.temporal_expression = clean_text
3021:                     imposed.append(decomp)
3022:                     seen_subtree_starts.add(tok.i)
3023: 
3024:         # --- Type 22: Infinitive purpose via xcomp (dep=xcomp with "to") ---
3025:         # "She exercises to stay healthy" -> "stay" is xcomp, not advcl
3026:         # Distinguish from subject-control: "I want to go" (control verb)
3027:         elif (dep == "xcomp" and pos == "VERB" and tag == "VB"):
3028:             # Check if this xcomp has a "to" particle child
3029:             has_to = False
3030:             to_tok = None
3031:             for child in tok.children:
3032:                 if child.lemma_ == "to" and child.tag_ == "TO":
3033:                     has_to = True
3034:                     to_tok = child
3035:                     break
3036:             if has_to and tok.head and tok.head.pos_ == "VERB":
3037:                 # Skip "used to" habitual construction
3038:                 if (tok.head.lemma_ == "use" and tok.head.tag_ == "VBD"):
3039:                     continue
3040: 
3041:                 # Skip "be going to" periphrastic future â€” not a purpose clause
3042:                 _is_going_to_future = False
3043:                 if tok.head.lemma_ == "go" and tok.head.tag_ == "VBG":
3044:                     for sibling in tok.head.children:
3045:                         if sibling.dep_ == "aux" and sibling.lemma_ == "be":
3046:                             _is_going_to_future = True
3047:                             break
3048:                 # Check if head verb is a control/framing verb via the
3049:                 # universal frame skipper. If _find_content_verb skips
3050:                 # past the head to this xcomp, it's subject control
3051:                 # (not purpose). If it stays on the head, head IS
3052:                 # the content verb and this xcomp is purpose.
3053:                 _content = _find_content_verb(tok.head)
3054:                 _is_control = (_content != tok.head)
3055:                 if (not _is_going_to_future and not _is_control):
3056:                     # This is a purpose xcomp â€” extract as imposed fact
3057:                     subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
3058: 
3059:                     purpose_subj = speaker or "user"
3060:                     for child in tok.head.children:
3061:                         if child.dep_ in ("nsubj", "nsubjpass"):
3062:                             purpose_subj = child.text
3063:                             break
3064: 
3065:                     # Build text without "to"
3066:                     filtered = [
3067:                         t.text for t in subtree_sorted
3068:                         if not (t.i == to_tok.i)
3069:                     ]
3070:                     clean_text = " ".join(filtered).strip()
3071: 
3072:                     # Extract object from the purpose verb
3073:                     purpose_obj = ""
3074:                     for child in tok.children:
3075:                         if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
3076:                             obj_subtree = sorted(
3077:                                 child.subtree, key=lambda t: t.i,
3078:                             )
3079:                             purpose_obj = " ".join(
3080:                                 t.text for t in obj_subtree
3081:                             )
3082:                             break
3083: 
3084:                     decomp = _build_imposed(
3085:                         clean_text, purpose_subj,
3086:                         extraction_rule="imposed_infinitive_purpose",
3087:                         predicate_override=tok.lemma_,
3088:                         skip_dedup=True,
3089:                     )
3090:                     if decomp is not None:
3091:                         if purpose_obj:
3092:                             decomp.object = purpose_obj
3093:                         imposed.append(decomp)
3094:                         seen_subtree_starts.add(tok.i)
3095: 
3096:         # --- Type 19: Appositives (dep=appos) ---
3097:         elif dep == "appos":
3098:             entity = tok.head.text if tok.head else ""
3099:             subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
3100:             description = " ".join(t.text for t in subtree_sorted)
3101:             if entity and description:
3102:                 imposed_text = f"{entity} is {description}"
3103:                 decomp = _build_imposed(
3104:                     imposed_text, entity,
3105:                     extraction_rule="imposed_appos",
3106:                     schematic_hint="identity",
3107:                     predicate_override="be",
3108:                 )
3109:                 if decomp is not None:
3110:                     imposed.append(decomp)
3111:                     seen_subtree_starts.add(tok.i)
3112: 
3113:         # --- Type 18: Participial phrases (dep=acl, VBG/VBN) ---
3114:         elif dep == "acl" and tag in ("VBG", "VBN"):
3115:             subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
3116:             clause_text = " ".join(t.text for t in subtree_sorted)
3117:             acl_subject = (
3118:                 tok.head.text if tok.head else (speaker or "user")
3119:             )
3120:             imposed_text = f"{acl_subject} {clause_text}".strip()
3121:             decomp = _build_imposed(
3122:                 imposed_text, acl_subject,
3123:                 extraction_rule="imposed_acl",
3124:                 predicate_override=tok.lemma_,
3125:             )
3126:             if decomp is not None:
3127:                 imposed.append(decomp)
3128:                 seen_subtree_starts.add(tok.i)
3129: 
3130:         # --- Type 14: Noun clauses as subject (dep=csubj) ---
3131:         elif dep in ("csubj", "csubjpass") and pos in ("VERB", "AUX"):
3132:             head_vc = VerbClass.UNKNOWN
3133:             if tok.head and tok.head.pos_ in ("VERB", "AUX"):
3134:                 head_vc = classify_verb_class(tok.head.lemma_)
3135:             if head_vc == VerbClass.SPEECH:
3136:                 continue
3137: 
3138:             subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
3139:             filtered = [
3140:                 t.text for t in subtree_sorted
3141:                 if not (t.dep_ == "mark" and t.lemma_.lower() == "that")
3142:             ]
3143:             clause_text = " ".join(filtered).strip()
3144: 
3145:             csubj_subject = speaker or "user"
3146:             for child in tok.children:
3147:                 if child.dep_ in ("nsubj", "nsubjpass"):
3148:                     csubj_subject = child.text
3149:                     break
3150: 
3151:             decomp = _build_imposed(
3152:                 clause_text, csubj_subject,
3153:                 extraction_rule="imposed_csubj",
3154:             )
3155:             if decomp is not None:
3156:                 imposed.append(decomp)
3157:                 seen_subtree_starts.add(tok.i)
3158: 
3159:         # --- Type 24: Existential there (dep=expl) ---
3160:         elif dep == "expl" and tok.text.lower() == "there":
3161:             be_verb = tok.head
3162:             if be_verb and be_verb.lemma_ == "be":
3163:                 content_tok = None
3164:                 for child in be_verb.children:
3165:                     if child.dep_ in ("attr", "nsubj"):
3166:                         content_tok = child
3167:                         break
3168:                 if content_tok is not None:
3169:                     subtree_sorted = sorted(
3170:                         content_tok.subtree, key=lambda t: t.i,
3171:                     )
3172:                     content_text = " ".join(
3173:                         t.text for t in subtree_sorted
3174:                     )
3175:                     preps = []
3176:                     for child in be_verb.children:
3177:                         if child.dep_ == "prep":
3178:                             prep_subtree = sorted(
3179:                                 child.subtree, key=lambda t: t.i,
3180:                             )
3181:                             preps.append(
3182:                                 " ".join(t.text for t in prep_subtree)
3183:                             )
3184:                     full_text = content_text
3185:                     if preps:
3186:                         full_text = f"{content_text} {' '.join(preps)}"
3187: 
3188:                     decomp = _build_imposed(
3189:                         full_text, speaker or "user",
3190:                         extraction_rule="imposed_expl",
3191:                     )
3192:                     if decomp is not None:
3193:                         imposed.append(decomp)
3194:                         seen_subtree_starts.add(tok.i)
3195: 
3196:         # --- Type 21: Gerund phrases (VBG as nsubj/dobj/pobj) ---
3197:         elif tag == "VBG" and dep in ("nsubj", "dobj", "pobj"):
3198:             subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
3199:             clause_text = " ".join(t.text for t in subtree_sorted)
3200:             gerund_subj = speaker or "user"
3201:             for child in tok.children:
3202:                 if child.dep_ in ("poss", "nsubj"):
3203:                     gerund_subj = child.text
3204:                     break
3205: 
3206:             decomp = _build_imposed(
3207:                 clause_text, gerund_subj,
3208:                 extraction_rule="imposed_gerund",
3209:             )
3210:             if decomp is not None:
3211:                 imposed.append(decomp)
3212:                 seen_subtree_starts.add(tok.i)
3213: 
3214:         # --- Type 15: Subjunctive wish (Grammar pp.735-739) ---
3215:         # "I wish I spoke French" -> wish clause is counterfactual
3216:         elif dep == "ccomp" and pos in ("VERB", "AUX") and tok.head and tok.head.lemma_ == "wish":
3217:             subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
3218:             wish_subj = speaker or "user"
3219:             for child in tok.children:
3220:                 if child.dep_ in ("nsubj", "nsubjpass"):
3221:                     wish_subj = child.text
3222:                     break
3223:             clause_text = " ".join(t.text for t in subtree_sorted).strip()
3224: 
3225:             decomp = _build_imposed(
3226:                 clause_text, wish_subj,
3227:                 mood="subjunctive",
3228:                 extraction_rule="imposed_subjunctive_wish",
3229:                 predicate_override=tok.lemma_,
3230:                 skip_dedup=True,
3231:             )
3232:             if decomp is not None:
3233:                 imposed.append(decomp)
3234:                 seen_subtree_starts.add(tok.i)
3235: 
3236:         # --- Type 12: Direct speech (ccomp with quotation marks) ---
3237:         elif dep == "ccomp" and pos in ("VERB", "AUX"):
3238:             has_quotes = any(
3239:                 t.text in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019")
3240:                 for t in doc
3241:             )
3242:             head_vc = VerbClass.UNKNOWN
3243:             if tok.head and tok.head.pos_ in ("VERB", "AUX"):
3244:                 head_vc = classify_verb_class(tok.head.lemma_)
3245:             if head_vc == VerbClass.SPEECH:
3246:                 continue
3247: 
3248:             if has_quotes:
3249:                 subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
3250:                 filtered = [
3251:                     t.text for t in subtree_sorted
3252:                     if t.text not in (
3253:                         '"', "'", "\u201c", "\u201d", "\u2018", "\u2019",
3254:                     )
3255:                 ]
3256:                 clause_text = " ".join(filtered).strip()
3257:                 ccomp_subj = speaker or "user"
3258:                 for child in tok.children:
3259:                     if child.dep_ in ("nsubj", "nsubjpass"):
3260:                         ccomp_subj = child.text
3261:                         break
3262: 
3263:                 decomp = _build_imposed(
3264:                     clause_text, ccomp_subj,
3265:                     extraction_rule="imposed_direct_speech",
3266:                 )
3267:                 if decomp is not None:
3268:                     imposed.append(decomp)
3269:                     seen_subtree_starts.add(tok.i)
3270:             else:
3271:                 # --- Type 14: Noun clause as object ---
3272:                 # Non-speech, non-quote ccomp: the embedded clause is a
3273:                 # known proposition.  "I realized self-care is important"
3274:                 # -> imposed: "self-care is important" (its own fact).
3275:                 subtree_sorted = sorted(tok.subtree, key=lambda t: t.i)
3276:                 filtered = [
3277:                     t.text for t in subtree_sorted
3278:                     if not (t.dep_ == "mark"
3279:                             and t.lemma_.lower() == "that")
3280:                 ]
3281:                 clause_text = " ".join(filtered).strip()
3282:                 ccomp_subj = speaker or "user"
3283:                 for child in tok.children:
3284:                     if child.dep_ in ("nsubj", "nsubjpass"):
3285:                         ccomp_subj = child.text
3286:                         break
3287: 
3288:                 decomp = _build_imposed(
3289:                     clause_text, ccomp_subj,
3290:                     extraction_rule="imposed_noun_clause",
3291:                     predicate_override=tok.lemma_,
3292:                 )
3293:                 if decomp is not None:
3294:                     imposed.append(decomp)
3295:                     seen_subtree_starts.add(tok.i)
3296: 
3297:         # --- Type 26: Correlative conjunctions ---
3298:         elif (dep == "preconj"
3299:                 and tok.lemma_.lower() in ("both", "either", "neither")):
3300:             head = tok.head
3301:             if head:
3302:                 conj_tok = None
3303:                 for child in head.children:
3304:                     if child.dep_ == "conj":
3305:                         conj_tok = child
3306:                         break
3307:                 if conj_tok:
3308:                     for element in (head, conj_tok):
3309:                         el_subtree = sorted(
3310:                             element.subtree, key=lambda t: t.i,
3311:                         )
3312:                         filtered = [
3313:                             t.text for t in el_subtree
3314:                             if t.dep_ not in ("preconj", "cc")
3315:                         ]
3316:                         el_text = " ".join(filtered).strip()
3317:                         if el_text and not _is_duplicate(el_text):
3318:                             decomp = _build_imposed(
3319:                                 f"{el_text} exists",
3320:                                 speaker or "user",
3321:                                 extraction_rule="imposed_correlative",
3322:                             )
3323:                             if decomp is not None:
3324:                                 imposed.append(decomp)
3325: 
3326:         # --- Type 27: Comparative/superlative ---
3327:         elif pos == "ADJ" and tag in ("JJR", "JJS"):
3328:             than_obj = None
3329:             for child in tok.children:
3330:                 if child.dep_ == "prep" and child.lemma_ == "than":
3331:                     for gc in child.children:
3332:                         if gc.dep_ == "pobj":
3333:                             than_obj = _span_text(gc)
3334:                             break
3335:                     break
3336:             if than_obj:
3337:                 base_adj = tok.lemma_ if tok.lemma_ else tok.text
3338:                 imposed_text = f"{than_obj} is {base_adj}"
3339:                 decomp = _build_imposed(
3340:                     imposed_text, than_obj,
3341:                     extraction_rule="imposed_comparative",
3342:                 )
3343:                 if decomp is not None:
3344:                     imposed.append(decomp)
3345: 
3346:         # --- Type 2: Negative frame entity extraction (dep=neg) ---
3347:         elif dep == "neg":
3348:             # "I don't have a car" -> "car" exists as a concept
3349:             # Walk to the negated verb, extract its dobj/pobj/attr
3350:             head_verb = tok.head
3351:             if head_verb and head_verb.pos_ in ("VERB", "AUX"):
3352:                 # Find extractable object from the negated verb
3353:                 neg_obj_tok = None
3354:                 for child in head_verb.children:
3355:                     if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
3356:                         neg_obj_tok = child
3357:                         break
3358:                 # Try pobj via prep if no direct object
3359:                 if neg_obj_tok is None:
3360:                     for child in head_verb.children:
3361:                         if child.dep_ == "prep":
3362:                             for gc in child.children:
3363:                                 if gc.dep_ == "pobj":
3364:                                     neg_obj_tok = gc
3365:                                     break
3366:                             if neg_obj_tok is not None:
3367:                                 break
3368:                 if neg_obj_tok is not None:
3369:                     obj_text = _span_text(neg_obj_tok)
3370:                     if obj_text and len(obj_text.strip()) >= 2:
3371:                         # Find subject of the negated verb
3372:                         neg_subj = speaker or "user"
3373:                         for child in head_verb.children:
3374:                             if child.dep_ in ("nsubj", "nsubjpass"):
3375:                                 neg_subj = child.text
3376:                                 break
3377:                         decomp = _build_imposed(
3378:                             obj_text, neg_subj,
3379:                             mood="indicative",
3380:                             extraction_rule="imposed_negative_frame",
3381:                             skip_dedup=True,
3382:                         )
3383:                         if decomp is not None:
3384:                             decomp.predicate = head_verb.lemma_.lower()
3385:                             decomp.object = obj_text
3386:                             imposed.append(decomp)
3387: 
3388:         # --- Type 16: Passive agent presupposition (dep=nsubjpass) ---
3389:         elif dep == "nsubjpass":
3390:             # "The window was broken by the storm" -> agent = "the storm"
3391:             head_verb = tok.head
3392:             if head_verb and head_verb.pos_ in ("VERB", "AUX"):
3393:                 # Check for "by" prep to find agent
3394:                 agent_tok = None
3395:                 for child in head_verb.children:
3396:                     if child.dep_ == "agent" or (
3397:                         child.dep_ == "prep" and child.lemma_ == "by"
3398:                     ):
3399:                         for gc in child.children:
3400:                             if gc.dep_ == "pobj":
3401:                                 agent_tok = gc
3402:                                 break
3403:                         break
3404: 
3405:                 # Guard: if no explicit agent AND nsubj is inanimate,
3406:                 # this is stative (not true passive). "The house was gone"
3407:                 # = state, not "someone made the house gone".
3408:                 if agent_tok is None:
3409:                     nsubj_is_animate = (
3410:                         tok.pos_ == "PRON"
3411:                         or tok.ent_type_ == "PERSON"
3412:                     )
3413:                     if not nsubj_is_animate:
3414:                         continue  # stative â€” skip passive agent extraction
3415: 
3416:                 patient_text = _span_text(tok)
3417:                 verb_lemma = head_verb.lemma_.lower()
3418: 
3419:                 if agent_tok is not None:
3420:                     agent_text = _span_text(agent_tok)
3421:                     imposed_text = (
3422:                         f"{agent_text} {verb_lemma} {patient_text}"
3423:                     )
3424:                     decomp = _build_imposed(
3425:                         imposed_text, agent_text,
3426:                         mood="indicative",
3427:                         extraction_rule="imposed_passive_agent",
3428:                         predicate_override=verb_lemma,
3429:                         skip_dedup=True,
3430:                     )
3431:                     if decomp is not None:
3432:                         decomp.subject = agent_text
3433:                         decomp.object = patient_text
3434:                         imposed.append(decomp)
3435:                 else:
3436:                     # No agent phrase â€” implied agent
3437:                     imposed_text = f"{verb_lemma} {patient_text}"
3438:                     decomp = _build_imposed(
3439:                         imposed_text, "",
3440:                         mood="indicative",
3441:                         extraction_rule="imposed_passive_agent",
3442:                         predicate_override=verb_lemma,
3443:                         skip_dedup=True,
3444:                     )
3445:                     if decomp is not None:
3446:                         decomp.subject = ""
3447:                         decomp.object = patient_text
3448:                         imposed.append(decomp)
3449: 
3450:     return imposed
3451: 
3452: 
3453: # ---------------------------------------------------------------------------
3454: # Triple derivation (backward compat)
3455: # ---------------------------------------------------------------------------
3456: 
3457: def _derive_triple(decomp: TraceDecomposition) -> Triple:
3458:     """Derive a backward-compatible Triple from a TraceDecomposition."""
3459:     return Triple(
3460:         subject=decomp.subject or decomp.relational_subject,
3461:         predicate=decomp.predicate,
3462:         object=decomp.object,  # Invariant 1: never fall back to episodic_fact (full clause)
3463:         is_historical=decomp.is_historical,
3464:         utterance_type=decomp.utterance_type,
3465:         negated=decomp.negated,
3466:         mood=decomp.mood,
3467:         extraction_rule=decomp.extraction_rule,
3468:     )
3469: 
3470: 
3471: # ---------------------------------------------------------------------------
3472: # Compound clause splitting
3473: # Grammar reference: compound sentences have two independent clauses
3474: # joined by a coordinating conjunction.
3475: # ---------------------------------------------------------------------------
3476: 
3477: def _head_chain_reaches_root(tok, _max_depth: int = 5) -> bool:
3478:     """Return True if following tok.head through conj deps reaches ROOT."""
3479:     current = tok.head
3480:     for _ in range(_max_depth):
3481:         if current.dep_ == "ROOT":
3482:             return True
3483:         if current.dep_ != "conj":
3484:             return False
3485:         current = current.head
3486:     return False
3487: 
3488: 
3489: def _span_has_subject_and_verb(tokens) -> bool:
3490:     """Return True if a list of tokens contains both a subject and a verb/root."""
3491:     has_subj = False
3492:     has_verb = False
3493:     for tok in tokens:
3494:         if tok.dep_ in ("nsubj", "nsubjpass"):
3495:             has_subj = True
3496:         if tok.dep_ == "ROOT" or tok.pos_ in ("VERB", "AUX"):
3497:             has_verb = True
3498:     return has_subj and has_verb
3499: 
3500: 
3501: _CONJUNCTION_TYPE_MAP: Dict[str, str] = {
3502:     "and": "additive",
3503:     "but": "contrast",
3504:     "yet": "contrast",
3505:     "for": "reason",
3506:     "so": "consequence",
3507:     "or": "alternative",
3508:     "nor": "alternative",
3509: }
3510: 
3511: 
3512: def _patch_fragment_lemmas(frag_doc, original_tokens) -> None:
3513:     """Fix fragment re-parse degradation: when spaCy re-parses a clause
3514:     fragment, POS/tag may change (e.g., VBDâ†’JJ for 'preferred'), causing
3515:     wrong lemmatization. Patch: for each token where the fragment's lemma
3516:     equals the surface form but the original had a different (correct)
3517:     lemma, copy the original's lemma and tag."""
3518:     orig_by_text = {}
3519:     for t in original_tokens:
3520:         key = t.text.lower()
3521:         if key not in orig_by_text:
3522:             orig_by_text[key] = t
3523: 
3524:     for tok in frag_doc:
3525:         key = tok.text.lower()
3526:         orig = orig_by_text.get(key)
3527:         if orig is None:
3528:             continue
3529:         # If fragment lemma equals surface form but original had a real lemma
3530:         if (tok.lemma_.lower() == tok.text.lower()
3531:                 and orig.lemma_.lower() != orig.text.lower()):
3532:             tok.lemma_ = orig.lemma_
3533:         # If fragment tag changed from verb to adjective (common degradation)
3534:         if (tok.tag_ in ("JJ", "NN") and orig.tag_ in ("VBD", "VBG", "VBN", "VBZ")):
3535:             tok.tag_ = orig.tag_
3536:             tok.pos_ = orig.pos_
3537:             # Copy morph features (tense, aspect, etc.) from original
3538:             tok.set_morph(str(orig.morph))
3539: 
3540: 
3541: def _split_compound_clauses(doc) -> List:
3542:     """Split compound sentences on coordinating conjunctions (FANBOYS)
3543:     AND semicolons (Grammar reference Page 912).
3544: 
3545:     Only splits when BOTH sides have a subject + verb (independent clauses).
3546:     "I like pottery and swimming" does NOT split.
3547:     "I like pottery and I went swimming" DOES split.
3548:     "She wanted tennis; he wanted basketball" DOES split.
3549: 
3550:     Returns a list of (doc, conjunction_type) tuples.  conjunction_type is
3551:     None for the first clause in each split group and for unsplit sentences;
3552:     subsequent clauses carry "additive", "contrast", "reason",
3553:     "consequence", or "alternative" based on the conjunction that
3554:     preceded them.
3555:     """
3556:     frag_nlp = _get_nlp_fragment()
3557:     clauses: List = []
3558: 
3559:     for sent in doc.sents:
3560:         sent_tokens = list(sent)
3561: 
3562:         # ---- Phase 1: split on semicolons (Grammar ref #8) ----
3563:         semicolon_indices = [
3564:             tok.i for tok in sent_tokens if tok.text == ";"
3565:         ]
3566: 
3567:         # Build spans separated by semicolons
3568:         semicolon_spans: List[List] = []
3569:         if semicolon_indices:
3570:             prev = sent.start
3571:             valid_semicolon_split = True
3572:             candidate_spans: List[List] = []
3573:             for sc_i in semicolon_indices:
3574:                 span_tokens = [t for t in sent_tokens
3575:                                if t.i >= prev and t.i < sc_i]
3576:                 candidate_spans.append(span_tokens)
3577:                 prev = sc_i + 1  # skip the semicolon itself
3578:             # remaining tokens after last semicolon
3579:             candidate_spans.append(
3580:                 [t for t in sent_tokens if t.i >= prev]
3581:             )
3582:             # Validate: every span must have subject + verb
3583:             for span_toks in candidate_spans:
3584:                 if not span_toks or not _span_has_subject_and_verb(span_toks):
3585:                     valid_semicolon_split = False
3586:                     break
3587:             if valid_semicolon_split:
3588:                 semicolon_spans = candidate_spans
3589: 
3590:         # If semicolons produced valid splits, re-parse each span
3591:         if semicolon_spans:
3592:             for idx, span_toks in enumerate(semicolon_spans):
3593:                 if span_toks:
3594:                     clause_text = " ".join(
3595:                         t.text for t in span_toks
3596:                     ).strip().rstrip(" ,;")
3597:                     if clause_text:
3598:                         clause_doc = frag_nlp(clause_text)
3599:                         _patch_fragment_lemmas(clause_doc, span_toks)
3600:                         conj_type = "additive" if idx > 0 else None
3601:                         clauses.append((clause_doc, conj_type))
3602:             continue  # skip conjunction splitting for this sentence
3603: 
3604:         # ---- Phase 1.5: comma splice detection ----
3605:         # Pattern: ccomp child with its own nsubj, preceded by comma,
3606:         # no subordinating conjunction. "I have a cat, they are rescues"
3607:         # spaCy treats first clause as ccomp of second. Split at comma.
3608:         _comma_split_done = False
3609:         root_tok = None
3610:         for tok in sent_tokens:
3611:             if tok.dep_ == "ROOT":
3612:                 root_tok = tok
3613:                 break
3614:         if root_tok:
3615:             for child in root_tok.children:
3616:                 if (child.dep_ == "ccomp" and child.pos_ in ("VERB", "AUX")
3617:                         and any(gc.dep_ in ("nsubj", "nsubjpass")
3618:                                 for gc in child.children)):
3619:                     # Check: no subordinating conjunction (mark) on ccomp
3620:                     has_mark = any(
3621:                         gc.dep_ == "mark" for gc in child.children
3622:                     )
3623:                     if has_mark:
3624:                         continue
3625:                     # Find comma between ccomp subtree and ROOT
3626:                     ccomp_end = max(t.i for t in child.subtree)
3627:                     comma_idx = None
3628:                     for t in sent_tokens:
3629:                         if (t.text == "," and t.i > ccomp_end
3630:                                 and t.i < root_tok.i):
3631:                             comma_idx = t.i
3632:                             break
3633:                         elif (t.text == "," and t.i < root_tok.i
3634:                               and t.i > min(tc.i for tc in child.subtree)):
3635:                             comma_idx = t.i
3636:                             break
3637:                     if comma_idx is None:
3638:                         # Comma might be between ccomp's last token and root's nsubj
3639:                         for t in sent_tokens:
3640:                             if t.text == ",":
3641:                                 comma_idx = t.i
3642:                                 break
3643:                     if comma_idx is not None:
3644:                         # Split: left = ccomp subtree, right = rest
3645:                         left_toks = [t for t in sent_tokens if t.i <= ccomp_end]
3646:                         right_toks = [t for t in sent_tokens
3647:                                       if t.i > comma_idx and t.text != ","]
3648:                         if (left_toks and right_toks
3649:                                 and _span_has_subject_and_verb(left_toks)
3650:                                 and _span_has_subject_and_verb(right_toks)):
3651:                             left_text = " ".join(
3652:                                 t.text for t in left_toks
3653:                             ).strip().rstrip(" ,")
3654:                             right_text = " ".join(
3655:                                 t.text for t in right_toks
3656:                             ).strip()
3657:                             if left_text and right_text:
3658:                                 left_doc = frag_nlp(left_text)
3659:                                 _patch_fragment_lemmas(left_doc, left_toks)
3660:                                 right_doc = frag_nlp(right_text)
3661:                                 _patch_fragment_lemmas(right_doc, right_toks)
3662:                                 clauses.append((left_doc, None))
3663:                                 clauses.append((right_doc, "additive"))
3664:                                 _comma_split_done = True
3665:                                 break
3666: 
3667:         if _comma_split_done:
3668:             continue
3669: 
3670:         # ---- Phase 2: split on coordinating conjunctions (FANBOYS) ----
3671:         # Each entry is (split_index, conjunction_type_string).
3672:         split_points: List[Tuple[int, Optional[str]]] = []
3673:         for tok in sent_tokens:
3674:             if (tok.dep_ == "conj"
3675:                     and tok.pos_ in ("VERB", "AUX")
3676:                     and _head_chain_reaches_root(tok)):
3677:                 has_own_subject = any(
3678:                     child.dep_ in ("nsubj", "nsubjpass")
3679:                     for child in tok.children
3680:                 )
3681:                 if has_own_subject:
3682:                     cc_tok = None
3683:                     for child in tok.head.children:
3684:                         if (child.dep_ == "cc"
3685:                                 and child.i > tok.head.i
3686:                                 and child.i < tok.i):
3687:                             cc_tok = child
3688:                             break
3689:                     if cc_tok is None:
3690:                         for child in tok.children:
3691:                             if child.dep_ == "cc":
3692:                                 cc_tok = child
3693:                                 break
3694: 
3695:                     split_idx = cc_tok.i if cc_tok else tok.i
3696:                     conj_lemma = (
3697:                         cc_tok.lemma_.lower() if cc_tok else None
3698:                     )
3699:                     conj_type = _CONJUNCTION_TYPE_MAP.get(
3700:                         conj_lemma, "additive",
3701:                     ) if conj_lemma else None
3702:                     split_points.append((split_idx, conj_type))
3703: 
3704:         if not split_points:
3705:             clauses.append((sent.as_doc(), None))
3706:         else:
3707:             prev_start = sent.start
3708:             clause_idx = 0
3709:             for split_i, conj_type in sorted(
3710:                 split_points, key=lambda x: x[0],
3711:             ):
3712:                 clause_tokens = [
3713:                     t for t in sent_tokens
3714:                     if t.i >= prev_start and t.i < split_i
3715:                 ]
3716:                 if clause_tokens:
3717:                     clause_text = " ".join(
3718:                         t.text for t in clause_tokens
3719:                     ).strip().rstrip(" ,")
3720:                     if clause_text:
3721:                         clause_doc = frag_nlp(clause_text)
3722:                         _patch_fragment_lemmas(clause_doc, clause_tokens)
3723:                         ct = None if clause_idx == 0 else conj_type
3724:                         clauses.append((clause_doc, ct))
3725:                         clause_idx += 1
3726:                 prev_start = split_i + 1
3727: 
3728:             remaining = [t for t in sent_tokens if t.i >= prev_start]
3729:             if remaining:
3730:                 clause_text = " ".join(
3731:                     t.text for t in remaining
3732:                 ).strip()
3733:                 if clause_text:
3734:                     clause_doc = frag_nlp(clause_text)
3735:                     _patch_fragment_lemmas(clause_doc, remaining)
3736:                     last_conj_type = split_points[-1][1] if split_points else None
3737:                     clauses.append((clause_doc, last_conj_type))
3738: 
3739:     return clauses
3740: 
3741: 
3742: # ---------------------------------------------------------------------------
3743: # process() -- the ONE entry point for the WRITE path
3744: # Spec: Text in -> classify -> extract traces -> derive triples
3745: # ---------------------------------------------------------------------------
3746: 
3747: def process(text: str, speaker: Optional[str] = None, listener: str = "user") -> GrammarResult:
3748:     """Main entry point.  Classify, extract traces, derive triples.
3749: 
3750:     Multi-sentence aware: each sentence classified and extracted independently.
3751:     Compound sentences split into independent clauses.
3752: 
3753:     Args:
3754:         text: Raw input text to process.
3755:         speaker: Name of the person speaking (resolves "I" -> speaker).
3756:         listener: Name of the person being addressed (resolves "you" -> listener).
3757:                   Defaults to "user" for backward compatibility.
3758: 
3759:     Spec Part 2: extraction rules by sentence type.
3760:     """
3761:     nlp = _get_nlp()
3762:     doc = nlp(text)
3763: 
3764:     clause_tuples = _split_compound_clauses(doc)
3765: 
3766:     decompositions: List[TraceDecomposition] = []
3767:     all_triples: List[Triple] = []
3768:     resolved_parts: List[str] = []
3769:     primary_classification: Optional[UtteranceClassification] = None
3770:     primary_sent_doc = None
3771: 
3772:     # Type 13 (free indirect speech): track perspective subject across
3773:     # sentences.  When sentence N has a PERSON/PROPN nsubj + mental verb,
3774:     # the next sentence may be narrated from that person's perspective.
3775:     _perspective_subject: Optional[str] = None
3776:     _perspective_gap: int = 0  # reset after 1 sentence gap
3777: 
3778:     for sent_doc, _conj_type in clause_tuples:
3779:         sent_cls = classify_utterance(sent_doc, speaker)
3780: 
3781:         if primary_classification is None or (
3782:             not primary_classification.is_storable and sent_cls.is_storable
3783:         ):
3784:             primary_classification = sent_cls
3785:             primary_sent_doc = sent_doc
3786: 
3787:         resolved_parts.append(resolve_pronouns(sent_doc, speaker, listener=listener))
3788:         sent_tense = detect_tense_aspect(sent_doc)
3789: 
3790:         # ============================================================
3791:         # SINGLE PATH: every clause goes through the same pipeline.
3792:         # Classification sets metadata (mood, storable), not code path.
3793:         # ============================================================
3794: 
3795:         # Step 1: Tag question pre-processing (strip the tag)
3796:         extraction_doc = sent_doc
3797:         if sent_cls.subcategory == "tag_question":
3798:             extraction_doc = _strip_tag_question(sent_doc)
3799:             sent_tense = detect_tense_aspect(extraction_doc)
3800: 
3801:         # Step 2: Extract traces â€” ALWAYS, for ALL sentence types
3802:         decomp = _extract_traces_from_sentence(
3803:             extraction_doc, speaker, sent_tense, listener=listener,
3804:         )
3805:         decomp.utterance_type = sent_cls.utterance_type_id
3806: 
3807:         # Step 3: Set mood from classification
3808:         if sent_cls.is_question:
3809:             decomp.mood = "interrogative"
3810:         elif sent_cls.is_command:
3811:             decomp.mood = "imperative"
3812:         elif sent_cls.subcategory == "tag_question":
3813:             decomp.mood = "indicative"
3814:         # else: mood from detect_mood() inside _extract_traces (already set)
3815: 
3816:         # Step 4: Conjunction semantics (Type 25)
3817:         if _conj_type:
3818:             decomp.extraction_rule = (
3819:                 f"{decomp.extraction_rule or 'trace'}_conj_{_conj_type}"
3820:             )
3821: 
3822:         # Step 5: Free indirect speech perspective (Type 13)
3823:         if _perspective_subject and _perspective_gap == 0:
3824:             root_tok = _get_root(sent_doc)
3825:             clause_subj_tok = None
3826:             if root_tok:
3827:                 for child in root_tok.children:
3828:                     if child.dep_ in ("nsubj", "nsubjpass"):
3829:                         clause_subj_tok = child
3830:                         break
3831:             _apply_fis = (
3832:                 clause_subj_tok is None
3833:                 or (clause_subj_tok.pos_ not in ("PROPN", "PRON")
3834:                     and clause_subj_tok.ent_type_ != "PERSON")
3835:             )
3836:             if _apply_fis:
3837:                 decomp.relational_subject = _perspective_subject
3838:                 decomp.subject = _perspective_subject
3839:                 decomp.extraction_rule = (
3840:                     f"{decomp.extraction_rule or 'trace'}"
3841:                     f"_free_indirect_speech"
3842:                 )
3843: 
3844:         # Step 6: Store or skip based on storability
3845:         # Questions are non-storable (the question itself isn't a fact).
3846:         # Everything else is stored.
3847:         if not sent_cls.is_question:
3848:             decompositions.append(decomp)
3849:             all_triples.append(_derive_triple(decomp))
3850: 
3851:         # Step 6b: Compound predicate (Grammar ref p.829, lines 5895-5911)
3852:         # One subject + multiple conj verbs â†’ each verb is a separate fact.
3853:         # "moved to Chicago, worked three years, relocated to Portland"
3854:         # â†’ 3 traces, all sharing ROOT's subject.
3855:         # Also handles gerund coordination: "Running, reading, playing"
3856:         # Walk the full conj chain (conj of conj of conj...).
3857:         root_tok = _get_root(sent_doc)
3858:         if root_tok and root_tok.pos_ in ("VERB", "AUX"):
3859:             conj_queue = [c for c in root_tok.children
3860:                           if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX", "NOUN")]
3861:             frag_nlp = _get_nlp_fragment()
3862:             while conj_queue:
3863:                 conj_child = conj_queue.pop(0)
3864:                 # Skip conj verbs with their own nsubj â€” those are
3865:                 # compound SENTENCES, already split by _split_compound_clauses.
3866:                 # Compound predicates share ROOT's subject (no own nsubj).
3867:                 _has_own_subj = any(
3868:                     c.dep_ in ("nsubj", "nsubjpass")
3869:                     for c in conj_child.children
3870:                 )
3871:                 if _has_own_subj:
3872:                     continue
3873:                 # Add this node's conj children to the queue (chain)
3874:                 conj_queue.extend(
3875:                     c for c in conj_child.children
3876:                     if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX", "NOUN")
3877:                 )
3878:                 conj_subtree = sorted(conj_child.subtree, key=lambda t: t.i)
3879:                 # Filter out conj children's subtrees (they get their own trace)
3880:                 conj_child_indices = set()
3881:                 for cc in conj_child.children:
3882:                     if cc.dep_ == "conj":
3883:                         conj_child_indices |= {t.i for t in cc.subtree}
3884:                 conj_tokens = [t for t in conj_subtree
3885:                                if t.i not in conj_child_indices
3886:                                and t.dep_ != "cc" and t.pos_ != "PUNCT"]
3887:                 conj_text = " ".join(t.text for t in conj_tokens).strip()
3888:                 if conj_text:
3889:                     conj_doc = frag_nlp(conj_text)
3890:                     _patch_fragment_lemmas(conj_doc, conj_tokens)
3891:                     conj_decomp = _extract_traces_from_sentence(
3892:                         conj_doc, speaker, sent_tense, listener=listener,
3893:                     )
3894:                     # Inherit subject from main trace (compound predicate)
3895:                     conj_decomp.subject = decomp.subject
3896:                     conj_decomp.relational_subject = decomp.relational_subject
3897:                     conj_decomp.extraction_rule = "trace_compound_predicate"
3898:                     decompositions.append(conj_decomp)
3899:                     all_triples.append(_derive_triple(conj_decomp))
3900: 
3901:         # Step 7: Extract imposed facts from subordinate constructions
3902:         # ALWAYS runs for ALL sentence types (questions, commands, statements).
3903:         existing_episodics = frozenset(
3904:             d.episodic_fact for d in decompositions if d.episodic_fact
3905:         )
3906:         sub_imposed = _extract_imposed_facts(
3907:             sent_doc, speaker, sent_tense, existing_episodics, listener=listener,
3908:         )
3909:         for imp_decomp in sub_imposed:
3910:             decompositions.append(imp_decomp)
3911:             all_triples.append(_derive_triple(imp_decomp))
3912: 
3913:         # --- Type 13: update perspective subject for next clause ---
3914:         # If this sentence has a PERSON/PROPN nsubj + mental/perception
3915:         # verb, store the subject as the perspective holder.
3916:         # Detect mental/perception verbs via VerbClass + clausal complement.
3917:         # No word list â€” uses WordNet hypernym closure (VerbClass.ABILITY
3918:         # covers know/understand/believe; EXPERIENCE covers feel/sense).
3919:         # Structural fallback: any verb with ccomp/xcomp that isn't SPEECH
3920:         # or PLANNING frames a proposition â†’ mental/perception.
3921:         _cur_root = _get_root(sent_doc)
3922:         _found_perspective = False
3923:         if _cur_root and _cur_root.pos_ in ("VERB", "AUX"):
3924:             _root_lemma = _cur_root.lemma_.lower()
3925:             _cur_vc = classify_verb_class(_root_lemma)
3926:             _has_clausal = any(
3927:                 c.dep_ in ("ccomp", "xcomp") for c in _cur_root.children
3928:             )
3929:             _is_mental = _cur_vc in (VerbClass.ABILITY, VerbClass.EXPERIENCE)
3930:             if not _is_mental and _has_clausal:
3931:                 if _cur_vc not in (VerbClass.PLANNING,):
3932:                     # SPEECH verbs are mental when used without a recipient
3933:                     # dobj. "She believed it" = mental. "She told me" = speech.
3934:                     if _cur_vc == VerbClass.SPEECH:
3935:                         _has_recipient = any(
3936:                             c.dep_ == "dobj" and c.pos_ == "PRON"
3937:                             for c in _cur_root.children
3938:                         )
3939:                         if not _has_recipient:
3940:                             _is_mental = True
3941:                     else:
3942:                         _is_mental = True
3943:             if _is_mental:
3944:                 for child in _cur_root.children:
3945:                     if child.dep_ in ("nsubj", "nsubjpass"):
3946:                         if (child.pos_ in ("PROPN", "PRON")
3947:                                 or child.ent_type_ == "PERSON"):
3948:                             _perspective_subject = child.text
3949:                             _perspective_gap = 0
3950:                             _found_perspective = True
3951:                         break
3952:         if not _found_perspective:
3953:             # Decay: if we had a perspective subject but this sentence
3954:             # didn't renew it, increment the gap counter.  Reset after
3955:             # 1 sentence gap (don't carry indefinitely).
3956:             if _perspective_subject is not None:
3957:                 _perspective_gap += 1
3958:                 if _perspective_gap >= 1:
3959:                     _perspective_subject = None
3960:                     _perspective_gap = 0
3961: 
3962:     # Fallback classification
3963:     if primary_classification is None:
3964:         primary_classification = classify_utterance(doc, speaker)
3965:         primary_sent_doc = doc
3966: 
3967:     # Derive grammatical features from primary sentence
3968:     # Tag questions: override mood to indicative (pragmatically assertions)
3969:     if (primary_classification is not None
3970:             and primary_classification.subcategory == "tag_question"):
3971:         mood = "indicative"
3972:     else:
3973:         mood = detect_mood(primary_sent_doc)
3974:     negated = detect_negation(primary_sent_doc)
3975:     tense_aspect = detect_tense_aspect(primary_sent_doc)
3976:     voice = detect_voice(primary_sent_doc)
3977:     resolved_text = " ".join(resolved_parts)
3978: 
3979:     emotion = None
3980:     if primary_classification.is_emotion:
3981:         root = _get_root(primary_sent_doc)
3982:         if root:
3983:             for child in root.children:
3984:                 if child.dep_ in ("acomp", "attr", "oprd") and child.pos_ == "ADJ":
3985:                     emotion = child.text.lower()
3986:                     break
3987: 
3988:     return GrammarResult(
3989:         trace_decompositions=decompositions,
3990:         triples=all_triples,
3991:         classification=primary_classification,
3992:         mood=mood,
3993:         negated=negated,
3994:         tense_aspect=tense_aspect,
3995:         voice=voice,
3996:         resolved_text=resolved_text,
3997:         emotion=emotion,
3998:     )
3999: 
4000: 
4001: # ---------------------------------------------------------------------------
4002: # extract_typed_triple -- architecture verifier compatibility
4003: # ---------------------------------------------------------------------------
4004: 
4005: def extract_typed_triple(
4006:     text: str,
4007:     speaker: Optional[str] = None,
4008: ) -> List[Tuple[str, str, str]]:
4009:     """Backward-compatibility wrapper for architecture verifier."""
4010:     result = process(text, speaker=speaker)
4011:     return [(t.subject, t.predicate, t.object) for t in result.triples]
4012: 
4013: 
4014: # ---------------------------------------------------------------------------
4015: # Query decomposition -- READ-PATH interface
4016: # Spec: classify_query parses questions the same way process() parses statements
4017: # ---------------------------------------------------------------------------
4018: 
4019: @dataclass
4020: class QueryDecomposition:
4021:     """Structural decomposition of a query for direct SQL lookup."""
4022:     wh_word: Optional[str] = None
4023:     match_subject: Optional[str] = None
4024:     match_predicate: Optional[str] = None
4025:     match_object: Optional[str] = None
4026:     match_entity: Optional[str] = None
4027:     match_schema: Optional[str] = None
4028:     return_field: str = "episodic"
4029:     utterance_type_id: int = 0
4030:     is_structural: bool = False
4031: 
4032: 
4033: def _wh_to_return_field(wh_tok) -> str:
4034:     """Map a WH-token to a return_field using spaCy POS/dep features.
4035:     Grammar reference Section 9: Question Decomposition.
4036: 
4037:     Root cause of prior bug: all WRB advmod tokens were handled by one
4038:     code path that used verb class to decide the return field.  But
4039:     "when" (temporal), "where" (locative), "why" (causal), and "how"
4040:     (manner) are structurally distinct question types in the closed
4041:     WH-word class and must be distinguished by lemma first.
4042: 
4043:     WRB "when"  -> temporal  (always â€” asks about time)
4044:     WRB "where" -> episodic  (location lives in object/prep trace)
4045:     WRB "why"   -> episodic  (reason/cause lives in episodic trace)
4046:     WRB "how"   -> temporal  if head is temporal ADV ("how long"),
4047:                    emotional if head has ADJ complement,
4048:                    else episodic
4049:     WP$         -> relational
4050:     WP subj/attr-> relational
4051:     Everything else -> episodic
4052:     """
4053:     if wh_tok is None:
4054:         return "episodic"
4055: 
4056:     dep = wh_tok.dep_
4057:     tag = wh_tok.tag_
4058:     lemma = wh_tok.lemma_.lower()
4059: 
4060:     # --- WRB advmod: distinguish by lemma (closed grammatical class) ---
4061:     if tag == "WRB" and dep == "advmod":
4062:         # "when" always asks about time
4063:         if lemma == "when":
4064:             return "temporal"
4065: 
4066:         # "where" asks about location â€” stored in episodic trace
4067:         if lemma == "where":
4068:             return "episodic"
4069: 
4070:         # "why" asks for reason/cause â€” episodic
4071:         if lemma == "why":
4072:             return "episodic"
4073: 
4074:         # "how" â€” context-dependent
4075:         head = wh_tok.head
4076:         # "how long" / "how long ago" -> temporal
4077:         if head.pos_ == "ADV" and head.lemma_.lower() == "long":
4078:             return "temporal"
4079:         # "how" + ADJ complement (e.g. "how did she feel") -> emotional
4080:         has_adj_complement = any(
4081:             c.dep_ in ("acomp", "oprd") and c.pos_ == "ADJ"
4082:             for c in head.children
4083:         )
4084:         if has_adj_complement:
4085:             return "emotional"
4086:         return "episodic"
4087: 
4088:     # --- WP$ ("whose") -> relational ---
4089:     if tag == "WP$":
4090:         return "relational"
4091: 
4092:     # --- WP ("who"/"whom"/"what") in subject position ---
4093:     # Note: WP in attr (e.g. "What is X?" / "Who is X?") maps to episodic
4094:     # because en_core_web_sm lacks morph features to distinguish [+human]
4095:     # "who" from [-human] "what" â€” both are WP with empty morph.
4096:     # Returning episodic (object field) is correct for the majority case
4097:     # ("What is X's Y?" queries outnumber "Who is X?" queries).
4098:     if tag == "WP":
4099:         if dep in ("nsubj", "nsubjpass"):
4100:             return "relational"
4101: 
4102:     return "episodic"
4103: 
4104: 
4105: def classify_query(query_text: str) -> QueryDecomposition:
4106:     """Decompose a query into structural fields for SQL lookup.
4107:     Spec: same spaCy parse as process(), extracts entity, keywords, schema, wh_type."""
4108:     nlp = _get_nlp()
4109:     doc = nlp(query_text)
4110:     result = QueryDecomposition()
4111: 
4112:     # WH-word extraction
4113:     wh_tok = None
4114:     for tok in doc:
4115:         if tok.pos_ == "SPACE":
4116:             continue
4117:         if tok.tag_ in ("WDT", "WP", "WP$", "WRB"):
4118:             wh_tok = tok
4119:             result.wh_word = tok.text.lower()
4120:             break
4121:         break
4122: 
4123:     result.return_field = _wh_to_return_field(wh_tok)
4124: 
4125:     root = _get_root(doc)
4126:     if root is None:
4127:         return result
4128: 
4129:     # Extract subject
4130:     subj_tok = None
4131:     for child in root.children:
4132:         if child.dep_ in ("nsubj", "nsubjpass"):
4133:             subj_tok = child
4134:             break
4135: 
4136:     if subj_tok is not None:
4137:         if subj_tok.tag_ not in ("WDT", "WP", "WP$", "WRB"):
4138:             if subj_tok.pos_ == "PRON":
4139:                 person = subj_tok.morph.get("Person")
4140:                 if person and "1" in person:
4141:                     result.match_subject = "user"
4142:             else:
4143:                 result.match_subject = _span_text(subj_tok).strip()
4144: 
4145:     # Extract predicate
4146:     if root.pos_ == "VERB":
4147:         result.match_predicate = root.lemma_.lower()
4148:     elif root.pos_ == "AUX":
4149:         for child in root.children:
4150:             if child.dep_ in ("xcomp", "ccomp", "acomp", "attr"):
4151:                 if child.pos_ in ("VERB", "NOUN", "ADJ"):
4152:                     result.match_predicate = child.lemma_.lower()
4153:                     break
4154: 
4155:     # Extract object
4156:     dobj_tok = None
4157:     for child in root.children:
4158:         if child.dep_ in ("dobj", "attr"):
4159:             dobj_tok = child
4160:             break
4161:     if dobj_tok is not None:
4162:         if dobj_tok.tag_ not in ("WDT", "WP", "WP$", "WRB"):
4163:             result.match_object = _span_text(dobj_tok).strip()
4164: 
4165:     if result.match_object is None:
4166:         prep, pobj = _get_prep_object(root)
4167:         if pobj:
4168:             result.match_object = pobj.strip()
4169: 
4170:     # Extract match_entity from NER
4171:     # Spec: relational_entities collects PERSON, ORG, GPE, LOC, FAC, NORP
4172:     for ent in doc.ents:
4173:         if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"):
4174:             result.match_entity = ent.text
4175:             break
4176: 
4177:     # Utterance type
4178:     cls = classify_utterance(doc)
4179:     result.utterance_type_id = cls.utterance_type_id
4180: 
4181:     # Structural flag
4182:     result.is_structural = bool(
4183:         result.match_subject or result.match_predicate
4184:     )
4185: 
4186:     # Derive match_schema â€” uses the SAME 7-step _extract_schematic() as the
4187:     # write path so that stored schema and query schema never diverge.
4188:     # Note: _reclassify_location_by_object is NOT called here because the
4189:     # write path calls it on content_root (after _find_content_verb), not on
4190:     # root. Adding it here on root caused Cat 5 to regress (29.8% -> 19.2%)
4191:     # without improving Cat 1-4.
4192:     try:
4193:         vc = VerbClass.UNKNOWN
4194:         if root.pos_ == "VERB":
4195:             vc = classify_verb_class(root.lemma_.lower())
4196:         elif root.pos_ == "AUX":
4197:             # For AUX roots (e.g. "is"), classify as BE so _extract_schematic
4198:             # Step 1 triggers xcomp/ccomp delegation correctly.
4199:             vc = classify_verb_class(root.lemma_.lower())
4200: 
4201:         schema = _extract_schematic(doc, root, vc)
4202: 
4203:         if schema and schema != "uncategorized":
4204:             result.match_schema = schema
4205:     except Exception:
4206:         pass
4207: 
4208:     return result
```
