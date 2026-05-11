# sdk-client

Source: [sdk/client.py](/D:/Nura/Code/nura_living_memory_code/sdk/client.py)

```text
0001: """
0002: Kenotic â€” the main SDK client.
0003: 
0004: One class, one entry point: process().
0005: 
0006:   k = Kenotic(user_id=1, db_path="test.db")
0007:   result = k.process("I got a job at Google starting Tuesday.")
0008:   result = k.process("Where do I work?")
0009:   result = k.process("What's going on in my life?")
0010:   result = k.process("Forget about Google.")
0011: 
0012: process() classifies intent via the grammar engine and routes internally
0013: to the appropriate capability (ingest, retrieve, reconstruct, forget,
0014: check_proactive). Returns a ProcessResult with action discriminator.
0015: 
0016: The 8 internal methods (ingest, retrieve, reconstruct, forget, show,
0017: trace, check_proactive, profile) remain available for direct use by
0018: runners and tests that need fine-grained control.
0019: 
0020: No runtime levers. Validation, grammar, and grounding are all always
0021: on in the SDK variant.
0022: """
0023: from __future__ import annotations
0024: 
0025: import logging
0026: import os
0027: from pathlib import Path
0028: from typing import Dict, List, Optional, Union
0029: 
0030: from sdk.types import Situation, Answer, ProcessResult
0031: 
0032: _log = logging.getLogger("kenotic.sdk")
0033: 
0034: 
0035: class Kenotic:
0036:     """Continuity-layer client. One per user/database."""
0037: 
0038:     def __init__(
0039:         self,
0040:         user_id: int = 0,
0041:         db_path: Union[str, Path] = "~/.kenotic/memory.db",
0042:         *,
0043:         embed_device: str = "cuda",
0044:     ):
0045:         """Initialize an isolated continuity layer bound to a SQLite file.
0046: 
0047:         Args:
0048:           user_id: integer partition key. Defaults to 0 (single-user mode
0049:                    used by the MCP Continuity Bridge). Edges are stored
0050:                    under this ID and never cross to other user_ids at
0051:                    retrieval time.
0052:           db_path: path to the SQLite file. Created if missing.
0053:           embed_device: 'cuda' or 'cpu' for MiniLM embedding inference.
0054:         """
0055:         self.user_id = int(user_id)
0056:         self.db_path = str(db_path)
0057: 
0058:         # Wire env used by the engines before import-time side effects
0059:         os.environ.setdefault("RAYA_EMBED_DEVICE", embed_device)
0060: 
0061:         # Point the engines at the caller's DB path
0062:         from config.settings import settings
0063:         settings.sqlite_path = self.db_path
0064: 
0065:         # Ensure schema exists
0066:         self._init_schema()
0067: 
0068:         # Lazy engine construction â€” deferred to first use to keep
0069:         # __init__ cheap
0070:         self._memory = None
0071:         self._temporal = None
0072:         self._retrieval = None
0073: 
0074:         # kenoticArchitectureV1 runtime verifier â€” runs ONCE per instance.
0075:         # Pure observability: delegates to architecture_verifier, logs summary.
0076:         # Never blocks init, never changes control flow.
0077:         self._architecture_status: List[Dict[str, str]] = self._run_verifier()
0078: 
0079:     # -- Architecture verifier --------------------------------------
0080: 
0081:     def _run_verifier(self) -> List[Dict[str, str]]:
0082:         """Run kenoticArchitectureV1 and log a one-line summary.
0083: 
0084:         Returns the full results list for caching. If the verifier itself
0085:         fails (import error, DB issue), returns an empty list and logs the
0086:         exception â€” never crashes init.
0087:         """
0088:         try:
0089:             from app.engines.architecture_verifier import (
0090:                 verify_kenotic_architecture_v1,
0091:             )
0092:             results = verify_kenotic_architecture_v1(db_path=self.db_path)
0093:         except Exception:
0094:             _log.warning(
0095:                 "kenoticArchitectureV1: verifier could not run",
0096:                 exc_info=True,
0097:             )
0098:             return []
0099: 
0100:         passed = sum(1 for r in results if r["status"] == "PASS")
0101:         failed = sum(1 for r in results if r["status"] == "FAIL")
0102:         total = len(results)
0103: 
0104:         _log.info(
0105:             "kenoticArchitectureV1: %d/%d PASS, %d FAIL",
0106:             passed, total, failed,
0107:         )
0108:         for r in results:
0109:             if r["status"] == "FAIL":
0110:                 _log.warning(
0111:                     "  FAIL: %s -- %s", r["check"], r["detail"],
0112:                 )
0113:         return results
0114: 
0115:     def architecture_status(self) -> List[Dict[str, str]]:
0116:         """Return cached kenoticArchitectureV1 verifier results.
0117: 
0118:         Each entry is a dict with keys: check, status, detail.
0119:         Ran once at init; this method returns the cached snapshot.
0120:         """
0121:         return self._architecture_status
0122: 
0123:     # -- Private engine loader --------------------------------------
0124: 
0125:     def _engines(self):
0126:         if self._retrieval is None:
0127:             from app.engines.memory import get_memory_engine
0128:             from app.engines.temporal import get_temporal_engine
0129:             from app.engines.retrieval import get_retrieval_engine
0130:             self._memory = get_memory_engine()
0131:             self._temporal = get_temporal_engine()
0132:             self._temporal.bind_memory(self._memory)
0133:             self._retrieval = get_retrieval_engine(self._memory, self._temporal)
0134:         return self._memory, self._temporal, self._retrieval
0135: 
0136:     def _init_schema(self):
0137:         import sqlite3
0138:         from app.db.models import MIGRATIONS, run_schema_upgrades
0139:         Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
0140:         conn = sqlite3.connect(self.db_path)
0141:         conn.executescript(MIGRATIONS)
0142:         run_schema_upgrades(conn)
0143:         # Phase 6 sequence_number column
0144:         try:
0145:             conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
0146:             conn.execute(
0147:                 "CREATE INDEX IF NOT EXISTS idx_rel_seq "
0148:                 "ON relationships(user_id, sequence_number)"
0149:             )
0150:         except Exception:
0151:             pass
0152:         conn.commit()
0153:         conn.close()
0154: 
0155:     # -- Public API -------------------------------------------------
0156: 
0157:     def ingest(
0158:         self,
0159:         text: str,
0160:         *,
0161:         source_timestamp: Optional[str] = None,
0162:         speaker: Optional[str] = None,
0163:         speaker_is_user: bool = True,
0164:         confidence: float = 0.9,
0165:         model_response: Optional[str] = None,
0166:     ) -> int:
0167:         """Extract triples from raw text and store them.
0168: 
0169:         The singular write path runs:
0170:         structural cleanup -> grammar correction -> grammar engine
0171:         extraction -> trace-primary persistence.
0172: 
0173:         There is no alternate extraction branch in the SDK path.
0174: 
0175:         If model_response is provided and non-empty, runs the same
0176:         extraction pipeline on the model's response and stores those
0177:         triples with source_tag="model_comprehension". The tag is fixed
0178:         and not caller-controlled.
0179: 
0180:         Args:
0181:           text: the raw user utterance or conversation turn.
0182:           source_timestamp: ISO datetime of the utterance. Used by the
0183:                             temporal engine for supersession + cluster
0184:                             recency.
0185:           speaker: if provided, 'I'/'me'/'myself' in extracted triples
0186:                    are resolved to this name.
0187:           confidence: 0.0â€”1.0 confidence for every resulting triple.
0188:           model_response: the model's response text. If non-empty, triples
0189:                           extracted from it are stored with
0190:                           source_tag="model_comprehension".
0191: 
0192:         Returns:
0193:           Combined number of triples stored from both passes.
0194:         """
0195:         memory, _, _ = self._engines()
0196:         count = memory.ingest_text(
0197:             user_id=self.user_id,
0198:             text=text,
0199:             source_timestamp=source_timestamp,
0200:             speaker=speaker,
0201:             speaker_is_user=speaker_is_user,
0202:             confidence=confidence,
0203:         )
0204:         if model_response and model_response.strip():
0205:             count += memory.ingest_text(
0206:                 user_id=self.user_id,
0207:                 text=model_response,
0208:                 source_timestamp=source_timestamp,
0209:                 speaker=speaker,
0210:                 speaker_is_user=speaker_is_user,
0211:                 confidence=confidence,
0212:                 source_tag="model_comprehension",
0213:             )
0214:         return count
0215: 
0216:     def retrieve(self, query: str) -> Union[Answer, Situation]:
0217:         """Query the continuity layer.
0218: 
0219:         Situational queries ('summarize X', 'why is Y', 'what's
0220:         happening with Z') route to reconstruction and return a
0221:         Situation. Lookup queries ('who is Maya's manager?') route to
0222:         Filter->Complete and return an Answer.
0223: 
0224:         Routing is determined by WH-grammar + discourse verbs in the
0225:         query â€” not by a flag.
0226:         """
0227:         # Route situational queries to the reconstruction path.
0228:         # When explicit_reconstruct_only is True, only fire reconstruct()
0229:         # if the query contains an explicit trigger word ("reconstruct",
0230:         # "what's going on", "summarize", "tell me about").
0231:         # When False, is_situational() routes automatically via WH-grammar.
0232:         from config.settings import settings
0233:         try:
0234:             if settings.explicit_reconstruct_only:
0235:                 _lower = query.lower()
0236:                 _triggers = ("reconstruct", "what's going on", "whats going on",
0237:                              "summarize", "tell me about", "what is going on",
0238:                              "what's happening", "whats happening",
0239:                              "how is everything", "catch me up",
0240:                              "give me a summary", "overview")
0241:                 if any(t in _lower for t in _triggers):
0242:                     return self.reconstruct(query)
0243:             else:
0244:                 from app.engines.wh_type import is_situational
0245:                 if is_situational(query):
0246:                     return self.reconstruct(query)
0247:         except Exception:
0248:             pass  # fail-open: fall through to lookup
0249: 
0250:         _, _, retrieval = self._engines()
0251:         return retrieval.retrieve(self.user_id, query)
0252: 
0253:     def forget(
0254:         self,
0255:         by: str,            # 'entity' | 'time_range' | 'source' | 'triple_id'
0256:         scope,              # str | int | (str, str) tuple / list
0257:     ) -> int:
0258:         """Tombstone memory by entity, time range, source tag, or triple id.
0259: 
0260:         Returns the count of tombstones emitted. Idempotent â€” calling
0261:         again with the same scope returns 0.
0262:         """
0263:         memory, _, _ = self._engines()
0264:         if by == 'entity':
0265:             return memory.forget_by_entity(self.user_id, scope)
0266:         if by == 'time_range':
0267:             start, end = scope  # tuple/list unpack
0268:             return memory.forget_by_time_range(self.user_id, start, end)
0269:         if by == 'source':
0270:             return memory.forget_by_source(self.user_id, scope)
0271:         if by == 'triple_id':
0272:             return memory.forget_by_triple_id(self.user_id, int(scope))
0273:         raise ValueError(f"Unknown forget scope: {by}")
0274: 
0275:     def show(
0276:         self,
0277:         facet: str,         # 'time' | 'entity' | 'source' | 'trace'
0278:         value: Optional[str] = None,
0279:         limit: int = 100,
0280:         export_raw_text: bool = False,
0281:     ) -> list:
0282:         """Return a browsable view of stored memory.
0283: 
0284:         Never returns trace internals (edge_*, embeddings, convergence
0285:         state, salience). If export_raw_text=True, includes the original
0286:         source_text field. Otherwise, returns only fact summaries.
0287:         """
0288:         memory, _, _ = self._engines()
0289:         rows = memory.list_by_facet(
0290:             self.user_id, facet, value=value, limit=limit
0291:         )
0292:         if not export_raw_text:
0293:             for r in rows:
0294:                 r.pop('source_text', None)
0295:         return rows
0296: 
0297:     def trace(self, subject: str, predicate: str) -> list:
0298:         """Return the full supersession history for a subject+predicate pair.
0299: 
0300:         Surfaces every object value ever stored for this (subject, predicate)
0301:         combination â€” both active and superseded â€” ordered oldest to newest.
0302:         Each entry is a dict with keys: id, object, is_active,
0303:         first_learned_at, source_timestamp, source_tag, superseded_by,
0304:         superseded_at, sequence_number.
0305: 
0306:         Pure delegation to MemoryEngine.trace(). No logic added here.
0307:         """
0308:         memory, _, _ = self._engines()
0309:         return memory.trace(self.user_id, subject, predicate)
0310: 
0311:     def check_proactive(self) -> list:
0312:         """Return proactive insights (arcs due for surfacing).
0313: 
0314:         Queries the arcs table for open arcs. Returns a list of
0315:         ProactiveInsight dataclasses. Returns [] if the arcs table
0316:         doesn't exist or no arcs are due.
0317:         """
0318:         from app.engines.proactive import get_proactive_engine
0319:         memory, temporal, _ = self._engines()
0320:         proactive = get_proactive_engine(memory, temporal)
0321:         return proactive.evaluate_due(self.user_id)
0322: 
0323:     def profile(self):
0324:         """Return the current adaptation profile for this user.
0325: 
0326:         Returns an AdaptationProfile dataclass with warmth, formality,
0327:         initiative, and check_in_frequency dimensions (all 0.0â€”1.0).
0328:         """
0329:         from app.engines.adaptability import get_adaptability_engine
0330:         memory, _, _ = self._engines()
0331:         adapt = get_adaptability_engine(memory)
0332:         return adapt.profile(self.user_id)
0333: 
0334: 
0335:     def process(
0336:         self,
0337:         text: str,
0338:         *,
0339:         speaker: str = "user",
0340:         speaker_is_user: bool = True,
0341:         source_timestamp: Optional[str] = None,
0342:         model_response: Optional[str] = None,
0343:         check_proactive: bool = False,
0344:     ) -> ProcessResult:
0345:         """Unified entry point. The architecture decides everything.
0346: 
0347:         Classifies intent via the grammar engine, routes to the correct
0348:         internal capability, and returns a ProcessResult.
0349: 
0350:         Args:
0351:           text: raw user utterance. Empty string with check_proactive=True
0352:                 triggers proactive-only mode.
0353:           speaker: identity for pronoun resolution in extraction.
0354:           source_timestamp: ISO datetime of the utterance.
0355:           model_response: assistant reply text for model_comprehension storage.
0356:           check_proactive: if True and text is empty, return proactive insights
0357:                            only. If True and text is non-empty, proactive
0358:                            insights are appended to store results.
0359: 
0360:         Returns:
0361:           ProcessResult with action discriminator and typed result.
0362:         """
0363:         # Proactive-only mode: no text, just check for due arcs.
0364:         if check_proactive and not text.strip():
0365:             insights = self.check_proactive()
0366:             return ProcessResult(
0367:                 action="proactive",
0368:                 result=insights,
0369:                 proactive=insights,
0370:             )
0371: 
0372:         # Classify intent via grammar engine (primary) or structural fallback.
0373:         is_question = False
0374:         is_command = False
0375:         is_backchannel = False
0376: 
0377:         try:
0378:             from app.engines.grammar_engine import process as grammar_process
0379:             gram = grammar_process(text, speaker=speaker)
0380:             is_question = gram.classification.is_question
0381:             is_command = gram.classification.is_command
0382:             is_backchannel = gram.classification.is_backchannel
0383:         except ImportError:
0384:             # Structural fallback: ? = question, imperative verbs = command
0385:             stripped = text.strip()
0386:             if stripped.endswith("?"):
0387:                 is_question = True
0388:             elif stripped.lower().split()[0] in ("forget", "delete", "remove") if stripped else False:
0389:                 is_command = True
0390: 
0391:         # BACKCHANNEL â€” skip entirely
0392:         if is_backchannel:
0393:             return ProcessResult(action="skipped")
0394: 
0395:         # COMMAND â€” route to forget
0396:         if is_command:
0397:             target = self._parse_forget_target(text)
0398:             if target:
0399:                 count = self.forget(by="entity", scope=target)
0400:                 return ProcessResult(action="forgot", result=count)
0401:             return ProcessResult(action="skipped")
0402: 
0403:         # QUESTION â€” sub-route: situational vs factual
0404:         # Respects explicit_reconstruct_only setting (same logic as retrieve())
0405:         if is_question:
0406:             try:
0407:                 if settings.explicit_reconstruct_only:
0408:                     _lower = text.lower()
0409:                     _triggers = ("reconstruct", "what's going on", "whats going on",
0410:                                  "summarize", "tell me about", "what is going on",
0411:                                  "what's happening", "whats happening",
0412:                                  "how is everything", "catch me up",
0413:                                  "give me a summary", "overview")
0414:                     if any(t in _lower for t in _triggers):
0415:                         situation = self.reconstruct(text)
0416:                         return ProcessResult(action="reconstructed", result=situation)
0417:                 else:
0418:                     from app.engines.wh_type import is_situational
0419:                     if is_situational(text):
0420:                         situation = self.reconstruct(text)
0421:                         return ProcessResult(action="reconstructed", result=situation)
0422:             except Exception:
0423:                 pass
0424:             answer = self.retrieve(text)
0425:             return ProcessResult(action="answered", result=answer)
0426: 
0427:         # STATEMENT â€” write path + optional proactive check
0428:         count = self.ingest(
0429:             text,
0430:             speaker=speaker,
0431:             speaker_is_user=speaker_is_user,
0432:             source_timestamp=source_timestamp,
0433:             model_response=model_response,
0434:         )
0435:         insights = self.check_proactive() if check_proactive else []
0436:         return ProcessResult(
0437:             action="stored",
0438:             result=count,
0439:             proactive=insights,
0440:             triples_stored=count,
0441:         )
0442: 
0443:     @staticmethod
0444:     def _parse_forget_target(text: str) -> Optional[str]:
0445:         """Extract the entity target from a forget/delete/remove command.
0446: 
0447:         Strips the imperative verb and common prepositions to isolate the
0448:         entity name. Returns None if nothing remains.
0449:         """
0450:         stripped = text.strip().rstrip(".").rstrip("!")
0451:         tokens = stripped.split()
0452:         if not tokens:
0453:             return None
0454:         # Drop the imperative verb
0455:         rest = tokens[1:]
0456:         # Drop leading prepositions ("about", "all about", "everything about")
0457:         skip = {"about", "all", "everything"}
0458:         while rest and rest[0].lower() in skip:
0459:             rest = rest[1:]
0460:         target = " ".join(rest).strip()
0461:         return target if target else None
0462: 
0463:     def reconstruct(self, query: str) -> Situation:
0464:         """Force the reconstruction path.
0465: 
0466:         Useful when the caller knows they want a structured Situation
0467:         (for dashboards, summarization, analytical workflows) even when
0468:         the query form doesn't naturally trigger situational routing.
0469:         Returns a Situation always; never an Answer.
0470:         """
0471:         _, _, retrieval = self._engines()
0472:         return retrieval.reconstruct(self.user_id, query)
0473: 
0474: 
0475: # -- Module-level singleton -----------------------------------------------
0476: 
0477: _singleton: Optional[Kenotic] = None
0478: 
0479: 
0480: def KenoticV1(
0481:     text: str,
0482:     *,
0483:     speaker: str = "user",
0484:     speaker_is_user: bool = True,
0485:     source_timestamp: Optional[str] = None,
0486:     model_response: Optional[str] = None,
0487:     check_proactive: bool = False,
0488:     user_id: int = 0,
0489:     db_path: Union[str, Path] = "~/.kenotic/memory.db",
0490:     embed_device: str = "cuda",
0491: ):
0492:     """The entire Kenotic continuity architecture in ONE function call.
0493: 
0494:     Send text. Get continuity. The architecture handles everything:
0495:     - Statements  -> grammar engine -> typed extraction -> 5-trace store -> supersession -> proactive arcs
0496:     - Questions   -> classify -> situational reconstruction OR factual lookup
0497:     - Commands    -> parse target -> soft tombstone
0498:     - Backchannels -> skip
0499: 
0500:     First call initializes the engine stack (lazy). Subsequent calls reuse it.
0501:     Runtime verifier (144 checks) runs once at init.
0502: 
0503:     Args:
0504:         text: any English text -- statement, question, command, anything.
0505:         speaker: who said it (for pronoun resolution).
0506:         source_timestamp: ISO datetime of the utterance.
0507:         model_response: the AI's response (stored as model_comprehension).
0508:         check_proactive: if True with empty text, returns arcs due for surfacing.
0509:         user_id: partition key (default 0 for single-user).
0510:         db_path: SQLite file path.
0511:         embed_device: 'cuda' or 'cpu'.
0512: 
0513:     Returns:
0514:         ProcessResult -- contains action, result, proactive insights, triples_stored.
0515:     """
0516:     global _singleton
0517:     if _singleton is None or _singleton.db_path != str(db_path) or _singleton.user_id != int(user_id):
0518:         _singleton = Kenotic(user_id=user_id, db_path=db_path, embed_device=embed_device)
0519: 
0520:     return _singleton.process(
0521:         text,
0522:         speaker=speaker,
0523:         speaker_is_user=speaker_is_user,
0524:         source_timestamp=source_timestamp,
0525:         model_response=model_response,
0526:         check_proactive=check_proactive,
0527:     )
```
