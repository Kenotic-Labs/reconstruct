# Engine Truth Master

This file has two parts:
1. A code-backed connection map.
2. The literal line-numbered source of each engine/db file.

## How They Connect

Write path, as directly wired in code:
1. `sdk.client.KenoticV1(...)` calls `Kenotic.process(...)` in `sdk/client.py`.
2. `Kenotic.process(...)` routes statements to `Kenotic.ingest(...)`.
3. `Kenotic.ingest(...)` calls `MemoryEngine.ingest_text(...)`.
4. `MemoryEngine.ingest_text(...)` calls `MemoryEngine._run_ingestion_path(...)`.
5. `_run_ingestion_path(...)` calls `sentence_model.cleanup(...)`.
6. `_run_ingestion_path(...)` then calls `grammar_engine.process(...)`.
7. `MemoryEngine.ingest_text(...)` flattens grammar output and calls `MemoryEngine.store(...)` once per row.
8. `MemoryEngine.store(...)` calls `_prepare_row(...)` then `_write_row(...)`.
9. `MemoryEngine.store(...)` then runs side-effects including fact/milestone routing, predicted query writes, temporal event-date resolution, supersession detection, cluster assignment, and arc detection.
10. `TemporalEngine` is bound to `MemoryEngine` in `Kenotic._engines()` via `self._temporal.bind_memory(self._memory)`.
11. `app/db/session.py` provides `get_db_context()` and connection setup used by persistence-facing engines.
12. `app/db/models.py` defines `MIGRATIONS` and `run_schema_upgrades(...)`, which `sdk/client.py` uses in `_init_schema()`.

Read path, as directly wired in code:
1. `Kenotic.process(...)` routes questions to `Kenotic.retrieve(...)` or `Kenotic.reconstruct(...)`.
2. `Kenotic.retrieve(...)` can route to reconstruction based on settings/WH helpers; otherwise it calls the retrieval engine.
3. `Kenotic.reconstruct(...)` calls `retrieval.reconstruct(...)`.

Database ownership visible in code:
- `sdk/client.py` initializes schema.
- `memory.py` writes and updates relationship-centered memory rows and related side tables.
- `temporal.py` reads/writes time-related relationship and arc state.
- `session.py` owns connection creation/context management.
- `models.py` owns schema/migrations.

## sdk/client.py

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

## app/engines/sentence_model.py

Source: [app/engines/sentence_model.py](/D:/Nura/Code/nura_living_memory_code/app/engines/sentence_model.py)

```text
0001: """
0002: SentenceModel â€” grammar-correcting seq2seq for reconstruction rendering.
0003: 
0004: Purpose: take an ungrammatical template-rendered sentence
0005: ("You has emotion sad.") and return its grammatical form
0006: ("You have an emotion of sadness." or similar).
0007: 
0008: The model is `jbochi/coedit-small` (flan-t5-small fine-tuned on CoEdit data, 77M params, ~300MB).
0009: Falls back to grammarly/coedit-large if small is unavailable.
0010: 
0011: Architectural contract:
0012:   - Polishing is LOSSLESS w.r.t. grounding. Input is one sentence
0013:     (derived from one edge). Output is one sentence (same edge).
0014:     Grounding map is preserved by the caller â€” this module does not
0015:     manipulate grounding.
0016:   - Model is OPTIONAL. If load fails or the flag is off, the original
0017:     template sentence is returned unchanged.
0018:   - Per-sentence cache: same input â†’ same output, zero recomputation.
0019:     Cache key is the input sentence string.
0020: 
0021: Public API:
0022:     polish(sentence: str) -> str
0023:     cleanup(text: str) -> str
0024:     is_available() -> bool
0025:     reset_cache()
0026: """
0027: from __future__ import annotations
0028: 
0029: import logging
0030: import os
0031: from threading import Lock
0032: from typing import Dict, Optional
0033: 
0034: logger = logging.getLogger(__name__)
0035: 
0036: # -----------------------------------------------------------------------
0037: # Model rules â€” "system prompts" for off-the-shelf models
0038: # -----------------------------------------------------------------------
0039: 
0040: # CoEdit task prefix â€” ONLY grammar correction. Other tasks (simplify,
0041: # paraphrase, formality) change meaning, violating the lossless contract.
0042: _COEDIT_TASK_PREFIX = "Fix grammatical errors in this sentence:"
0043: 
0044: # Structural cleanup rules â€” spaCy dep labels that signal noise
0045: _FILLER_POS = frozenset({"INTJ"})
0046: _SCAFFOLDING_DEPS = frozenset({"parataxis"})
0047: _DISCOURSE_FRAME_LEMMAS = frozenset({
0048:     "wait", "remind", "mean", "say", "tell", "know",
0049:     "think", "wonder", "guess", "suppose", "remember",
0050:     "hear", "listen", "look",
0051: })
0052: _DISCOURSE_SUBJECT_LEMMAS = frozenset({"i", "we", "you"})
0053: _DISCOURSE_FILLER_NOUNS = frozenset({
0054:     "thing", "point", "deal", "fact", "truth", "matter",
0055:     "problem", "issue", "question",
0056: })
0057: _IDIOM_FRAMES = frozenset({
0058:     "long story short", "bottom line", "at the end of the day",
0059:     "truth be told", "between you and me", "to be honest",
0060:     "to be fair", "for what it's worth", "if you ask me",
0061:     "believe it or not", "here's the thing", "here's the deal",
0062: })
0063: _RETRACTION_PHRASES = frozenset({
0064:     "never mind", "nevermind", "forget it", "forget that",
0065:     "scratch that", "disregard that", "ignore that",
0066: })
0067:     # No contraction map â€” spaCy was trained on web text including informal
0068:     # speech. Regex replacement before parse changes tokenization and can
0069:     # break dep trees. Let spaCy handle what it was trained on.
0070: 
0071: _MODEL = None
0072: _TOKENIZER = None
0073: _DEVICE: Optional[str] = None
0074: _LOCK = Lock()
0075: _UNAVAILABLE = False
0076: _CACHE: Dict[str, str] = {}
0077: 
0078: _ENV_MODEL = "RAYA_SENTENCE_MODEL"
0079: _ENV_DEVICE = "RAYA_SENTENCE_DEVICE"
0080: _ENV_ENABLE = "RAYA_SENTENCE_POLISH"   # set to "0" to disable, default ON
0081: _DEFAULT_MODEL = "models/coedit-raya"
0082: 
0083: 
0084: def is_enabled() -> bool:
0085:     """Polish is on by default. Set RAYA_SENTENCE_POLISH=0 to disable."""
0086:     return os.environ.get(_ENV_ENABLE, "1") != "0"
0087: 
0088: 
0089: def is_available() -> bool:
0090:     """True if the model loaded successfully and polishing can run."""
0091:     return _MODEL is not None and not _UNAVAILABLE
0092: 
0093: 
0094: def _load() -> bool:
0095:     global _MODEL, _TOKENIZER, _DEVICE, _UNAVAILABLE
0096:     if _UNAVAILABLE:
0097:         return False
0098:     if _MODEL is not None:
0099:         return True
0100:     with _LOCK:
0101:         if _MODEL is not None:
0102:             return True
0103:         if _UNAVAILABLE:
0104:             return False
0105:         try:
0106:             import torch
0107:             from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
0108: 
0109:             dev = os.environ.get(_ENV_DEVICE,
0110:                                  os.environ.get("RAYA_EMBED_DEVICE", "cuda"))
0111:             # GPU only â€” no CPU fallback
0112: 
0113:             # Force offline â€” load from local cache, never ping HF
0114:             os.environ["HF_HUB_OFFLINE"] = "1"
0115:             os.environ["TRANSFORMERS_OFFLINE"] = "1"
0116: 
0117:             name = os.environ.get(_ENV_MODEL, _DEFAULT_MODEL)
0118:             tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
0119:             model = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=True)
0120:             if dev == "cuda":
0121:                 model = model.half()
0122:             model = model.to(dev).eval()
0123: 
0124:             _MODEL = model
0125:             _TOKENIZER = tok
0126:             _DEVICE = dev
0127:             print(f"[SentenceModel] Loaded {name} on {dev} (fp16={dev=='cuda'})")
0128:             return True
0129:         except (OSError, ImportError, ValueError) as e:
0130:             # Model not installed or transformers not available â€” expected.
0131:             print(f"[SentenceModel] Unavailable ({e}); polishing disabled")
0132:             _UNAVAILABLE = True
0133:             return False
0134:         except Exception as e:
0135:             # Unexpected error (CUDA init failure, corrupt weights, etc.)
0136:             # â€” propagate so the caller sees it rather than silently degrading.
0137:             raise RuntimeError(
0138:                 f"[SentenceModel] Unexpected error loading model: {e}"
0139:             ) from e
0140: 
0141: 
0142: def polish(sentence: str) -> str:
0143:     """Polish one sentence via CoEdit grammar error correction.
0144: 
0145:     Model: jbochi/coedit-small (77M params, flan-t5-small fine-tuned)
0146:     Task prefix: "Fix grammatical errors in this sentence:" (GEC task)
0147:     Max input: 128 tokens (truncated by tokenizer)
0148:     Max output: 96 tokens (per sentence â€” callers must split multi-sentence input)
0149:     Beam search: 2 beams, no sampling (deterministic)
0150:     Source: CoEdIT paper (EMNLP 2023, Raheja et al.)
0151: 
0152:     Supported models (via RAYA_SENTENCE_MODEL env var):
0153:         - jbochi/coedit-small (default, 77M params, fastest)
0154:         - grammarly/coedit-large (770M params, higher quality, needs GPU)
0155:         - grammarly/coedit-xl (3B params, highest quality, needs large GPU)
0156: 
0157:     Returns input unchanged if polish disabled, model unavailable, or
0158:     sentence is trivial."""
0159:     if not sentence or not sentence.strip():
0160:         return sentence
0161:     if not is_enabled():
0162:         return sentence
0163: 
0164:     key = sentence.strip()
0165:     if key in _CACHE:
0166:         return _CACHE[key]
0167: 
0168:     if not _load():
0169:         return sentence
0170: 
0171:     try:
0172:         import torch
0173:         prompt = _COEDIT_TASK_PREFIX + " " + key
0174:         inp = _TOKENIZER(
0175:             prompt, return_tensors="pt", max_length=128, truncation=True,
0176:         ).to(_DEVICE)
0177:         with torch.no_grad():
0178:             out = _MODEL.generate(
0179:                 **inp, max_length=96, num_beams=2, do_sample=False,
0180:             )
0181:         result = _TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
0182:         if not result:
0183:             logger.warning(
0184:                 "[SentenceModel] CoEdit returned empty output for: %r "
0185:                 "â€” model may be broken, returning input unchanged", key
0186:             )
0187:             result = sentence
0188:         _CACHE[key] = result
0189:         return result
0190:     except (RuntimeError, ValueError) as e:
0191:         # RuntimeError: CUDA OOM or tensor errors
0192:         # ValueError: tokenizer encoding issues
0193:         logger.warning("[SentenceModel] polish error (%s); returning input unchanged", e)
0194:         _CACHE[key] = sentence
0195:         return sentence
0196: 
0197: 
0198: def cleanup(text: str, speaker: str = None) -> str:
0199:     """Two-pass dialogue cleanup:
0200: 
0201:     Pass 1 (structural): spaCy dep-label rules from pipeline_config.
0202:       - 7 detection patterns, all structural
0203:       - No model inference beyond spaCy's dep parse
0204: 
0205:     Pass 2 (model): CoEdit grammar polish (GEC only).
0206:       - One task prefix, one model, semantic guards
0207:       - Does NOT change meaning â€” only fixes grammar
0208: 
0209:     Rules are defined in pipeline_config.py (the "system prompt" for each model).
0210:     """
0211:     if not text or not text.strip():
0212:         return text or ""
0213: 
0214: 
0215:     # -- Pass 1: structural cleanup via spaCy dep labels --
0216:     stripped = text.strip()
0217:     try:
0218:         from app.engines.grammar_engine import _get_nlp, _get_root
0219:         _nlp = _get_nlp()
0220: 
0221:         # Step 5: Retraction detection (before full parse â€” cheap check)
0222:         _lower_stripped = stripped.lower().rstrip(".,!? ")
0223:         for phrase in _RETRACTION_PHRASES:
0224:             if _lower_stripped.endswith(phrase):
0225:                 return ""  # speaker withdrew â€” discard entire utterance
0226: 
0227:         _doc = _nlp(stripped)
0228: 
0229:         # Step 0: rhetorical/discourse question followed by factual answer.
0230:         # Keep the factual answer clause and drop the scaffolding question.
0231:         _raw_segments = [seg.strip() for seg in stripped.split("?") if seg.strip()]
0232:         if len(_raw_segments) >= 2:
0233:             _first_doc = _nlp(_raw_segments[0] + "?")
0234:             _first_root = _get_root(_first_doc)
0235:             _first_text = _raw_segments[0].lower()
0236:             _is_discourse_question = (
0237:                 _first_text.startswith("you know")
0238:                 or any(_first_text.startswith(frame) for frame in _IDIOM_FRAMES)
0239:                 or (
0240:                     _first_root is not None
0241:                     and _first_root.lemma_.lower() in _DISCOURSE_FRAME_LEMMAS
0242:                 )
0243:             )
0244:             if _is_discourse_question:
0245:                 _tail = "? ".join(_raw_segments[1:]).strip()
0246:                 if _tail:
0247:                     stripped = _tail
0248:                     _doc = _nlp(stripped)
0249: 
0250:         # Step 1: Collect indices to remove
0251:         _remove_indices = set()
0252: 
0253:         # 1a: INTJ tokens (POS-based)
0254:         for tok in _doc:
0255:             if tok.pos_ in _FILLER_POS:
0256:                 _remove_indices.add(tok.i)
0257: 
0258:         # 1b: Parataxis subtrees (dep-based)
0259:         for tok in _doc:
0260:             if tok.dep_ in _SCAFFOLDING_DEPS:
0261:                 _remove_indices |= {t.i for t in tok.subtree}
0262: 
0263:         # Step 3: Discourse frame detection
0264:         _root = _get_root(_doc)
0265: 
0266:         # 3a: ROOT or ccomp is discourse frame verb
0267:         if _root and _root.pos_ in ("VERB", "AUX"):
0268:             _frame_verb = None
0269:             _content_verb = None
0270: 
0271:             # Check ROOT as frame
0272:             if _root.lemma_.lower() in _DISCOURSE_FRAME_LEMMAS:
0273:                 _subj_ok = any(
0274:                     c.dep_ in ("nsubj", "nsubjpass")
0275:                     and c.text.lower() in _DISCOURSE_SUBJECT_LEMMAS
0276:                     for c in _root.children
0277:                 )
0278:                 if _subj_ok:
0279:                     # Find content clause with its own subject
0280:                     for c in _root.children:
0281:                         if (c.dep_ in ("ccomp", "xcomp", "conj", "parataxis")
0282:                                 and c.pos_ in ("VERB", "AUX")
0283:                                 and any(gc.dep_ in ("nsubj", "nsubjpass")
0284:                                         for gc in c.children)):
0285:                             # Guard: only strip if content subject differs
0286:                             # from ROOT subject (else it's the same person's fact)
0287:                             _root_subj = next(
0288:                                 (rc for rc in _root.children
0289:                                  if rc.dep_ in ("nsubj", "nsubjpass")), None
0290:                             )
0291:                             _content_subj = next(
0292:                                 (gc for gc in c.children
0293:                                  if gc.dep_ in ("nsubj", "nsubjpass")), None
0294:                             )
0295:                             if (_root_subj and _content_subj
0296:                                     and _root_subj.text.lower()
0297:                                     != _content_subj.text.lower()):
0298:                                 _frame_verb = _root
0299:                                 _content_verb = c
0300:                             break
0301: 
0302:             # Check ROOT as filler noun frame:
0303:             # "The thing is, I applied" â†’ ROOT=is, nsubj=thing, ccomp=applied
0304:             if _frame_verb is None and _root.lemma_.lower() == "be":
0305:                 _root_subj = next(
0306:                     (c for c in _root.children
0307:                      if c.dep_ in ("nsubj", "nsubjpass")), None
0308:                 )
0309:                 if (_root_subj
0310:                         and _root_subj.lemma_.lower() in _DISCOURSE_FILLER_NOUNS):
0311:                     _content_ccomp = next(
0312:                         (c for c in _root.children
0313:                          if c.dep_ == "ccomp" and c.pos_ in ("VERB", "AUX")),
0314:                         None,
0315:                     )
0316:                     if _content_ccomp is not None:
0317:                         _frame_verb = _root
0318:                         _content_verb = _content_ccomp
0319: 
0320:             # Check ccomp as frame (discourse in subordinate position)
0321:             if _frame_verb is None:
0322:                 for c in _root.children:
0323:                     if c.dep_ == "ccomp" and c.pos_ in ("VERB", "AUX"):
0324:                         if c.lemma_.lower() == "be":
0325:                             _ccomp_subj = next(
0326:                                 (gc for gc in c.children
0327:                                  if gc.dep_ in ("nsubj", "nsubjpass")), None
0328:                             )
0329:                             if (_ccomp_subj and _ccomp_subj.lemma_.lower()
0330:                                     in _DISCOURSE_FILLER_NOUNS):
0331:                                 _frame_verb = c
0332:                                 _content_verb = _root
0333:                                 break
0334: 
0335:             if _frame_verb is not None and _content_verb is not None:
0336:                 _content_indices = {t.i for t in _content_verb.subtree}
0337:                 _frame_indices = {t.i for t in _doc} - _content_indices
0338:                 # Keep punctuation
0339:                 _frame_indices -= {t.i for t in _doc if t.pos_ == "PUNCT"}
0340:                 _remove_indices |= _frame_indices
0341: 
0342:         # 3b: Idiom adverbial frames at sentence start
0343:         _sent_text_lower = stripped.lower()
0344:         for idiom in _IDIOM_FRAMES:
0345:             if _sent_text_lower.startswith(idiom):
0346:                 # Remove the full idiom span as tokenized by spaCy, not
0347:                 # by naive whitespace. This handles "here's the deal"
0348:                 # -> ["here", "'s", "the", "deal"].
0349:                 _idiom_doc = _nlp(idiom)
0350:                 _idiom_tokens = [
0351:                     t.text.lower() for t in _idiom_doc
0352:                     if not t.is_space
0353:                 ]
0354:                 _removed = 0
0355:                 for tok in _doc:
0356:                     if _removed < len(_idiom_tokens):
0357:                         _remove_indices.add(tok.i)
0358:                         if tok.text.lower() == _idiom_tokens[_removed]:
0359:                             _removed += 1
0360:                     elif tok.text == ",":
0361:                         _remove_indices.add(tok.i)
0362:                     else:
0363:                         break
0364:                 break
0365: 
0366:         # Strip rhetorical tail fragments left after idiom removal:
0367:         # "Deal? I work at Google..." -> "I work at Google..."
0368:         _clean_probe = [tok for tok in _doc if tok.i not in _remove_indices]
0369:         if _clean_probe:
0370:             _probe_tokens = [tok.text for tok in _clean_probe]
0371:             while _probe_tokens:
0372:                 if not _clean_probe:
0373:                     break
0374:                 _first = _probe_tokens[0].lower()
0375:                 if _first in _DISCOURSE_FILLER_NOUNS:
0376:                     _remove_indices.add(_clean_probe[0].i)
0377:                     _clean_probe = _clean_probe[1:]
0378:                     _probe_tokens = [tok.text for tok in _clean_probe]
0379:                     continue
0380:                 if _first in ("?", ",", ":", ";", "-", "â€”", "â€“"):
0381:                     _remove_indices.add(_clean_probe[0].i)
0382:                     _clean_probe = _clean_probe[1:]
0383:                     _probe_tokens = [tok.text for tok in _clean_probe]
0384:                     continue
0385:                 if len(_clean_probe) > 1 and _clean_probe[1].text == "?":
0386:                     _remove_indices.add(_clean_probe[0].i)
0387:                     _remove_indices.add(_clean_probe[1].i)
0388:                     _clean_probe = _clean_probe[2:]
0389:                     _probe_tokens = [tok.text for tok in _clean_probe]
0390:                 else:
0391:                     break
0392: 
0393:         # Build cleaned text
0394:         clean_tokens = [tok for tok in _doc if tok.i not in _remove_indices]
0395: 
0396:         if clean_tokens:
0397:             _normalized_tokens = []
0398:             _last_text = None
0399:             for tok in clean_tokens:
0400:                 _text = tok.text
0401:                 if (_text == "," and (_last_text is None or _last_text == ",")):
0402:                     continue
0403:                 _normalized_tokens.append(_text)
0404:                 _last_text = _text
0405:             stripped = " ".join(tok.text for tok in clean_tokens).strip()
0406:             if _normalized_tokens:
0407:                 stripped = " ".join(_normalized_tokens).strip()
0408:             stripped = " ".join(stripped.split())
0409:             # Strip leading conjunctions left after filler removal
0410:             if stripped:
0411:                 first_word = stripped.split()[0].lower()
0412:                 if first_word in ("and", "but", "so", "or"):
0413:                     stripped = " ".join(stripped.split()[1:])
0414: 
0415:         # Guard: if cleanup stripped everything, return original
0416:         if not stripped or not stripped.strip():
0417:             stripped = text.strip()
0418: 
0419:     except (OSError, ImportError):
0420:         pass  # spaCy not installed â€” use raw text
0421: 
0422:     if not stripped:
0423:         stripped = text.strip()
0424: 
0425:     # Pass 2: CoEdit "Rewrite to be formal:" per-sentence.
0426:     # coedit-small has max_length=96 output tokens. Multi-sentence
0427:     # turns get truncated. Fix: split into sentences, rewrite each
0428:     # individually, rejoin. No content loss.
0429:     if not is_enabled():
0430:         return stripped
0431: 
0432:     key = ("cleanup", stripped, speaker or "")
0433:     cached = _CACHE.get(key)
0434:     if cached is not None:
0435:         return cached
0436: 
0437:     if not _load():
0438:         return stripped
0439: 
0440:     try:
0441:         import torch
0442: 
0443:         def _rewrite_one(sentence: str) -> str:
0444:             """Rewrite a single sentence via coedit.
0445:             Semantic guard: if CoEdit changes ROOT verb, loses NER entities,
0446:             or significantly changes length, reject the rewrite â€” it changed
0447:             meaning, not just grammar."""
0448:             prompt = _COEDIT_TASK_PREFIX + " " + sentence
0449:             inp = _TOKENIZER(
0450:                 prompt, return_tensors="pt", max_length=128, truncation=True,
0451:             ).to(_DEVICE)
0452:             with torch.no_grad():
0453:                 out = _MODEL.generate(
0454:                     **inp, max_length=96, num_beams=2, do_sample=False,
0455:                 )
0456:             r = _TOKENIZER.decode(out[0], skip_special_tokens=True).strip()
0457:             if not r:
0458:                 logger.warning(
0459:                     "[SentenceModel] CoEdit returned empty output for: %r",
0460:                     sentence,
0461:                 )
0462:                 return sentence
0463:             # Guardrail: reject triple/tuple syntax hallucination
0464:             if "(" in r and ")" in r:
0465:                 oi = r.find("(")
0466:                 ci = r.find(")", oi + 1)
0467:                 if ci != -1 and r[oi + 1:ci].count(",") >= 2:
0468:                     return sentence
0469:             # Semantic guard: compare input vs output via spaCy.
0470:             # If CoEdit changed the ROOT verb, lost NER entities, or
0471:             # significantly shortened the text, it changed meaning â€”
0472:             # reject the rewrite and keep the structurally cleaned input.
0473:             try:
0474:                 from app.engines.grammar_engine import (
0475:                     _get_nlp_fragment, _get_root,
0476:                 )
0477:                 _snlp = _get_nlp_fragment()
0478:                 _in_doc = _snlp(sentence)
0479:                 _out_doc = _snlp(r)
0480:                 # Check 1: NER entities not lost. Cleanup is allowed to
0481:                 # rewrite syntax, split clauses, and improve punctuation,
0482:                 # but it should not drop the named entities that anchor
0483:                 # the stored fact.
0484:                 _in_ents = {e.text.lower() for e in _in_doc.ents}
0485:                 _out_ents = {e.text.lower() for e in _out_doc.ents}
0486:                 if _in_ents and not (_in_ents & _out_ents):
0487:                     logger.warning(
0488:                         "[SentenceModel] CoEdit lost NER entities: %r -> %r",
0489:                         sentence, r,
0490:                     )
0491:                     return sentence
0492:                 # Check 2: output not drastically shorter (content lost)
0493:                 if len(r.split()) < len(sentence.split()) * 0.5:
0494:                     logger.warning(
0495:                         "[SentenceModel] CoEdit truncated content: %r -> %r",
0496:                         sentence, r,
0497:                     )
0498:                     return sentence
0499:                 # Check 3: do not let the rewrite invent a question form
0500:                 # from a declarative cleanup input.
0501:                 if "?" not in sentence and "?" in r:
0502:                     logger.warning(
0503:                         "[SentenceModel] CoEdit changed sentence mood: %r -> %r",
0504:                         sentence, r,
0505:                     )
0506:                     return sentence
0507:             except (ImportError, OSError):
0508:                 pass  # spaCy not available for guard â€” accept rewrite
0509:             return r
0510: 
0511:         # Split into sentences â€” rewrite each individually so
0512:         # coedit-small's 96-token output limit doesn't truncate.
0513:         from app.engines.grammar_engine import _get_nlp
0514:         # Split on sentence boundaries AND ellipsis/dash breaks.
0515:         # spaCy may not split on "..." or "â€”" but these are natural
0516:         # sentence boundaries in conversational text.
0517:         import re
0518:         _presplit = re.split(r'\.{2,}|â€”|â€“', stripped)
0519:         _presplit = [s.strip() for s in _presplit if s.strip()]
0520:         _sentences = []
0521:         for _seg in _presplit:
0522:             _seg_doc = _get_nlp()(_seg)
0523:             _sentences.extend(
0524:                 s.text.strip() for s in _seg_doc.sents if s.text.strip()
0525:             )
0526: 
0527:         if len(_sentences) <= 1:
0528:             result = _rewrite_one(stripped)
0529:         else:
0530:             rewritten = [_rewrite_one(s) for s in _sentences]
0531:             result = " ".join(rewritten)
0532: 
0533:         try:
0534:             _result_doc = _get_nlp()(result)
0535:             _result_sents = [s for s in _result_doc.sents if s.text.strip()]
0536:             if _result_sents:
0537:                 _lead_tokens = [
0538:                     tok for tok in _result_sents[0]
0539:                     if not tok.is_punct and not tok.is_space
0540:                 ]
0541:                 _lead_lemmas = {tok.lemma_.lower() for tok in _lead_tokens}
0542:                 if (
0543:                     _lead_tokens
0544:                     and len(_lead_tokens) <= 3
0545:                     and _lead_lemmas <= (_DISCOURSE_FRAME_LEMMAS | {"actually"})
0546:                 ):
0547:                     result = " ".join(
0548:                         sent.text.strip() for sent in _result_sents[1:]
0549:                     ).strip()
0550: 
0551:             _trim_doc = _get_nlp()(result) if result else None
0552:             if _trim_doc is not None:
0553:                 _trim_tokens = [
0554:                     tok for tok in _trim_doc
0555:                     if not tok.is_space
0556:                 ]
0557:                 if (
0558:                     len(_trim_tokens) >= 3
0559:                     and _trim_tokens[-1].text in (".", "!", "?")
0560:                     and _trim_tokens[-2].lemma_.lower() == "also"
0561:                     and _trim_tokens[-3].lemma_.lower() == "but"
0562:                 ):
0563:                     result = "".join(
0564:                         tok.text_with_ws for tok in _trim_doc[:-3]
0565:                     ).strip()
0566:                     if result:
0567:                         result = result + "."
0568:         except (OSError, ImportError):
0569:             pass
0570: 
0571:         if not result:
0572:             logger.warning(
0573:                 "[SentenceModel] cleanup produced empty result for: %r", stripped
0574:             )
0575:             result = text.strip()
0576: 
0577:         _CACHE[key] = result
0578:         return result
0579:     except (RuntimeError, ValueError) as e:
0580:         # RuntimeError: CUDA OOM or tensor errors
0581:         # ValueError: tokenizer encoding issues
0582:         logger.warning("[SentenceModel] cleanup error (%s); returning input unchanged", e)
0583:         fallback = text.strip()
0584:         _CACHE[key] = fallback
0585:         return fallback
0586: 
0587: 
0588: def reset_cache():
0589:     global _CACHE
0590:     _CACHE = {}
```

## app/engines/grammar_engine.py

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

## app/engines/memory.py

Source: [app/engines/memory.py](/D:/Nura/Code/nura_living_memory_code/app/engines/memory.py)

```text
0001: # ============================================================================
0002: # TRACE-PRIMARY MEMORY ENGINE (2026-04-29 three-phase rewrite)
0003: #
0004: # Three-phase store(): PREPARE (pure computation) -> WRITE (one SQL) -> SIDE-EFFECTS
0005: # No cosine anchors. No triple validation gate.
0006: # If the grammar engine produced a trace decomposition, store it.
0007: # ============================================================================
0008: """
0009: MemoryEngine -- trace-primary store + all derived views.
0010: 
0011: Three-phase store path:
0012:   Phase 1 PREPARE: Extract all column values into a flat dict (no DB access)
0013:   Phase 2 WRITE:   One INSERT or UPDATE for the relationship row + FTS5
0014:   Phase 3 SIDE-EFFECTS: facts/milestones, predicted queries, clustering, arcs
0015: 
0016: Public interface:
0017:     store()                 -> three-phase write + side-effects
0018:     ingest_text()           -> grammar_engine.process() -> store() per decomp
0019:     get_relationships()     -> read triples by any combination of fields
0020:     get_entity()            -> read an entity row by name
0021:     entity_link()           -> resolve a mention to an entity via cosine
0022:     supersede()             -> supersession hook called by TemporalEngine
0023:     summarize()             -> natural-language summary
0024:     forget_by_triple_id()   -> tombstone single triple
0025:     forget_by_entity()      -> tombstone all triples for entity
0026:     forget_by_time_range()  -> tombstone triples in time window
0027:     forget_by_source()      -> tombstone triples by source_tag
0028:     list_by_facet()         -> faceted listing (time/entity/source/trace)
0029:     clean()                 -> singular cleanup path
0030:     extract()               -> singular extraction path
0031: """
0032: from __future__ import annotations
0033: 
0034: import hashlib
0035: import json
0036: import logging
0037: import sqlite3
0038: from dataclasses import dataclass, field
0039: from datetime import datetime, timezone
0040: from typing import Any, Dict, List, Optional, Tuple
0041: 
0042: import numpy as np
0043: 
0044: from app.db.session import get_db_context
0045: from app.utils.cosine import cosine_sim
0046: from app.vector.embedder import embed_text
0047: 
0048: log = logging.getLogger(__name__)
0049: 
0050: 
0051: # =============================================================================
0052: # DATA MODEL
0053: # =============================================================================
0054: 
0055: @dataclass
0056: class Entity:
0057:     name: str
0058:     entity_type: str = "unknown"
0059:     attributes: Dict[str, Any] = field(default_factory=dict)
0060:     embedding: Optional[np.ndarray] = None
0061:     mention_count: int = 1
0062: 
0063: 
0064: @dataclass
0065: class Relationship:
0066:     subject: str
0067:     predicate: str
0068:     object: str
0069:     confidence: float = 0.9
0070:     id: Optional[int] = None
0071:     is_current: bool = True
0072:     sequence_number: Optional[int] = None
0073: 
0074: 
0075: @dataclass
0076: class Traces:
0077:     valence: float = 0.5
0078:     affiliation: float = 0.5
0079:     emotional_intensity: Optional[float] = None
0080:     relational_type: Optional[str] = None
0081:     relational_proximity: float = 0.5
0082:     relational_valence: float = 0.5
0083:     episodic_significance: str = "routine"
0084:     episodic_narrative_position: str = "ongoing"
0085:     temporal_context: Optional[str] = None
0086:     schema_category: Optional[str] = None
0087:     schema_confidence: float = 0.0
0088: 
0089: 
0090: # =============================================================================
0091: # MEMORY ENGINE
0092: # =============================================================================
0093: 
0094: class MemoryEngine:
0095:     """Trace-primary engine owning the relationships store.
0096: 
0097:     Grammar engine is the canonical extraction path. No T5, no cosine
0098:     anchors. Edge embeddings and predicted queries are computed at
0099:     write time (best-effort, fail-open).
0100:     """
0101: 
0102:     def __init__(self) -> None:
0103:         pass
0104: 
0105:     # ------------------------------------------------------------------
0106:     # Singular ingestion path
0107:     # ------------------------------------------------------------------
0108: 
0109:     def _run_ingestion_path(
0110:         self,
0111:         text: str,
0112:         speaker: Optional[str] = None,
0113:     ) -> Tuple[str, List[Tuple[str, str, str, Any]], Any]:
0114:         """The one write-path entry: cleanup -> grammar -> SPO/decomp rows.
0115: 
0116:         This is the only ingestion pipeline. `clean()`, `extract()`, and
0117:         `ingest_text()` all route through it so KenoticV1 cannot drift onto
0118:         a different extraction path.
0119:         """
0120:         if not text or not text.strip():
0121:             return "", [], None
0122: 
0123:         from app.engines.sentence_model import cleanup as _cleanup
0124:         from app.engines import grammar_engine as _grammar_eng
0125: 
0126:         cleaned = _cleanup(text.strip(), speaker=speaker)
0127:         grammar_result = _grammar_eng.process(cleaned, speaker=speaker)
0128: 
0129:         rows: List[Tuple[str, str, str, Any]] = []
0130:         decomps = grammar_result.trace_decompositions or []
0131:         for idx, triple in enumerate(grammar_result.triples):
0132:             decomp = decomps[idx] if idx < len(decomps) else None
0133:             rows.append((triple.subject, triple.predicate, triple.object, decomp))
0134: 
0135:         for idx in range(len(grammar_result.triples), len(decomps)):
0136:             decomp = decomps[idx]
0137:             rows.append(
0138:                 (
0139:                     getattr(decomp, "subject", ""),
0140:                     getattr(decomp, "predicate", ""),
0141:                     getattr(decomp, "object", ""),
0142:                     decomp,
0143:                 )
0144:             )
0145: 
0146:         final_text = grammar_result.resolved_text or cleaned
0147:         return final_text, rows, grammar_result
0148: 
0149:     def clean(self, text: str) -> str:
0150:         """Run the singular cleanup path used by live ingestion."""
0151:         if not text or not text.strip():
0152:             return text or ""
0153:         from app.engines.sentence_model import cleanup as _cleanup
0154:         return _cleanup(text.strip())
0155: 
0156:     def extract(self, text: str) -> List[Tuple[str, str, str, bool]]:
0157:         """Run the singular extraction path used by live ingestion."""
0158:         _, rows, _ = self._run_ingestion_path(text)
0159:         extracted: List[Tuple[str, str, str, bool]] = []
0160:         for s, p, o, decomp in rows:
0161:             extracted.append(
0162:                 (
0163:                     s,
0164:                     p,
0165:                     o,
0166:                     bool(getattr(decomp, "is_historical", False)) if decomp else False,
0167:                 )
0168:             )
0169:         return extracted
0170: 
0171:     # ------------------------------------------------------------------
0172:     # ingest_text -- singular cleanup -> grammar -> store
0173:     # ------------------------------------------------------------------
0174: 
0175:     def ingest_text(
0176:         self,
0177:         user_id: int,
0178:         text: str,
0179:         source_timestamp: Optional[str] = None,
0180:         speaker: Optional[str] = None,
0181:         speaker_is_user: bool = True,
0182:         confidence: float = 0.9,
0183:         source_tag: Optional[str] = None,
0184:     ) -> int:
0185:         """End-to-end write-path entry for raw text.
0186: 
0187:         1. Grammar engine produces trace decompositions
0188:         2. Speaker resolution (I/me/myself -> speaker name)
0189:         3. Preserve speaker identity in decomp.relational_subject
0190:         4. For each decomposition -> store()
0191: 
0192:         No _validate_triple(). If grammar produced it, store it.
0193: 
0194:         Returns the number of rows actually stored.
0195:         """
0196:         if not text or not text.strip():
0197:             return 0
0198: 
0199:         cleaned, triples_with_decomp, _grammar_result = self._run_ingestion_path(
0200:             text,
0201:             speaker=speaker,
0202:         )
0203: 
0204:         # -- Speaker resolution --
0205:         def _resolve(tok: str) -> str:
0206:             if not tok:
0207:                 return tok
0208:             lower = tok.strip().lower()
0209:             if lower in ("user", "i", "me", "myself"):
0210:                 return speaker if speaker else "user"
0211:             return tok
0212: 
0213:         count = 0
0214:         for s, p, o, decomp in triples_with_decomp:
0215:             resolved_s = _resolve(s)
0216:             resolved_o = _resolve(o)
0217:             # Preserve speaker identity in trace decomposition
0218:             if decomp is not None and speaker:
0219:                 if not getattr(decomp, 'relational_subject', None) or \
0220:                    getattr(decomp, 'relational_subject', '') == 'user':
0221:                     decomp.relational_subject = speaker
0222: 
0223:             has_spo = bool(resolved_s and p and resolved_o)
0224: 
0225:             # Use the decomposition's sentence-level source_text, not
0226:             # the full cleaned turn. Each sentence gets its own hash
0227:             # so multi-sentence turns produce multiple edges, not one.
0228:             decomp_src = getattr(decomp, 'source_text', '') if decomp else ''
0229:             # FIX 1: Wire utterance_type from grammar decomposition to store
0230:             _utt_type = None
0231:             if decomp is not None:
0232:                 _utt_type = getattr(decomp, 'utterance_type', None)
0233: 
0234:             rel_id = self.store(
0235:                 user_id=user_id,
0236:                 trace_decomposition=decomp,
0237:                 source_text=decomp_src or cleaned,
0238:                 confidence=confidence,
0239:                 source_timestamp=source_timestamp,
0240:                 source_tag=source_tag,
0241:                 utterance_type_id=_utt_type,
0242:                 subject=resolved_s if has_spo else None,
0243:                 predicate=p if has_spo else None,
0244:                 object=resolved_o if has_spo else None,
0245:             )
0246:             if rel_id:
0247:                 count += 1
0248:         return count
0249: 
0250:     # ------------------------------------------------------------------
0251:     # store -- three phases: PREPARE -> WRITE -> SIDE-EFFECTS
0252:     # ------------------------------------------------------------------
0253: 
0254:     def store(
0255:         self,
0256:         user_id: int,
0257:         subject: Optional[str] = None,
0258:         predicate: Optional[str] = None,
0259:         object: Optional[str] = None,
0260:         source_text: str = "",
0261:         confidence: float = 0.9,
0262:         utterance_type_id: Optional[int] = None,
0263:         source_timestamp: Optional[str] = None,
0264:         source_tag: Optional[str] = None,
0265:         trace_decomposition: Any = None,
0266:     ) -> int:
0267:         """Three-phase store: PREPARE -> WRITE -> SIDE-EFFECTS.
0268: 
0269:         Phase 1 PREPARE: Extract all column values from TraceDecomposition
0270:                          into a flat dict. Pure computation, no DB access.
0271:         Phase 2 WRITE:   One INSERT or UPDATE for the relationship row,
0272:                          plus FTS5 sync. ~2-4 SQL statements total.
0273:         Phase 3 SIDE-EFFECTS: Entity types + affiliation (one combined
0274:                          UPDATE), facts/milestones, predicted queries,
0275:                          event date, supersession, clustering, arcs.
0276: 
0277:         Returns: relationship_id (0 on failure).
0278:         """
0279:         td = trace_decomposition
0280: 
0281:         # Fill S/P/O from decomposition if not provided directly
0282:         if td is not None:
0283:             subject = subject or getattr(td, 'subject', '') or ''
0284:             predicate = predicate or getattr(td, 'predicate', '') or ''
0285:             object = object or getattr(td, 'object', '') or ''
0286:             if not source_text and getattr(td, 'source_text', ''):
0287:                 source_text = td.source_text
0288: 
0289:         subject = (subject or "").strip()
0290:         predicate = (predicate or "").strip().lower().replace(" ", "_")
0291:         object = (object or "").strip()
0292: 
0293:         has_triple = bool(subject and predicate and object)
0294:         has_source = bool(source_text and source_text.strip())
0295:         if not has_triple and not has_source:
0296:             return 0
0297: 
0298:         if not has_source and has_triple:
0299:             source_text = f"{subject} {predicate.replace('_', ' ')} {object}"
0300: 
0301:         source_text_hash = hashlib.sha256(
0302:             source_text.encode('utf-8')
0303:         ).hexdigest()
0304: 
0305:         # â”€â”€ Phase 1: PREPARE (pure computation, no DB) â”€â”€
0306:         row = self._prepare_row(
0307:             user_id, td, subject, predicate, object, source_text,
0308:             confidence, utterance_type_id, source_timestamp, source_tag,
0309:             source_text_hash,
0310:         )
0311: 
0312:         try:
0313:             with get_db_context() as conn:
0314:                 # â”€â”€ Phase 2: WRITE (one row write + FTS5 sync) â”€â”€
0315:                 rel_id = self._write_row(conn, row, user_id, source_text_hash)
0316:                 if not rel_id:
0317:                     return 0
0318: 
0319:                 # â”€â”€ Phase 3: SIDE-EFFECTS â”€â”€
0320: 
0321:                 # 3-pre: Populate entities (before affiliation reads them)
0322:                 self._upsert_entities(conn, user_id, td)
0323: 
0324:                 # 3a: Entity type + affiliation (read DB, write back as ONE update)
0325:                 _side_updates: Dict[str, Any] = {}
0326:                 try:
0327:                     _speaker = getattr(td, 'relational_subject', None) or subject or ''
0328:                     _aff = self._compute_affiliation(conn, user_id, td, _speaker)
0329:                     if _aff is not None:
0330:                         _side_updates["edge_affiliation"] = round(_aff, 4)
0331:                 except Exception:
0332:                     pass
0333:                 try:
0334:                     for col, name in [("subject_type", subject), ("object_type", object)]:
0335:                         if not name:
0336:                             continue
0337:                         ent_row = conn.execute(
0338:                             "SELECT entity_type FROM entities WHERE user_id = ? AND name = ? LIMIT 1",
0339:                             (user_id, name),
0340:                         ).fetchone()
0341:                         if ent_row and ent_row["entity_type"]:
0342:                             _side_updates[col] = ent_row["entity_type"]
0343:                 except Exception:
0344:                     pass
0345:                 if _side_updates:
0346:                     _cols = ", ".join(f"{k} = ?" for k in _side_updates)
0347:                     _vals: List[Any] = list(_side_updates.values()) + [rel_id]
0348:                     try:
0349:                         conn.execute(
0350:                             f"UPDATE relationships SET {_cols} WHERE id = ?",
0351:                             tuple(_vals),
0352:                         )
0353:                     except Exception:
0354:                         pass
0355: 
0356:                 # 3b: Tier routing (separate tables)
0357:                 tier = row.get("_tier", "episode")
0358:                 if tier == "fact" and td is not None:
0359:                     self._upsert_fact(conn, user_id, rel_id, td, confidence)
0360:                 elif tier == "milestone" and td is not None:
0361:                     self._append_milestone(conn, user_id, rel_id, td, confidence)
0362: 
0363:                 # 3c: Predicted queries (separate table)
0364:                 if has_triple:
0365:                     try:
0366:                         from app.engines.predicted_queries import generate_predicted_queries as _gen_pqs
0367:                         _pqs = _gen_pqs(subject, predicate, object)
0368:                         answer_text = source_text or f"{subject} {predicate.replace('_', ' ')} {object}"
0369:                         for q_text, q_emb in _pqs:
0370:                             conn.execute(
0371:                                 """INSERT INTO predicted_queries
0372:                                      (relationship_id, user_id, predicted_question,
0373:                                       answer_text, answer_subject, question_embedding,
0374:                                       confidence)
0375:                                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
0376:                                 (rel_id, user_id, q_text, answer_text, subject,
0377:                                  q_emb.tobytes(), confidence),
0378:                             )
0379:                     except Exception:
0380:                         pass
0381: 
0382:                 # 3d: Event date resolution
0383:                 self._resolve_event_date(
0384:                     conn, rel_id, source_text,
0385:                     source_timestamp, trace_decomposition,
0386:                 )
0387: 
0388:                 # 3e: Supersession detection
0389:                 try:
0390:                     from app.engines.temporal import get_temporal_engine
0391:                     get_temporal_engine().detect_supersession(
0392:                         user_id, source_text or "", rel_id, None
0393:                     )
0394:                 except Exception:
0395:                     pass
0396: 
0397:                 # 3f: Cluster assignment + arc membership
0398:                 self._assign_cluster(conn, user_id, rel_id, td)
0399:                 try:
0400:                     _cr = conn.execute(
0401:                         "SELECT cluster_id FROM relationships WHERE id = ?",
0402:                         (rel_id,),
0403:                     ).fetchone()
0404:                     _cid = _cr["cluster_id"] if _cr else None
0405:                     # Arc detection via temporal engine (single owner)
0406:                     from app.engines.temporal import get_temporal_engine
0407:                     _arc_id = get_temporal_engine().detect_arcs(user_id, rel_id, _cid)
0408:                     if _arc_id:
0409:                         conn.execute(
0410:                             "UPDATE relationships SET arc_id = ? WHERE id = ?",
0411:                             (_arc_id, rel_id),
0412:                         )
0413:                 except Exception as e:
0414:                     log.warning("arc assignment failed for rel_id=%s: %s", rel_id, e)
0415: 
0416:                 conn.commit()
0417:                 return rel_id
0418:         except Exception as e:
0419:             try:
0420:                 conn.rollback()  # type: ignore[possibly-undefined]
0421:             except Exception:
0422:                 pass  # conn may not be bound if get_db_context() itself failed
0423:             log.error("store failed on (%s, %s, %s): %s", subject, predicate, object, e)
0424:             return 0
0425: 
0426:     # ------------------------------------------------------------------
0427:     # Phase 1: _prepare_row -- pure computation, no DB
0428:     # ------------------------------------------------------------------
0429: 
0430:     def _prepare_row(
0431:         self,
0432:         user_id: int,
0433:         td: Any,
0434:         subject: str,
0435:         predicate: str,
0436:         object: str,
0437:         source_text: str,
0438:         confidence: float,
0439:         utterance_type_id: Optional[int],
0440:         source_timestamp: Optional[str],
0441:         source_tag: Optional[str],
0442:         source_text_hash: str,
0443:     ) -> Dict[str, Any]:
0444:         """Prepare all column values for one relationship row.
0445: 
0446:         Pure computation -- no database access. Returns a flat dict
0447:         of column->value. Internal keys (prefixed with '_') carry
0448:         metadata used by later phases but are NOT written to SQL.
0449:         """
0450:         row: Dict[str, Any] = {}
0451: 
0452:         # Core fields
0453:         row["user_id"] = user_id
0454:         row["subject"] = subject or None
0455:         row["predicate"] = predicate or None
0456:         row["object"] = object or None
0457:         row["source_text"] = source_text
0458:         row["source_text_hash"] = source_text_hash
0459:         row["confidence"] = confidence
0460:         row["utterance_type_id"] = utterance_type_id
0461:         row["source_timestamp"] = source_timestamp
0462:         row["source_tag"] = source_tag
0463: 
0464:         # 5 DTCM trace dimensions (from TraceDecomposition)
0465:         if td is not None:
0466:             _ev = getattr(td, 'emotional_valence', None)
0467:             row["edge_emotional_valence"] = round(
0468:                 float(_ev if _ev is not None else 0.5), 4
0469:             )
0470:             row["edge_emotional_label"] = getattr(td, 'emotional_state', None)
0471:             row["edge_schematic_category"] = (
0472:                 getattr(td, 'schematic_category', None) or 'uncategorized'
0473:             )
0474:             row["edge_episodic_significance"] = (
0475:                 getattr(td, 'episodic_significance', None) or 'routine'
0476:             )
0477:             row["edge_relational_type"] = (
0478:                 getattr(td, 'relational_type', None) or 'personal'
0479:             )
0480:             row["edge_temporal_context"] = (
0481:                 getattr(td, 'temporal_direction', None) or 'present'
0482:             )
0483:             row["edge_negated"] = 1 if getattr(td, 'negated', False) else 0
0484:             row["edge_mood"] = getattr(td, 'mood', 'indicative') or 'indicative'
0485: 
0486:             # Extended trace fields
0487:             row["is_historical"] = 1 if getattr(td, 'is_historical', False) else 0
0488:             row["episodic_fact"] = getattr(td, 'episodic_fact', None) or None
0489:             row["emotional_target"] = getattr(td, 'emotional_target', None) or None
0490:             row["extraction_rule"] = getattr(td, 'extraction_rule', None) or None
0491: 
0492:             # Temporal expression + relational entities
0493:             row["temporal_expression"] = getattr(td, 'temporal_expression', None)
0494:             _rel_subj = getattr(td, 'relational_subject', None)
0495:             _rel_ents = getattr(td, 'relational_entities', None) or []
0496:             if isinstance(_rel_ents, list) and _rel_subj and _rel_subj.lower() != 'user':
0497:                 if _rel_subj not in _rel_ents:
0498:                     _rel_ents = list(_rel_ents) + [_rel_subj]
0499:             row["relational_entities"] = json.dumps(_rel_ents) if _rel_ents else None
0500: 
0501:         # Embeddings (computed here, not in separate steps)
0502:         try:
0503:             from app.vector.embedder import embed_text as _embed
0504:             if source_text:
0505:                 row["edge_embedding"] = _embed(source_text).tobytes()
0506:             if predicate:
0507:                 row["predicate_embedding"] = _embed(
0508:                     predicate.replace("_", " ")
0509:                 ).tobytes()
0510:         except Exception:
0511:             pass  # embedder unavailable
0512: 
0513:         # Tier classification (pure logic, no DB)
0514:         row["_tier"] = self._classify_tier(td)
0515: 
0516:         return row
0517: 
0518:     # ------------------------------------------------------------------
0519:     # Phase 2: _write_row -- one INSERT or UPDATE + FTS5
0520:     # ------------------------------------------------------------------
0521: 
0522:     def _write_row(
0523:         self,
0524:         conn: sqlite3.Connection,
0525:         row: Dict[str, Any],
0526:         user_id: int,
0527:         source_text_hash: str,
0528:     ) -> int:
0529:         """Write one relationship row. Dedup on source_text_hash.
0530: 
0531:         UPDATE path: one SELECT (dedup+tombstone check) + one UPDATE + FTS5 sync.
0532:         INSERT path: one INSERT + FTS5 sync + one sequence_number UPDATE.
0533: 
0534:         Returns relationship_id (0 on failure).
0535:         """
0536:         # â”€â”€ Dedup check â”€â”€
0537:         existing = None
0538:         if source_text_hash:
0539:             try:
0540:                 existing = conn.execute(
0541:                     "SELECT id, tombstoned_at, subject, predicate, object, source_text "
0542:                     "FROM relationships WHERE user_id = ? AND source_text_hash = ?",
0543:                     (user_id, source_text_hash),
0544:                 ).fetchone()
0545:             except Exception:
0546:                 existing = None  # fail-open: column not yet migrated
0547: 
0548:         if existing:
0549:             rel_id: int = existing["id"]
0550: 
0551:             # Tombstone guard: never revive a forgotten row
0552:             if existing["tombstoned_at"] is not None:
0553:                 return rel_id
0554: 
0555:             # Capture old values for FTS5 delete
0556:             old_subj = existing["subject"] or ""
0557:             old_pred = (existing["predicate"] or "").replace("_", " ")
0558:             old_obj = existing["object"] or ""
0559:             old_src = existing["source_text"] or ""
0560: 
0561:             # â”€â”€ Build ONE UPDATE with all columns â”€â”€
0562:             update_cols: List[str] = []
0563:             update_vals: List[Any] = []
0564: 
0565:             # Always update these
0566:             update_cols.append("confidence = MAX(confidence, ?)")
0567:             update_vals.append(row.get("confidence", 0.9))
0568:             update_cols.append("last_confirmed_at = datetime('now')")
0569:             update_cols.append("is_current = 1")
0570: 
0571:             # COALESCE for S/P/O (keep existing if new is NULL)
0572:             update_cols.append("subject = COALESCE(?, subject)")
0573:             update_vals.append(row.get("subject"))
0574:             update_cols.append("predicate = COALESCE(?, predicate)")
0575:             update_vals.append(row.get("predicate"))
0576:             update_cols.append("object = COALESCE(?, object)")
0577:             update_vals.append(row.get("object"))
0578: 
0579:             # Source text (always overwrite if provided)
0580:             if row.get("source_text"):
0581:                 update_cols.append("source_text = ?")
0582:                 update_vals.append(row["source_text"])
0583:             if row.get("source_tag"):
0584:                 update_cols.append("source_tag = COALESCE(source_tag, ?)")
0585:                 update_vals.append(row["source_tag"])
0586: 
0587:             # All trace + embedding columns in one pass
0588:             _trace_cols = (
0589:                 "edge_emotional_valence", "edge_emotional_label",
0590:                 "edge_schematic_category", "edge_episodic_significance",
0591:                 "edge_relational_type", "edge_temporal_context",
0592:                 "edge_negated", "edge_mood",
0593:                 "is_historical", "episodic_fact", "emotional_target",
0594:                 "extraction_rule", "temporal_expression", "relational_entities",
0595:                 "edge_embedding", "predicate_embedding",
0596:             )
0597:             for col in _trace_cols:
0598:                 if col in row and row[col] is not None:
0599:                     update_cols.append(f"{col} = ?")
0600:                     update_vals.append(row[col])
0601: 
0602:             update_vals.append(rel_id)
0603:             sql = f"UPDATE relationships SET {', '.join(update_cols)} WHERE id = ?"
0604: 
0605:             try:
0606:                 conn.execute(sql, tuple(update_vals))
0607:             except sqlite3.OperationalError as e:
0608:                 # Some columns may not exist yet in old DBs -- fall back to core
0609:                 log.warning("full update failed, falling back to core: %s", e)
0610:                 conn.execute(
0611:                     """UPDATE relationships SET
0612:                          confidence = MAX(confidence, ?),
0613:                          last_confirmed_at = datetime('now'),
0614:                          is_current = 1,
0615:                          subject = COALESCE(?, subject),
0616:                          predicate = COALESCE(?, predicate),
0617:                          object = COALESCE(?, object)
0618:                        WHERE id = ?""",
0619:                     (row.get("confidence", 0.9), row.get("subject"),
0620:                      row.get("predicate"), row.get("object"), rel_id),
0621:                 )
0622: 
0623:             # FTS5 sync (delete old + insert new)
0624:             try:
0625:                 conn.execute(
0626:                     "INSERT INTO relationships_fts(relationships_fts, rowid, "
0627:                     "subject, predicate, object, source_text) "
0628:                     "VALUES('delete', ?, ?, ?, ?, ?)",
0629:                     (rel_id, old_subj, old_pred, old_obj, old_src),
0630:                 )
0631:                 conn.execute(
0632:                     """INSERT INTO relationships_fts
0633:                          (rowid, subject, predicate, object, source_text)
0634:                        VALUES (?, ?, ?, ?, ?)""",
0635:                     (rel_id, row.get("subject") or "",
0636:                      (row.get("predicate") or "").replace("_", " "),
0637:                      row.get("object") or "", row.get("source_text") or ""),
0638:                 )
0639:             except Exception:
0640:                 pass  # FTS5 table may not exist yet
0641: 
0642:             return rel_id
0643: 
0644:         # â”€â”€ INSERT new row â”€â”€
0645:         # Build column list dynamically from what's in row
0646:         insert_cols: List[str] = ["user_id", "is_current"]
0647:         insert_vals: List[Any] = [user_id, 1]
0648: 
0649:         for col, val in row.items():
0650:             if col.startswith("_"):  # skip internal keys like _tier
0651:                 continue
0652:             if val is not None and col != "user_id":
0653:                 insert_cols.append(col)
0654:                 insert_vals.append(val)
0655: 
0656:         placeholders = ", ".join(["?"] * len(insert_vals))
0657:         col_names = ", ".join(insert_cols)
0658: 
0659:         try:
0660:             cur = conn.execute(
0661:                 f"INSERT INTO relationships ({col_names}) VALUES ({placeholders})",
0662:                 tuple(insert_vals),
0663:             )
0664:         except sqlite3.OperationalError:
0665:             # Fall back to core columns only for old DBs
0666:             cur = conn.execute(
0667:                 """INSERT INTO relationships
0668:                      (user_id, subject, predicate, object, confidence,
0669:                       utterance_type_id, source_timestamp, is_current,
0670:                       source_text, source_tag, source_text_hash)
0671:                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
0672:                 (user_id, row.get("subject"), row.get("predicate"),
0673:                  row.get("object"), row.get("confidence", 0.9),
0674:                  row.get("utterance_type_id"), row.get("source_timestamp"),
0675:                  row.get("source_text"), row.get("source_tag"),
0676:                  row.get("source_text_hash")),
0677:             )
0678: 
0679:         rel_id = cur.lastrowid
0680: 
0681:         # FTS5 sync
0682:         try:
0683:             conn.execute(
0684:                 """INSERT INTO relationships_fts
0685:                      (rowid, subject, predicate, object, source_text)
0686:                    VALUES (?, ?, ?, ?, ?)""",
0687:                 (rel_id, row.get("subject") or "",
0688:                  (row.get("predicate") or "").replace("_", " "),
0689:                  row.get("object") or "", row.get("source_text") or ""),
0690:             )
0691:         except Exception:
0692:             pass
0693: 
0694:         # Sequence number
0695:         try:
0696:             conn.execute(
0697:                 "UPDATE relationships SET sequence_number = ? WHERE id = ?",
0698:                 (rel_id, rel_id),
0699:             )
0700:         except Exception:
0701:             pass
0702: 
0703:         return rel_id
0704: 
0705:     # ------------------------------------------------------------------
0706:     # _compute_affiliation -- in-group/out-group scoring
0707:     # ------------------------------------------------------------------
0708: 
0709:     def _compute_affiliation(
0710:         self,
0711:         conn: sqlite3.Connection,
0712:         user_id: int,
0713:         td: Any,
0714:         speaker: str,
0715:     ) -> Optional[float]:
0716:         """Compute edge_affiliation: how personally connected a fact is to the speaker.
0717: 
0718:         1.0 = speaker's own life fact
0719:         0.8 = well-known entity (mention_count >= 5)
0720:         0.6 = mentioned before (mention_count >= 2)
0721:         0.4 = first mention but named
0722:         0.2 = unknown entity
0723:         0.1 = no entities at all (no personal connection)
0724: 
0725:         Returns MAX across all entity scores (one close entity makes the fact personal).
0726:         """
0727:         if td is None:
0728:             return None
0729: 
0730:         # If relational_subject IS the speaker -> speaker's own fact
0731:         rel_subj = getattr(td, 'relational_subject', None) or ''
0732:         if rel_subj:
0733:             subj_lower = rel_subj.strip().lower()
0734:             speaker_lower = (speaker or '').strip().lower()
0735:             if subj_lower in ('user', 'i', 'me', 'myself') or (
0736:                 speaker_lower and subj_lower == speaker_lower
0737:             ):
0738:                 return 1.0
0739: 
0740:         # Gather candidate entities from decomposition
0741:         candidates: List[str] = []
0742:         for ent in (getattr(td, 'relational_entities', None) or []):
0743:             if ent and ent.strip().lower() not in ('', 'user'):
0744:                 candidates.append(ent.strip())
0745:         for attr in ('subject', 'object'):
0746:             val = getattr(td, attr, None)
0747:             if val and val.strip().lower() not in ('', 'user'):
0748:                 candidates.append(val.strip())
0749: 
0750:         # Deduplicate (case-insensitive) while preserving order
0751:         seen: set[str] = set()
0752:         unique: List[str] = []
0753:         for c in candidates:
0754:             key = c.lower()
0755:             if key not in seen:
0756:                 seen.add(key)
0757:                 unique.append(c)
0758: 
0759:         if not unique:
0760:             return 0.1  # no entities -> no personal connection
0761: 
0762:         # Score each entity by mention_count
0763:         scores: List[float] = []
0764:         for name in unique:
0765:             row = None
0766:             try:
0767:                 row = conn.execute(
0768:                     "SELECT mention_count FROM entities WHERE user_id = ? AND name = ? LIMIT 1",
0769:                     (user_id, name),
0770:                 ).fetchone()
0771:             except Exception:
0772:                 pass  # entities table may not exist
0773: 
0774:             if row is None:
0775:                 scores.append(0.2)
0776:             else:
0777:                 mc = row["mention_count"] or 1
0778:                 if mc >= 5:
0779:                     scores.append(0.8)
0780:                 elif mc >= 2:
0781:                     scores.append(0.6)
0782:                 else:
0783:                     scores.append(0.4)
0784: 
0785:         return max(scores)
0786: 
0787:     # ------------------------------------------------------------------
0788:     # Entity population -- upsert entities from trace decomposition
0789:     # ------------------------------------------------------------------
0790: 
0791:     def _upsert_entities(self, conn: sqlite3.Connection, user_id: str, td: Any) -> None:
0792:         """Upsert entities from TraceDecomposition into the entities table.
0793: 
0794:         Only NER-labeled entities (already filtered by grammar engine).
0795:         Title-case normalized to prevent duplicate rows from case differences.
0796:         """
0797:         if td is None:
0798:             return
0799: 
0800:         # Gather entities ONLY from relational_entities (already NER-filtered
0801:         # by grammar_engine._extract_relational). Subject/object fields carry
0802:         # arbitrary phrases ("Really Nervous About The Interview") that are
0803:         # not proper entities.
0804:         names: set[str] = set()
0805:         rel_ents = getattr(td, 'relational_entities', None) or []
0806:         if isinstance(rel_ents, list):
0807:             for e in rel_ents:
0808:                 if isinstance(e, str) and e.strip():
0809:                     if e.strip().lower() not in ('user', 'i', 'me', 'myself'):
0810:                         names.add(e.strip().title())
0811: 
0812:         for name in names:
0813:             try:
0814:                 existing = conn.execute(
0815:                     "SELECT id, mention_count FROM entities WHERE user_id = ? AND name = ?",
0816:                     (user_id, name),
0817:                 ).fetchone()
0818:                 if existing:
0819:                     conn.execute(
0820:                         "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?",
0821:                         (existing["id"],),
0822:                     )
0823:                 else:
0824:                     conn.execute(
0825:                         "INSERT INTO entities (user_id, name, mention_count) VALUES (?, ?, 1)",
0826:                         (user_id, name),
0827:                     )
0828:             except Exception:
0829:                 pass  # fail-open
0830: 
0831:     # ------------------------------------------------------------------
0832:     # Tier routing -- classify edge as fact/milestone/episode
0833:     # ------------------------------------------------------------------
0834: 
0835:     def _classify_tier(self, td: Any) -> str:
0836:         """Classify a TraceDecomposition into fact/milestone/episode tier.
0837: 
0838:         Significance alone determines tier â€” no schema allow-list.
0839:         The grammar engine's significance classifier already did the work
0840:         of determining whether this is a persisting state or a one-time event.
0841:         """
0842:         if td is None:
0843:             return "episode"
0844:         sig = getattr(td, 'episodic_significance', 'routine') or 'routine'
0845: 
0846:         if sig == "milestone":
0847:             return "milestone"
0848:         if sig == "stative":
0849:             return "fact"
0850:         return "episode"
0851: 
0852:     # ------------------------------------------------------------------
0853:     # _classify_supersession_type -- structural signal, no word lists
0854:     # ------------------------------------------------------------------
0855: 
0856:     def _classify_supersession_type(self, td: Any) -> str:
0857:         """Classify how a fact was superseded: update, correction, or reversal.
0858: 
0859:         Uses signals already present on the TraceDecomposition:
0860:         - correction: grammar engine flagged via extraction_rule
0861:         - reversal:   decomposition carries negated=True
0862:         - update:     natural temporal progression (default)
0863:         """
0864:         if td is None:
0865:             return "update"
0866: 
0867:         extraction_rule = getattr(td, 'extraction_rule', '') or ''
0868:         if 'correction' in extraction_rule.lower():
0869:             return "correction"
0870: 
0871:         negated = getattr(td, 'negated', False)
0872:         if negated:
0873:             return "reversal"
0874: 
0875:         return "update"
0876: 
0877:     # ------------------------------------------------------------------
0878:     # _upsert_fact -- semantic key dedup with history chain
0879:     # ------------------------------------------------------------------
0880: 
0881:     def _upsert_fact(
0882:         self,
0883:         conn: sqlite3.Connection,
0884:         user_id: int,
0885:         rel_id: int,
0886:         td: Any,
0887:         confidence: float,
0888:     ) -> None:
0889:         """Upsert a fact row keyed on schema::VerbClass::subject.
0890: 
0891:         Root cause addressed: the old key schema::predicate::subject used
0892:         surface predicates, so "work_at" and "start_at" created separate
0893:         fact rows for the same career fact. Now uses WordNet open-vocabulary
0894:         hypernym classification (classify_verb_class) to group synonymous
0895:         verbs under one VerbClass label. The verb lemma is extracted from
0896:         compound predicates by splitting on underscore before lookup.
0897: 
0898:         If the value changed, the old value is appended to the history
0899:         JSON array for contradiction tracking. If the value is the same,
0900:         only last_confirmed_at is bumped.
0901:         """
0902:         from app.engines.grammar_engine import classify_verb_class
0903: 
0904:         schema = getattr(td, 'schematic_category', 'uncategorized') or 'uncategorized'
0905:         predicate = getattr(td, 'predicate', '') or ''
0906:         subject = getattr(td, 'subject', '') or ''
0907:         value = getattr(td, 'object', '') or ''
0908: 
0909:         # Extract verb lemma from compound predicates (e.g., "work_at" -> "work")
0910:         # then classify via WordNet hypernym closure (open-vocabulary, not a word list)
0911:         verb_lemma = predicate.split('_')[0] if predicate else ''
0912:         verb_class = classify_verb_class(verb_lemma).name if verb_lemma else 'UNKNOWN'
0913:         key = f"{schema}::{verb_class}::{subject}"
0914:         if not key or not value:
0915:             return
0916: 
0917:         try:
0918:             existing = conn.execute(
0919:                 "SELECT id, value, history FROM facts WHERE user_id = ? AND key = ?",
0920:                 (user_id, key),
0921:             ).fetchone()
0922: 
0923:             if existing:
0924:                 old_value = existing["value"] or ""
0925:                 if old_value != value:
0926:                     # Value changed -- append old to history chain
0927:                     history_raw = existing["history"] or "[]"
0928:                     try:
0929:                         history: List[Dict[str, str]] = json.loads(history_raw)
0930:                     except (json.JSONDecodeError, TypeError):
0931:                         history = []
0932:                     history.append({
0933:                         "value": old_value,
0934:                         "at": datetime.now(timezone.utc).isoformat(),
0935:                         "type": self._classify_supersession_type(td),
0936:                     })
0937:                     conn.execute(
0938:                         """UPDATE facts SET
0939:                              value = ?,
0940:                              confidence = ?,
0941:                              last_confirmed_at = datetime('now'),
0942:                              provenance_memory_id = ?,
0943:                              history = ?
0944:                            WHERE id = ?""",
0945:                         (value, confidence, rel_id,
0946:                          json.dumps(history), existing["id"]),
0947:                     )
0948:                 else:
0949:                     # Same value -- just bump confirmation timestamp
0950:                     conn.execute(
0951:                         "UPDATE facts SET last_confirmed_at = datetime('now') WHERE id = ?",
0952:                         (existing["id"],),
0953:                     )
0954:             else:
0955:                 # New fact
0956:                 conn.execute(
0957:                     """INSERT INTO facts
0958:                          (user_id, key, value, confidence,
0959:                           provenance_memory_id)
0960:                        VALUES (?, ?, ?, ?, ?)""",
0961:                     (user_id, key, value, confidence, rel_id),
0962:                 )
0963:         except Exception as e:
0964:             log.warning("_upsert_fact failed for key=%s: %s", key, e)
0965: 
0966:     # ------------------------------------------------------------------
0967:     # _append_milestone -- dedup on exact description text
0968:     # ------------------------------------------------------------------
0969: 
0970:     def _append_milestone(
0971:         self,
0972:         conn: sqlite3.Connection,
0973:         user_id: int,
0974:         rel_id: int,
0975:         td: Any,
0976:         confidence: float,
0977:     ) -> None:
0978:         """Insert a milestone row if no exact-text duplicate exists.
0979: 
0980:         Event date is left NULL; _resolve_event_date fills it later.
0981:         """
0982:         description = (
0983:             getattr(td, 'source_text', '') or
0984:             getattr(td, 'episodic_fact', '') or ''
0985:         ).strip()
0986:         event_type = getattr(td, 'schematic_category', '') or 'life_event'
0987: 
0988:         if not description:
0989:             return
0990: 
0991:         try:
0992:             dup = conn.execute(
0993:                 "SELECT id FROM milestones WHERE user_id = ? AND description = ?",
0994:                 (user_id, description),
0995:             ).fetchone()
0996:             if dup:
0997:                 return  # exact duplicate -- skip
0998: 
0999:             conn.execute(
1000:                 """INSERT INTO milestones
1001:                      (user_id, event_type, description, confidence,
1002:                       provenance_memory_id)
1003:                    VALUES (?, ?, ?, ?, ?)""",
1004:                 (user_id, event_type, description, confidence, rel_id),
1005:             )
1006:         except Exception as e:
1007:             log.warning("_append_milestone failed: %s", e)
1008: 
1009:     # ------------------------------------------------------------------
1010:     # _resolve_event_date -- 4-tier waterfall (extracted from old store)
1011:     # ------------------------------------------------------------------
1012: 
1013:     def _resolve_event_date(
1014:         self,
1015:         conn: sqlite3.Connection,
1016:         rel_id: int,
1017:         source_text: str,
1018:         source_timestamp: Optional[str],
1019:         trace_decomposition: Any,
1020:     ) -> None:
1021:         """Resolve event date via 4-tier waterfall and write to row."""
1022:         td = trace_decomposition
1023:         try:
1024:             from app.engines.temporal import get_temporal_engine
1025:             _te_date = get_temporal_engine()
1026: 
1027:             # Tier 1: resolve from source text
1028:             _resolved = _te_date.resolve_event_date(
1029:                 source_text or "", source_timestamp
1030:             )
1031: 
1032:             # Tier 2: grammar engine temporal_expression
1033:             if not _resolved and td is not None:
1034:                 _temp_expr = getattr(td, 'temporal_expression', None)
1035:                 if _temp_expr:
1036:                     _resolved = _te_date.resolve_event_date(
1037:                         _temp_expr, source_timestamp
1038:                     )
1039: 
1040:             # Tier 3: grammar engine pre-resolved date
1041:             if not _resolved and td is not None:
1042:                 _temp_resolved = getattr(td, 'temporal_resolved', None)
1043:                 if _temp_resolved:
1044:                     _resolved = _temp_resolved
1045: 
1046:             # Tier 4: source_timestamp (session time)
1047:             if not _resolved and source_timestamp:
1048:                 _resolved = source_timestamp
1049: 
1050:             if _resolved:
1051:                 conn.execute(
1052:                     "UPDATE relationships SET resolved_event_date = ? WHERE id = ?",
1053:                     (_resolved, rel_id),
1054:                 )
1055:         except Exception:
1056:             pass  # fail-open: date resolution is observational
1057: 
1058:     # ------------------------------------------------------------------
1059:     # _assign_cluster -- bind related edges via shared entities
1060:     # ------------------------------------------------------------------
1061: 
1062:     def _assign_cluster(
1063:         self,
1064:         conn: sqlite3.Connection,
1065:         user_id: int,
1066:         rel_id: int,
1067:         td: Any,
1068:     ) -> None:
1069:         """Assign cluster_id to a newly written edge via entity overlap.
1070: 
1071:         Algorithm:
1072:           1. Build entity set from subject, object, and relational_entities
1073:           2. Find existing clusters that share at least one entity
1074:           3. No match   -> new cluster (str(rel_id))
1075:              One match  -> join that cluster
1076:              N matches  -> merge all into smallest cluster_id, then join
1077:           4. Write cluster_id to the new edge
1078: 
1079:         Fail-open: entire method is wrapped in try/except so a clustering
1080:         bug never blocks the store path.
1081:         """
1082:         try:
1083:             # -- 1. Build entity set for this edge --
1084:             entities: set[str] = set()
1085: 
1086:             # From subject/object columns (already written to the row)
1087:             try:
1088:                 row = conn.execute(
1089:                     "SELECT subject, object, relational_entities FROM relationships WHERE id = ?",
1090:                     (rel_id,),
1091:                 ).fetchone()
1092:                 if row:
1093:                     if row["subject"]:
1094:                         entities.add(row["subject"])
1095:                     if row["object"]:
1096:                         entities.add(row["object"])
1097:                     if row["relational_entities"]:
1098:                         try:
1099:                             rel_ents = json.loads(row["relational_entities"])
1100:                             if isinstance(rel_ents, list):
1101:                                 for e in rel_ents:
1102:                                     if isinstance(e, str) and e.strip():
1103:                                         entities.add(e.strip())
1104:                         except (json.JSONDecodeError, TypeError):
1105:                             pass
1106:             except Exception:
1107:                 pass  # relational_entities column may not exist
1108: 
1109:             # Fallback: pull from trace decomposition if DB read missed
1110:             if td is not None:
1111:                 for attr in ('subject', 'object'):
1112:                     val = getattr(td, attr, None)
1113:                     if val and isinstance(val, str) and val.strip():
1114:                         entities.add(val.strip())
1115:                 td_ents = getattr(td, 'relational_entities', None) or []
1116:                 if isinstance(td_ents, list):
1117:                     for e in td_ents:
1118:                         if isinstance(e, str) and e.strip():
1119:                             entities.add(e.strip())
1120: 
1121:             # -- 2. Filter noise --
1122:             entities.discard("")
1123:             entities.discard("user")
1124: 
1125:             if not entities:
1126:                 return  # no clustering signal
1127: 
1128:             # -- 3. Find existing clusters sharing at least one entity --
1129:             entity_list = list(entities)
1130:             placeholders = ",".join("?" for _ in entity_list)
1131: 
1132:             # Build LIKE clauses for relational_entities JSON.
1133:             # Each entity gets: relational_entities LIKE '%"entity"%'
1134:             # The quotes prevent substring false positives ("Sam" won't
1135:             # match "Samantha" because the JSON stores ["Sam"]).
1136:             like_clauses = " OR ".join(
1137:                 "relational_entities LIKE ?" for _ in entity_list
1138:             )
1139:             like_vals = [f'%"{e}"%' for e in entity_list]
1140: 
1141:             existing_clusters = conn.execute(
1142:                 f"""SELECT DISTINCT cluster_id FROM relationships
1143:                     WHERE user_id = ? AND cluster_id IS NOT NULL AND cluster_id != ''
1144:                     AND id != ?
1145:                     AND (subject IN ({placeholders})
1146:                          OR object IN ({placeholders})
1147:                          OR ({like_clauses}))""",
1148:                 [user_id, rel_id] + entity_list + entity_list + like_vals,
1149:             ).fetchall()
1150: 
1151:             cluster_ids = [r["cluster_id"] for r in existing_clusters if r["cluster_id"]]
1152: 
1153:             if not cluster_ids:
1154:                 # -- 4a. No match -> new cluster --
1155:                 chosen = str(rel_id)
1156:             elif len(cluster_ids) == 1:
1157:                 # -- 4b. One match -> join --
1158:                 chosen = cluster_ids[0]
1159:             else:
1160:                 # -- 4c. Multiple matches -> merge into smallest --
1161:                 chosen = min(cluster_ids, key=lambda cid: int(cid) if cid.isdigit() else float('inf'))
1162:                 old_ids = [cid for cid in cluster_ids if cid != chosen]
1163:                 if old_ids:
1164:                     merge_placeholders = ",".join("?" for _ in old_ids)
1165:                     conn.execute(
1166:                         f"""UPDATE relationships SET cluster_id = ?
1167:                             WHERE user_id = ? AND cluster_id IN ({merge_placeholders})""",
1168:                         [chosen, user_id] + old_ids,
1169:                     )
1170: 
1171:             # -- 5. Write cluster_id to the new edge --
1172:             conn.execute(
1173:                 "UPDATE relationships SET cluster_id = ? WHERE id = ?",
1174:                 (chosen, rel_id),
1175:             )
1176: 
1177:         except Exception as e:
1178:             log.warning("_assign_cluster failed for rel_id=%s: %s", rel_id, e)
1179: 
1180:     # ------------------------------------------------------------------
1181:     # _assign_arc -- match new edge's cluster to open arcs
1182:     # ------------------------------------------------------------------
1183: 
1184:     def _assign_arc(
1185:         self,
1186:         conn: sqlite3.Connection,
1187:         user_id: int,
1188:         rel_id: int,
1189:         cluster_id: Optional[str],
1190:     ) -> None:
1191:         """Assign arc_id to a newly written edge by matching its cluster
1192:         to open arcs whose start_edge shares the same cluster_id.
1193: 
1194:         Fail-open: entire method is wrapped in try/except so an arc
1195:         assignment bug never blocks the store path.
1196:         """
1197:         try:
1198:             if not cluster_id:
1199:                 return  # can't match without a cluster
1200: 
1201:             # Find an open arc whose start_edge belongs to the same cluster
1202:             arc_row = conn.execute(
1203:                 """SELECT a.id FROM arcs a
1204:                    JOIN relationships r ON r.id = a.start_edge_id
1205:                    WHERE a.user_id = ? AND a.status = 'open'
1206:                      AND r.cluster_id = ?
1207:                    LIMIT 1""",
1208:                 (user_id, cluster_id),
1209:             ).fetchone()
1210: 
1211:             if not arc_row:
1212:                 return  # no matching open arc
1213: 
1214:             arc_id: str = arc_row["id"] if isinstance(arc_row, sqlite3.Row) else arc_row[0]
1215: 
1216:             # Write arc_id to the new edge
1217:             conn.execute(
1218:                 "UPDATE relationships SET arc_id = ? WHERE id = ?",
1219:                 (arc_id, rel_id),
1220:             )
1221: 
1222:             # Touch the arc's last_checked_at
1223:             conn.execute(
1224:                 "UPDATE arcs SET last_checked_at = datetime('now') WHERE id = ?",
1225:                 (arc_id,),
1226:             )
1227: 
1228:             log.debug("_assign_arc: rel_id=%s -> arc_id=%s", rel_id, arc_id)
1229: 
1230:         except Exception as e:
1231:             log.warning("_assign_arc failed for rel_id=%s: %s", rel_id, e)
1232: 
1233:     # ------------------------------------------------------------------
1234:     # Read methods
1235:     # ------------------------------------------------------------------
1236: 
1237:     def get_relationships(
1238:         self,
1239:         user_id: int,
1240:         subject: Optional[str] = None,
1241:         predicate: Optional[str] = None,
1242:         object: Optional[str] = None,
1243:         only_current: bool = True,
1244:     ) -> List[Relationship]:
1245:         conditions = ["user_id = ?"]
1246:         params: List[Any] = [user_id]
1247:         if only_current:
1248:             conditions.append("COALESCE(is_current, 1) = 1")
1249:             conditions.append("tombstoned_at IS NULL")
1250:         if subject:
1251:             conditions.append("LOWER(subject) = LOWER(?)")
1252:             params.append(subject)
1253:         if predicate:
1254:             conditions.append("LOWER(predicate) = LOWER(?)")
1255:             params.append(predicate)
1256:         if object:
1257:             conditions.append("LOWER(object) = LOWER(?)")
1258:             params.append(object)
1259: 
1260:         sql = (
1261:             f"SELECT id, subject, predicate, object, confidence, is_current "
1262:             f"FROM relationships WHERE {' AND '.join(conditions)} "
1263:             f"ORDER BY id DESC"
1264:         )
1265:         with get_db_context() as conn:
1266:             rows = conn.execute(sql, params).fetchall()
1267:             return [
1268:                 Relationship(
1269:                     id=r["id"],
1270:                     subject=r["subject"],
1271:                     predicate=r["predicate"],
1272:                     object=r["object"],
1273:                     confidence=r["confidence"] or 0.9,
1274:                     is_current=bool(r["is_current"] if r["is_current"] is not None else 1),
1275:                 )
1276:                 for r in rows
1277:             ]
1278: 
1279:     def get_entity(self, user_id: int, name: str) -> Optional[Entity]:
1280:         with get_db_context() as conn:
1281:             row = conn.execute(
1282:                 "SELECT * FROM entities WHERE user_id = ? AND LOWER(name) = LOWER(?)",
1283:                 (user_id, name),
1284:             ).fetchone()
1285:         if not row:
1286:             return None
1287:         emb = None
1288:         if row["embedding"]:
1289:             try:
1290:                 emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
1291:             except Exception:
1292:                 emb = None
1293:         attrs: Dict[str, Any] = {}
1294:         try:
1295:             attrs = json.loads(row["attributes"] or "{}")
1296:         except Exception:
1297:             pass
1298:         return Entity(
1299:             name=row["name"],
1300:             entity_type=row["entity_type"] or "unknown",
1301:             attributes=attrs,
1302:             embedding=emb,
1303:             mention_count=row["mention_count"] or 1,
1304:         )
1305: 
1306:     # ------------------------------------------------------------------
1307:     # Neural entity linker -- pure cosine, no regex, no stop-word list
1308:     # ------------------------------------------------------------------
1309: 
1310:     def entity_link(
1311:         self,
1312:         user_id: int,
1313:         mention_text: str,
1314:     ) -> Optional[Entity]:
1315:         """Resolve a free-text mention to a canonical entity via cosine
1316:         over entities.embedding for this user."""
1317:         if not mention_text or not mention_text.strip():
1318:             return None
1319: 
1320:         try:
1321:             query_emb = embed_text(mention_text.strip())
1322:         except Exception:
1323:             return None
1324: 
1325:         with get_db_context() as conn:
1326:             rows = conn.execute(
1327:                 """SELECT id, name, entity_type, attributes, embedding, mention_count
1328:                    FROM entities WHERE user_id = ? AND embedding IS NOT NULL""",
1329:                 (user_id,),
1330:             ).fetchall()
1331: 
1332:         if not rows:
1333:             return None
1334: 
1335:         sims: List[Tuple[float, sqlite3.Row]] = []
1336:         for r in rows:
1337:             try:
1338:                 emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
1339:                 sims.append((cosine_sim(query_emb, emb), r))
1340:             except Exception:
1341:                 continue
1342: 
1343:         if not sims:
1344:             return None
1345: 
1346:         sims.sort(key=lambda x: -x[0])
1347:         best_sim, best_row = sims[0]
1348: 
1349:         # Adaptive population threshold
1350:         scores = np.array([s for s, _ in sims], dtype=np.float32)
1351:         if len(scores) >= 3:
1352:             mean = float(scores.mean())
1353:             std = float(scores.std())
1354:             threshold = mean + std
1355:             if best_sim < threshold:
1356:                 return None
1357: 
1358:         attrs: Dict[str, Any] = {}
1359:         try:
1360:             attrs = json.loads(best_row["attributes"] or "{}")
1361:         except Exception:
1362:             pass
1363:         emb = None
1364:         try:
1365:             emb = np.frombuffer(best_row["embedding"], dtype=np.float32).copy()
1366:         except Exception:
1367:             pass
1368:         return Entity(
1369:             name=best_row["name"],
1370:             entity_type=best_row["entity_type"] or "unknown",
1371:             attributes=attrs,
1372:             embedding=emb,
1373:             mention_count=best_row["mention_count"] or 1,
1374:         )
1375: 
1376:     # ------------------------------------------------------------------
1377:     # Supersession hook (called by TemporalEngine)
1378:     # ------------------------------------------------------------------
1379: 
1380:     def supersede(self, relationship_id: int, superseded_by: int) -> None:
1381:         """Mark a prior relationship as superseded by a newer one."""
1382:         try:
1383:             with get_db_context() as conn:
1384:                 conn.execute(
1385:                     """UPDATE relationships SET
1386:                          is_current = 0,
1387:                          superseded_at = datetime('now'),
1388:                          superseded_by = ?
1389:                        WHERE id = ?""",
1390:                     (superseded_by, relationship_id),
1391:                 )
1392:                 # Clean predicted_queries for superseded edge (legacy table)
1393:                 try:
1394:                     conn.execute(
1395:                         "DELETE FROM predicted_queries WHERE relationship_id = ?",
1396:                         (relationship_id,),
1397:                     )
1398:                 except sqlite3.OperationalError:
1399:                     pass  # table may not exist
1400:                 conn.commit()
1401:         except Exception as e:
1402:             log.error("supersede failed: %s", e)
1403: 
1404:     # ------------------------------------------------------------------
1405:     # Summarize
1406:     # ------------------------------------------------------------------
1407: 
1408:     def summarize(self, user_id: int, entity: Optional[str] = None) -> str:
1409:         """Natural-language summary of entity state."""
1410:         rels = (
1411:             self.get_relationships(user_id, subject=entity)
1412:             if entity
1413:             else self.get_relationships(user_id, subject="user")
1414:         )
1415:         if not rels:
1416:             return ""
1417:         lines = [
1418:             f"{r.subject} {r.predicate.replace('_', ' ')} {r.object}"
1419:             for r in rels[:10]
1420:         ]
1421:         return ". ".join(lines) + "."
1422: 
1423:     # ------------------------------------------------------------------
1424:     # Forget (soft tombstone) + faceted show
1425:     # ------------------------------------------------------------------
1426: 
1427:     def _append_forget_audit(
1428:         self, conn: sqlite3.Connection, user_id: int, op_id: str,
1429:         reason: str, count: int,
1430:     ) -> None:
1431:         """Append an audit log row into the memories table."""
1432:         try:
1433:             conn.execute(
1434:                 """INSERT INTO memories (user_id, content, memory_type, importance)
1435:                    VALUES (?, ?, 'summary', 0.9)""",
1436:                 (user_id, f"forget op {op_id}: reason={reason}, n={count}"),
1437:             )
1438:         except Exception as e:
1439:             log.warning("forget audit write failed: %s", e)
1440: 
1441:     def _tombstone_rows(
1442:         self, conn: sqlite3.Connection, user_id: int,
1443:         ids: List[int], reason: str, op_id: str,
1444:     ) -> int:
1445:         """Flip tombstone flags on a set of relationship rows."""
1446:         if not ids:
1447:             return 0
1448:         placeholders = ",".join("?" for _ in ids)
1449:         cur = conn.execute(
1450:             f"""UPDATE relationships SET
1451:                   tombstoned_at = datetime('now'),
1452:                   tombstone_reason = ?,
1453:                   tombstone_op_id = ?
1454:                 WHERE user_id = ? AND tombstoned_at IS NULL
1455:                   AND id IN ({placeholders})""",
1456:             [reason, op_id, user_id] + list(ids),
1457:         )
1458:         return cur.rowcount or 0
1459: 
1460:     def forget_by_triple_id(self, user_id: int, triple_id: int) -> int:
1461:         """Tombstone a single triple by its relationship id."""
1462:         from uuid import uuid4
1463:         op_id = uuid4().hex
1464:         reason = f"by_triple_id:{int(triple_id)}"
1465:         try:
1466:             with get_db_context() as conn:
1467:                 row = conn.execute(
1468:                     """SELECT id FROM relationships
1469:                        WHERE user_id = ? AND id = ?
1470:                          AND tombstoned_at IS NULL
1471:                          AND COALESCE(is_current, 1) = 1""",
1472:                     (user_id, int(triple_id)),
1473:                 ).fetchone()
1474:                 if not row:
1475:                     return 0
1476:                 count = self._tombstone_rows(
1477:                     conn, user_id, [row["id"]], reason, op_id
1478:                 )
1479:                 if count:
1480:                     self._append_forget_audit(conn, user_id, op_id, reason, count)
1481:                 conn.commit()
1482:                 return count
1483:         except Exception as e:
1484:             log.error("forget_by_triple_id failed: %s", e)
1485:             return 0
1486: 
1487:     def forget_by_entity(self, user_id: int, entity_name: str) -> int:
1488:         """Tombstone every live triple where the entity appears as subject
1489:         or object (case-insensitive exact match)."""
1490:         from uuid import uuid4
1491:         if not entity_name or not entity_name.strip():
1492:             return 0
1493:         name = entity_name.strip()
1494:         op_id = uuid4().hex
1495:         reason = f"by_entity:{name}"
1496:         try:
1497:             with get_db_context() as conn:
1498:                 rows = conn.execute(
1499:                     """SELECT id FROM relationships
1500:                        WHERE user_id = ? AND tombstoned_at IS NULL
1501:                          AND COALESCE(is_current, 1) = 1
1502:                          AND (LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))""",
1503:                     (user_id, name, name),
1504:                 ).fetchall()
1505:                 ids = [r["id"] for r in rows]
1506:                 count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
1507:                 if count:
1508:                     self._append_forget_audit(conn, user_id, op_id, reason, count)
1509:                 conn.commit()
1510:                 return count
1511:         except Exception as e:
1512:             log.error("forget_by_entity failed: %s", e)
1513:             return 0
1514: 
1515:     def forget_by_time_range(
1516:         self, user_id: int, start_iso: str, end_iso: str,
1517:     ) -> int:
1518:         """Tombstone every live triple whose source_timestamp falls
1519:         within [start_iso, end_iso]."""
1520:         from uuid import uuid4
1521:         op_id = uuid4().hex
1522:         reason = f"by_time_range:{start_iso}..{end_iso}"
1523:         try:
1524:             with get_db_context() as conn:
1525:                 rows = conn.execute(
1526:                     """SELECT id FROM relationships
1527:                        WHERE user_id = ? AND tombstoned_at IS NULL
1528:                          AND COALESCE(is_current, 1) = 1
1529:                          AND source_timestamp IS NOT NULL
1530:                          AND source_timestamp >= ?
1531:                          AND source_timestamp <= ?""",
1532:                     (user_id, start_iso, end_iso),
1533:                 ).fetchall()
1534:                 ids = [r["id"] for r in rows]
1535:                 count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
1536:                 if count:
1537:                     self._append_forget_audit(conn, user_id, op_id, reason, count)
1538:                 conn.commit()
1539:                 return count
1540:         except Exception as e:
1541:             log.error("forget_by_time_range failed: %s", e)
1542:             return 0
1543: 
1544:     def forget_by_source(self, user_id: int, source_tag: str) -> int:
1545:         """Tombstone every live triple whose source_tag exactly matches."""
1546:         from uuid import uuid4
1547:         if source_tag is None:
1548:             return 0
1549:         op_id = uuid4().hex
1550:         reason = f"by_source:{source_tag}"
1551:         try:
1552:             with get_db_context() as conn:
1553:                 rows = conn.execute(
1554:                     """SELECT id FROM relationships
1555:                        WHERE user_id = ? AND tombstoned_at IS NULL
1556:                          AND COALESCE(is_current, 1) = 1
1557:                          AND source_tag = ?""",
1558:                     (user_id, source_tag),
1559:                 ).fetchall()
1560:                 ids = [r["id"] for r in rows]
1561:                 count = self._tombstone_rows(conn, user_id, ids, reason, op_id)
1562:                 if count:
1563:                     self._append_forget_audit(conn, user_id, op_id, reason, count)
1564:                 conn.commit()
1565:                 return count
1566:         except Exception as e:
1567:             log.error("forget_by_source failed: %s", e)
1568:             return 0
1569: 
1570:     # ------------------------------------------------------------------
1571:     # Faceted listing
1572:     # ------------------------------------------------------------------
1573: 
1574:     def list_by_facet(
1575:         self,
1576:         user_id: int,
1577:         facet: str,
1578:         value: Optional[str] = None,
1579:         limit: int = 100,
1580:     ) -> List[Dict[str, Any]]:
1581:         """Return live triples matching a facet view.
1582: 
1583:         Facets: 'time', 'entity', 'source', 'trace'.
1584:         """
1585:         if facet not in ("time", "entity", "source", "trace"):
1586:             raise ValueError(f"Unknown facet: {facet}")
1587:         limit = max(1, min(int(limit or 100), 1000))
1588: 
1589:         base_cols = (
1590:             "id, subject, predicate, object, "
1591:             "source_timestamp, source_tag, confidence, source_text"
1592:         )
1593:         where = [
1594:             "user_id = ?", "tombstoned_at IS NULL",
1595:             "COALESCE(is_current, 1) = 1",
1596:         ]
1597:         params: List[Any] = [user_id]
1598:         order = "id DESC"
1599: 
1600:         if facet == "time":
1601:             if value:
1602:                 where.append("source_timestamp LIKE ?")
1603:                 params.append(f"{value}%")
1604:             order = (
1605:                 "CASE WHEN source_timestamp IS NULL THEN 1 ELSE 0 END, "
1606:                 "source_timestamp DESC, id DESC"
1607:             )
1608:         elif facet == "entity":
1609:             if not value:
1610:                 return []
1611:             where.append(
1612:                 "(LOWER(subject) = LOWER(?) OR LOWER(object) = LOWER(?))"
1613:             )
1614:             params.extend([value, value])
1615:         elif facet == "source":
1616:             if value is None:
1617:                 return []
1618:             where.append("source_tag = ?")
1619:             params.append(value)
1620:         elif facet == "trace":
1621:             if value:
1622:                 where.append("edge_schematic_category = ?")
1623:                 params.append(value)
1624: 
1625:         sql = (
1626:             f"SELECT {base_cols} FROM relationships "
1627:             f"WHERE {' AND '.join(where)} "
1628:             f"ORDER BY {order} LIMIT ?"
1629:         )
1630:         params.append(limit)
1631: 
1632:         try:
1633:             with get_db_context() as conn:
1634:                 rows = conn.execute(sql, params).fetchall()
1635:         except Exception as e:
1636:             log.error("list_by_facet failed: %s", e)
1637:             return []
1638: 
1639:         return [
1640:             {
1641:                 "id": r["id"],
1642:                 "subject": r["subject"],
1643:                 "predicate": r["predicate"],
1644:                 "object": r["object"],
1645:                 "source_timestamp": r["source_timestamp"],
1646:                 "source_tag": r["source_tag"],
1647:                 "confidence": r["confidence"] or 0.9,
1648:                 "source_text": r["source_text"],
1649:             }
1650:             for r in rows
1651:         ]
1652: 
1653: 
1654: # =============================================================================
1655: # Module-level singleton
1656: # =============================================================================
1657: 
1658: _singleton: Optional[MemoryEngine] = None
1659: 
1660: 
1661: def get_memory_engine() -> MemoryEngine:
1662:     global _singleton
1663:     if _singleton is None:
1664:         _singleton = MemoryEngine()
1665:     return _singleton
```

## app/engines/temporal.py

Source: [app/engines/temporal.py](/D:/Nura/Code/nura_living_memory_code/app/engines/temporal.py)

```text
0001: # ============================================================================
0002: # NO HARDCODED LISTS. NO THRESHOLDS. NO SCORING MAGIC NUMBERS. NO REGEX.
0003: # ============================================================================
0004: """
0005: TemporalEngine â€” all time-related concerns + supersession.
0006: 
0007: Absorbs: temporal_engine, temporal_patterns, time_humanizer, time_authority,
0008: calendar_model, temporal_staleness, duration_extractor, correction_handler,
0009: contradiction_detector, semantic/temporal_concepts.
0010: 
0011: COMPLEMENTING design:
0012:     - Owns wall-clock time (parse, now, calendar).
0013:     - Owns narrative time via sequence_number assigned by MemoryEngine.
0014:     - Owns the temporal knowledge graph (sequence ordering + clusters).
0015:     - Owns SUPERSESSION: cluster-based detection of corrections + contradictions.
0016:       correction_handler and contradiction_detector do NOT exist as separate
0017:       modules anymore â€” their semantics live here as `detect_supersession`.
0018: 
0019: Single public interface:
0020:     parse(text, now) â†’ TemporalResult
0021:     resolve_event_date(text, reference_timestamp) â†’ Optional[str]
0022:     cluster(user_id, utterance, src_ts) â†’ Cluster
0023:     detect_supersession(user_id, new_utt, new_rel_id, cluster) â†’ Optional[SupersessionEvent]
0024:     recluster_for_reconstruction(user_id, edge_ids) â†’ List[dict]
0025:     detect_arcs(user_id, new_rel_id, cluster_id) â†’ Optional[str]
0026:     edges_valid_at(user_id, timestamp) â†’ List[dict]
0027:     temporal_neighbors(user_id, relationship_id, window) â†’ List[dict]
0028:     patterns(user_id) â†’ list[TemporalPattern]
0029:     staleness(relationship_id) â†’ float
0030:     humanize(delta) â†’ str
0031:     now() â†’ datetime
0032: """
0033: from __future__ import annotations
0034: 
0035: import uuid
0036: from dataclasses import dataclass, field
0037: from datetime import datetime, timezone, timedelta
0038: from typing import Any, Dict, List, Optional, Tuple
0039: 
0040: import numpy as np
0041: 
0042: from app.db.session import get_db_context
0043: from app.vector.embedder import embed_text
0044: 
0045: 
0046: # =============================================================================
0047: # LAZY LOADERS â€” spaCy and dateparser are heavy; load once on first use
0048: # =============================================================================
0049: 
0050: _spacy_nlp = None
0051: 
0052: 
0053: def _get_spacy():
0054:     global _spacy_nlp
0055:     if _spacy_nlp is None:
0056:         import spacy
0057:         _spacy_nlp = spacy.load("en_core_web_sm")
0058:     return _spacy_nlp
0059: 
0060: 
0061: # =============================================================================
0062: # DATA MODEL
0063: # =============================================================================
0064: 
0065: @dataclass
0066: class TemporalResult:
0067:     parsed_datetime: Optional[datetime] = None
0068:     duration_seconds: Optional[float] = None
0069:     direction: str = "present"  # "past" | "present" | "future" | "ongoing"
0070:     retrieval_window_days: Optional[int] = None
0071:     disable_recency: bool = False
0072: 
0073: 
0074: @dataclass
0075: class Cluster:
0076:     cluster_id: int
0077:     member_relationship_ids: List[int] = field(default_factory=list)
0078:     span_seconds: float = 0.0
0079:     centroid_embedding: Optional[np.ndarray] = None
0080:     last_updated_at: Optional[str] = None
0081: 
0082: 
0083: @dataclass
0084: class SupersessionEvent:
0085:     superseded_relationship_id: int
0086:     superseding_relationship_id: int
0087:     reason: str  # "correction" | "contradiction" | "update" | "negation"
0088:     cluster_id: int
0089: 
0090: 
0091: @dataclass
0092: class TemporalPattern:
0093:     pattern_type: str
0094:     confidence: float
0095:     exemplar_relationship_ids: List[int] = field(default_factory=list)
0096: 
0097: 
0098: # =============================================================================
0099: # ANCHOR DESCRIPTIONS â€” supersession + temporal semantics.
0100: # Stable concept anchors, matched via cosine. Not curation.
0101: # =============================================================================
0102: 
0103: _SUPERSESSION_CORRECTION_ANCHOR = (
0104:     "actually I meant no it's not I misspoke correction let me fix sorry wrong"
0105: )
0106: _SUPERSESSION_UPDATE_ANCHOR = (
0107:     "now it's updated changed moved to switched from earlier no longer as of"
0108: )
0109: _SUPERSESSION_NEGATION_ANCHOR = (
0110:     "not never no longer doesn't isn't didn't aren't nobody nothing nowhere"
0111: )
0112: 
0113: _TEMP_PAST_ANCHOR = "happened before previously ago yesterday last year earlier"
0114: _TEMP_FUTURE_ANCHOR = "will happen soon tomorrow next week upcoming plan"
0115: _TEMP_ONGOING_ANCHOR = "always every day regularly currently routine habitual"
0116: _TEMP_PRESENT_ANCHOR = "today now currently this week recent lately"
0117: 
0118: 
0119: # =============================================================================
0120: # TEMPORAL ENGINE
0121: # =============================================================================
0122: 
0123: class TemporalEngine:
0124:     """Owner of all time-related concerns."""
0125: 
0126:     def __init__(self, memory_engine=None):
0127:         self._memory = memory_engine
0128:         self._anchor_cache: Dict[str, np.ndarray] = {}
0129:         self._frozen_now: Optional[datetime] = None
0130: 
0131:     def bind_memory(self, memory_engine) -> None:
0132:         """Late binding to avoid circular construction at startup."""
0133:         self._memory = memory_engine
0134: 
0135:     # â”€â”€ Now / time authority â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0136: 
0137:     def now(self) -> datetime:
0138:         if self._frozen_now is not None:
0139:             return self._frozen_now
0140:         return datetime.now(timezone.utc)
0141: 
0142:     def freeze(self, dt: datetime) -> None:
0143:         self._frozen_now = dt
0144: 
0145:     def unfreeze(self) -> None:
0146:         self._frozen_now = None
0147: 
0148: 
0149:     # == Calendar model ====================================================
0150:     # Real-time understanding of time â€” not just wall-clock, but what
0151:     # time MEANS: proximity, urgency, ordering, duration, upcoming events.
0152: 
0153:     def temporal_context(self):
0154:         """What the engine knows about NOW."""
0155:         n = self.now()
0156:         return {
0157:             "now": n.isoformat(), "weekday": n.strftime("%A"),
0158:             "date": n.strftime("%Y-%m-%d"), "time": n.strftime("%H:%M"),
0159:             "hour": n.hour, "day_of_week": n.weekday(),
0160:             "is_weekend": n.weekday() >= 5,
0161:             "month": n.strftime("%B"), "year": n.year,
0162:         }
0163: 
0164:     def days_until(self, target):
0165:         """Days from now until target. Negative = past."""
0166:         target_dt = self._to_datetime(target)
0167:         if target_dt is None:
0168:             return None
0169:         now_naive = self.now().replace(tzinfo=None)
0170:         return (target_dt - now_naive).total_seconds() / 86400.0
0171: 
0172:     def days_since(self, target):
0173:         """Days since target until now. Negative = future."""
0174:         d = self.days_until(target)
0175:         return -d if d is not None else None
0176: 
0177:     def proximity(self, target):
0178:         """Human proximity: 'tomorrow', 'in 3 days', '2 weeks ago'."""
0179:         days = self.days_until(target)
0180:         if days is None:
0181:             return None
0182:         ad = abs(days)
0183:         fut = days > 0
0184:         if ad < 0.04:
0185:             return "right now"
0186:         if ad < 1:
0187:             return "today" if fut else "earlier today"
0188:         if ad < 2:
0189:             return "tomorrow" if fut else "yesterday"
0190:         if ad < 7:
0191:             n = round(ad)
0192:             return f"in {n} days" if fut else f"{n} days ago"
0193:         if ad < 30:
0194:             n = round(ad / 7)
0195:             u = "week" if n == 1 else "weeks"
0196:             return f"in {n} {u}" if fut else f"{n} {u} ago"
0197:         if ad < 365:
0198:             n = round(ad / 30)
0199:             u = "month" if n == 1 else "months"
0200:             return f"in {n} {u}" if fut else f"{n} {u} ago"
0201:         n = round(ad / 365)
0202:         u = "year" if n == 1 else "years"
0203:         return f"in {n} {u}" if fut else f"{n} {u} ago"
0204: 
0205:     def is_before(self, a, b):
0206:         """Is event A before event B?"""
0207:         dt_a, dt_b = self._to_datetime(a), self._to_datetime(b)
0208:         if dt_a is None or dt_b is None:
0209:             return None
0210:         return dt_a < dt_b
0211: 
0212:     def duration_between(self, a, b):
0213:         """Seconds between two events. Always positive."""
0214:         dt_a, dt_b = self._to_datetime(a), self._to_datetime(b)
0215:         if dt_a is None or dt_b is None:
0216:             return None
0217:         return abs((dt_b - dt_a).total_seconds())
0218: 
0219:     def urgency(self, user_id, relationship_id):
0220:         """How urgent is this edge right now?"""
0221:         try:
0222:             with get_db_context() as conn:
0223:                 row = conn.execute(
0224:                     "SELECT resolved_event_date, temporal_expression, "
0225:                     "edge_emotional_valence, edge_schematic_category, is_current "
0226:                     "FROM relationships WHERE id = ? AND user_id = ?",
0227:                     (relationship_id, user_id),
0228:                 ).fetchone()
0229:             if not row:
0230:                 return None
0231:             date_str = row["resolved_event_date"] or row["temporal_expression"]
0232:             if not date_str:
0233:                 return None
0234:             days = self.days_until(date_str)
0235:             if days is None:
0236:                 return None
0237:             return {
0238:                 "days": round(days, 1),
0239:                 "proximity": self.proximity(date_str),
0240:                 "is_upcoming": 0 < days <= 7,
0241:                 "is_overdue": days < 0 and bool(row["is_current"]),
0242:                 "schema": row["edge_schematic_category"],
0243:                 "valence": row["edge_emotional_valence"],
0244:             }
0245:         except Exception:
0246:             return None
0247: 
0248:     def upcoming_events(self, user_id, window_days=7):
0249:         """Edges with resolved dates within the next N days."""
0250:         n = self.now()
0251:         cutoff = (n + timedelta(days=window_days)).isoformat()
0252:         now_str = n.isoformat()
0253:         try:
0254:             with get_db_context() as conn:
0255:                 rows = conn.execute(
0256:                     "SELECT id, subject, predicate, object, "
0257:                     "resolved_event_date, source_text, edge_schematic_category "
0258:                     "FROM relationships WHERE user_id = ? "
0259:                     "AND resolved_event_date IS NOT NULL "
0260:                     "AND resolved_event_date >= ? AND resolved_event_date <= ? "
0261:                     "AND COALESCE(is_current, 1) = 1 AND tombstoned_at IS NULL "
0262:                     "ORDER BY resolved_event_date ASC",
0263:                     (user_id, now_str, cutoff),
0264:                 ).fetchall()
0265:             result = []
0266:             for r in rows:
0267:                 days = self.days_until(r["resolved_event_date"])
0268:                 result.append({
0269:                     "id": r["id"], "subject": r["subject"],
0270:                     "predicate": r["predicate"], "object": r["object"],
0271:                     "date": r["resolved_event_date"],
0272:                     "days_until": round(days, 1) if days is not None else None,
0273:                     "proximity": self.proximity(r["resolved_event_date"]),
0274:                     "schema": r["edge_schematic_category"],
0275:                 })
0276:             return result
0277:         except Exception:
0278:             return []
0279: 
0280:     def _to_datetime(self, value):
0281:         """Convert anything temporal to naive datetime."""
0282:         if value is None:
0283:             return None
0284:         if isinstance(value, datetime):
0285:             return value.replace(tzinfo=None) if value.tzinfo else value
0286:         if isinstance(value, str):
0287:             try:
0288:                 dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
0289:                 return dt.replace(tzinfo=None) if dt.tzinfo else dt
0290:             except (ValueError, TypeError):
0291:                 pass
0292:             resolved = self.resolve_event_date(value, self.now().isoformat())
0293:             if resolved:
0294:                 try:
0295:                     dt = datetime.fromisoformat(resolved)
0296:                     return dt.replace(tzinfo=None) if dt.tzinfo else dt
0297:                 except (ValueError, TypeError):
0298:                     pass
0299:         return None
0300: 
0301: 
0302:     # â”€â”€ Anchor cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0303: 
0304:     def _anchor(self, key: str, text: str) -> np.ndarray:
0305:         cached = self._anchor_cache.get(key)
0306:         if cached is not None:
0307:             return cached
0308:         emb = embed_text(text)
0309:         self._anchor_cache[key] = emb
0310:         return emb
0311: 
0312:     def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
0313:         """Cosine similarity, not raw dot product."""
0314:         na = float(np.linalg.norm(a))
0315:         nb = float(np.linalg.norm(b))
0316:         if na == 0.0 or nb == 0.0:
0317:             return 0.0
0318:         return float(np.dot(a, b) / (na * nb))
0319: 
0320:     # â”€â”€ resolve_event_date â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0321:     # ROOT CAUSE: memory.py line 1000 calls this but it did not exist,
0322:     # leaving resolved_event_date NULL on every edge.
0323: 
0324:     def resolve_event_date(
0325:         self,
0326:         text: str,
0327:         reference_timestamp: Optional[str] = None,
0328:     ) -> Optional[str]:
0329:         """Extract and resolve temporal expressions to ISO 8601.
0330: 
0331:         Called by memory.py at write time. Uses spaCy NER for DATE/TIME
0332:         span extraction, then dateparser to resolve against the reference
0333:         timestamp. Returns ISO string or None.
0334:         """
0335:         if not text or not text.strip():
0336:             return None
0337: 
0338:         # Parse reference timestamp into datetime for dateparser
0339:         ref_dt = self.now()
0340:         if reference_timestamp:
0341:             try:
0342:                 ref_dt = datetime.fromisoformat(
0343:                     reference_timestamp.replace("Z", "+00:00")
0344:                 )
0345:             except (ValueError, TypeError):
0346:                 pass
0347: 
0348:         # Step 1: Extract DATE/TIME spans via spaCy NER
0349:         try:
0350:             nlp = _get_spacy()
0351:             doc = nlp(text)
0352:             temporal_spans = [
0353:                 ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")
0354:             ]
0355:         except Exception:
0356:             temporal_spans = []
0357: 
0358:         if not temporal_spans:
0359:             return None
0360: 
0361:         # Step 2: Resolve spans via dateparser
0362:         #
0363:         # ROOT CAUSE: dateparser 1.4.0 returns None for compound expressions
0364:         # like "next Tuesday" (confirmed via bash testing). It resolves bare
0365:         # day names with PREFER_DATES_FROM. Fallback: strip all tokens except
0366:         # the head noun (the date entity itself) via spaCy dep parse, then
0367:         # try both past and future preferences. The overall sentence embedding
0368:         # cosine against temporal anchors determines which result to keep.
0369:         import dateparser
0370: 
0371:         ref_naive = ref_dt.replace(tzinfo=None) if ref_dt.tzinfo else ref_dt
0372: 
0373:         def _dateparser_resolve(span: str, prefer: str) -> Optional[datetime]:
0374:             return dateparser.parse(span, settings={
0375:                 "RELATIVE_BASE": ref_naive,
0376:                 "PREFER_DATES_FROM": prefer,
0377:                 "RETURN_AS_TIMEZONE_AWARE": False,
0378:             })
0379: 
0380:         def _head_noun(span: str) -> Optional[str]:
0381:             """Extract the syntactic head of a span via spaCy dep parse.
0382:             For 'next Tuesday', returns 'Tuesday'. For 'January', returns
0383:             'January'. Returns None if spaCy fails."""
0384:             try:
0385:                 nlp = _get_spacy()
0386:                 span_doc = nlp(span)
0387:                 # The root of the span is the head noun
0388:                 for token in span_doc:
0389:                     if token.dep_ == "ROOT" or token.head == token:
0390:                         return token.text
0391:             except Exception:
0392:                 pass
0393:             return None
0394: 
0395:         # Determine sentence-level temporal direction using the same
0396:         # max-cosine pattern as parse() (line ~263). This guides the
0397:         # head-noun fallback when dateparser cannot parse the full span.
0398:         # ROOT CAUSE: "next Tuesday" -> head noun "Tuesday" -> tried
0399:         # "past" first -> got April 25 instead of May 2. The sentence
0400:         # context ("next" = future) must be preserved in the fallback.
0401:         try:
0402:             text_emb = embed_text(text)
0403:             direction_scores = {
0404:                 "past": self._cos(text_emb, self._anchor("temp_past", _TEMP_PAST_ANCHOR)),
0405:                 "future": self._cos(text_emb, self._anchor("temp_future", _TEMP_FUTURE_ANCHOR)),
0406:             }
0407:             sentence_prefer = max(direction_scores, key=direction_scores.get)
0408:         except Exception:
0409:             sentence_prefer = "past"
0410: 
0411:         def _try_parse(span: str) -> Optional[datetime]:
0412:             # Try direct parse with sentence-inferred preference
0413:             result = _dateparser_resolve(span, sentence_prefer)
0414:             if result is not None:
0415:                 return result
0416:             # Try the opposite preference
0417:             alt_prefer = "future" if sentence_prefer == "past" else "past"
0418:             result = _dateparser_resolve(span, alt_prefer)
0419:             if result is not None:
0420:                 return result
0421:             # Fallback: extract head noun, try sentence-inferred then opposite
0422:             head = _head_noun(span)
0423:             if head and head != span:
0424:                 result = _dateparser_resolve(head, sentence_prefer)
0425:                 if result is not None:
0426:                     return result
0427:                 result = _dateparser_resolve(head, alt_prefer)
0428:                 if result is not None:
0429:                     return result
0430:             return None
0431: 
0432:         # Try combined span first (e.g., "next Tuesday at 3 PM"), then
0433:         # individual spans. The longest resolved result wins.
0434:         candidates: List[Tuple[str, datetime]] = []
0435: 
0436:         if len(temporal_spans) > 1:
0437:             combined = " ".join(temporal_spans)
0438:             parsed = _try_parse(combined)
0439:             if parsed:
0440:                 candidates.append((combined, parsed))
0441: 
0442:         for span_text in temporal_spans:
0443:             parsed = _try_parse(span_text)
0444:             if parsed:
0445:                 candidates.append((span_text, parsed))
0446: 
0447:         if not candidates:
0448:             return None
0449: 
0450:         # Pick the candidate with the most specificity (longest input text)
0451:         best_text, best_dt = max(candidates, key=lambda c: len(c[0]))
0452:         return best_dt.isoformat()
0453: 
0454:     # â”€â”€ Parse â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0455: 
0456:     def parse(
0457:         self,
0458:         text: str,
0459:         now: Optional[datetime] = None,
0460:     ) -> TemporalResult:
0461:         """Parse temporal references. Returns direction + parsed datetime + duration.
0462: 
0463:         Direction comes from cosine against temporal anchors.
0464:         Parsed datetime comes from resolve_event_date.
0465:         Duration comes from dateparser on duration-like spans.
0466:         """
0467:         if not text or not text.strip():
0468:             return TemporalResult()
0469: 
0470:         try:
0471:             emb = embed_text(text)
0472:         except Exception:
0473:             return TemporalResult()
0474: 
0475:         scores = {
0476:             "past": self._cos(emb, self._anchor("temp_past", _TEMP_PAST_ANCHOR)),
0477:             "future": self._cos(emb, self._anchor("temp_future", _TEMP_FUTURE_ANCHOR)),
0478:             "ongoing": self._cos(emb, self._anchor("temp_ongoing", _TEMP_ONGOING_ANCHOR)),
0479:             "present": self._cos(emb, self._anchor("temp_present", _TEMP_PRESENT_ANCHOR)),
0480:         }
0481:         direction = max(scores, key=scores.get)
0482: 
0483:         # Resolve concrete datetime
0484:         ref_ts = (now or self.now()).isoformat()
0485:         resolved_iso = self.resolve_event_date(text, ref_ts)
0486:         parsed_dt = None
0487:         if resolved_iso:
0488:             try:
0489:                 parsed_dt = datetime.fromisoformat(resolved_iso)
0490:             except (ValueError, TypeError):
0491:                 pass
0492: 
0493:         # Resolve duration for "for 6 months" / "since January" patterns
0494:         duration_seconds = self._extract_duration(text, now or self.now())
0495: 
0496:         return TemporalResult(
0497:             direction=direction,
0498:             parsed_datetime=parsed_dt,
0499:             duration_seconds=duration_seconds,
0500:         )
0501: 
0502:     def _extract_duration(
0503:         self,
0504:         text: str,
0505:         ref: datetime,
0506:     ) -> Optional[float]:
0507:         """Extract duration in seconds from duration-like expressions.
0508: 
0509:         Uses dateparser to resolve "since January" or "for 6 months" by
0510:         computing the difference between the resolved date and reference.
0511:         """
0512:         import dateparser
0513: 
0514:         # Look for temporal spans via spaCy NER
0515:         try:
0516:             nlp = _get_spacy()
0517:             doc = nlp(text)
0518:             duration_spans = [
0519:                 ent.text for ent in doc.ents
0520:                 if ent.label_ in ("DATE", "TIME")
0521:             ]
0522:         except Exception:
0523:             return None
0524: 
0525:         if not duration_spans:
0526:             return None
0527: 
0528:         ref_naive = ref.replace(tzinfo=None) if ref.tzinfo else ref
0529:         settings = {
0530:             "RELATIVE_BASE": ref_naive,
0531:             "PREFER_DATES_FROM": "past",
0532:             "RETURN_AS_TIMEZONE_AWARE": False,
0533:         }
0534: 
0535:         # Check for duration structure via spaCy dependency parse:
0536:         # Duration expressions have prep/mark tokens ('since', 'for')
0537:         # governing a temporal NP. Using dep parse, not string matching.
0538:         has_duration_dep = False
0539:         try:
0540:             nlp = _get_spacy()
0541:             doc = nlp(text)
0542:             for tok in doc:
0543:                 if tok.dep_ in ("prep", "mark") and tok.pos_ == "ADP":
0544:                     # Check if this preposition governs a DATE/TIME entity
0545:                     for child in tok.children:
0546:                         if child.ent_type_ in ("DATE", "TIME"):
0547:                             has_duration_dep = True
0548:                             break
0549:                 if has_duration_dep:
0550:                     break
0551:         except Exception:
0552:             pass
0553: 
0554:         if not has_duration_dep:
0555:             return None
0556: 
0557:         for span_text in duration_spans:
0558:             parsed = dateparser.parse(span_text, settings=settings)
0559:             if parsed:
0560:                 diff = abs((ref_naive - parsed).total_seconds())
0561:                 if diff > 0:
0562:                     return diff
0563: 
0564:         return None
0565: 
0566:     # â”€â”€ Supersession â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0567:     # ROOT CAUSE: memory.py line 370 passes an int (rel_id) but old
0568:     # code expected a Relationship object, causing silent failure.
0569:     # Also: no same-source guard, no predicate normalization.
0570: 
0571:     def detect_supersession(
0572:         self,
0573:         user_id: int,
0574:         new_utterance: str,
0575:         new_relationship: Any,  # int (rel_id) or Relationship object
0576:         cluster: Optional[Cluster],
0577:     ) -> Optional[SupersessionEvent]:
0578:         """Given a newly-stored triple, decide whether it supersedes any
0579:         prior member of its temporal cluster.
0580: 
0581:         Accepts either an int (relationship ID) or a Relationship object
0582:         to support both memory.py call sites.
0583: 
0584:         Neural decision:
0585:           1. Score the new utterance's cosine against supersession-shape
0586:              anchors (correction, update, negation).
0587:           2. Same-source guard: skip if source_text_hash matches.
0588:           3. Predicate normalization via grammar_engine.classify_verb_class.
0589:           4. Find prior with same (subject, predicate class) â€” that's the
0590:              superseded one.
0591: 
0592:         Returns SupersessionEvent if detected; caller invokes
0593:         MemoryEngine.supersede(). Returns None otherwise.
0594:         """
0595:         if self._memory is None:
0596:             return None
0597:         if not new_utterance or not new_utterance.strip():
0598:             return None
0599: 
0600:         # Resolve relationship ID â€” accept int or object with .id
0601:         new_id = None
0602:         if isinstance(new_relationship, int):
0603:             new_id = new_relationship
0604:         elif new_relationship is not None and hasattr(new_relationship, "id"):
0605:             new_id = new_relationship.id
0606:         else:
0607:             return None
0608: 
0609:         # Fetch the new edge's fields from DB
0610:         try:
0611:             with get_db_context() as conn:
0612:                 new_row = conn.execute(
0613:                     """SELECT id, subject, predicate, object, source_text_hash
0614:                        FROM relationships WHERE id = ?""",
0615:                     (new_id,),
0616:                 ).fetchone()
0617:         except Exception:
0618:             return None
0619: 
0620:         if not new_row:
0621:             return None
0622: 
0623:         new_subj = new_row["subject"] or ""
0624:         new_pred = new_row["predicate"] or ""
0625:         new_obj = (new_row["object"] or "").lower()
0626:         new_hash = new_row["source_text_hash"] or ""
0627: 
0628:         # Does the utterance "sound like" a correction / update / negation?
0629:         try:
0630:             utt_emb = embed_text(new_utterance)
0631:         except Exception:
0632:             return None
0633: 
0634:         sup_scores = {
0635:             "correction": self._cos(utt_emb, self._anchor("sup_corr", _SUPERSESSION_CORRECTION_ANCHOR)),
0636:             "update": self._cos(utt_emb, self._anchor("sup_upd", _SUPERSESSION_UPDATE_ANCHOR)),
0637:             "negation": self._cos(utt_emb, self._anchor("sup_neg", _SUPERSESSION_NEGATION_ANCHOR)),
0638:         }
0639:         best_shape, best_shape_score = max(sup_scores.items(), key=lambda kv: kv[1])
0640: 
0641:         # Shape must stand out above the mean of the other scores
0642:         # (population-relative, not absolute threshold).
0643:         other_scores = [s for k, s in sup_scores.items() if k != best_shape]
0644:         other_mean = sum(other_scores) / max(len(other_scores), 1)
0645:         if best_shape_score <= other_mean:
0646:             return None
0647: 
0648:         # Predicate normalization via grammar_engine (open-vocabulary)
0649:         pred_class = None
0650:         try:
0651:             from app.engines.grammar_engine import classify_verb_class
0652:             pred_class = classify_verb_class(new_pred.lower().split()[0] if new_pred else "")
0653:         except Exception:
0654:             pass
0655: 
0656:         # Find priors with same subject
0657:         cluster_id = cluster.cluster_id if cluster else 0
0658:         with get_db_context() as conn:
0659:             rows = conn.execute(
0660:                 """SELECT id, subject, predicate, object, source_text_hash
0661:                    FROM relationships
0662:                    WHERE user_id = ? AND id != ?
0663:                      AND LOWER(subject) = LOWER(?)
0664:                      AND COALESCE(is_current, 1) = 1
0665:                    ORDER BY id DESC LIMIT 20""",
0666:                 (user_id, new_id, new_subj),
0667:             ).fetchall()
0668: 
0669:         for prior in rows:
0670:             # Same-source guard: skip if same utterance produced both edges
0671:             prior_hash = prior["source_text_hash"] or ""
0672:             if new_hash and prior_hash and new_hash == prior_hash:
0673:                 continue
0674: 
0675:             prior_pred = (prior["predicate"] or "").lower()
0676:             new_pred_lower = new_pred.lower()
0677: 
0678:             # Check predicate match: exact match or same verb class
0679:             pred_match = (prior_pred == new_pred_lower)
0680:             if not pred_match and pred_class is not None:
0681:                 try:
0682:                     from app.engines.grammar_engine import classify_verb_class
0683:                     prior_class = classify_verb_class(
0684:                         prior_pred.split()[0] if prior_pred else ""
0685:                     )
0686:                     pred_match = (prior_class == pred_class)
0687:                 except Exception:
0688:                     pass
0689: 
0690:             if not pred_match:
0691:                 continue
0692: 
0693:             prior_obj = (prior["object"] or "").lower()
0694:             if prior_obj != new_obj:
0695:                 # Different object with same (subject, predicate class) = supersession
0696:                 # Temporal engine owns the write â€” mark old edge directly.
0697:                 try:
0698:                     conn.execute(
0699:                         "UPDATE relationships SET is_current = 0, "
0700:                         "superseded_at = datetime('now'), superseded_by = ? "
0701:                         "WHERE id = ?",
0702:                         (new_id, prior["id"]),
0703:                     )
0704:                     conn.commit()
0705:                 except Exception:
0706:                     pass
0707:                 return SupersessionEvent(
0708:                     superseded_relationship_id=prior["id"],
0709:                     superseding_relationship_id=new_id,
0710:                     reason=best_shape,
0711:                     cluster_id=cluster_id,
0712:                 )
0713: 
0714:         return None
0715: 
0716:     # â”€â”€ Recluster for reconstruction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0717:     # ROOT CAUSE: architecture_verifier expects this method but it did
0718:     # not exist. Retrieval reconstruct() cannot group edges for narrative.
0719: 
0720:     def recluster_for_reconstruction(
0721:         self,
0722:         user_id: int,
0723:         edge_ids: List[int],
0724:     ) -> List[dict]:
0725:         """Union-find clustering on shared entities across edges.
0726: 
0727:         Groups edges that share subject or object entities, orders
0728:         clusters by mean sequence_number, and returns list of
0729:         {cluster_id, edge_ids, participants}.
0730: 
0731:         Excludes generic "user" entity from union-find to prevent
0732:         collapsing all edges into one cluster.
0733:         """
0734:         if not edge_ids:
0735:             return []
0736: 
0737:         # Fetch edge data
0738:         placeholders = ",".join("?" for _ in edge_ids)
0739:         with get_db_context() as conn:
0740:             rows = conn.execute(
0741:                 f"""SELECT id, subject, object, sequence_number,
0742:                            relational_entities
0743:                     FROM relationships
0744:                     WHERE id IN ({placeholders}) AND user_id = ?""",
0745:                 (*edge_ids, user_id),
0746:             ).fetchall()
0747: 
0748:         if not rows:
0749:             return []
0750: 
0751:         # Build entity-to-edge mapping
0752:         _GENERIC = {"user", "i", "me", "my"}
0753:         entity_to_edges: Dict[str, List[int]] = {}
0754:         edge_data: Dict[int, dict] = {}
0755: 
0756:         for row in rows:
0757:             eid = row["id"]
0758:             edge_data[eid] = dict(row)
0759:             entities: set = set()
0760: 
0761:             for field_name in ("subject", "object"):
0762:                 val = row[field_name]
0763:                 if val and val.lower() not in _GENERIC:
0764:                     entities.add(val.lower())
0765: 
0766:             # Parse relational_entities JSON if present
0767:             rel_ents = row["relational_entities"]
0768:             if rel_ents:
0769:                 try:
0770:                     import json
0771:                     parsed = json.loads(rel_ents)
0772:                     if isinstance(parsed, list):
0773:                         for e in parsed:
0774:                             if isinstance(e, str) and e.lower() not in _GENERIC:
0775:                                 entities.add(e.lower())
0776:                 except Exception:
0777:                     pass
0778: 
0779:             for ent in entities:
0780:                 entity_to_edges.setdefault(ent, []).append(eid)
0781: 
0782:         # Union-find
0783:         parent: Dict[int, int] = {eid: eid for eid in edge_data}
0784: 
0785:         def find(x: int) -> int:
0786:             while parent[x] != x:
0787:                 parent[x] = parent[parent[x]]
0788:                 x = parent[x]
0789:             return x
0790: 
0791:         def union(a: int, b: int) -> None:
0792:             ra, rb = find(a), find(b)
0793:             if ra != rb:
0794:                 parent[ra] = rb
0795: 
0796:         for ent, eids in entity_to_edges.items():
0797:             for i in range(1, len(eids)):
0798:                 union(eids[0], eids[i])
0799: 
0800:         # Collect clusters
0801:         cluster_map: Dict[int, List[int]] = {}
0802:         for eid in edge_data:
0803:             root = find(eid)
0804:             cluster_map.setdefault(root, []).append(eid)
0805: 
0806:         # Build result sorted by mean sequence_number
0807:         result = []
0808:         for idx, (root, members) in enumerate(cluster_map.items()):
0809:             seq_nums = [
0810:                 edge_data[m].get("sequence_number") or 0
0811:                 for m in members
0812:             ]
0813:             mean_seq = sum(seq_nums) / max(len(seq_nums), 1)
0814: 
0815:             participants = set()
0816:             for m in members:
0817:                 for field_name in ("subject", "object"):
0818:                     val = edge_data[m].get(field_name)
0819:                     if val and val.lower() not in _GENERIC:
0820:                         participants.add(val)
0821: 
0822:             result.append({
0823:                 "cluster_id": idx,
0824:                 "edge_ids": sorted(members),
0825:                 "participants": sorted(participants),
0826:                 "_mean_seq": mean_seq,
0827:             })
0828: 
0829:         result.sort(key=lambda c: c["_mean_seq"])
0830: 
0831:         # Remove internal sort key
0832:         for c in result:
0833:             c.pop("_mean_seq", None)
0834: 
0835:         return result
0836: 
0837:     # â”€â”€ Arc detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0838:     # ROOT CAUSE: architecture_verifier expects this method but it did
0839:     # not exist. Arcs table exists in schema but nothing writes to it.
0840: 
0841:     def detect_arcs(
0842:         self,
0843:         user_id: int,
0844:         new_rel_id: int,
0845:         cluster_id: Optional[str] = None,
0846:     ) -> Optional[str]:
0847:         """Check arcs table for open arc matching subject + schema.
0848: 
0849:         If found, add edge to arc. If not, create new arc.
0850:         Return arc_id string.
0851:         """
0852:         try:
0853:             with get_db_context() as conn:
0854:                 # Fetch new edge data
0855:                 edge = conn.execute(
0856:                     """SELECT subject, object, predicate, source_text
0857:                        FROM relationships WHERE id = ? AND user_id = ?""",
0858:                     (new_rel_id, user_id),
0859:                 ).fetchone()
0860: 
0861:                 if not edge:
0862:                     return None
0863: 
0864:                 subject = edge["subject"] or ""
0865:                 source_text = edge["source_text"] or edge["object"] or ""
0866: 
0867:                 # Embed the edge topic for arc matching
0868:                 try:
0869:                     edge_emb = embed_text(source_text)
0870:                 except Exception:
0871:                     return None
0872: 
0873:                 # Find open arcs for this user
0874:                 open_arcs = conn.execute(
0875:                     """SELECT id, topic, topic_embedding
0876:                        FROM arcs
0877:                        WHERE user_id = ? AND status = 'open'
0878:                        ORDER BY last_checked_at DESC LIMIT 20""",
0879:                     (user_id,),
0880:                 ).fetchall()
0881: 
0882:                 best_arc_id = None
0883:                 best_sim = -1.0
0884: 
0885:                 all_sims = []
0886:                 for arc in open_arcs:
0887:                     arc_emb_blob = arc["topic_embedding"]
0888:                     if not arc_emb_blob:
0889:                         continue
0890:                     try:
0891:                         arc_emb = np.frombuffer(arc_emb_blob, dtype=np.float32)
0892:                         sim = self._cos(edge_emb, arc_emb)
0893:                         all_sims.append(sim)
0894:                         if sim > best_sim:
0895:                             best_sim = sim
0896:                             best_arc_id = arc["id"]
0897:                     except Exception:
0898:                         continue
0899: 
0900:                 # Population-relative: best must exceed mean of all open arcs
0901:                 if best_arc_id is not None and all_sims:
0902:                     mean_sim = sum(all_sims) / len(all_sims)
0903:                     if best_sim > mean_sim:
0904:                         conn.execute(
0905:                             "UPDATE arcs SET last_checked_at = datetime('now') WHERE id = ?",
0906:                             (best_arc_id,),
0907:                         )
0908:                         conn.execute(
0909:                             "UPDATE relationships SET arc_id = ? WHERE id = ?",
0910:                             (best_arc_id, new_rel_id),
0911:                         )
0912:                         return best_arc_id
0913: 
0914:                 # Create new arc
0915:                 arc_id = str(uuid.uuid4())
0916:                 topic = f"{subject}: {edge['predicate'] or ''} {edge['object'] or ''}".strip()
0917:                 conn.execute(
0918:                     """INSERT INTO arcs (id, user_id, topic, topic_embedding,
0919:                                         start_edge_id, status)
0920:                        VALUES (?, ?, ?, ?, ?, 'open')""",
0921:                     (arc_id, user_id, topic, edge_emb.tobytes(), new_rel_id),
0922:                 )
0923:                 conn.execute(
0924:                     "UPDATE relationships SET arc_id = ? WHERE id = ?",
0925:                     (arc_id, new_rel_id),
0926:                 )
0927:                 return arc_id
0928: 
0929:         except Exception:
0930:             return None
0931: 
0932:     # â”€â”€ Temporal graph helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0933: 
0934:     def edges_valid_at(
0935:         self,
0936:         user_id: int,
0937:         timestamp: Any,  # str or datetime
0938:     ) -> List[dict]:
0939:         """Return edges valid at a given timestamp.
0940: 
0941:         An edge is valid if resolved_event_date <= timestamp AND
0942:         (tombstoned_at IS NULL OR tombstoned_at > timestamp).
0943:         """
0944:         try:
0945:             ts_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
0946:             with get_db_context() as conn:
0947:                 rows = conn.execute(
0948:                     """SELECT id, subject, predicate, object, resolved_event_date,
0949:                               source_timestamp, is_current
0950:                        FROM relationships
0951:                        WHERE user_id = ?
0952:                          AND (resolved_event_date IS NOT NULL AND resolved_event_date <= ?)
0953:                          AND (tombstoned_at IS NULL OR tombstoned_at > ?)
0954:                        ORDER BY resolved_event_date ASC""",
0955:                     (user_id, ts_str, ts_str),
0956:                 ).fetchall()
0957:             return [dict(r) for r in rows]
0958:         except Exception:
0959:             return []
0960: 
0961:     def temporal_neighbors(
0962:         self,
0963:         user_id: int,
0964:         relationship_id: int,
0965:         window: int = 5,
0966:     ) -> List[dict]:
0967:         """Return edges before and after the given edge in sequence order.
0968: 
0969:         Returns up to `window` edges on each side based on sequence_number.
0970:         """
0971:         try:
0972:             with get_db_context() as conn:
0973:                 target = conn.execute(
0974:                     "SELECT sequence_number FROM relationships WHERE id = ? AND user_id = ?",
0975:                     (relationship_id, user_id),
0976:                 ).fetchone()
0977: 
0978:                 if not target or target["sequence_number"] is None:
0979:                     return []
0980: 
0981:                 seq = target["sequence_number"]
0982: 
0983:                 rows = conn.execute(
0984:                     """SELECT id, subject, predicate, object, sequence_number,
0985:                               resolved_event_date, source_timestamp
0986:                        FROM relationships
0987:                        WHERE user_id = ?
0988:                          AND sequence_number BETWEEN ? AND ?
0989:                          AND id != ?
0990:                          AND COALESCE(is_current, 1) = 1
0991:                        ORDER BY sequence_number ASC""",
0992:                     (user_id, seq - window, seq + window, relationship_id),
0993:                 ).fetchall()
0994:             return [dict(r) for r in rows]
0995:         except Exception:
0996:             return []
0997: 
0998:     # â”€â”€ Patterns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
0999: 
1000:     def patterns(self, user_id: int) -> List[TemporalPattern]:
1001:         """Day-of-week / hour-of-day patterns from stored timestamps.
1002: 
1003:         Identifies circadian patterns by grouping edges by hour-of-day
1004:         and day-of-week from source_timestamp. Peak detection uses
1005:         population-relative comparison (above mean density).
1006:         """
1007:         try:
1008:             with get_db_context() as conn:
1009:                 rows = conn.execute(
1010:                     """SELECT id, source_timestamp
1011:                        FROM relationships
1012:                        WHERE user_id = ? AND source_timestamp IS NOT NULL
1013:                        ORDER BY id DESC LIMIT 200""",
1014:                     (user_id,),
1015:                 ).fetchall()
1016: 
1017:             if not rows:
1018:                 return []
1019: 
1020:             hour_counts: Dict[int, List[int]] = {}
1021:             dow_counts: Dict[int, List[int]] = {}
1022: 
1023:             for row in rows:
1024:                 try:
1025:                     ts = datetime.fromisoformat(
1026:                         row["source_timestamp"].replace("Z", "+00:00")
1027:                     )
1028:                     hour_counts.setdefault(ts.hour, []).append(row["id"])
1029:                     dow_counts.setdefault(ts.weekday(), []).append(row["id"])
1030:                 except Exception:
1031:                     continue
1032: 
1033:             patterns_out: List[TemporalPattern] = []
1034: 
1035:             if hour_counts:
1036:                 mean_count = len(rows) / 24.0
1037:                 for hour, ids in hour_counts.items():
1038:                     if len(ids) > mean_count:
1039:                         ratio = len(ids) / max(len(rows), 1)
1040:                         patterns_out.append(TemporalPattern(
1041:                             pattern_type=f"peak_hour_{hour}",
1042:                             confidence=ratio,
1043:                             exemplar_relationship_ids=ids[:5],
1044:                         ))
1045: 
1046:             if dow_counts:
1047:                 mean_count = len(rows) / 7.0
1048:                 day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
1049:                 for dow, ids in dow_counts.items():
1050:                     if len(ids) > mean_count:
1051:                         ratio = len(ids) / max(len(rows), 1)
1052:                         patterns_out.append(TemporalPattern(
1053:                             pattern_type=f"peak_day_{day_names[dow]}",
1054:                             confidence=ratio,
1055:                             exemplar_relationship_ids=ids[:5],
1056:                         ))
1057: 
1058:             return patterns_out
1059: 
1060:         except Exception:
1061:             return []
1062: 
1063:     # â”€â”€ Staleness â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
1064: 
1065:     def staleness_batch(self, relationship_ids: List[int]) -> Dict[int, float]:
1066:         """Batch recency decay for a set of edges. Single DB query.
1067:         Returns {rel_id: score} where score in [0,1], 1.0=fresh."""
1068:         if not relationship_ids:
1069:             return {}
1070:         result: Dict[int, float] = {rid: 1.0 for rid in relationship_ids}
1071:         try:
1072:             placeholders = ",".join("?" for _ in relationship_ids)
1073:             with get_db_context() as conn:
1074:                 rows = conn.execute(
1075:                     f"SELECT id, last_confirmed_at FROM relationships "
1076:                     f"WHERE id IN ({placeholders})",
1077:                     tuple(relationship_ids),
1078:                 ).fetchall()
1079:             now = self.now()
1080:             time_constant = 30 * 86400.0
1081:             for row in rows:
1082:                 rid = row["id"]
1083:                 lc = row["last_confirmed_at"]
1084:                 if not lc:
1085:                     continue
1086:                 try:
1087:                     last_conf = datetime.fromisoformat(lc.replace("Z", "+00:00"))
1088:                     if last_conf.tzinfo is None:
1089:                         last_conf = last_conf.replace(tzinfo=timezone.utc)
1090:                     age_s = (now - last_conf).total_seconds()
1091:                     if age_s > 0:
1092:                         result[rid] = float(np.exp(-age_s / time_constant))
1093:                 except Exception:
1094:                     pass
1095:         except Exception:
1096:             pass
1097:         return result
1098: 
1099:     def staleness(self, relationship_id: int) -> float:
1100:         """Single-edge staleness. Delegates to batch for consistency."""
1101:         batch = self.staleness_batch([relationship_id])
1102:         return batch.get(relationship_id, 1.0)
1103: 
1104:     # â”€â”€ Humanize â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
1105: 
1106:     def humanize(self, delta_seconds: float) -> str:
1107:         if delta_seconds < 60:
1108:             return "just now"
1109:         if delta_seconds < 3600:
1110:             return f"{int(delta_seconds // 60)} minutes ago"
1111:         if delta_seconds < 86400:
1112:             return f"{int(delta_seconds // 3600)} hours ago"
1113:         if delta_seconds < 604800:
1114:             return f"{int(delta_seconds // 86400)} days ago"
1115:         if delta_seconds < 2592000:
1116:             return f"{int(delta_seconds // 604800)} weeks ago"
1117:         return f"{int(delta_seconds // 2592000)} months ago"
1118: 
1119: 
1120: _singleton: Optional[TemporalEngine] = None
1121: 
1122: 
1123: def get_temporal_engine() -> TemporalEngine:
1124:     global _singleton
1125:     if _singleton is None:
1126:         _singleton = TemporalEngine()
1127:     return _singleton
```

## app/db/session.py

Source: [app/db/session.py](/D:/Nura/Code/nura_living_memory_code/app/db/session.py)

```text
0001: import sqlite3
0002: from contextlib import contextmanager
0003: from pathlib import Path
0004: from config.settings import settings
0005: from app.db.models import MIGRATIONS, run_schema_upgrades
0006: 
0007: 
0008: def init_wal_mode(conn: sqlite3.Connection) -> None:
0009:     """Enable Write-Ahead Logging mode for better concurrency."""
0010:     conn.execute("PRAGMA journal_mode=WAL;")
0011:     conn.commit()
0012: 
0013: 
0014: def get_db_connection() -> sqlite3.Connection:
0015:     """Create and return a new database connection with WAL mode enabled."""
0016:     conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
0017:     conn.row_factory = sqlite3.Row
0018:     init_wal_mode(conn)
0019:     return conn
0020: 
0021: 
0022: def get_conn() -> sqlite3.Connection:
0023:     """
0024:     Get database connection.
0025: 
0026:     Note: Returns a fresh connection. Callers should use context managers
0027:     or ensure proper connection cleanup.
0028:     """
0029:     return get_db_connection()
0030: 
0031: 
0032: @contextmanager
0033: def get_db_context():
0034:     """
0035:     Context manager for database connections with automatic cleanup.
0036: 
0037:     Usage:
0038:         with get_db_context() as conn:
0039:             conn.execute("SELECT ...")
0040:             conn.commit()
0041:     """
0042:     conn = get_db_connection()
0043:     try:
0044:         yield conn
0045:     finally:
0046:         conn.close()
0047: 
0048: 
0049: def init_db(sqlite_path: str) -> None:
0050:     """Initialize database with schema and WAL mode."""
0051:     # Ensure DB file path directory exists
0052:     p = Path(sqlite_path)
0053:     if p.parent and str(p.parent) != ".":
0054:         p.parent.mkdir(parents=True, exist_ok=True)
0055: 
0056:     conn = sqlite3.connect(sqlite_path, check_same_thread=False)
0057:     conn.executescript(MIGRATIONS)
0058:     conn.commit()
0059: 
0060:     # Run schema upgrades for existing databases
0061:     run_schema_upgrades(conn)
0062: 
0063:     # Enable WAL mode
0064:     init_wal_mode(conn)
0065: 
0066:     conn.close()
0067: 
0068: 
0069: # =============================================================================
0070: # PROACTIVE COOLDOWN PERSISTENCE
0071: # =============================================================================
0072: 
0073: def get_proactive_cooldown(user_id: int) -> dict:
0074:     """
0075:     Load proactive cooldown state from database.
0076: 
0077:     Returns:
0078:         {"last_asked_at": ISO string or None, "asks_today": int}
0079:     """
0080:     from datetime import datetime
0081: 
0082:     with get_db_context() as conn:
0083:         row = conn.execute(
0084:             "SELECT last_asked_at, asks_today, asks_date FROM proactive_cooldown WHERE user_id = ?",
0085:             (user_id,)
0086:         ).fetchone()
0087: 
0088:         if not row:
0089:             return {"last_asked_at": None, "asks_today": 0}
0090: 
0091:         # Reset asks_today if it's a new day
0092:         today = datetime.now().strftime("%Y-%m-%d")
0093:         if row["asks_date"] != today:
0094:             return {"last_asked_at": row["last_asked_at"], "asks_today": 0}
0095: 
0096:         return {
0097:             "last_asked_at": row["last_asked_at"],
0098:             "asks_today": row["asks_today"] or 0
0099:         }
0100: 
0101: 
0102: def update_proactive_cooldown(user_id: int, last_asked_at: str) -> None:
0103:     """
0104:     Update proactive cooldown after a successful proactive ask.
0105: 
0106:     Increments asks_today and sets last_asked_at.
0107:     Resets asks_today if it's a new day.
0108:     """
0109:     from datetime import datetime
0110: 
0111:     today = datetime.now().strftime("%Y-%m-%d")
0112: 
0113:     with get_db_context() as conn:
0114:         # Check current state
0115:         row = conn.execute(
0116:             "SELECT asks_today, asks_date FROM proactive_cooldown WHERE user_id = ?",
0117:             (user_id,)
0118:         ).fetchone()
0119: 
0120:         if row:
0121:             # Reset if new day
0122:             if row["asks_date"] != today:
0123:                 asks_today = 1
0124:             else:
0125:                 asks_today = (row["asks_today"] or 0) + 1
0126: 
0127:             conn.execute(
0128:                 """UPDATE proactive_cooldown
0129:                    SET last_asked_at = ?, asks_today = ?, asks_date = ?, updated_at = datetime('now')
0130:                    WHERE user_id = ?""",
0131:                 (last_asked_at, asks_today, today, user_id)
0132:             )
0133:         else:
0134:             conn.execute(
0135:                 """INSERT INTO proactive_cooldown (user_id, last_asked_at, asks_today, asks_date)
0136:                    VALUES (?, ?, 1, ?)""",
0137:                 (user_id, last_asked_at, today)
0138:             )
0139: 
0140:         conn.commit()
```

## app/db/models.py

Source: [app/db/models.py](/D:/Nura/Code/nura_living_memory_code/app/db/models.py)

```text
0001: # SQLite migrations (kept simple for v1). Use Alembic later if needed.
0002: 
0003: MIGRATIONS = """
0004: -- =============================================================================
0005: -- EPISODES TABLE (Episodic Memory) - Day-to-day conversations, DOES decay
0006: -- =============================================================================
0007: CREATE TABLE IF NOT EXISTS memories (
0008:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0009:     user_id INTEGER NOT NULL,
0010:     content TEXT NOT NULL,
0011:     memory_type TEXT CHECK(memory_type IN ('episodic','semantic','summary')) NOT NULL,
0012:     importance REAL DEFAULT 0.5,
0013:     embedding BLOB,
0014:     created_at TEXT DEFAULT (datetime('now')),
0015:     last_accessed_at TEXT,
0016:     temporal_tags TEXT,
0017:     metadata TEXT
0018: );
0019: 
0020: CREATE INDEX IF NOT EXISTS idx_memories_user_time ON memories(user_id, created_at);
0021: CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(user_id, memory_type);
0022: CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(user_id, importance DESC);
0023: 
0024: -- =============================================================================
0025: -- FACTS TABLE (Semantic Memory) - Personal truths, ONE value per key, NO decay
0026: -- Key-value pairs that get UPDATED, not accumulated
0027: -- =============================================================================
0028: CREATE TABLE IF NOT EXISTS facts (
0029:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0030:     user_id INTEGER NOT NULL,
0031:     key TEXT NOT NULL,
0032:     value TEXT NOT NULL,
0033:     confidence REAL DEFAULT 0.9,
0034:     last_confirmed_at TEXT DEFAULT (datetime('now')),
0035:     first_learned_at TEXT DEFAULT (datetime('now')),
0036:     provenance_memory_id INTEGER,
0037:     embedding BLOB,
0038:     history TEXT,  -- JSON array of previous values for contradiction tracking
0039:     UNIQUE(user_id, key)
0040: );
0041: 
0042: CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
0043: CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(user_id, key);
0044: 
0045: -- =============================================================================
0046: -- MILESTONES TABLE (Life Events) - Timestamped events, NO decay, permanent
0047: -- Deaths, marriages, graduations, major life changes
0048: -- =============================================================================
0049: CREATE TABLE IF NOT EXISTS milestones (
0050:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0051:     user_id INTEGER NOT NULL,
0052:     event_type TEXT NOT NULL,
0053:     event_date TEXT,
0054:     description TEXT NOT NULL,
0055:     confidence REAL DEFAULT 0.9,
0056:     created_at TEXT DEFAULT (datetime('now')),
0057:     provenance_memory_id INTEGER,
0058:     embedding BLOB,
0059:     metadata TEXT
0060: );
0061: 
0062: CREATE INDEX IF NOT EXISTS idx_milestones_user ON milestones(user_id);
0063: CREATE INDEX IF NOT EXISTS idx_milestones_type ON milestones(user_id, event_type);
0064: CREATE INDEX IF NOT EXISTS idx_milestones_date ON milestones(user_id, event_date);
0065: 
0066: CREATE TABLE IF NOT EXISTS adaptation_profiles (
0067:     user_id INTEGER PRIMARY KEY,
0068:     warmth REAL DEFAULT 0.5,
0069:     formality REAL DEFAULT 0.5,
0070:     initiative REAL DEFAULT 0.5,
0071:     check_in_frequency REAL DEFAULT 0.5,
0072:     updated_at TEXT DEFAULT (datetime('now'))
0073: );
0074: 
0075: CREATE TABLE IF NOT EXISTS relationship_metrics (
0076:     user_id INTEGER NOT NULL,
0077:     week INTEGER NOT NULL,
0078:     relationship_depth REAL,
0079:     disclosure_avg REAL,
0080:     emotional_events INTEGER,
0081:     return_rate REAL,
0082:     created_at TEXT DEFAULT (datetime('now')),
0083:     PRIMARY KEY (user_id, week)
0084: );
0085: 
0086: CREATE TABLE IF NOT EXISTS temporal_patterns (
0087:     user_id INTEGER NOT NULL,
0088:     pattern_type TEXT NOT NULL,
0089:     confidence REAL DEFAULT 0.5,
0090:     detected_at TEXT DEFAULT (datetime('now')),
0091:     example_memory_ids TEXT
0092: );
0093: 
0094: CREATE INDEX IF NOT EXISTS idx_temporal_patterns_lookup ON temporal_patterns(user_id, pattern_type);
0095: 
0096: -- =============================================================================
0097: -- PROACTIVE COOLDOWN (Rate limiting for proactive outreach)
0098: -- Persists ask counts and last ask time across sessions
0099: -- =============================================================================
0100: CREATE TABLE IF NOT EXISTS proactive_cooldown (
0101:     user_id INTEGER PRIMARY KEY,
0102:     last_asked_at TEXT,
0103:     asks_today INTEGER DEFAULT 0,
0104:     asks_date TEXT,  -- Date for asks_today (resets daily)
0105:     updated_at TEXT DEFAULT (datetime('now'))
0106: );
0107: 
0108: -- =============================================================================
0109: -- ENTITIES TABLE (Graph nodes) - People, places, things mentioned by user
0110: -- =============================================================================
0111: CREATE TABLE IF NOT EXISTS entities (
0112:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0113:     user_id INTEGER NOT NULL,
0114:     name TEXT NOT NULL,           -- "Emily", "FitLife", "Jake"
0115:     entity_type TEXT,             -- "person", "place", "organization", "thing"
0116:     attributes TEXT,              -- JSON: {"status": "pregnant", "occupation": "trainer"}
0117:     first_mentioned_at TEXT DEFAULT (datetime('now')),
0118:     last_mentioned_at TEXT DEFAULT (datetime('now')),
0119:     mention_count INTEGER DEFAULT 1,
0120:     embedding BLOB,
0121:     UNIQUE(user_id, name)
0122: );
0123: 
0124: CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);
0125: CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(user_id, name);
0126: CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(user_id, entity_type);
0127: 
0128: -- =============================================================================
0129: -- RELATIONSHIPS TABLE (Graph edges) - Trace-primary schema (2026-04-25)
0130: -- source_text_hash is the dedup key; subject/predicate/object are optional
0131: -- derived fields. 5 trace columns carry NOT NULL defaults (same sentinel
0132: -- values already used in memory.py _write_edge_traces() at lines 867-869
0133: -- and in scripts/migrate_traces_primary.py v2 table at lines 151-156).
0134: -- Root cause: UNIQUE(user_id, subject, predicate, object) caused two
0135: -- distinct utterances producing the same triple to collide, losing trace
0136: -- data from the first. source_text_hash dedup preserves each utterance.
0137: -- =============================================================================
0138: CREATE TABLE IF NOT EXISTS relationships (
0139:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0140:     user_id INTEGER NOT NULL,
0141:     source_text TEXT NOT NULL DEFAULT '',
0142:     source_text_hash TEXT NOT NULL DEFAULT '',
0143:     created_at TEXT DEFAULT (datetime('now')),
0144: 
0145:     -- 5 traces (primary data)
0146:     edge_schematic_category TEXT NOT NULL DEFAULT 'uncategorized',
0147:     edge_temporal_context TEXT NOT NULL DEFAULT 'present',
0148:     edge_relational_type TEXT NOT NULL DEFAULT 'personal',
0149:     edge_episodic_significance TEXT NOT NULL DEFAULT 'routine',
0150:     edge_emotional_valence REAL NOT NULL DEFAULT 0.5,
0151:     edge_emotional_label TEXT,
0152:     edge_affiliation REAL,
0153: 
0154:     -- Derived triple
0155:     subject TEXT,
0156:     predicate TEXT,
0157:     object TEXT,
0158: 
0159:     -- Type resolution (grammar engine + type_resolver)
0160:     subject_type TEXT,
0161:     object_type TEXT,
0162:     subject_type_confidence REAL,
0163:     object_type_confidence REAL,
0164: 
0165:     -- Temporal (temporal engine)
0166:     source_timestamp TEXT,
0167:     temporal_expression TEXT,
0168:     resolved_event_date TEXT,
0169:     is_historical INTEGER DEFAULT 0,
0170:     is_current INTEGER DEFAULT 1,
0171:     superseded_at TEXT,
0172:     superseded_by INTEGER,
0173:     tombstoned_at TEXT,
0174:     tombstone_reason TEXT,
0175:     tombstone_op_id INTEGER,
0176: 
0177:     -- Embeddings (MiniLM)
0178:     edge_embedding BLOB,
0179:     predicate_embedding BLOB,
0180: 
0181:     -- Structure (memory engine)
0182:     cluster_id TEXT,
0183:     arc_id TEXT,
0184:     sequence_number INTEGER,
0185:     utterance_type_id INTEGER,
0186:     source_tag TEXT,
0187:     relational_entities TEXT,
0188: 
0189:     -- Grammar traces
0190:     edge_negated INTEGER DEFAULT 0,
0191:     edge_mood TEXT DEFAULT 'indicative',
0192:     episodic_fact TEXT,
0193:     emotional_target TEXT,
0194:     extraction_rule TEXT,
0195:     canonical_fields TEXT,
0196: 
0197:     -- Metadata
0198:     confidence REAL DEFAULT 0.9,
0199:     first_learned_at TEXT DEFAULT (datetime('now')),
0200:     last_confirmed_at TEXT DEFAULT (datetime('now')),
0201:     provenance_memory_id INTEGER,
0202:     UNIQUE(user_id, source_text_hash, created_at)
0203: );
0204: 
0205: CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id);
0206: CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(user_id, subject);
0207: CREATE INDEX IF NOT EXISTS idx_relationships_predicate ON relationships(user_id, predicate);
0208: CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(user_id, object);
0209: CREATE INDEX IF NOT EXISTS idx_rel_schema_cat ON relationships(user_id, edge_schematic_category);
0210: CREATE INDEX IF NOT EXISTS idx_rel_subject_schema ON relationships(user_id, subject, edge_schematic_category);
0211: CREATE INDEX IF NOT EXISTS idx_rel_source_hash ON relationships(user_id, source_text_hash);
0212: CREATE INDEX IF NOT EXISTS idx_rel_is_current ON relationships(user_id, is_current);
0213: CREATE INDEX IF NOT EXISTS idx_rel_seq ON relationships(user_id, sequence_number);
0214: CREATE INDEX IF NOT EXISTS idx_rel_arc ON relationships(user_id, arc_id);
0215: CREATE INDEX IF NOT EXISTS idx_rel_cluster ON relationships(user_id, cluster_id);
0216: CREATE INDEX IF NOT EXISTS idx_rel_resolved_date ON relationships(user_id, resolved_event_date);
0217: CREATE INDEX IF NOT EXISTS idx_rel_tombstoned ON relationships(user_id, tombstoned_at);
0218: 
0219: -- Full-text search on relationships (subject, predicate, object, source_text)
0220: CREATE VIRTUAL TABLE IF NOT EXISTS relationships_fts USING fts5(
0221:     subject, predicate, object, source_text,
0222:     content='relationships', content_rowid='id'
0223: );
0224: 
0225: -- =============================================================================
0226: -- MEMORY TRACES TABLE (Distributed Trace Convergence Memory)
0227: -- DEPRECATED: table retained for backward-compat reads; get_traces() removed 2026-04-24.
0228: -- Decomposed trace metadata per relationship triple
0229: -- =============================================================================
0230: CREATE TABLE IF NOT EXISTS memory_traces (
0231:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0232:     relationship_id INTEGER UNIQUE,
0233:     user_id INTEGER NOT NULL,
0234:     -- Emotional trace (3 dimensions)
0235:     valence REAL DEFAULT 0.5,
0236:     affiliation REAL DEFAULT 0.5,
0237:     emotional_intensity REAL,
0238:     -- Relational trace (3 dimensions)
0239:     relational_type TEXT,
0240:     relational_proximity REAL DEFAULT 0.5,
0241:     relational_valence REAL DEFAULT 0.5,
0242:     -- Episodic trace (2 dimensions)
0243:     episodic_significance TEXT DEFAULT 'routine',
0244:     episodic_narrative_position TEXT DEFAULT 'ongoing',
0245:     -- Temporal trace
0246:     temporal_context TEXT CHECK(temporal_context IN ('past','present','future','ongoing')),
0247:     -- Schematic trace
0248:     schema_category TEXT,
0249:     schema_confidence REAL,
0250:     -- Rehearsal
0251:     access_count INTEGER DEFAULT 0,
0252:     last_accessed_at TEXT,
0253:     created_at TEXT DEFAULT (datetime('now')),
0254:     FOREIGN KEY (relationship_id) REFERENCES relationships(id)
0255: );
0256: 
0257: CREATE INDEX IF NOT EXISTS idx_memory_traces_user ON memory_traces(user_id);
0258: 
0259: -- =============================================================================
0260: -- PREDICTED QUERIES TABLE (Distributed Trace Convergence Memory)
0261: -- Pre-computed query-answer fingerprints written at extraction time
0262: -- =============================================================================
0263: CREATE TABLE IF NOT EXISTS predicted_queries (
0264:     id INTEGER PRIMARY KEY AUTOINCREMENT,
0265:     relationship_id INTEGER,
0266:     user_id INTEGER NOT NULL,
0267:     predicted_question TEXT NOT NULL,
0268:     answer_text TEXT NOT NULL,
0269:     answer_subject TEXT,
0270:     question_embedding BLOB NOT NULL,
0271:     confidence REAL DEFAULT 0.9,
0272:     created_at TEXT DEFAULT (datetime('now')),
0273:     FOREIGN KEY (relationship_id) REFERENCES relationships(id)
0274: );
0275: 
0276: CREATE INDEX IF NOT EXISTS idx_predicted_queries_user ON predicted_queries(user_id);
0277: 
0278: -- =============================================================================
0279: -- ARCS TABLE (Story arcs for proactive engine)
0280: -- =============================================================================
0281: CREATE TABLE IF NOT EXISTS arcs (
0282:     id TEXT PRIMARY KEY,
0283:     user_id INTEGER NOT NULL,
0284:     topic TEXT NOT NULL,
0285:     topic_embedding BLOB,
0286:     start_edge_id INTEGER,
0287:     emotional_baseline REAL,
0288:     status TEXT DEFAULT 'open',
0289:     created_at TEXT DEFAULT (datetime('now')),
0290:     last_checked_at TEXT DEFAULT (datetime('now')),
0291:     resolved_at TEXT
0292: );
0293: 
0294: CREATE INDEX IF NOT EXISTS idx_arcs_user_status ON arcs(user_id, status);
0295: CREATE INDEX IF NOT EXISTS idx_arcs_last_checked ON arcs(user_id, last_checked_at);
0296: 
0297: -- =============================================================================
0298: -- TIMERS TABLE (Short-term reminders for proactive engine)
0299: -- =============================================================================
0300: CREATE TABLE IF NOT EXISTS timers (
0301:     id TEXT PRIMARY KEY,
0302:     user_id INTEGER NOT NULL,
0303:     fire_at TEXT NOT NULL,
0304:     callback_type TEXT,
0305:     payload TEXT,
0306:     fired INTEGER DEFAULT 0,
0307:     created_at TEXT DEFAULT (datetime('now'))
0308: );
0309: 
0310: CREATE INDEX IF NOT EXISTS idx_timers_user ON timers(user_id, fired);
0311: CREATE INDEX IF NOT EXISTS idx_timers_fire ON timers(fire_at, fired);
0312: 
0313: -- =============================================================================
0314: -- SCHEMA UPGRADES (for existing databases)
0315: -- These are safe to run multiple times
0316: -- =============================================================================
0317: 
0318: -- Add missing columns to facts table (for older databases)
0319: -- SQLite doesn't support IF NOT EXISTS for columns, so we use a workaround
0320: """
0321: 
0322: # Additional migration to add missing columns
0323: SCHEMA_UPGRADES = """
0324: -- Upgrade facts table if columns are missing
0325: -- These will fail silently if columns already exist
0326: """
0327: 
0328: def run_schema_upgrades(conn) -> None:
0329:     """Add missing columns to existing databases."""
0330:     # Check facts table columns
0331:     cursor = conn.execute("PRAGMA table_info(facts)")
0332:     existing_columns = {row[1] for row in cursor.fetchall()}
0333: 
0334:     upgrades = []
0335:     if "history" not in existing_columns:
0336:         upgrades.append("ALTER TABLE facts ADD COLUMN history TEXT")
0337:     if "first_learned_at" not in existing_columns:
0338:         upgrades.append("ALTER TABLE facts ADD COLUMN first_learned_at TEXT DEFAULT (datetime('now'))")
0339:     if "embedding" not in existing_columns:
0340:         upgrades.append("ALTER TABLE facts ADD COLUMN embedding BLOB")
0341: 
0342:     for sql in upgrades:
0343:         try:
0344:             conn.execute(sql)
0345:             print(f"[DB] Applied: {sql[:50]}...")
0346:         except Exception:
0347:             pass
0348: 
0349:     if upgrades:
0350:         conn.commit()
0351: 
0352:     # --- Grammar Engine: object_type column on relationships ---
0353:     rel_cursor = conn.execute("PRAGMA table_info(relationships)")
0354:     rel_columns = {row[1] for row in rel_cursor.fetchall()}
0355: 
0356:     if "object_type" not in rel_columns:
0357:         try:
0358:             conn.execute("ALTER TABLE relationships ADD COLUMN object_type TEXT DEFAULT 'unknown'")
0359:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_object_type ON relationships(user_id, object_type)")
0360:             conn.commit()
0361:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN object_type")
0362:         except Exception:
0363:             pass
0364: 
0365:     # --- 594 Equation System: utterance_type_id + canonical_fields on relationships ---
0366:     if "utterance_type_id" not in rel_columns:
0367:         try:
0368:             conn.execute("ALTER TABLE relationships ADD COLUMN utterance_type_id INTEGER")
0369:             conn.commit()
0370:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN utterance_type_id")
0371:         except Exception:
0372:             pass
0373: 
0374:     if "canonical_fields" not in rel_columns:
0375:         try:
0376:             conn.execute("ALTER TABLE relationships ADD COLUMN canonical_fields TEXT")
0377:             conn.commit()
0378:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN canonical_fields")
0379:         except Exception:
0380:             pass
0381: 
0382:     # --- Supersession columns on relationships (update handling) ---
0383:     if "is_current" not in rel_columns:
0384:         try:
0385:             conn.execute("ALTER TABLE relationships ADD COLUMN is_current INTEGER DEFAULT 1")
0386:             conn.commit()
0387:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN is_current")
0388:         except Exception:
0389:             pass
0390: 
0391:     if "superseded_at" not in rel_columns:
0392:         try:
0393:             conn.execute("ALTER TABLE relationships ADD COLUMN superseded_at TEXT")
0394:             conn.commit()
0395:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN superseded_at")
0396:         except Exception:
0397:             pass
0398: 
0399:     if "superseded_by" not in rel_columns:
0400:         try:
0401:             conn.execute("ALTER TABLE relationships ADD COLUMN superseded_by INTEGER")
0402:             conn.commit()
0403:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN superseded_by")
0404:         except Exception:
0405:             pass
0406: 
0407:     # --- situation_id: DEPRECATED (2026-04-23). Column was never populated.
0408:     # Existing databases retain the column for backward compat but it is
0409:     # no longer added to new databases. No code reads or writes it. ---
0410: 
0411:     # --- Source timestamp: when the source utterance occurred ---
0412:     # Used by temporal engine to resolve "when did X happen?" queries.
0413:     # For Raya: this is the wall-clock time the user said it.
0414:     # For LOCOMO/SDK: this is the session_date_time from the source data.
0415:     if "source_timestamp" not in rel_columns:
0416:         try:
0417:             conn.execute("ALTER TABLE relationships ADD COLUMN source_timestamp TEXT")
0418:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_ts ON relationships(user_id, source_timestamp)")
0419:             conn.commit()
0420:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN source_timestamp")
0421:         except Exception:
0422:             pass
0423: 
0424:     # --- Continuous dimension columns on memory_traces (legacy T5 SRL naming, kept for schema compat) ---
0425:     trace_cursor = conn.execute("PRAGMA table_info(memory_traces)")
0426:     trace_columns = {row[1] for row in trace_cursor.fetchall()}
0427: 
0428:     t5_upgrades = []
0429:     for col in ("valence", "affiliation"):
0430:         if col not in trace_columns:
0431:             t5_upgrades.append(f"ALTER TABLE memory_traces ADD COLUMN {col} REAL DEFAULT 0.5")
0432: 
0433:     # Migrate existing categorical emotional_valence â†’ continuous valence
0434:     migrate_valence = "emotional_valence" in trace_columns and "valence" not in trace_columns
0435: 
0436:     for sql in t5_upgrades:
0437:         try:
0438:             conn.execute(sql)
0439:             print(f"[DB] Applied: {sql[:60]}...")
0440:         except Exception:
0441:             pass
0442: 
0443:     if t5_upgrades:
0444:         conn.commit()
0445: 
0446:     # Backfill: convert old categorical values to continuous
0447:     if migrate_valence:
0448:         try:
0449:             conn.execute("""UPDATE memory_traces SET valence = CASE
0450:                 WHEN emotional_valence = 'positive' THEN 0.8
0451:                 WHEN emotional_valence = 'negative' THEN 0.2
0452:                 WHEN emotional_valence = 'mixed' THEN 0.5
0453:                 ELSE 0.5 END
0454:                 WHERE valence = 0.5 AND emotional_valence IS NOT NULL""")
0455:             conn.commit()
0456:             print("[DB] Applied: backfill emotional_valence â†’ valence")
0457:         except Exception:
0458:             pass
0459: 
0460:     # --- DTCM 5-trace upgrade: relational + episodic columns ---
0461:     new_trace_cols = {
0462:         "relational_type": "TEXT",
0463:         "relational_proximity": "REAL DEFAULT 0.5",
0464:         "relational_valence": "REAL DEFAULT 0.5",
0465:         "episodic_significance": "TEXT DEFAULT 'routine'",
0466:         "episodic_narrative_position": "TEXT DEFAULT 'ongoing'",
0467:     }
0468:     for col, col_type in new_trace_cols.items():
0469:         if col not in trace_columns:
0470:             try:
0471:                 conn.execute(f"ALTER TABLE memory_traces ADD COLUMN {col} {col_type}")
0472:                 conn.commit()
0473:                 print(f"[DB] Applied: ALTER TABLE memory_traces ADD COLUMN {col}")
0474:             except Exception:
0475:                 pass
0476: 
0477:     # --- Filter â†’ Complete migration (2026-04-12): trace columns
0478:     # and edge_embedding live DIRECTLY on the relationships row.
0479:     # memory_traces is kept for backward-compat read but no longer
0480:     # written. predicted_queries is no longer used â€” edge_embedding
0481:     # is the access path. ---
0482:     edge_trace_cols = {
0483:         "edge_embedding": "BLOB",
0484:         "edge_emotional_valence": "REAL DEFAULT 0.5",
0485:         "edge_emotional_label": "TEXT",
0486:         "edge_schematic_category": "TEXT",
0487:         "edge_episodic_significance": "TEXT",
0488:         "edge_relational_type": "TEXT",
0489:         "edge_temporal_context": "TEXT",
0490:         "edge_affiliation": "REAL DEFAULT 0.5",
0491:     }
0492:     for col, col_type in edge_trace_cols.items():
0493:         if col not in rel_columns:
0494:             try:
0495:                 conn.execute(f"ALTER TABLE relationships ADD COLUMN {col} {col_type}")
0496:                 conn.commit()
0497:                 print(f"[DB] Applied: ALTER TABLE relationships ADD COLUMN {col}")
0498:             except Exception:
0499:                 pass
0500: 
0501:     # --- Predicate embedding cache (2026-04-24): store the embedded
0502:     # predicate at write time so retrieval can skip embed_text() per edge.
0503:     if "predicate_embedding" not in rel_columns:
0504:         try:
0505:             conn.execute("ALTER TABLE relationships ADD COLUMN predicate_embedding BLOB")
0506:             conn.commit()
0507:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN predicate_embedding")
0508:         except Exception:
0509:             pass
0510: 
0511:     # --- Forget/Show migration (2026-04-14): tombstone columns +
0512:     # source provenance on relationships, plus live-query indexes. ---
0513:     forget_cols = {
0514:         "tombstoned_at": "TEXT",
0515:         "tombstone_reason": "TEXT",
0516:         "tombstone_op_id": "TEXT",
0517:         "source_text": "TEXT",
0518:         "source_tag": "TEXT",
0519:     }
0520:     for col, col_type in forget_cols.items():
0521:         if col not in rel_columns:
0522:             try:
0523:                 conn.execute(f"ALTER TABLE relationships ADD COLUMN {col} {col_type}")
0524:                 conn.commit()
0525:                 print(f"[DB] Applied: ALTER TABLE relationships ADD COLUMN {col}")
0526:             except Exception:
0527:                 pass
0528: 
0529:     forget_indexes = [
0530:         "CREATE INDEX IF NOT EXISTS idx_rel_live_subject "
0531:         "ON relationships(user_id, tombstoned_at, subject)",
0532:         "CREATE INDEX IF NOT EXISTS idx_rel_live_object "
0533:         "ON relationships(user_id, tombstoned_at, object)",
0534:         "CREATE INDEX IF NOT EXISTS idx_rel_live_source_ts "
0535:         "ON relationships(user_id, tombstoned_at, source_timestamp)",
0536:         "CREATE INDEX IF NOT EXISTS idx_rel_live_source_tag "
0537:         "ON relationships(user_id, tombstoned_at, source_tag)",
0538:     ]
0539:     for sql in forget_indexes:
0540:         try:
0541:             conn.execute(sql)
0542:             conn.commit()
0543:         except Exception:
0544:             pass
0545: 
0546:     # --- Set-op 9-axis pipeline (2026-04-14): cluster_id + arc_id.
0547:     # cluster_id is persisted at write time via TemporalEngine.cluster.
0548:     # arc_id stays NULL (placeholder for multi-session arc grouping).
0549:     if "cluster_id" not in rel_columns:
0550:         try:
0551:             conn.execute("ALTER TABLE relationships ADD COLUMN cluster_id TEXT")
0552:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_cluster ON relationships(user_id, cluster_id)")
0553:             conn.commit()
0554:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN cluster_id")
0555:         except Exception:
0556:             pass
0557: 
0558:     if "arc_id" not in rel_columns:
0559:         try:
0560:             conn.execute("ALTER TABLE relationships ADD COLUMN arc_id TEXT")
0561:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_arc ON relationships(user_id, arc_id)")
0562:             conn.commit()
0563:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN arc_id")
0564:         except Exception:
0565:             pass
0566: 
0567:     # --- Write-path foundation (2026-04-14): entity-type tagging on the
0568:     # triple. object_type already exists from an earlier migration; add
0569:     # subject_type on relationships and ensure entity_type is indexed on
0570:     # entities. Values drawn from the closed vocabulary resolved by
0571:     # app.engines.type_resolver: PERSON | ORG | LOCATION | TIME | EVENT |
0572:     # QUANTITY | WORK_OF_ART | PRODUCT | GENERIC.
0573:     if "subject_type" not in rel_columns:
0574:         try:
0575:             conn.execute("ALTER TABLE relationships ADD COLUMN subject_type TEXT")
0576:             conn.commit()
0577:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN subject_type")
0578:         except Exception:
0579:             pass
0580: 
0581:     # --- Resolved event date (2026-04-25): the actual date of the event
0582:     # described in the source text, resolved from DATE/TIME NER spans
0583:     # against source_timestamp. Distinct from source_timestamp (session
0584:     # wall-clock) â€” this is what the user said happened WHEN.
0585:     # Root cause: "when" queries found the right edge but returned no
0586:     # date because source_timestamp is session time, not event time.
0587:     if "resolved_event_date" not in rel_columns:
0588:         try:
0589:             conn.execute("ALTER TABLE relationships ADD COLUMN resolved_event_date TEXT")
0590:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_event_date ON relationships(user_id, resolved_event_date)")
0591:             conn.commit()
0592:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN resolved_event_date")
0593:         except Exception:
0594:             pass
0595: 
0596:     # --- Grammar engine decomposition columns (2026-04-25): persist
0597:     # temporal_expression and relational_entities from TraceDecomposition.
0598:     if "temporal_expression" not in rel_columns:
0599:         try:
0600:             conn.execute("ALTER TABLE relationships ADD COLUMN temporal_expression TEXT")
0601:             conn.commit()
0602:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN temporal_expression")
0603:         except Exception:
0604:             pass
0605: 
0606:     if "relational_entities" not in rel_columns:
0607:         try:
0608:             conn.execute("ALTER TABLE relationships ADD COLUMN relational_entities TEXT")
0609:             conn.commit()
0610:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN relational_entities")
0611:         except Exception:
0612:             pass
0613: 
0614:     # --- Type confidence columns (2026-04-24): signals whether NER and
0615:     # WordNet agreed on the entity type. HIGH = agreement or single-source,
0616:     # LOW = disagreement (type is uncertain). Used by retrieval as a soft
0617:     # tiebreaker â€” low-confidence types don't gate, only nudge.
0618:     if "subject_type_confidence" not in rel_columns:
0619:         try:
0620:             conn.execute("ALTER TABLE relationships ADD COLUMN subject_type_confidence TEXT DEFAULT 'high'")
0621:             conn.commit()
0622:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN subject_type_confidence")
0623:         except Exception:
0624:             pass
0625:     if "object_type_confidence" not in rel_columns:
0626:         try:
0627:             conn.execute("ALTER TABLE relationships ADD COLUMN object_type_confidence TEXT DEFAULT 'high'")
0628:             conn.commit()
0629:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN object_type_confidence")
0630:         except Exception:
0631:             pass
0632: 
0633:     # Ensure object_type index exists with the canonical name expected by
0634:     # the coherence gate (older migration used a DEFAULT 'unknown' and a
0635:     # different index; we add the canonical one idempotently).
0636:     for sql in (
0637:         "CREATE INDEX IF NOT EXISTS idx_rel_object_type ON relationships(user_id, object_type)",
0638:         "CREATE INDEX IF NOT EXISTS idx_rel_subject_type ON relationships(user_id, subject_type)",
0639:     ):
0640:         try:
0641:             conn.execute(sql)
0642:             conn.commit()
0643:         except Exception:
0644:             pass
0645: 
0646:     # --- sequence_number column (2026-04-25): narrative-time axis.
0647:     # Root cause: memory.py lines 747-755 writes sequence_number via
0648:     # UPDATE but the column was never formally added via ALTER TABLE or
0649:     # included in the base CREATE TABLE. Retrieval.py references it in
0650:     # ORDER BY clauses (lines 774, 956, 1183, 1309, 1351, 1396, 1439).
0651:     # On databases created from MIGRATIONS alone, the column does not
0652:     # exist and those ORDER BY clauses silently get NULLs.
0653:     if "sequence_number" not in rel_columns:
0654:         try:
0655:             conn.execute("ALTER TABLE relationships ADD COLUMN sequence_number INTEGER")
0656:             conn.commit()
0657:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN sequence_number")
0658:         except Exception:
0659:             pass
0660: 
0661:     # --- Negation column (2026-04-27): grammar engine sets negated=True
0662:     # on TraceDecomposition but _write_edge_traces never persisted it.
0663:     if "edge_negated" not in rel_columns:
0664:         try:
0665:             conn.execute("ALTER TABLE relationships ADD COLUMN edge_negated INTEGER DEFAULT 0")
0666:             conn.commit()
0667:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN edge_negated")
0668:         except Exception:
0669:             pass
0670: 
0671:     # --- Mood column (2026-04-28): stores "indicative", "interrogative",
0672:     # "imperative", "conditional" from grammar engine TraceDecomposition.
0673:     # Retrieval filters out non-indicative edges so imposed facts from
0674:     # questions/commands don't leak as answers.
0675:     if "edge_mood" not in rel_columns:
0676:         try:
0677:             conn.execute("ALTER TABLE relationships ADD COLUMN edge_mood TEXT DEFAULT 'indicative'")
0678:             conn.commit()
0679:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN edge_mood")
0680:         except Exception:
0681:             pass
0682: 
0683:     # --- is_historical column (2026-04-29): grammar engine sets
0684:     # is_historical=True for past-tense facts (e.g. "I used to work at
0685:     # Google"). Retrieval can distinguish current vs historical facts.
0686:     if "is_historical" not in rel_columns:
0687:         try:
0688:             conn.execute("ALTER TABLE relationships ADD COLUMN is_historical INTEGER DEFAULT 0")
0689:             conn.commit()
0690:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN is_historical")
0691:         except Exception:
0692:             pass
0693: 
0694:     # --- episodic_fact column (2026-04-29): normalized sentence-level
0695:     # fact from TraceDecomposition. The canonical natural-language form
0696:     # of what was stored, independent of triple decomposition.
0697:     if "episodic_fact" not in rel_columns:
0698:         try:
0699:             conn.execute("ALTER TABLE relationships ADD COLUMN episodic_fact TEXT")
0700:             conn.commit()
0701:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN episodic_fact")
0702:         except Exception:
0703:             pass
0704: 
0705:     # --- emotional_target column (2026-04-29): what the emotion is
0706:     # about (e.g. "the job interview" when user says "I'm nervous about
0707:     # the job interview"). From TraceDecomposition.emotional_target.
0708:     if "emotional_target" not in rel_columns:
0709:         try:
0710:             conn.execute("ALTER TABLE relationships ADD COLUMN emotional_target TEXT")
0711:             conn.commit()
0712:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN emotional_target")
0713:         except Exception:
0714:             pass
0715: 
0716:     # --- extraction_rule column (2026-04-29): provenance trace showing
0717:     # which grammar rule produced this edge (trace|imposed|
0718:     # free_indirect_speech etc.). From TraceDecomposition.extraction_rule.
0719:     if "extraction_rule" not in rel_columns:
0720:         try:
0721:             conn.execute("ALTER TABLE relationships ADD COLUMN extraction_rule TEXT")
0722:             conn.commit()
0723:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN extraction_rule")
0724:         except Exception:
0725:             pass
0726: 
0727:     # --- source_text_hash column (Phase 6, 2026-04-25): SHA-256 of
0728:     # source_text. Root cause: the full trace-primary migration
0729:     # (scripts/migrate_traces_primary.py) replaces the UNIQUE constraint,
0730:     # but memory.py store() needs to start writing hashes before the full
0731:     # migration runs. This column-add is the incremental bridge so
0732:     # store() can populate the hash on each INSERT.
0733:     if "source_text_hash" not in rel_columns:
0734:         try:
0735:             conn.execute("ALTER TABLE relationships ADD COLUMN source_text_hash TEXT")
0736:             conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_hash ON relationships(user_id, source_text_hash)")
0737:             conn.commit()
0738:             print("[DB] Applied: ALTER TABLE relationships ADD COLUMN source_text_hash")
0739:         except Exception:
0740:             pass
0741: 
0742:     # entities.entity_type already exists in the base CREATE TABLE; ensure
0743:     # its index is named idx_entity_type for the coherence gate.
0744:     try:
0745:         conn.execute(
0746:             "CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(user_id, entity_type)"
0747:         )
0748:         conn.commit()
0749:     except Exception:
0750:         pass
0751: 
0752:     # --- FTS5 full-text search on relationships (BM25 hybrid retrieval) ---
0753:     # Content-sync FTS5 virtual table: external content points to the
0754:     # relationships table. Queries use BM25 ranking over the concatenated
0755:     # subject + predicate + object + source_text. Zero new dependencies â€”
0756:     # FTS5 is built into SQLite 3.9+ (Python 3.10 ships 3.37+).
0757:     #
0758:     # Migration (2026-04-24): added source_text as 4th FTS column so BM25
0759:     # can match the raw user utterance. Existing 3-column FTS tables are
0760:     # detected and rebuilt automatically.
0761:     _fts_needs_rebuild = False
0762:     try:
0763:         # Detect whether the FTS table exists and has the expected columns.
0764:         # FTS5 content-sync tables expose columns via PRAGMA; if source_text
0765:         # is missing we must drop + recreate.
0766:         _fts_cols = {
0767:             row[1]
0768:             for row in conn.execute("PRAGMA table_info(relationships_fts)").fetchall()
0769:         }
0770:         if _fts_cols and "source_text" not in _fts_cols:
0771:             conn.execute("DROP TABLE IF EXISTS relationships_fts")
0772:             conn.commit()
0773:             _fts_needs_rebuild = True
0774:             print("[DB] Dropped old 3-column relationships_fts for source_text migration")
0775:     except Exception:
0776:         pass
0777: 
0778:     try:
0779:         conn.execute("""
0780:             CREATE VIRTUAL TABLE IF NOT EXISTS relationships_fts
0781:             USING fts5(
0782:                 subject, predicate, object, source_text,
0783:                 content='relationships',
0784:                 content_rowid='id'
0785:             )
0786:         """)
0787:         conn.commit()
0788:         print("[DB] Applied: CREATE VIRTUAL TABLE relationships_fts (FTS5, 4-col)")
0789:     except Exception:
0790:         pass
0791: 
0792:     # Populate FTS5 index for any existing rows not yet indexed.
0793:     # This is idempotent: INSERT OR IGNORE semantics via FTS5's
0794:     # content-sync mechanism. For content-sync tables, we rebuild
0795:     # if the table is empty (fresh migration) or after a schema rebuild.
0796:     try:
0797:         fts_count = conn.execute(
0798:             "SELECT COUNT(*) FROM relationships_fts"
0799:         ).fetchone()[0]
0800:         if fts_count == 0 or _fts_needs_rebuild:
0801:             # Clear any stale rows from a partial state before full backfill.
0802:             if _fts_needs_rebuild and fts_count > 0:
0803:                 conn.execute(
0804:                     "INSERT INTO relationships_fts(relationships_fts) VALUES('delete-all')"
0805:                 )
0806:             conn.execute("""
0807:                 INSERT INTO relationships_fts(rowid, subject, predicate, object, source_text)
0808:                 SELECT id, subject, REPLACE(predicate, '_', ' '), object,
0809:                        COALESCE(source_text, '')
0810:                 FROM relationships
0811:                 WHERE tombstoned_at IS NULL
0812:             """)
0813:             conn.commit()
0814:             print("[DB] Applied: backfill relationships_fts from existing rows")
0815:     except Exception:
0816:         pass
```
