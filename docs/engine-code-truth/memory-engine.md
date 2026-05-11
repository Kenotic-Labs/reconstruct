# memory-engine

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
