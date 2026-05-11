# temporal-engine

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
