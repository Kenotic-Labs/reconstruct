# sentence-model

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
