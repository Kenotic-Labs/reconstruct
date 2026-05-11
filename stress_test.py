"""
NURA Engine Stress Test — Hard Questions from LoCoMo Conversation 0

This test ingests REAL messy dialogue (pronouns, fragments, backchannels,
long rambling turns) from the Caroline/Melanie conversation, then fires
15 questions designed to break the engine:

  1. Pronoun resolution ("she" → who?)
  2. Object-position entities ("Who painted the sunrise?")
  3. Relative temporal references ("The sunday before 25 May 2023")
  4. Multi-hop across sessions ("Where does Caroline's grandma live?")
  5. Cat 5 adversarial (wrong speaker attribution)
  6. Facts buried in long conversational turns
  7. Implied answers (not stated directly)
  8. No NER entity questions ("What is self-care?")
  9. Count/aggregation questions
 10. Negation / absence questions

Uses the official LoCoMo eval_question_answering() for scoring.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT))

_LOCOMO_PKG = _PROJECT / "locomo_bench" / "locomo"
sys.path.insert(0, str(_LOCOMO_PKG))

from task_eval.evaluation import eval_question_answering  # noqa: E402
from sdk import KenoticV1                                 # noqa: E402
from app.engines.retrieval import Answer, Situation, StructuralRefusal  # noqa: E402

DATA_PATH = _PROJECT / "locomo_bench" / "locomo" / "data" / "locomo10.json"


# ---------------------------------------------------------------------------
# Helpers (lifted from run_locomo.py)
# ---------------------------------------------------------------------------

def _parse_locomo_timestamp(raw: str) -> str | None:
    import re
    from datetime import datetime
    if not raw:
        return None
    cleaned = raw.strip().replace(",", "")
    m = re.match(
        r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        cleaned, re.IGNORECASE,
    )
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    ampm = m.group(3).lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    day = int(m.group(4))
    month_name = m.group(5)
    year = int(m.group(6))
    try:
        dt = datetime.strptime(f"{year} {month_name} {day}", "%Y %B %d")
        dt = dt.replace(hour=hour, minute=minute)
        return dt.isoformat()
    except ValueError:
        return None


def _extract_answer_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, Answer):
        return result.text or ""
    if isinstance(result, Situation):
        return result.narrative or ""
    if isinstance(result, StructuralRefusal):
        return "This information is not mentioned in the conversation."
    return str(result) if result else ""


# ---------------------------------------------------------------------------
# Hand-picked dialogue turns — messy, real, covering key evidence spans
# ---------------------------------------------------------------------------
# We pick 30 turns from sessions 1-5 of conversation 0 that contain the
# facts our hard questions target. These are the RAW dialogue strings —
# pronouns, fragments, backchannels included.

SELECTED_TURNS = [
    # Session 1 (8 May 2023) — D1:1 through D1:18
    ("1:56 pm on 8 May, 2023", [
        ("Caroline", "Hey Mel! Good to see you! How have you been?", True),
        ("Melanie", "Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?", False),
        ("Caroline", "I went to a LGBTQ support group yesterday and it was so powerful.", True),
        ("Melanie", "Wow, that's cool, Caroline! What happened that was so awesome? Did you hear any inspiring stories?", False),
        ("Caroline", "The transgender stories were so inspiring! I was so happy and thankful for all the support.", True),
        ("Melanie", "Wow, love that painting! So cool you found such a helpful group. What's it done for you?", False),
        ("Caroline", "The support group has made me feel accepted and given me courage to embrace myself.", True),
        ("Melanie", "That's really cool. You've got guts. What now?", False),
        ("Caroline", "Gonna continue my edu and check out career options, which is pretty exciting!", True),
        ("Caroline", "I'm keen on counseling or working in mental health - I'd love to support those with similar issues.", True),
        ("Melanie", "You'd be a great counselor! Your empathy and understanding will really help the people you work with. By the way, take a look at this.", False),
        ("Caroline", "Thanks, Melanie! That's really sweet. Is this your own painting?", True),
        ("Melanie", "Yeah, I painted that lake sunrise last year! It's special to me.", False),
        ("Caroline", "Wow, Melanie! The colors really blend nicely. Painting looks like a great outlet for expressing yourself.", True),
        ("Melanie", "Thanks, Caroline! Painting's a fun way to express my feelings and get creative. It's a great way to relax after a long day.", False),
        ("Melanie", "Yep, Caroline. Taking care of ourselves is vital. I'm off to go swimming with the kids. Talk to you soon!", False),
    ]),
    # Session 2 (25 May 2023) — D2:1 through D2:17
    ("1:14 pm on 25 May, 2023", [
        ("Melanie", "Hey Caroline, since we last chatted, I've had a lot of things happening to me. I ran a charity race for mental health last Saturday \u2013 it was really rewarding. Really made me think about taking care of our minds.", False),
        ("Caroline", "That charity race sounds great, Mel! Making a difference & raising awareness for mental health is super rewarding - I'm really proud of you for taking part!", True),
        ("Melanie", "Thanks, Caroline! The event was really thought-provoking. I'm starting to realize that self-care is really important. It's a journey for me, but when I look after myself, I'm able to better look after my family.", False),
        ("Melanie", "Yeah, it's tough. So I'm carving out some me-time each day - running, reading, or playing my violin - which refreshes me and helps me stay present for my fam!", False),
        ("Caroline", "Researching adoption agencies \u2014 it's been a dream to have a family and give a loving home to kids who need it.", True),
        ("Caroline", "I chose them 'cause they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me.", True),
        ("Caroline", "I'm thrilled to make a family for kids who need one. It'll be tough as a single parent, but I'm up for the challenge!", True),
    ]),
    # Session 3 (9 June 2023) — D3:1 through D3:16
    ("7:55 pm on 9 June, 2023", [
        ("Caroline", "Hey Melanie! How's it going? I wanted to tell you about my school event last week. It was awesome! I talked about my transgender journey and encouraged students to get involved in the LGBTQ community. It was great to see their reactions. It made me reflect on how far I've come since I started transitioning three years ago.", True),
        ("Caroline", "Thanks Mel! Your kind words mean a lot. Sharing our experiences isn't always easy, but I feel it's important to help promote understanding and acceptance. I've been blessed with loads of love and support throughout this journey, and I want to pass it on to others. By sharing our stories, we can build a strong, supportive community of hope.", True),
        ("Caroline", "Thanks, Mel! My friends, family and mentors are my rocks \u2013 they motivate me and give me the strength to push on. Here's a pic from when we met up last week!", True),
        ("Caroline", "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?", True),
        ("Melanie", "I'm lucky to have my husband and kids; they keep me motivated.", False),
        ("Melanie", "5 years already! Time flies- feels like just yesterday I put this dress on! Thanks, Caroline!", False),
    ]),
    # Session 4 (27 June 2023) — D4:1 through D4:18
    ("10:37 am on 27 June, 2023", [
        ("Caroline", "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family.", True),
        ("Caroline", "Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago. The pattern and colors are awesome-- it reminds me of art and self-expression.", True),
        ("Melanie", "That sounds great, Caroline! It's awesome having stuff around that make us think of good connections and times. Actually, I just took my fam camping in the mountains last week - it was a really nice time together!", False),
        ("Melanie", "It was an awesome time, Caroline! We explored nature, roasted marshmallows around the campfire and even went on a hike. The view from the top was amazing! The 2 younger kids love nature. It was so special having these moments together as a family - I'll never forget it!", False),
        ("Caroline", "I'm still figuring out the details, but I'm thinking of working with trans people, helping them accept themselves and supporting their mental health. Last Friday, I went to an LGBTQ+ counseling workshop and it was really enlightening. They talked about different therapeutic methods and how to best work with trans people. Seeing how passionate these pros were about making a safe space for people like me was amazing.", True),
        ("Caroline", "Thanks, Melanie. It really mattered. My own journey and the support I got made a huge difference. Now I want to help people go through it too. I saw how counseling and support groups improved my life, so I started caring more about mental health and understanding myself. Now I'm passionate about creating a safe, inviting place for people to grow.", True),
    ]),
]


# ---------------------------------------------------------------------------
# 15 HARD questions — designed to break the engine
# ---------------------------------------------------------------------------

HARD_QUESTIONS = [
    # --- 1. Pronoun resolution: "she" painted it, but who is "she"? ---
    {
        "id": 1,
        "question": "Who painted the sunrise?",
        "answer": "Melanie",
        "category": 4,
        "difficulty": "object-position entity + pronoun chain",
        "evidence": ["D1:14"],
        "notes": "Melanie says 'I painted that lake sunrise last year'. The entity is in the OBJECT of the sentence. Engine must resolve 'I' -> Melanie.",
    },
    # --- 2. Relative temporal reference ---
    {
        "id": 2,
        "question": "When did Melanie run a charity race?",
        "answer": "The sunday before 25 May 2023",
        "category": 2,
        "difficulty": "relative date from session timestamp",
        "evidence": ["D2:1"],
        "notes": "Melanie says 'last Saturday' in session dated 25 May 2023. Gold answer is 'The sunday before 25 May 2023' (LoCoMo says sunday but means saturday -- this is LoCoMo's own gold).",
    },
    # --- 3. Multi-hop: connecting grandma -> country ---
    {
        "id": 3,
        "question": "Where did Caroline move from 4 years ago?",
        "answer": "Sweden",
        "category": 1,
        "difficulty": "multi-hop: friends-for-4-years + home-country = Sweden",
        "evidence": ["D3:13", "D4:3"],
        "notes": "D3:13 says 'known these friends for 4 years, since I moved from my home country'. D4:3 reveals home country is Sweden. Two separate sessions.",
    },
    # --- 4. Cat 5 adversarial: wrong speaker attribution ---
    {
        "id": 4,
        "question": "What are Melanie's plans for the summer with respect to adoption?",
        "answer": "not mentioned",
        "category": 5,
        "difficulty": "adversarial - adoption is CAROLINE's plan, not Melanie's",
        "evidence": ["D2:8"],
        "notes": "Caroline researches adoption. Melanie never mentions adopting. Engine must refuse, not hallucinate.",
    },
    # --- 5. Cat 5 adversarial: attribute swap ---
    {
        "id": 5,
        "question": "What country is Melanie's grandma from?",
        "answer": "not mentioned",
        "category": 5,
        "difficulty": "adversarial - Sweden is CAROLINE's grandma, not Melanie's",
        "evidence": ["D4:3"],
        "notes": "Caroline's grandma is from Sweden. Melanie's grandma is never mentioned.",
    },
    # --- 6. Fact buried in long conversational turn ---
    {
        "id": 6,
        "question": "How long ago was Caroline's 18th birthday?",
        "answer": "10 years ago",
        "category": 2,
        "difficulty": "fact buried mid-sentence in a long turn",
        "evidence": ["D4:5"],
        "notes": "D4:5: 'A friend made it for my 18th birthday ten years ago.' Buried in a sentence about a bowl.",
    },
    # --- 7. No NER entity: abstract concept question ---
    {
        "id": 7,
        "question": "What did Melanie realize after the charity race?",
        "answer": "self-care is important",
        "category": 4,
        "difficulty": "answer is an abstract concept, no named entity",
        "evidence": ["D2:3"],
        "notes": "D2:3: 'I'm starting to realize that self-care is really important.' No NER entity in the answer.",
    },
    # --- 8. Implied answer: career inference ---
    {
        "id": 8,
        "question": "What fields would Caroline be likely to pursue in her education?",
        "answer": "Psychology, counseling certification",
        "category": 3,
        "difficulty": "implied from career interest, not directly stated",
        "evidence": ["D1:9", "D1:11"],
        "notes": "Caroline says 'continue my edu' and 'keen on counseling or working in mental health'. The answer 'psychology, counseling certification' is inferred.",
    },
    # --- 9. Cat 5 adversarial: swap who gave what ---
    {
        "id": 9,
        "question": "What was grandpa's gift to Caroline?",
        "answer": "not mentioned",
        "category": 5,
        "difficulty": "adversarial - it was GRANDMA's gift, not grandpa's",
        "evidence": ["D4:3"],
        "notes": "D4:3 says 'a gift from my grandma'. No grandpa is ever mentioned.",
    },
    # --- 10. Cross-session temporal: how long married ---
    {
        "id": 10,
        "question": "How long have Mel and her husband been married?",
        "answer": "Mel and her husband have been married for 5 years.",
        "category": 4,
        "difficulty": "answer requires linking 'husband' from D3:14 to '5 years' from D3:16",
        "evidence": ["D3:16"],
        "notes": "D3:16: '5 years already! Time flies'. Must connect this to marriage context.",
    },
    # --- 11. Cat 5 adversarial: Caroline did NOT go camping ---
    {
        "id": 11,
        "question": "What did Caroline and her family do while camping?",
        "answer": "not mentioned",
        "category": 5,
        "difficulty": "adversarial - MELANIE went camping, not Caroline",
        "evidence": ["D4:8"],
        "notes": "Melanie's family went camping. Caroline never mentions camping with her family.",
    },
    # --- 12. Aggregation across sessions ---
    {
        "id": 12,
        "question": "What activities does Melanie partake in?",
        "answer": "pottery, camping, painting, swimming",
        "category": 1,
        "difficulty": "multi-hop aggregation across 4 evidence spans",
        "evidence": ["D5:4", "D9:1", "D1:12", "D1:18"],
        "notes": "Facts spread across sessions 1, 5, and 9. Engine must aggregate.",
    },
    # --- 13. Cat 5 adversarial: wrong person's counseling ---
    {
        "id": 13,
        "question": "What kind of counseling and mental health services is Melanie interested in pursuing?",
        "answer": "not mentioned",
        "category": 5,
        "difficulty": "adversarial - counseling is CAROLINE's interest, not Melanie's",
        "evidence": ["D4:13"],
        "notes": "Caroline discusses counseling for trans people. Melanie never expresses interest in pursuing counseling.",
    },
    # --- 14. Precise detail extraction from messy turn ---
    {
        "id": 14,
        "question": "What motivated Caroline to pursue counseling?",
        "answer": "her own journey and the support she received, and how counseling improved her life",
        "category": 4,
        "difficulty": "long answer embedded in a rambling conversational turn",
        "evidence": ["D4:15"],
        "notes": "D4:15 is a long turn. The answer is spread across multiple clauses.",
    },
    # --- 15. Identity question requiring inference ---
    {
        "id": 15,
        "question": "What is Caroline's identity?",
        "answer": "Transgender woman",
        "category": 1,
        "difficulty": "never stated as 'I am a transgender woman' — must infer from context",
        "evidence": ["D1:5"],
        "notes": "D1:5 mentions 'The transgender stories were so inspiring' plus the context of the LGBTQ support group and transition journey. Never a direct 'I am transgender' statement.",
    },
]


def main():
    print("=" * 70)
    print("NURA ENGINE STRESS TEST — 15 Hard Questions")
    print("=" * 70)

    # Load the real conversation data for session keys we DON'T have
    # (we'll need sessions 5+ for the multi-hop aggregation Q12)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    conv0 = all_data[0]["conversation"]

    # Create temp DB
    tmp = tempfile.NamedTemporaryFile(
        suffix=".db", prefix="stress_test_", delete=False,
    )
    db_path = tmp.name
    tmp.close()

    print(f"\nDB: {db_path}")
    print(f"\nPhase 1: Ingesting dialogue turns...")

    t0 = time.time()
    turn_count = 0

    # Ingest our hand-picked turns from sessions 1-4
    for raw_ts, turns in SELECTED_TURNS:
        iso_ts = _parse_locomo_timestamp(raw_ts)
        for speaker, text, is_user in turns:
            KenoticV1(
                text,
                speaker=speaker,
                speaker_is_user=is_user,
                source_timestamp=iso_ts,
                db_path=db_path,
            )
            turn_count += 1

    # Also ingest sessions 5-9 from the real data to cover evidence spans
    # for Q12 (activities aggregation) which needs D5:4, D9:1 etc.
    for session_num in range(5, 20):
        session_key = f"session_{session_num}"
        date_key = f"{session_key}_date_time"
        if session_key not in conv0:
            continue
        raw_ts = conv0.get(date_key, "")
        iso_ts = _parse_locomo_timestamp(raw_ts)
        turns = conv0[session_key]
        if not isinstance(turns, list):
            continue
        for turn in turns:
            text = turn.get("text", "")
            if not text:
                continue
            speaker = turn.get("speaker", "unknown")
            speaker_a = conv0.get("speaker_a", "Caroline")
            KenoticV1(
                text,
                speaker=speaker,
                speaker_is_user=(speaker == speaker_a),
                source_timestamp=iso_ts,
                db_path=db_path,
            )
            turn_count += 1

    ingest_time = time.time() - t0
    print(f"  Ingested {turn_count} turns in {ingest_time:.1f}s")

    # Phase 2: Ask the hard questions
    print(f"\nPhase 2: Querying {len(HARD_QUESTIONS)} hard questions...")
    print("-" * 70)

    t1 = time.time()
    scored_qas = []

    for q in HARD_QUESTIONS:
        result = KenoticV1(q["question"], db_path=db_path)
        prediction = _extract_answer_text(result.result)

        item = {
            "question": q["question"],
            "category": q["category"],
            "evidence": q["evidence"],
            "prediction": prediction,
            "answer": q["answer"],
        }
        scored_qas.append(item)

    query_time = time.time() - t1

    # Score using official LoCoMo eval
    f1_scores, _, recall_scores = eval_question_answering(scored_qas, eval_key="prediction")

    # Attach scores
    for i, item in enumerate(scored_qas):
        item["f1"] = float(f1_scores[i])
        item["id"] = HARD_QUESTIONS[i]["id"]
        item["difficulty"] = HARD_QUESTIONS[i]["difficulty"]
        item["notes"] = HARD_QUESTIONS[i]["notes"]

    # Phase 3: Report
    print(f"\n{'=' * 70}")
    print("STRESS TEST RESULTS")
    print(f"{'=' * 70}")

    pass_count = 0
    fail_count = 0
    cat_scores = defaultdict(list)

    for item in scored_qas:
        cat_scores[item["category"]].append(item["f1"])
        passed = item["f1"] >= 0.5  # generous threshold
        if passed:
            pass_count += 1
        else:
            fail_count += 1

        status = "PASS" if passed else "FAIL"
        marker = "  " if passed else ">>"

        print(f"\n{marker} Q{item['id']:2d} [{status}] F1={item['f1']:.4f}")
        print(f"   Category: {item['category']} | Difficulty: {item['difficulty']}")
        print(f"   Question:   {item['question']}")
        print(f"   Expected:   {item['answer']}")
        print(f"   Got:        {item['prediction'][:200]}")
        if not passed:
            print(f"   Diagnosis:  {item['notes']}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total:  {len(scored_qas)}")
    print(f"  Pass:   {pass_count}  ({100*pass_count/len(scored_qas):.0f}%)")
    print(f"  Fail:   {fail_count}  ({100*fail_count/len(scored_qas):.0f}%)")
    print()

    overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"  Overall mean F1: {overall_f1:.4f}")
    print()

    CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "narrative", 5: "adversarial"}
    for cat in sorted(cat_scores):
        scores = cat_scores[cat]
        mean = sum(scores) / len(scores) if scores else 0.0
        label = CATEGORY_NAMES.get(cat, f"cat{cat}")
        print(f"  Cat {cat} ({label:>12s}): F1 = {mean:.4f}  ({len(scores)} questions)")

    print(f"\n  Query time: {query_time:.1f}s")
    print(f"  Total time: {time.time() - t0:.1f}s")

    # Failure pattern analysis
    failures = [item for item in scored_qas if item["f1"] < 0.5]
    if failures:
        print(f"\n{'=' * 70}")
        print("FAILURE PATTERN ANALYSIS")
        print(f"{'=' * 70}")

        patterns = defaultdict(list)
        for f in failures:
            # Classify failure pattern
            if f["category"] == 5:
                if "not mentioned" not in f["prediction"].lower() and "no information" not in f["prediction"].lower():
                    patterns["adversarial: failed to refuse"].append(f)
                else:
                    patterns["adversarial: other"].append(f)
            elif not f["prediction"].strip():
                patterns["empty response"].append(f)
            elif f["category"] == 1:
                patterns["multi-hop: incomplete aggregation"].append(f)
            elif f["category"] == 2:
                patterns["temporal: wrong/missing date"].append(f)
            elif f["category"] == 3:
                patterns["open-domain: inference failure"].append(f)
            elif f["category"] == 4:
                if len(f["prediction"].split()) > 3 * len(f["answer"].split()):
                    patterns["narrative: answer too verbose"].append(f)
                else:
                    patterns["narrative: wrong content"].append(f)
            else:
                patterns["other"].append(f)

        for pattern, items in sorted(patterns.items()):
            print(f"\n  [{pattern}] — {len(items)} failure(s)")
            for item in items:
                print(f"    Q{item['id']}: {item['question'][:60]}")

    # Cleanup
    try:
        os.unlink(db_path)
        for ext in ("-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                os.unlink(p)
    except OSError:
        pass

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
