"""Test question formation on actual LOCOMO conversation sentences."""
import json
import spacy
from test_pq_formula import form_question, _find_target, _get_root

nlp = spacy.load("en_core_web_md")

# Load LOCOMO conv 0
with open("locomo_bench/locomo/data/locomo10.json") as f:
    data = json.load(f)

conv = data[0]["conversation"]
speaker_a = conv["speaker_a"]  # Caroline
speaker_b = conv["speaker_b"]  # Melanie

# Collect first 50 statement turns (skip questions, backchannels)
session_keys = sorted(
    [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
    key=lambda k: int(k.split("_")[1]),
)

turns = []
for sk in session_keys:
    session_turns = conv[sk]
    if not isinstance(session_turns, list):
        continue
    for turn in session_turns:
        text = turn.get("text", "")
        speaker = turn.get("speaker", "")
        if text and speaker:
            turns.append((speaker, text))

# Process first 30 turns, generate questions for each sentence
total_questions = 0
total_sentences = 0

for speaker, text in turns[:30]:
    doc = nlp(text)
    for sent in doc.sents:
        sent_doc = sent.as_doc()
        root = _get_root(sent_doc)
        if not root:
            continue
        # Skip questions, backchannels, fragments
        if any(tok.text == "?" for tok in sent_doc):
            continue
        if len(sent_doc) < 4:
            continue

        total_sentences += 1
        questions = []

        # Try each answer constituent type
        for target_dep in ("dobj", "attr", "acomp", "pobj", "ccomp", "xcomp", "advmod", "nsubj"):
            target = _find_target(sent_doc, root, target_dep)
            if target:
                q = form_question(sent_doc, target, speaker)
                if q and not q.startswith("("):
                    questions.append((target_dep, q))

        if questions:
            print(f'[{speaker}] "{str(sent_doc).strip()[:70]}"')
            for dep, q in questions:
                total_questions += 1
                print(f"  {dep:8s} -> {q}")
            print()

print(f"\n{total_sentences} sentences processed, {total_questions} questions generated")
