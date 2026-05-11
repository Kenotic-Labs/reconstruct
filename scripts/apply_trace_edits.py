"""Apply Task 1 + Task 2 edits to grammar_engine.py surgically.

Task 1: Fill emotional_valence structurally via dep-label negation detection.
Task 2: Broaden schematic_category via additional OntoNotes NER mappings.

Run: python scripts/apply_trace_edits.py
"""
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "app" / "engines" / "grammar_engine.py"

lines = TARGET.read_text(encoding="utf-8").splitlines(keepends=True)

# --- TASK 1: Insert emotional_valence derivation after emotional_state block ---
# Find the line "                break" after "decomp.emotional_state = tok.text.lower()"
# inside _enrich_defaults_from_doc.
#
# Anchor: the exact line sequence:
#   if tok.pos_ == "ADJ" and tok.dep_ in ("acomp", "attr", "oprd"):
#       decomp.emotional_state = tok.text.lower()
#       break
#
# Insert AFTER the blank line following "break".

task1_anchor = '                decomp.emotional_state = tok.text.lower()\n'
task1_insert_after = '                break\n'

task1_block = '''
    # -- emotional_valence: derive from structural negation on emotion token --
    # Root cause: emotional_valence defaults to None (TraceDecomposition L130).
    # _build_trace_decomposition never sets it.  memory.py L898 falls back to
    # 0.5 unconditionally.
    #
    # Structural fix: when emotional_state is set, find the emotion ADJ token
    # in the doc and check for the UD dep label "neg" on it or its head verb.
    # "neg" is a closed Universal Dependencies relation for syntactic negation
    # (same dep label class used throughout this file: acomp, attr, oprd, etc.).
    #
    # Encoding: float(not has_negation) produces 1.0 (unnegated/positive
    # polarity) or 0.0 (negated/inverted polarity).  This is Python standard
    # bool-to-float conversion -- the same pattern as decomp.negated which
    # stores a boolean dep-tree feature.  NOT a threshold or magic number.
    if decomp.emotional_valence is None and decomp.emotional_state is not None:
        for tok in doc:
            if tok.pos_ == "ADJ" and tok.text.lower() == decomp.emotional_state:
                # Check for structural negation: "neg" dep on the token itself
                # or on its syntactic head (the copular/linking verb).
                has_negation = any(
                    child.dep_ == "neg" for child in tok.children
                )
                if not has_negation and tok.head is not None:
                    has_negation = any(
                        child.dep_ == "neg" for child in tok.head.children
                    )
                decomp.emotional_valence = float(not has_negation)
                break

'''

# --- TASK 2a: Extend NER chain in _build_trace_decomposition ---
# Anchor: the existing block ends with:
#   elif doc_ner_labels & frozenset({"GPE", "FAC"}):
#       schematic_category = "housing"
# Insert additional elif branches after "housing".

task2a_anchor = '            schematic_category = "housing"\n'
task2a_block = '''\
            elif "EVENT" in doc_ner_labels:
                schematic_category = "experience"
            elif "MONEY" in doc_ner_labels:
                schematic_category = "finance"
            elif "NORP" in doc_ner_labels:
                schematic_category = "social"
            elif "LAW" in doc_ner_labels:
                schematic_category = "legal"
            elif "WORK_OF_ART" in doc_ner_labels:
                schematic_category = "culture"
            elif "PRODUCT" in doc_ner_labels:
                schematic_category = "commercial"
            elif "QUANTITY" in doc_ner_labels:
                schematic_category = "measurement"
'''

# --- TASK 2b: Mirror NER extension in classify_query ---
# Anchor: the existing block ends with:
#   elif ner & frozenset({"GPE", "FAC"}):
#       schema = "housing"
# (inside the try block of classify_query)

task2b_anchor_line = '                schema = "housing"\n'
# This anchor appears TWICE (task2a and task2b). We need to patch the SECOND occurrence.
task2b_block = '''\
            elif "EVENT" in ner:
                schema = "experience"
            elif "MONEY" in ner:
                schema = "finance"
            elif "NORP" in ner:
                schema = "social"
            elif "LAW" in ner:
                schema = "legal"
            elif "WORK_OF_ART" in ner:
                schema = "culture"
            elif "PRODUCT" in ner:
                schema = "commercial"
            elif "QUANTITY" in ner:
                schema = "measurement"
'''

# ---- Apply edits ----

content = TARGET.read_text(encoding="utf-8")

# Task 1: Find the exact location.
# We look for the emotional_state block inside _enrich_defaults_from_doc
# and insert after the "break" + blank line.
marker_t1 = (
    "            if tok.pos_ == \"ADJ\" and tok.dep_ in (\"acomp\", \"attr\", \"oprd\"):\n"
    "                decomp.emotional_state = tok.text.lower()\n"
    "                break\n"
    "\n"
    "    # -- temporal_expression"
)
if marker_t1 not in content:
    print("ERROR: Task 1 anchor not found. File may have changed.")
    sys.exit(1)

replacement_t1 = (
    "            if tok.pos_ == \"ADJ\" and tok.dep_ in (\"acomp\", \"attr\", \"oprd\"):\n"
    "                decomp.emotional_state = tok.text.lower()\n"
    "                break\n"
    + task1_block +
    "    # -- temporal_expression"
)
content = content.replace(marker_t1, replacement_t1, 1)
print("Task 1: emotional_valence block inserted.")

# Task 2a: Extend NER chain in _build_trace_decomposition.
# Find the first occurrence of 'schematic_category = "housing"' which is
# inside _build_trace_decomposition (uses schematic_category variable).
marker_t2a = (
    '        elif doc_ner_labels & frozenset({"GPE", "FAC"}):\n'
    '            schematic_category = "housing"\n'
    '\n'
    '    # Possessive-subject family'
)
if marker_t2a not in content:
    print("ERROR: Task 2a anchor not found. File may have changed.")
    sys.exit(1)

replacement_t2a = (
    '        elif doc_ner_labels & frozenset({"GPE", "FAC"}):\n'
    '            schematic_category = "housing"\n'
    + task2a_block +
    '\n'
    '    # Possessive-subject family'
)
content = content.replace(marker_t2a, replacement_t2a, 1)
print("Task 2a: NER chain extended in _build_trace_decomposition.")

# Task 2b: Extend NER chain in classify_query.
# Find the occurrence inside classify_query's try block.
marker_t2b = (
    '            elif ner & frozenset({"GPE", "FAC"}):\n'
    '                schema = "housing"\n'
    '\n'
    '        if schema in ("uncategorized", "identity", "planning"):\n'
    '            root_tok = _get_root(doc)'
)
if marker_t2b not in content:
    print("ERROR: Task 2b anchor not found. File may have changed.")
    sys.exit(1)

replacement_t2b = (
    '            elif ner & frozenset({"GPE", "FAC"}):\n'
    '                schema = "housing"\n'
    + task2b_block +
    '\n'
    '        if schema in ("uncategorized", "identity", "planning"):\n'
    '            root_tok = _get_root(doc)'
)
content = content.replace(marker_t2b, replacement_t2b, 1)
print("Task 2b: NER chain extended in classify_query.")

# Write back
TARGET.write_text(content, encoding="utf-8")
print(f"\nAll edits applied to {TARGET}")
print("Run tests to verify: python -m pytest tests/test_trace_decomposition.py -v")
