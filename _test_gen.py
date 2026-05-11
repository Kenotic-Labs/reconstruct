import sys

RESOLVER = chr(10).join([
    '',
    '    # ===================================================================',
    '    # INDIRECT REFERENCE RESOLUTION',
    '    # ===================================================================',
    '',
    '    _SPEAKER_REFS = frozenset({',
    '        "speaker", "user", "narrator",',
    '    })',
    '',
    '    _GENERIC_PERSON_NOUNS = frozenset({',
    '        "person", "one", "guy", "woman", "man", "friend", "people",',
    '        "someone", "somebody", "individual",',
    '    })',
    '',
])
print(len(RESOLVER))