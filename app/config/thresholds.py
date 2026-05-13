# Central place for tuning thresholds (keep deterministic v1).

DROP_MAX_CHARS = 6              # "ok", "yes", "thx" etc.
IMPORTANCE_DEFAULT = 0.5
IMPORTANCE_HIGH = 0.8

# Breakthrough detection — semantic anchor descriptions for embedding similarity
# These feed cosine similarity, not keyword matching.
GRATITUDE_ANCHOR = "thank you thanks grateful appreciate thankful blessed"
PRAYER_ANCHOR = "pray prayer psalm god jesus faith worship spiritual divine"
VULNERABILITY_ANCHOR = "scared afraid fear anxious panic lonely terrified petrified helpless overwhelmed"

# Retrieval + memory context limits
MIN_MEMORY_SIMILARITY = 0.30
MAX_MEMORY_HITS = 5
MEMORY_CONTEXT_TOKEN_BUDGET = 350  # Approx tokens for memory block
MEMORY_CONTEXT_MAX_CHARS = 600     # Per-memory clamp

# Safety keyword signals — these MUST remain explicit keyword matching.
# Safety guards are auditable, not probabilistic. This is intentional.
SAFETY_SELF_HARM = {
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "cut myself", "overdose", "i want to die", "i don't want to live"
}
SAFETY_VIOLENCE = {
    "kill them", "hurt them", "shoot", "stab", "bomb", "attack", "murder"
}
SAFETY_ILLEGAL = {
    "make a bomb", "build a bomb", "buy a gun illegally", "credit card fraud",
    "steal a car", "hack", "phishing", "malware"
}
