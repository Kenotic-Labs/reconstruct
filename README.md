# Nura — Offline-First AI Companion with Persistent Memory

> A fully offline, privacy-first AI assistant with long-lived conversational memory, real-time voice interaction, and semantic understanding — no cloud required.

## Overview

Nura is a six-engine memory architecture designed for long-horizon conversational AI with persistent, adaptive memory capabilities. The system runs entirely offline on consumer hardware, prioritizing privacy, low latency, and architectural discipline.

**Key Differentiators:**
- **100% Offline** — No cloud APIs, no data leaves your device
- **Semantic Understanding** — ML-based comprehension, not regex/keyword matching
- **Persistent Memory** — Remembers facts, preferences, and conversations across sessions
- **Real-Time Voice** — Sub-second speech-to-speech latency (~800ms warm)
- **Privacy-First** — Your conversations stay on your machine

## Architecture

### Six-Engine Design
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Memory    │  │  Retrieval  │  │  Temporal   │
│   Engine    │  │   Engine    │  │   Engine    │
│             │  │             │  │             │
│  Storage &  │  │  Semantic   │  │    Time     │
│   Facts     │  │   Search    │  │  Reasoning  │
└─────────────┘  └─────────────┘  └─────────────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              ┌─────────┴─────────┐
              │    Orchestrator   │
              └─────────┬─────────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Adaptation  │  │  Proactive  │  │  Semantic   │
│   Engine    │  │   Engine    │  │   Router    │
│             │  │             │  │             │
│  Behavior   │  │  Reminders  │  │    NLU      │
│  Learning   │  │  & Nudges   │  │ Understanding│
└─────────────┘  └─────────────┘  └─────────────┘
```

**Memory Engine** — Event ingestion, semantic classification, fact extraction, persistent storage
**Retrieval Engine** — FAISS-accelerated semantic search, temporal-aware ranking
**Temporal Engine** — Time phrase parsing, temporal context generation, deadline tracking
**Adaptation Engine** — User profile evolution, warmth/formality tuning, behavioral adaptation
**Proactive Engine** — Reminder scheduling, follow-up nudges, narrative boundary detection
**Semantic Router** — ML-based intent/emotion/importance detection (replaces all regex)

### Voice Pipeline
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  TEN VAD │───►│ Whisper  │───►│Orchestr- │───►│  Local   │───►│  Piper   │
│  (50ms)  │    │   STT    │    │  ator    │    │   LLM    │    │   TTS    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
   Voice          Speech          Memory          Response         Speech
 Activity        to Text         + Context        Generation       Output
Detection                         Injection
```

**Target Latency:** <500ms end-to-end (achieved: ~806ms warm, ~1200ms cold)

### Semantic Understanding (No Regex)

Nura uses ML-based embeddings for all natural language understanding:

```
┌─────────────────────────────────────────────────────────────┐
│                    SEMANTIC ROUTER                          │
│                                                             │
│   User Input: "my dog name is Shiro"                       │
│                      │                                      │
│                      ▼                                      │
│              ┌──────────────┐                               │
│              │   EMBED ONCE │  (all-MiniLM-L6-v2)          │
│              └──────┬───────┘                               │
│                     │                                       │
│     ┌───────────────┼───────────────┐                      │
│     ▼               ▼               ▼                      │
│ ┌────────┐    ┌──────────┐    ┌────────────┐              │
│ │ Intent │    │   Fact   │    │ Importance │              │
│ │ 95%    │    │ dog_name │    │   HIGH     │              │
│ │PERSONAL│    │  90%     │    │   85%      │              │
│ └────────┘    └──────────┘    └────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

**Concept Domains:**
- `intent_concepts.py` — 7 intent types (personal_state, question, greeting, etc.)
- `temporal_concepts.py` — 30 temporal concepts (tomorrow, next week, etc.)
- `fact_concepts.py` — 20 fact types (name, age, pet, location, etc.)
- `query_concepts.py` — 16 query types (recall, search, compare, etc.)
- `emotion_concepts.py` — 20 emotion states (happy, stressed, anxious, etc.)
- `importance_concepts.py` — 16 importance levels (urgent, trivial, etc.)

### Design Principles

- **Offline-First** — Every component runs locally; no network required
- **Semantic Over Regex** — ML embeddings understand meaning, not patterns
- **Embed Once, Understand Everywhere** — Single embedding serves all engines
- **Strict Separation of Concerns** — Each engine has non-overlapping responsibilities
- **Privacy by Architecture** — No telemetry, no cloud, no data collection

## Current Status

### Phase 1–5: Core Architecture ✅ **COMPLETED**

- ✅ Architectural boundary enforcement
- ✅ Cross-engine orchestration
- ✅ Dead code resolution
- ✅ Protocol interfaces
- ✅ Testing & validation (92.9% pass rate)

---

### Phase 6: Scale Preparation ✅ **COMPLETED**

**Objective:** Replace development implementations with production-grade components.

- ✅ **6.1 Sentence Transformers** — `all-MiniLM-L6-v2` for semantic embeddings
- ✅ **6.2 FAISS Integration** — O(log N) vector search with `IndexFlatIP`
- ✅ **6.3 Database Optimizations** — WAL mode, connection pooling, indexes

---

### Phase 7: Voice Pipeline ✅ **COMPLETED**

**Objective:** Real-time speech-to-speech interaction.

- ✅ **7.1 TEN VAD** — 50ms latency voice activity detection
- ✅ **7.2 Whisper STT** — Local speech recognition (faster-whisper)
- ✅ **7.3 Piper TTS** — Neural text-to-speech (Jenny voice)
- ✅ **7.4 Streaming Pipeline** — Token-by-token TTS for low latency

**Achieved Latency:**
| Component | Time |
|-----------|------|
| VAD | ~50ms |
| STT | ~150-200ms |
| Semantic Analysis | ~15-50ms |
| LLM Inference | ~400-600ms |
| TTS | ~100-150ms |
| **Total (warm)** | **~806ms** |

---

### Phase 8: Semantic Engine Migration ✅ **COMPLETED**

**Objective:** Replace all regex/keyword matching with ML-based semantic understanding.

- ✅ **8.1 Intent Classification** — Semantic embeddings replace regex patterns
- ✅ **8.2 Temporal Parsing** — Semantic concepts replace time regex
- ✅ **8.3 Fact Extraction** — Semantic detection of personal facts
- ✅ **8.4 Memory Classification** — Semantic importance replaces CSV keywords
- ✅ **8.5 STT Prompting** — initial_prompt generated from semantic concepts

**Migration Summary:**
| Component | Before | After |
|-----------|--------|-------|
| Intent Detection | 50+ regex patterns | Semantic embeddings |
| Temporal Parsing | 100+ time patterns | 30 temporal concepts |
| Fact Extraction | Hardcoded patterns | 20 fact type concepts |
| Memory Classification | CSV trigger words | Semantic importance |
| Query Detection | Keyword lists | 16 query concepts |
| Emotion Detection | Word lists | 20 emotion concepts |

---

### Phase 9: Proactive Intelligence ✅ **COMPLETED**

**Objective:** Autonomous reminders and follow-ups without user prompting.

- ✅ **9.1 Reminder Scheduling** — "Remind me tomorrow" creates scheduled nudges
- ✅ **9.2 Follow-up Detection** — Detects unresolved commitments
- ✅ **9.3 Narrative Boundaries** — Understands event conclusions
- ✅ **9.4 Cooldown System** — Prevents reminder spam

---

### Phase 10: LLM Fine-Tuning ✅ **COMPLETED**

**Objective:** Custom personality and response style.

- ✅ **10.1 Base Model Selection** — Qwen 2.5 3B Instruct
- ✅ **10.2 LoRA Training** — Identity injection, memory awareness
- ✅ **10.3 GGUF Export** — Quantized for CPU inference (Q4_K_M)
- ✅ **10.4 Personality Embedding** — Warm, supportive, memory-aware responses

---

### Phase 11: Production Hardening 📋 **IN PROGRESS**

**Objective:** Installer, error handling, edge cases.

- ✅ **11.1 Windows Installer** — NSIS-based one-click setup
- ✅ **11.2 Model Downloads** — Automatic first-run model fetching
- [ ] **11.3 Error Recovery** — Graceful degradation on component failure
- [ ] **11.4 Multi-user Support** — User profile switching

## Project Structure

```
nura/
├── app/
│   ├── orchestrator/       # Central coordination
│   │   ├── orchestrator.py # Main engine coordinator
│   │   └── engine_policy.py # Engine activation rules
│   │
│   ├── semantic/           # ML-based understanding
│   │   ├── semantic_router.py    # Unified NLU entry point
│   │   ├── concept_store.py      # Embedding cache
│   │   └── concepts/             # Domain-specific concepts
│   │       ├── intent_concepts.py
│   │       ├── temporal_concepts.py
│   │       ├── fact_concepts.py
│   │       ├── query_concepts.py
│   │       ├── emotion_concepts.py
│   │       └── importance_concepts.py
│   │
│   ├── memory/             # Persistent storage
│   │   ├── memory_engine.py      # Event ingestion
│   │   ├── memory_store.py       # SQLite operations
│   │   ├── memory_classifier.py  # Semantic classification
│   │   └── memory_summarizer.py  # Session compression
│   │
│   ├── retrieval/          # Semantic search
│   │   ├── retrieval_engine.py   # Search orchestration
│   │   ├── ranker.py             # Relevance scoring
│   │   └── query_parser.py       # Query understanding
│   │
│   ├── temporal/           # Time reasoning
│   │   ├── temporal_engine.py    # Time awareness
│   │   └── temporal_patterns.py  # Pattern detection
│   │
│   ├── adaptation/         # User modeling
│   │   └── adaptation_engine.py  # Profile evolution
│   │
│   ├── proactive/          # Autonomous actions
│   │   └── proactive_engine.py   # Reminder scheduling
│   │
│   ├── services/           # External interfaces
│   │   ├── realtime_stt.py       # Whisper + TEN VAD
│   │   ├── streaming_tts.py      # Piper neural TTS
│   │   ├── nura_llm_interface.py # Local LLM inference
│   │   └── wake_word_listener.py # "Hey Nura" detection
│   │
│   ├── vector/             # Embeddings & search
│   │   ├── embedding_service.py  # all-MiniLM-L6-v2
│   │   └── vector_index.py       # FAISS index
│   │
│   ├── guards/             # Safety & limits
│   │   ├── safety_layer.py       # Content filtering
│   │   └── token_budget.py       # Context management
│   │
│   ├── db/                 # Database
│   │   └── session.py            # SQLite connection pool
│   │
│   └── api/                # REST endpoints
│       └── memory_routes.py      # Memory CRUD
│
├── config/
│   ├── settings.py         # Global configuration
│   ├── thresholds.py       # Tunable parameters
│   └── model_paths.py      # Model file locations
│
├── models/                 # Downloaded models
│   ├── nura-v3-q4_k_m.gguf      # Fine-tuned LLM
│   ├── all-MiniLM-L6-v2/        # Embedding model
│   └── jenny_piper/             # TTS voice
│
├── Training/               # Fine-tuning scripts
│   ├── train_lora.py
│   └── export_gguf.py
│
└── Docs/                   # Documentation
    ├── SketchArchitecture.md
    └── NURA_DEVELOPMENT_STATUS.md
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **LLM** | Qwen 2.5 3B (LoRA fine-tuned, Q4_K_M quantized) |
| **LLM Runtime** | llama-cpp-python |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) |
| **Vector Search** | FAISS (IndexFlatIP) |
| **STT** | faster-whisper (small.en) |
| **VAD** | TEN VAD (50ms latency) |
| **TTS** | Piper (Jenny neural voice) |
| **Database** | SQLite (WAL mode) |
| **API** | FastAPI |
| **Testing** | pytest |

## Hardware Requirements

| Tier | RAM | Storage | Performance |
|------|-----|---------|-------------|
| **Minimum** | 8GB | 5GB | ~2s latency |
| **Recommended** | 16GB | 6GB | ~800ms latency |
| **Optimal** | 32GB + GPU | 8GB | ~400ms latency |

## Installation

### Windows (Recommended)
```bash
# Download and run the installer
Nura_Setup.exe

# Or manual installation
git clone https://github.com/Talknura/Nura.git
cd Nura
pip install -r requirements.txt
python first_run_setup.py  # Downloads models
python run_ultra.py        # Start Nura
```

### Voice Interaction
```
Say: "Hey Nura"           # Wake word
Say: "My name is Sam"     # Nura remembers
Say: "What's my name?"    # Nura recalls: "Sam"
Say: "Bye Nura"           # Session ends, memories summarized
```

## Privacy & Security

- **No Cloud** — All processing happens locally
- **No Telemetry** — No usage data collected
- **No Network** — Works in airplane mode
- **Local Storage** — SQLite database in user directory
- **Your Data** — Stays on your device, always

## Roadmap

- [x] Phase 1–5: Core Architecture
- [x] Phase 6: Scale Preparation (FAISS, Embeddings)
- [x] Phase 7: Voice Pipeline
- [x] Phase 8: Semantic Engine Migration
- [x] Phase 9: Proactive Intelligence
- [x] Phase 10: LLM Fine-Tuning
- [ ] Phase 11: Production Hardening
- [ ] Phase 12: Mobile Companion App
- [ ] Phase 13: Multi-modal (Vision)
- [ ] Phase 14: Edge Deployment (NVIDIA Jetson Nano Orin)

## Model-Agnostic Architecture

Nura's six-engine architecture is **model-agnostic** — designed to work with any LLM, not locked to a single provider.

**Current:** Phi-3.5 3B (local, offline)
**Next:** NVIDIA PersonaPlex 7B (full-duplex speech-to-speech for demo)
**Future:** Custom Nura model (in development)

The engines (Memory, Retrieval, Temporal, Adaptation, Proactive, Semantic Router) plug into ANY model. As better models emerge, Nura evolves — same engines, upgraded brain. That's the business model.

```
┌─────────────────────────────────────────────────┐
│           NURA ENGINE LAYER                     │
│  Memory | Temporal | Proactive | Adaptation     │
│  Retrieval | Semantic Router | Safety           │
└───────────────────────┬─────────────────────────┘
                        │ Context Injection
                        ▼
┌─────────────────────────────────────────────────┐
│              MODEL LAYER (Swappable)            │
├─────────────────────────────────────────────────┤
│  Today:    Phi-3.5 3B + Whisper + Kokoro TTS   │
│  Next:     PersonaPlex 7B (full-duplex voice)  │
│  Future:   Custom Nura Model                    │
└─────────────────────────────────────────────────┘
```

## Research Context

This project explores:
- **Offline-first AI** — Bringing cloud-level capabilities to local devices
- **Semantic memory architectures** — Long-horizon conversational persistence
- **Privacy-preserving AI** — No compromise between capability and privacy
- **Model-agnostic design** — Engine layer decoupled from model layer

## Author

**Samuel Sameer Tanguturi**
Master of Science in Information Systems
Central Michigan University

**Contact:** Tangu1s@cmich.edu
**LinkedIn:** [linkedin.com/in/tanguturi-sameer](https://www.linkedin.com/in/tanguturi-sameer-3a5b57303)
**Project Started:** October 2025

---

## License

**Proprietary** — All rights reserved. This is a private research project.

---

*Nura proves that truly private AI assistants are possible. No cloud required. No compromises on capability. Your memories, your device, your control.*
