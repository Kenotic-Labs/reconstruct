from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NURA_",
        extra="ignore"
    )

    # Database
    sqlite_path: str = "Memory Storage/nura.db"

    # Embeddings
    embedding_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"

    # Classifier model (e5-small-v2 — dedicated, always-on, sub-16ms)
    classifier_model: str = "intfloat/e5-small-v2"
    classifier_dim: int = 384

    # Memory settings
    summary_every_n_turns: int = 12
    default_top_k: int = 8
    use_real_embeddings: bool = True
    vector_index_path: str = "Memory Storage/vector_index.faiss"
    memory_jsonl_path: str = "Memory Storage/memory.jsonl"

    # Retrieval mode
    explicit_reconstruct_only: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LLM (Qwen3-4B via llama.cpp)
    llm_model: str = ""

    # TTS (Kokoro - local)
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0

settings = Settings()
