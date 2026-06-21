from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps configuration importable before optional dependencies are installed.
    def load_dotenv() -> bool:
        return False



@dataclass
class Settings:
    data_dir: Path = Path("data")
    index_dir: Path = Path(".rag_index")
    collection_name: str = "document_qa"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "ollama"
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    openai_chat_model: str = "gpt-4o-mini"
    gemini_chat_model: str = "gemini-2.5-flash"
    top_k: int = 4
    max_distance: float = 0.85
    chunk_size: int = 1200
    chunk_overlap: int = 180
    embedding_batch_size: int = 64

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            index_dir=Path(os.getenv("INDEX_DIR", ".rag_index")),
            collection_name=os.getenv("COLLECTION_NAME", "document_qa"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama").lower(),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            top_k=int(os.getenv("TOP_K", "4")),
            max_distance=float(os.getenv("MAX_DISTANCE", "0.85")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        )

    def validate(self) -> None:
        if self.chunk_size < 200:
            raise ValueError("CHUNK_SIZE must be at least 200 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE")
        if self.top_k < 1:
            raise ValueError("TOP_K must be at least 1")
        if self.llm_provider not in {"ollama", "openai", "gemini"}:
            raise ValueError("LLM_PROVIDER must be 'ollama', 'openai', or 'gemini'")
