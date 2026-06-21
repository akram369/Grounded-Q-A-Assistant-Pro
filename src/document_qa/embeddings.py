from __future__ import annotations

import os
from typing import Protocol
import requests


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(1)  # Prevent CPU thread explosion and memory spikes in Docker containers

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch
        with torch.no_grad():
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        return vectors.tolist()


class GeminiEmbedder:
    def __init__(self, api_key: str | None = None, model: str = "models/gemini-embedding-2") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiEmbedder")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        embeddings = []
        # Gemini batch limit is 100 per request
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents?key={self.api_key}"
            payload = {
                "requests": [
                    {
                        "model": self.model,
                        "content": {"parts": [{"text": t}]}
                    }
                    for t in batch
                ]
            }
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                for emb in data.get("embeddings", []):
                    embeddings.append(emb["values"])
            except Exception as e:
                raise RuntimeError(f"Gemini embedding API call failed: {e}")
        return embeddings


class OpenAIEmbedder:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            url = "https://api.openai.com/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "input": batch,
                "model": self.model
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                for item in data.get("data", []):
                    embeddings.append(item["embedding"])
            except Exception as e:
                raise RuntimeError(f"OpenAI embedding API call failed: {e}")
        return embeddings


def get_embedder(llm_provider: str, embedding_model: str, embedding_batch_size: int) -> Embedder:
    if llm_provider == "gemini":
        return GeminiEmbedder()
    elif llm_provider == "openai":
        return OpenAIEmbedder()
    else:
        return SentenceTransformerEmbedder(embedding_model, embedding_batch_size)
