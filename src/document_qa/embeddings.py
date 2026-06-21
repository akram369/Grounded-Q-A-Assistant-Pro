from __future__ import annotations

from typing import Protocol


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
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return vectors.tolist()

