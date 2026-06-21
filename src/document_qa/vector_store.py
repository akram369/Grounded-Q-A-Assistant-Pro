from __future__ import annotations

from pathlib import Path

from document_qa.models import Chunk, RetrievedChunk


class ChromaVectorStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        for start in range(0, len(chunks), batch_size):
            group = chunks[start : start + batch_size]
            self.collection.add(
                ids=[chunk.id for chunk in group],
                documents=[chunk.text for chunk in group],
                metadatas=[
                    {
                        "source": chunk.source,
                        "locator": chunk.locator,
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in group
                ],
                embeddings=embeddings[start : start + batch_size],
            )

    def count(self) -> int:
        return self.collection.count()

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        found = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = Chunk(
                id=chunk_id,
                text=text,
                source=str(metadata["source"]),
                locator=str(metadata["locator"]),
                chunk_index=int(metadata["chunk_index"]),
            )
            found.append(RetrievedChunk(chunk, float(distance)))
        return found

