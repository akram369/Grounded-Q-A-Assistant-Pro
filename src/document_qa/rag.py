from __future__ import annotations

from document_qa.embeddings import Embedder
from document_qa.generation import Generator
from document_qa.models import Answer

NOT_FOUND = "I couldn't find that information in the indexed documents."


class RAGPipeline:
    def __init__(self, embedder: Embedder, store, generator: Generator, top_k: int, max_distance: float) -> None:
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.top_k = top_k
        self.max_distance = max_distance

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        if self.store.count() == 0:
            raise RuntimeError("The index is empty. Run the indexing command first.")
        
        search_query = question
        if history:
            search_query = self.generator.rewrite_query(question, history)

        query_vector = self.embedder.embed([search_query])[0]
        candidates = self.store.search(query_vector, self.top_k)
        relevant = [item for item in candidates if item.distance <= self.max_distance]
        if not relevant:
            return Answer(NOT_FOUND, candidates, answerable=False, search_query=search_query)
        text = self.generator.generate(question, relevant)
        return Answer(text or NOT_FOUND, relevant, answerable=bool(text), search_query=search_query)

