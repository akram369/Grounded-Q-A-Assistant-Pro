from unittest import TestCase

from document_qa.models import Chunk, RetrievedChunk
from document_qa.rag import NOT_FOUND, RAGPipeline


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, results):
        self.results = results

    def count(self):
        return len(self.results)

    def search(self, query_embedding, top_k):
        return self.results[:top_k]


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, sources):
        self.calls.append((question, sources))
        return "Trees cool streets through shade and evapotranspiration [S1]."

    def rewrite_query(self, question, history):
        return question


class RAGTests(TestCase):
    def test_generates_only_from_relevant_sources(self):
        relevant = RetrievedChunk(Chunk("1", "Trees cool streets.", "heat.txt", "text", 0), 0.2)
        distant = RetrievedChunk(Chunk("2", "Unrelated material.", "other.txt", "text", 0), 1.1)
        generator = FakeGenerator()
        pipeline = RAGPipeline(FakeEmbedder(), FakeStore([relevant, distant]), generator, 4, 0.8)

        answer = pipeline.ask("How do trees cool streets?")

        self.assertTrue(answer.answerable)
        self.assertEqual(answer.sources, [relevant])
        self.assertEqual(generator.calls[0][1], [relevant])
        self.assertIn("[S1]", answer.text)

    def test_refuses_when_retrieval_is_too_distant(self):
        distant = RetrievedChunk(Chunk("2", "No answer here.", "other.txt", "text", 0), 1.2)
        generator = FakeGenerator()
        pipeline = RAGPipeline(FakeEmbedder(), FakeStore([distant]), generator, 4, 0.8)

        answer = pipeline.ask("Who won a match yesterday?")

        self.assertFalse(answer.answerable)
        self.assertEqual(answer.text, NOT_FOUND)
        self.assertEqual(generator.calls, [])

    def test_query_rewriting_with_history(self):
        relevant = RetrievedChunk(Chunk("1", "Trees cool streets.", "heat.txt", "text", 0), 0.2)
        generator = FakeGenerator()
        
        def mock_rewrite(q, h):
            return "How do trees cool streets?"
        generator.rewrite_query = mock_rewrite
        
        pipeline = RAGPipeline(FakeEmbedder(), FakeStore([relevant]), generator, 4, 0.8)
        history = [{"role": "user", "content": "Tell me about trees."}]
        
        answer = pipeline.ask("How do they cool streets?", history=history)
        
        self.assertTrue(answer.answerable)
        self.assertEqual(answer.search_query, "How do trees cool streets?")

