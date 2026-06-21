from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from document_qa.config import Settings
from document_qa.indexer import build_index


class RecordingEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class RecordingStore:
    def replace(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings


class IndexerTests(TestCase):
    def test_embeds_all_chunks_in_one_batched_call(self):
        with TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "one.txt").write_text("First substantial paragraph. " * 30, encoding="utf-8")
            (data / "two.txt").write_text("Second substantial paragraph. " * 30, encoding="utf-8")
            settings = Settings(data_dir=data, chunk_size=220, chunk_overlap=30)
            embedder, store = RecordingEmbedder(), RecordingStore()

            document_count, chunk_count = build_index(settings, embedder, store)

            self.assertEqual(document_count, 2)
            self.assertEqual(len(embedder.calls), 1)
            self.assertEqual(len(embedder.calls[0]), chunk_count)
            self.assertEqual(len(store.chunks), len(store.embeddings))

