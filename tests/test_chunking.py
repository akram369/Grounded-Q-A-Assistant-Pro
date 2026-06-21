from pathlib import Path
from unittest import TestCase

from document_qa.chunking import chunk_pages
from document_qa.models import DocumentPage


class ChunkingTests(TestCase):
    def test_preserves_source_locator_and_overlap(self):
        text = (
            "Cities store heat in roads and buildings. Trees provide shade and evaporative cooling.\n\n"
            "Cooling centers need transport, accessible entrances, water, and useful opening hours.\n\n"
            "Long-term plans should measure health outcomes rather than count activities alone."
        )
        chunks = chunk_pages(
            [DocumentPage(Path("heat.txt"), "section: response", text)],
            chunk_size=130,
            overlap=28,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.source == "heat.txt" for chunk in chunks))
        self.assertTrue(all(chunk.locator == "section: response" for chunk in chunks))
        self.assertIn(chunks[0].text[-20:].strip(), chunks[1].text)
        self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))

    def test_long_sentence_is_bounded(self):
        chunks = chunk_pages(
            [DocumentPage(Path("long.txt"), "text", "word " * 150)],
            chunk_size=100,
            overlap=10,
        )
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.text) <= 110 for chunk in chunks))

