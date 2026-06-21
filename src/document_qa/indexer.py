from __future__ import annotations

from document_qa.chunking import chunk_pages
from document_qa.config import Settings
from document_qa.embeddings import Embedder
from document_qa.ingestion import load_documents


def build_index(settings: Settings, embedder: Embedder, store) -> tuple[int, int]:
    pages = load_documents(settings.data_dir)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    # The complete text list is passed in one call; the model handles it in configured batches.
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    store.replace(chunks, embeddings)
    import gc
    gc.collect()  # Force reclaim memory from the temporary lists and arrays
    return len({page.source for page in pages}), len(chunks)

