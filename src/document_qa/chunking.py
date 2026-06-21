from __future__ import annotations

import hashlib
import re

from document_qa.models import Chunk, DocumentPage


def _split_oversized(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
        elif not current or len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts


def chunk_pages(pages: list[DocumentPage], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    per_source_index: dict[str, int] = {}
    for page in pages:
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        units = [unit for p in paragraphs for unit in _split_oversized(p, chunk_size)]
        current = ""
        page_chunks: list[str] = []
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip()
            if current and len(candidate) > chunk_size:
                page_chunks.append(current)
                tail = current[-overlap:].lstrip() if overlap else ""
                current = f"{tail}\n\n{unit}".strip()
                if len(current) > chunk_size:
                    page_chunks.extend(_split_oversized(current, chunk_size)[:-1])
                    current = _split_oversized(current, chunk_size)[-1]
            else:
                current = candidate
        if current:
            page_chunks.append(current)

        source = page.source.name
        start_index = per_source_index.get(source, 0)
        for offset, text in enumerate(page_chunks):
            index = start_index + offset
            digest = hashlib.sha1(f"{source}:{page.locator}:{index}:{text}".encode()).hexdigest()[:16]
            chunks.append(Chunk(digest, text, source, page.locator, index))
        per_source_index[source] = start_index + len(page_chunks)
    return chunks

