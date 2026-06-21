from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from document_qa.models import DocumentPage

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}


def _clean_lines(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def _pdf_pages(path: Path) -> list[DocumentPage]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    raw_pages = [(page.extract_text() or "") for page in reader.pages]

    # Remove repeated short first/last lines, a common header/footer pattern.
    edge_lines: list[str] = []
    page_lines: list[list[str]] = []
    for text in raw_pages:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_lines.append(lines)
        edge_lines.extend(lines[:2] + lines[-2:])
    repeated = {
        line
        for line, count in Counter(edge_lines).items()
        if count >= max(2, len(raw_pages) // 2) and len(line) < 100
    }

    pages = []
    for number, lines in enumerate(page_lines, start=1):
        kept = [line for line in lines if line not in repeated and not re.fullmatch(r"Page \d+", line)]
        text = _clean_lines("\n".join(kept))
        if text:
            pages.append(DocumentPage(path, f"page {number}", text))
    return pages


def _txt_pages(path: Path) -> list[DocumentPage]:
    text = path.read_text(encoding="utf-8")
    return [DocumentPage(path, "text", _clean_lines(text))]


def _docx_pages(path: Path) -> list[DocumentPage]:
    from docx import Document

    document = Document(path)
    sections: list[DocumentPage] = []
    heading = "document"
    body: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style.name.lower().startswith("heading"):
            if body:
                sections.append(DocumentPage(path, f"section: {heading}", _clean_lines("\n".join(body))))
            heading, body = text, []
        else:
            body.append(text)
    if body:
        sections.append(DocumentPage(path, f"section: {heading}", _clean_lines("\n".join(body))))
    return sections


def load_documents(data_dir: Path) -> list[DocumentPage]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Document folder does not exist: {data_dir}")
    paths = sorted(path for path in data_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported documents found in {data_dir}")

    pages: list[DocumentPage] = []
    for path in paths:
        suffix = path.suffix.lower()
        loaded = _pdf_pages(path) if suffix == ".pdf" else _docx_pages(path) if suffix == ".docx" else _txt_pages(path)
        pages.extend(page for page in loaded if page.text)
    return pages

