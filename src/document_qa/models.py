from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocumentPage:
    source: Path
    locator: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    locator: str
    chunk_index: int

    @property
    def citation(self) -> str:
        return f"{self.source}, {self.locator}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    distance: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    answerable: bool = True
    search_query: str | None = None

