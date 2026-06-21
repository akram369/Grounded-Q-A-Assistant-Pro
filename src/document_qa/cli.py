from __future__ import annotations

import argparse
import sys

from document_qa.config import Settings
from document_qa.embeddings import SentenceTransformerEmbedder
from document_qa.generation import GeminiGenerator, OllamaGenerator, OpenAIGenerator
from document_qa.indexer import build_index
from document_qa.rag import RAGPipeline
from document_qa.vector_store import ChromaVectorStore


def _components(settings: Settings):
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
    store = ChromaVectorStore(settings.index_dir, settings.collection_name)
    if settings.llm_provider == "gemini":
        generator = GeminiGenerator(settings.gemini_chat_model)
    elif settings.llm_provider == "openai":
        generator = OpenAIGenerator(settings.openai_chat_model)
    else:
        generator = OllamaGenerator(settings.ollama_base_url, settings.ollama_model)
    return embedder, store, generator


def _display(answer) -> None:
    print(f"\nAnswer\n------\n{answer.text}\n")
    if answer.sources:
        print("Retrieved sources\n-----------------")
        for number, item in enumerate(answer.sources, start=1):
            preview = " ".join(item.chunk.text.split())[:280]
            print(f"[S{number}] {item.chunk.citation} | distance={item.distance:.3f}\n{preview}...\n")


def _pipeline(settings: Settings) -> RAGPipeline:
    embedder, store, generator = _components(settings)
    return RAGPipeline(embedder, store, generator, settings.top_k, settings.max_distance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grounded document Q&A with RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="ingest documents and rebuild the persistent index")
    ask_parser = subparsers.add_parser("ask", help="ask one question")
    ask_parser.add_argument("question")
    subparsers.add_parser("chat", help="start an interactive question loop")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
        settings.validate()
        if args.command == "index":
            embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
            store = ChromaVectorStore(settings.index_dir, settings.collection_name)
            document_count, chunk_count = build_index(settings, embedder, store)
            print(f"Indexed {document_count} documents into {chunk_count} chunks at {settings.index_dir}")
            return 0

        pipeline = _pipeline(settings)
        if args.command == "ask":
            _display(pipeline.ask(args.question))
            return 0

        print("Document Q&A is ready. Type 'quit' to stop.")
        history: list[dict[str, str]] = []
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"quit", "exit"}:
                break
            if question:
                answer = pipeline.ask(question, history=history)
                _display(answer)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer.text})
        return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130

