from __future__ import annotations

import os
from typing import Protocol

import requests

from document_qa.models import RetrievedChunk


SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer only from the supplied context. If the context does not contain the answer, say exactly:
I couldn't find that information in the indexed documents.
Use concise prose and cite factual claims with the supplied source labels such as [S1].
Never use outside knowledge and never invent a source."""

REWRITE_SYSTEM_PROMPT = """You are a retrieval assistant. Given a conversation history and a follow-up question, rewrite the question to be a standalone search query that can be understood without the conversation history. Do NOT answer the question. Only return the rewritten question. If the question is already standalone or doesn't refer to context from the history, return the original question exactly as is."""


class Generator(Protocol):
    def generate(self, question: str, sources: list[RetrievedChunk]) -> str: ...
    def rewrite_query(self, question: str, history: list[dict[str, str]]) -> str: ...


def build_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[S{i}] {item.chunk.citation}\n{item.chunk.text}"
        for i, item in enumerate(sources, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with inline source labels:"


class OllamaGenerator:
    def __init__(self, base_url: str, model: str) -> None:
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model

    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(question, sources)},
                    ],
                    "options": {"temperature": 0},
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach Ollama. Start Ollama and pull the configured model first."
            ) from exc
        return response.json()["message"]["content"].strip()

    def rewrite_query(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question
        messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": f"Follow-up question: {question}"})
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": messages,
                    "options": {"temperature": 0},
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception:
            return question


class OpenAIGenerator:
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.client = OpenAI()
        self.model = model

    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, sources)},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def rewrite_query(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question
        messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": f"Follow-up question: {question}"})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return question


class GeminiGenerator:
    def __init__(self, model: str) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        self.model = model

    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        prompt = build_prompt(question, sources)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0
            },
            "systemInstruction": {
                "parts": [
                    {"text": SYSTEM_PROMPT}
                ]
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            res_json = response.json()
            return res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")

    def rewrite_query(self, question: str, history: list[dict[str, str]]) -> str:
        if not history:
            return question
        history_lines = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {turn['content']}")
        history_str = "\n".join(history_lines)

        prompt = f"Conversation History:\n{history_str}\n\nFollow-up question: {question}\n\nRewritten standalone question:"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0
            },
            "systemInstruction": {
                "parts": [
                    {"text": REWRITE_SYSTEM_PROMPT}
                ]
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            res_json = response.json()
            return res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            return question

