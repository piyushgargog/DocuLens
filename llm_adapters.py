"""One adapter per chat-API family, so any model can be used.

llm_client builds the same OpenAI-style `messages` list for every provider; an
adapter turns that into the provider's own request, reads its reply, and reads
its server-sent-event stream as ("content" | "reasoning", text) pieces:

* `openai`     -- `/chat/completions`. Covers OpenAI, Groq, OpenRouter, NVIDIA
                  (Nemotron, gpt-oss, Llama ...), Hugging Face, Mistral, DeepSeek,
                  xAI, Together, Azure OpenAI, Ollama, vLLM, LM Studio and
                  Google's OpenAI-compatible Gemini endpoint.
* `anthropic`  -- the Messages API (Claude).
* `gemini`     -- Google's native `generateContent` API.
* `cohere`     -- Cohere's v2 chat API.

Adding another family means one more class here and one entry in ADAPTERS.
Adapters only shape requests and replies; keys, cooldowns and failover stay in
providers.py / llm_client.py. Nothing here ever sees user-controlled URLs: the
base URL comes from the operator's configuration.
"""

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from urllib.parse import urlencode

import requests

from llm_errors import LLMRequestError

DEFAULT_MAX_TOKENS = 4096  # APIs that require a cap on the reply length


def max_tokens() -> int:
    try:
        return max(16, int(os.environ.get("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    except ValueError:
        return DEFAULT_MAX_TOKENS


def _sse_data(response) -> Iterator[tuple[str, str]]:
    """(event name, data) from a server-sent-events response. SSE is UTF-8 by
    definition, but providers send `text/event-stream` without a charset and
    requests then guesses ISO-8859-1, which mangles non-ASCII text (found in
    v2.0.0 testing), so the encoding is forced."""
    response.encoding = "utf-8"
    event = ""
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                event = ""
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                yield event, line[5:].strip()
    except requests.RequestException as exc:
        raise LLMRequestError(f"Streaming read failed: {type(exc).__name__}") from exc


def _merge_same_role(turns: list[dict]) -> list[dict]:
    """APIs that need strictly alternating roles: join neighbours with the same role."""
    out: list[dict] = []
    for turn in turns:
        if out and out[-1]["role"] == turn["role"]:
            out[-1] = {"role": turn["role"], "content": out[-1]["content"] + "\n\n" + turn["content"]}
        else:
            out.append(dict(turn))
    return out


@dataclass(frozen=True)
class Request:
    url: str
    headers: dict
    payload: dict


class Adapter:
    name = ""
    default_auth = "bearer"

    def auth_headers(self, route) -> dict:
        auth = route.auth or self.default_auth
        key = route.api_key
        if auth == "none" or not key:
            return {}
        if auth == "bearer":
            return {"Authorization": f"Bearer {key}"}
        if auth in ("x-api-key", "api-key", "x-goog-api-key"):
            return {auth: key}
        return {}

    def request(self, route, messages: list[dict], stream: bool) -> Request:
        raise NotImplementedError

    def parse(self, data: dict) -> str:
        raise NotImplementedError

    def stream(self, response) -> Iterator[tuple[str, str]]:
        raise NotImplementedError


class OpenAIAdapter(Adapter):
    name = "openai"

    def request(self, route, messages, stream):
        payload = {"model": route.model, "messages": messages, "temperature": 0.0}
        if stream:
            payload["stream"] = True
        url = f"{route.base_url}/chat/completions"
        if route.query:
            url += "?" + urlencode(route.query)  # e.g. Azure's api-version
        headers = {**self.auth_headers(route), "Content-Type": "application/json", **dict(route.headers)}
        return Request(url, headers, payload)

    def parse(self, data):
        return (data["choices"][0]["message"].get("content") or "").strip()

    def stream(self, response):
        """Pieces from an OpenAI-compatible stream, where kind is "content" (the
        answer) or "reasoning" (the model's own thinking, which gpt-oss and
        Nemotron send in a separate `reasoning` delta field). Reasoning is shown
        as a collapsible panel and is never stored as the answer."""
        for _, data in _sse_data(response):
            if not data:
                continue
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if "error" in event:
                raise LLMRequestError("The LLM API reported an error while streaming.")
            try:
                delta = event["choices"][0]["delta"]
            except (KeyError, IndexError, TypeError):
                continue
            if not isinstance(delta, dict):
                continue
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                yield ("reasoning", reasoning)
            content = delta.get("content")
            if content:
                yield ("content", content)


class AnthropicAdapter(Adapter):
    name = "anthropic"
    default_auth = "x-api-key"
    VERSION = "2023-06-01"

    def request(self, route, messages, stream):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = _merge_same_role(
            [{"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]} for m in messages if m["role"] != "system"]
        )
        payload = {"model": route.model, "max_tokens": max_tokens(), "temperature": 0.0, "messages": turns}
        if system:
            payload["system"] = system
        if stream:
            payload["stream"] = True
        base = route.base_url.removesuffix("/v1")
        headers = {**self.auth_headers(route), "anthropic-version": self.VERSION, "Content-Type": "application/json", **dict(route.headers)}
        return Request(f"{base}/v1/messages", headers, payload)

    def parse(self, data):
        return "".join(b.get("text", "") for b in data["content"] if isinstance(b, dict) and b.get("type") == "text").strip()

    def stream(self, response):
        for _, data in _sse_data(response):
            try:
                event = json.loads(data)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "error":
                raise LLMRequestError("The LLM API reported an error while streaming.")
            if kind == "message_stop":
                return
            if kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield ("content", delta["text"])
                elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                    yield ("reasoning", delta["thinking"])


class GeminiAdapter(Adapter):
    name = "gemini"
    default_auth = "x-goog-api-key"

    def request(self, route, messages, stream):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = _merge_same_role(
            [{"role": "model" if m["role"] == "assistant" else "user", "content": m["content"]} for m in messages if m["role"] != "system"]
        )
        payload = {
            "contents": [{"role": t["role"], "parts": [{"text": t["content"]}]} for t in turns],
            "generationConfig": {"temperature": 0.0},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        model = route.model.removeprefix("models/")
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        headers = {**self.auth_headers(route), "Content-Type": "application/json", **dict(route.headers)}
        return Request(f"{route.base_url}/v1beta/models/{model}:{method}", headers, payload)

    @staticmethod
    def _parts(data: dict) -> list[dict]:
        return ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []

    def parse(self, data):
        return "".join(p.get("text", "") for p in self._parts(data) if not p.get("thought")).strip()

    def stream(self, response):
        for _, data in _sse_data(response):
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if "error" in event:
                raise LLMRequestError("The LLM API reported an error while streaming.")
            for part in self._parts(event):
                text = part.get("text")
                if text:
                    yield ("reasoning" if part.get("thought") else "content", text)


class CohereAdapter(Adapter):
    name = "cohere"

    def request(self, route, messages, stream):
        payload = {"model": route.model, "messages": messages, "temperature": 0.0}
        if stream:
            payload["stream"] = True
        base = route.base_url.removesuffix("/v2")
        headers = {**self.auth_headers(route), "Content-Type": "application/json", **dict(route.headers)}
        return Request(f"{base}/v2/chat", headers, payload)

    def parse(self, data):
        content = (data.get("message") or {}).get("content") or []
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type", "text") == "text").strip()

    def stream(self, response):
        for event_name, data in _sse_data(response):
            try:
                event = json.loads(data)
            except ValueError:
                continue
            kind = event.get("type") or event_name
            if kind == "message-end":
                return
            if kind == "content-delta":
                text = (((event.get("delta") or {}).get("message") or {}).get("content") or {}).get("text")
                if text:
                    yield ("content", text)


ADAPTERS: dict[str, Adapter] = {a.name: a for a in (OpenAIAdapter(), AnthropicAdapter(), GeminiAdapter(), CohereAdapter())}


def get(api: str) -> Adapter:
    return ADAPTERS.get(api, ADAPTERS["openai"])
