"""Any model, any chat API: request shapes, reply parsing and streaming for each
API family; provider presets, comma-separated models and LLM_ROUTES; and the
failover chain across families."""

import json

import pytest

import llm_adapters
import llm_client
import providers
from providers import Route

MESSAGES = [
    {"role": "system", "content": "SYS RULES"},
    {"role": "user", "content": "Earlier question"},
    {"role": "assistant", "content": "Earlier answer"},
    {"role": "user", "content": "Now the question"},
]


def route(api="openai", base="https://api.example.com/v1", model="m1", key="sk-secret", **kw):
    return Route("p", "P", base, model, key, api=api, **kw)


class FakeStream:
    def __init__(self, lines):
        self.lines, self.encoding, self.closed = lines, None, False

    def iter_lines(self, decode_unicode=False):
        yield from self.lines

    def close(self):
        self.closed = True


def sse(*events, named=False):
    lines = []
    for ev in events:
        if named and isinstance(ev, tuple):
            lines += [f"event: {ev[0]}", f"data: {json.dumps(ev[1])}", ""]
        else:
            lines += [f"data: {ev if isinstance(ev, str) else json.dumps(ev)}", ""]
    return FakeStream(lines)


# ---------- OpenAI family ----------


def test_openai_request_shape():
    r = llm_adapters.get("openai").request(route(), MESSAGES, stream=False)
    assert r.url == "https://api.example.com/v1/chat/completions"
    assert r.headers["Authorization"] == "Bearer sk-secret"
    assert r.payload == {"model": "m1", "messages": MESSAGES, "temperature": 0.0}
    assert llm_adapters.get("openai").request(route(), MESSAGES, stream=True).payload["stream"] is True


def test_openai_extra_headers_and_query_for_azure_style_endpoints():
    r = llm_adapters.get("openai").request(
        route(auth="api-key", headers=(("X-Org", "acme"),), query=(("api-version", "2024-06-01"),)), MESSAGES, False
    )
    assert r.url.endswith("/chat/completions?api-version=2024-06-01")
    assert r.headers["api-key"] == "sk-secret" and "Authorization" not in r.headers and r.headers["X-Org"] == "acme"


def test_keyless_local_servers_send_no_credentials():
    r = llm_adapters.get("openai").request(route(key="", base="http://localhost:11434/v1"), MESSAGES, False)
    assert "Authorization" not in r.headers
    assert "Authorization" not in llm_adapters.get("openai").request(route(auth="none"), MESSAGES, False).headers


def test_openai_parse_and_stream_with_reasoning():
    adapter = llm_adapters.get("openai")
    assert adapter.parse({"choices": [{"message": {"content": " hi "}}]}) == "hi"
    stream = sse(
        {"choices": [{"delta": {"reasoning": "thinking"}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        "[DONE]",
        {"choices": [{"delta": {"content": "NEVER"}}]},
    )
    assert list(adapter.stream(stream)) == [("reasoning", "thinking"), ("content", "Hel"), ("content", "lo")]
    assert stream.encoding == "utf-8"  # forced: providers omit the charset


def test_openai_stream_error_event_raises():
    with pytest.raises(llm_client.LLMRequestError):
        list(llm_adapters.get("openai").stream(sse({"error": {"message": "x"}})))


# ---------- Anthropic ----------


def test_anthropic_request_separates_the_system_prompt_and_alternates_roles():
    r = llm_adapters.get("anthropic").request(route("anthropic", "https://api.anthropic.com", "claude-x"), MESSAGES, False)
    assert r.url == "https://api.anthropic.com/v1/messages"
    assert r.headers["x-api-key"] == "sk-secret" and r.headers["anthropic-version"] == "2023-06-01" and "Authorization" not in r.headers
    p = r.payload
    assert p["system"] == "SYS RULES" and p["model"] == "claude-x" and p["max_tokens"] >= 16 and p["temperature"] == 0.0
    assert [m["role"] for m in p["messages"]] == ["user", "assistant", "user"] and all(m["role"] != "system" for m in p["messages"])


def test_anthropic_merges_consecutive_user_turns_and_tolerates_a_v1_base():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    r = llm_adapters.get("anthropic").request(route("anthropic", "https://api.anthropic.com/v1", "c"), msgs, True)
    assert r.url == "https://api.anthropic.com/v1/messages" and r.payload["stream"] is True
    assert r.payload["messages"] == [{"role": "user", "content": "a\n\nb"}]


def test_anthropic_parse_ignores_non_text_blocks():
    data = {"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Answer "}, {"type": "text", "text": "here"}]}
    assert llm_adapters.get("anthropic").parse(data) == "Answer here"


def test_anthropic_stream_text_thinking_and_stop():
    stream = sse(
        {"type": "message_start"},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "plan"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "B"}},
        {"type": "message_stop"},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "NEVER"}},
    )
    assert list(llm_adapters.get("anthropic").stream(stream)) == [("reasoning", "plan"), ("content", "A"), ("content", "B")]
    with pytest.raises(llm_client.LLMRequestError):
        list(llm_adapters.get("anthropic").stream(sse({"type": "error", "error": {"type": "overloaded_error"}})))


# ---------- Gemini ----------


def test_gemini_request_shape():
    r = llm_adapters.get("gemini").request(route("gemini", "https://generativelanguage.googleapis.com", "models/gemini-x"), MESSAGES, False)
    assert r.url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"
    assert r.headers["x-goog-api-key"] == "sk-secret"
    p = r.payload
    assert p["systemInstruction"] == {"parts": [{"text": "SYS RULES"}]}
    assert [c["role"] for c in p["contents"]] == ["user", "model", "user"] and p["contents"][1]["parts"] == [{"text": "Earlier answer"}]
    assert p["generationConfig"]["temperature"] == 0.0
    streaming = llm_adapters.get("gemini").request(route("gemini", "https://g.example", "gemini-x"), MESSAGES, True)
    assert streaming.url.endswith(":streamGenerateContent?alt=sse")


def test_gemini_parse_skips_thought_parts_and_streams_them_as_reasoning():
    parts = {"candidates": [{"content": {"parts": [{"text": "secret thoughts", "thought": True}, {"text": "Final "}, {"text": "answer"}]}}]}
    assert llm_adapters.get("gemini").parse(parts) == "Final answer"
    assert llm_adapters.get("gemini").parse({"candidates": []}) == ""
    stream = sse(parts, {"candidates": [{"content": {"parts": [{"text": " more"}]}}]})
    assert list(llm_adapters.get("gemini").stream(stream)) == [("reasoning", "secret thoughts"), ("content", "Final "), ("content", "answer"), ("content", " more")]
    with pytest.raises(llm_client.LLMRequestError):
        list(llm_adapters.get("gemini").stream(sse({"error": {"code": 503}})))


# ---------- Cohere ----------


def test_cohere_request_parse_and_stream():
    adapter = llm_adapters.get("cohere")
    r = adapter.request(route("cohere", "https://api.cohere.com", "command-x"), MESSAGES, True)
    assert r.url == "https://api.cohere.com/v2/chat" and r.headers["Authorization"] == "Bearer sk-secret"
    assert r.payload["messages"] == MESSAGES and r.payload["stream"] is True
    assert adapter.parse({"message": {"content": [{"type": "text", "text": "ok"}]}}) == "ok"
    stream = sse(
        {"type": "content-delta", "delta": {"message": {"content": {"text": "He"}}}},
        {"type": "content-delta", "delta": {"message": {"content": {"text": "llo"}}}},
        {"type": "message-end"},
    )
    assert list(adapter.stream(stream)) == [("content", "He"), ("content", "llo")]


def test_unknown_api_names_fall_back_to_openai():
    assert llm_adapters.get("nonsense").name == "openai"


# ---------- end to end through llm_client, every family ----------


def fake_requests(monkeypatch, replies):
    """replies: host-substring -> (status, json). Records every call."""
    calls = []

    class Resp:
        def __init__(self, status, data):
            self.status_code, self._data, self.headers, self.text = status, data, {}, ""

        def json(self):
            return self._data

        def close(self):
            pass

    def post(url, json, headers, timeout, stream=False):
        calls.append({"url": url, "payload": json, "headers": headers})
        for host, (status, data) in replies.items():
            if host in url:
                return Resp(status, data)
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(llm_client.requests, "post", post)
    return calls


@pytest.fixture
def clean_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.endswith(("_API_KEY", "_MODEL", "_BASE_URL")) or name in ("LLM_PROVIDERS", "LLM_ROUTES", "HF_TOKEN", "LLM_FALLBACK_MODEL"):
            monkeypatch.delenv(name, raising=False)
    providers.reset()
    return monkeypatch


@pytest.mark.parametrize(
    "provider,key_var,host,reply",
    [
        ("anthropic", "ANTHROPIC_API_KEY", "anthropic.com/v1/messages", {"content": [{"type": "text", "text": "claude says hi"}]}),
        ("gemini", "GEMINI_API_KEY", "generativelanguage.googleapis.com/v1beta/models/", {"candidates": [{"content": {"parts": [{"text": "gemini says hi"}]}}]}),
        ("cohere", "COHERE_API_KEY", "cohere.com/v2/chat", {"message": {"content": [{"type": "text", "text": "cohere says hi"}]}}),
        ("openai", "OPENAI_API_KEY", "api.openai.com/v1/chat/completions", {"choices": [{"message": {"content": "gpt says hi"}}]}),
        ("mistral", "MISTRAL_API_KEY", "api.mistral.ai/v1/chat/completions", {"choices": [{"message": {"content": "mistral says hi"}}]}),
    ],
)
def test_every_family_answers_through_the_same_client(clean_env, provider, key_var, host, reply):
    clean_env.setenv("LLM_PROVIDERS", provider)
    clean_env.setenv(key_var, "k-test")
    calls = fake_requests(clean_env, {host: (200, reply)})
    answer = llm_client.ask("What?", [{"text": "passage", "page": 1}])
    assert str(answer).endswith("says hi") and len(calls) == 1
    assert llm_client.answered_by(answer)["model"]


def test_a_failing_provider_fails_over_to_a_different_api_family(clean_env):
    clean_env.setenv("LLM_PROVIDERS", "groq,anthropic")
    clean_env.setenv("GROQ_API_KEY", "g")
    clean_env.setenv("ANTHROPIC_API_KEY", "a")
    calls = fake_requests(
        clean_env,
        {"groq.com": (500, {}), "anthropic.com": (200, {"content": [{"type": "text", "text": "claude to the rescue"}]})},
    )
    answer = llm_client.ask("What?", [{"text": "p", "page": 1}])
    assert str(answer) == "claude to the rescue" and llm_client.answered_by(answer)["provider"] == "Anthropic"
    assert [c["url"].split("/")[2] for c in calls] == ["api.groq.com", "api.anthropic.com"]


def test_the_streaming_path_works_for_a_non_openai_family(clean_env):
    clean_env.setenv("LLM_PROVIDERS", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "a")
    fake = sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "streamed "}},
               {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "claude"}}, {"type": "message_stop"})
    fake.status_code = 200
    clean_env.setattr(llm_client.requests, "post", lambda url, json, headers, timeout, stream=False: fake)
    pieces = list(llm_client.ask_stream("What?", [{"text": "p", "page": 1}]))
    assert "".join(t for k, t in pieces if k == "content") == "streamed claude"
    assert llm_client.answered_by(pieces[0][1])["provider"] == "Anthropic"


# ---------- provider configuration ----------


def test_the_nemotron_example_is_one_environment_variable(clean_env):
    clean_env.setenv("LLM_PROVIDERS", "nvidia")
    clean_env.setenv("NVIDIA_API_KEY", "k")
    clean_env.setenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-ultra-253b-v1")
    (r,) = providers.routes()
    assert r.model == "nvidia/llama-3.1-nemotron-ultra-253b-v1" and r.base_url == "https://integrate.api.nvidia.com/v1" and r.api == "openai"


def test_several_models_for_one_provider_become_a_chain(clean_env):
    clean_env.setenv("LLM_PROVIDERS", "nvidia")
    clean_env.setenv("NVIDIA_API_KEY", "k")
    clean_env.setenv("NVIDIA_MODEL", "nvidia/nemotron-a, openai/gpt-oss-120b ,nvidia/nemotron-a")
    assert [r.model for r in providers.routes()] == ["nvidia/nemotron-a", "openai/gpt-oss-120b"]
    assert len({r.id for r in providers.routes()}) == 2  # each has its own cooldown


def test_presets_need_their_key_and_the_local_server_needs_a_model(clean_env):
    clean_env.setenv("LLM_PROVIDERS", "anthropic,openai,ollama,deepseek")
    assert providers.routes() == []
    clean_env.setenv("DEEPSEEK_API_KEY", "k")
    clean_env.setenv("OLLAMA_MODEL", "llama3.2")
    assert [(r.provider, r.model, r.api) for r in providers.routes()] == [("ollama", "llama3.2", "openai"), ("deepseek", "deepseek-chat", "openai")] or \
        [(r.provider, r.model) for r in providers.routes()] == [("ollama", "llama3.2"), ("deepseek", "deepseek-chat")]
    local = next(r for r in providers.routes() if r.provider == "ollama")
    assert local.api_key == "" and local.base_url == "http://localhost:11434/v1"


def test_llm_routes_adds_any_endpoint_without_code(clean_env):
    clean_env.setenv("LLM_PROVIDERS", "groq")
    clean_env.setenv("GROQ_API_KEY", "g")
    clean_env.setenv("MY_KEY", "secret-from-env")
    clean_env.setenv(
        "LLM_ROUTES",
        json.dumps(
            [
                {"name": "nemotron", "api": "openai", "base_url": "https://integrate.api.nvidia.com/v1", "model": "nvidia/nemotron-x", "key_env": "MY_KEY", "label": "Nemotron"},
                {"name": "claude-eu", "api": "anthropic", "base_url": "https://eu.example.com", "model": "claude-x", "key_env": "MY_KEY", "headers": {"X-Team": "a"}},
                {"name": "azure", "api": "openai", "base_url": "https://r.openai.azure.com/openai/deployments/d", "model": "d", "key_env": "MY_KEY", "auth": "api-key", "query": {"api-version": "2024-06-01"}},
                {"name": "local", "api": "openai", "base_url": "http://127.0.0.1:8000/v1", "model": "qwen"},
            ]
        ),
    )
    chain = providers.routes()
    assert [r.provider for r in chain] == ["groq", "nemotron", "claude-eu", "azure", "local"]
    by = {r.provider: r for r in chain}
    assert by["nemotron"].label == "Nemotron" and by["nemotron"].api_key == "secret-from-env"
    assert by["claude-eu"].api == "anthropic" and ("X-Team", "a") in by["claude-eu"].headers
    assert by["azure"].auth == "api-key" and by["azure"].query == (("api-version", "2024-06-01"),)
    assert by["local"].api_key == ""


def test_a_custom_route_without_its_key_is_skipped_not_called_without_credentials(clean_env):
    clean_env.setenv("LLM_ROUTES", json.dumps([{"name": "x", "base_url": "https://a.example/v1", "model": "m", "key_env": "NOT_SET_ANYWHERE"}]))
    assert providers.routes() == []


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "BAD NAME", "base_url": "https://a.example/v1", "model": "m", "key_env": "K"},
        {"name": "ok", "base_url": "ftp://a.example", "model": "m", "key_env": "K"},
        {"name": "ok", "base_url": "https://user:pass@a.example/v1", "model": "m", "key_env": "K"},
        {"name": "ok", "base_url": "https://a.example/v1", "model": "", "key_env": "K"},
        {"name": "ok", "base_url": "https://a.example/v1", "model": "m", "key_env": "K", "api": "telepathy"},
        {"name": "ok", "base_url": "https://a.example/v1", "model": "m", "key_env": "bad key!"},
        {"name": "ok", "base_url": "https://a.example/v1", "model": "m", "key_env": "K", "auth": "magic"},
        {"name": "ok", "base_url": "https://a.example/v1", "model": "m", "key_env": "K", "headers": ["x"]},
        {"name": "groq", "base_url": "https://a.example/v1", "model": "m", "key_env": "K"},  # shadows a built-in
        "just a string",
    ],
)
def test_malformed_route_entries_are_ignored(clean_env, entry):
    clean_env.setenv("K", "v")
    clean_env.setenv("LLM_ROUTES", json.dumps([entry]))
    assert providers.routes() == []


@pytest.mark.parametrize("raw", ["not json", "{}", "null", "x" * 30000])
def test_malformed_routes_json_is_ignored(clean_env, raw):
    clean_env.setenv("LLM_ROUTES", raw)
    assert providers.routes() == []


def test_duplicate_custom_names_keep_only_the_first(clean_env):
    clean_env.setenv("K", "v")
    entry = {"base_url": "https://a.example/v1", "key_env": "K", "name": "dup"}
    clean_env.setenv("LLM_ROUTES", json.dumps([{**entry, "model": "first"}, {**entry, "model": "second"}]))
    assert [r.model for r in providers.routes()] == ["first"]


def test_keys_never_appear_in_repr_status_or_log_output(clean_env, caplog):
    clean_env.setenv("LLM_PROVIDERS", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret")
    clean_env.setenv("LLM_ROUTES", "{bad json with sk-ant-very-secret")
    text = repr(providers.routes()) + repr(providers.status()) + caplog.text
    assert "sk-ant-very-secret" not in text


def test_the_old_environment_variables_still_configure_groq(clean_env):
    clean_env.setenv("LLM_API_KEY", "legacy")
    clean_env.setenv("LLM_MODEL", "legacy-model")
    (r,) = providers.routes()
    assert (r.provider, r.model, r.api_key, r.api) == ("groq", "legacy-model", "legacy", "openai")
