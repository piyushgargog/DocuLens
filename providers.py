"""Several LLM providers behind one failover chain, for any model.

Each provider speaks one chat-API family (`api`, see llm_adapters.py): `openai`
(the `/chat/completions` shape that Groq, OpenRouter, NVIDIA, Hugging Face,
OpenAI, Mistral, DeepSeek, xAI, Together, Azure, Ollama and vLLM all serve),
`anthropic` (Claude), `gemini` (Google's native API) or `cohere`. Free tiers are
small (Groq's is 200k tokens a day per model), so one provider alone runs out;
with a chain, the app keeps answering.

A provider is used only when its API key is set. `LLM_PROVIDERS` sets the
order (default: groq, openrouter, nvidia, huggingface), `<NAME>_MODEL` and
`<NAME>_BASE_URL` override the defaults, and `LLM_FALLBACK_MODEL` adds a
second Groq model at the end of the chain (each Groq model has its own
quota). The older single-provider settings (`LLM_API_KEY`, `LLM_BASE_URL`,
`LLM_MODEL`) still configure the first slot, so existing .env files keep
working.

Any model works without code changes:

* `<NAME>_MODEL` may list several models, comma separated: each becomes its own
  route in the chain (e.g. `NVIDIA_MODEL=nvidia/llama-3.1-nemotron-ultra-253b-v1,openai/gpt-oss-120b`).
* Presets exist for the common services (see KNOWN); add one to LLM_PROVIDERS to
  use it once its key is set.
* `LLM_ROUTES` is a JSON list for anything else -- a different endpoint, another
  model family, a local server, Azure -- e.g.
  `[{"name":"nemotron","api":"openai","base_url":"https://integrate.api.nvidia.com/v1",
     "model":"nvidia/llama-3.1-nemotron-ultra-253b-v1","key_env":"NVIDIA_API_KEY"}]`.
  Fields: name, api (openai|anthropic|gemini|cohere), base_url, model, key_env
  (the NAME of the environment variable holding the key; omit for keyless local
  servers), label, auth (bearer|x-api-key|api-key|x-goog-api-key|none),
  headers (object), query (object). Routes listed there join the chain after the
  ones in LLM_PROVIDERS unless their name is listed in it.

A provider that fails is put on a short cooldown, so the next questions go
straight to one that works instead of waiting on it again:
rate limits for as long as the provider asks (capped), a bad key, used-up
credits or an unknown model for 15 minutes, timeouts and server errors for 30 seconds.
"""

import json
import os
import logging
import re
import threading
import time
from typing import Any
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

import llm_adapters

# Google AI Studio is supported but opt-in (add "google" to LLM_PROVIDERS):
# in testing on 2026-09-29 its free tier was overloaded (503) or rate
# limited on most calls, so it could not be verified against the
# prompt-injection tests the other providers pass.
DEFAULT_ORDER = "groq,openrouter,nvidia,huggingface"

KNOWN: dict[str, dict[str, Any]] = {
    # Google AI Studio (Gemini API) through its OpenAI-compatible endpoint.
    "google": {
        "label": "Google AI Studio",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash",
        "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "keys": ("GROQ_API_KEY", "LLM_API_KEY"),
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        # gpt-oss-120b is paid-only on OpenRouter now; this free 120B model
        # answered as well in testing (2026-09-29).
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "keys": ("OPENROUTER_API_KEY",),
        # Optional attribution headers OpenRouter asks apps to send.
        "headers": (
            ("HTTP-Referer", "https://doculens.duckdns.org"),
            ("X-Title", "DocuLens"),
        ),
    },
    "nvidia": {
        "label": "NVIDIA",
        "base_url": "https://integrate.api.nvidia.com/v1",
        # gpt-oss-120b reached end of life on NVIDIA's API on 2026-09-03.
        "model": "openai/gpt-oss-20b",
        "keys": ("NVIDIA_API_KEY",),
    },
    "huggingface": {
        "label": "Hugging Face",
        "base_url": "https://router.huggingface.co/v1",
        "model": "openai/gpt-oss-120b",
        "keys": ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
    },
    # ---- opt-in presets: add the name to LLM_PROVIDERS and set the key ----
    # Other chat-API families, each with its own adapter.
    "anthropic": {
        "label": "Anthropic",
        "api": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-5-5",
        "keys": ("ANTHROPIC_API_KEY",),
    },
    "gemini": {
        "label": "Google Gemini",
        "api": "gemini",
        "base_url": "https://generativelanguage.googleapis.com",
        "model": "gemini-3.6-flash",
        "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
    "cohere": {
        "label": "Cohere",
        "api": "cohere",
        "base_url": "https://api.cohere.com",
        "model": "command-a-03-2025",
        "keys": ("COHERE_API_KEY",),
    },
    # OpenAI-compatible services (override the model with <NAME>_MODEL).
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-5", "keys": ("OPENAI_API_KEY",)},
    "mistral": {"label": "Mistral", "base_url": "https://api.mistral.ai/v1", "model": "mistral-large-latest", "keys": ("MISTRAL_API_KEY",)},
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat", "keys": ("DEEPSEEK_API_KEY",)},
    "xai": {"label": "xAI", "base_url": "https://api.x.ai/v1", "model": "grok-4", "keys": ("XAI_API_KEY",)},
    "together": {
        "label": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "keys": ("TOGETHER_API_KEY",),
    },
    # A local server (Ollama, vLLM, LM Studio): no key, used once OLLAMA_MODEL is set.
    "ollama": {"label": "Local (Ollama)", "base_url": "http://localhost:11434/v1", "model": "", "keys": (), "keyless": True},
}

_log = logging.getLogger("doculens.providers")

RATE_LIMIT_COOLDOWN_DEFAULT = 60.0
RATE_LIMIT_COOLDOWN_MAX = 900.0
BROKEN_COOLDOWN = 900.0  # bad key, no credits, unknown or retired model
FLAKY_COOLDOWN = 30.0  # timeout, connection error, 5xx


@dataclass(frozen=True)
class Route:
    """One provider + model the app can send a chat completion to."""

    provider: str
    label: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    headers: tuple = ()
    api: str = "openai"  # which chat-API family: see llm_adapters.ADAPTERS
    auth: str = ""  # how the key is sent; "" = the family's default
    query: tuple = ()  # extra URL query parameters, e.g. Azure's api-version

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}"

    def describe(self) -> dict:
        """What the UI shows about who answered (never the key)."""
        return {"provider": self.label, "model": self.model}


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,29}")
_KEY_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_AUTHS = ("", "bearer", "x-api-key", "api-key", "x-goog-api-key", "none")
MAX_ROUTES_JSON = 20_000


def _custom_specs() -> dict[str, dict[str, Any]]:
    """Providers declared in LLM_ROUTES (JSON). Anything malformed is skipped
    with a warning that names the problem but never prints the value."""
    raw = _env("LLM_ROUTES")
    if not raw:
        return {}
    if len(raw) > MAX_ROUTES_JSON:
        _log.warning("LLM_ROUTES ignored: too large")
        return {}
    try:
        entries = json.loads(raw)
    except ValueError:
        _log.warning("LLM_ROUTES ignored: not valid JSON")
        return {}
    if not isinstance(entries, list):
        _log.warning("LLM_ROUTES ignored: expected a JSON list")
        return {}
    specs: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            _log.warning("LLM_ROUTES entry #%d ignored: not an object", position)
            continue
        name = str(entry.get("name", "")).strip().lower()
        api = str(entry.get("api", "openai")).strip().lower()
        base_url = str(entry.get("base_url", "")).strip().rstrip("/")
        model = str(entry.get("model", "")).strip()
        key_env = str(entry.get("key_env", "")).strip()
        auth = str(entry.get("auth", "")).strip().lower()
        parts = urlsplit(base_url)
        problems = []
        if not _NAME_RE.fullmatch(name):
            problems.append("name")
        if api not in llm_adapters.ADAPTERS:
            problems.append("api")
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            problems.append("base_url")
        if not model:
            problems.append("model")
        if key_env and not _KEY_ENV_RE.fullmatch(key_env):
            problems.append("key_env")
        if auth not in _AUTHS:
            problems.append("auth")
        headers, query = entry.get("headers", {}), entry.get("query", {})
        if not isinstance(headers, dict) or not isinstance(query, dict):
            problems.append("headers/query")
        if problems or name in KNOWN or name in specs:
            _log.warning("LLM_ROUTES entry #%d ignored: bad or duplicate %s", position, ", ".join(problems) or "name")
            continue
        specs[name] = {
            "label": str(entry.get("label") or name)[:40],
            "api": api,
            "base_url": base_url,
            "model": model,
            "keys": (key_env,) if key_env else (),
            "keyless": not key_env,
            "auth": auth,
            "headers": tuple((str(k), str(v)) for k, v in headers.items()),
            "query": tuple((str(k), str(v)) for k, v in query.items()),
        }
    return specs


def _models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def routes() -> list[Route]:
    """The configured chain, in order. Read from the environment on every
    call, so tests (and a restarted container) see changes immediately."""
    custom = _custom_specs()
    known: dict[str, dict[str, Any]] = {**KNOWN, **custom}
    order = [p.strip().lower() for p in (_env("LLM_PROVIDERS") or DEFAULT_ORDER).split(",") if p.strip()]
    order += [name for name in custom if name not in order]  # LLM_ROUTES entries join the end
    chain = []
    for name in dict.fromkeys(order):
        spec = known.get(name)
        if spec is None:
            continue
        key = next((_env(k) for k in spec["keys"] if _env(k)), "")
        prefix = re.sub(r"[^A-Z0-9]", "_", name.upper())
        if not key and not spec.get("keyless"):
            continue
        legacy = name == "groq"
        base_url = (_env(f"{prefix}_BASE_URL") or (legacy and _env("LLM_BASE_URL")) or spec["base_url"]).rstrip("/")
        models = _models(_env(f"{prefix}_MODEL") or (legacy and _env("LLM_MODEL")) or spec["model"])
        if not models:
            continue  # e.g. the local-server preset until its model is named
        label = spec["label"]
        if legacy and "groq.com" not in base_url:
            # LLM_BASE_URL pointed at some other OpenAI-compatible server.
            label = urlsplit(base_url).hostname or "Custom"
        for model in dict.fromkeys(models):
            chain.append(
                Route(
                    name,
                    label,
                    base_url,
                    model,
                    key,
                    spec.get("headers", ()),
                    spec.get("api", "openai"),
                    spec.get("auth", ""),
                    spec.get("query", ()),
                )
            )

    fallback = _env("LLM_FALLBACK_MODEL")
    first = next((r for r in chain if r.provider == "groq"), None)
    if fallback and first is not None and fallback != first.model:
        chain.append(replace(first, model=fallback))
    return chain


# --- Cooldowns -----------------------------------------------------------------

_lock = threading.Lock()
_cooldown_until: dict[str, float] = {}
_last_error: dict[str, str] = {}


def reset() -> None:
    with _lock:
        _cooldown_until.clear()
        _last_error.clear()


def cooling(route: Route) -> bool:
    with _lock:
        return _cooldown_until.get(route.id, 0.0) > time.monotonic()


def candidates() -> list[Route]:
    """Routes to try, in order: those not cooling down first. When every
    route is cooling down they are all tried anyway -- a provider may have
    recovered early, and trying beats refusing outright."""
    chain = routes()
    ready = [r for r in chain if not cooling(r)]
    return ready + [r for r in chain if r not in ready]


def cool_down(route: Route, seconds: float, reason: str) -> None:
    with _lock:
        _cooldown_until[route.id] = time.monotonic() + seconds
        _last_error[route.id] = reason
    _log.warning("Provider %s (%s) unavailable: %s; skipping for %.0fs", route.label, route.model, reason, seconds)


def mark_ok(route: Route) -> None:
    with _lock:
        _cooldown_until.pop(route.id, None)
        _last_error.pop(route.id, None)


def status() -> list[dict]:
    """Per configured route: who it is and whether it's usable right now.
    No keys, URLs or error details -- this is shown on the public page."""
    now = time.monotonic()
    out = []
    for route in routes():
        with _lock:
            until = _cooldown_until.get(route.id, 0.0)
        out.append({**route.describe(), "state": "cooling" if until > now else "ready"})
    return out
