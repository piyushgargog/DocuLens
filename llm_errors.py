"""Exceptions shared by llm_client.py and llm_adapters.py (kept apart so the
adapters can raise them without importing the client)."""


class LLMConfigError(RuntimeError):
    """Raised when required LLM configuration (e.g. API key) is missing."""


class LLMRequestError(RuntimeError):
    """Raised when the LLM API call itself fails."""

    status: int | None = None  # the HTTP status, when the provider answered


class LLMRateLimitError(LLMRequestError):
    """The provider kept answering 429, or asked to wait longer than we hold for
    (e.g. a daily token quota is used up)."""

    wait: float | None = None  # seconds the provider asked us to wait, if it said
