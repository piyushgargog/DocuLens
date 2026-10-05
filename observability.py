"""Structured logging, per-request IDs, and optional error reporting.

Replaces scattered print() calls with a single configured logger so the
deployed service has readable, greppable, level-filtered logs that carry the
id of the request they belong to. Error tracking (Sentry) turns on only when
SENTRY_DSN is set, so nothing is sent anywhere by default and the privacy
posture is unchanged.

Configuration (all optional, via environment):
  LOG_LEVEL          DEBUG|INFO|WARNING|ERROR            (default INFO)
  SENTRY_DSN         enable Sentry error reporting when set
  SENTRY_TRACES_RATE float 0..1 performance sampling     (default 0)
  APP_ENV            environment tag for Sentry          (default production)
"""

import contextvars
import logging
import os
import secrets

# The id of the request currently being handled, injected into every log line.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

log = logging.getLogger("doculens")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def new_request_id() -> str:
    return secrets.token_hex(6)


_logging_configured = False


def setup_logging() -> None:
    """Configure the `doculens` logger once, with a request-id-aware format.
    Idempotent so imports/tests/reloads don't stack handlers."""
    global _logging_configured
    if _logging_configured:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s")
    )
    handler.addFilter(_RequestIdFilter())
    log.setLevel(getattr(logging, level, logging.INFO))
    log.addHandler(handler)
    log.propagate = False
    _logging_configured = True


def init_sentry(release: str | None = None) -> bool:
    """Enable Sentry only if SENTRY_DSN is set and the SDK is installed.
    Returns True when active. No DSN => no network, no data leaves the box."""
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
    except Exception:
        log.warning("SENTRY_DSN is set but sentry-sdk is not installed; error reporting is off")
        return False
    try:
        rate = float(os.environ.get("SENTRY_TRACES_RATE", "0") or 0)
    except ValueError:
        rate = 0.0
    sentry_sdk.init(
        dsn=dsn,
        release=release,
        environment=os.environ.get("APP_ENV", "production"),
        traces_sample_rate=rate,
        # Don't attach request bodies / headers: questions and passages are
        # the user's document content and must not be shipped to a third party.
        send_default_pii=False,
    )
    log.info("Sentry error reporting enabled")
    return True
