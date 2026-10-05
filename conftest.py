import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def pytest_ignore_collect(collection_path, config):
    """Browser end-to-end tests (tests/e2e) need Playwright and a browser, so
    they run only when asked: DOCULENS_E2E=1 pytest tests/e2e"""
    if "e2e" in collection_path.parts and os.environ.get("DOCULENS_E2E") != "1":
        return True
    return None


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """Bytes of the bundled minimal demo PDF (sample_docs/sample.pdf)."""
    return (ROOT / "sample_docs" / "sample.pdf").read_bytes()


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Rate-limit counters and logins live in the process-wide store; give every test a clean slate."""
    import main
    import providers
    import store

    main._store = store.MemoryStore()
    main._user_sessions.clear()
    main._sessions.clear()
    providers.reset()
    yield
    main._store = store.MemoryStore()
    main._user_sessions.clear()
    main._sessions.clear()
    providers.reset()
