"""Browser end-to-end fixtures: a real uvicorn server and a Playwright browser.

These tests are opt-in (DOCULENS_E2E=1, see the root conftest.py) so the plain
`pytest` run never needs a browser. They need no LLM key: the server starts with
every provider key blanked, and the LLM-backed endpoints are faked in the
browser with `page.route`, while upload/ingest runs for real.

Environment knobs:
    DOCULENS_E2E_PORT     port for the test server (default 8031)
    DOCULENS_E2E_PYTHON   interpreter that has the app's requirements
                          (default: the one running pytest)
    DOCULENS_E2E_CHANNEL  Playwright browser channel, e.g. "chrome" locally;
                          unset = Playwright's bundled Chromium (CI)
"""

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PORT = int(os.environ.get("DOCULENS_E2E_PORT", "8031"))
BASE_URL = f"http://127.0.0.1:{PORT}"

# Every provider key the app reads. Blank values win over a local .env
# (python-dotenv never overrides variables that are already set).
PROVIDER_KEYS = (
    "LLM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
    "NVIDIA_API_KEY", "HF_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY", "COHERE_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY", "XAI_API_KEY", "TOGETHER_API_KEY", "LLM_ROUTES",
    # shared state and sign-in: the e2e server must be self-contained, never
    # touching a developer's Redis or Firebase project from a local .env
    "REDIS_URL", "FIREBASE_PROJECT_ID", "FIREBASE_API_KEY", "FIREBASE_AUTH_DOMAIN",
)


@pytest.fixture(scope="session")
def app_version() -> str:
    """APP_VERSION read from main.py's source, so these tests don't import the app."""
    match = re.search(r'^APP_VERSION = "([^"]+)"', (ROOT / "main.py").read_text("utf-8"), re.M)
    assert match, "APP_VERSION not found in main.py"
    return match.group(1)


@pytest.fixture(scope="session")
def server():
    env = dict(os.environ, PREFETCH_MODEL="1", LOG_LEVEL="WARNING")
    env.update({key: "" for key in PROVIDER_KEYS})
    python = os.environ.get("DOCULENS_E2E_PYTHON", sys.executable)
    proc = subprocess.Popen(
        [python, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 180  # first run downloads the embedding model
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"test server exited early (code {proc.returncode})")
            try:
                with urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.time() > deadline:
                raise RuntimeError("test server did not become healthy in time")
            time.sleep(0.5)
        yield BASE_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def browser():
    channel = os.environ.get("DOCULENS_E2E_CHANNEL") or None
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        yield browser
        browser.close()


@pytest.fixture
def make_page(browser):
    """make_page(reduced_motion=...) -> (page, console_errors list)."""
    contexts = []

    def _make(reduced_motion="no-preference"):
        context = browser.new_context(reduced_motion=reduced_motion, viewport={"width": 1280, "height": 800})
        contexts.append(context)
        page = context.new_page()
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        return page, errors

    yield _make
    for context in contexts:
        context.close()
