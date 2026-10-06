import pytest

import auth
from firebase_helpers import PROJECT, FakeCerts


@pytest.fixture
def signin_on(monkeypatch):
    """Firebase sign-in configured, Google's certificates served by a fake."""
    monkeypatch.setenv("FIREBASE_PROJECT_ID", PROJECT)
    monkeypatch.setenv("FIREBASE_API_KEY", "web-api-key")
    fake = FakeCerts()
    monkeypatch.setattr(auth.requests, "get", lambda url, timeout: fake.get())
    auth.certs.clear()
    yield fake
    auth.certs.clear()
