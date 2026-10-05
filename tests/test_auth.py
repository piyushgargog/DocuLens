"""Firebase sign-in and the guest / signed-in limits.

Tokens are real RS256 JWTs signed with a throwaway key; the key's self-signed
certificate stands in for Google's published one, so signature, audience,
issuer and expiry are all genuinely verified. Nothing touches the network."""

import time

import jwt
import pytest
from fastapi.testclient import TestClient

import auth
import main
from firebase_helpers import KEY, KID, OTHER_KEY, PROJECT, TXT, make_token

@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def sign_in(client, token=None):
    return client.post("/api/login", json={"id_token": make_token() if token is None else token})


def fake_answer(question, states, history=None):
    return {"answer": "ok", "sources": []}


# ---------- off by default ----------


def test_sign_in_is_off_without_configuration(client, monkeypatch):
    monkeypatch.delenv("FIREBASE_PROJECT_ID", raising=False)
    assert client.get("/api/me").json() == {
        "auth_enabled": False,
        "firebase": None,
        "turnstile_site_key": None,
        "human": True,
        "user": None,
        "limits": None,
    }
    assert sign_in(client).status_code == 503
    # and nothing is limited: two documents at once, as before
    assert client.post("/api/ingest", files=TXT).status_code == 200
    assert client.post("/api/ingest", files=TXT).status_code == 200
    assert "apis.google.com" not in client.get("/").headers["content-security-policy"]


# ---------- sign-in ----------


def test_me_exposes_only_the_public_firebase_config(client, signin_on):
    me = client.get("/api/me").json()
    assert me["firebase"] == {
        "apiKey": "web-api-key",
        "authDomain": f"{PROJECT}.firebaseapp.com",
        "projectId": PROJECT,
    }
    assert me["limits"]["tier"] == "guest"


def test_valid_token_signs_in_then_logout_ends_it(client, signin_on):
    response = sign_in(client)
    assert response.status_code == 200
    assert response.json()["user"] == {"name": "Ada", "email": "ada@example.com"}
    assert "HttpOnly" in response.headers["set-cookie"] and "SameSite=lax" in response.headers["set-cookie"]

    me = client.get("/api/me").json()
    assert me["user"] == {"name": "Ada", "email": "ada@example.com"}
    assert me["limits"]["tier"] == "user" and me["limits"]["max_docs"] == main.MAX_DOCS_PER_SESSION
    assert "uid-123" not in str(me)

    assert client.post("/api/logout").json() == {"ok": True}
    assert client.get("/api/me").json()["user"] is None


def test_certificates_are_cached_between_sign_ins(client, signin_on):
    sign_in(client)
    sign_in(client)
    assert signin_on.fetches == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"aud": "someone-elses-project"},
        {"iss": "https://securetoken.google.com/someone-elses-project"},
        {"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200},
        {"email_verified": False},
        {"email": None},
        {"sub": None},
    ],
)
def test_bad_claims_are_rejected(client, signin_on, overrides):
    response = sign_in(client, make_token(**overrides))
    assert response.status_code == 401
    assert client.get("/api/me").json()["user"] is None


def test_token_signed_by_another_key_is_rejected(client, signin_on):
    assert sign_in(client, make_token(key=OTHER_KEY)).status_code == 401


def test_unsigned_and_hs256_tokens_are_rejected(client, signin_on):
    unsigned = jwt.encode({"aud": PROJECT, "sub": "x"}, key=None, algorithm="none", headers={"kid": KID})
    hs256 = jwt.encode({"aud": PROJECT, "sub": "x"}, "secret" * 8, algorithm="HS256", headers={"kid": KID})
    for token in (unsigned, hs256, "not-a-jwt", ""):
        assert sign_in(client, token).status_code == 401
    assert client.post("/api/login", json={}).status_code == 401


def test_unknown_key_id_is_rejected(client, signin_on):
    now = int(time.time())
    claims = {"iss": f"https://securetoken.google.com/{PROJECT}", "aud": PROJECT, "sub": "u", "iat": now, "exp": now + 60}
    token = jwt.encode(claims, KEY, algorithm="RS256", headers={"kid": "rotated-away"})
    assert sign_in(client, token).status_code == 401


def test_google_certificate_outage_is_a_clean_401(client, signin_on, monkeypatch):
    def boom(url, timeout):
        raise auth.requests.ConnectionError("down")

    monkeypatch.setattr(auth.requests, "get", boom)
    auth.certs.clear()
    assert sign_in(client).status_code == 401


def test_login_rejects_cross_site_posts(client, signin_on):
    response = client.post("/api/login", json={"id_token": make_token()}, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_login_is_rate_limited(client, signin_on, monkeypatch):
    monkeypatch.setitem(main.RATE_LIMITS, "login", [(2, 600)])
    for _ in range(2):
        sign_in(client, "x" * 30)
    assert sign_in(client, "x" * 30).status_code == 429


# ---------- headers needed by the Firebase SDK ----------


def test_headers_allow_only_what_the_firebase_popup_needs(client, signin_on):
    headers = client.get("/").headers
    csp = headers["content-security-policy"]
    assert "script-src 'self' https://apis.google.com;" in csp
    assert "connect-src 'self' https://identitytoolkit.googleapis.com https://securetoken.googleapis.com;" in csp
    assert f"frame-src https://{PROJECT}.firebaseapp.com https://accounts.google.com" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "frame-ancestors 'none'" in csp
    assert headers["cross-origin-opener-policy"] == "same-origin-allow-popups"


# ---------- tiers ----------


def test_guest_may_load_only_one_document(client, signin_on):
    assert client.post("/api/ingest", files=TXT).status_code == 200
    second = client.post("/api/ingest", files=TXT)
    assert second.status_code == 400
    assert "sign in with Google" in second.json()["error"]


def test_signed_in_user_may_load_several_documents(client, signin_on):
    sign_in(client)
    for _ in range(3):
        assert client.post("/api/ingest", files=TXT).status_code == 200


def test_guest_uploads_are_capped_per_day_even_after_removing(client, signin_on, monkeypatch):
    monkeypatch.setenv("GUEST_DAILY_UPLOADS", "2")
    for _ in range(2):
        assert client.post("/api/ingest", files=TXT).status_code == 200
        client.post("/api/remove", json={})
    third = client.post("/api/ingest", files=TXT)
    assert third.status_code == 429
    assert "Guest limit reached" in third.json()["error"]


def test_guest_questions_are_capped_per_ip_per_day(client, signin_on, monkeypatch):
    monkeypatch.setenv("GUEST_DAILY_QUESTIONS", "2")
    monkeypatch.setattr(main.pipeline, "answer", fake_answer)
    client.post("/api/ingest", files=TXT)
    for _ in range(2):
        assert client.post("/api/ask", json={"question": "what?"}).status_code == 200
    blocked = client.post("/api/ask", json={"question": "what?"})
    assert blocked.status_code == 429
    assert "Guest limit reached (2 questions per day)" in blocked.json()["error"]
    assert "Retry-After" in blocked.headers

    # a fresh browser (no cookies) on the same IP shares the guest budget
    other = TestClient(main.app)
    other.post("/api/ingest", files=TXT)
    assert other.post("/api/ask", json={"question": "what?"}).status_code == 429


def test_signing_in_lifts_the_guest_question_cap(client, signin_on, monkeypatch):
    monkeypatch.setenv("GUEST_DAILY_QUESTIONS", "1")
    monkeypatch.setattr(main.pipeline, "answer", fake_answer)
    client.post("/api/ingest", files=TXT)
    assert client.post("/api/ask", json={"question": "a?"}).status_code == 200
    assert client.post("/api/ask", json={"question": "b?"}).status_code == 429
    sign_in(client)
    assert client.post("/api/ask", json={"question": "c?"}).status_code == 200


def test_logout_also_drops_loaded_documents(client, signin_on):
    sign_in(client)
    client.post("/api/ingest", files=TXT)
    assert client.get("/api/session").json()["documents"]
    client.post("/api/logout")
    assert client.get("/api/session").json()["documents"] == []


def test_logout_rejects_cross_site_posts(client, signin_on):
    response = client.post("/api/logout", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
