"""Firebase Authentication (Google sign-in) and per-tier limits.

Pure logic with no FastAPI import: main.py owns the routes and cookies. The flow:

1. The browser signs in with the Firebase JS SDK (Google provider, popup) and
   receives a Firebase *ID token* -- a JWT signed by Google.
2. It POSTs that token to /api/login. `verify_id_token()` checks the signature
   against Google's published certificates for Firebase, plus audience
   (= our Firebase project id), issuer, expiry and a verified e-mail.
3. main.py then stores the profile as a login session in the store and sets its
   own cookie. ID tokens last an hour; our login lasts 7 days, so the token is
   only ever used once, at sign-in.

No secret lives on the server: the Firebase web config (api key, project id) is
public by design, and tokens are verified with public certificates.

When FIREBASE_PROJECT_ID / FIREBASE_API_KEY are not set, sign-in is off and the
app behaves as before (no tiers) -- local runs and tests need no setup.
"""

import os
import re
import threading
import time
from dataclasses import dataclass

import jwt
import requests
from cryptography import x509

CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
CERTS_TIMEOUT = 10
CERTS_DEFAULT_TTL = 60 * 60
CERTS_MIN_REFRESH = 60  # seconds between refetches triggered by an unknown key id

LOGIN_TTL = 7 * 24 * 60 * 60  # login sessions slide: renewed when used...
LOGIN_REFRESH = 60 * 60  # ...but at most once an hour (one store write, not one per request)

DAY = 24 * 60 * 60

_PROJECT_ID_RE = re.compile(r"[a-z][a-z0-9-]{4,29}")


class AuthError(Exception):
    """Sign-in could not be completed (bad or expired token, unverified e-mail)."""


@dataclass(frozen=True)
class Tier:
    """What one kind of visitor may do. `ask_per_day` / `uploads_per_day` are
    sliding 24-hour windows kept in the store (per user, or per IP for guests)."""

    name: str
    max_docs: int
    ask_per_day: int
    uploads_per_day: int


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def guest_tier() -> Tier:
    return Tier(
        "guest",
        max_docs=max(1, _env_int("GUEST_MAX_DOCS", 1)),
        ask_per_day=_env_int("GUEST_DAILY_QUESTIONS", 5),
        uploads_per_day=_env_int("GUEST_DAILY_UPLOADS", 3),
    )


def user_tier(max_docs: int) -> Tier:
    return Tier(
        "user",
        max_docs=max_docs,
        ask_per_day=_env_int("USER_DAILY_QUESTIONS", 200),
        uploads_per_day=_env_int("USER_DAILY_UPLOADS", 30),
    )


def project_id() -> str:
    value = os.environ.get("FIREBASE_PROJECT_ID", "").strip()
    return value if _PROJECT_ID_RE.fullmatch(value) else ""


def enabled() -> bool:
    return bool(project_id() and os.environ.get("FIREBASE_API_KEY", "").strip())


def web_config() -> dict:
    """The public Firebase web config the browser SDK needs."""
    pid = project_id()
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY", "").strip(),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", "").strip() or f"{pid}.firebaseapp.com",
        "projectId": pid,
    }


class _Certs:
    """Google's signing certificates for Firebase ID tokens, cached for as long
    as Google says (Cache-Control max-age), and refetched when a token names a
    key id we have not seen (key rotation)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, object] = {}
        self._expires = 0.0
        self._fetched = 0.0

    def get(self, kid: str):
        with self._lock:
            now = time.time()
            if kid in self._keys and now < self._expires:
                return self._keys[kid]
            # An unknown key id means "rotated" or "forged". Refetching on every
            # forged token would let anyone make this server call Google at will,
            # so a refetch happens at most once per CERTS_MIN_REFRESH seconds.
            if self._keys and now - self._fetched < CERTS_MIN_REFRESH and now < self._expires:
                return None
            self._refresh()
            return self._keys.get(kid)

    def _refresh(self) -> None:
        try:
            response = requests.get(CERTS_URL, timeout=CERTS_TIMEOUT)
            if response.status_code != 200:
                raise AuthError("Could not load Google's signing keys.")
            pems = response.json()
            self._keys = {
                kid: x509.load_pem_x509_certificate(pem.encode("ascii")).public_key() for kid, pem in pems.items()
            }
        except (requests.RequestException, ValueError, AttributeError) as e:
            raise AuthError("Could not load Google's signing keys.") from e
        match = re.search(r"max-age=(\d+)", response.headers.get("Cache-Control", ""))
        self._fetched = time.time()
        self._expires = self._fetched + (int(match.group(1)) if match else CERTS_DEFAULT_TTL)

    def clear(self) -> None:
        with self._lock:
            self._keys, self._expires, self._fetched = {}, 0.0, 0.0


certs = _Certs()


def verify_id_token(token: str) -> dict:
    """Blocking (may fetch certificates; call in a worker thread).
    Returns {"sub", "email", "name"} for a valid Firebase ID token."""
    pid = project_id()
    if not pid or not isinstance(token, str) or not 20 <= len(token) <= 4096:
        raise AuthError("Invalid sign-in token.")
    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256" or not header.get("kid"):
            raise AuthError("Invalid sign-in token.")
        key = certs.get(header["kid"])
        if key is None:
            raise AuthError("Invalid sign-in token.")
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=pid,
            issuer=f"https://securetoken.google.com/{pid}",
            options={"require": ["exp", "iat", "aud", "iss", "sub"]},
            leeway=10,
        )
    except jwt.PyJWTError as e:
        raise AuthError("Invalid or expired sign-in token.") from e

    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        raise AuthError("Invalid sign-in token.")
    if not claims.get("email") or claims.get("email_verified") is not True:
        raise AuthError("Your Google account needs a verified e-mail address.")
    return {
        "sub": claims["sub"],
        "email": str(claims["email"])[:254],
        "name": str(claims.get("name") or claims["email"])[:120],
    }
