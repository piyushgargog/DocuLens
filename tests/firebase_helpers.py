"""Shared Firebase test doubles: real RS256 tokens signed with a throwaway key,
and a fake Google certificate endpoint, so token verification is exercised for
real without touching the network."""

import datetime
import time

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

PROJECT = "doculens-test"
KID = "test-key-1"
TXT = {"file": ("notes.txt", b"Mercury is the smallest planet.", "text/plain")}


def _make_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


KEY = _make_key()
OTHER_KEY = _make_key()


def _cert_pem(key) -> str:
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode()


CERT_PEM = _cert_pem(KEY)


def make_token(key=KEY, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": f"https://securetoken.google.com/{PROJECT}",
        "aud": PROJECT,
        "sub": "uid-123",
        "iat": now,
        "exp": now + 3600,
        "email": "ada@example.com",
        "email_verified": True,
        "name": "Ada",
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


class FakeCerts:
    def __init__(self):
        self.fetches = 0

    def get(self, response_headers=None):
        self.fetches += 1

        class Resp:
            status_code = 200
            headers = response_headers or {"Cache-Control": "public, max-age=3600"}

            def json(self_inner):
                return {KID: CERT_PEM}

        return Resp()
