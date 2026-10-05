"""Shared key-value state: login sessions and rate limits.

Two backends behind one async interface:

* `RedisStore` -- used when REDIS_URL is set (any Redis, e.g. Upstash with a
  `rediss://` URL). State survives a restart and is shared by every worker;
  keys expire on their own (Redis TTL).
* `MemoryStore` -- the default with no REDIS_URL (local runs, the test suite).
  Same behaviour inside one process.

`ResilientStore` wraps Redis and falls back to a private MemoryStore when
Redis errors, so a Redis outage degrades the site (logins and counters reset)
instead of taking it down.

Besides JSON values it holds binary blobs and small hashes, which docstore.py
uses for persisted documents. Nothing here knows about documents or users.
"""

import json
import os
import secrets
import time
from collections import deque

from observability import log

# Sliding-window limiter, run atomically inside Redis. ARGV: now_ms, a unique
# member id, then (limit, window_ms) pairs. Returns 0 if the hit was allowed
# and counted, otherwise the seconds to wait until a slot frees up.
_RATE_LUA = """
local now = tonumber(ARGV[1])
local longest = 0
for i = 3, #ARGV, 2 do
  longest = math.max(longest, tonumber(ARGV[i + 1]))
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - longest)
for i = 3, #ARGV, 2 do
  local limit = tonumber(ARGV[i])
  local window = tonumber(ARGV[i + 1])
  local count = redis.call('ZCOUNT', KEYS[1], now - window, '+inf')
  if count >= limit then
    local blocker = redis.call('ZRANGEBYSCORE', KEYS[1], now - window, '+inf',
                               'WITHSCORES', 'LIMIT', count - limit, 1)
    return math.floor((window - (now - tonumber(blocker[2]))) / 1000) + 1
  end
end
redis.call('ZADD', KEYS[1], now, ARGV[2])
redis.call('PEXPIRE', KEYS[1], longest)
return 0
"""


class MemoryStore:
    """In-process store with the same semantics as RedisStore."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, float]] = {}  # key -> (json, expires_at)
        self._blobs: dict[str, tuple[bytes, float]] = {}
        self._hashes: dict[str, tuple[dict[str, str], float]] = {}
        self._hits: dict[str, deque] = {}

    def _live(self, key: str) -> str | None:
        entry = self._values.get(key)
        if entry is None:
            return None
        if entry[1] <= time.time():
            del self._values[key]
            return None
        return entry[0]

    async def get_json(self, key: str):
        raw = self._live(key)
        return None if raw is None else json.loads(raw)

    async def set_json(self, key: str, value, ttl: int) -> None:
        self._values[key] = (json.dumps(value), time.time() + ttl)

    async def pop_json(self, key: str):
        raw = self._live(key)
        self._values.pop(key, None)
        return None if raw is None else json.loads(raw)

    async def delete(self, key: str) -> None:
        self._values.pop(key, None)
        self._hashes.pop(key, None)

    async def set_bytes(self, key: str, data: bytes, ttl: int) -> None:
        self._blobs[key] = (data, time.time() + ttl)

    async def get_bytes(self, key: str) -> bytes | None:
        entry = self._blobs.get(key)
        if entry is None:
            return None
        if entry[1] <= time.time():
            del self._blobs[key]
            return None
        return entry[0]

    async def delete_bytes(self, keys: list[str]) -> None:
        for key in keys:
            self._blobs.pop(key, None)

    def _live_hash(self, key: str) -> dict | None:
        entry = self._hashes.get(key)
        if entry is None:
            return None
        if entry[1] <= time.time():
            del self._hashes[key]
            return None
        return entry[0]

    async def hset_json(self, key: str, field: str, value, ttl: int) -> None:
        fields = self._live_hash(key) or {}
        fields[field] = json.dumps(value)
        self._hashes[key] = (fields, time.time() + ttl)

    async def hgetall_json(self, key: str) -> dict:
        return {field: json.loads(raw) for field, raw in (self._live_hash(key) or {}).items()}

    async def hdel(self, key: str, field: str) -> None:
        fields = self._live_hash(key)
        if fields is not None:
            fields.pop(field, None)
            if not fields:
                del self._hashes[key]

    async def rate_hit(self, key: str, limits: list[tuple[int, int]]) -> int:
        now = time.time()
        hits = self._hits.setdefault(key, deque())
        longest = max(window for _, window in limits)
        while hits and now - hits[0] > longest:
            hits.popleft()
        for limit, window in limits:
            recent = [t for t in hits if now - t <= window]
            if len(recent) >= limit:
                return int(window - (now - recent[len(recent) - limit])) + 1
        hits.append(now)
        return 0

    async def sweep(self) -> None:
        """Drop expired values and idle rate-limit history."""
        now = time.time()
        for table in (self._values, self._blobs, self._hashes):
            for key in [k for k, (_, exp) in table.items() if exp <= now]:
                del table[key]
        for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] > 24 * 3600]:
            del self._hits[key]

    def clear(self) -> None:
        self._values.clear()
        self._blobs.clear()
        self._hashes.clear()
        self._hits.clear()

    async def close(self) -> None:
        return None


class RedisStore:
    """Redis backend (redis.asyncio). Values are JSON with a TTL. The client
    returns bytes (blobs are binary); json.loads accepts bytes."""

    def __init__(self, url: str, client=None) -> None:
        import redis.asyncio as aioredis

        # `client` lets tests inject a fake; production connects from the URL.
        self._redis = client or aioredis.from_url(
            url, socket_connect_timeout=3, socket_timeout=3, health_check_interval=30
        )
        self._rate_script = self._redis.register_script(_RATE_LUA)

    async def get_json(self, key: str):
        raw = await self._redis.get(key)
        return await self._decode(key, raw)

    async def _decode(self, key: str, raw):
        """JSON from Redis. A value that is not valid JSON (corruption, or something
        else wrote the key) is deleted and treated as absent: it is bad *data*, not
        an outage, so it must not make the whole store look down."""
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            log.warning("Dropping a corrupt value in the state store")
            await self._redis.delete(key)
            return None

    async def set_json(self, key: str, value, ttl: int) -> None:
        await self._redis.set(key, json.dumps(value), ex=ttl)

    async def pop_json(self, key: str):
        raw = await self._redis.getdel(key)
        return await self._decode(key, raw)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def set_bytes(self, key: str, data: bytes, ttl: int) -> None:
        await self._redis.set(key, data, ex=ttl)

    async def get_bytes(self, key: str) -> bytes | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return raw if isinstance(raw, bytes) else str(raw).encode("utf-8")

    async def delete_bytes(self, keys: list[str]) -> None:
        if keys:
            await self._redis.delete(*keys)

    async def hset_json(self, key: str, field: str, value, ttl: int) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, field, json.dumps(value))
            pipe.expire(key, ttl)
            await pipe.execute()

    async def hgetall_json(self, key: str) -> dict:
        raw = await self._redis.hgetall(key)
        out = {}
        for field, value in raw.items():
            name = field.decode() if isinstance(field, bytes) else field
            try:
                out[name] = json.loads(value)
            except ValueError:  # one corrupt field: drop it, keep the rest
                log.warning("Dropping a corrupt hash field in the state store")
                await self._redis.hdel(key, field)
        return out

    async def hdel(self, key: str, field: str) -> None:
        await self._redis.hdel(key, field)

    async def rate_hit(self, key: str, limits: list[tuple[int, int]]) -> int:
        args: list[int | str] = [int(time.time() * 1000), secrets.token_hex(6)]
        for limit, window in limits:
            args += [limit, window * 1000]
        return int(await self._rate_script(keys=[key], args=args))

    async def sweep(self) -> None:
        return None  # Redis expires keys itself

    async def ping(self) -> None:
        await self._redis.ping()

    async def close(self) -> None:
        await self._redis.aclose()


class ResilientStore:
    """Redis first; on any Redis error, the same call runs on a local
    MemoryStore instead. After a failure Redis is skipped for RETRY_AFTER
    seconds, so a remote Redis that is down costs one timeout, not one per
    request. Errors are logged at most once a minute."""

    RETRY_AFTER = 30

    def __init__(self, primary: RedisStore) -> None:
        self.primary = primary
        self.fallback = MemoryStore()
        self._last_warned = 0.0
        self._skip_until = 0.0

    @property
    def degraded(self) -> bool:
        """True while Redis is being skipped after a failure."""
        return time.time() < self._skip_until

    def _failed(self, err: Exception) -> None:
        now = time.time()
        self._skip_until = now + self.RETRY_AFTER
        if now - self._last_warned > 60:
            self._last_warned = now
            log.error("Redis unavailable (%s); using in-process state until it recovers", type(err).__name__)

    async def _call(self, name: str, *args):
        if time.time() >= self._skip_until:
            try:
                return await getattr(self.primary, name)(*args)
            except Exception as e:  # redis.RedisError, OSError, timeouts, bad JSON
                self._failed(e)
        return await getattr(self.fallback, name)(*args)

    async def get_json(self, key):
        return await self._call("get_json", key)

    async def set_json(self, key, value, ttl):
        return await self._call("set_json", key, value, ttl)

    async def pop_json(self, key):
        return await self._call("pop_json", key)

    async def delete(self, key):
        return await self._call("delete", key)

    async def set_bytes(self, key, data, ttl):
        return await self._call("set_bytes", key, data, ttl)

    async def get_bytes(self, key):
        return await self._call("get_bytes", key)

    async def delete_bytes(self, keys):
        return await self._call("delete_bytes", keys)

    async def hset_json(self, key, field, value, ttl):
        return await self._call("hset_json", key, field, value, ttl)

    async def hgetall_json(self, key):
        return await self._call("hgetall_json", key)

    async def hdel(self, key, field):
        return await self._call("hdel", key, field)

    async def rate_hit(self, key, limits):
        return await self._call("rate_hit", key, limits)

    async def sweep(self):
        await self.fallback.sweep()

    async def close(self):
        try:
            await self.primary.close()
        except Exception as e:  # shutting down during an outage must still shut down
            log.warning("Closing the Redis connection failed (%s)", type(e).__name__)


def make_store() -> MemoryStore | ResilientStore:
    """RedisStore (with outage fallback) if REDIS_URL is set, else MemoryStore."""
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return MemoryStore()
    return ResilientStore(RedisStore(url))
