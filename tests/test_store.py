"""The state store: login values with TTLs, single-use pops and sliding-window
rate limits. Every test runs against both the in-process store and the Redis
store (backed by fakeredis, which executes the real Lua limiter script)."""

import asyncio

import fakeredis
import pytest

import store


@pytest.fixture(params=["memory", "redis"])
def backend(request):
    if request.param == "memory":
        return store.MemoryStore()
    return store.RedisStore("redis://unused", client=fakeredis.FakeAsyncRedis())


def run(coro):
    return asyncio.run(coro)


def test_set_get_delete_roundtrip(backend):
    async def go():
        await backend.set_json("k", {"a": 1}, 60)
        assert await backend.get_json("k") == {"a": 1}
        await backend.delete("k")
        assert await backend.get_json("k") is None

    run(go())


def test_pop_is_single_use(backend):
    async def go():
        await backend.set_json("handshake", {"v": "x"}, 60)
        assert await backend.pop_json("handshake") == {"v": "x"}
        assert await backend.pop_json("handshake") is None

    run(go())


def test_values_expire(backend):
    async def go():
        await backend.set_json("k", 1, 1)
        await asyncio.sleep(1.2)
        assert await backend.get_json("k") is None

    run(go())


def test_rate_hit_allows_up_to_the_limit_then_reports_a_wait(backend):
    async def go():
        assert [await backend.rate_hit("rl", [(3, 60)]) for _ in range(3)] == [0, 0, 0]
        wait = await backend.rate_hit("rl", [(3, 60)])
        assert 1 <= wait <= 61

    run(go())


def test_rate_hit_honours_every_window(backend):
    async def go():
        limits = [(2, 60), (5, 3600)]
        assert await backend.rate_hit("rl", limits) == 0
        assert await backend.rate_hit("rl", limits) == 0
        assert await backend.rate_hit("rl", limits) > 0  # 3rd in a minute

    run(go())


def test_rate_limits_are_per_key(backend):
    async def go():
        assert await backend.rate_hit("a", [(1, 60)]) == 0
        assert await backend.rate_hit("b", [(1, 60)]) == 0
        assert await backend.rate_hit("a", [(1, 60)]) > 0

    run(go())


def test_rejected_hits_are_not_counted(backend):
    async def go():
        await backend.rate_hit("rl", [(1, 2)])
        for _ in range(5):
            assert await backend.rate_hit("rl", [(1, 2)]) > 0
        await asyncio.sleep(2.1)
        assert await backend.rate_hit("rl", [(1, 2)]) == 0  # only the first hit ever counted

    run(go())


class _Down:
    """A Redis stand-in whose every call fails."""

    def __getattr__(self, name):
        async def fail(*args, **kwargs):
            raise ConnectionError("redis is down")

        return fail


def test_resilient_store_falls_back_when_redis_is_down():
    primary = store.RedisStore("redis://unused", client=fakeredis.FakeAsyncRedis())
    primary._redis = _Down()
    primary._rate_script = _Down().script
    resilient = store.ResilientStore(primary)

    async def go():
        await resilient.set_json("login:x", {"sub": "1"}, 60)
        assert await resilient.get_json("login:x") == {"sub": "1"}
        assert await resilient.rate_hit("rl", [(1, 60)]) == 0
        assert await resilient.rate_hit("rl", [(1, 60)]) > 0

    run(go())


def test_resilient_store_skips_a_down_redis_after_the_first_failure():
    calls = []

    class Counting(_Down):
        def __getattr__(self, name):
            calls.append(name)
            return super().__getattr__(name)

    primary = store.RedisStore("redis://unused", client=fakeredis.FakeAsyncRedis())
    primary._redis = Counting()
    resilient = store.ResilientStore(primary)

    async def go():
        for i in range(5):
            await resilient.set_json(f"k{i}", i, 60)

    run(go())
    assert calls == ["set"]  # one failed attempt, then the breaker keeps Redis out of the way


def test_blobs_roundtrip_and_delete(backend):
    async def go():
        blob = bytes(range(256)) * 10  # arbitrary binary, not valid UTF-8
        await backend.set_bytes("b1", blob, 60)
        assert await backend.get_bytes("b1") == blob
        assert await backend.get_bytes("missing") is None
        await backend.delete_bytes(["b1", "never-existed"])
        assert await backend.get_bytes("b1") is None

    run(go())


def test_blobs_expire(backend):
    async def go():
        await backend.set_bytes("b", b"x", 1)
        await asyncio.sleep(1.2)
        assert await backend.get_bytes("b") is None

    run(go())


def test_hash_fields_are_independent_json(backend):
    async def go():
        await backend.hset_json("h", "a", {"n": 1}, 60)
        await backend.hset_json("h", "b", [1, 2], 60)
        assert await backend.hgetall_json("h") == {"a": {"n": 1}, "b": [1, 2]}
        await backend.hdel("h", "a")
        assert await backend.hgetall_json("h") == {"b": [1, 2]}
        await backend.hdel("h", "b")
        assert await backend.hgetall_json("h") == {}
        assert await backend.hgetall_json("never") == {}

    run(go())


def test_hash_expires_as_a_whole(backend):
    async def go():
        await backend.hset_json("h", "a", 1, 1)
        await asyncio.sleep(1.2)
        assert await backend.hgetall_json("h") == {}

    run(go())
