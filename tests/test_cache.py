from __future__ import annotations

import pytest

from mreg_api import CacheConfig
from mreg_api.cache import MregApiCache
from mreg_api.exceptions import CacheMiss


def test_cache_enable() -> None:
    # Use default config
    config = CacheConfig(enable=False, tag="test_cache_enable")
    cache = MregApiCache[str].new(config)

    # Cache should be disabled
    assert not cache.is_enabled
    cache.set("test_key", "test_value")
    with pytest.raises(CacheMiss) as exc_info:
        cache.get("test_key")
    assert "disabled" in str(exc_info.value)

    # Enable cache
    cache.enable()
    cache.set("test_key", "test_value")
    assert cache.is_enabled
    value = cache.get("test_key")
    assert value == "test_value"

    # Re-disable cache
    cache.disable()
    assert not cache.is_enabled
    with pytest.raises(CacheMiss) as exc_info:
        cache.get("test_key")
    assert "disabled" in str(exc_info.value)


def test_cache_disable() -> None:
    # Use default config
    config = CacheConfig(enable=True, tag="test_cache_disable")
    cache = MregApiCache[str].new(config)

    # Cache should be enabled
    cache.set("test_key", "test_value")
    assert cache.is_enabled
    value = cache.get("test_key")
    assert value == "test_value"

    # Disable cache
    cache.disable()
    assert not cache.is_enabled
    with pytest.raises(CacheMiss) as exc_info:
        cache.get("test_key")
    assert "disabled" in str(exc_info.value)

    # Re-enable cache and check value is still there
    cache.enable()
    assert cache.is_enabled
    value = cache.get("test_key")
    assert value == "test_value"
