from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from inline_snapshot import snapshot

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
        _ = cache.get("test_key")
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
        _ = cache.get("test_key")
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
        _ = cache.get("test_key")
    assert "disabled" in str(exc_info.value)

    # Re-enable cache and check value is still there
    cache.enable()
    assert cache.is_enabled
    value = cache.get("test_key")
    assert value == "test_value"


def test_cache_readonly_fs_without_directory_arg(caplog: pytest.LogCaptureFixture) -> None:
    """Test cache instantiation with no directory argument on read-only filesystem(uses tmpdir)."""
    caplog.set_level(logging.WARNING)
    with patch("tempfile.mkdtemp") as mock_mkdtemp:
        mock_mkdtemp.side_effect = PermissionError("Read-only directory")
        cache = MregApiCache.new(CacheConfig(enable=True, directory=None))

        # Tempdir creation failed, cache should be disabled
        assert caplog.record_tuples == snapshot(
            [
                (
                    "mreg_api.cache",
                    30,
                    "Failed to create diskcache.Cache: Read-only directory. Cache will be disabled.",
                )
            ]
        )
        assert not cache.is_enabled
        assert cache._cache is None  # pyright: ignore[reportPrivateUsage]


def test_cache_readonly_fs_with_directory_arg(caplog: pytest.LogCaptureFixture) -> None:
    """Test cache instantiation with no directory argument on read-only filesystem."""
    caplog.set_level(logging.WARNING)
    with patch("os.makedirs") as mock_makedirs:
        # Mock both directory strategies
        mock_makedirs.side_effect = PermissionError("Read-only directory")
        cache = MregApiCache.new(CacheConfig(enable=True, directory="/tmp/mreg_api_cache_test"))

        # Creation of specified directory failed, cache should be disabled
        assert caplog.record_tuples == snapshot(
            [
                (
                    "mreg_api.cache",
                    30,
                    'Failed to create diskcache.Cache: [Errno None] Cache directory "/tmp/mreg_api_cache_test" does not exist and could not be created. Cache will be disabled.',
                )
            ]
        )
        assert not cache.is_enabled
        assert cache._cache is None  # pyright: ignore[reportPrivateUsage]
