"""Caching utilities for mreg-api."""

from __future__ import annotations

import logging
from typing import Generic
from typing import ParamSpec
from typing import TypeVar
from typing import cast
from typing import final

from diskcache import Cache
from pydantic import BaseModel
from pydantic import ByteSize
from pydantic import field_serializer

from mreg_api.exceptions import CacheError
from mreg_api.exceptions import CacheMiss

logger = logging.getLogger(__name__)


P = ParamSpec("P")
T = TypeVar("T")

DEFAULT_CACHE_TAG = "mreg-api"
DEFAULT_CACHE_TTL = 300  # seconds

_CACHE_MISS = object()


class CacheInfo(BaseModel):
    """Information about the cache."""

    items: int
    size: ByteSize
    hits: int
    misses: int
    ttl: int
    directory: str

    @field_serializer("size")
    def _serialize_size(self, value: ByteSize) -> str:
        return value.human_readable()


def _create_cache(config: CacheConfig) -> Cache | None:
    """Create a diskcache.Cache based on the provided configuration.

    If the diskcache.Cache cannot be created (e.g., filesystem access denied),
    returns None.

    Args:
        config: Cache configuration.

    Returns:
        diskcache.Cache instance, or None if creation failed.
    """
    if not config.enable:
        logger.debug("Cache is disabled in config, not creating cache.")
        return None

    try:
        return Cache(directory=config.directory, timeout=config.timeout)
    except Exception as e:
        logger.warning("Failed to create diskcache.Cache: %s. Cache will be disabled.", e)
        return None


class CacheConfig(BaseModel):
    """Configuration for the mreg-api cache."""

    enable: bool = True
    ttl: int = 300
    tag: str = "mreg-api"
    timeout: int = 60
    directory: str | None = None


@final
class MregApiCache(Generic[T]):
    """Wrapper around the mreg-api cache.

    This class can operate in two modes:
    1. Enabled: When a valid diskcache.Cache is provided
    2. Disabled: When cache is None (no-op mode, all operations are safe to call)

    The disabled mode allows callers to use the cache interface without null checks.
    """

    _cache: Cache | None
    """The underlying diskcache.Cache instance, or None if disabled.

    Warning:
        NEVER use `if self._cache` checks! diskcache.Cache implements `__bool__`
        to check if the cache is non-empty, not whether it exists. Always use
        `if self._cache is None` instead.
    """

    def __init__(self, cache: Cache | None, config: CacheConfig) -> None:
        """Initialize the cache wrapper.

        Args:
            cache: The underlying diskcache.Cache instance, or None for disabled mode.
            config: Configuration for cache behavior.
        """
        self._cache = cache
        self.config = config

    @property
    def has_backend(self) -> bool:
        """Return True if an underlying cache backend is available."""
        return self._cache is not None

    @property
    def is_enabled(self) -> bool:
        """Return True if cache is operational (has backend and is enabled in config)."""
        return self.has_backend and self.config.enable

    @classmethod
    def new(cls, config: CacheConfig) -> MregApiCache[T]:
        """Create a new MregApiCache instance based on the provided configuration.

        Args:
            config: Cache configuration.

        Returns:
            MregApiCache instance (may be in disabled mode if creation failed).
        """
        cache = _create_cache(config)
        return cls(cache, config=config)

    def enable(self) -> None:
        """Enable the cache."""
        self.config.enable = True
        # We have no cache backend yet, try to create one
        if self._cache is None:
            self._cache = _create_cache(self.config)

    def disable(self) -> None:
        """Disable the cache."""
        self.config.enable = False

    def get_info(self) -> CacheInfo | None:
        """Get information about the cache.

        Returns:
            CacheInfo object with cache statistics, or None if cache is disabled.
        """
        # No cache object exists.
        # We can return stats for disabled cache as long as we have a backend.
        if self._cache is None:
            return None

        hits, misses = self._cache.stats()

        return CacheInfo(
            size=self._cache.volume(),  # pyright: ignore[reportAny]
            hits=hits,  # pyright: ignore[reportArgumentType]
            misses=misses,  # pyright: ignore[reportArgumentType]
            items=len(self._cache),  # pyright: ignore[reportArgumentType]
            directory=self._cache.directory,
            ttl=self.config.ttl,
        )

    def set(self, key: str, value: T | None, expire: int | None = None) -> None:
        """Set a value in the cache.

        No-op if cache is disabled.
        """
        if not self.is_enabled or self._cache is None:
            return
        try:
            self._cache.set(key, value, expire=expire or self.config.ttl, tag=self.config.tag)
        except Exception as e:
            raise CacheError(f"Failed to set cache key {key}: {e}") from e

    def get(self, key: str) -> T | None:
        """Get a value from the cache.

        Raises:
            CacheMiss: If the key is not found in the cache or cache is disabled.
        """
        if not self.is_enabled or self._cache is None:
            raise CacheMiss(f"Cache disabled, key: {key}")

        # Use sentinel to distinguish between None and missing key
        try:
            value = self._cache.get(key, default=_CACHE_MISS)  # pyright: ignore[reportReturnType, reportUnknownVariableType]
        except Exception as e:
            raise CacheError(f"Failed to get cache key {key}: {e}") from e

        if value is _CACHE_MISS:
            raise CacheMiss(f"Cache miss for key: {key}")
        # NOTE: diskcache.Cache.get has poor type annotations
        # so we cannot properly type hint the return value here.
        return cast(T | None, value)

    def clear(self) -> int:
        """Clear the cache and reset statistics.

        Returns 0 if cache is disabled.
        """
        if not self.is_enabled or self._cache is None:
            return 0
        try:
            n_items = self._cache.evict(self.config.tag)
            logger.info("Cleared %d items from cache with tag %s", n_items, self.config.tag)
            return n_items
        except Exception as e:
            raise CacheError(f"Failed to clear cache: {e}") from e
