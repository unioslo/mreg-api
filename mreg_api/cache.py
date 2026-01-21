"""Caching utilities for mreg-api."""

from __future__ import annotations

import logging
from typing import Generic
from typing import ParamSpec
from typing import Self
from typing import TypeVar
from typing import final

from diskcache import Cache
from pydantic import BaseModel
from pydantic import ByteSize
from pydantic import field_serializer

logger = logging.getLogger(__name__)


P = ParamSpec("P")
T = TypeVar("T")

DEFAULT_CACHE_TAG = "mreg-api"
DEFAULT_CACHE_TTL = 300  # seconds


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

    def as_table_args(self) -> tuple[list[str], list[str], list[Self]]:
        """Get a tuple of string arguments for table display."""
        return (
            ["Items", "Hits", "Misses", "Size", "TTL", "Directory"],
            ["items", "hits", "misses", "size", "ttl", "directory"],
            [self],
        )


def create_cache(config: CacheConfig, item_type: type[T]) -> MregApiCache[T] | None:
    """Create the global mreg-cli cache.

    Falls back to a no-op cache object if the diskcache cache cannot be created.
    """
    try:
        return MregApiCache[item_type](Cache(), config=config)
    except Exception as e:
        logger.exception("Failed to create cache: %s", e)
        return None


class CacheConfig(BaseModel):
    """Configuration for the mreg-api cache."""

    enabled: bool = True
    ttl: int = DEFAULT_CACHE_TTL
    tag: str = DEFAULT_CACHE_TAG
    directory: str | None = None


@final
class MregApiCache(Generic[T]):
    """Wrapper around the mreg-cli cache."""

    def __init__(self, cache: Cache, config: CacheConfig) -> None:
        """Initialize the cache wrapper."""
        self.cache = cache
        self.config = config

    def get_info(self) -> CacheInfo:
        """Get information about the cache.

        Raises:
            ValueError: If the cache statistics cannot be retrieved.
        """
        hits, misses = self.cache.stats()

        return CacheInfo(
            size=self.cache.volume(),  # pyright: ignore[reportAny]
            hits=hits,  # pyright: ignore[reportArgumentType]
            misses=misses,  # pyright: ignore[reportArgumentType]
            items=len(self.cache),  # pyright: ignore[reportArgumentType]
            directory=self.cache.directory,
            ttl=self.config.ttl,
        )

    def set(self, key: str, value: T | None, expire: int | None = None) -> None:
        """Set a value in the cache."""
        try:
            self.cache.set(key, value, expire=expire or self.config.ttl, tag=self.config.tag)
        except Exception as e:
            logger.exception("Failed to set cache key %s: %s", key, e)

    def get(self, key: str) -> T | None:
        """Get a value from the cache."""
        # NOTE: diskcache.Cache.get has poor type annotations
        # so we cannot properly type hint the return value here.
        try:
            return self.cache.get(key, default=None)  # pyright: ignore[reportReturnType, reportUnknownVariableType]
        except Exception as e:
            logger.exception("Failed to get cache key %s: %s", key, e)
            return None

    def clear(self) -> None:
        """Clear the cache and reset statistics."""
        try:
            items = self.cache.evict(self.config.tag)
            logger.info("Cleared %d items from cache with tag %s", items, self.config.tag)
        except Exception as e:
            logger.exception("Failed to clear cache for tag %s: %s", self.config.tag, e)
