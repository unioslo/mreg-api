"""MREG API - Python client library for MREG."""

from __future__ import annotations

from mreg_api import cache
from mreg_api import dirs
from mreg_api import endpoints
from mreg_api import exceptions
from mreg_api import models
from mreg_api import types
from mreg_api.__about__ import __version__
from mreg_api.cache import CacheConfig
from mreg_api.client import MregClient

__all__ = [
    "__version__",
    "MregClient",
    "CacheConfig",
    "cache",
    "dirs",
    "endpoints",
    "exceptions",
    "models",
    "types",
]
