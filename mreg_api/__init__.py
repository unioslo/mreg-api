"""MREG API - Python client library for MREG."""

from __future__ import annotations

from mreg_api import cache
from mreg_api import endpoints
from mreg_api import events
from mreg_api import exceptions
from mreg_api import models
from mreg_api import types
from mreg_api.__about__ import __version__
from mreg_api.cache import CacheConfig
from mreg_api.client import MregClient
from mreg_api.events import Event
from mreg_api.events import EventKind
from mreg_api.events import EventLog
from mreg_api.events import ObjectRef

__all__ = [
    "__version__",
    "MregClient",
    "CacheConfig",
    "Event",
    "EventKind",
    "EventLog",
    "ObjectRef",
    "cache",
    "endpoints",
    "events",
    "exceptions",
    "models",
    "types",
]
