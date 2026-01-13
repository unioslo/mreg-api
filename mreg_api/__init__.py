"""MREG API - Python client library for MREG."""

from __future__ import annotations

from mreg_api.__about__ import __version__
from mreg_api.client import MregApiClient

__all__ = [
    "__version__",
    "MregApiClient",
]
