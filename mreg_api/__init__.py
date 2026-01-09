"""MREG API - Python client library for MREG network registry."""

from __future__ import annotations

from mreg_api.__about__ import __version__
from mreg_api.client import MregApiClient
from mreg_api.client import get_client
from mreg_api.config import MregCliConfig
from mreg_api.exceptions import APIError
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import LoginFailedError
from mreg_api.exceptions import MregApiBaseError
from mreg_api.exceptions import MregValidationError

__all__ = [
    "__version__",
    "MregApiClient",
    "get_client",
    "MregCliConfig",
    "APIError",
    "EntityAlreadyExists",
    "EntityNotFound",
    "LoginFailedError",
    "MregApiBaseError",
    "MregValidationError",
]
