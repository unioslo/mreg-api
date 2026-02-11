"""MREG API client package."""

from __future__ import annotations

from mreg_api.client.core import Header
from mreg_api.client.core import MregClient
from mreg_api.client.core import RequestRecord
from mreg_api.client.core import invalidate_cache
from mreg_api.client.transport import PaginatedResponse
from mreg_api.client.transport import validate_list_response
from mreg_api.client.transport import validate_paginated_response
from mreg_api.request_context import last_request_method
from mreg_api.request_context import last_request_url

__all__ = [
    "Header",
    "MregClient",
    "PaginatedResponse",
    "RequestRecord",
    "invalidate_cache",
    "last_request_method",
    "last_request_url",
    "validate_list_response",
    "validate_paginated_response",
]
