from __future__ import annotations

import pytest
from pytest_httpserver import HTTPServer

from mreg_api.client import MregClient


@pytest.fixture
def client(httpserver: HTTPServer) -> MregClient:
    """Get an MregClient instance for the current httpserver context."""
    return MregClient(url=httpserver.url_for(""), domain="example.com")
