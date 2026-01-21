from __future__ import annotations

import pytest
from pytest_httpserver import HTTPServer

from mreg_api.client import MregClient


@pytest.fixture(autouse=True, scope="function")
def reset_instance_after_test() -> None:
    """Reset MregClient singleton instance after each test."""
    try:
        MregClient.reset_instance()
    except KeyError:
        pass


@pytest.fixture
def client(httpserver: HTTPServer) -> MregClient:
    """Get an MregClient instance for the current httpserver context."""
    return MregClient(url=httpserver.url_for(""), domain="example.com")
