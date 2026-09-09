from __future__ import annotations

import logging
from collections.abc import Generator

import pytest
from pytest_httpserver import HTTPServer

from mreg_api.client import MregClient


@pytest.fixture
def client(httpserver: HTTPServer) -> MregClient:
    """Get an MregClient instance for the current httpserver context."""
    return MregClient(url=httpserver.url_for(""), domain="example.com")


@pytest.fixture
def fail_on_error_log(caplog: pytest.LogCaptureFixture) -> Generator[None, None, None]:
    """Fixture that fails the test if any error logs are emitted during the test."""
    with caplog.at_level("ERROR"):
        yield
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, f"Unexpected error logs: {[r.getMessage() for r in errors]}"
