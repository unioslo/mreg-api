from __future__ import annotations

from typing import Generator

import pytest
from pytest_httpserver import HTTPServer

from mreg_api.client import MregClient
from mreg_api.models.fields import hostname_domain


@pytest.fixture(autouse=True, scope="function")
def reset_instance_after_test() -> None:
    """Reset MregClient singleton instance after each test."""
    try:
        MregClient.reset_instance()
    except KeyError:
        pass


@pytest.fixture(autouse=True, scope="function")
def reset_hostname_after_test() -> Generator[None, None, None]:
    """Reset hostname domain context variable after each test."""
    pre = hostname_domain.get()
    try:
        yield
    finally:
        hostname_domain.set(pre)


@pytest.fixture
def client(httpserver: HTTPServer) -> MregClient:
    """Get an MregClient instance for the current httpserver context."""
    return MregClient(url=httpserver.url_for(""), domain="example.com")
