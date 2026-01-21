from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from pytest_httpserver import HTTPServer

from mreg_api.client import MregClient
from mreg_api.exceptions import GetError
from mreg_api.models.models import Host


def test_client_singleton() -> None:
    client1 = MregClient(url="http://example.com", domain="example.com", timeout=30)
    client2 = MregClient()
    assert client1 is client2


def test_client_reset_instance() -> None:
    client1 = MregClient()
    MregClient.reset_instance()
    client2 = MregClient()
    assert client1 is not client2


def test_client_caching(httpserver: HTTPServer) -> None:
    client = MregClient(url=httpserver.url_for(""), domain="example.com", cache=True)
    assert client._cache is not None

    def init_endpoint() -> None:
        httpserver.expect_oneshot_request("/api/v1/hosts/").respond_with_json(
            [
                {
                    "id": 1,
                    "name": "host1.example.com",
                    "ipaddresses": [],
                    "comment": "My comment",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ]
        )

    # Using Host model to access endpoint

    init_endpoint()
    # First fetch - should hit the server
    hosts1 = Host.get_list()

    # Second fetch - should use the cache
    hosts2 = Host.get_list()

    assert hosts1 == hosts2

    # Using client directly to access endpoint
    init_endpoint()

    # First fetch - should hit the server
    resp = client.get(str(Host.endpoint()), params=None, ok404=False)

    # Ensure that trying to access the endpoint now raises an exception (oneshot)
    with pytest.raises(GetError) as exc_info:
        client._do_get(str(Host.endpoint()), params=None, ok404=False)
    assert "No handler found" in exc_info.value.response.text
    assert exc_info.value.response.status_code == snapshot(500)

    resp2 = client.get(str(Host.endpoint()), params=None, ok404=False)

    assert resp.content == resp2.content
    assert resp.json() == snapshot(
        [
            {
                "id": 1,
                "name": "host1.example.com",
                "ipaddresses": [],
                "comment": "My comment",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        ]
    )
