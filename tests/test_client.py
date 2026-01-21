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


@pytest.mark.parametrize("method", ["POST", "PATCH", "DELETE"])
def test_client_cache_invalidate_on_mutation(httpserver: HTTPServer, method: str) -> None:
    client = MregClient(url=httpserver.url_for(""), domain="example.com", cache=True)
    assert client._cache is not None

    httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
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

    httpserver.expect_oneshot_request("/api/v1/hosts/", method=method).respond_with_json(
        {"detail": "Mutation successful"}
    )

    hosts_pre_mutation = Host.get_list()
    assert len(hosts_pre_mutation) == 1

    # Assert we can access the cached data
    hosts_pre_mutation_cached = Host.get_list()
    assert len(hosts_pre_mutation_cached) == 1

    # We don't care about the mutation response or respecting what it would actually do
    if method == "POST":
        client.post("/api/v1/hosts/", params={"name": "newhost.example.com"})
    elif method == "PATCH":
        client.patch("/api/v1/hosts/", params={"comment": "Updated comment"})
    elif method == "DELETE":
        client.delete("/api/v1/hosts/")

    # Pretend response has changed after mutation
    httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
        [
            {
                "id": 1,
                "name": "host1.example.com",
                "ipaddresses": [],
                "comment": "My comment",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 2,
                "name": "host2.example.com",
                "ipaddresses": [],
                "comment": "My other comment",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
            },
        ]
    )

    hosts_post_mutation = Host.get_list()
    assert len(hosts_post_mutation) == 2


def test_client_cache_contextmanager_disabled(httpserver: HTTPServer) -> None:
    client = MregClient(url=httpserver.url_for(""), domain="example.com", cache=True)
    assert client._cache is not None

    # Do some stuff that gets cached
    httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
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
    hosts1 = Host.get_list()
    assert len(client.get_client_history()) == 1

    # Perform same fetches within the context manager - should bypass cache
    with client.cache_disabled():
        httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
            [
                {
                    "id": 1,
                    "name": "host1.example.com",
                    "ipaddresses": [],
                    "comment": "My comment",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                },
                {
                    "id": 2,
                    "name": "host2.example.com",
                    "ipaddresses": [],
                    "comment": "My other comment",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                },
            ]
        )
        hosts2 = Host.get_list()
        assert len(client.get_client_history()) == 2

        assert len(hosts1) == 1
        assert len(hosts2) == 2

    # Fetching outside the context manager should use the cache again
    stats_pre = client.get_cache_stats()
    assert stats_pre is not None

    hosts3 = Host.get_list()
    assert len(client.get_client_history()) == 2  # History unchanged
    assert len(hosts3) == len(hosts1) == 1

    # Compare cache stats
    stats_post = client.get_cache_stats()
    assert stats_post is not None

    assert stats_post.hits == stats_pre.hits + 1
    assert stats_post.misses == stats_pre.misses


def test_client_cache_contextmanager_enabled(httpserver: HTTPServer) -> None:
    client = MregClient(url=httpserver.url_for(""), domain="example.com", cache=False)
    assert client._cache is None

    with client.cache_enabled():
        httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
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
        hosts1 = Host.get_list()
        assert len(client.get_client_history()) == 1

        httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
            [
                {
                    "id": 1,
                    "name": "host1.example.com",
                    "ipaddresses": [],
                    "comment": "My comment",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                },
                {
                    "id": 2,
                    "name": "host2.example.com",
                    "ipaddresses": [],
                    "comment": "My other comment",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                },
            ]
        )
        # Second fetch should hit the cache - not the new handler
        hosts2 = Host.get_list()
        assert len(client.get_client_history()) == 1
        assert len(hosts1) == len(hosts2) == 1

    # Fetching outside the context manager should hit the server again
    httpserver.expect_oneshot_request("/api/v1/hosts/", method="GET").respond_with_json(
        [
            {
                "id": 1,
                "name": "host1.example.com",
                "ipaddresses": [],
                "comment": "My comment",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 2,
                "name": "host2.example.com",
                "ipaddresses": [],
                "comment": "My other comment",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z",
            },
        ]
    )
    hosts3 = Host.get_list()
    assert len(client.get_client_history()) == 2
    assert len(hosts3) == 2
