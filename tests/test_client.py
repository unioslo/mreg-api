from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot
from pytest_httpserver import HTTPServer
from werkzeug import Response

from mreg_api.client import MregClient
from mreg_api.exceptions import GetError
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import MultipleEntitiesFound
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
    client.clear_cache()
    init_endpoint()

    # First fetch - should hit the server
    resp = client.get(str(Host.endpoint()), params=None, ok404=False)

    # Ensure that trying to access the endpoint now raises an exception (oneshot)
    with pytest.raises(GetError) as exc_info:
        client._do_get(str(Host.endpoint()), params=None, ok404=False)
    assert "No handler found" in exc_info.value.response.text
    assert exc_info.value.response.status_code == snapshot(500)

    # We know the endpoint doesn't work, so this is certain to go via the cache
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


def test_client_caching_contextmanager_disabled(httpserver: HTTPServer) -> None:
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
    with client.caching(enable=False):
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
    info_pre = client.get_cache_info()
    assert info_pre is not None

    hosts3 = Host.get_list()
    assert len(client.get_client_history()) == 2  # History unchanged
    assert len(hosts3) == len(hosts1) == 1

    # Compare cache info
    info_post = client.get_cache_info()
    assert info_post is not None

    assert info_post.hits == info_pre.hits + 1
    assert info_post.misses == info_pre.misses


def test_client_caching_contextmanager_enabled(httpserver: HTTPServer) -> None:
    client = MregClient(url=httpserver.url_for(""), domain="example.com", cache=False)
    assert client._cache is None

    with client.caching(enable=True):
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


@pytest.mark.parametrize(
    "inp,expect",
    [
        # Empty dict
        ({}, {}),
        # Mixed values
        ({"foo": "a", "bar": None}, {"foo": "a"}),
        # Multiple keys with None values
        ({"foo": None, "bar": None}, {}),
        # Nested dicts
        ({"foo": {"bar": {"baz": None}}}, {}),
        (
            {"foo": {"bar": {"baz": None}}, "qux": {}, "quux": ["a", "b", "c"]},
            {"quux": ["a", "b", "c"]},
        ),
    ],
)
def test_strip_none(inp: dict[str, Any], expect: dict[str, Any]) -> None:
    client = MregClient()
    assert client._strip_none(inp) == expect  # pyright: ignore[reportPrivateUsage]


def test_client_get_list_paginated(httpserver: HTTPServer, client: MregClient) -> None:
    httpserver.expect_oneshot_request("/test_get_list_paginated").respond_with_json(
        {
            "results": [{"foo": "bar"}],
            "count": 1,
            "next": None,
            "previous": None,
        }
    )
    resp = client.get_list("/test_get_list_paginated")
    assert resp == snapshot([{"foo": "bar"}])


def test_client_get_list_paginated_empty(httpserver: HTTPServer, client: MregClient) -> None:
    httpserver.expect_oneshot_request("/test_get_list_paginated_empty").respond_with_json(
        {
            "results": [],
            "count": 0,
            "next": None,
            "previous": None,
        }
    )
    resp = client.get_list("/test_get_list_paginated_empty")
    assert resp == snapshot([])


def test_client_get_list_paginated_multiple_pages(httpserver: HTTPServer, client: MregClient) -> None:
    httpserver.expect_oneshot_request(
        "/test_client_get_list_paginated_multiple_pages"
    ).respond_with_json(
        {
            "results": [{"foo": "bar"}],
            "count": 1,
            "next": "/test_client_get_list_paginated_multiple_pages?page=2",
            "previous": None,
        }
    )
    httpserver.expect_oneshot_request(
        "/test_client_get_list_paginated_multiple_pages", query_string="page=2"
    ).respond_with_json(
        {
            "results": [{"baz": "qux"}],
            "count": 1,
            "next": None,
            "previous": "/test_client_get_list_paginated_multiple_pages",
        }
    )
    resp = client.get_list("/test_client_get_list_paginated_multiple_pages")
    assert resp == snapshot([{"foo": "bar"}, {"baz": "qux"}])


def test_client_get_list_paginated_multiple_pages_ok404(
    httpserver: HTTPServer, client: MregClient
) -> None:
    """Paginated response with 404 on next page is ignored when `ok404=True`."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_paginated_multiple_pages_ok404"
    ).respond_with_json(
        {
            "results": [{"foo": "bar"}],
            "count": 1,
            "next": "/test_client_get_list_paginated_multiple_pages_ok404?page=2",
            "previous": None,
        }
    )
    httpserver.expect_oneshot_request(
        "/test_client_get_list_paginated_multiple_pages_ok404", query_string="page=2"
    ).respond_with_response(Response(status=404))
    assert client.get_list(
        "/test_client_get_list_paginated_multiple_pages_ok404", ok404=True
    ) == snapshot([{"foo": "bar"}])


def test_client_get_list_paginated_multiple_pages_inconsistent_count(
    httpserver: HTTPServer, client: MregClient
) -> None:
    """Inconsistent count in paginated response is ignored."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_paginated_multiple_pages_inconsistent_count"
    ).respond_with_json(
        {
            "results": [{"foo": "bar"}, {"baz": "qux"}],
            "count": 1,  # wrong count
            "next": "/test_client_get_list_paginated_multiple_pages_inconsistent_count?page=2",
            "previous": None,
        }
    )
    httpserver.expect_oneshot_request(
        "/test_client_get_list_paginated_multiple_pages_inconsistent_count", query_string="page=2"
    ).respond_with_json(
        {
            "results": [{"quux": "spam"}],
            "count": 2,  # wrong count
            "next": None,
            "previous": "/test_client_get_list_paginated_multiple_pages_inconsistent_count?page=1",
        }
    )
    resp = client.get_list("/test_client_get_list_paginated_multiple_pages_inconsistent_count")
    assert resp == snapshot([{"foo": "bar"}, {"baz": "qux"}, {"quux": "spam"}])


@pytest.mark.parametrize(
    "results",
    [
        '"foo"',  # Not a list
        "42",  # Not a list
        '{"foo": "bar"}',  # Not a list
        "{'foo': 'bar'}",  # Invalid JSON + not a list
        "[{'foo': 'bar'}]",  # Invalid JSON
    ],
)
def test_client_get_list_paginated_invalid(
    httpserver: HTTPServer, client: MregClient, results: Any
) -> None:
    """Invalid JSON or non-array response is an error."""
    httpserver.expect_oneshot_request("/test_client_get_list_paginated_invalid").respond_with_data(
        f"""{{
            "results": {results},
            "count": 1,
            "next": None,
            "previous": None,
            }}"""
    )
    with pytest.raises(MregValidationError) as exc_info:
        client.get_list("/test_client_get_list_paginated_invalid")
    assert "did not return valid paginated JSON" in exc_info.exconly()


def test_client_get_list_non_paginated(httpserver: HTTPServer, client: MregClient) -> None:
    """Inconsistent count in paginated response is ignored."""
    httpserver.expect_oneshot_request("/test_client_get_list_non_paginated").respond_with_json(
        [
            "foo",
            "bar",
            {"baz": "qux"},
        ]
    )
    resp = client.get_list("/test_client_get_list_non_paginated")
    assert resp == snapshot(["foo", "bar", {"baz": "qux"}])


def test_client_get_list_non_paginated_empty(httpserver: HTTPServer, client: MregClient) -> None:
    """Inconsistent count in paginated response is ignored."""
    httpserver.expect_oneshot_request("/test_client_get_list_non_paginated_empty").respond_with_json([])
    resp = client.get_list("/test_client_get_list_non_paginated_empty")
    assert resp == snapshot([])


def test_client_get_list_non_paginated_non_array(httpserver: HTTPServer, client: MregClient) -> None:
    """Non-paginated non-array response is an error."""
    httpserver.expect_oneshot_request("/test_client_get_list_non_paginated_non_array").respond_with_json(
        {
            "not": "an array",
        }
    )
    with pytest.raises(MregValidationError) as exc_info:
        client.get_list("/test_client_get_list_non_paginated_non_array")
    assert "did not return a valid JSON" in exc_info.exconly()


def test_client_get_list_non_paginated_invalid_json(httpserver: HTTPServer, client: MregClient) -> None:
    """Non-paginated response with invalid JSON is an error."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_non_paginated_invalid_json"
    ).respond_with_data(
        "[{'key': 'value'}, 'foo',]",  # strings must be double quoted
        content_type="application/json",
    )
    with pytest.raises(MregValidationError) as exc_info:
        client.get_list("/test_client_get_list_non_paginated_invalid_json")
    assert "did not return a valid JSON" in exc_info.exconly()


def test_client_get_list_unique_paginated(httpserver: HTTPServer, client: MregClient) -> None:
    """Non-paginated response with invalid JSON is an error."""
    httpserver.expect_oneshot_request("/test_client_get_list_unique_paginated").respond_with_json(
        {
            "results": [{"foo": "bar"}],
            "count": 1,
            "next": None,
            "previous": None,
        }
    )
    resp = client.get_list_unique("/test_client_get_list_unique_paginated", params={})
    assert resp == snapshot({"foo": "bar"})


def test_client_get_list_unique_paginated_too_many_results(
    httpserver: HTTPServer, client: MregClient
) -> None:
    """get_list_unique with multiple unique results is an error."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_unique_paginated_too_many_results"
    ).respond_with_json(
        {
            "results": [{"foo": "bar"}],
            "count": 1,
            "next": "/test_client_get_list_unique_paginated_too_many_results?page=2",
            "previous": None,
        }
    )
    httpserver.expect_oneshot_request(
        "/test_client_get_list_unique_paginated_too_many_results", query_string="page=2"
    ).respond_with_json(
        {
            "results": [{"baz": "qux"}],
            "count": 1,
            "next": None,
            "previous": "/test_client_get_list_unique_paginated_too_many_results?page=1",
        }
    )
    with pytest.raises(MultipleEntitiesFound) as exc_info:
        client.get_list_unique("/test_client_get_list_unique_paginated_too_many_results", params={})
    assert exc_info.exconly() == snapshot(
        "mreg_api.exceptions.MultipleEntitiesFound: Expected a unique result, got 2 distinct results."
    )


def test_client_get_list_unique_paginated_duplicate_result_ok(
    httpserver: HTTPServer, client: MregClient
) -> None:
    """get_list_unique with _only_ duplicate results is ok."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_unique_paginated_duplicate_result_ok"
    ).respond_with_json(
        {
            "results": [{"foo": "bar"}],
            "count": 1,
            "next": "/test_client_get_list_unique_paginated_duplicate_result_ok?page=2",
            "previous": None,
        }
    )
    httpserver.expect_oneshot_request(
        "/test_client_get_list_unique_paginated_duplicate_result_ok", query_string="page=2"
    ).respond_with_json(
        {
            "results": [{"foo": "bar"}],
            "count": 1,
            "next": None,
            "previous": "/test_client_get_list_unique_paginated_duplicate_result_ok?page=1",
        }
    )
    resp = client.get_list_unique(
        "/test_client_get_list_unique_paginated_duplicate_result_ok", params={}
    )
    assert resp == snapshot({"foo": "bar"})


def test_client_get_list_unique_paginated_no_result(httpserver: HTTPServer, client: MregClient) -> None:
    """No result is None."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_unique_paginated_no_result"
    ).respond_with_json(
        {
            "results": [],
            "count": 0,
            "next": None,
            "previous": None,
        }
    )
    resp = client.get_list_unique("/test_client_get_list_unique_paginated_no_result", params={})
    assert resp is None


def test_client_get_list_unique_non_paginated_no_result(
    httpserver: HTTPServer, client: MregClient
) -> None:
    """No result is None."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_unique_non_paginated_no_result"
    ).respond_with_json([])
    resp = client.get_list_unique("/test_client_get_list_unique_non_paginated_no_result", params={})
    assert resp is None
