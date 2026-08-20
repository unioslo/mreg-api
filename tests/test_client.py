from __future__ import annotations

import datetime
import gc
from typing import Any

import pytest
from httpx import Request as HttpxRequest
from httpx import Response as HttpxResponse
from inline_snapshot import snapshot
from pytest_httpserver import HTTPServer
from werkzeug import Response

from mreg_api.__about__ import __version__
from mreg_api.client import MregClient
from mreg_api.client import check_response
from mreg_api.client import strip_none
from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import DeleteError
from mreg_api.exceptions import GetError
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.exceptions import PatchError
from mreg_api.exceptions import PostError
from mreg_api.managers import HostManager
from mreg_api.models.models import Host
from mreg_api.types import HTTPMethod


def test_client_user_agent() -> None:
    """Test MregClient `user_agent` parameter."""
    expect = f"mreg-api-{__version__}"
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.session.headers["User-Agent"] == expect

    custom_agent = "my-custom-agent/1.0"
    client2 = MregClient(url="http://example.com", domain="example.com", user_agent=custom_agent)
    assert client2.session.headers["User-Agent"] == custom_agent


def test_client_caching(httpserver: HTTPServer) -> None:
    client = MregClient(url=httpserver.url_for(""), domain="example.com", cache=True)
    assert client.cache.is_enabled

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
    hosts1 = client.host.list()

    # Second fetch - should use the cache
    hosts2 = client.host.list()

    assert hosts1 == hosts2

    # Using client directly to access endpoint
    client.clear_cache()
    init_endpoint()

    # First fetch - should hit the server
    resp = client.get(str(Endpoint.Hosts), params=None, ok404=False)

    # Ensure that trying to access the endpoint now raises an exception (oneshot)
    with pytest.raises(GetError) as exc_info:
        client._do_get(str(Endpoint.Hosts), params=None, ok404=False)
    assert "No handler found" in exc_info.value.response.text
    assert exc_info.value.response.status_code == snapshot(500)

    # We know the endpoint doesn't work, so this is certain to go via the cache
    resp2 = client.get(str(Endpoint.Hosts), params=None, ok404=False)

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
    assert client.cache.is_enabled

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

    hosts_pre_mutation = client.host.list()
    assert len(hosts_pre_mutation) == 1

    # Assert we can access the cached data
    hosts_pre_mutation_cached = client.host.list()
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

    hosts_post_mutation = client.host.list()
    assert len(hosts_post_mutation) == 2


def test_client_caching_contextmanager_disabled(httpserver: HTTPServer) -> None:
    client = MregClient(url=httpserver.url_for(""), domain="example.com", cache=True)
    assert client.cache.is_enabled

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
    hosts1 = client.host.list()
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
        hosts2 = client.host.list()
        assert len(client.get_client_history()) == 2

        assert len(hosts1) == 1
        assert len(hosts2) == 2

    # Fetching outside the context manager should use the cache again
    info_pre = client.get_cache_info()
    assert info_pre is not None

    hosts3 = client.host.list()
    assert len(client.get_client_history()) == 2  # History unchanged
    assert len(hosts3) == len(hosts1) == 1

    # Compare cache info
    info_post = client.get_cache_info()
    assert info_post is not None

    assert info_post.hits == info_pre.hits + 1
    assert info_post.misses == info_pre.misses


def test_client_caching_contextmanager_enabled(httpserver: HTTPServer) -> None:
    client = MregClient(url=httpserver.url_for(""), domain="example.com", cache=False)
    assert not client.cache.is_enabled

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
        hosts1 = client.host.list()
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
        hosts2 = client.host.list()
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
    hosts3 = client.host.list()
    assert len(client.get_client_history()) == 2
    assert len(hosts3) == 2


def test_client_destructor_close_cache() -> None:
    """Test that MregClient destructor closes its _own_ but not others' cache on deletion."""
    client1 = MregClient(url="http://example.com", domain="example.com", cache=True)
    client2 = MregClient(url="http://example.com", domain="example.com", cache=True)

    # Ensure both clients have their own cache
    assert client1.cache is not client2.cache

    # Delete client1 and ensure gc collects deleted objects
    del client1
    gc.collect()

    assert client2.cache.is_enabled  # client2's cache should still be enabled


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
    assert strip_none(inp) == expect


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
    httpserver.expect_oneshot_request("/test_client_get_list_paginated_multiple_pages").respond_with_json(
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


def test_client_get_list_paginated_multiple_pages_ok404(httpserver: HTTPServer, client: MregClient) -> None:
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
    assert client.get_list("/test_client_get_list_paginated_multiple_pages_ok404", ok404=True) == snapshot(
        [{"foo": "bar"}]
    )


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
def test_client_get_list_paginated_invalid(httpserver: HTTPServer, client: MregClient, results: Any) -> None:
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
    exc_msg = exc_info.exconly().replace(httpserver.url_for("/"), "<server_url>/")
    assert "Failed to validate paginated JSON" in exc_msg


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
    exc_msg = exc_info.exconly().replace(httpserver.url_for("/"), "<server_url>/")
    assert "Failed to validate JSON list" in exc_msg


def test_client_get_list_non_paginated_invalid_json(httpserver: HTTPServer, client: MregClient) -> None:
    """Non-paginated response with invalid JSON is an error."""
    httpserver.expect_oneshot_request("/test_client_get_list_non_paginated_invalid_json").respond_with_data(
        "[{'key': 'value'}, 'foo',]",  # strings must be double quoted
        content_type="application/json",
    )
    with pytest.raises(MregValidationError) as exc_info:
        client.get_list("/test_client_get_list_non_paginated_invalid_json")
    exc_msg = exc_info.exconly().replace(httpserver.url_for("/"), "<server_url>/")
    assert "Failed to validate JSON list" in exc_msg


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
    resp = client.get_list_unique("/test_client_get_list_unique_paginated_duplicate_result_ok", params={})
    assert resp == snapshot({"foo": "bar"})


def test_client_get_list_unique_paginated_no_result(httpserver: HTTPServer, client: MregClient) -> None:
    """No result is None."""
    httpserver.expect_oneshot_request("/test_client_get_list_unique_paginated_no_result").respond_with_json(
        {
            "results": [],
            "count": 0,
            "next": None,
            "previous": None,
        }
    )
    resp = client.get_list_unique("/test_client_get_list_unique_paginated_no_result", params={})
    assert resp is None


def test_client_get_list_unique_non_paginated_no_result(httpserver: HTTPServer, client: MregClient) -> None:
    """No result is None."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_unique_non_paginated_no_result"
    ).respond_with_json([])
    resp = client.get_list_unique("/test_client_get_list_unique_non_paginated_no_result", params={})
    assert resp is None


def test_client_get_list_unique_invalid_json(httpserver: HTTPServer, client: MregClient) -> None:
    """get_list_unique with multiple unique results is an error."""
    httpserver.expect_oneshot_request(
        "/test_client_get_list_unique_paginated_too_many_results"
    ).respond_with_json(
        {
            # Invalid result: expected list containing a single dict
            "results": ["invalid_not_a_dict"],
            "count": 1,
            "next": None,
            "previous": None,
        }
    )
    with pytest.raises(MregValidationError) as exc_info:
        client.get_list_unique("/test_client_get_list_unique_paginated_too_many_results", params={})

    exc_msg = exc_info.exconly().replace(httpserver.url_for("/"), "<server_url>/")
    assert "Failed to validate JSON mapping" in exc_msg


def test_client_fqdn_expands_bare_hostname() -> None:
    """Fqdn appends the client domain to bare hostnames."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.fqdn("myhost") == "myhost.example.com"


def test_client_fqdn_passes_through_qualified() -> None:
    """Fqdn does not alter hostnames that already contain a dot."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.fqdn("myhost.other.org") == "myhost.other.org"


def test_client_fqdn_strips_trailing_dot() -> None:
    """Fqdn strips trailing dot then returns as-is (already qualified)."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.fqdn("myhost.example.com.") == "myhost.example.com"


def test_client_fqdn_bare_no_domain() -> None:
    """Fqdn returns bare hostname when no domain is configured."""
    client = MregClient(url="http://example.com", domain="")
    assert client.fqdn("myhost") == "myhost"


def test_client_domain_override_context_manager() -> None:
    """domain_override temporarily changes the hostname domain within a context."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.domain == "example.com"

    with client.domain_override("temp.net"):
        assert client.domain == "temp.net"
        assert client.fqdn("myhost") == "myhost.temp.net"

    assert client.domain == "example.com"
    assert client.fqdn("myhost") == "myhost.example.com"


def test_client_domain_override_restores_on_exception() -> None:
    """domain_override restores the domain even when an exception occurs."""
    client = MregClient(url="http://example.com", domain="example.com")

    with pytest.raises(ValueError, match="test error"):
        with client.domain_override("temp.net"):
            assert client.domain == "temp.net"
            raise ValueError("test error")

    assert client.domain == "example.com"


def test_client_domain_override_nested() -> None:
    """Nested domain_override contexts work correctly."""
    client = MregClient(url="http://example.com", domain="example.com")

    with client.domain_override("outer.org"):
        assert client.domain == "outer.org"

        with client.domain_override("inner.net"):
            assert client.domain == "inner.net"

        assert client.domain == "outer.org"

    assert client.domain == "example.com"


def test_client_domain_direct_assignment() -> None:
    """Domain attribute can be changed directly."""
    client = MregClient(url="http://example.com", domain="example.com")
    client.domain = "changed.org"
    assert client.fqdn("web") == "web.changed.org"

    with client.domain_override("temp.net"):
        assert client.domain == "temp.net"

    assert client.domain == "changed.org"


def test_client_manager_composition_usage(client: MregClient, httpserver: HTTPServer) -> None:
    """Test using the HostManager via the MregClient."""
    assert isinstance(client.host, HostManager)

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

    assert client.host.list() == snapshot(
        [
            Host(
                created_at=datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.UTC),
                updated_at=datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.UTC),
                id=1,
                name="host1.example.com",
                ipaddresses=[],
                comment="My comment",
            )
        ]
    )


def get_mreg_client_managers(client: MregClient) -> list[tuple[str, str, type[Any]]]:
    """Return a list of (attribute_name, class_name, class_type) for all managers in the client."""
    # XXX: This is very similar to tests.test_managers.get_resource_managers :\
    managers: list[tuple[str, str, type[Any]]] = []
    for name in dir(client):
        obj = getattr(client, name)
        cls_name = obj.__class__.__name__
        if cls_name.endswith("Manager"):
            managers.append((name, cls_name, obj.__class__))
    return managers


def test_client_manager_composition_snapshot(client: MregClient) -> None:
    """Snapshot test to verify which managers are exported from the client."""
    managers = get_mreg_client_managers(client)
    names = sorted([f"{attr} ({cls_name})" for attr, cls_name, _ in managers])
    assert names == snapshot(
        [
            "atom (AtomManager)",
            "bacnetid (BacnetIDManager)",
            "cname (CNAMEManager)",
            "community (CommunityManager)",
            "delegation (DelegationManager)",
            "dhcphostipv4 (DhcpHostIPv4Manager)",
            "dhcphostipv6 (DhcpHostIPv6Manager)",
            "dhcphostipv6byipv4 (DhcpHostIPv6ByIPv4Manager)",
            "hinfo (HInfoManager)",
            "host (HostManager)",
            "hostgroup (HostGroupManager)",
            "ipaddress (IPAddressManager)",
            "label (LabelManager)",
            "location (LocationManager)",
            "mx (MXManager)",
            "nameserver (NameServerManager)",
            "naptr (NAPTRManager)",
            "network (NetworkManager)",
            "networkpolicy (NetworkPolicyManager)",
            "networkpolicyattribute (NetworkPolicyAttributeManager)",
            "permission (PermissionManager)",
            "ptroverride (PTROverrideManager)",
            "role (RoleManager)",
            "srv (SrvManager)",
            "sshfp (SSHFPManager)",
            "txt (TXTManager)",
            "zone (ZoneManager)",
        ]
    )


def test_client_manager_composition_has_get_method(client: MregClient) -> None:
    # Test that they all export a `get()` method (both via class attr and composed attr on MregClient)
    managers = get_mreg_client_managers(client)
    for client_attr, cls_name, manager_cls in managers:
        assert hasattr(manager_cls, "get"), f"Manager class {cls_name} does not have a get() method"

        # MregClient instance attr exists (`MregClient.<attr>.get()`)
        assert hasattr(getattr(client, client_attr), "get"), (
            f"Manager instance {client.__class__.__name__}.{client_attr} does not have a get() method"
        )


@pytest.mark.parametrize(
    "method,expected_exc",
    [
        ("GET", GetError),
        ("POST", PostError),
        ("PATCH", PatchError),
        ("DELETE", DeleteError),
    ],
)
def test_request_exception_handling(
    httpserver: HTTPServer, client: MregClient, method: HTTPMethod, expected_exc: type[Exception]
) -> None:
    """Test that check_response raises the appropriate exception type on non-2xx response."""
    httpserver.expect_oneshot_request("/test_check_response").respond_with_response(Response(status=400))
    with pytest.raises(expected_exc):
        _ = client.request(method, "/test_check_response")


@pytest.mark.parametrize(
    "method",
    ["GET", "POST", "PATCH", "DELETE"],
)
@pytest.mark.parametrize(
    "status_code",
    [
        200,
        201,
        202,
        204,
    ],
)
def test_check_response_2xx(method: HTTPMethod, status_code: int) -> None:
    """Test that check_response does not raise on non-2xx response."""
    # Valid 2xx response should not raise
    response = HttpxResponse(status_code=status_code, request=HttpxRequest(method=method, url="http://test"))
    check_response(response, method, str(response.request.url))


@pytest.mark.parametrize(
    "method, expected_exc",
    [
        ("GET", GetError),
        ("POST", PostError),
        ("PATCH", PatchError),
        ("DELETE", DeleteError),
    ],
)
@pytest.mark.parametrize(
    "status_code",
    [400, 403, 404, 409, 500],
)
def test_check_response_error(method: HTTPMethod, expected_exc: type[Exception], status_code: int) -> None:
    """Test that check_response raises the correct error on non-2xx response."""
    # Valid 2xx response should not raise
    response = HttpxResponse(status_code=status_code, request=HttpxRequest(method=method, url="http://test"))
    with pytest.raises(expected_exc) as exc_info:
        check_response(response, method, str(response.request.url))
    assert exc_info.type is expected_exc
