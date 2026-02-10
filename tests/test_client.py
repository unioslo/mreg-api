from __future__ import annotations

from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from pytest_httpserver import HTTPServer
from werkzeug import Response

from mreg_api import models
from mreg_api.client import MregClient
from mreg_api.exceptions import EntityNotFound, GetError, MregValidationError, MultipleEntitiesFound
from mreg_api.models.fields import HostName, hostname_domain
from mreg_api.models.manager import ModelList, ModelManager, to_snake_case
from mreg_api.models.models import Host


def test_client_not_singleton() -> None:
    client1 = MregClient(url="http://example.com", domain="example.com", timeout=30)
    client2 = MregClient(url="http://example.com", domain="example.com", timeout=30)
    assert client1 is not client2


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
    hosts1 = client.host().get_list()

    # Second fetch - should use the cache
    hosts2 = client.host().get_list()

    assert hosts1 == hosts2

    # Using client directly to access endpoint
    _ = client.clear_cache()  # Clear cache to test caching again
    init_endpoint()

    # First fetch - should hit the server
    resp = client.get(str(Host.endpoint()), params=None, ok404=False)
    assert resp is not None

    # Ensure that trying to access the endpoint now raises an exception (oneshot)
    with pytest.raises(GetError) as exc_info:
        _ = client._do_get(str(Host.endpoint()), params=None, ok404=False)
    assert exc_info.value.response is not None
    assert "No handler found" in exc_info.value.response.text
    assert exc_info.value.response.status_code == snapshot(500)

    # We know the endpoint doesn't work, so this is certain to go via the cache
    resp2 = client.get(str(Host.endpoint()), params=None, ok404=False)
    assert resp2 is not None

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

    httpserver.expect_oneshot_request("/api/v1/hosts/", method=method).respond_with_json({"detail": "Mutation successful"})

    hosts_pre_mutation = client.host().get_list()
    assert len(hosts_pre_mutation) == 1

    # Assert we can access the cached data
    hosts_pre_mutation_cached = client.host().get_list()
    assert len(hosts_pre_mutation_cached) == 1

    # We don't care about the mutation response or respecting what it would actually do
    if method == "POST":
        _ = client.post("/api/v1/hosts/", params={"name": "newhost.example.com"})
    elif method == "PATCH":
        _ = client.patch("/api/v1/hosts/", params={"comment": "Updated comment"})
    elif method == "DELETE":
        _ = client.delete("/api/v1/hosts/")

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

    hosts_post_mutation = client.host().get_list()
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
    hosts1 = client.host().get_list()
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
        hosts2 = client.host().get_list()
        assert len(client.get_client_history()) == 2

        assert len(hosts1) == 1
        assert len(hosts2) == 2

    # Fetching outside the context manager should use the cache again
    info_pre = client.get_cache_info()
    assert info_pre is not None

    hosts3 = client.host().get_list()
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
        hosts1 = client.host().get_list()
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
        hosts2 = client.host().get_list()
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
    hosts3 = client.host().get_list()
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
    httpserver.expect_oneshot_request("/test_client_get_list_paginated_multiple_pages").respond_with_json(
        {
            "results": [{"foo": "bar"}],
            "count": 1,
            "next": "/test_client_get_list_paginated_multiple_pages?page=2",
            "previous": None,
        }
    )
    httpserver.expect_oneshot_request("/test_client_get_list_paginated_multiple_pages", query_string="page=2").respond_with_json(
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
    httpserver.expect_oneshot_request("/test_client_get_list_paginated_multiple_pages_ok404").respond_with_json(
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
    assert client.get_list("/test_client_get_list_paginated_multiple_pages_ok404", ok404=True) == snapshot([{"foo": "bar"}])


def test_client_get_list_paginated_multiple_pages_inconsistent_count(httpserver: HTTPServer, client: MregClient) -> None:
    """Inconsistent count in paginated response is ignored."""
    httpserver.expect_oneshot_request("/test_client_get_list_paginated_multiple_pages_inconsistent_count").respond_with_json(
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
        _ = client.get_list("/test_client_get_list_paginated_invalid")
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
        _ = client.get_list("/test_client_get_list_non_paginated_non_array")
    exc_msg = exc_info.exconly().replace(httpserver.url_for("/"), "<server_url>/")
    assert "Failed to validate JSON list" in exc_msg


def test_manager_get_list_returns_model_list(httpserver: HTTPServer, client: MregClient) -> None:
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

    hosts = client.host().get_list()
    assert isinstance(hosts, ModelList)
    assert [str(host.name) for host in hosts] == ["host1.example.com"]


def test_model_list_delete_bulk(httpserver: HTTPServer, client: MregClient) -> None:
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
    httpserver.expect_oneshot_request("/api/v1/hosts/host1.example.com", method="DELETE").respond_with_json({})
    httpserver.expect_oneshot_request("/api/v1/hosts/host2.example.com", method="DELETE").respond_with_json({})

    hosts = client.host().get_list()
    hosts.delete()


def test_model_list_patch_dispatches() -> None:
    class Patchable:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], bool]] = []

        def patch(self, fields: dict[str, Any] | None = None, validate: bool = True, **field_kwargs: Any) -> Patchable:
            payload = fields if fields is not None else field_kwargs
            self.calls.append((payload, validate))
            return self

    one = Patchable()
    two = Patchable()
    objects = ModelList([one, two])

    patched = objects.patch(comment="updated", validate=False)

    assert isinstance(patched, ModelList)
    assert patched == [one, two]
    assert one.calls == [({"comment": "updated"}, False)]
    assert two.calls == [({"comment": "updated"}, False)]


def test_model_list_delete_raises_on_failure() -> None:
    class Deletable:
        def __init__(self, ok: bool) -> None:
            self.ok = ok

        def delete(self) -> bool:
            return self.ok

    objects = ModelList([Deletable(True), Deletable(False)])
    with pytest.raises(RuntimeError, match="Failed to delete"):
        _ = objects.delete()


def test_model_list_patch_raises_if_missing_method() -> None:
    objects = ModelList([object()])
    with pytest.raises(TypeError, match="Cannot bulk-patch"):
        _ = objects.patch({"comment": "updated"})


def test_model_list_patch_matches_manual_loop() -> None:
    class Patchable:
        def __init__(self, identifier: int) -> None:
            self.identifier = identifier
            self.calls: list[tuple[dict[str, Any], bool]] = []

        def patch(self, fields: dict[str, Any] | None = None, validate: bool = True, **field_kwargs: Any) -> Patchable:
            payload = fields if fields is not None else field_kwargs
            self.calls.append((payload, validate))
            return self

    bulk_objects = ModelList([Patchable(1), Patchable(2)])
    loop_objects = ModelList([Patchable(1), Patchable(2)])

    bulk_result = bulk_objects.patch(comment="hi")

    loop_result = ModelList[Patchable]()
    for host in loop_objects:
        loop_result.append(host.patch(comment="hi"))

    assert isinstance(bulk_result, ModelList)
    assert bulk_result == bulk_objects
    assert [obj.identifier for obj in bulk_result] == [obj.identifier for obj in loop_result]
    assert [obj.calls for obj in bulk_objects] == [obj.calls for obj in loop_objects]


def test_model_list_patch_raw_dispatches() -> None:
    class Patchable:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], bool]] = []

        def patch_raw(self, fields: dict[str, Any], validate: bool = True) -> Patchable:
            self.calls.append((fields, validate))
            return self

    one = Patchable()
    two = Patchable()
    objects = ModelList([one, two])

    patched = objects.patch_raw({"comment": "updated"}, validate=False)

    assert isinstance(patched, ModelList)
    assert patched == [one, two]
    assert one.calls == [({"comment": "updated"}, False)]
    assert two.calls == [({"comment": "updated"}, False)]


def test_model_list_patch_typed_dispatches() -> None:
    class Patchable:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], bool]] = []

        def patch_typed(self, *, validate: bool = True, **field_kwargs: Any) -> Patchable:
            self.calls.append((field_kwargs, validate))
            return self

    one = Patchable()
    two = Patchable()
    objects = ModelList([one, two])

    patched = objects.patch_typed(comment="updated", validate=False)

    assert isinstance(patched, ModelList)
    assert patched == [one, two]
    assert one.calls == [({"comment": "updated"}, False)]
    assert two.calls == [({"comment": "updated"}, False)]


def test_model_list_patch_typed_raises_if_missing_method() -> None:
    objects = ModelList([object()])
    with pytest.raises(TypeError, match="Cannot bulk-patch_typed"):
        _ = objects.patch_typed(comment="updated")


def test_model_manager_get_by_id_or_raise_returns_object() -> None:
    class DummyModel:
        @classmethod
        def get_by_id(cls, client: object, _id: int, _manager: bool = False) -> object | None:
            _ = (client, _id, _manager)
            return {"id": 1}

    manager = ModelManager(MregClient(), DummyModel)
    assert manager.get_by_id_or_raise(1) == {"id": 1}


def test_model_manager_get_by_id_or_raise_raises_with_default_message() -> None:
    class DummyModel:
        @classmethod
        def get_by_id(cls, client: object, _id: int, _manager: bool = False) -> object | None:
            _ = (client, _id, _manager)
            return None

    manager = ModelManager(MregClient(), DummyModel)
    with pytest.raises(EntityNotFound, match="DummyModel with ID 1 not found"):
        _ = manager.get_by_id_or_raise(1)


def test_client_get_list_non_paginated_invalid_json(httpserver: HTTPServer, client: MregClient) -> None:
    """Non-paginated response with invalid JSON is an error."""
    httpserver.expect_oneshot_request("/test_client_get_list_non_paginated_invalid_json").respond_with_data(
        "[{'key': 'value'}, 'foo',]",  # strings must be double quoted
        content_type="application/json",
    )
    with pytest.raises(MregValidationError) as exc_info:
        _ = client.get_list("/test_client_get_list_non_paginated_invalid_json")
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


def test_client_get_list_unique_paginated_too_many_results(httpserver: HTTPServer, client: MregClient) -> None:
    """get_list_unique with multiple unique results is an error."""
    httpserver.expect_oneshot_request("/test_client_get_list_unique_paginated_too_many_results").respond_with_json(
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
        _ = client.get_list_unique("/test_client_get_list_unique_paginated_too_many_results", params={})
    assert exc_info.exconly() == snapshot(
        "mreg_api.exceptions.MultipleEntitiesFound: Expected a unique result, got 2 distinct results."
    )


def test_client_get_list_unique_paginated_duplicate_result_ok(httpserver: HTTPServer, client: MregClient) -> None:
    """get_list_unique with _only_ duplicate results is ok."""
    httpserver.expect_oneshot_request("/test_client_get_list_unique_paginated_duplicate_result_ok").respond_with_json(
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
    httpserver.expect_oneshot_request("/test_client_get_list_unique_non_paginated_no_result").respond_with_json([])
    resp = client.get_list_unique("/test_client_get_list_unique_non_paginated_no_result", params={})
    assert resp is None


def test_client_get_list_unique_invalid_json(httpserver: HTTPServer, client: MregClient) -> None:
    """get_list_unique with multiple unique results is an error."""
    httpserver.expect_oneshot_request("/test_client_get_list_unique_paginated_too_many_results").respond_with_json(
        {
            # Invalid result: expected list containing a single dict
            "results": ["invalid_not_a_dict"],
            "count": 1,
            "next": None,
            "previous": None,
        }
    )
    with pytest.raises(MregValidationError) as exc_info:
        _ = client.get_list_unique("/test_client_get_list_unique_paginated_too_many_results", params={})

    exc_msg = exc_info.exconly().replace(httpserver.url_for("/"), "<server_url>/")
    assert "Failed to validate JSON mapping" in exc_msg


def test_client_set_domain() -> None:
    """set_domain changes the hostname domain used for validation."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert hostname_domain.get() == "example.com"
    assert HostName.validate_hostname("myhost") == "myhost.example.com"

    _ = client.set_domain("other.org")
    assert client.get_domain() == "other.org"
    assert client.get_domain() == hostname_domain.get()
    assert HostName.validate_hostname("myhost") == "myhost.other.org"


def test_client_reset_domain() -> None:
    """reset_domain restores the hostname domain to the value from initialization."""
    assert hostname_domain.get() == snapshot("uio.no")  # default domain

    client = MregClient(url="http://example.com", domain="example.com")
    assert hostname_domain.get() == "example.com"

    # Set another domain
    _ = client.set_domain("other.org")
    assert hostname_domain.get() == "other.org"

    # Reset to original
    client.reset_domain()
    assert client.get_domain() == snapshot("example.com")

    # Can be called multiple times (also in destructor)
    client.reset_domain()
    client.reset_domain()
    assert client.get_domain() == snapshot("example.com")


def test_client_reset_domain_after_multiple_set_domain() -> None:
    """reset_domain always restores to the initialization value, not the previous value."""
    client = MregClient(url="http://example.com", domain="example.com")

    _ = client.set_domain("first.org")
    _ = client.set_domain("second.org")
    _ = client.set_domain("third.org")
    assert client.get_domain() == "third.org"
    assert client.get_domain() == hostname_domain.get()

    # reset_domain should go back to example.com, not third.org or second.org
    client.reset_domain()
    assert client.get_domain() == "example.com"
    assert client.get_domain() == hostname_domain.get()


def test_client_domain_override_context_manager() -> None:
    """domain_override temporarily changes the hostname domain within a context."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.get_domain() == "example.com"

    with client.domain_override("temp.net"):
        assert client.get_domain() == "temp.net"
        assert HostName.validate_hostname("myhost") == "myhost.temp.net"

    # After exiting context, domain should be restored
    assert client.get_domain() == "example.com"
    assert HostName.validate_hostname("myhost") == "myhost.example.com"


def test_client_domain_override_restores_on_exception() -> None:
    """domain_override restores the domain even when an exception occurs."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.get_domain() == "example.com"

    with pytest.raises(ValueError, match="test error"):
        with client.domain_override("temp.net"):
            assert client.get_domain() == "temp.net"
            raise ValueError("test error")

    # Domain should still be restored after exception
    assert client.get_domain() == "example.com"


def test_client_domain_override_nested() -> None:
    """Nested domain_override contexts work correctly."""
    client = MregClient(url="http://example.com", domain="example.com")
    assert client.get_domain() == "example.com"

    with client.domain_override("outer.org"):
        assert client.get_domain() == "outer.org"

        with client.domain_override("inner.net"):
            assert client.get_domain() == "inner.net"

        # After inner context exits, should restore to outer value
        assert client.get_domain() == "outer.org"

    # After outer context exits, should restore to original
    assert client.get_domain() == "example.com"


def test_client_domain_override_after_set_domain() -> None:
    """domain_override works correctly after set_domain has been called."""
    client = MregClient(url="http://example.com", domain="example.com")

    _ = client.set_domain("changed.org")
    assert client.get_domain() == "changed.org"

    with client.domain_override("temp.net"):
        assert client.get_domain() == "temp.net"

    # After context exits, should restore to the value before the context (changed.org)
    assert client.get_domain() == "changed.org"

    # reset_domain should still restore to original initialization value
    client.reset_domain()
    assert client.get_domain() == "example.com"


def test_client_model_composition(client: MregClient, httpserver: HTTPServer) -> None:
    """MregClient has models composed as attributes for easy access."""
    assert hasattr(client, "host")
    assert client.host().model is Host

    # Test that we can use the composed model to make requests
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

    hosts = client.host().get_list()
    assert [(host.id, host.name, host.comment) for host in hosts] == snapshot([(1, "host1.example.com", "My comment")])


def _client_models() -> list[type]:
    client_models: list[type] = []
    for model in models.__all__:
        model_obj = getattr(models, model)
        # All models with a get or fetch method should be accessible via the client
        if any(hasattr(model_obj, attr) for attr in ["get", "fetch"]):
            client_models.append(model_obj)
    return client_models


@pytest.mark.parametrize("model", _client_models())
def test_client_model_composition_dynamic(model: type, client: MregClient) -> None:
    """Ensure all models with get/fetch are accessible via client attributes."""
    attr_name = to_snake_case(model.__name__)
    assert hasattr(client, attr_name)
    manager: Any = getattr(client, attr_name)
    if callable(manager):
        manager = manager()
    typed_manager = cast(ModelManager[Any], manager)
    assert typed_manager.model is model
