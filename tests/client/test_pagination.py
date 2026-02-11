from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot
from pytest_httpserver import HTTPServer
from werkzeug import Response

from mreg_api.client import MregClient
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import MultipleEntitiesFound


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


