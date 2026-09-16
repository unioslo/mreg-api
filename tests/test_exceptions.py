from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import ValidationError as PydanticValidationError
from pytest_httpserver import HTTPServer

from mreg_api import MregClient
from mreg_api.client import last_request_url
from mreg_api.exceptions import APIError
from mreg_api.exceptions import ForceMissing
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import PreconditionError
from mreg_api.exceptions import ResponseError
from mreg_api.exceptions import UnexpectedResponseError
from mreg_api.models import Host


def make_mock_response(
    status_code: int = 400,
    json_body: dict[str, Any] | None = None,
    text_body: str | None = None,
    method: str = "POST",
    url: str = "http://localhost/api/v1/hosts/",
) -> httpx.Response:
    """Create a mock httpx Response for testing."""
    request = httpx.Request(method, url)
    if json_body is not None:
        content = json.dumps(json_body).encode()
        headers = {"content-type": "application/json"}
    else:
        content = (text_body or "").encode()
        headers = {"content-type": "text/plain"}
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=headers,
        request=request,
    )


def test_validation_error_get_host(httpserver: HTTPServer) -> None:
    """Test a validation error stemming from a GET request."""
    client = MregClient(url=httpserver.url_for("/"), domain="example.com")

    httpserver.expect_oneshot_request("/hosts/foobar").respond_with_json(
        {
            "created_at": "2022-06-16T09:15:40.775601+02:00",
            "updated_at": "2024-01-26T10:23:06.631486+01:00",
            "id": 76036,
            "name": "_.--host123_example.com",  # invalid name
            "ipaddresses": [
                {
                    "host": 76036,
                    "created_at": "2022-06-16T09:47:43.761478+02:00",
                    "updated_at": "2022-06-16T12:20:40.722808+02:00",
                    "id": 78492,
                    "macaddress": "e4:54:e8:80:73:73",
                    "ipaddress": "192.168.0.1",
                }
            ],
            "cnames": [],
            "mxs": [],
            "txts": [],
            "ptr_overrides": [],
            "hinfo": None,
            "loc": None,
            "bacnetid": None,
            "contact": "user@example.com",
            "ttl": None,
            "comment": "",
            "zone": 5,
        }
    )
    resp = client.get("/hosts/foobar")
    with pytest.raises(PydanticValidationError) as exc_info:
        Host.model_validate_json(resp.text)

    assert exc_info.value.error_count() == snapshot(1)
    assert [repr(err) for err in exc_info.value.errors(include_url=False)] == snapshot(
        [
            "{'type': 'value_error', 'loc': ('name',), 'msg': 'Value error, Invalid input for hostname: _.--host123_example.com', 'input': '_.--host123_example.com', 'ctx': {'error': InputFailure('Invalid input for hostname: _.--host123_example.com')}}"
        ]
    )

    validationerror = MregValidationError.from_pydantic(exc_info.value)

    # port-number is non-determinstic, so we need to replace that before comparing
    err = validationerror.args[0].replace(f":{httpserver.port}", ":12345")
    assert "Failed to validate Host" in err


def test_validation_error_no_request() -> None:
    """Test a validation error that did not originate from an API request."""
    last_request_url.set(None)  # Ensure no last request URL is set

    with pytest.raises(PydanticValidationError) as exc_info:
        Host.model_validate({"name": "test"})  # Missing required fields

    assert exc_info.value.error_count() == snapshot(5)
    assert [repr(err) for err in exc_info.value.errors(include_url=False)] == snapshot(
        [
            "{'type': 'missing', 'loc': ('created_at',), 'msg': 'Field required', 'input': {'name': 'test'}}",
            "{'type': 'missing', 'loc': ('updated_at',), 'msg': 'Field required', 'input': {'name': 'test'}}",
            "{'type': 'missing', 'loc': ('id',), 'msg': 'Field required', 'input': {'name': 'test'}}",
            "{'type': 'missing', 'loc': ('ipaddresses',), 'msg': 'Field required', 'input': {'name': 'test'}}",
            "{'type': 'missing', 'loc': ('comment',), 'msg': 'Field required', 'input': {'name': 'test'}}",
        ]
    )

    validationerror = MregValidationError.from_pydantic(exc_info.value)
    err_str = str(validationerror)
    assert "Failed to validate Host" in err_str


class TestAPIErrorFormattedMessage:
    """Tests for APIError.formatted_message with MREG error responses."""

    def test_formatted_message_single_error(self) -> None:
        """Test formatted_message with a single MREG error."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [{"code": "required", "detail": "This field is required.", "attr": "name"}],
            },
        )
        error = APIError(response=response)

        assert error.formatted_message() == snapshot("""\
400 Bad Request
  POST http://localhost/api/v1/hosts/
  name: Required - This field is required\
""")

    def test_formatted_message_multiple_errors(self) -> None:
        """Test formatted_message with multiple MREG errors."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [
                    {"code": "required", "detail": "This field is required.", "attr": "name"},
                    {"code": "invalid", "detail": "Enter a valid email.", "attr": "contact"},
                ],
            },
        )
        error = APIError(response=response)

        assert error.formatted_message() == snapshot("""\
400 Bad Request
  POST http://localhost/api/v1/hosts/
  name: Required - This field is required
  contact: Invalid - Enter a valid email\
""")

    def test_formatted_message_error_without_attr(self) -> None:
        """Test formatted_message with an error that has no attr field."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "client_error",
                "errors": [{"code": "authentication_failed", "detail": "Invalid credentials.", "attr": None}],
            },
        )
        error = APIError(response=response)

        assert error.formatted_message() == snapshot("""\
400 Bad Request
  POST http://localhost/api/v1/hosts/
  Authentication Failed - Invalid credentials\
""")

    def test_formatted_message_json_mode(self) -> None:
        """Test formatted_message with json=True returns JSON details."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [{"code": "required", "detail": "This field is required.", "attr": "name"}],
            },
        )
        error = APIError(response=response)

        assert error.formatted_message(json=True) == snapshot("""\
400 Bad Request
  POST http://localhost/api/v1/hosts/
  {
    "type": "validation_error",
    "errors": [
      {
        "code": "required",
        "detail": "This field is required.",
        "attr": "name"
      }
    ]
  }\
""")

    def test_formatted_message_fallback_to_response_text(self) -> None:
        """Test formatted_message falls back to response text when not MREG error format."""
        response = make_mock_response(
            status_code=500,
            text_body="Internal Server Error: Something went wrong",
        )
        error = APIError(response=response)

        assert error.formatted_message() == snapshot("""\
500 Internal Server Error
  POST http://localhost/api/v1/hosts/
  Internal Server Error: Something went wrong\
""")

    def test_formatted_message_not_found_hint(self) -> None:
        """A missing-endpoint 404 renders the version hint, not the raw text."""
        from mreg_api.__about__ import __version__  # noqa: PLC0415

        response = make_mock_response(
            status_code=404,
            method="GET",
            url="http://localhost/api/v1/does/not/exist",
            text_body="The requested resource was not found on this server.",
        )
        error = APIError(response=response)

        assert error.formatted_message() == snapshot(f"""\
404 Not Found
  GET http://localhost/api/v1/does/not/exist
  Endpoint not found: 'does/not/exist'
  This may be because your library version ({__version__}) is:
    - Too old: The endpoint has been removed from the server
    - Too new: You're using a beta feature not yet available on the server\
""")

    def test_formatted_message_404_without_sentinel(self) -> None:
        """A 404 without the sentinel text renders the raw body, no hint."""
        response = make_mock_response(
            status_code=404,
            method="GET",
            url="http://localhost/api/v1/hosts/foo",
            text_body="host not found",
        )
        error = APIError(response=response)

        assert error.formatted_message() == snapshot("""\
404 Not Found
  GET http://localhost/api/v1/hosts/foo
  host not found\
""")

    def test_formatted_message_explicit_message_wins(self) -> None:
        """An explicit message (enrichment override) beats the parsed detail."""
        response = make_mock_response(
            status_code=409,
            json_body={
                "type": "client_error",
                "errors": [{"code": "conflict", "detail": "unhelpful server message", "attr": None}],
            },
        )
        error = APIError("Host is already a member of the group.", response=response)

        assert error.formatted_message() == snapshot("""\
409 Conflict
  POST http://localhost/api/v1/hosts/
  Host is already a member of the group.\
""")

    def test_formatted_message_json_mode_no_structured_errors(self) -> None:
        """Test formatted_message with json=True falls back to plain text when no structured errors."""
        response = make_mock_response(
            status_code=500,
            text_body="Internal Server Error",
        )
        error = APIError(response=response)

        # json=True but no structured errors, should fall back to plain text details
        assert error.formatted_message(json=True) == snapshot("""\
500 Internal Server Error
  POST http://localhost/api/v1/hosts/
  {
    "type": "unknown",
    "errors": []
  }\
""")

    def test_detail_property(self) -> None:
        """Test that detail returns the clean human-readable message."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [{"code": "required", "detail": "This field is required.", "attr": "name"}],
            },
        )
        error = APIError(response=response)
        assert error.detail == "This field is required."

    def test_detail_joins_multiple_errors(self) -> None:
        """Test that detail joins multiple error details with '; '."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [
                    {"code": "required", "detail": "This field is required.", "attr": "name"},
                    {"code": "invalid", "detail": "Enter a valid email.", "attr": "contact"},
                ],
            },
        )
        error = APIError(response=response)
        assert error.detail == "This field is required.; Enter a valid email."

    def test_detail_empty_without_structured_errors(self) -> None:
        """Detail is empty when the response has no parseable MREG errors."""
        response = make_mock_response(status_code=500, text_body="boom")
        error = APIError(response=response)
        assert error.detail == ""

    def test_str_is_formatted_message(self) -> None:
        """Test that str(exc) returns the full formatted_message block."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [{"code": "required", "detail": "This field is required.", "attr": "name"}],
            },
        )
        error = APIError(response=response)
        assert str(error) == error.formatted_message()
        assert str(error) == snapshot("""\
400 Bad Request
  POST http://localhost/api/v1/hosts/
  name: Required - This field is required\
""")


class TestFormattedMessageVerbose:
    """Tests for the verbose (Format 3) rendering."""

    def test_verbose_single_error(self) -> None:
        """Single error renders labeled Request/Attr/Detail/Code fields."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [{"code": "invalid", "detail": "Enter a valid hostname.", "attr": "name"}],
            },
        )
        error = APIError(response=response)
        assert error.formatted_message(verbose=True) == snapshot("""\
400 Bad Request
  Request:   POST http://localhost/api/v1/hosts/
  Attr:      name
  Detail:    Enter a valid hostname.
  Code:      invalid\
""")

    def test_verbose_multiple_errors(self) -> None:
        """Multiple errors render a numbered Errors (N) sub-list."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [
                    {"code": "required", "detail": "This field is required.", "attr": "ipaddress"},
                    {
                        "code": "max_length",
                        "detail": "Ensure this field has no more than 255 characters.",
                        "attr": "comment",
                    },
                ],
            },
        )
        error = APIError(response=response)
        assert error.formatted_message(verbose=True) == snapshot("""\
400 Bad Request
  Request:   POST http://localhost/api/v1/hosts/
  Errors (2):
    [1] ipaddress: This field is required.  (required)
    [2] comment: Ensure this field has no more than 255 characters.  (max_length)\
""")

    def test_verbose_not_found_hint(self) -> None:
        """The endpoint-missing hint renders as a Hint field in verbose mode."""
        from mreg_api.__about__ import __version__  # noqa: PLC0415

        response = make_mock_response(
            status_code=404,
            method="GET",
            url="http://localhost/api/v1/does/not/exist",
            text_body="The requested resource was not found on this server.",
        )
        error = APIError(response=response)
        assert error.formatted_message(verbose=True) == snapshot(f"""\
404 Not Found
  Request:   GET http://localhost/api/v1/does/not/exist
  Hint:      Endpoint not found: 'does/not/exist'
             This may be because your library version ({__version__}) is:
               - Too old: The endpoint has been removed from the server
               - Too new: You're using a beta feature not yet available on the server\
""")


class TestResponseErrors:
    """Errors under `ResponseError`: they always carry an HTTP response."""

    def test_requires_response(self) -> None:
        """ResponseError cannot be constructed without a response."""
        with pytest.raises(TypeError):
            ResponseError("boom")  # pyright: ignore[reportCallIssue]

    def test_unexpected_response_error_carries_2xx(self) -> None:
        """UnexpectedResponseError renders its message atop a success status."""
        response = make_mock_response(
            status_code=201,
            method="POST",
            text_body="",
        )
        error = UnexpectedResponseError("Failed to fetch host after creation.", response=response)
        assert error.status_code == 201
        assert error.formatted_message() == snapshot("""\
201 Created
  POST http://localhost/api/v1/hosts/
  Failed to fetch host after creation.\
""")


class TestPreconditionErrors:
    """Client-side guards: responseless, raised before any HTTP request."""

    def test_precondition_error_is_responseless(self) -> None:
        """PreconditionError is a plain message-only error with no response."""
        error = PreconditionError("Zone has 3 registered entries. Can not delete.")
        assert str(error) == "Zone has 3 registered entries. Can not delete."
        assert not hasattr(error, "response")

    def test_force_missing_is_a_precondition_error(self) -> None:
        """ForceMissing is a force-overridable PreconditionError; catchable as either."""
        error = ForceMissing("Atom 'a' used in roles: r1")
        assert isinstance(error, PreconditionError)
        assert not hasattr(error, "response")
