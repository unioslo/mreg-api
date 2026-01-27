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
from mreg_api.exceptions import MregValidationError
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
    if MregClient._instances:  # pyright: ignore[reportPrivateUsage]
        MregClient.reset_instance()
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
    assert err == snapshot(
        """\
Failed to validate Host response from GET http://localhost:12345/hosts/foobar
  Input: _.--host123_example.com
  Errors:
    Field: name
    Reason: Value error, Invalid input for hostname: _.--host123_example.com\
"""
    )


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
    assert err_str == snapshot("""\
Failed to validate Host
  Input: {'name': 'test'}
  Errors:
    Field: created_at
    Reason: Field required

    Field: updated_at
    Reason: Field required

    Field: id
    Reason: Field required

    Field: ipaddresses
    Reason: Field required

    Field: comment
    Reason: Field required\
""")


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
        error = APIError("Request failed", response=response)

        assert error.formatted_message() == snapshot("""\
POST "http://localhost/api/v1/hosts/": 400: Bad Request
Validation Error: Required - This field is required.: name\
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
        error = APIError("Request failed", response=response)

        assert error.formatted_message() == snapshot("""\
POST "http://localhost/api/v1/hosts/": 400: Bad Request
Validation Error: Required - This field is required.: name; Invalid - Enter a valid email.: contact\
""")

    def test_formatted_message_error_without_attr(self) -> None:
        """Test formatted_message with an error that has no attr field."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "client_error",
                "errors": [
                    {"code": "authentication_failed", "detail": "Invalid credentials.", "attr": None}
                ],
            },
        )
        error = APIError("Request failed", response=response)

        assert error.formatted_message() == snapshot("""\
POST "http://localhost/api/v1/hosts/": 400: Bad Request
Client Error: Authentication Failed - Invalid credentials.\
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
        error = APIError("Request failed", response=response)

        assert error.formatted_message(json=True) == snapshot("""\
POST "http://localhost/api/v1/hosts/": 400: Bad Request
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
        error = APIError("Request failed", response=response)

        assert error.formatted_message() == snapshot("""\
POST "http://localhost/api/v1/hosts/": 500: Internal Server Error
Internal Server Error: Something went wrong\
""")

    def test_formatted_message_fallback_to_exception_args(self) -> None:
        """Test formatted_message falls back to exception args when no response."""
        error = APIError("Connection timed out", response=None)

        assert error.formatted_message() == snapshot("Connection timed out")

    def test_formatted_message_json_mode_no_structured_errors(self) -> None:
        """Test formatted_message with json=True falls back to plain text when no structured errors."""
        response = make_mock_response(
            status_code=500,
            text_body="Internal Server Error",
        )
        error = APIError("Request failed", response=response)

        # json=True but no structured errors, should fall back to plain text details
        assert error.formatted_message(json=True) == snapshot("""\
POST "http://localhost/api/v1/hosts/": 500: Internal Server Error
Internal Server Error\
""")

    def test_details_property(self) -> None:
        """Test that details property returns formatted error string."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [{"code": "required", "detail": "This field is required.", "attr": "name"}],
            },
        )
        error = APIError("Request failed", response=response)

        assert error.details == snapshot("Validation Error: Required - This field is required.: name")

    def test_details_json_property(self) -> None:
        """Test that details_json property returns JSON string."""
        response = make_mock_response(
            status_code=400,
            json_body={
                "type": "validation_error",
                "errors": [{"code": "required", "detail": "This field is required.", "attr": "name"}],
            },
        )
        error = APIError("Request failed", response=response)

        assert error.details_json == snapshot("""\
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

    def test_details_empty_when_no_response(self) -> None:
        """Test that details returns empty string when no response."""
        error = APIError("Connection failed", response=None)
        assert error.details == ""

    def test_details_json_none_when_no_response(self) -> None:
        """Test that details_json returns None when no response."""
        error = APIError("Connection failed", response=None)
        assert error.details_json is None
