"""MREG API client for managing HTTP sessions and authentication."""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Self
from typing import TypeVar
from typing import get_origin
from typing import overload
from urllib.parse import urljoin
from uuid import uuid4

import requests
from pydantic import BaseModel
from pydantic import TypeAdapter
from pydantic import field_validator
from requests import Response

from mreg_api.__about__ import __version__
from mreg_api.exceptions import APIError
from mreg_api.exceptions import LoginFailedError
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.exceptions import TooManyResults
from mreg_api.models.endpoints import Endpoint
from mreg_api.models.errors import parse_mreg_error
from mreg_api.types import Json
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter

logger = logging.getLogger(__name__)

# Context variables for tracking request info (used in error reporting)
last_request_url: ContextVar[str | None] = ContextVar("last_request_url", default=None)
last_request_method: ContextVar[str | None] = ContextVar("last_request_method", default=None)


T = TypeVar("T")

JsonMappingValidator = TypeAdapter(JsonMapping)


class MregApiClient:
    """Client for interacting with MREG API.

    This client manages HTTP sessions, authentication, and provides
    methods for making API requests. It is designed to be used as
    a singleton, similar to MregCliConfig.

    Authentication modes:
    1. Token: Provide token directly via set_token()
    2. Token file: Load from file via load_token_from_file()
    3. Username/password: Call login() with credentials

    Example:
        >>> client = MregApiClient()
        >>> client.login("username", "password")
        >>> from mreg_api.models import Host
        >>> Host.get_by_any_means("example.uio.no")

        Or with token:
        >>> client = MregApiClient()
        >>> client.set_token("your-token-here")
        >>> from mreg_api.models import Host
        >>> Host.get_by_any_means("example.uio.no")
    """

    _instance: ClassVar[Self | None] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:  # noqa: ARG004 # __new__ needs to match __init__
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        url: str = "https://mreg.uio.no",
        domain: str = "uio.no",
        user: str | None = None,
        timeout: int = 20,
        cache: bool = True,
        cache_ttl: int = 300,
    ) -> None:
        """Initialize the client (only once for singleton)."""
        if self._initialized:
            return

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"mreg-api-{__version__}"})

        self.url = url
        self.domain = domain
        self.timeout = timeout
        self.user = user

        self.cache = cache
        self.cache_ttl = cache_ttl

        # State
        self._token: str | None = None

        self._initialized = True

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def set_token(self, token: str) -> None:
        """Set the authorization token for API requests.

        Args:
            token: Bearer token for authentication

        """
        self.session.headers.update({"Authorization": f"Token {token}"})
        self._token = token

    def get_token(self) -> str | None:
        """Get the current authorization token if set.

        Returns:
            Token string or None if not authenticated

        """
        return self._token

    def set_correlation_id(self, suffix: str) -> str:
        """Set correlation ID for request tracking.

        Args:
            suffix: Human-readable suffix for the correlation ID

        Returns:
            The generated correlation ID

        """
        suffix = re.sub(r"\s+", "_", suffix)
        correlation_id = f"{uuid4()}-{suffix}"
        self.session.headers.update({"X-Correlation-ID": correlation_id})
        return correlation_id

    def get_correlation_id(self) -> str:
        """Get the current correlation ID.

        Returns:
            Current correlation ID or empty string

        """
        return str(self.session.headers.get("X-Correlation-ID", ""))

    def login(self, username: str, password: str) -> str:
        """Authenticate with username and password.

        Args:
            username: MREG username
            password: MREG password

        Raises:
            LoginFailedError: If authentication fails

        Returns:
            The authentication token

        """
        token_url = urljoin(self.url, Endpoint.TokenAuth)
        logger.info("Authenticating %s @ %s", username, token_url)

        try:
            result = requests.post(
                token_url,
                {"username": username, "password": password},
                timeout=self.timeout,
            )
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            raise LoginFailedError(f"Connection failed: {e}") from e

        if not result.ok:
            err = parse_mreg_error(result)
            msg = err.as_str() if err else result.text
            raise LoginFailedError(msg)

        token = result.json().get("token")
        if not token:
            raise LoginFailedError("No token received from server")
        token = str(token)
        self.set_token(token)

        logger.info("Authentication successful for %s", username)

        return token

    def logout(self) -> None:
        """Logout from MREG (invalidate token on server)."""
        path = urljoin(self.url, Endpoint.TokenLogout)
        try:
            self.session.post(path, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            logger.warning("Failed to log out: %s", e)

    def test_connection(self) -> bool:
        """Test if the current token/connection is valid.

        Returns:
            True if connection is valid, False otherwise

        """
        try:
            ret = self.session.get(
                urljoin(self.url, Endpoint.Hosts),
                params={"page_size": 1},
                timeout=5,
            )
            return ret.status_code != 401
        except requests.exceptions.ConnectionError:
            return False

    def _strip_none(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively strip None values from a dictionary."""
        new: dict[str, Any] = {}
        for key, value in data.items():
            if value is not None:
                if isinstance(value, dict):
                    v = self._strip_none(value)  # pyright: ignore[reportUnknownArgumentType]
                    if v:
                        new[key] = v
                else:
                    new[key] = value
        return new

    def _result_check(self, result: Response, operation_type: str, url: str) -> None:
        """Check the result of a request and raise on error."""
        if not result.ok:
            if err := parse_mreg_error(result):
                res_text = err.as_json_str()
            elif result.status_code == 404:
                endpoint = url.split("/api/v1/")[-1] if "/api/v1/" in url else url
                res_text = (
                    f"Endpoint not found: '{endpoint}'\n"
                    f"This may be because your library version ({__version__}) is:\n"
                    f"  - Too old: The endpoint has been removed from the server\n"
                    f"  - Too new: You're using a beta feature not yet available on the server"
                )
            else:
                res_text = result.text
            message = f'{operation_type} "{url}": {result.status_code}: {result.reason}\n{res_text}'
            raise APIError(message, result)

    def request(
        self,
        method: Literal["get", "post", "patch", "delete"],
        path: str,
        params: QueryParams | None = None,
        ok404: bool = False,
        **data: Any,
    ) -> Response | None:
        """Make an HTTP request to the MREG API.

        Args:
            method: HTTP method
            path: API path (relative to base URL)
            params: Query parameters
            ok404: Whether to return None on 404 instead of raising
            **data: Request body data

        Returns:
            Response object or None if ok404=True and status is 404

        Raises:
            APIError: If request fails

        """
        if params is None:
            params = {}

        url = urljoin(self.url, path)

        # Build log URL
        logurl = url
        if method.upper() == "GET" and params:
            logurl = logurl + "?" + "&".join(f"{k}={v}" for k, v in params.items())

        logger.info("Request: %s %s [%s]", method.upper(), logurl, self.get_correlation_id())

        if method.upper() != "GET" and params:
            logger.debug("Params: %s", params)

        if data:
            logger.debug("Data: %s", data)

        # Strip None values from data (except for PATCH)
        if data and method != "patch":
            data = self._strip_none(data)

        # Select the appropriate session method
        func = getattr(self.session, method)

        result = func(
            url,
            params=params,
            json=data or None,
            timeout=self.timeout,
        )

        # Update context variables for error reporting
        last_request_url.set(logurl)
        last_request_method.set(method)

        # Log response
        request_id = result.headers.get("X-Request-Id", "?")
        correlation_id = result.headers.get("X-Correlation-ID", "?")
        id_str = f"[R:{request_id} C:{correlation_id}]"
        log_message = f"Response: {method.upper()} {logurl} {result.status_code} {id_str}"

        if result.status_code >= 300:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Handle 404 if ok404 is True
        if result.status_code == 404 and ok404:
            return None

        self._result_check(result, method.upper(), url)
        return result

    @overload
    def get(self, path: str, params: QueryParams | None, ok404: Literal[True]) -> Response | None: ...

    @overload
    def get(self, path: str, params: QueryParams | None, ok404: Literal[False]) -> Response: ...

    @overload
    def get(self, path: str, params: QueryParams | None = ..., *, ok404: bool) -> Response | None: ...

    @overload
    def get(self, path: str, params: QueryParams | None = ...) -> Response: ...

    def get(self, path: str, params: QueryParams | None = None, ok404: bool = False) -> Response | None:
        """Make a standard get request."""
        return self._do_get(path, params, ok404)

    def _do_get(
        self, path: str, params: QueryParams | None = None, ok404: bool = False
    ) -> Response | None:
        """Perform a GET request.

        Separated out from get(), so that we can patch this function with a memoized version
        without affecting other modules that do `from mreg_api.utilities.api import get`, as
        they would then operate on the unpatched version instead of the modified one,
        since their `get` symbol differs from the `get` symbol in this module
        in a scenario where `get` itself is patched _after_ it is imported elsewhere.

        The caching module can modify this function instead of `get`,
        allowing other modules to be oblivious of the caching behavior, and freely
        import `get` into their namespaces.

        Yes... Patching sucks when you have other modules that import scoped symbols
        into their own namespace.
        """
        if params is None:
            params = {}
        return self.request("get", path, params=params, ok404=ok404)

    def post(
        self,
        path: str,
        params: QueryParams | None = None,
        **kwargs: Any,
    ) -> Response | None:
        """Make a POST request."""
        return self.request("post", path, params, **kwargs)

    def patch(
        self,
        path: str,
        params: QueryParams | None = None,
        **kwargs: Any,
    ) -> Response | None:
        """Make a PATCH request."""
        return self.request("patch", path, params, **kwargs)

    def delete(
        self,
        path: str,
        params: QueryParams | None = None,
        **kwargs: Any,
    ) -> Response | None:
        """Make a DELETE request."""
        return self.request("delete", path, params, **kwargs)

    def get_list(
        self,
        path: str,
        params: QueryParams | None = None,
        ok404: bool = False,
        limit: int | None = None,
    ) -> list[Json]:
        """Make a get request that produces a list.

        Will iterate over paginated results and return result as list. If the number of hits is
        greater than limit, the function will raise an exception.

        :param path: The path to the API endpoint.
        :param params: The parameters to pass to the API endpoint.
        :param ok404: Whether to allow 404 responses.
        :param limit: The maximum number of hits to allow.
            If the number of hits is greater than this, the function will raise an exception.
            Set to None to disable this check.
        :raises CliError: If the result from get_list_generic is not a list.

        :returns: A list of dictionaries.
        """
        return self.get_list_generic(path, params, ok404, limit, expect_one_result=False)

    def get_list_in(
        self,
        path: str,
        search_field: str,
        search_values: list[int],
        ok404: bool = False,
    ) -> list[Json]:
        """Get a list of items by a key value pair.

        :param path: The path to the API endpoint.
        :param search_field: The field to search for.
        :param search_values: The values to search for.
        :param ok404: Whether to allow 404 responses.

        :returns: A list of dictionaries.
        """
        return self.get_list(
            path,
            params={f"{search_field}__in": ",".join(str(x) for x in search_values)},
            ok404=ok404,
        )

    def get_item_by_key_value(
        self,
        path: str,
        search_field: str,
        search_value: str | int,
        ok404: bool = False,
    ) -> None | JsonMapping:
        """Get an item by a key value pair.

        :param path: The path to the API endpoint.
        :param search_field: The field to search for.
        :param search_value: The value to search for.
        :param ok404: Whether to allow 404 responses.

        :raises CliWarning: If no result was found and ok404 is False.

        :returns: A single dictionary, or None if no result was found and ok404 is True.
        """
        return self.get_list_unique(path, params={search_field: search_value}, ok404=ok404)

    def get_list_unique(
        self,
        path: str,
        params: QueryParams | None = None,
        ok404: bool = False,
    ) -> None | JsonMapping:
        """Do a get request that returns a single result from a search.

        :param path: The path to the API endpoint.
        :param params: The parameters to pass to the API endpoint.
        :param ok404: Whether to allow 404 responses.

        :raises CliWarning: If no result was found and ok404 is False.

        :returns: A single dictionary, or None if no result was found and ok404 is True.
        """
        ret = self.get_list_generic(path, params, ok404, expect_one_result=True)
        if not ret:
            return None
        try:
            return JsonMappingValidator.validate_python(ret)
        except ValueError as e:
            raise MregValidationError(f"Failed to validate response from {path}: {e}") from e

    @overload
    def get_list_generic(
        self,
        path: str,
        params: QueryParams | None = ...,
        ok404: bool = ...,
        limit: int | None = ...,
        expect_one_result: Literal[False] = False,
    ) -> list[Json]: ...

    @overload
    def get_list_generic(
        self,
        path: str,
        params: QueryParams | None = ...,
        ok404: bool = ...,
        limit: int | None = ...,
        expect_one_result: Literal[True] = True,
    ) -> Json: ...

    def get_list_generic(
        self,
        path: str,
        params: QueryParams | None = None,
        ok404: bool = False,
        limit: int | None = None,
        expect_one_result: bool | None = False,
    ) -> Json | list[Json]:
        """Make a get request that produces a list.

        Will iterate over paginated results and return result as list. If the number of hits is
        greater than limit, the function will raise an exception.

        :param path: The path to the API endpoint.
        :param params: The parameters to pass to the API endpoint.
        :param ok404: Whether to allow 404 responses.
        :param limit: The maximum number of hits to allow.
            If the number of hits is greater than this, the function will raise an exception.
            Set to None to disable this check.
        :param expect_one_result: If True, expect exactly one result and return it as a list.

        :raises CliError: If expect_one_result is True and the number of results is not zero or one.
        :raises CliError: If expect_one_result is True and there is a response without a 'results' key.
        :raises CliError: If the number of hits is greater than limit.

        :returns: A list of dictionaries or a dictionary if expect_one_result is True.
        """
        response = self.get(path, params)

        # Non-paginated results, return them directly
        if "count" not in response.text:
            return validate_list_response(response)

        resp = validate_paginated_response(response)

        if limit and resp.count > abs(limit):
            raise TooManyResults(f"Too many hits ({resp.count}), please refine your search criteria.")

        # Iterate over all pages and collect the results
        ret: list[Json] = resp.results
        while resp.next:
            response = self.get(resp.next, ok404=ok404)
            if response is None:
                break
            resp = validate_paginated_response(response)
            ret.extend(resp.results)
        if expect_one_result:
            if len(ret) == 0:
                return {}
            if len(ret) > 1 and any(ret[0] != x for x in ret):
                raise MultipleEntitiesFound(
                    f"Expected a unique result, got {len(ret)} distinct results."
                )
            return ret[0]
        return ret

    def get_typed(
        self,
        path: str,
        type_: type[T],
        params: QueryParams | None = None,
        limit: int | None = None,
    ) -> T:
        """Fetch and deserialize JSON from an endpoint into a specific type.

        This function is a wrapper over the `get()` function, adding the additional
        functionality of validating and converting the response data to the specified type.

        :param path: The path to the API endpoint.
        :param type_: The type to which the response data should be deserialized.
        :param params: The parameters to pass to the API endpoint.
        :param limit: The maximum number of hits to allow for paginated responses.

        :raises pydantic.ValidationError: If the response cannot be deserialized into the given type.

        :returns: An instance of `type_` populated with data from the response.
        """
        adapter = get_type_adapter(type_)
        if type_ is list or get_origin(type_) is list:
            resp = self.get_list(path, params=params, limit=limit)
            return adapter.validate_python(resp)
        else:
            resp = self.get(path, params=params)
            return adapter.validate_json(resp.text)


class PaginatedResponse(BaseModel):
    """Paginated response data from the API."""

    count: int
    next: str | None  # noqa: A003
    previous: str | None
    results: list[Json]

    @field_validator("count", mode="before")
    @classmethod
    def _none_count_is_0(cls, v: Any) -> Any:
        """Ensure `count` is never `None`."""
        # Django count doesn't seem to be guaranteed to be an integer.
        # https://github.com/django/django/blob/bcbc4b9b8a4a47c8e045b060a9860a5c038192de/django/core/paginator.py#L105-L111
        # Theoretically any callable can be passed to the "count" attribute of the paginator.
        # Ensures here that None (and any falsey value) is treated as 0.
        return v or 0

    @classmethod
    def from_response(cls, response: Response) -> PaginatedResponse:
        """Create a PaginatedResponse from a Response."""
        return cls.model_validate_json(response.text)


ListResponse = TypeAdapter(list[Json])
"""JSON list (array) response adapter."""


# TODO: Provide better validation error introspection
def validate_list_response(response: Response) -> list[Json]:
    """Parse and validate that a response contains a JSON array.

    :param response: The response to validate.
    :raises MregValidationError: If the response does not contain a valid JSON array.
    :returns: Parsed response data as a list of Python objects.
    """
    try:
        return ListResponse.validate_json(response.text)
    # NOTE: ValueError catches custom Pydantic errors too
    except ValueError as e:
        raise MregValidationError(f"{response.url} did not return a valid JSON array") from e


def validate_paginated_response(response: Response) -> PaginatedResponse:
    """Validate and parse that a response contains paginated JSON data.

    :param response: The response to validate.
    :raises MregValidationError: If the response does not contain valid paginated JSON.
    :returns: Parsed response data as a PaginatedResponse object.
    """
    try:
        return PaginatedResponse.from_response(response)
    except ValueError as e:
        raise MregValidationError(f"{response.url} did not return valid paginated JSON") from e


# Global client instance getter
_client: MregApiClient | None = None


def get_client() -> MregApiClient:
    """Get the global MREG API client instance.

    Returns:
        The singleton MregApiClient instance

    """
    global _client  # noqa: PLW0603
    if _client is None:
        _client = MregApiClient()
    return _client
