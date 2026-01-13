"""Utility functions for interacting with the MREG API.

NOTE: Slated for deletion. mreg_api.client.MregApiCliens implements these functions now.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any
from typing import Callable
from typing import Literal
from typing import ParamSpec
from typing import TypeVar
from typing import get_origin
from typing import overload

from pydantic import BaseModel
from pydantic import TypeAdapter
from pydantic import field_validator
from requests import Response

from mreg_api.cache import get_cache
from mreg_api.client import get_client
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.exceptions import TooManyResults
from mreg_api.types import Json
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter

logger = logging.getLogger(__name__)


T = TypeVar("T")
P = ParamSpec("P")

JsonMappingValidator = TypeAdapter(JsonMapping)


def create_and_set_corrolation_id(suffix: str) -> str:
    """Set currently active correlation id.

    This will take a suffix and append it to a generated UUIDv4 and set it as the correlation id.

    :param suffix: The suffix to use for the correlation id.

    :returns: The generated correlation id.
    """
    return get_client().set_correlation_id(suffix)


def get_correlation_id() -> str:
    """Get the currently active correlation id.

    :returns: The currently active correlation id.
    """
    return get_client().get_correlation_id()


def set_session_token(token: str) -> None:
    """Update session headers with an authorization token.

    :param token: The token to use.
    """
    get_client().set_token(token)


def get_session_token() -> str | None:
    """Get the authorization token from an active session if it exists.

    :returns: The token if it exists, otherwise None.
    """
    return get_client().get_token()


def login(username: str, password: str, save_token: bool = True) -> None:
    """Authenticate with MREG API.

    :param username: MREG username
    :param password: MREG password
    :param save_token: Whether to save token to token file

    :raises LoginFailedError: If authentication fails
    """
    get_client().login(username, password, save_token)


def logout() -> None:
    """Logout from MREG."""
    get_client().logout()


def _do_get(path: str, params: QueryParams | None = None, ok404: bool = False) -> Response | None:
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
    return get_client().request("get", path, params=params, ok404=ok404)


@overload
def get(path: str, params: QueryParams | None, ok404: Literal[True]) -> Response | None: ...


@overload
def get(path: str, params: QueryParams | None, ok404: Literal[False]) -> Response: ...


@overload
def get(path: str, params: QueryParams | None = ..., *, ok404: bool) -> Response | None: ...


@overload
def get(path: str, params: QueryParams | None = ...) -> Response: ...


def get(path: str, params: QueryParams | None = None, ok404: bool = False) -> Response | None:
    """Make a standard get request."""
    return _do_get(path, params, ok404)


def get_list(
    path: str,
    params: QueryParams | None = None,
    ok404: bool = False,
    limit: int | None = 500,
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
    return get_list_generic(path, params, ok404, limit, expect_one_result=False)


def get_list_in(
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
    return get_list(
        path,
        params={f"{search_field}__in": ",".join(str(x) for x in search_values)},
        ok404=ok404,
    )


def get_item_by_key_value(
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
    return get_list_unique(path, params={search_field: search_value}, ok404=ok404)


def get_list_unique(
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
    ret = get_list_generic(path, params, ok404, expect_one_result=True)
    if not ret:
        return None
    try:
        return JsonMappingValidator.validate_python(ret)
    except ValueError as e:
        raise MregValidationError(f"Failed to validate response from {path}: {e}") from e


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


@overload
def get_list_generic(
    path: str,
    params: QueryParams | None = ...,
    ok404: bool = ...,
    limit: int | None = ...,
    expect_one_result: Literal[False] = False,
) -> list[Json]: ...


@overload
def get_list_generic(
    path: str,
    params: QueryParams | None = ...,
    ok404: bool = ...,
    limit: int | None = ...,
    expect_one_result: Literal[True] = True,
) -> Json: ...


def get_list_generic(
    path: str,
    params: QueryParams | None = None,
    ok404: bool = False,
    limit: int | None = 500,
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
    response = get(path, params)

    # Non-paginated results, return them directly
    if "count" not in response.text:
        return validate_list_response(response)

    resp = validate_paginated_response(response)

    if limit and resp.count > abs(limit):
        raise TooManyResults(f"Too many hits ({resp.count}), please refine your search criteria.")

    # Iterate over all pages and collect the results
    ret: list[Json] = resp.results
    while resp.next:
        response = get(resp.next, ok404=ok404)
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
    path: str,
    type_: type[T],
    params: QueryParams | None = None,
    limit: int | None = 500,
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
        resp = get_list(path, params=params, limit=limit)
        return adapter.validate_python(resp)
    else:
        resp = get(path, params=params)
        return adapter.validate_json(resp.text)


def clear_cache(f: Callable[P, T]) -> Callable[P, T]:
    """Clear the API cache after running the function."""

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        result = f(*args, **kwargs)
        try:
            cache = get_cache()
            cache.cache.evict("api")  # does not reset stats
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        return result

    return wrapper


@clear_cache
def post(path: str, params: QueryParams | None = None, **kwargs: Any) -> Response | None:
    """Use requests to make a post request. Assumes that all kwargs are data fields."""
    if params is None:
        params = {}
    return get_client().request("post", path, params=params, **kwargs)


@clear_cache
def patch(path: str, params: QueryParams | None = None, **kwargs: Any) -> Response | None:
    """Use requests to make a patch request. Assumes that all kwargs are data fields."""
    if params is None:
        params = {}
    return get_client().request("patch", path, params=params, **kwargs)


@clear_cache
def delete(path: str, params: QueryParams | None = None, **kwargs: Any) -> Response | None:
    """Use requests to make a delete request."""
    if params is None:
        params = {}
    return get_client().request("delete", path, params=params, **kwargs)
