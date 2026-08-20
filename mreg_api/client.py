"""MREG API client for managing HTTP sessions and authentication."""

from __future__ import annotations

import functools
import logging
import re
from collections import deque
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import Any
from typing import Concatenate
from typing import Literal
from typing import NamedTuple
from typing import ParamSpec
from typing import TypeVar
from typing import get_origin
from typing import overload
from urllib.parse import urljoin
from uuid import uuid4

import httpx
from httpx import Request
from httpx import Response
from pydantic import BaseModel
from pydantic import TypeAdapter
from pydantic import ValidationError
from pydantic import field_validator

from mreg_api.__about__ import __version__
from mreg_api.cache import CacheConfig
from mreg_api.cache import CacheInfo
from mreg_api.cache import MregApiCache
from mreg_api.endpoints import Endpoint
from mreg_api.events import EventLog
from mreg_api.exceptions import APIError
from mreg_api.exceptions import CacheMiss
from mreg_api.exceptions import DeleteError
from mreg_api.exceptions import GetError
from mreg_api.exceptions import InvalidAuthTokenError
from mreg_api.exceptions import LoginFailedError
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.exceptions import PatchError
from mreg_api.exceptions import PostError
from mreg_api.exceptions import TooManyResults
from mreg_api.exceptions import determine_http_error_class
from mreg_api.managers import AtomManager
from mreg_api.managers import BacnetIDManager
from mreg_api.managers import CNAMEManager
from mreg_api.managers import CommunityManager
from mreg_api.managers import DelegationManager
from mreg_api.managers import DhcpHostIPv4Manager
from mreg_api.managers import DhcpHostIPv6ByIPv4Manager
from mreg_api.managers import DhcpHostIPv6Manager
from mreg_api.managers import HInfoManager
from mreg_api.managers import HostGroupManager
from mreg_api.managers import HostManager
from mreg_api.managers import IPAddressManager
from mreg_api.managers import LabelManager
from mreg_api.managers import LocationManager
from mreg_api.managers import MetaManagerNamespace
from mreg_api.managers import MXManager
from mreg_api.managers import NameServerManager
from mreg_api.managers import NAPTRManager
from mreg_api.managers import NetworkManager
from mreg_api.managers import NetworkPolicyAttributeManager
from mreg_api.managers import NetworkPolicyManager
from mreg_api.managers import PermissionManager
from mreg_api.managers import PTROverrideManager
from mreg_api.managers import RoleManager
from mreg_api.managers import SrvManager
from mreg_api.managers import SSHFPManager
from mreg_api.managers import TXTManager
from mreg_api.managers import ZoneManager
from mreg_api.models.fields import HostName
from mreg_api.models.fields import parse_hostname
from mreg_api.models.models import TokenAuth
from mreg_api.types import HTTPMethod
from mreg_api.types import Json
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter

logger = logging.getLogger(__name__)

# Context variables for tracking request info (used in error reporting)
last_request_url: ContextVar[str | None] = ContextVar("last_request_url", default=None)
last_request_method: ContextVar[str | None] = ContextVar("last_request_method", default=None)


T = TypeVar("T")
P = ParamSpec("P")

JsonMappingValidator = TypeAdapter(JsonMapping)


class Header(StrEnum):
    """HTTP headers used by the MREG API."""

    AUTH = "Authorization"
    CORRELATION_ID = "X-Correlation-ID"
    REQUEST_ID = "X-Request-Id"


class RequestRecord(NamedTuple):
    """A complete record of an HTTP request and its response.

    Captures all relevant details of an API call including the request
    parameters, response data, and metadata like status code and URL.
    """

    method: str
    request: Request
    response: Response
    status: int
    data: JsonMapping | None
    json: Json | None

    @property
    def path(self) -> str:
        """Get the request path (URL without base)."""
        # TODO: Make this less hacky! Can we get the path + query directly from httpx?
        parts = [f"{self.response.request.url.scheme}://", self.response.request.url.host]

        if (port_s := f":{self.response.request.url.port}") in self.url:
            parts.append(port_s)

        return self.url.removeprefix("".join(parts))

    @property
    def url(self) -> str:
        """Get the full request URL."""
        return str(self.request.url)


def invalidate_cache(
    func: Callable[Concatenate[MregClient, P], T],
) -> Callable[Concatenate[MregClient, P], T]:
    """Decorator that clears the cache after a successful mutating request."""

    @functools.wraps(func)
    def wrapper(self: MregClient, *args: P.args, **kwargs: P.kwargs) -> T:
        result = func(self, *args, **kwargs)
        # Clear cache after successful mutation (if it exists)
        self.clear_cache()
        return result

    return wrapper


def strip_none(data: JsonMapping) -> JsonMapping:
    """Recursively strip None values from a dictionary."""
    new: dict[str, Json] = {}
    for key, value in data.items():
        if value is not None:
            if isinstance(value, dict):
                v = strip_none(value)
                if v:
                    new[key] = v
            else:
                new[key] = value
    return new


def check_response(response: Response, operation_type: HTTPMethod, url: str) -> None:
    """Check the result of a request and raise on error."""
    if not response.is_success:
        # Produce a more helpful error message for 404s on endpoints that don't exist
        if (
            response.status_code == 404
            and "The requested resource was not found on this server." in response.text
        ):
            endpoint = url.split("/api/v1/")[-1] if "/api/v1/" in url else url
            msg = (
                f"Endpoint not found: '{endpoint}'\n"
                f"This may be because your library version ({__version__}) is:\n"
                f"  - Too old: The endpoint has been removed from the server\n"
                f"  - Too new: You're using a beta feature not yet available on the server"
            )
        else:
            msg = response.text

        # Fall back on reason phrase if derived message is empty
        if not msg:
            msg = response.reason_phrase

        cls = determine_http_error_class(operation_type)
        raise cls(msg, response)


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

    Args:
        response: The response to validate.

    Raises:
        MregValidationError: If the response does not contain a valid JSON array.

    Returns:
        Parsed response data as a list of Python objects.
    """
    try:
        return ListResponse.validate_json(response.text)
    # NOTE: ValueError catches custom Pydantic errors too
    except ValidationError as e:
        raise MregValidationError.from_pydantic(e, "JSON list") from e


def validate_paginated_response(response: Response) -> PaginatedResponse:
    """Validate and parse that a response contains paginated JSON data.

    Args:
        response: The response to validate.

    Raises:
        MregValidationError: If the response does not contain valid paginated JSON.

    Returns:
        Parsed response data as a PaginatedResponse object.
    """
    try:
        return PaginatedResponse.from_response(response)
    except ValidationError as e:
        raise MregValidationError.from_pydantic(e, "paginated JSON") from e


class MregClient:
    """Client for interacting with MREG API.

    This client manages HTTP sessions, authentication, and provides
    methods for making API requests.

    Authentication modes:
    1. Token: Provide token directly via set_token()
    2. Username/password: Call login() with credentials

    Example:
        >>> client = MregClient(url="https://mreg.example.com", domain="example.com")
        >>> client.login("username", "password")
        >>> hosts = client.host.list()

        Or with token:
        >>> client = MregClient(url="https://mreg.example.com", domain="example.com")
        >>> client.set_token("your-token-here")
    """

    @functools.cached_property
    def atom(self) -> AtomManager:
        """Manager for host policy atom resources."""
        return AtomManager(self)

    @functools.cached_property
    def bacnetid(self) -> BacnetIDManager:
        """Manager for BACnet ID resources."""
        return BacnetIDManager(self)

    @functools.cached_property
    def cname(self) -> CNAMEManager:
        """Manager for CNAME resources."""
        return CNAMEManager(self)

    @functools.cached_property
    def community(self) -> CommunityManager:
        """Manager for network community resources."""
        return CommunityManager(self, self.network)

    @functools.cached_property
    def delegation(self) -> DelegationManager:
        """Manager for zone delegation resources."""
        return DelegationManager(self)

    @functools.cached_property
    def dhcphostipv4(self) -> DhcpHostIPv4Manager:
        """Manager for IPv4 DHCP host records."""
        return DhcpHostIPv4Manager(self)

    @functools.cached_property
    def dhcphostipv6(self) -> DhcpHostIPv6Manager:
        """Manager for IPv6 DHCP host records."""
        return DhcpHostIPv6Manager(self)

    @functools.cached_property
    def dhcphostipv6byipv4(self) -> DhcpHostIPv6ByIPv4Manager:
        """Manager for IPv6-via-IPv4 DHCP host records."""
        return DhcpHostIPv6ByIPv4Manager(self)

    @functools.cached_property
    def hinfo(self) -> HInfoManager:
        """Manager for HINFO resources."""
        return HInfoManager(self)

    @functools.cached_property
    def host(self) -> HostManager:
        """Manager for host resources."""
        return HostManager(self)

    @functools.cached_property
    def hostgroup(self) -> HostGroupManager:
        """Manager for host group resources."""
        return HostGroupManager(self)

    @functools.cached_property
    def ipaddress(self) -> IPAddressManager:
        """Manager for IP address resources."""
        return IPAddressManager(self)

    @functools.cached_property
    def label(self) -> LabelManager:
        """Manager for label resources."""
        return LabelManager(self)

    @functools.cached_property
    def location(self) -> LocationManager:
        """Manager for location resources."""
        return LocationManager(self)

    @functools.cached_property
    def meta(self) -> MetaManagerNamespace:
        """Manager for meta endpoints."""
        return MetaManagerNamespace(self)

    @functools.cached_property
    def mx(self) -> MXManager:
        """Manager for MX record resources."""
        return MXManager(self)

    @functools.cached_property
    def nameserver(self) -> NameServerManager:
        """Manager for nameserver resources (read-only)."""
        return NameServerManager(self)

    @functools.cached_property
    def naptr(self) -> NAPTRManager:
        """Manager for NAPTR record resources."""
        return NAPTRManager(self)

    @functools.cached_property
    def network(self) -> NetworkManager:
        """Manager for network resources."""
        return NetworkManager(self)

    @functools.cached_property
    def networkpolicy(self) -> NetworkPolicyManager:
        """Manager for network policy resources."""
        return NetworkPolicyManager(self)

    @functools.cached_property
    def networkpolicyattribute(self) -> NetworkPolicyAttributeManager:
        """Manager for network policy attribute resources."""
        return NetworkPolicyAttributeManager(self)

    @functools.cached_property
    def permission(self) -> PermissionManager:
        """Manager for permission resources."""
        return PermissionManager(self)

    @functools.cached_property
    def ptroverride(self) -> PTROverrideManager:
        """Manager for PTR override resources."""
        return PTROverrideManager(self)

    @functools.cached_property
    def role(self) -> RoleManager:
        """Manager for host policy role resources."""
        return RoleManager(self)

    @functools.cached_property
    def srv(self) -> SrvManager:
        """Manager for SRV record resources."""
        return SrvManager(self)

    @functools.cached_property
    def sshfp(self) -> SSHFPManager:
        """Manager for SSHFP record resources."""
        return SSHFPManager(self)

    @functools.cached_property
    def txt(self) -> TXTManager:
        """Manager for TXT record resources."""
        return TXTManager(self)

    @functools.cached_property
    def zone(self) -> ZoneManager:
        """Manager for forward/reverse zone resources."""
        return ZoneManager(self)

    def __init__(
        self,
        url: str,
        domain: str | None = None,
        user: str | None = None,
        timeout: int | float | None = 60,
        cache: CacheConfig | bool = False,
        follow_redirects: bool = False,
        page_size: int | None = None,
        history_size: int | None = 100,
        event_log_size: int | None = 100,
        user_agent: str | None = None,
    ) -> None:
        """Initialize the client."""
        if not user_agent:
            user_agent = f"mreg-api-{__version__}"
        self.session: httpx.Client = httpx.Client(
            headers={"User-Agent": user_agent},
            follow_redirects=follow_redirects,
            timeout=timeout,
        )

        self.url: str = url
        self.domain: str | None = domain
        self._page_size: int | None = page_size
        self.user: str | None = user

        if isinstance(cache, bool):
            _cache_config: CacheConfig = CacheConfig(enable=cache)
        else:
            _cache_config = cache
        self.cache: MregApiCache[Response] = self._create_cache(_cache_config)

        # State setup/reset
        self._token: str | None = None
        self.history: deque[RequestRecord] = deque(maxlen=history_size)
        self.events: EventLog = EventLog(max_size=event_log_size)

    @property
    def timeout(self) -> float | None:
        """Get the current timeout setting."""
        return self.session.timeout.read

    @timeout.setter
    def timeout(self, value: int | float | None) -> None:
        """Set the timeout for requests."""
        self.session.timeout = value

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self._reset_contextvars()
        self.session.close()

        # Attempt to close open connections if cache is enabled
        # In practice, we rarely need this, but in tests and multihreaded applications,
        # we may see "ResourceWarning: unclosed database in [...]" when running
        # via coverage and vscode test runner.
        if self.cache.has_backend:  # always close if cache _exists_ (may be disabled)
            self.cache.close()

    def _reset_contextvars(self) -> None:
        """Reset context variables used for request tracking."""
        _ = last_request_url.set(None)
        _ = last_request_method.set(None)

    def fqdn(self, name: str) -> HostName:
        """Normalise and expand a hostname using this client's configured domain.

        Not _technically_ a FQDN, since we don't include a trailing dot,
        but that's not a concept MREG operates on anyway.
        """
        return parse_hostname(name, self.domain)

    @contextmanager
    def domain_override(self, domain: str | None) -> Generator[None, None, None]:
        """Temporarily override the hostname domain used by :meth:`fqdn`.

        Args:
            domain: The domain to use within the context.

        Example:
            >>> with client.domain_override("example.com"):
            ...     client.fqdn("web")  # → "web.example.com"
        """
        old = self.domain
        self.domain = domain
        try:
            yield
        finally:
            self.domain = old

    def _get_cache_tag(self) -> str:
        """Get the cache tag for this client."""
        return f"mreg_api_client_cache_{self.url.replace('://', '_').replace('/', '_')}"

    def _create_cache(self, config: CacheConfig) -> MregApiCache[Response]:
        """Create the cache wrapper."""
        return MregApiCache[Response].new(config)

    def enable_cache(self, config: CacheConfig | None = None) -> None:
        """Enable caching of GET responses for this client.

        If a new config is provided, the cache is recreated with that config.
        Otherwise, just enables caching on the existing cache.
        """
        if config is not None:
            # Recreate cache with new config
            self.cache = self._create_cache(config)
        else:
            self.cache.enable()

    def disable_cache(self, *, clear: bool = True) -> None:
        """Disable caching of GET responses for this client.

        Args:
            clear: If True, clear the existing cache data.
                If False, leave the cache data intact.
        """
        if clear:
            self.clear_cache()
        self.cache.disable()

    def clear_cache(self) -> int:
        """Clear the client's GET response cache."""
        return self.cache.clear()

    @contextmanager
    def caching(self, enable: bool = True):
        """Context manager to temporarily enable or disable caching.

        Args:
            enable: If True, enable caching within the context.
                    If False, disable caching within the context.

        Example:
            >>> with client.caching(enable=True):
            ...     # caching is enabled here
            ...     pass
            >>> with client.caching(enable=False):
            ...     # caching is disabled here
            ...     pass
        """
        was_enabled = self.cache.is_enabled

        if enable != was_enabled:
            if enable:
                self.enable_cache()
            else:
                self.disable_cache(clear=False)

        try:
            yield
        finally:
            if enable != was_enabled:
                if was_enabled:
                    self.enable_cache()
                else:
                    self.disable_cache(clear=False)

    def get_cache_info(self) -> CacheInfo | None:
        """Get statistics about the client's cache.

        Returns:
            CacheInfo object or None if caching is disabled

        """
        return self.cache.get_info()

    def set_token(self, token: str) -> None:
        """Set the authorization token for API requests.

        Args:
            token: Bearer token for authentication

        """
        self.session.headers.update({Header.AUTH: f"Token {token}"})
        self._token = token

    def get_token(self) -> str | None:
        """Get the current authorization token if set.

        Returns:
            Token string or None if not authenticated

        """
        return self._token

    def unset_token(self) -> None:
        """Unset the current authorization token."""
        self.session.headers.pop(Header.AUTH, None)
        self._token = None

    def set_correlation_id(self, suffix: str) -> str:
        """Set correlation ID for request tracking.

        Args:
            suffix: Human-readable suffix for the correlation ID

        Returns:
            The generated correlation ID

        """
        suffix = re.sub(r"\s+", "_", suffix)
        correlation_id = f"{uuid4()}-{suffix}"
        self.session.headers.update({Header.CORRELATION_ID: correlation_id})
        return correlation_id

    def get_correlation_id(self) -> str:
        """Get the current correlation ID.

        Returns:
            Current correlation ID or empty string

        """
        return str(self.session.headers.get(Header.CORRELATION_ID, ""))

    def _add_to_history(
        self,
        response: Response,
        data: JsonMapping | None = None,
        json: Json | None = None,
    ) -> None:
        """Add a request/response pair to the history log."""
        self.history.append(
            RequestRecord(
                method=response.request.method,
                request=response.request,
                response=response,
                status=response.status_code,
                data=data,
                json=json,
            )
        )

    def get_client_history(self) -> deque[RequestRecord]:
        """Get the request/response history for this client.

        Returns:
            List of RequestRecord objects representing the history

        """
        return self.history

    def clear_client_history(self) -> None:
        """Clear the request/response history for this client."""
        self.history.clear()

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
            response = httpx.post(
                token_url,
                data={"username": username, "password": password},
                timeout=self.timeout,
            )
        except httpx.RequestError as e:
            raise LoginFailedError(f"Connection failed: {e}") from e

        if not response.is_success:
            # NOTE: Exception uses parsed API error message if possible
            raise LoginFailedError(response.text, response)

        if not (json_str := response.text):
            raise LoginFailedError("No token received from server")
        try:
            token = TokenAuth.model_validate_json(json_str).token
        except ValidationError as e:
            raise LoginFailedError(f"Failed to parse authentication token: {e}") from e

        self.set_token(token)

        logger.info("Authentication successful for %s", username)

        return token

    def logout(self) -> None:
        """Logout from MREG (invalidate token on server)."""
        path = urljoin(self.url, Endpoint.TokenLogout)
        try:
            self.session.post(path, timeout=self.timeout)
        except httpx.RequestError as e:
            logger.warning("Failed to log out: %s", e)

    def test_auth(self) -> None:
        """Test if the current authorization token is valid.

        Does not handle connection errors.

        Raises:
            APIError: If the authorization test fails

        Returns:
            True if authorization is valid, False otherwise
        """
        ret: Response | None = None
        try:
            ret = self.session.get(
                urljoin(self.url, Endpoint.Hosts),
                params={"page_size": 1},
                timeout=self.timeout,
            )
            ret.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise InvalidAuthTokenError(f"Authorization test failed: {e}", ret) from e

    def request(
        self,
        method: HTTPMethod,
        path: str,
        *,
        params: QueryParams | None = None,
        ok404: bool = False,
        json: Json | None = None,
    ) -> Response | None:
        """Make an HTTP request to the MREG API.

        Args:
            method: HTTP method
            path: API path (relative to base URL)
            params: Query parameters
            ok404: Whether to return None on 404 instead of raising
            json: Request body data as a JSON-serializable value

        Returns:
            Response object or None if ok404=True and status is 404

        Raises:
            APIError: If request fails

        """
        # Ensure that we never pass in params when we are paginating,
        # as that would overwrite the query parameters in the URL.
        if "?" in path:
            params = None
        else:
            params = params or {}
            # Add default GET params if we are passing in params
            # Only passes default params to GET requests while NOT paginating
            if method == "GET":
                if self._page_size:
                    _ = params.setdefault("page_size", self._page_size)

        url = urljoin(self.url, path)

        if method != "GET" and params:
            logger.debug("Params: %s", params)

        if json is not None:
            logger.debug("JSON: %s", json)

        # Strip None values from json body (except for PATCH)
        if json is not None and method != "PATCH" and isinstance(json, Mapping):
            json = strip_none(json) or None

        # Construct the request object
        request = self.session.build_request(method=method, url=url, params=params, json=json)
        logger.info("Request: %s %s [%s]", method, request.url, self.get_correlation_id())

        # Update context variables for error reporting
        last_request_url.set(str(request.url))
        last_request_method.set(method)

        result = self.session.send(request)
        # Log response in response log
        self._add_to_history(result, json=json)

        # # This is a workaround for old server versions that can't handle JSON data in requests
        # if result.is_redirect and not result.history and params == {} and data:
        #     self.session.build_request(method=method, url=url, params=params, data=data or None)

        # Log response
        request_id = result.headers.get(Header.REQUEST_ID, "?")
        correlation_id = result.headers.get(Header.CORRELATION_ID, "?")
        id_str = f"[R:{request_id} C:{correlation_id}]"
        log_message = f"Response: {method} {request.url} {result.status_code} {id_str}"

        if result.status_code >= 300:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        # Handle 404 if ok404 is True
        if result.status_code == 404 and ok404:
            return None

        check_response(result, method, url)
        return result

    def _make_cache_key(self, path: str, params: QueryParams | None, ok404: bool) -> str:
        """Create a unique cache key from request path and parameters.

        Args:
            path: API endpoint path
            params: Query parameters dict
            ok404: ok404 parameter

        Returns:
            A deterministic string key for caching
        """
        parts = [path, f"ok404={ok404}"]
        if params:
            # Sort params for consistent key generation
            sorted_params = sorted((k, str(v)) for k, v in params.items())
            param_str = "&".join(f"{k}={v}" for k, v in sorted_params)
            parts.append(param_str)
        return "?".join(parts)

    @overload
    def get(
        self, path: str, *, params: QueryParams | None = ..., ok404: Literal[True]
    ) -> Response | None: ...

    @overload
    def get(self, path: str, *, params: QueryParams | None = ..., ok404: Literal[False]) -> Response: ...

    @overload
    def get(self, path: str, *, params: QueryParams | None = ..., ok404: bool) -> Response | None: ...

    @overload
    def get(self, path: str, *, params: QueryParams | None = None) -> Response: ...

    def get(self, path: str, *, params: QueryParams | None = None, ok404: bool = False) -> Response | None:
        """Make a standard get request."""
        if self.cache.is_enabled:
            cache_key = self._make_cache_key(path, params, ok404)

            try:
                return self.cache.get(cache_key)
            except CacheMiss:
                logger.debug("Cache miss for key: %s", cache_key)

            ret = self._do_get(path, params=params, ok404=ok404)
            self.cache.set(cache_key, ret)
            return ret
        return self._do_get(path, params=params, ok404=ok404)

    def _do_get(
        self, path: str, *, params: QueryParams | None = None, ok404: bool = False
    ) -> Response | None:
        try:
            return self.request("GET", path, params=params, ok404=ok404)
        except GetError as e:
            raise e
        except APIError as e:
            raise GetError(e.details, e.response) from e

    @overload
    def post(
        self,
        path: str,
        *,
        json: Json | None = ...,
        params: QueryParams | None = ...,
        ok404: Literal[True],
    ) -> Response | None: ...

    @overload
    def post(
        self,
        path: str,
        *,
        json: Json | None = ...,
        params: QueryParams | None = ...,
        ok404: Literal[False],
    ) -> Response: ...

    @overload
    def post(
        self, path: str, *, json: Json | None = ..., params: QueryParams | None = ..., ok404: bool
    ) -> Response | None: ...

    @overload
    def post(self, path: str, *, json: Json | None = None, params: QueryParams | None = None) -> Response: ...

    @invalidate_cache
    def post(
        self,
        path: str,
        *,
        json: Json | None = None,
        params: QueryParams | None = None,
        ok404: bool = False,
    ) -> Response | None:
        """Make a POST request."""
        try:
            return self.request("POST", path, params=params, ok404=ok404, json=json)
        except PostError as e:
            raise e
        except APIError as e:
            raise PostError(e.details, e.response) from e

    @overload
    def patch(
        self,
        path: str,
        *,
        json: Json | None = ...,
        params: QueryParams | None = ...,
        ok404: Literal[True],
    ) -> Response | None: ...

    @overload
    def patch(
        self,
        path: str,
        *,
        json: Json | None = ...,
        params: QueryParams | None = ...,
        ok404: Literal[False],
    ) -> Response: ...

    @overload
    def patch(
        self, path: str, *, json: Json | None = ..., params: QueryParams | None = ..., ok404: bool
    ) -> Response | None: ...

    @overload
    def patch(
        self, path: str, *, json: Json | None = None, params: QueryParams | None = None
    ) -> Response: ...

    @invalidate_cache
    def patch(
        self,
        path: str,
        *,
        json: Json | None = None,
        params: QueryParams | None = None,
        ok404: bool = False,
    ) -> Response | None:
        """Make a PATCH request."""
        try:
            return self.request("PATCH", path, params=params, ok404=ok404, json=json)
        except PatchError as e:
            raise e
        except APIError as e:
            raise PatchError(e.details, e.response) from e

    @overload
    def delete(
        self,
        path: str,
        *,
        json: Json | None = ...,
        params: QueryParams | None = ...,
        ok404: Literal[True],
    ) -> Response | None: ...

    @overload
    def delete(
        self,
        path: str,
        *,
        json: Json | None = ...,
        params: QueryParams | None = ...,
        ok404: Literal[False],
    ) -> Response: ...

    @overload
    def delete(
        self, path: str, *, json: Json | None = ..., params: QueryParams | None = ..., ok404: bool
    ) -> Response | None: ...

    @overload
    def delete(
        self, path: str, *, json: Json | None = None, params: QueryParams | None = None
    ) -> Response: ...

    @invalidate_cache
    def delete(
        self,
        path: str,
        *,
        json: Json | None = None,
        params: QueryParams | None = None,
        ok404: bool = False,
    ) -> Response | None:
        """Make a DELETE request."""
        try:
            return self.request("DELETE", path, params=params, ok404=ok404, json=json)
        except DeleteError as e:
            raise e
        except APIError as e:
            raise DeleteError(e.details, e.response) from e

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

        Args:
            path: The path to the API endpoint.
            params: The parameters to pass to the API endpoint.
            ok404: Whether to allow 404 responses.
            limit: The maximum number of hits to allow. If the number of hits is greater
                than this, the function will raise an exception. Set to None to disable
                this check.

        Returns:
            A list of dictionaries.
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

        Args:
            path: The path to the API endpoint.
            search_field: The field to search for.
            search_values: The values to search for.
            ok404: Whether to allow 404 responses.

        Returns:
            A list of dictionaries.
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

        Args:
            path: The path to the API endpoint.
            search_field: The field to search for.
            search_value: The value to search for.
            ok404: Whether to allow 404 responses.

        Returns:
            A single dictionary, or None if no result was found and ok404 is True.
        """
        return self.get_list_unique(path, params={search_field: search_value}, ok404=ok404)

    def get_list_unique(
        self,
        path: str,
        params: QueryParams | None = None,
        ok404: bool = False,
    ) -> None | JsonMapping:
        """Do a get request that returns a single result from a search.

        Args:
            path: The path to the API endpoint.
            params: The parameters to pass to the API endpoint.
            ok404: Whether to allow 404 responses.

        Returns:
            A single dictionary, or None if no result was found and ok404 is True.
        """
        ret = self.get_list_generic(path, params, ok404, expect_one_result=True)
        if not ret:
            return None
        try:
            return JsonMappingValidator.validate_python(ret)
        except ValidationError as e:
            raise MregValidationError.from_pydantic(e, "JSON mapping") from e

    def get_first(self, path: str, params: QueryParams | None = None) -> JsonMapping | None:
        """Get the first item from a list endpoint."""
        if params is None:
            params = {}

        # Set a default page size of 1 to minimize data transfer.
        # Callers may override this (but they really shouldn't!)
        params.setdefault("page_size", 1)

        response = self.get(path, params=params)

        if "count" not in response.text:
            content = validate_list_response(response)
        else:
            resp = validate_paginated_response(response)
            content = resp.results
        if not content:
            return None
        # Ensure that the first item is a mapping (dictionary)
        return JsonMappingValidator.validate_python(content[0])

    def get_count(self, path: str, *, strict: bool = False) -> int:
        """Get the count of items from a list endpoint.

        Warning:
            Returns the length of the results if the endpoint does not implement pagination.

        Returns:
            The count of items.
        """
        response = self.get(path, params={"page_size": 1})
        try:
            resp = validate_paginated_response(response)
            return resp.count
        except MregValidationError:
            if strict:
                raise GetError(
                    f"Endpoint {path} does not support counting. "  # pyright: ignore[reportImplicitStringConcatenation]
                    "Pass `strict=False` to fall back on client-side counting."
                ) from None

            content = validate_list_response(response)
            logger.warning(
                "Endpoint %s did not return a paginated response, returning length of results as count",
                path,
            )
            return len(content)

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

        Args:
            path: The path to the API endpoint.
            params: The parameters to pass to the API endpoint.
            ok404: Whether to allow 404 responses.
            limit: The maximum number of hits to allow. If the number of hits is greater
                than this, the function will raise an exception. Set to None to disable
                this check.
            expect_one_result: If True, expect exactly one result and return it as a list.

        Raises:
            MultipleEntitiesFound: If expect_one_result is True and the number of results
                is greater than one with distinct results.
            TooManyResults: If the number of hits is greater than limit.

        Returns:
            A list of dictionaries or a dictionary if expect_one_result is True.
        """
        response = self.get(path, params=params)

        # Non-paginated results, return them directly
        if "count" not in response.text:
            return validate_list_response(response)

        resp = validate_paginated_response(response)

        if limit and resp.count > abs(limit):
            # TODO: append to error message that limit can be increased/disabled
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
                raise MultipleEntitiesFound(f"Expected a unique result, got {len(ret)} distinct results.")
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

        Args:
            path: The path to the API endpoint.
            type_: The type to which the response data should be deserialized.
            params: The parameters to pass to the API endpoint.
            limit: The maximum number of hits to allow for paginated responses.

        Raises:
            pydantic.ValidationError: If the response cannot be deserialized into the given type.

        Returns:
            An instance of `type_` populated with data from the response.
        """
        adapter = get_type_adapter(type_)
        if type_ is list or get_origin(type_) is list:
            resp = self.get_list(path, params=params, limit=limit)
            return adapter.validate_python(resp)
        else:
            resp = self.get(path, params=params)
            return adapter.validate_json(resp.text)
