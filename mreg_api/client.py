"""MREG API client for managing HTTP sessions and authentication."""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Self
from urllib.parse import urljoin
from uuid import uuid4

import requests
from requests import Response

from mreg_api.__about__ import __version__
from mreg_api.api.errors import parse_mreg_error
from mreg_api.config import MregCliConfig
from mreg_api.exceptions import APIError
from mreg_api.exceptions import LoginFailedError
from mreg_api.tokenfile import TokenFile
from mreg_api.types import QueryParams

logger = logging.getLogger(__name__)

# Context variables for tracking request info (used in error reporting)
last_request_url: ContextVar[str | None] = ContextVar("last_request_url", default=None)
last_request_method: ContextVar[str | None] = ContextVar("last_request_method", default=None)


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
        >>> response = client.get("/api/v1/hosts/")

        Or with token:
        >>> client = MregApiClient()
        >>> client.set_token("your-token-here")
        >>> response = client.get("/api/v1/hosts/")

        Or from token file:
        >>> client = MregApiClient()
        >>> client.load_token_from_file()
        >>> response = client.get("/api/v1/hosts/")

    """

    _instance: ClassVar[Self | None] = None

    def __new__(cls) -> Self:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the client (only once for singleton)."""
        if self._initialized:
            return

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"mreg-api-{__version__}"})
        self._config = MregCliConfig()
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

    def get_token(self) -> str | None:
        """Get the current authorization token if set.

        Returns:
            Token string or None if not authenticated

        """
        auth = self.session.headers.get("Authorization", "")
        if isinstance(auth, str):
            return auth.partition(" ")[2] or None
        return None

    def load_token_from_file(self) -> bool:
        """Load token from token file if available.

        Returns:
            True if token was loaded, False otherwise

        """
        if not self._config.user or not self._config.url:
            logger.warning("Cannot load token: user or URL not configured")
            return False

        token_entry = TokenFile.get_entry(self._config.user, self._config.url)
        if token_entry:
            self.set_token(token_entry.token)
            logger.info("Loaded token from file for %s @ %s", self._config.user, self._config.url)
            return True

        logger.debug("No token found in file for %s @ %s", self._config.user, self._config.url)
        return False

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

    def login(self, username: str, password: str, save_token: bool = True) -> None:
        """Authenticate with username and password.

        Args:
            username: MREG username
            password: MREG password
            save_token: Whether to save token to token file

        Raises:
            LoginFailedError: If authentication fails

        """
        token_url = urljoin(self._config.url, "/api/token-auth/")
        logger.info("Authenticating %s @ %s", username, token_url)

        try:
            result = requests.post(
                token_url,
                {"username": username, "password": password},
                timeout=self._config.http_timeout,
            )
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            raise LoginFailedError(f"Connection failed: {e}") from e

        if not result.ok:
            err = parse_mreg_error(result)
            msg = err.as_str() if err else result.text
            raise LoginFailedError(msg)

        token = result.json()["token"]
        self.set_token(token)

        if save_token:
            TokenFile.set_entry(username, self._config.url, token)

        logger.info("Authentication successful for %s", username)

    def logout(self) -> None:
        """Logout from MREG (invalidate token on server)."""
        path = urljoin(self._config.url, "/api/token-logout/")
        try:
            self.session.post(path, timeout=self._config.http_timeout)
        except requests.exceptions.ConnectionError as e:
            logger.warning("Failed to log out: %s", e)

    def test_connection(self) -> bool:
        """Test if the current token/connection is valid.

        Returns:
            True if connection is valid, False otherwise

        """
        try:
            ret = self.session.get(
                urljoin(self._config.url, "/api/v1/hosts/"),
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
            message = (
                f'{operation_type} "{url}": {result.status_code}: {result.reason}\n{res_text}'
            )
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

        url = urljoin(self._config.url, path)

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
            timeout=self._config.http_timeout,
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

    def get(
        self,
        path: str,
        params: QueryParams | None = None,
        ok404: bool = False,
    ) -> Response | None:
        """Make a GET request."""
        return self.request("get", path, params, ok404)

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
