"""Pydantic models for API resources."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Protocol
from typing import Self
from typing import TypeVar

from pydantic import BaseModel
from pydantic import computed_field
from typing_extensions import override

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import GetError
from mreg_api.models.abstracts import FrozenModel
from mreg_api.models.abstracts import manager_only
from mreg_api.models.infra import APIModelMixin
from mreg_api.types import ClientProtocol

logger = logging.getLogger(__name__)

class UserDjangoStatus(BaseModel):
    """Model for Django status in the user response."""

    superuser: bool
    staff: bool
    active: bool


class UserMregStatus(BaseModel):
    """Model for Mreg status in the user response."""

    superuser: bool
    admin: bool
    group_admin: bool
    network_admin: bool
    hostpolicy_admin: bool
    dns_wildcard_admin: bool
    underscore_admin: bool


class UserPermission(BaseModel):
    """Model for permissions in the user response."""

    group: str
    range: str
    regex: str
    labels: list[str]

    # NOTE: _needs_ to be a computed field in order to use it in
    # OutputManager.add_formatted_table, since we dump the model to a dict
    # inside that method.
    @computed_field
    @property
    def labels_str(self) -> str:
        """Return the labels as a string."""
        return ", ".join(self.labels)


FetchT_co = TypeVar("FetchT_co", covariant=True)


class Fetchable(Protocol[FetchT_co]):
    """Interface for classes that implement a `fetch()` method.

    This protocol is designed for endpoints that don't operate on resource IDs,
    unlike standard APIMixin classes which implement `get()` with an ID parameter.

    The `fetch()` method sends a GET request to the class's endpoint without
    requiring an ID, typically for singleton resources, status endpoints, or
    metadata endpoints that return a single object or status response rather
    than a collection.

    Since the response structure and object construction vary by endpoint type,
    there is no general `fetch()` implementation provided by APIMixin. Each
    implementing class must define its own `fetch()` method to handle the
    specific response format and construct the appropriate object.
    """

    def fetch(self, client: "ClientProtocol", *, ignore_errors: bool = True) -> FetchT_co: ...  # noqa: D102


class ServerVersion(BaseModel):
    """Model for server version metadata."""

    version: str

    @classmethod
    def endpoint(cls) -> str:
        """Return the endpoint for the class."""
        return Endpoint.MetaVersion

    @classmethod
    @manager_only
    def fetch(cls, client: "ClientProtocol", *, ignore_errors: bool = True) -> ServerVersion:
        """Fetch the server version from the endpoint.

        :param ignore_errors: Whether to ignore errors.
        :raises GetError: If the response data is invalid and ignore_errors is False.
        :raises httpx.RequestError: If the HTTP request fails and ignore_errors is False.
        :returns: An instance of ServerVersion with the fetched data.
        """
        try:
            response = client.get(cls.endpoint())
            return cls.model_validate(response.json())
        except Exception as e:
            if ignore_errors:
                return cls(version="Unknown")
            raise e


class Library(BaseModel):
    """Model for library metadata."""

    name: str
    version: str


class ServerLibraries(BaseModel):
    """Model for server libraries metadata."""

    libraries: list[Library]

    @classmethod
    def endpoint(cls) -> str:
        """Return the endpoint for the class."""
        return Endpoint.MetaLibraries

    @classmethod
    @manager_only
    def fetch(cls, client: "ClientProtocol", *, ignore_errors: bool = True) -> ServerLibraries:
        """Fetch the server libraries from the endpoint.

        :param ignore_errors: Whether to ignore errors.
        :raises GetError: If the response data is invalid and ignore_errors is False.
        :raises httpx.RequestError: If the HTTP request fails and ignore_errors is False.
        :returns: An instance of ServerLibraries with the fetched data.
        """
        try:
            response = client.get_typed(cls.endpoint(), dict[str, str])
            libraries: list[Library] = []

            for name, version in response.items():
                libraries.append(Library(name=name, version=version))
            return cls(libraries=libraries)
        except Exception as e:
            if ignore_errors:
                return cls(libraries=[])
            raise e


class TokenInfo(BaseModel):
    """Model for token information."""

    is_valid: bool
    created: str
    expire: str
    last_used: str | None = None
    lifespan: str


class UserInfo(BaseModel):
    """Model for the user information."""

    username: str
    last_login: str | None = None
    token: TokenInfo | None = None
    django_status: UserDjangoStatus
    mreg_status: UserMregStatus
    groups: list[str]
    permissions: list[UserPermission]

    @classmethod
    def endpoint(cls) -> str:
        """Return the endpoint for the class."""
        return Endpoint.MetaUser

    @classmethod
    @manager_only
    def fetch(
        cls, client: "ClientProtocol", *, ignore_errors: bool = True, user: str | None = None
    ) -> UserInfo:
        """Fetch the user information from the endpoint.

        :param ignore_errors: Whether to ignore errors.
        :param user: The username to fetch information for. If None, fetch information for the
                              current user.

        :raises GetError: If the response data is invalid and ignore_errors is False.
        :raises httpx.RequestError: If the HTTP request fails and ignore_errors is False.
        :returns: An instance of UserInfo with the fetched data.
        """
        try:
            endpoint = cls.endpoint()
            if user:
                endpoint = f"{endpoint}?username={user}"

            response = client.get(endpoint)
            return cls.model_validate(response.json())
        except Exception as e:
            if ignore_errors:
                return cls(
                    username="Unknown",
                    django_status=UserDjangoStatus(superuser=False, staff=False, active=False),
                    mreg_status=UserMregStatus(
                        superuser=False,
                        admin=False,
                        group_admin=False,
                        network_admin=False,
                        hostpolicy_admin=False,
                        dns_wildcard_admin=False,
                        underscore_admin=False,
                    ),
                    groups=[],
                    permissions=[],
                )
            raise e


class LDAPHealth(APIModelMixin):
    """Model for LDAP health endpoint."""

    status: str

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HealthLDAP

    @classmethod
    @manager_only
    def fetch(cls, client: "ClientProtocol", *, ignore_errors: bool = True) -> Self:
        """Fetch the LDAP status from the endpoint.

        :param ignore_errors: Ignore non-503 errors. 503 means LDAP is down,
            and should not be treated as an error in the traditional sense.
        :raises GetError: If the response code is not 200 or 503.
        :returns: An instance of LDAPStatus.
        """
        try:
            _ = client.get(cls.endpoint())
            return cls(status="OK")
        except GetError as e:
            if ignore_errors:
                logger.error("Failed to fetch LDAP health: %s", e)
                if e.response and e.response.status_code == 503:
                    return cls(status="Down")
                else:
                    return cls(status="Unknown")
            raise e


class HeartbeatHealth(APIModelMixin):
    """Model for heartbeat health endpoint."""

    uptime: int
    start_time: int

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HealthHeartbeat

    def as_str(self) -> str:
        """Return the uptime as a string."""
        # If we got a negative datetime, we weren't able to fetch the heartbeat
        return str(timedelta(seconds=self.uptime)) if self.uptime > 0 else "Unknown"

    @classmethod
    @manager_only
    def fetch(cls, client: "ClientProtocol", *, ignore_errors: bool = True) -> Self:
        """Fetch the heartbeat status from the endpoint.

        :param ignore_errors: Ignore HTTP errors and return dummy object with negative uptime.
        :returns: An instance of HeartbeatHealth with the fetched data.
        """
        try:
            result = client.get(cls.endpoint())
            return cls.model_validate_json(result.text)
        except Exception as e:
            if ignore_errors:
                logger.error("Failed to fetch heartbeat: %s", e)
                return cls(uptime=-1, start_time=0)
            raise e


class HealthInfo(BaseModel):
    """Combined information from all health endpoints."""

    heartbeat: HeartbeatHealth
    ldap: LDAPHealth

    @classmethod
    @manager_only
    def fetch(cls, client: "ClientProtocol") -> HealthInfo:
        """Fetch the health information from the endpoint.

        :returns: An instance of HealthInfo with the fetched data.
        """
        return cls(
            heartbeat=HeartbeatHealth.fetch(client),
            ldap=LDAPHealth.fetch(client),
        )


class TokenAuth(FrozenModel):
    """Model for token authentication response."""

    token: str
