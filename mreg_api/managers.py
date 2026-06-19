"""Resource managers: the public, client-bound API surface.

Each manager owns the operations for one API resource. A manager holds a
back-reference to the :class:`~mreg_api.client.MregClient` that constructed it and
uses the client's low-level HTTP/pagination primitives to fetch data and build
(dumb) models from it.

The generic :class:`ResourceManager` provides the shared CRUD verbs once; concrete
per-resource managers bind the model type and add resource-specific methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import Literal
from typing import Protocol
from typing import TypeVar
from typing import overload

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import GetError
from mreg_api.exceptions import InternalError
from mreg_api.exceptions import PostError
from mreg_api.models import Host
from mreg_api.models.history import HistoryItem
from mreg_api.models.history import HistoryResource
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter

if TYPE_CHECKING:
    from mreg_api.client import MregClient
    from mreg_api.models.fields import HostName
    from mreg_api.types import IP_AddressT
    from mreg_api.types import IP_NetworkT


class APIResource(Protocol):
    """Structural type for a model that maps to an API resource.

    Resources only need to expose their endpoint as pure metadata; all I/O lives on
    the managers.
    """

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the resource."""
        ...


T = TypeVar("T", bound=APIResource)


class ResourceManager(Generic[T]):
    """Manager for performing CRUD operations on an API resource type."""

    # NOTE TO AGENT: ClassVar[type[Any]] is not ideal.
    # It should be narrowed down to some common model base class.
    # While, yes, the `model` property does convince the type checker,
    # it's still a hack, as there is nothing preventing `_model` from being set to an
    # incompatible type. If we can do this without using `Any`, it's for the best.
    # Furthermore, `APIResource` may have to become a concrete base class
    # that all models inherit from if need be.
    _model: ClassVar[type[Any]]
    """The model type this manager operates on. Bound by each subclass."""

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client that owns it."""
        self._client: MregClient = client

    # --- internal helpers ------------------------------------------------

    @property
    def model(self) -> type[T]:
        """The model type this manager operates on."""
        return self._model

    def _endpoint(self) -> Endpoint:
        return self.model.endpoint()

    def _validate(self, data: Any) -> T:
        return get_type_adapter(self.model).validate_python(data)

    def _id_field(self, obj: T) -> str:
        return obj.endpoint().external_id_field()

    def _endpoint_with_id(self, obj: T) -> str:
        return obj.endpoint().with_id(getattr(obj, self._id_field(obj)))

    # NOTE TO AGENT: these internal helpers will not work in their current iteration
    # since the client performs a check on each response and raises if the response
    # is not successful. Therefore, any code that assumes the CRUD methods will return
    # a response object that can be checked for success or failure will not work.
    # Client may need to be refactored to always return a response object, and
    # let the manager raise the appropriate error based on the request and response.
    # I.e. `GetError`, `PostError`, etc. like we currently do.
    #
    # Furthermore: `_patch` has no exception handling at all. Must be fixed.
    #              All CRUD methods should be refactored to have consistent
    #              and robust error handling.
    #
    # Idea for handling Pydantic errors (implement this _last_ after everything else works):
    # - Decorator or helper method that catches ValidationError and is able
    # to use similar logic to the current `MregValidationError.from_pydantic`
    # to determine the last URL we fetched from (which likely caused the error).
    # This error handling should then be available to all subclasses, so they can
    # simply try to fetch the relevant resources and delegate error handling to
    # the helper method/decorator.
    # Decorator may be worth exploring.

    def _fetch_by_field(self, field: str, value: str | int) -> T | None:
        """Fetch a single object by a field, mirroring the old APIMixin.get_by_field."""
        endpoint = self._endpoint()

        # Field is the "external" ID field. i.e. /hosts/{name} instead of /hosts/{id}
        if endpoint.requires_search_for_id() and field == endpoint.external_id_field():
            resp = self._client.get(endpoint.with_id(value), ok404=True)
            if not resp:
                return None
            return self._validate(resp.json())

        # Lookup by non-ID field, i.e. `/hosts?ipaddress=foo` instead of `/hosts/{id}`.
        data = self._client.get_item_by_key_value(endpoint, field, value, ok404=True)
        if not data:
            return None
        return self._validate(data)

    def _refetch(self, obj: T) -> T:
        """Fetch a fresh copy of ``obj`` from the server."""
        id_field = self._id_field(obj)
        lookup = getattr(obj, "id", None) if hasattr(obj, "id") else getattr(obj, id_field, None)
        if not lookup:
            raise InternalError(f"Could not determine identifier for {self.model.__name__}.")
        fresh = self._fetch_by_field("id" if hasattr(obj, "id") else id_field, lookup)
        if not fresh:
            raise GetError(f"Could not refetch {self.model.__name__} ({lookup!r}).")
        return fresh

    def _create(self, data: dict[str, Any], *, fetch_after_create: bool = True) -> T | None:
        """POST ``data`` to the resource endpoint, optionally fetching the result."""
        response = self._client.post(self._endpoint(), json=data)
        if not (response and response.is_success):
            raise PostError(f"Failed to create {self.model.__name__} with {data}.")
        if fetch_after_create and "Location" in response.headers:
            return self._client.get_typed(response.headers["Location"], self.model)
        return None

    def _patch(self, obj: T, data: dict[str, Any], *, params: QueryParams | None = None) -> T:
        """PATCH ``obj`` with ``data`` and return the refetched object."""
        _ = self._client.patch(self._endpoint_with_id(obj), json=data, params=params)
        return self._refetch(obj)

    # --- public CRUD -----------------------------------------------------

    @overload
    def get(self, ident: str | int, *, should_exist: Literal[True]) -> T: ...
    @overload
    def get(self, ident: str | int, *, should_exist: Literal[False]) -> None: ...
    @overload
    def get(self, ident: str | int, *, should_exist: None = ...) -> T | None: ...
    def get(self, ident: str | int, *, should_exist: bool | None = None) -> T | None:
        """Get a resource by its natural identifier.

        Args:
            ident: The external identifier (id / name / network, per resource).
            should_exist: ``None`` returns ``T | None``; ``True`` raises

        Raises:
            :class:`EntityNotFound` if `should_exist` is True and the resource is not found.
            :class:`EntityAlreadyExists` if `should_exist` is False and the resource is found.

        Returns:
            The resource, or ``None``.
        """
        obj = self._fetch_by_field(self._endpoint().external_id_field(), ident)
        if should_exist is True and obj is None:
            raise EntityNotFound(f"{self.model.__name__} {ident!r} not found.")
        if should_exist is False and obj is not None:
            raise EntityAlreadyExists(f"{self.model.__name__} {ident!r} already exists.")
        return obj

    def list(
        self,
        *,
        limit: int | None = 500,
        **query: str | int | float | bool | None,
    ) -> list[T]:
        """List resources, optionally filtered by query parameters."""
        params: QueryParams = dict(query)
        return self._client.get_typed(self._endpoint(), list[self.model], params=params, limit=limit)

    def delete(self, obj: T) -> bool:
        """Delete a resource. Returns True on success."""
        response = self._client.delete(self._endpoint_with_id(obj))
        return bool(response and response.is_success)


class NamedResourceManager(ResourceManager[T]):
    """Extended manager for resources that support name-based lookups."""

    name_field: ClassVar[str] = "name"
    name_lowercase: ClassVar[bool] = False

    def _case_name(self, name: str) -> str:
        return name.lower() if self.name_lowercase else name

    @overload
    def get_by_name(self, name: str, *, should_exist: Literal[True]) -> T: ...
    @overload
    def get_by_name(self, name: str, *, should_exist: Literal[False]) -> None: ...
    @overload
    def get_by_name(self, name: str, *, should_exist: None = ...) -> T | None: ...
    def get_by_name(self, name: str, *, should_exist: bool | None = None) -> T | None:
        """Get a resource by name (searches the name field)."""
        obj = self._fetch_by_field(self.name_field, self._case_name(name))
        if should_exist is True and obj is None:
            raise EntityNotFound(f"{self.model.__name__} {name!r} not found.")
        if should_exist is False and obj is not None:
            raise EntityAlreadyExists(f"{self.model.__name__} {name!r} already exists.")
        return obj


class HistoryCapableManager(Generic[T]):
    """Mixin for managers whose resource records audit history.

    A pure mixin (no ``__init__``) so it composes with a concrete
    :class:`ResourceManager` without unsafe multiple inheritance. Only managers
    that mix this in expose :meth:`history`, so e.g. ``client.networks.history`` is
    a static type error.
    """

    history_resource: ClassVar[HistoryResource]

    def history(self, name: str) -> list[HistoryItem]:
        """Get the audit history for a named resource."""
        return HistoryItem.get(name, self.history_resource)


class HostManager(NamedResourceManager[Host], HistoryCapableManager[Host]):
    """Operations on :class:`~mreg_api.models.Host` resources."""

    _model: ClassVar[type[Host]] = Host
    history_resource: ClassVar[HistoryResource] = HistoryResource.Host

    # NOTE TO AGENT: Hosts should be resolvable through both ID or Name when fetching them.
    # Having a bespoke `get_by_name` method inherited from `NamedResourceManager` is not ideal.
    # We would ideally like that any model that currently implements
    # `get_by_any_means(or_raise)` would implement similar logic that
    # is called under the hood in the public API of `Host` (and `Network`)
    # that performs the same more complicated lookup logic that they currently do.
    # It is not acceptable for users to call `get` and `get_by_name` and
    # protect them in try..except, etc. etc. It needs to be a clean and
    # opaque method that simply does the right thing.
    #
    # In detail, this means the following:
    #
    # `Host` can fetch by:
    # - hostname (name field)
    # - ID (id field)
    # - IP address (any of its IPs)
    # - MAC address (any of its MACs)
    #
    # `Network` can fetch by:
    # - IP address
    # - network address (CIDR)
    # - name
    # - ID
    #
    # In order to write "correct" Python, we need to either always pass in
    # the IP, MAC and Network addresses as strings, or we need to implement
    # a separate method for this `get_by_any_means`-type behavior. I lean towards
    # the former, so we can have a clean `get` interface for all models that
    # perform some magic under the hood.
    def _resolve(self, ref: int | Host) -> Host:
        """Resolve a host reference (instance or numeric id) to a Host."""
        if isinstance(ref, Host):
            return ref
        host = self._fetch_by_field("id", ref)
        if host is None:
            raise EntityNotFound(f"Host with id {ref!r} not found.")
        return host

    def create(
        self,
        *,
        name: str | HostName,
        comment: str = "",
        contacts: list[str] | None = None,
        ipaddress: IP_AddressT | str | None = None,
        network: IP_NetworkT | str | None = None,
        ttl: int | None = None,
        fetch_after_create: bool = True,
    ) -> Host | None:
        """Create a host."""
        data: dict[str, Any] = {"name": str(name)}
        if comment:
            data["comment"] = comment
        if contacts:
            data["contacts"] = contacts
        if ipaddress:
            data["ipaddress"] = str(ipaddress)
        if network:
            data["network"] = str(network)
        if ttl is not None:
            data["ttl"] = ttl
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | Host,
        *,
        name: str | HostName | None = None,
        comment: str | None = None,
        ttl: int | None = None,
    ) -> Host:
        """Update a host's mutable fields. ``None`` kwargs are left unchanged."""
        host = self._resolve(ref)
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = str(name)
        if comment is not None:
            data["comment"] = comment
        if ttl is not None:
            data["ttl"] = ttl
        return self._patch(host, data)
