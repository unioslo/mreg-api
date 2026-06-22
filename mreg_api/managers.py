"""Resource managers: the public, client-bound API surface.

Each manager owns the operations for one API resource. A manager holds a
back-reference to the :class:`~mreg_api.client.MregClient` that constructed it and
uses the client's low-level HTTP/pagination primitives to fetch data and build
(dumb) models from it.

The generic :class:`ResourceManager` provides the shared CRUD verbs once; concrete
per-resource managers bind the model type and add resource-specific methods.

Error handling: the client's HTTP verbs (``get``/``post``/``patch``/``delete``)
already raise the appropriate typed error (:class:`GetError`, :class:`PostError`,
:class:`PatchError`, :class:`DeleteError`) when the server returns a non-success,
non-404 status. Managers therefore never re-check ``response.is_success`` — a
returned response is always successful — and let those typed errors propagate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import Literal
from typing import Protocol
from typing import TypeVar
from typing import cast
from typing import overload

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import GetError
from mreg_api.exceptions import InternalError
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
    the managers. This is the lower bound for the model type a manager can operate
    on, and the type of :attr:`ResourceManager._model`.
    """

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the resource."""
        ...


T = TypeVar("T", bound=APIResource)


class ResourceManager(Generic[T]):
    """Manager for performing CRUD operations on an API resource type."""

    # `_model` is typed against the `APIResource` bound rather than `Any`, so
    # assigning an incompatible class in a subclass is a static error. It cannot be
    # typed `ClassVar[type[T]]` directly because a `ClassVar` may not reference a
    # `TypeVar`; the `model` property bridges that gap with a single, contained cast.
    _model: ClassVar[type[APIResource]]
    """The model type this manager operates on. Bound by each subclass."""

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client that owns it."""
        self._client: MregClient = client

    # --- internal helpers ------------------------------------------------

    @property
    def model(self) -> type[T]:
        """The model type this manager operates on."""
        # Safe: each subclass binds `_model` to its concrete `type[T]` (e.g.
        # `HostManager._model = Host`); the cast only recovers the precise `T` that
        # the `ClassVar` declaration cannot express.
        return cast("type[T]", self._model)

    def _endpoint(self) -> Endpoint:
        return self.model.endpoint()

    def _validate(self, data: Any) -> T:
        return get_type_adapter(self.model).validate_python(data)

    def _id_field(self, obj: T) -> str:
        return obj.endpoint().external_id_field()

    def _endpoint_with_id(self, obj: T) -> str:
        return obj.endpoint().with_id(getattr(obj, self._id_field(obj)))

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
        """POST ``data`` to the resource endpoint, optionally fetching the result.

        Raises :class:`PostError` (from the client) if the server rejects the create.
        Returns ``None`` when the create succeeds but the server provides no
        ``Location`` header to refetch from (many endpoints don't), or when
        ``fetch_after_create`` is ``False``.
        """
        response = self._client.post(self._endpoint(), json=data)
        if fetch_after_create and "Location" in response.headers:
            return self._client.get_typed(response.headers["Location"], self.model)
        return None

    def _patch(self, obj: T, data: dict[str, Any], *, params: QueryParams | None = None) -> T:
        """PATCH ``obj`` with ``data`` and return the refetched object.

        Raises :class:`PatchError` (from the client) if the server rejects the patch.
        """
        _ = self._client.patch(self._endpoint_with_id(obj), json=data, params=params)
        return self._refetch(obj)

    # --- public CRUD -----------------------------------------------------

    @overload
    def get(self, ident: str | int, *, required: Literal[True]) -> T: ...
    @overload
    def get(self, ident: str | int, *, required: Literal[False] = ...) -> T | None: ...
    def get(self, ident: str | int, *, required: bool = False) -> T | None:
        """Get a resource by its natural identifier.

        Args:
            ident: The external identifier (id / name / network, per resource).
            required: When ``True``, raise if the resource is missing (returns ``T``).
                When ``False`` (default), return ``T | None``.

        Raises:
            EntityNotFound: If `required` is True and the resource is not found.

        Returns:
            The resource, or ``None`` when `required` is False.
        """
        obj = self._fetch_by_field(self._endpoint().external_id_field(), ident)
        if required and obj is None:
            raise EntityNotFound(f"{self.model.__name__} {ident!r} not found.")
        return obj

    def ensure_absent(self, ident: str | int) -> None:
        """Assert that no resource with ``ident`` exists.

        The "must not exist" guard (replaces the old ``get_x_and_raise``); kept a
        distinct verb rather than a return-typed-``None`` overload of :meth:`get`.

        Raises:
            EntityAlreadyExists: If a resource with `ident` exists.
        """
        if self._fetch_by_field(self._endpoint().external_id_field(), ident) is not None:
            raise EntityAlreadyExists(f"{self.model.__name__} {ident!r} already exists.")

    def list(
        self,
        *,
        limit: int | None = 500,
        **query: str | int | float | bool | None,
    ) -> list[T]:
        """List resources, optionally filtered by query parameters."""
        params: QueryParams = dict(query)
        return self._client.get_typed(self._endpoint(), list[self.model], params=params, limit=limit)

    def delete(self, obj: T) -> None:
        """Delete a resource.

        Raises :class:`DeleteError` (from the client) if the server rejects the
        delete. Returns ``None`` — success is implied by returning normally (the old
        ``bool`` was always ``True``-or-raise; see ADR-0003).
        """
        _ = self._client.delete(self._endpoint_with_id(obj))


class NamedResourceManager(ResourceManager[T]):
    """Extended manager for resources that support name-based lookups."""

    name_field: ClassVar[str] = "name"
    name_lowercase: ClassVar[bool] = False

    def _case_name(self, name: str) -> str:
        return name.lower() if self.name_lowercase else name

    @overload
    def get_by_name(self, name: str, *, required: Literal[True]) -> T: ...
    @overload
    def get_by_name(self, name: str, *, required: Literal[False] = ...) -> T | None: ...
    def get_by_name(self, name: str, *, required: bool = False) -> T | None:
        """Get a resource by name (searches the name field).

        For the "must not exist" guard use :meth:`ensure_absent` (for named resources
        the external id-field is the name, so it covers name absence).
        """
        obj = self._fetch_by_field(self.name_field, self._case_name(name))
        if required and obj is None:
            raise EntityNotFound(f"{self.model.__name__} {name!r} not found.")
        return obj


class HistoryCapableManager:
    """Mixin for managers whose resource records audit history.

    A standalone mixin (no ``__init__``, does not inherit :class:`ResourceManager`)
    so it composes without unsafe multiple inheritance. It declares the ``_client``
    binding that the concrete :class:`ResourceManager` it is mixed into supplies at
    runtime, letting :meth:`history` use it with full type information. Only managers
    that mix this in expose :meth:`history`, so e.g. ``client.networks.history`` is a
    static type error.
    """

    # Supplied at runtime by the ResourceManager this mixin is combined with; the
    # mixin itself has no __init__, hence the targeted ignore.
    _client: MregClient  # pyright: ignore[reportUninitializedInstanceVariable]
    history_resource: ClassVar[HistoryResource]

    def history(self, name: str) -> list[HistoryItem]:
        """Get the audit history for a named resource.

        Relocated from the former ``HistoryItem.get``: fetches history through the
        owning client and constructs :class:`HistoryItem` models from the result.
        """
        resource = self.history_resource
        params: QueryParams = {"resource": resource.resource(), "name": name}
        ret = self._client.get_typed(Endpoint.History, list[HistoryItem], params=params)
        if len(ret) == 0:
            # No-history is a valid state, not a not-found error (ADR-0003).
            return []

        model_ids = ",".join({str(i.mid) for i in ret})
        params = {"resource": resource.resource(), "model_id__in": model_ids}
        ret = self._client.get_typed(Endpoint.History, list[HistoryItem], params=params)

        params = {"data__relation": resource.relation(), "data__id__in": model_ids}
        ret.extend(self._client.get_typed(Endpoint.History, list[HistoryItem], params=params))

        return ret


class HostManager(NamedResourceManager[Host], HistoryCapableManager):
    """Operations on :class:`~mreg_api.models.Host` resources.

    Hosts are fetchable by several identifiers (hostname, id, IP, MAC). The opaque
    "fetch by any means" resolution (the relocated ``Host.get_by_any_means``
    algorithm) lands here as part of the dedicated relocation step; until then
    :meth:`get` resolves by the endpoint id-field (hostname) and :meth:`get_by_name`
    is inherited. The intended end state is a single clean :meth:`get` that does the
    right thing under the hood, with IP/MAC/network always passed as strings.
    """

    # Declared type matches the base exactly (a `ClassVar` is invariant, so narrowing
    # to `type[Host]` would be an override error); `model` recovers the precise type.
    _model: ClassVar[type[APIResource]] = Host
    history_resource: ClassVar[HistoryResource] = HistoryResource.Host

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
        fetch_after_create: bool = True,
    ) -> Host | None:
        """Create a host.

        Keys mirror the create payload the CLI builds (``name`` required; ``contacts``,
        ``comment``, ``ipaddress``, ``network`` optional). ``ipaddress`` and ``network``
        are mutually exclusive at the server.
        """
        data: dict[str, Any] = {"name": str(name)}
        if comment:
            data["comment"] = comment
        if contacts:
            data["contacts"] = contacts
        if ipaddress:
            data["ipaddress"] = str(ipaddress)
        if network:
            data["network"] = str(network)
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | Host,
        *,
        name: str | HostName | None = None,
        comment: str | None = None,
        contacts: list[str] | None = None,
        ttl: int | None = None,
    ) -> Host:
        """Update a host's mutable fields. ``None`` kwargs are left unchanged.

        Keys mirror the host PATCH operations the CLI performs: ``name`` (rename),
        ``comment`` (set_comment), ``contacts`` (set_contacts, replaces existing),
        ``ttl`` (set_ttl).
        """
        host = self._resolve(ref)
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = str(name)
        if comment is not None:
            data["comment"] = comment
        if contacts is not None:
            data["contacts"] = contacts
        if ttl is not None:
            data["ttl"] = ttl
        return self._patch(host, data)
