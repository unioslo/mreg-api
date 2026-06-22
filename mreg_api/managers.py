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
from mreg_api.events import Event
from mreg_api.events import EventKind
from mreg_api.events import ObjectRef
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import GetError
from mreg_api.exceptions import InternalError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.models import Host
from mreg_api.models import NetworkOrIP
from mreg_api.models.fields import MacAddress
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


class Unset:
    """Sentinel for "unchanged" parameters in update methods."""

    def __repr__(self) -> str:
        """Return a string representation of the Unset sentinel."""
        return "UNSET"


UNSET = Unset()


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

    def _fetch_list_by_field(self, field: str, value: str | int) -> list[T]:
        """Fetch all objects matching a field, mirroring old APIMixin.get_list_by_field."""
        params: QueryParams = {field: value}
        return self._client.get_typed(self._endpoint(), list[self.model], params=params, limit=500)

    def _refetch(self, obj: T) -> T:
        """Fetch a fresh copy of ``obj`` from the server."""
        # TODO: rewrite models to specify their ID field explicitly
        # and write tests that ensure it is always present and populated.
        # Perhaps make it an abstract property like the endpoint.
        # Furthermore, make the ID field an object that carries info
        # we currently hardcode in the endpoint's `requires_search_for_id` and `external_id_field` logic
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

    Hosts are fetchable by several identifiers, each via an explicit per-kind getter:
    :meth:`get`/:meth:`get_by_name` (hostname), :meth:`get_by_id`, :meth:`get_by_ip`,
    :meth:`get_by_mac` (plus the :meth:`list_by_ip`/:meth:`list_by_mac` plural forms).

    There is deliberately **no** "fetch by any means" method (ADR-0001): guessing the
    *kind* of a free-text identifier is a CLI/UX concern. The CLI composes these getters
    in its documented order (id → IP → MAC → name → CNAME), resolving CNAMEs itself via
    ``client.cnames`` — the library never follows a CNAME and emits no CNAME event.
    PTR fallback, by contrast, is library-internal, so :meth:`get_by_ip`/:meth:`list_by_ip`
    always record a ``RESOLUTION`` event when an IP matches via a PTR override.
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

    # --- explicit per-kind getters (resolution is the CLI's job, ADR-0001) ---

    def _record_ptr_event(self, host: Host, ip: str) -> None:
        """Record that ``ip`` resolved to ``host`` via a PTR override.

        PTR fallback is library-internal (the caller cannot otherwise tell the match
        came from a PTR override), so it is always surfaced as an event; the CLI filters
        events at display time (ADR-0001). CNAME resolution, in contrast, is composed by
        the CLI and is not a library event.
        """
        self._client.events.record(
            Event(
                kind=EventKind.RESOLUTION,
                message=f"{ip} is a PTR override for {host.name}",
                subject=ObjectRef.new(host),
                related=(ObjectRef("PTR_override", ip, field="ipaddress"),),
                correlation_id=self._client.get_correlation_id(),
            )
        )

    @overload
    def get_by_id(self, host_id: int, *, required: Literal[True]) -> Host: ...
    @overload
    def get_by_id(self, host_id: int, *, required: Literal[False] = ...) -> Host | None: ...
    def get_by_id(self, host_id: int, *, required: bool = False) -> Host | None:
        """Get a host by its numeric id.

        Distinct from :meth:`get`: the Host endpoint id-field is the hostname, so
        :meth:`get` resolves by name while this resolves by the numeric ``id``.
        """
        obj = self._fetch_by_field("id", host_id)
        if required and obj is None:
            raise EntityNotFound(f"Host with id {host_id!r} not found.")
        return obj

    @overload
    def get_by_ip(self, ip: str | IP_AddressT, *, required: Literal[True]) -> Host: ...
    @overload
    def get_by_ip(self, ip: str | IP_AddressT, *, required: Literal[False] = ...) -> Host | None: ...
    def get_by_ip(self, ip: str | IP_AddressT, *, required: bool = False) -> Host | None:
        """Get a host by IP address (A/AAAA, falling back to PTR override).

        Falls back to a PTR override when no direct A/AAAA match exists; a PTR match
        always records a ``RESOLUTION`` event (no opt-out — see ADR-0001).
        """
        addr = str(NetworkOrIP.parse_or_raise(str(ip), mode="ip"))
        try:
            host = self._fetch_by_field("ipaddresses__ipaddress", addr)
            if host is None:
                host = self._fetch_by_field("ptr_overrides__ipaddress", addr)
                if host is not None:
                    self._record_ptr_event(host, addr)
        except MultipleEntitiesFound as e:
            raise MultipleEntitiesFound(f"Multiple hosts found with IP address {addr}.") from e
        if required and host is None:
            raise EntityNotFound(f"Host with IP address {addr} not found.")
        return host

    @overload
    def get_by_mac(self, mac: str | MacAddress, *, required: Literal[True]) -> Host: ...
    @overload
    def get_by_mac(self, mac: str | MacAddress, *, required: Literal[False] = ...) -> Host | None: ...
    def get_by_mac(self, mac: str | MacAddress, *, required: bool = False) -> Host | None:
        """Get a host by MAC address."""
        addr = MacAddress.parse_or_raise(mac)
        host = self._fetch_by_field("ipaddresses__macaddress", str(addr))
        if required and host is None:
            raise EntityNotFound(f"Host with MAC address {addr} not found.")
        return host

    # NOTE: not sure if it makes sense to fetch _multiple_ hosts given the same IP/MAC
    # but these methods are implemented for parity with the old `get_list_by_ip_or_raise`
    def list_by_ip(self, ip: str | IP_AddressT) -> list[Host]:
        """List hosts by IP address (A/AAAA, falling back to PTR override).

        Args:
            ip (str | IP_AddressT): IP address to filter by.

        Returns:
            list[Host]: List of hosts matching the IP address.
        """
        addr = str(NetworkOrIP.parse_or_raise(str(ip), mode="ip"))
        hosts = self._fetch_list_by_field("ipaddresses__ipaddress", addr)
        if not hosts:
            hosts = self._fetch_list_by_field("ptr_overrides__ipaddress", addr)
            for host in hosts:
                self._record_ptr_event(host, addr)
        return hosts

    def list_by_mac(self, mac: str | MacAddress) -> list[Host]:
        """List hosts by MAC address. Returns ``[]`` when nothing matches."""
        addr = MacAddress.parse_or_raise(mac)
        return self._fetch_list_by_field("ipaddresses__macaddress", str(addr))

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

        Args:
            name (str | HostName): Name of the host to create.
            comment (str, optional): Comment for the host. Defaults to "".
            contacts (list[str] | None, optional): List of contacts for the host.
            ipaddress (IP_AddressT | str | None, optional): IP address of the host.
            network (IP_NetworkT | str | None, optional): Network of the host.
            fetch_after_create (bool, optional): Whether to fetch the host after creation.

        Returns:
            Host | None: _description_
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
        name: str | HostName | Unset = UNSET,
        comment: str | None | Unset = UNSET,
        contacts: list[str] | Unset = UNSET,
        ttl: int | None | Unset = UNSET,
    ) -> Host:
        """Update a host's mutable fields."""
        host = self._resolve(ref)
        data: dict[str, Any] = {}
        if name is not UNSET:
            data["name"] = str(name)
        if comment is not UNSET:
            data["comment"] = comment
        if contacts is not UNSET:
            data["contacts"] = contacts
        if ttl is not UNSET:
            data["ttl"] = ttl
        return self._patch(host, data)
