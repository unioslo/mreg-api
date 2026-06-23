"""Resource managers: encapsulated access to API resources.

Each manager operates on a specific model type and exposes a set of operations.
Managers are bound to an `MregClient` instance.

As a baseline, each manager exposes:
- `get`
- `list`
- `ensure_absent`  (TODO: rename)

Additionally, managers for resources that support name-based lookups:
- `get_by_name`

For resources that support write operations:
- `create`
- `update`
- `delete`

For resources that support history:
- `history`
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import Literal
from typing import Protocol
from typing import TypeVar
from typing import overload

from typing_extensions import override

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
from mreg_api.models import HostGroup
from mreg_api.models import NetworkOrIP
from mreg_api.models.fields import MacAddress
from mreg_api.models.history import HistoryItem
from mreg_api.models.history import HistoryResource
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter
from mreg_api.utilities.shared import convert_wildcard_to_regex

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


class ResourceManager(Generic[T], ABC):
    """Basic manager for performing read operations on an API resource type."""

    _url_identifier: ClassVar[str] = "id"
    """The name of the field that holds the URL MID field for the resource.

    I.e. for most resources the URL ID field is the numeric ID field:
        GET /api/sshfps/123 # 200

    But for other resources, the URL ID field is a different field (e.g. name, network, host):
        GET /api/hosts/example.com # 200
        GET /api/v1/networks/192.168.0.0/24 # 200
    """

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client that owns it."""
        self._client: MregClient = client

    @property
    @abstractmethod
    def model(self) -> type[T]:
        """The model type this manager operates on."""
        raise NotImplementedError

    def _endpoint(self) -> Endpoint:
        return self.model.endpoint()

    def _resolve(self, ref: int | T) -> T:
        """Resolve an object reference (instance or numeric id) to an instance of the model.

        Fetches the object from the server if only an ID is provided.

        Raises:
            EntityNotFound: If the object cannot be found by its ID.
        """
        if isinstance(ref, self.model):
            return ref

        # Ideally this would use the primary path (URL identifier)
        # but we would need to either resolve the type of identifier, coerce
        # the value to str, or use a generic type variable for the identifier type.
        # For now, we only accept the id field for resolution.
        obj = self._fetch_by_field("id", ref)  # TODO: resolve type narrowing!
        if obj is None:
            raise EntityNotFound(f"{self.model.__name__} with id {ref!r} not found.")
        return obj

    def _validate(self, data: Any) -> T:
        return get_type_adapter(self.model).validate_python(data)

    def id_field_value(self, obj: T) -> str | int:
        """Get the value of the field that is used as the URL ID for ``obj``."""
        return getattr(obj, self._url_identifier)

    # TODO: test for all model types/managers! getattr is not great!
    def _endpoint_with_id(self, obj: T) -> str:
        return self._endpoint().with_id(self.id_field_value(obj))

    def _fetch_by_field(self, field: str, value: str | int) -> T | None:
        """Fetch a single object by a field, mirroring the old APIMixin.get_by_field."""
        endpoint = self._endpoint()

        # Field is the ID used in the URL path. i.e. /hosts/example.com, /sshfps/123, etc.
        if self._url_identifier != "id" and field == self._url_identifier:
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

    # NOTE: add toggleable _refetch behavior? Return None if refetching is disabled?
    def _refetch(self, obj: T) -> T:
        """Fetch a fresh copy of ``obj`` from the server."""
        lookup = (
            getattr(obj, "id", None) if hasattr(obj, "id") else getattr(obj, self._url_identifier, None)
        )
        if not lookup:
            raise InternalError(f"Could not determine identifier for {self.model.__name__}.")
        fresh = self._fetch_by_field("id" if hasattr(obj, "id") else self._url_identifier, lookup)
        if not fresh:
            raise GetError(f"Could not refetch {self.model.__name__} ({lookup!r}).")
        return fresh

    @overload
    def get(self, ident: str | int, *, required: Literal[True]) -> T: ...
    @overload
    def get(self, ident: str | int, *, required: Literal[False] = ...) -> T | None: ...
    def get(self, ident: str | int, *, required: bool = False) -> T | None:
        """Get a resource by its natural identifier.

        Args:
            ident: The URL identifier (id / name / network, per resource).
            required: When ``True``, raise if the resource is missing (returns ``T``).
                When ``False`` (default), return ``T | None``.

        Raises:
            EntityNotFound: If `required` is True and the resource is not found.

        Returns:
            The resource, or ``None`` when `required` is False.
        """
        obj = self._fetch_by_field(self._url_identifier, ident)
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
        if self._fetch_by_field(self._url_identifier, ident) is not None:
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


class WriteResourceManager(ResourceManager[T], ABC):
    """Manager for performing CRUD operations on an API resource type."""

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

    def delete(self, obj: T) -> None:
        """Delete a resource.

        Args:
            obj (T): The resource to delete.
        """
        _ = self._client.delete(self._endpoint_with_id(obj))


class NamedResourceManager(WriteResourceManager[T], ABC):
    """Extended WriteResourceManager for resources that support name-based lookups."""

    # NOTE: does not currently handle name-based lookups for resources that do
    # not support writes (but they don't exist per now!)

    name_field: ClassVar[str] = "name"
    name_lowercase: ClassVar[bool] = False

    def _case_name(self, name: str) -> str:
        return name.lower() if self.name_lowercase else name

    # FIXME: Do we actually need an explicit name based lookup? Can we just use get_by_field() instead?
    #       get() should handle the "primary" path i.e. the URL identifier.
    #       The only affordance this gives us is the ability to format the name (lowercase, etc.)
    #       before performing the lookup.
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

    def rename(self, obj: T, new_name: str) -> T:
        """Rename the resource.

        Args:
            obj: The resource to rename.
            new_name: The new name to set.

        Returns:
            The patched resource.
        """
        return self._patch(obj, {self.name_field: self._case_name(new_name)})

    # TODO: add list_by_name ? Didn't exist in prior version

    # RENAMED: get_list_by_name_regex -> list_by_name_regex
    def list_by_name_regex(self, name: str) -> list[T]:
        """Get multiple resources by a name regex.

        Args:
            name: The regex pattern for names to search for.

        Returns:
            A list of resource objects.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        param, value = convert_wildcard_to_regex(self.name_field, self._case_name(name), True)
        # NOTE: why can we use T here? Is this a Python 3.11 thing?
        # return MregClient().get_typed(cls.endpoint(), list[T], params={param: value})
        return MregClient().get_typed(self._endpoint(), list[self.model], params={param: value})


class HistoryManager(ResourceManager[T], ABC):
    """Manager capable of fetching history for a resource."""

    @property
    @abstractmethod
    def history_resource(self) -> HistoryResource:
        """The history resource corresponding to this manager's resource.

        Used to construct the query for fetching history items.
        """
        ...

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


class HostManager(NamedResourceManager[Host], HistoryManager[Host]):
    """Operations on :class:`~mreg_api.models.Host` resources."""

    _url_identifier: ClassVar[str] = "name"

    @property
    @override
    def model(self) -> type[Host]:
        return Host

    @property
    @override
    def history_resource(self) -> HistoryResource:
        return HistoryResource.Host

    def _resolve(self, ref: int | Host) -> Host:
        """Resolve a host reference (instance or numeric id) to a Host."""
        if isinstance(ref, Host):
            return ref
        host = self._fetch_by_field("id", ref)
        if host is None:
            raise EntityNotFound(f"Host with id {ref!r} not found.")
        return host

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
            Host | None: The created host, or None if creation failed.
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


class HostGroupManager(NamedResourceManager[HostGroup], HistoryManager[HostGroup]):
    """Operations on :class:`~mreg_api.models.HostGroup` resources."""

    _url_identifier: ClassVar[str] = "name"

    @property
    @override
    def model(self) -> type[HostGroup]:
        return HostGroup

    @property
    @override
    def history_resource(self) -> HistoryResource:
        return HistoryResource.Group

    def create(
        self,
        *,
        name: str,
        description: str | Unset = UNSET,
        fetch_after_create: bool = True,
    ) -> HostGroup | None:
        """Create a host group."""
        data: dict[str, Any] = {"name": name}
        if description is not UNSET:
            data["description"] = description
        return self._create(data, fetch_after_create=fetch_after_create)

    # Not much to update here, but we implement update for future expansion + consistent interface
    def update(
        self,
        ref: int | HostGroup,
        *,
        description: str | Unset = UNSET,
    ) -> HostGroup:
        """Update a host group's mutable fields."""
        group = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(group, data)

    def set_description(self, group: int | HostGroup, description: str) -> HostGroup:
        """Set the description for the host group."""
        group = self._resolve(group)
        return self.update(group, description=description)

    def add_group(self, group: int | HostGroup, name: str) -> HostGroup:
        """Add a group to a host group."""
        group = self._resolve(group)

        self._client.post(
            Endpoint.HostGroupsAddHostGroups.with_params(group.name),
            json={"name": name},
        )
        return self._refetch(group)

    def remove_group(self, group: int | HostGroup, name: str) -> HostGroup:
        """Remove a group from a host group."""
        group = self._resolve(group)

        self._client.delete(
            Endpoint.HostGroupsRemoveHostGroups.with_params(group.name, name),
        )
        return self._refetch(group)

    def add_host(self, group: int | HostGroup, name: str) -> HostGroup:
        """Add a host to a host group."""
        group = self._resolve(group)

        self._client.post(
            Endpoint.HostGroupsAddHosts.with_params(group.name),
            json={"name": name},
        )
        return self._refetch(group)

    def remove_host(self, group: int | HostGroup, name: str) -> HostGroup:
        """Remove a host from a host group."""
        group = self._resolve(group)

        self._client.delete(
            Endpoint.HostGroupsRemoveHosts.with_params(group.name, name),
        )
        return self._refetch(group)

    def add_owner(self, group: int | HostGroup, name: str) -> HostGroup:
        """Add an owner to a host group."""
        group = self._resolve(group)

        self._client.post(
            Endpoint.HostGroupsAddOwner.with_params(group.name),
            json={"name": name},
        )
        return self._refetch(group)

    def remove_owner(self, group: int | HostGroup, name: str) -> HostGroup:
        """Remove an owner from a host group."""
        group = self._resolve(group)

        self._client.delete(
            Endpoint.HostGroupsRemoveOwner.with_params(group.name, name),
        )
        return self._refetch(group)

    # RENAMED: get_all_parents -> list_parents
    def list_parents(self, group: int | HostGroup) -> list[HostGroup]:
        """Get all parent groups of a host group.

        Renamed from `get_all_parents` to `list_parents`
        """
        group = self._resolve(group)
        parents: list[HostGroup] = []
        for parent in group.parent:  # why singular name?
            pobj = self._fetch_by_field("name", parent)
            if pobj:
                parents.append(pobj)
                parents.extend(self.list_parents(pobj))
        return parents
