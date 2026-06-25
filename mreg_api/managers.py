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

import ipaddress
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
from mreg_api.exceptions import InputFailure
from mreg_api.exceptions import InternalError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.models import CNAME
from mreg_api.models import MX
from mreg_api.models import NAPTR
from mreg_api.models import SSHFP
from mreg_api.models import TXT
from mreg_api.models import BacnetID
from mreg_api.models import HInfo
from mreg_api.models import Host
from mreg_api.models import HostGroup
from mreg_api.models import IPAddress
from mreg_api.models import Label
from mreg_api.models import Location
from mreg_api.models import Network
from mreg_api.models import NetworkOrIP
from mreg_api.models import NetworkPolicy
from mreg_api.models import NetworkPolicyAttribute
from mreg_api.models import NetworkPolicyAttributeValue
from mreg_api.models import Permission
from mreg_api.models import PTR_override
from mreg_api.models import Srv
from mreg_api.models.fields import HostName
from mreg_api.models.fields import MacAddress
from mreg_api.models.history import HistoryItem
from mreg_api.models.history import HistoryResource
from mreg_api.types import IP_AddressT
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter
from mreg_api.utilities.shared import convert_wildcard_to_regex

if TYPE_CHECKING:
    from mreg_api.client import MregClient
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

    # TODO: rename var to indicate more clearly that it is the field used in the URL path
    _url_identifier: ClassVar[str] = "id"
    """The name of the field that is used to identify the resource in the URL path.

    I.e. for most resources the URL ID field is the numeric ID field:
        GET /api/sshfps/123 # 200

    But for other resources, the URL ID field is a different field (e.g. name, network, host):
        GET /api/hosts/example.com # 200
        GET /api/hosts/123 # 404

        GET /api/v1/networks/192.168.0.0/24 # 200
        GET /api/v1/networks/123 # 404
    """

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client that owns it."""
        self._client: MregClient = client

    def _normalize_id(self, ident: str | int) -> str | int:
        """Normalise an identifier before use in lookups. Override to add hostname expansion."""
        return ident

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

        # Ideally this method call would use the primary path (URL identifier) not just int,
        # but we would need to either resolve the type of identifier, coerce
        # the value to str, or use a generic type variable for the identifier type.
        # For now, we only accept the id field for resolution.
        obj = self._fetch_by_field("id", ref)  # TODO: resolve type narrowing!
        if obj is None:
            raise EntityNotFound(f"{self.model.__name__} with id {ref!r} not found.")
        return obj

    def _validate_json(self, data: str) -> T:
        """Attempt to construct the manager's model type from JSON data.

        Args:
            data (str): JSON data to construct model with.

        Returns:
            T: An instance of the manager's model type.

        """
        return get_type_adapter(self.model).validate_json(data)

    def _validate(self, data: Any) -> T:
        """Attempt to construct the manager's model type from data.

        Args:
            data (Any): Data to construct model with. Usually a dict.

        Returns:
            T: An instance of the manager's model type.

        """
        return get_type_adapter(self.model).validate_python(data)

    def id_field_value(self, obj: T) -> str | int:
        """Get the value of the field that is used as the URL ID for ``obj``."""
        return getattr(obj, self._url_identifier)

    # TODO: test for all model types/managers! getattr is not great!
    def _endpoint_with_id(self, obj: T) -> str:
        return self._endpoint().with_id(self.id_field_value(obj))

    def _fetch_by_field(self, field: str, value: str | int) -> T | None:
        """Fetch a single object, querying by a field."""
        endpoint = self._endpoint()

        # Field is the ID used in the URL path. i.e. /hosts/example.com, /sshfps/123, etc.
        if self._url_identifier != "id" and field == self._url_identifier:
            resp = self._client.get(endpoint.with_id(value), ok404=True)
            if not resp:
                return None
            return self._validate_json(resp.text)

        # Lookup by non-ID field, i.e. `/hosts?ipaddress=foo` instead of `/hosts/{id}`.
        data = self._client.get_item_by_key_value(endpoint, field, value, ok404=True)
        if not data:
            return None
        # TODO: refactor the get_list/get_item_by_key_value methods to return a response
        # OR JSON strings, so we can pass everything through _validate_json().
        # Pydantic's JSON parser is faster than passing everything through the stdlib
        # JSON parser, and then through _validate().
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
        ident = self._normalize_id(ident)
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
        ident = self._normalize_id(ident)
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

    # TODO: should override _url_identifier instead?!
    name_field: ClassVar[str] = "name"
    name_lowercase: ClassVar[bool] = False

    def _case_name(self, name: str) -> str:
        return name.lower() if self.name_lowercase else name

    def _normalize_name(self, name: str) -> str:
        """Normalise a name before use in lookups. Override to add hostname expansion."""
        return self._case_name(name)

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
        name = self._normalize_name(name)
        obj = self._fetch_by_field(self.name_field, name)
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
        return self._patch(obj, {self.name_field: self._normalize_name(new_name)})

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

        param, value = convert_wildcard_to_regex(self.name_field, self._normalize_name(name), True)
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

    def _normalize_history_name(self, name: str) -> str:
        """Normalise a name before use in history lookups. Override for hostname expansion."""
        return name

    def history(self, name: str) -> list[HistoryItem]:
        """Get the audit history for a named resource.

        Relocated from the former ``HistoryItem.get``: fetches history through the
        owning client and constructs :class:`HistoryItem` models from the result.
        """
        name = self._normalize_history_name(name)
        resource = self.history_resource
        params: QueryParams = {"resource": resource.resource(), "name": name}
        ret = self._client.get_typed(Endpoint.History, list[HistoryItem], params=params)
        if len(ret) == 0:
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

    def _normalize_id(self, ident: str | int) -> str | int:
        return self._client.fqdn(ident) if isinstance(ident, str) else ident

    # NOTE: why do we have a separate _normalize_history_name?
    #       They do the exact same thing...
    def _normalize_name(self, name: str) -> str:
        return self._client.fqdn(name)

    def _normalize_history_name(self, name: str) -> str:
        return self._client.fqdn(name)

    def _resolve(self, ref: int | Host) -> Host:
        """Resolve a host reference (instance or numeric id) to a Host."""
        if isinstance(ref, Host):
            return ref
        host = self._fetch_by_field("id", ref)
        if host is None:
            raise EntityNotFound(f"Host with id {ref!r} not found.")
        return host

    def _record_ptr_event(self, host: Host, ip: str) -> None:
        """Record that ``ip`` resolved to ``host`` via a PTR override."""
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
        """List hosts by MAC address."""
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
        data: dict[str, Any] = {"name": self._client.fqdn(name)}
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
            data["name"] = self._client.fqdn(str(name))
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
        name = self._client.fqdn(name)

        self._client.post(
            Endpoint.HostGroupsAddHosts.with_params(group.name),
            json={"name": name},
        )
        return self._refetch(group)

    def remove_host(self, group: int | HostGroup, name: str) -> HostGroup:
        """Remove a host from a host group."""
        group = self._resolve(group)
        name = self._client.fqdn(name)

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


class LabelManager(NamedResourceManager[Label]):
    """Operations on :class:`~mreg_api.models.Label` resources."""

    @property
    @override
    def model(self) -> type[Label]:
        return Label

    def create(
        self,
        *,
        name: str,
        description: str,
        fetch_after_create: bool = True,
    ) -> Label | None:
        """Create a label."""
        return self._create(
            {"name": name, "description": description}, fetch_after_create=fetch_after_create
        )

    def update(
        self,
        ref: int | Label,
        *,
        description: str | Unset = UNSET,
    ) -> Label:
        """Update a label's mutable fields."""
        label = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(label, data)

    def set_description(self, label: int | Label, description: str) -> Label:
        """Set the description for the label."""
        label = self._resolve(label)
        return self.update(label, description=description)


class PermissionManager(WriteResourceManager[Permission]):
    """Operations on :class:`~mreg_api.models.Permission` resources."""

    @property
    @override
    def model(self) -> type[Permission]:
        return Permission

    def create(
        self,
        *,
        group: str,
        range: str,  # noqa: A002
        regex: str,
        labels: list[int] | None = None,
        fetch_after_create: bool = True,
    ) -> Permission | None:
        """Create a permission.

        Args:
            group: The netgroup the permission applies to.
            range: The network range (CIDR) the permission covers.
            regex: The host regex pattern for the permission.
            labels: Optional list of label IDs to attach.
            fetch_after_create: Whether to fetch and return the created object.
        """
        data: dict[str, Any] = {"group": group, "range": range, "regex": regex}

        # TODO: check if we can actually pass labels during creation
        # TODO: also accept Label objects?
        if labels is not None:
            data["labels"] = labels
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | Permission,
        *,
        group: str | Unset = UNSET,
        range: str | Unset = UNSET,  # noqa: A002
        regex: str | Unset = UNSET,
        labels: list[int] | Unset = UNSET,
    ) -> Permission:
        """Update a permission's mutable fields."""
        perm = self._resolve(ref)
        data: dict[str, Any] = {}
        if group is not UNSET:
            data["group"] = group
        if range is not UNSET:
            data["range"] = range
        if regex is not UNSET:
            data["regex"] = regex
        if labels is not UNSET:
            data["labels"] = labels
        return self._patch(perm, data)

    def add_label(self, ref: int | Permission, label_name: str) -> Permission:
        """Add a label to the permission by label name.

        Args:
            ref: The permission (instance or numeric id).
            label_name: The name of the label to add.

        Raises:
            EntityNotFound: If the label does not exist.
            EntityAlreadyExists: If the permission already has this label.
        """
        perm = self._resolve(ref)
        label = LabelManager(self._client).get_by_name(label_name, required=True)
        if label.id in perm.labels:
            raise EntityAlreadyExists(f"Permission already has label {label_name!r}.")
        return self.update(perm, labels=[*perm.labels, label.id])

    def remove_label(self, ref: int | Permission, label_name: str) -> Permission:
        """Remove a label from the permission by label name.

        Args:
            ref: The permission (instance or numeric id).
            label_name: The name of the label to remove.

        Raises:
            EntityNotFound: If the label does not exist or the permission lacks it.
        """
        perm = self._resolve(ref)
        label = LabelManager(self._client).get_by_name(label_name, required=True)
        if label.id not in perm.labels:
            raise EntityNotFound(f"Permission does not have label {label_name!r}.")
        return self.update(perm, labels=[lid for lid in perm.labels if lid != label.id])

    @overload
    def get_by_triplet(
        self,
        group: str,
        range: str,  # noqa: A002
        regex: str,
        *,
        required: Literal[True],
    ) -> Permission: ...
    @overload
    def get_by_triplet(
        self,
        group: str,
        range: str,  # noqa: A002
        regex: str,
        *,
        required: Literal[False] = ...,
    ) -> Permission | None: ...
    def get_by_triplet(
        self,
        group: str,
        range: str,  # noqa: A002
        regex: str,
        *,
        required: bool = False,
    ) -> Permission | None:
        """Get a permission by the (group, range, regex) triplet.

        Replaces ``Permission.get_by_query_unique_or_raise`` from the old model API.

        Raises:
            MultipleEntitiesFound: If more than one permission matches the triplet.
            EntityNotFound: If ``required`` is True and no match is found.
        """
        # NOTE: Should only return a single result. do we have to use `list` to fetch it?
        results = self.list(group=group, range=range, regex=regex)
        if len(results) > 1:
            raise MultipleEntitiesFound(
                f"Multiple permissions found for group={group!r}, range={range!r}, regex={regex!r}."
            )
        obj = results[0] if results else None
        if required and obj is None:
            raise EntityNotFound(
                f"Permission not found for group={group!r}, range={range!r}, regex={regex!r}."
            )
        return obj


class NetworkPolicyAttributeManager(NamedResourceManager[NetworkPolicyAttribute]):
    """Operations on :class:`~mreg_api.models.NetworkPolicyAttribute` resources."""

    name_lowercase: ClassVar[bool] = True

    @property
    @override
    def model(self) -> type[NetworkPolicyAttribute]:
        return NetworkPolicyAttribute

    def create(
        self,
        *,
        name: str,
        description: str,
        fetch_after_create: bool = True,
    ) -> NetworkPolicyAttribute | None:
        """Create a network policy attribute."""
        return self._create(
            {"name": name, "description": description}, fetch_after_create=fetch_after_create
        )

    def update(
        self,
        ref: int | NetworkPolicyAttribute,
        *,
        description: str | Unset = UNSET,
    ) -> NetworkPolicyAttribute:
        """Update a network policy attribute's mutable fields."""
        attr = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(attr, data)

    def set_description(
        self, ref: int | NetworkPolicyAttribute, description: str
    ) -> NetworkPolicyAttribute:
        """Set the description for the attribute."""
        attr = self._resolve(ref)
        return self.update(attr, description=description)

    def get_policies(self, ref: int | NetworkPolicyAttribute) -> list[NetworkPolicy]:
        """Get all network policies that use this attribute."""
        attr = self._resolve(ref)
        return self._client.get_typed(
            NetworkPolicy.endpoint(), list[NetworkPolicy], params={"attributes": attr.id}
        )


class NetworkPolicyManager(NamedResourceManager[NetworkPolicy]):
    """Operations on :class:`~mreg_api.models.NetworkPolicy` resources."""

    name_lowercase: ClassVar[bool] = True

    @property
    @override
    def model(self) -> type[NetworkPolicy]:
        return NetworkPolicy

    def create(
        self,
        *,
        name: str,
        description: str = "",
        attributes: list[NetworkPolicyAttributeValue] | None = None,
        community_template_pattern: str | None | Unset = UNSET,
        fetch_after_create: bool = True,
    ) -> NetworkPolicy | None:
        """Create a network policy.

        Args:
            name: The policy name (lowercased).
            description: Optional description.
            attributes: Optional list of attribute name/value pairs to attach at creation.
            community_template_pattern: Optional community name template pattern.
            fetch_after_create: Whether to fetch and return the created object.
        """
        data: dict[str, Any] = {"name": name, "description": description}
        if attributes is not None:
            data["attributes"] = [{"name": a.name, "value": a.value} for a in attributes]
        if community_template_pattern is not UNSET:
            data["community_template_pattern"] = community_template_pattern
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | NetworkPolicy,
        *,
        description: str | Unset = UNSET,
        community_template_pattern: str | None | Unset = UNSET,
    ) -> NetworkPolicy:
        """Update a network policy's mutable fields.

        Pass ``community_template_pattern=None`` to unset it.
        """
        pol = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        if community_template_pattern is not UNSET:
            data["community_template_pattern"] = community_template_pattern
        return self._patch(pol, data)

    def set_description(self, ref: int | NetworkPolicy, description: str) -> NetworkPolicy:
        """Set the description for the policy."""
        pol = self._resolve(ref)
        return self.update(pol, description=description)

    def add_attribute(
        self,
        ref: int | NetworkPolicy,
        attr: NetworkPolicyAttribute,
        value: bool = True,
    ) -> NetworkPolicy:
        """Add an attribute to a policy.

        Raises:
            EntityAlreadyExists: If the policy already has this attribute.
        """
        pol = self._resolve(ref)
        if pol.get_attribute(attr.name):
            raise EntityAlreadyExists(f"Policy {pol.name!r} already has attribute {attr.name!r}.")
        attrs = [*pol.attributes, NetworkPolicyAttributeValue(name=attr.name, value=value)]
        return self._patch(pol, {"attributes": [{"name": a.name, "value": a.value} for a in attrs]})

    def remove_attribute(self, ref: int | NetworkPolicy, attribute_name: str) -> NetworkPolicy:
        """Remove an attribute from a policy by name.

        Raises:
            EntityNotFound: If the policy does not have this attribute.
        """
        pol = self._resolve(ref)
        if not pol.get_attribute(attribute_name):
            raise EntityNotFound(f"Policy {pol.name!r} does not have attribute {attribute_name!r}.")
        attrs = [a for a in pol.attributes if a.name != attribute_name]
        return self._patch(pol, {"attributes": [{"name": a.name, "value": a.value} for a in attrs]})

    def networks(self, ref: int | NetworkPolicy) -> list[Network]:
        """Get all networks that use this policy."""
        pol = self._resolve(ref)
        return self._client.get_typed(Network.endpoint(), list[Network], params={"policy": pol.id})


class NetworkManager(WriteResourceManager[Network]):
    """Operations on :class:`~mreg_api.models.Network` resources."""

    _url_identifier: ClassVar[str] = "network"

    @property
    @override
    def model(self) -> type[Network]:
        return Network

    def _resolve_net(self, ref: str | int | Network) -> Network:
        """Resolve a network reference (address string, numeric id, or instance)."""
        if isinstance(ref, Network):
            return ref
        if isinstance(ref, int):
            obj = self._fetch_by_field("id", ref)
        else:
            obj = self.get(ref)
        if obj is None:
            raise EntityNotFound(f"Network {ref!r} not found.")
        return obj

    @overload
    def get_by_ip(self, ip: str | IP_AddressT, *, required: Literal[True]) -> Network: ...
    @overload
    def get_by_ip(self, ip: str | IP_AddressT, *, required: Literal[False] = ...) -> Network | None: ...
    def get_by_ip(self, ip: str | IP_AddressT, *, required: bool = False) -> Network | None:
        """Get the network containing an IP address."""
        addr = str(NetworkOrIP.parse_or_raise(str(ip), mode="ip"))
        resp = self._client.get(Endpoint.NetworksByIP.with_id(addr), ok404=True)
        if not resp:
            if required:
                raise EntityNotFound(f"Network containing IP {addr!r} not found.")
            return None
        return self._validate_json(resp.text)

    def create(
        self,
        *,
        network: str,
        description: str = "",
        vlan: int | None | Unset = UNSET,
        dns_delegated: bool | Unset = UNSET,
        category: str | Unset = UNSET,
        location: str | Unset = UNSET,
        frozen: bool | Unset = UNSET,
        reserved: int | Unset = UNSET,
        fetch_after_create: bool = True,
    ) -> Network | None:
        """Create a network."""
        data: dict[str, Any] = {"network": network, "description": description}
        if vlan is not UNSET:
            data["vlan"] = vlan
        if dns_delegated is not UNSET:
            data["dns_delegated"] = dns_delegated
        if category is not UNSET:
            data["category"] = category
        if location is not UNSET:
            data["location"] = location
        if frozen is not UNSET:
            data["frozen"] = frozen
        if reserved is not UNSET:
            data["reserved"] = reserved
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: str | int | Network,
        *,
        description: str | Unset = UNSET,
        vlan: int | None | Unset = UNSET,
        dns_delegated: bool | Unset = UNSET,
        category: str | Unset = UNSET,
        location: str | Unset = UNSET,
        frozen: bool | Unset = UNSET,
        reserved: int | Unset = UNSET,
        policy: int | None | Unset = UNSET,
        max_communities: int | None | Unset = UNSET,
    ) -> Network:
        """Update a network's mutable fields.

        Pass ``policy=None`` or ``max_communities=None`` to unset; omit to leave unchanged.
        """
        net = self._resolve_net(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        if vlan is not UNSET:
            data["vlan"] = vlan
        if dns_delegated is not UNSET:
            data["dns_delegated"] = dns_delegated
        if category is not UNSET:
            data["category"] = category
        if location is not UNSET:
            data["location"] = location
        if frozen is not UNSET:
            data["frozen"] = frozen
        if reserved is not UNSET:
            data["reserved"] = reserved
        if policy is not UNSET:
            data["policy"] = policy
        if max_communities is not UNSET:
            data["max_communities"] = max_communities
        return self._patch(net, data)

    def get_first_available_ip(self, ref: str | int | Network) -> IP_AddressT:
        """Return the first available IP address in the network."""
        net = self._resolve_net(ref)
        return ipaddress.ip_address(
            self._client.get_typed(Endpoint.NetworksFirstUnused.with_params(net.network), str)
        )

    def get_used_count(self, ref: str | int | Network) -> int:
        """Return the number of used IP addresses in the network."""
        net = self._resolve_net(ref)
        return self._client.get_typed(Endpoint.NetworksUsedCount.with_params(net.network), int)

    def get_unused_count(self, ref: str | int | Network) -> int:
        """Return the number of unused IP addresses in the network."""
        net = self._resolve_net(ref)
        return self._client.get_typed(Endpoint.NetworksUnusedCount.with_params(net.network), int)

    def get_used_list(self, ref: str | int | Network) -> list[IP_AddressT]:
        """Return the used IP addresses in the network."""
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksUsedList.with_params(net.network), list[IP_AddressT]
        )

    def get_unused_list(self, ref: str | int | Network) -> list[IP_AddressT]:
        """Return the unused IP addresses in the network."""
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksUnusedList.with_params(net.network), list[IP_AddressT]
        )

    def get_reserved_ips(self, ref: str | int | Network) -> list[IP_AddressT]:
        """Return the reserved IP addresses of the network."""
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksReservedList.with_params(net.network), list[IP_AddressT]
        )

    def get_used_host_list(self, ref: str | int | Network) -> dict[str, list[str]]:
        """Return a dict of used IP addresses to their associated hostnames."""
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksUsedHostList.with_params(net.network), dict[str, list[str]]
        )

    def get_ptroverride_host_list(self, ref: str | int | Network) -> dict[str, str]:
        """Return a dict of PTR override IPs to their associated hostnames."""
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksPTROverrideHostList.with_params(net.network), dict[str, str]
        )

    def get_ptr_overrides(self, ref: str | int | Network) -> list[IP_AddressT]:
        """Return IP addresses that have PTR overrides in the network."""
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksPTROverrideList.with_params(net.network), list[IP_AddressT]
        )

    def add_excluded_range(self, ref: str | int | Network, start: str, end: str) -> None:
        """Add an excluded IP range to the network."""
        net = self._resolve_net(ref)
        start_ip = NetworkOrIP.parse_or_raise(start, mode="ip")
        end_ip = NetworkOrIP.parse_or_raise(end, mode="ip")
        if start_ip.version != end_ip.version:
            raise InputFailure("Start and end IP addresses must be of the same version.")
        self._client.post(
            Endpoint.NetworksAddExcludedRanges.with_params(net.network),
            json={"network": net.id, "start_ip": str(start_ip), "end_ip": str(end_ip)},
        )

    def remove_excluded_range(self, ref: str | int | Network, start: str, end: str) -> None:
        """Remove an excluded IP range from the network."""
        net = self._resolve_net(ref)
        exrange = next(
            (r for r in net.excluded_ranges if str(r.start_ip) == start and str(r.end_ip) == end),
            None,
        )
        if exrange is None:
            raise EntityNotFound(f"Excluded range {start} - {end} not found in {net.network!r}.")
        self._client.delete(Endpoint.NetworksRemoveExcludedRanges.with_params(net.network, exrange.id))


def resolve_host_id(host: Host | int) -> int:
    """Resolve a host reference to its numeric ID."""
    if isinstance(host, Host):
        return host.id
    return host


class IPAddressManager(WriteResourceManager[IPAddress]):
    """Operations on :class:`~mreg_api.models.IPAddress` resources."""

    @property
    @override
    def model(self) -> type[IPAddress]:
        return IPAddress

    def create(
        self,
        *,
        host: int | Host,
        ipaddress: IP_AddressT | str,
        macaddress: str | MacAddress | None = None,
        fetch_after_create: bool = True,
    ) -> IPAddress | None:
        """Create an IP address record."""
        host_id = resolve_host_id(host)
        data: dict[str, Any] = {"host": host_id, "ipaddress": str(ipaddress)}
        if macaddress is not None:
            data["macaddress"] = str(macaddress)
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | IPAddress,
        *,
        ipaddress: IP_AddressT | str | Unset = UNSET,
        macaddress: str | MacAddress | None | Unset = UNSET,
    ) -> IPAddress:
        """Update an IP address record's mutable fields."""
        ip = self._resolve(ref)
        data: dict[str, Any] = {}
        if ipaddress is not UNSET:
            data["ipaddress"] = str(ipaddress)
        if macaddress is not UNSET:
            data["macaddress"] = str(macaddress) if macaddress is not None else ""
        return self._patch(ip, data)

    def associate_mac(
        self, ref: int | IPAddress, mac: str | MacAddress, *, force: bool = False
    ) -> IPAddress:
        """Associate a MAC address with an IP address.

        Raises:
            EntityAlreadyExists: If the IP already has a MAC and ``force`` is False.
        """
        ip = self._resolve(ref)
        if ip.macaddress and not force:
            raise EntityAlreadyExists(
                f"IP address {ip.ipaddress} already has MAC address {ip.macaddress}."
            )
        return self.update(ip, macaddress=mac)

    def disassociate_mac(self, ref: int | IPAddress) -> IPAddress:
        """Remove the MAC address from an IP address."""
        ip = self._resolve(ref)
        return self.update(ip, macaddress=None)

    def list_by_host(self, host: int | Host) -> list[IPAddress]:
        """List all IP address records for a host."""
        host_id = resolve_host_id(host)
        return self._fetch_list_by_field("host", host_id)

    def list_by_ip(self, ip: IP_AddressT | str) -> list[IPAddress]:
        """List all IP address records with a given IP address."""
        return self._fetch_list_by_field("ipaddress", str(ip))


class CNAMEManager(WriteResourceManager[CNAME]):
    """Operations on :class:`~mreg_api.models.CNAME` resources."""

    @property
    @override
    def model(self) -> type[CNAME]:
        return CNAME

    def _normalize_name(self, name: str) -> str:
        return self._client.fqdn(name)

    def create(
        self,
        *,
        host: int | Host,
        name: str | HostName,
        fetch_after_create: bool = True,
    ) -> CNAME | None:
        """Create a CNAME record."""
        host_id = resolve_host_id(host)
        return self._create(
            {"host": str(host_id), "name": self._client.fqdn(name)},
            fetch_after_create=fetch_after_create,
        )

    def update(
        self,
        ref: int | CNAME,
        *,
        host: int | Host | Unset = UNSET,
        name: str | HostName | Unset = UNSET,
    ) -> CNAME:
        """Update a CNAME record's mutable fields."""
        cname = self._resolve(ref)
        data: dict[str, Any] = {}
        if host is not UNSET:
            data["host"] = host.id if isinstance(host, Host) else host
        if name is not UNSET:
            data["name"] = self._client.fqdn(str(name))
        return self._patch(cname, data)

    @overload
    def get_by_name(self, name: str, *, required: Literal[True]) -> CNAME: ...
    @overload
    def get_by_name(self, name: str, *, required: Literal[False] = ...) -> CNAME | None: ...
    def get_by_name(self, name: str, *, required: bool = False) -> CNAME | None:
        """Get a CNAME record by alias name."""
        obj = self._fetch_by_field("name", self._client.fqdn(name))
        if required and obj is None:
            raise EntityNotFound(f"CNAME {name!r} not found.")
        return obj

    def list_by_host(self, host: int | Host) -> list[CNAME]:
        """List all CNAME records for a host."""
        host_id = resolve_host_id(host)
        return self._fetch_list_by_field("host", host_id)


class HInfoManager(WriteResourceManager[HInfo]):
    """Operations on :class:`~mreg_api.models.HInfo` resources.

    HInfo is a 1-per-host record; the URL identifier is the host ID (not a numeric row id).
    """

    _url_identifier: ClassVar[str] = "host"

    @property
    @override
    def model(self) -> type[HInfo]:
        return HInfo

    def create(
        self,
        *,
        host: int | Host,
        cpu: str,
        os: str,
        fetch_after_create: bool = True,
    ) -> HInfo | None:
        """Create an HInfo record."""
        host_id = resolve_host_id(host)
        return self._create(
            {"host": host_id, "cpu": cpu, "os": os}, fetch_after_create=fetch_after_create
        )

    def update(
        self,
        ref: int | HInfo,
        *,
        cpu: str | Unset = UNSET,
        os: str | Unset = UNSET,
    ) -> HInfo:
        """Update an HInfo record's mutable fields."""
        hinfo = self._resolve(ref)
        data: dict[str, Any] = {}
        if cpu is not UNSET:
            data["cpu"] = cpu
        if os is not UNSET:
            data["os"] = os
        return self._patch(hinfo, data)

    def get_by_host(self, host: int | Host, *, required: bool = False) -> HInfo | None:
        """Get the HInfo record for a host."""
        host_id = resolve_host_id(host)
        obj = self._fetch_by_field("host", host_id)
        if required and obj is None:
            raise EntityNotFound(f"HInfo for host id {host_id!r} not found.")
        return obj


class TXTManager(WriteResourceManager[TXT]):
    """Operations on :class:`~mreg_api.models.TXT` resources."""

    @property
    @override
    def model(self) -> type[TXT]:
        return TXT

    def create(
        self,
        *,
        host: int | Host,
        txt: str,
        fetch_after_create: bool = True,
    ) -> TXT | None:
        """Create a TXT record."""
        host_id = resolve_host_id(host)
        return self._create({"host": host_id, "txt": txt}, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | TXT,
        *,
        txt: str | Unset = UNSET,
    ) -> TXT:
        """Update a TXT record's mutable fields."""
        txt_obj = self._resolve(ref)
        data: dict[str, Any] = {}
        if txt is not UNSET:
            data["txt"] = txt
        return self._patch(txt_obj, data)

    def list_by_host(self, host: int | Host) -> list[TXT]:
        """List all TXT records for a host."""
        host_id = resolve_host_id(host)
        return self._fetch_list_by_field("host", host_id)


class MXManager(WriteResourceManager[MX]):
    """Operations on :class:`~mreg_api.models.MX` resources."""

    @property
    @override
    def model(self) -> type[MX]:
        return MX

    def create(
        self,
        *,
        host: int | Host,
        mx: str,
        priority: int,
        fetch_after_create: bool = True,
    ) -> MX | None:
        """Create an MX record."""
        host_id = resolve_host_id(host)
        return self._create(
            {"host": host_id, "mx": mx, "priority": priority},
            fetch_after_create=fetch_after_create,
        )

    def update(
        self,
        ref: int | MX,
        *,
        mx: str | Unset = UNSET,
        priority: int | Unset = UNSET,
    ) -> MX:
        """Update an MX record's mutable fields."""
        mx_obj = self._resolve(ref)
        data: dict[str, Any] = {}
        if mx is not UNSET:
            data["mx"] = mx
        if priority is not UNSET:
            data["priority"] = priority
        return self._patch(mx_obj, data)

    def list_by_host(self, host: int | Host) -> list[MX]:
        """List all MX records for a host."""
        host_id = resolve_host_id(host)
        return self._fetch_list_by_field("host", host_id)


class NAPTRManager(WriteResourceManager[NAPTR]):
    """Operations on :class:`~mreg_api.models.NAPTR` resources."""

    @property
    @override
    def model(self) -> type[NAPTR]:
        return NAPTR

    def create(
        self,
        *,
        host: int | Host,
        preference: int,
        order: int,
        flag: str = "",
        service: str = "",
        regex: str = "",
        replacement: str,
        fetch_after_create: bool = True,
    ) -> NAPTR | None:
        """Create a NAPTR record."""
        host_id = resolve_host_id(host)
        return self._create(
            {
                "host": host_id,
                "preference": preference,
                "order": order,
                "flag": flag,
                "service": service,
                "regex": regex,
                "replacement": replacement,
            },
            fetch_after_create=fetch_after_create,
        )

    def update(
        self,
        ref: int | NAPTR,
        *,
        preference: int | Unset = UNSET,
        order: int | Unset = UNSET,
        flag: str | Unset = UNSET,
        service: str | Unset = UNSET,
        regex: str | Unset = UNSET,
        replacement: str | Unset = UNSET,
    ) -> NAPTR:
        """Update a NAPTR record's mutable fields."""
        naptr = self._resolve(ref)
        data: dict[str, Any] = {}
        if preference is not UNSET:
            data["preference"] = preference
        if order is not UNSET:
            data["order"] = order
        if flag is not UNSET:
            data["flag"] = flag
        if service is not UNSET:
            data["service"] = service
        if regex is not UNSET:
            data["regex"] = regex
        if replacement is not UNSET:
            data["replacement"] = replacement
        return self._patch(naptr, data)

    def list_by_host(self, host: int | Host) -> list[NAPTR]:
        """List all NAPTR records for a host."""
        host_id = resolve_host_id(host)
        return self._fetch_list_by_field("host", host_id)


class SrvManager(WriteResourceManager[Srv]):
    """Operations on :class:`~mreg_api.models.Srv` resources."""

    @property
    @override
    def model(self) -> type[Srv]:
        return Srv

    def create(
        self,
        *,
        host: int | Host,
        name: str,
        priority: int,
        weight: int,
        port: int,
        ttl: int | None | Unset = UNSET,
        fetch_after_create: bool = True,
    ) -> Srv | None:
        """Create a SRV record."""
        host_id = resolve_host_id(host)
        data: dict[str, Any] = {
            "host": host_id,
            "name": name,
            "priority": priority,
            "weight": weight,
            "port": port,
        }
        if ttl is not UNSET:
            data["ttl"] = ttl
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | Srv,
        *,
        name: str | Unset = UNSET,
        priority: int | Unset = UNSET,
        weight: int | Unset = UNSET,
        port: int | Unset = UNSET,
        ttl: int | None | Unset = UNSET,
    ) -> Srv:
        """Update a SRV record's mutable fields. Pass ``ttl=None`` to reset to default."""
        srv = self._resolve(ref)
        data: dict[str, Any] = {}
        if name is not UNSET:
            data["name"] = name
        if priority is not UNSET:
            data["priority"] = priority
        if weight is not UNSET:
            data["weight"] = weight
        if port is not UNSET:
            data["port"] = port
        if ttl is not UNSET:
            data["ttl"] = ttl
        return self._patch(srv, data)

    def list_by_host(self, host: int | Host) -> list[Srv]:
        """List all SRV records for a host."""
        host_id = resolve_host_id(host)
        return self._fetch_list_by_field("host", host_id)


class PTROverrideManager(WriteResourceManager[PTR_override]):
    """Operations on :class:`~mreg_api.models.PTR_override` resources."""

    @property
    @override
    def model(self) -> type[PTR_override]:
        return PTR_override

    def create(
        self,
        *,
        host: int | Host,
        ipaddress: IP_AddressT | str,
        fetch_after_create: bool = True,
    ) -> PTR_override | None:
        """Create a PTR override record."""
        host_id = resolve_host_id(host)
        return self._create(
            {"host": host_id, "ipaddress": str(ipaddress)},
            fetch_after_create=fetch_after_create,
        )

    def update(
        self,
        ref: int | PTR_override,
        *,
        host: int | Host | Unset = UNSET,
        ipaddress: IP_AddressT | str | Unset = UNSET,
    ) -> PTR_override:
        """Update a PTR override record's mutable fields."""
        ptr = self._resolve(ref)
        data: dict[str, Any] = {}
        if host is not UNSET:
            data["host"] = host.id if isinstance(host, Host) else host
        if ipaddress is not UNSET:
            data["ipaddress"] = str(ipaddress)
        return self._patch(ptr, data)

    def list_by_host(self, host: int | Host) -> list[PTR_override]:
        """List all PTR override records for a host."""
        host_id = resolve_host_id(host)
        return self._fetch_list_by_field("host", host_id)


class SSHFPManager(WriteResourceManager[SSHFP]):
    """Operations on :class:`~mreg_api.models.SSHFP` resources."""

    @property
    @override
    def model(self) -> type[SSHFP]:
        return SSHFP

    def create(
        self,
        *,
        host: int | Host,
        algorithm: int,
        hash_type: int,
        fingerprint: str,
        ttl: int | None | Unset = UNSET,
        fetch_after_create: bool = True,
    ) -> SSHFP | None:
        """Create an SSHFP record."""
        host_id = resolve_host_id(host)
        data: dict[str, Any] = {
            "host": host_id,
            "algorithm": algorithm,
            "hash_type": hash_type,
            "fingerprint": fingerprint,
        }
        if ttl is not UNSET:
            data["ttl"] = ttl
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | SSHFP,
        *,
        algorithm: int | Unset = UNSET,
        hash_type: int | Unset = UNSET,
        fingerprint: str | Unset = UNSET,
        ttl: int | None | Unset = UNSET,
    ) -> SSHFP:
        """Update an SSHFP record's mutable fields. Pass ``ttl=None`` to reset to default."""
        sshfp = self._resolve(ref)
        data: dict[str, Any] = {}
        if algorithm is not UNSET:
            data["algorithm"] = algorithm
        if hash_type is not UNSET:
            data["hash_type"] = hash_type
        if fingerprint is not UNSET:
            data["fingerprint"] = fingerprint
        if ttl is not UNSET:
            data["ttl"] = ttl
        return self._patch(sshfp, data)

    def list_by_host(self, host: int | Host) -> list[SSHFP]:
        """List all SSHFP records for a host."""
        host_id = resolve_host_id(host)
        return self._fetch_list_by_field("host", host_id)


class BacnetIDManager(WriteResourceManager[BacnetID]):
    """Operations on :class:`~mreg_api.models.BacnetID` resources."""

    @property
    @override
    def model(self) -> type[BacnetID]:
        return BacnetID

    def create(
        self,
        *,
        hostname: str | HostName,
        id: int,  # noqa: A002
        fetch_after_create: bool = True,
    ) -> BacnetID | None:
        """Create a BacnetID record.

        Args:
            hostname: The FQDN of the host (used directly by the API, not a host id).
            id: The BACnet device instance number (0–4194302).
            fetch_after_create: Whether to fetch and return the created object.
        """
        return self._create(
            {"hostname": str(hostname), "id": id},
            fetch_after_create=fetch_after_create,
        )

    def update(
        self,
        ref: int | BacnetID,
        *,
        hostname: str | HostName | Unset = UNSET,
    ) -> BacnetID:
        """Update a BacnetID record's mutable fields."""
        bacnet = self._resolve(ref)
        data: dict[str, Any] = {}
        if hostname is not UNSET:
            data["hostname"] = str(hostname)
        return self._patch(bacnet, data)

    def list_in_range(self, start: int, end: int) -> list[BacnetID]:
        """List BacnetID records within a numeric id range (inclusive)."""
        return self._client.get_typed(
            self._endpoint(), list[BacnetID], params={"id__range": f"{start},{end}"}
        )


class LocationManager(WriteResourceManager[Location]):
    """Operations on :class:`~mreg_api.models.Location` resources."""

    _url_identifier: ClassVar[str] = "host"

    @property
    @override
    def model(self) -> type[Location]:
        return Location

    def create(
        self,
        *,
        host: int | Host,
        loc: str,
        fetch_after_create: bool = True,
    ) -> Location | None:
        """Create a LOC record."""
        host_id = resolve_host_id(host)
        return self._create({"host": host_id, "loc": loc}, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | Location,
        *,
        loc: str | Unset = UNSET,
    ) -> Location:
        """Update a LOC record's mutable fields."""
        loc_obj = self._resolve(ref)
        data: dict[str, Any] = {}
        if loc is not UNSET:
            data["loc"] = loc
        return self._patch(loc_obj, data)

    def get_by_host(self, host: int | Host, *, required: bool = False) -> Location | None:
        """Get the LOC record for a host."""
        host_id = resolve_host_id(host)
        obj = self._fetch_by_field("host", host_id)
        if required and obj is None:
            raise EntityNotFound(f"Location for host id {host_id!r} not found.")
        return obj
