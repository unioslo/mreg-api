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

import functools
import ipaddress
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import Literal
from typing import TypeVar
from typing import overload

from typing_extensions import override

from mreg_api.endpoints import Endpoint
from mreg_api.events import Event
from mreg_api.events import EventKind
from mreg_api.events import ObjectRef
from mreg_api.exceptions import DeleteError
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import EntityOwnershipMismatch
from mreg_api.exceptions import ForceMissing
from mreg_api.exceptions import GetError
from mreg_api.exceptions import InputFailure
from mreg_api.exceptions import InternalError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.exceptions import UnexpectedDataError
from mreg_api.models import CNAME
from mreg_api.models import MX
from mreg_api.models import NAPTR
from mreg_api.models import SSHFP
from mreg_api.models import TXT
from mreg_api.models import Atom
from mreg_api.models import BacnetID
from mreg_api.models import Community
from mreg_api.models import DhcpHostIPv4
from mreg_api.models import DhcpHostIPv6
from mreg_api.models import DhcpHostIPv6ByIPv4
from mreg_api.models import ForwardZone
from mreg_api.models import ForwardZoneDelegation
from mreg_api.models import HInfo
from mreg_api.models import Host
from mreg_api.models import HostContactModification
from mreg_api.models import HostGroup
from mreg_api.models import IPAddress
from mreg_api.models import Label
from mreg_api.models import Location
from mreg_api.models import NameServer
from mreg_api.models import Network
from mreg_api.models import NetworkOrIP
from mreg_api.models import NetworkPolicy
from mreg_api.models import NetworkPolicyAttribute
from mreg_api.models import NetworkPolicyAttributeValue
from mreg_api.models import Permission
from mreg_api.models import PTR_override
from mreg_api.models import ReverseZone
from mreg_api.models import ReverseZoneDelegation
from mreg_api.models import Role
from mreg_api.models import Srv
from mreg_api.models import Zone
from mreg_api.models import ZoneFile
from mreg_api.models.abstracts import MregModel
from mreg_api.models.fields import HostName
from mreg_api.models.fields import MacAddress
from mreg_api.models.history import HistoryItem
from mreg_api.models.history import HistoryResource
from mreg_api.models.models import is_reverse_zone_name
from mreg_api.types import IP_AddressT
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter
from mreg_api.utilities.shared import convert_wildcard_to_regex

if TYPE_CHECKING:
    from mreg_api.client import MregClient
    from mreg_api.types import IP_NetworkT


class Unset:
    """Sentinel for "unchanged" parameters in update methods."""

    def __repr__(self) -> str:
        """Return a string representation of the Unset sentinel."""
        return "UNSET"


UNSET = Unset()


T = TypeVar("T", bound=MregModel)


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

    @property
    @abstractmethod
    def endpoint(self) -> Endpoint:
        """Return the API endpoint for this manager's resource type."""
        ...

    def _resolve(self, ref: int | T) -> T:
        """Resolve an object reference (instance or numeric id) to an instance of the model.

        Fetches the object from the server if only an ID is provided.

        Raises:
            EntityNotFound: If the object cannot be found by its ID.
        """
        if isinstance(ref, self.model):
            return ref

        # TODO: use url identifier to resolve?
        # I.e. `_resolve("testhost.example.com")` would use `_url_identifier`
        # in the _fetch_by_field call. Requires us to support str args, as well
        # as ensuring all other methods that call this method are calling it
        # with the primary identifier in mind. Should potentially also
        # always coerce `ref` arg to `str` here.
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
        return self.endpoint.with_id(self.id_field_value(obj))

    def _fetch_by_field(self, field: str, value: str | int) -> T | None:
        """Fetch a single object, querying by a field."""
        endpoint = self.endpoint

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
        return self._client.get_typed(self.endpoint, list[self.model], params=params, limit=500)

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
        return self._client.get_typed(self.endpoint, list[self.model], params=params, limit=limit)

    # TODO: add warning or similar when used on non-paginated endpoints somehow.
    # Manually? Use contextvar? Who knows.
    @overload
    def first(self, *, required: Literal[True], **query: str | int | float | bool | None) -> T: ...
    @overload
    def first(
        self, *, required: Literal[False] = ..., **query: str | int | float | bool | None
    ) -> T | None: ...
    def first(self, *, required: bool = False, **query: str | int | float | bool | None) -> T | None:
        """Return the first resource, or ``None`` when ``required`` is False.

        Passes ``page_size=1`` to avoid over-fetching.

        Suboptimal behavior on non-paginated endpoints, i.e.
        `networks/{network}/unused_list`, `/dhcphosts`, etc.

        Args:
            required: When ``True``, raise if no resource exists (returns ``T``).
            **query: Optional filter parameters forwarded to the list endpoint.

        Raises:
            EntityNotFound: If ``required`` is True and no resource is found.
        """
        params: QueryParams = {**dict(query), "page_size": 1}
        results = self._client.get_typed(self.endpoint, list[self.model], params=params, limit=None)
        obj = results[0] if results else None
        if required and obj is None:
            raise EntityNotFound(f"No {self.model.__name__} found.")
        return obj


class WriteResourceManager(ResourceManager[T], ABC):
    """Manager for performing CRUD operations on an API resource type."""

    def _create(self, data: dict[str, Any], *, fetch_after_create: bool = True) -> T | None:
        """POST ``data`` to the resource endpoint, optionally fetching the result.

        Raises :class:`PostError` (from the client) if the server rejects the create.
        Returns ``None`` when the create succeeds but the server provides no
        ``Location`` header to refetch from (many endpoints don't), or when
        ``fetch_after_create`` is ``False``.
        """
        response = self._client.post(self.endpoint, json=data)
        if fetch_after_create and "Location" in response.headers:
            return self._client.get_typed(response.headers["Location"], self.model)
        return None

    def _patch(self, obj: T, data: dict[str, Any], *, params: QueryParams | None = None) -> T:
        """PATCH ``obj`` with ``data`` and return the refetched object.

        Raises :class:`PatchError` (from the client) if the server rejects the patch.
        """
        _ = self._client.patch(self._endpoint_with_id(obj), json=data, params=params)
        return self._refetch(obj)

    def delete(self, obj: T) -> None:
        """Delete a resource.

        Args:
            obj (T): The resource to delete.
        """
        _ = self._client.delete(self._endpoint_with_id(obj))


class CountableResourceManager(WriteResourceManager[T], ABC):
    """Opt-in capability mixin: adds ``count`` to a ``WriteResourceManager``.

    Inherit this alongside (or instead of) ``WriteResourceManager`` for any resource
    whose list endpoint returns a DRF-paginated response with a ``count`` field.
    DhcpHost managers must NOT inherit this (their endpoints are non-paginated).

    Combine with other capabilities via multiple inheritance, the same way
    ``HostManager`` combines ``NamedResourceManager`` + ``HistoryManager``.
    """

    def count(self) -> int:
        """Return the server-reported total count of resources at this endpoint."""
        return self._client.get_count(self.endpoint)


class NamedResourceManager(WriteResourceManager[T], ABC):
    """Extended WriteResourceManager for resources that support name-based lookups."""

    # NOTE: write methods are inherited even for managers where the underlying
    # API endpoint may not support writes (e.g. NameServerManager).

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
        return MregClient().get_typed(self.endpoint, list[self.model], params={param: value})


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

    # TODO: refactor to also accept a resource instance
    #       HOWEVER, we _must_ also accept names, because
    #       history may refer to deleted items, thus no concrete instance!
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
    def endpoint(self) -> Endpoint:
        return Endpoint.Hosts

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

    def add_contacts(self, ref: int | Host, contacts: list[str]) -> HostContactModification:
        """Add contacts to a host (atomic; POST to /hosts/{name}/contacts/)."""
        host = self._resolve(ref)
        resp = self._client.post(
            Endpoint.HostsContacts.with_params(host.name), json={"emails": contacts}
        )
        return get_type_adapter(HostContactModification).validate_json(resp.text)

    def clear_contacts(self, ref: int | Host) -> HostContactModification:
        """Remove all contacts from a host (atomic; DELETE /hosts/{name}/contacts/)."""
        host = self._resolve(ref)
        resp = self._client.delete(Endpoint.HostsContacts.with_params(host.name))
        return get_type_adapter(HostContactModification).validate_json(resp.text)

    def remove_contacts(self, ref: int | Host, contacts: list[str]) -> HostContactModification:
        """Remove specific contacts from a host (atomic; DELETE /hosts/{name}/contacts/)."""
        host = self._resolve(ref)
        resp = self._client.delete(
            Endpoint.HostsContacts.with_params(host.name), json={"emails": contacts}
        )
        return get_type_adapter(HostContactModification).validate_json(resp.text)


class HostGroupManager(NamedResourceManager[HostGroup], HistoryManager[HostGroup]):
    """Operations on :class:`~mreg_api.models.HostGroup` resources."""

    _url_identifier: ClassVar[str] = "name"

    @property
    @override
    def model(self) -> type[HostGroup]:
        return HostGroup

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.HostGroups

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

    def list_by_host(self, host: int | Host, *, traverse: bool = False) -> list[HostGroup]:
        """List all hostgroups that include the given host.

        Args:
            host: Host instance or numeric host ID.
            traverse: If True, also include all parent groups recursively.
        """
        host_id = host.id if isinstance(host, Host) else host
        direct = self._fetch_list_by_field("hosts", host_id)
        if not traverse:
            return sorted(direct, key=lambda g: g.name)
        groups: list[HostGroup] = list(direct)
        for group in direct:
            groups.extend(self.list_parents(group))
        return sorted(groups, key=lambda g: g.name)


class LabelManager(NamedResourceManager[Label]):
    """Operations on :class:`~mreg_api.models.Label` resources."""

    @property
    @override
    def model(self) -> type[Label]:
        return Label

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Labels

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


class RoleManager(NamedResourceManager[Role], HistoryManager[Role]):
    """Operations on :class:`~mreg_api.models.Role` resources."""

    _url_identifier: ClassVar[str] = "name"

    @property
    @override
    def model(self) -> type[Role]:
        return Role

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.HostPolicyRoles

    @property
    @override
    def history_resource(self) -> HistoryResource:
        return HistoryResource.HostPolicy_Role

    def create(
        self,
        *,
        name: str,
        description: str = "",
        fetch_after_create: bool = True,
    ) -> Role | None:
        """Create a role."""
        return self._create(
            {"name": name, "description": description}, fetch_after_create=fetch_after_create
        )

    def update(
        self,
        ref: int | Role,
        *,
        description: str | Unset = UNSET,
    ) -> Role:
        """Update a role's mutable fields."""
        role = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(role, data)

    def set_description(self, ref: int | Role, description: str) -> Role:
        """Set the description for the role."""
        role = self._resolve(ref)
        return self.update(role, description=description)

    @override
    def delete(self, obj: Role) -> None:
        """Delete a role.

        Raises:
            DeleteError: If the role is still in use on any hosts.
        """
        if obj.hosts:
            hosts = ", ".join(obj.hosts)
            raise DeleteError(f"Role {obj.name!r} used on hosts: {hosts}")
        super().delete(obj)

    def list_with_atom(self, atom_name: str) -> list[Role]:
        """List all roles that contain a given atom.

        Renamed from ``Role.get_roles_with_atom``.
        """
        return self._fetch_list_by_field("atoms__name__exact", atom_name)

    def add_atom(self, ref: int | Role, atom_name: str) -> Role:
        """Add an atom to the role.

        Raises:
            EntityNotFound: If the atom does not exist.
            EntityAlreadyExists: If the atom is already a member of the role.
        """
        role = self._resolve(ref)
        _ = AtomManager(self._client).get_by_name(atom_name, required=True)
        if atom_name in role.atoms:
            raise EntityAlreadyExists(f"Atom {atom_name!r} already a member of role {role.name!r}")
        self._client.post(
            Endpoint.HostPolicyRolesAddAtom.with_params(role.name), json={"name": atom_name}
        )
        return self._refetch(role)

    def remove_atom(self, ref: int | Role, atom_name: str) -> Role:
        """Remove an atom from the role.

        Raises:
            EntityOwnershipMismatch: If the atom is not a member of the role.
        """
        role = self._resolve(ref)
        if atom_name not in role.atoms:
            raise EntityOwnershipMismatch(f"Atom {atom_name!r} not a member of {role.name!r}")
        self._client.delete(Endpoint.HostPolicyRolesRemoveAtom.with_params(role.name, atom_name))
        return self._refetch(role)

    def add_host(self, ref: int | Role, name: str) -> Role:
        """Add a host to the role by name."""
        role = self._resolve(ref)
        name = self._client.fqdn(name)
        self._client.post(Endpoint.HostPolicyRolesAddHost.with_params(role.name), json={"name": name})
        return self._refetch(role)

    def remove_host(self, ref: int | Role, name: str) -> Role:
        """Remove a host from the role by name."""
        role = self._resolve(ref)
        name = self._client.fqdn(name)
        self._client.delete(Endpoint.HostPolicyRolesRemoveHost.with_params(role.name, name))
        return self._refetch(role)

    def get_labels(self, ref: int | Role) -> list[Label]:
        """Get the labels associated with the role."""
        role = self._resolve(ref)
        labels = LabelManager(self._client)
        return [labels.get(lid, required=True) for lid in role.labels]

    def add_label(self, ref: int | Role, label_name: str) -> Role:
        """Add a label to the role by name.

        Raises:
            EntityNotFound: If the label does not exist.
            EntityAlreadyExists: If the role already has the label.
        """
        role = self._resolve(ref)
        label = LabelManager(self._client).get_by_name(label_name, required=True)
        if label.id in role.labels:
            raise EntityAlreadyExists(f"Role {role.name!r} already has label {label_name!r}")
        return self._patch(role, {"labels": [*role.labels, label.id]})

    def remove_label(self, ref: int | Role, label_name: str) -> Role:
        """Remove a label from the role by name.

        Raises:
            EntityNotFound: If the label does not exist.
            EntityOwnershipMismatch: If the role does not have the label.
        """
        role = self._resolve(ref)
        label = LabelManager(self._client).get_by_name(label_name, required=True)
        if label.id not in role.labels:
            raise EntityOwnershipMismatch(f"Role {role.name!r} does not have label {label_name!r}")
        return self._patch(role, {"labels": [lid for lid in role.labels if lid != label.id]})

    def list_by_host(self, host: int | Host) -> list[Role]:
        """List all roles that include the given host."""
        host_id = host.id if isinstance(host, Host) else host
        return self._fetch_list_by_field("hosts", host_id)


class AtomManager(NamedResourceManager[Atom], HistoryManager[Atom]):
    """Operations on :class:`~mreg_api.models.Atom` resources."""

    _url_identifier: ClassVar[str] = "name"

    @property
    @override
    def model(self) -> type[Atom]:
        return Atom

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.HostPolicyAtoms

    @property
    @override
    def history_resource(self) -> HistoryResource:
        return HistoryResource.HostPolicy_Atom

    def create(
        self,
        *,
        name: str,
        description: str = "",
        fetch_after_create: bool = True,
    ) -> Atom | None:
        """Create an atom."""
        return self._create(
            {"name": name, "description": description}, fetch_after_create=fetch_after_create
        )

    def update(
        self,
        ref: int | Atom,
        *,
        description: str | Unset = UNSET,
    ) -> Atom:
        """Update an atom's mutable fields."""
        atom = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(atom, data)

    def set_description(self, ref: int | Atom, description: str) -> Atom:
        """Set the description for the atom."""
        atom = self._resolve(ref)
        return self.update(atom, description=description)

    @override
    def delete(self, obj: Atom) -> None:
        """Delete an atom.

        Raises:
            DeleteError: If the atom is still used in any roles.
        """
        if obj.roles:
            roles = ", ".join(obj.roles)
            raise DeleteError(f"Atom {obj.name!r} used in roles: {roles}")
        super().delete(obj)


class PermissionManager(WriteResourceManager[Permission]):
    """Operations on :class:`~mreg_api.models.Permission` resources."""

    @property
    @override
    def model(self) -> type[Permission]:
        return Permission

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.PermissionNetgroupRegex

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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.NetworkPolicyAttributes

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
            Endpoint.NetworkPolicies, list[NetworkPolicy], params={"attributes": attr.id}
        )


class NetworkPolicyManager(NamedResourceManager[NetworkPolicy]):
    """Operations on :class:`~mreg_api.models.NetworkPolicy` resources."""

    name_lowercase: ClassVar[bool] = True

    @property
    @override
    def model(self) -> type[NetworkPolicy]:
        return NetworkPolicy

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.NetworkPolicies

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
        return self._client.get_typed(Endpoint.Networks, list[Network], params={"policy": pol.id})

    @functools.cached_property
    def attribute(self) -> NetworkPolicyAttributeManager:
        """Manager for network policy attributes (``client.networks.policy.attribute``)."""
        return NetworkPolicyAttributeManager(self._client)


class CommunityManager:
    """Operations on network communities (``client.networks.communities``).

    Communities are always scoped to a network — every method takes a network
    reference (address string or :class:`~mreg_api.models.Network` instance).
    Exposed as ``client.networks.communities`` via :class:`NetworkManager`.
    """

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client."""
        self._client = client

    @staticmethod
    def _net_addr(ref: str | Network) -> str:
        """Return the network address string from a string or Network instance."""
        return ref.network if isinstance(ref, Network) else ref

    def list(self, network: str | Network) -> list[Community]:
        """List all communities for a network."""
        addr = self._net_addr(network)
        return self._client.get_typed(Endpoint.NetworkCommunities.with_params(addr), list[Community])

    def get_by_name(
        self, network: str | Network, name: str, *, required: bool = False
    ) -> Community | None:
        """Get a community by name within a network."""
        community = next((c for c in self.list(network) if c.name == name), None)
        if required and community is None:
            raise EntityNotFound(f"Community {name!r} not found in network {network!r}.")
        return community

    def create(self, network: str | Network, *, name: str, description: str) -> bool:
        """Create a community in a network."""
        addr = self._net_addr(network)
        resp = self._client.post(
            Endpoint.NetworkCommunities.with_params(addr),
            json={"name": name, "description": description},
        )
        return resp.is_success if resp else False

    def delete(self, network: str | Network, community: int | Community) -> None:
        """Delete a community from a network."""
        addr = self._net_addr(network)
        community_id = community.id if isinstance(community, Community) else community
        self._client.delete(Endpoint.NetworkCommunity.with_params(addr, community_id))

    def get_hosts(self, network: str | Network, community: int | Community) -> list[Host]:
        """List all hosts in a community."""
        addr = self._net_addr(network)
        community_id = community.id if isinstance(community, Community) else community
        return self._client.get_typed(
            Endpoint.NetworkCommunityHosts.with_params(addr, community_id), list[Host]
        )

    def add_host(
        self,
        network: str | Network,
        community: int | Community,
        host: int | Host,
        *,
        ipaddress: IP_AddressT | str | None = None,
    ) -> bool:
        """Add a host to a community."""
        addr = self._net_addr(network)
        community_id = community.id if isinstance(community, Community) else community
        host_id = host.id if isinstance(host, Host) else host
        data: dict[str, Any] = {"id": host_id}
        if ipaddress is not None:
            data["ipaddress"] = str(ipaddress)
        resp = self._client.post(
            Endpoint.NetworkCommunityHosts.with_params(addr, community_id), json=data
        )
        return resp.is_success if resp else False

    def remove_host(
        self,
        network: str | Network,
        community: int | Community,
        host: int | Host,
    ) -> None:
        """Remove a host from a community."""
        addr = self._net_addr(network)
        community_id = community.id if isinstance(community, Community) else community
        host_id = host.id if isinstance(host, Host) else host
        self._client.delete(Endpoint.NetworkCommunityHost.with_params(addr, community_id, host_id))


class NetworkManager(WriteResourceManager[Network]):
    """Operations on :class:`~mreg_api.models.Network` resources."""

    _url_identifier: ClassVar[str] = "network"

    @property
    @override
    def model(self) -> type[Network]:
        return Network

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Networks

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

    @functools.cached_property
    def policy(self) -> NetworkPolicyManager:
        """Manager for network policies (``client.networks.policy``)."""
        return NetworkPolicyManager(self._client)

    @functools.cached_property
    def communities(self) -> CommunityManager:
        """Manager for network communities (``client.networks.communities``)."""
        return CommunityManager(self._client)


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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Ipaddresses

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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Cnames

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
        ttl: int | None | Unset = UNSET,
    ) -> CNAME:
        """Update a CNAME record's mutable fields. Pass ``ttl=None`` to reset to default."""
        cname = self._resolve(ref)
        data: dict[str, Any] = {}
        if not isinstance(host, Unset):
            data["host"] = resolve_host_id(host)
        if name is not UNSET:
            data["name"] = self._client.fqdn(str(name))
        if ttl is not UNSET:
            data["ttl"] = ttl
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

    def get_by_host_and_name(
        self, host: int | Host, name: str, *, required: bool = False
    ) -> CNAME | None:
        """Get a CNAME record matching both the host and alias name."""
        host_id = resolve_host_id(host)
        fqdn = self._client.fqdn(name)
        cnamas = self._fetch_list_by_field("host", host_id)
        obj = next((c for c in cnamas if c.name == fqdn), None)
        if required and obj is None:
            raise EntityNotFound(f"CNAME {name!r} for host {host_id} not found.")
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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Hinfos

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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Txts

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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Mxs

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

    def get_by_all(self, host: int | Host, mx: str, priority: int) -> MX:
        """Get an MX record matching host, mx value, and priority.

        Raises:
            EntityNotFound: If no matching MX record exists.
        """
        host_id = resolve_host_id(host)
        obj = self._client.get_list_unique(
            self.endpoint, params={"host": str(host_id), "mx": mx, "priority": str(priority)}
        )
        if not obj:
            raise EntityNotFound(f"MX {mx!r} with priority {priority} not found for host {host_id}.")
        return MX.model_validate(obj)


class NAPTRManager(WriteResourceManager[NAPTR]):
    """Operations on :class:`~mreg_api.models.NAPTR` resources."""

    @property
    @override
    def model(self) -> type[NAPTR]:
        return NAPTR

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Naptrs

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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Srvs

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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.PTR_overrides

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
        if not isinstance(host, Unset):
            data["host"] = resolve_host_id(host)
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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Sshfps

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

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.BacnetID

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
            self.endpoint, list[BacnetID], params={"id__range": f"{start},{end}"}
        )

    def get_by_host(self, hostname: str | HostName, *, required: bool = False) -> BacnetID | None:
        """Get the BacnetID record for a host by its FQDN."""
        name = self._client.fqdn(hostname)
        obj = self._fetch_by_field("hostname", name)
        if required and obj is None:
            raise EntityNotFound(f"BacnetID record for host {name!r} not found.")
        return obj


class LocationManager(WriteResourceManager[Location]):
    """Operations on :class:`~mreg_api.models.Location` resources."""

    _url_identifier: ClassVar[str] = "host"

    @property
    @override
    def model(self) -> type[Location]:
        return Location

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Locs

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


_ZONE_TTL_MIN = 300
_ZONE_TTL_MAX = 68400


def _valid_zone_ttl(ttl: int) -> int:
    """Validate a zone TTL value (300–68400), mirroring the old model bounds."""
    if ttl < _ZONE_TTL_MIN or ttl > _ZONE_TTL_MAX:
        raise InputFailure(f"Invalid TTL value: {ttl} ({_ZONE_TTL_MIN}->{_ZONE_TTL_MAX})")
    return ttl


def _verify_nameservers(client: MregClient, nameservers: list[str], force: bool = False) -> None:
    """Verify nameservers exist in mreg and have an A-record / glue.

    Ported from ``Zone.verify_nameservers``; resolution of each nameserver now uses the
    explicit ``client.hosts.get_by_name`` instead of the dropped ``get_by_any_means``.

    Raises:
        InputFailure: If no nameservers are given.
        ForceMissing: If a nameserver is missing (or lacks glue) and ``force`` is False.
    """
    if not nameservers:
        raise InputFailure("At least one nameserver is required")

    errors: list[str] = []
    for nameserver in nameservers:
        host = client.hosts.get_by_name(nameserver)
        if host is None:
            if not force:
                errors.append(f"{nameserver} is not in mreg, must force")
        elif host.zone is None and not host.ipaddresses and not force:
            errors.append(f"{nameserver} has no A-record/glue, must force")
    if errors:
        raise ForceMissing("\n".join(errors))


_ZoneT = TypeVar("_ZoneT", bound=Zone)


class _ZoneSubManager(NamedResourceManager[_ZoneT], ABC):
    """Private single-endpoint manager shared by the forward/reverse zone managers.

    Each concrete subclass binds one model (hence one endpoint), honoring the
    one-endpoint invariant of the base. The public surface is :class:`ZoneManager`,
    which dispatches to these by zone-name shape.
    """

    _url_identifier: ClassVar[str] = "name"
    nameservers_endpoint: ClassVar[Endpoint]
    """The per-type nameservers endpoint (forward/reverse differ)."""

    def create(
        self,
        *,
        name: str,
        email: str,
        primary_ns: list[str],
        fetch_after_create: bool = True,
    ) -> _ZoneT | None:
        """Create a zone of this manager's type. Caller verifies nameservers/absence."""
        return self._create(
            {"name": name, "email": email, "primary_ns": primary_ns},
            fetch_after_create=fetch_after_create,
        )

    def update_soa(
        self,
        zone: _ZoneT,
        *,
        primary_ns: str | Unset = UNSET,
        email: str | Unset = UNSET,
        serialno: int | Unset = UNSET,
        refresh: int | Unset = UNSET,
        retry: int | Unset = UNSET,
        expire: int | Unset = UNSET,
        soa_ttl: int | Unset = UNSET,
    ) -> _ZoneT:
        """Update the zone's SOA fields. At least one field must be provided."""
        data: dict[str, Any] = {}
        if primary_ns is not UNSET:
            data["primary_ns"] = primary_ns
        if email is not UNSET:
            data["email"] = email
        if serialno is not UNSET:
            data["serialno"] = serialno
        if refresh is not UNSET:
            data["refresh"] = refresh
        if retry is not UNSET:
            data["retry"] = retry
        if expire is not UNSET:
            data["expire"] = expire
        if isinstance(soa_ttl, int):
            data["soa_ttl"] = _valid_zone_ttl(soa_ttl)
        if not data:
            raise InputFailure("No fields to update")
        return self._patch(zone, data)

    def set_default_ttl(self, zone: _ZoneT, ttl: int) -> _ZoneT:
        """Set the zone's default TTL."""
        return self._patch(zone, {"default_ttl": _valid_zone_ttl(ttl)})

    def set_nameservers(self, zone: _ZoneT, nameservers: list[str], *, force: bool = False) -> None:
        """Replace the zone's nameservers (hits the per-type nameservers endpoint)."""
        _verify_nameservers(self._client, nameservers, force=force)
        self._client.patch(
            self.nameservers_endpoint.with_params(zone.name), json={"primary_ns": nameservers}
        )

    def list_subzones(self, zone: _ZoneT) -> list[_ZoneT]:
        """List subzones of the zone (excluding the zone itself)."""
        zones = self._fetch_list_by_field("name__endswith", f".{zone.name}")
        return [z for z in zones if z.name != zone.name]

    def _ensure_deletable(self, zone: _ZoneT) -> None:
        """Raise if the zone has registered entries or subzones."""
        # XXX: Not foolproof (e.g. SRVs are not hosts), parity with old Zone.ensure_deletable.
        hosts = self._client.hosts.list(zone=zone.id)
        if hosts:
            raise DeleteError(f"Zone has {len(hosts)} registered entries. Can not delete.")
        subzones = self.list_subzones(zone)
        if subzones:
            names = ", ".join(z.name for z in subzones)
            raise DeleteError(f"Zone has registered subzones: '{names}'. Can not delete")

    @override
    def delete(self, obj: _ZoneT, *, force: bool = False) -> None:
        """Delete the zone, guarding against non-empty zones unless ``force``."""
        if not force:
            self._ensure_deletable(obj)
        super().delete(obj)


class _ForwardZoneManager(_ZoneSubManager[ForwardZone]):
    """Private manager for forward zones."""

    nameservers_endpoint: ClassVar[Endpoint] = Endpoint.ForwardZonesNameservers

    @property
    @override
    def model(self) -> type[ForwardZone]:
        return ForwardZone

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.ForwardZones

    def get_from_hostname(self, hostname: str) -> ForwardZoneDelegation | ForwardZone | None:
        """Get the forward zone (or delegation) responsible for a hostname.

        May return a :class:`ForwardZoneDelegation` when the hostname falls under a
        delegated subzone.
        """
        name = self._client.fqdn(hostname)
        resp = self._client.get(Endpoint.ForwardZoneForHost.with_id(name), ok404=True)
        if not resp:
            return None
        blob = resp.json()
        if "delegate" in blob:
            return ForwardZoneDelegation.model_validate(blob)
        if "zone" in blob:
            return ForwardZone.model_validate(blob["zone"])
        if "delegation" in blob:
            return ForwardZoneDelegation.model_validate(blob["delegation"])
        raise UnexpectedDataError(f"Unexpected response from server: {blob}", resp)


class _ReverseZoneManager(_ZoneSubManager[ReverseZone]):
    """Private manager for reverse zones."""

    nameservers_endpoint: ClassVar[Endpoint] = Endpoint.ReverseZonesNameservers

    @property
    @override
    def model(self) -> type[ReverseZone]:
        return ReverseZone

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.ReverseZones


class ZoneManager:
    """Public facade over the forward/reverse zone managers (``client.zones``).

    Zones split into forward/reverse only because their endpoints differ; the
    distinction is an endpoint artifact, not a domain one. This facade dispatches by
    zone-name shape and keeps the per-type managers private. See ADR-0007.

    Methods that take an existing zone accept either an instance or a name (resolved
    by name shape).
    """

    def __init__(self, client: MregClient) -> None:
        """Bind the facade and its private sub-managers to the client."""
        self._client: MregClient = client
        self._forward: _ForwardZoneManager = _ForwardZoneManager(client)
        self._reverse: _ReverseZoneManager = _ReverseZoneManager(client)

    def _sub_for_name(self, name: str) -> _ZoneSubManager[ForwardZone] | _ZoneSubManager[ReverseZone]:
        return self._reverse if is_reverse_zone_name(name) else self._forward

    def _resolve_zone(self, ref: str | ForwardZone | ReverseZone) -> ForwardZone | ReverseZone:
        if isinstance(ref, (ForwardZone, ReverseZone)):
            return ref
        return self.get_by_name(ref, required=True)

    def verify_nameservers(self, nameservers: list[str], force: bool = False) -> None:
        """Verify nameservers exist in mreg and have glue (raises otherwise)."""
        _verify_nameservers(self._client, nameservers, force=force)

    @overload
    def get_by_name(self, name: str, *, required: Literal[True]) -> ForwardZone | ReverseZone: ...
    @overload
    def get_by_name(
        self, name: str, *, required: Literal[False] = ...
    ) -> ForwardZone | ReverseZone | None: ...
    def get_by_name(self, name: str, *, required: bool = False) -> ForwardZone | ReverseZone | None:
        """Get a zone by name; forward/reverse chosen by name shape."""
        return self._sub_for_name(name).get_by_name(name, required=required)

    def ensure_absent(self, name: str) -> None:
        """Raise EntityAlreadyExists if a zone with ``name`` exists."""
        self._sub_for_name(name).ensure_absent(name)

    def list_forward(self) -> list[ForwardZone]:
        """List forward zones."""
        return self._forward.list()

    def list_reverse(self) -> list[ReverseZone]:
        """List reverse zones."""
        return self._reverse.list()

    def get_from_hostname(self, hostname: str) -> ForwardZoneDelegation | ForwardZone | None:
        """Get the forward zone (or delegation) responsible for a hostname."""
        return self._forward.get_from_hostname(hostname)

    def create(
        self,
        *,
        name: str,
        email: str,
        primary_ns: list[str],
        force: bool = False,
        fetch_after_create: bool = True,
    ) -> ForwardZone | ReverseZone | None:
        """Create a forward or reverse zone (type chosen by name shape).

        Verifies the nameservers and that no zone with this name exists first.
        """
        self.verify_nameservers(primary_ns, force=force)
        sub = self._sub_for_name(name)
        sub.ensure_absent(name)
        return sub.create(
            name=name, email=email, primary_ns=primary_ns, fetch_after_create=fetch_after_create
        )

    def update_soa(
        self,
        zone: str | ForwardZone | ReverseZone,
        *,
        primary_ns: str | Unset = UNSET,
        email: str | Unset = UNSET,
        serialno: int | Unset = UNSET,
        refresh: int | Unset = UNSET,
        retry: int | Unset = UNSET,
        expire: int | Unset = UNSET,
        soa_ttl: int | Unset = UNSET,
    ) -> ForwardZone | ReverseZone:
        """Update the zone's SOA fields."""
        z = self._resolve_zone(zone)
        kwargs: dict[str, Any] = {
            "primary_ns": primary_ns,
            "email": email,
            "serialno": serialno,
            "refresh": refresh,
            "retry": retry,
            "expire": expire,
            "soa_ttl": soa_ttl,
        }
        if isinstance(z, ReverseZone):
            return self._reverse.update_soa(z, **kwargs)
        return self._forward.update_soa(z, **kwargs)

    def set_default_ttl(
        self, zone: str | ForwardZone | ReverseZone, ttl: int
    ) -> ForwardZone | ReverseZone:
        """Set the zone's default TTL."""
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            return self._reverse.set_default_ttl(z, ttl)
        return self._forward.set_default_ttl(z, ttl)

    def set_nameservers(
        self, zone: str | ForwardZone | ReverseZone, nameservers: list[str], *, force: bool = False
    ) -> None:
        """Replace the zone's nameservers."""
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            self._reverse.set_nameservers(z, nameservers, force=force)
        else:
            self._forward.set_nameservers(z, nameservers, force=force)

    def list_subzones(
        self, zone: str | ForwardZone | ReverseZone
    ) -> list[ForwardZone] | list[ReverseZone]:
        """List subzones of the zone (excluding the zone itself)."""
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            return self._reverse.list_subzones(z)
        return self._forward.list_subzones(z)

    def delete(self, zone: str | ForwardZone | ReverseZone, *, force: bool = False) -> None:
        """Delete the zone, guarding against non-empty zones unless ``force``."""
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            self._reverse.delete(z, force=force)
        else:
            self._forward.delete(z, force=force)

    def zone_file(self, zone_name: str, *, exclude_private: bool = False) -> str | None:
        """Return the zone file content for the named zone, or None if not found."""
        params: QueryParams = {}
        if exclude_private:
            params["excludePrivate"] = "1"
        resp = self._client.get(Endpoint.Zonefiles.with_id(zone_name), params=params, ok404=True)
        if not resp:
            return None
        return ZoneFile.model_validate(resp.text).root

    @functools.cached_property
    def delegations(self) -> DelegationManager:
        """Manager for zone delegations (``client.zones.delegations``)."""
        return DelegationManager(self._client)


class DelegationManager:
    """Operations on zone delegations (``client.delegations``).

    Delegations have no standalone endpoint; their type (forward/reverse) is derived
    from the parent zone, so every method takes the parent zone as its first argument.
    Kept separate from :class:`ZoneManager` to stay composition-ready (a future
    ``client.zones.delegations``). See ADR-0007.
    """

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client."""
        self._client: MregClient = client

    def _model_for(self, zone: Zone) -> type[ForwardZoneDelegation | ReverseZoneDelegation]:
        return ReverseZoneDelegation if zone.is_reverse() else ForwardZoneDelegation

    def _endpoint_for(self, zone: Zone) -> Endpoint:
        """Return the delegations endpoint for the given zone's type."""
        return (
            Endpoint.ReverseZonesDelegations if zone.is_reverse() else Endpoint.ForwardZonesDelegations
        )

    def _ensure_in_zone(self, zone: Zone, name: str) -> None:
        if not name.endswith(f".{zone.name}"):
            raise InputFailure(f"Delegation {name!r} is not in {zone.name!r}")

    @overload
    def get(
        self, zone: Zone, name: str, *, required: Literal[True]
    ) -> ForwardZoneDelegation | ReverseZoneDelegation: ...
    @overload
    def get(
        self, zone: Zone, name: str, *, required: Literal[False] = ...
    ) -> ForwardZoneDelegation | ReverseZoneDelegation | None: ...
    def get(
        self, zone: Zone, name: str, *, required: bool = False
    ) -> ForwardZoneDelegation | ReverseZoneDelegation | None:
        """Get a delegation in ``zone`` by name."""
        self._ensure_in_zone(zone, name)
        cls = self._model_for(zone)
        resp = self._client.get(cls.endpoint_with_name(zone, name), ok404=True)
        if not resp:
            if required:
                raise EntityNotFound(f"Could not find delegation {name!r} in zone {zone.name!r}")
            return None
        return cls.model_validate_json(resp.text)

    def list_by_zone(self, zone: Zone) -> list[ForwardZoneDelegation | ReverseZoneDelegation]:
        """List all delegations for a zone."""
        cls = self._model_for(zone)
        return self._client.get_typed(self._endpoint_for(zone).with_params(zone.name), list[cls])

    def create(
        self,
        zone: Zone,
        *,
        name: str,
        nameservers: list[str],
        comment: str = "",
        force: bool = False,
        fetch_after_create: bool = True,
    ) -> ForwardZoneDelegation | ReverseZoneDelegation | None:
        """Create a delegation in ``zone``.

        Verifies the delegation name is within the zone and the nameservers exist.
        Unless ``force``, also checks the delegated zone exists and matches the parent
        zone type, and that the delegation does not already exist.
        """
        self._ensure_in_zone(zone, name)
        _verify_nameservers(self._client, nameservers, force=force)

        if not force:
            delegated = self._client.zones.get_by_name(name)
            if not delegated:
                raise InputFailure(f"Zone {name!r} does not exist. Must force.")
            if delegated.is_reverse() != zone.is_reverse():
                raise InputFailure(f"Delegation {name!r} is not a {type(zone).__name__} zone")

        if self.get(zone, name) is not None:
            raise EntityAlreadyExists(f"Zone {zone.name!r} already has a delegation named {name!r}")

        self._client.post(
            self._endpoint_for(zone).with_params(zone.name),
            json={"name": name, "nameservers": nameservers, "comment": comment},
        )
        if fetch_after_create:
            return self.get(zone, name, required=True)
        return None

    def delete(self, zone: Zone, name: str) -> None:
        """Delete a delegation from ``zone``."""
        self._ensure_in_zone(zone, name)
        _ = self.get(zone, name, required=True)
        cls = self._model_for(zone)
        self._client.delete(cls.endpoint_with_name(zone, name))

    def set_comment(
        self, zone: Zone, name: str, comment: str
    ) -> ForwardZoneDelegation | ReverseZoneDelegation:
        """Set (or clear, with ``""``) the comment for a delegation."""
        _ = self.get(zone, name, required=True)
        cls = self._model_for(zone)
        self._client.patch(cls.endpoint_with_name(zone, name), json={"comment": comment})
        return self.get(zone, name, required=True)


# TODO: test subclassing here!


class DhcpHostManager(ResourceManager[T], ABC):
    """Read-only manager for DHCP host records (``client.dhcphost``).

    This is a generic base class for the IPv4, IPv6, and IPv6-via-IPv4 managers.
    Cannot be instantiated directly; use one of the concrete subclasses instead.
    """

    def list_by_range(self, ip: str | IP_AddressT, range: str) -> list[T]:  # noqa: A002
        """List DHCP hosts within the given IP range."""
        return self._client.get_typed(
            Endpoint.DhcpHostsByRange.with_params(str(ip), range),
            list[self.model],
        )


class DhcpHostIPv4Manager(DhcpHostManager[DhcpHostIPv4]):
    """Read-only manager for IPv4 DHCP host records (``client.dhcphost_ipv4``)."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv4]:
        return DhcpHostIPv4

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv4


class DhcpHostIPv6Manager(DhcpHostManager[DhcpHostIPv6]):
    """Read-only manager for IPv6 DHCP host records (``client.dhcphost_ipv6``)."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv6]:
        return DhcpHostIPv6

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv6


class DhcpHostIPv6ByIPv4Manager(DhcpHostManager[DhcpHostIPv6ByIPv4]):
    """Read-only manager for IPv6-via-IPv4 DHCP host records (``client.dhcphost_ipv6byipv4``)."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv6ByIPv4]:
        return DhcpHostIPv6ByIPv4

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv6ByIpv4


class NameServerManager(NamedResourceManager[NameServer]):
    """Access to :class:`~mreg_api.models.NameServer` resources (``client.nameservers``).

    Nameservers are created and deleted implicitly through zone and delegation
    operations; this manager exposes listing and lookup.
    """

    _url_identifier: ClassVar[str] = "name"

    @property
    @override
    def model(self) -> type[NameServer]:
        return NameServer

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Nameservers
