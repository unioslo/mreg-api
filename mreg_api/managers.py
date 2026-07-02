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
import logging
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
from mreg_api.models import HealthInfo
from mreg_api.models import HeartbeatHealth
from mreg_api.models import HInfo
from mreg_api.models import Host
from mreg_api.models import HostContactModification
from mreg_api.models import HostGroup
from mreg_api.models import IPAddress
from mreg_api.models import Label
from mreg_api.models import LDAPHealth
from mreg_api.models import Library
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
from mreg_api.models import ServerLibraries
from mreg_api.models import ServerVersion
from mreg_api.models import Srv
from mreg_api.models import UserDjangoStatus
from mreg_api.models import UserInfo
from mreg_api.models import UserMregStatus
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

logger = logging.getLogger(__name__)


class Unset:
    """Sentinel for "unchanged" parameters in update methods."""

    def __repr__(self) -> str:
        """Return a string representation of the Unset sentinel."""
        return "UNSET"


UNSET = Unset()


T = TypeVar("T", bound=MregModel)


def resolve_host_id(host: int | str | Host, client: MregClient) -> int:
    """Resolve a host reference to its numeric ID.

    Args:
        host (int | str | Host): Host instance or numeric ID.
            Passing a hostname performs an API call to fetch the host.
        client (MregClient): API client instance. Used to fetch host by name.

    Returns:
        int: The numeric ID of the host.
    """
    if isinstance(host, Host):
        return host.id
    elif isinstance(host, int):
        return host
    return HostManager(client).get_by_name(host, required=True).id


def resolve_host_name(host: Host | str | HostName, client: MregClient) -> str:
    """Resolve a host reference to its fully qualified name.

    Performs no API calls.

    Args:
        host (Host | str | HostName): Host instance, name string, or HostName instance.
        client (MregClient): API client instance. Used to determine FQDN.

    Returns:
        str: The fully qualified domain name of the host.
    """
    if isinstance(host, Host):
        return host.name
    return client.fqdn(str(host))


def resolve_host(host: int | str | Host, client: MregClient) -> Host:
    """Resolve a host ID, name or Host to a Host object.

    Performs API call to fetch host if argument is ID or name

    Args:
        host (int | str | Host): Host ID, name or object.
        client (MregClient): Client to fetch with.

    Returns:
        Host: Resolved host object.
    """
    if isinstance(host, Host):
        return host
    elif isinstance(host, int):
        return HostManager(client).get_by_id(host, required=True)
    else:
        return HostManager(client).get_by_name(host, required=True)


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

    # NOTE: this won't work for hosts!
    # We need to define ref as `int | str | T`, and determine the endpoint
    # to hit OR field to search for based on the URL identifier
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

    def _resolve_hostname(self, host: str | HostName | Host) -> str:
        """Resolve a host reference to its fully qualified name.

        Args:
            host: A host reference, which can be a string, HostName, or Host instance.

        Returns:
            The fully qualified name of the host as a string.
        """
        return resolve_host_name(host, self._client)

    def _resolve_label_id(self, label: int | str | Label) -> int:
        """Resolve a label reference to its numeric ID."""
        if isinstance(label, int):
            return label
        if isinstance(label, Label):
            return label.id
        return LabelManager(self._client).get_by_name(str(label), required=True).id

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
        field = field.casefold()

        # Field is the ID used in the URL path. i.e. /hosts/example.com, /sshfps/123, etc.
        if field == self._url_identifier.casefold():
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
        lookup = getattr(obj, "id", None) if hasattr(obj, "id") else getattr(obj, self._url_identifier, None)
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

        Args:
            ident (str | int): The URL identifier to check (id / name / network, per resource).

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
        """List resources, optionally filtered by query parameters.

        Args:
            limit (int | None): Maximum number of results to return. Defaults to 500.
            **query: Optional filter parameters forwarded to the list endpoint.
        """
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

        Over-fetches on certain non-standard endpoints that do not implement pagination
        such as `networks/{network}/unused_list`, `/dhcphosts`, and others.

        Args:
            required: When ``True``, raise if no resource exists (returns ``T``).
            **query: Optional filter parameters forwarded to the list endpoint.

        Raises:
            EntityNotFound: If ``required`` is True and no resource is found.
        """
        params: QueryParams = dict(query)
        res = self._client.get_first(self.endpoint, params)
        if res is None:
            if required:
                raise EntityNotFound(f"No {self.model.__name__} found.")
            else:
                return None
        return get_type_adapter(self.model).validate_python(res)


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

    @override
    def ensure_absent(self, ident: str | int) -> None:
        """Assert that no resource with ``name`` exists.

        The "must not exist" guard (replaces the old ``get_x_and_raise``); kept a
        distinct verb rather than a return-typed-``None`` overload of :meth:`get`.

        Args:
            ident (str | int): The URL identifier or name to check.

        Raises:
            EntityAlreadyExists: If a resource with `name` exists.
        """
        if isinstance(ident, int):
            # If the identifier is an integer, we assume it's an ID and not a name.
            # In this case, we can delegate to the parent class's ensure_absent method.
            super().ensure_absent(ident)
            return
        # If we have a string, delegate to internal get_by_name method
        if self.get_by_name(ident, required=False) is not None:
            raise EntityAlreadyExists(f"{self.model.__name__} {ident!r} already exists.")

    # FIXME: Do we actually need an explicit name based lookup? Can we just use get_by_field() instead?
    #       get() should IDEALLY handle the "primary" path i.e. the URL identifier.
    #       So if we could do `client.host.get("ExAMple.com")` and it know that
    #       the URL identifier is "name" and it should normalize the name, then we don't need
    #       a separate get_by_name() method.
    @overload
    def get_by_name(self, name: str, *, required: Literal[True]) -> T: ...
    @overload
    def get_by_name(self, name: str, *, required: Literal[False] = ...) -> T | None: ...
    def get_by_name(self, name: str, *, required: bool = False) -> T | None:
        """Get a resource by name (searches the name field).

        For the "must not exist" guard use :meth:`ensure_absent` (for named resources
        the external id-field is the name, so it covers name absence).

        Args:
            name (str): The name to look up.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and the resource is not found.
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
        # NOTE: no normalization or validation - this is a regex pattern, not a host name
        param, value = convert_wildcard_to_regex(self.name_field, name, True)
        return self._client.get_typed(self.endpoint, list[self.model], params={param: value})


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

        Args:
            name (str): The name of the resource to fetch history for.
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

        Args:
            host_id (int): The numeric id of the host.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and the host is not found.
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

        Args:
            ip (str | IP_AddressT): The IP address to look up.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            MultipleEntitiesFound: If more than one host matches the IP address.
            EntityNotFound: If ``required`` is True and no host is found.
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
        """Get a host by MAC address.

        Args:
            mac (str | MacAddress): The MAC address to look up.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and no host is found.
        """
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
        """List hosts by MAC address.

        Args:
            mac (str | MacAddress): The MAC address to filter by.
        """
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
        ref: int | str | Host,
        *,
        name: str | HostName | Unset = UNSET,
        comment: str | None | Unset = UNSET,
        contacts: list[str] | Unset = UNSET,
        ttl: int | None | Unset = UNSET,
    ) -> Host:
        """Update a host's mutable fields.

        Args:
            ref (int | str | Host): Host instance or numeric ID.
            name (str | HostName | Unset): New name for the host. Omit to leave unchanged.
            comment (str | None | Unset): New comment. Pass None to unset, omit to leave unchanged.
            contacts (list[str] | Unset): New contacts list. Omit to leave unchanged.
            ttl (int | None | Unset): New TTL. Pass None to reset to default, omit to leave unchanged.
        """
        host = resolve_host(ref, self._client)
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

    def add_contacts(self, ref: int | str | Host, contacts: list[str]) -> HostContactModification:
        """Add contacts to a host (atomic; POST to /hosts/{name}/contacts/).

        Args:
            ref (int | str | Host): Host instance or numeric ID.
            contacts (list[str]): Email addresses to add as contacts.
        """
        host = resolve_host(ref, self._client)
        resp = self._client.post(Endpoint.HostsContacts.with_params(host.name), json={"emails": contacts})
        return get_type_adapter(HostContactModification).validate_json(resp.text)

    def clear_contacts(self, ref: int | str | Host) -> HostContactModification:
        """Remove all contacts from a host (atomic; DELETE /hosts/{name}/contacts/).

        Args:
            ref (int | str | Host): Host instance or numeric ID.
        """
        host = resolve_host(ref, self._client)
        resp = self._client.delete(Endpoint.HostsContacts.with_params(host.name))
        return get_type_adapter(HostContactModification).validate_json(resp.text)

    def remove_contacts(self, ref: int | str | Host, contacts: list[str]) -> HostContactModification:
        """Remove specific contacts from a host (atomic; DELETE /hosts/{name}/contacts/).

        Args:
            ref (int | str | Host): Host instance or numeric ID.
            contacts (list[str]): Email addresses to remove from contacts.
        """
        host = resolve_host(ref, self._client)
        resp = self._client.delete(Endpoint.HostsContacts.with_params(host.name), json={"emails": contacts})
        return get_type_adapter(HostContactModification).validate_json(resp.text)

    def networks(self, ref: int | str | Host) -> dict[Network, list[IPAddress]]:
        """Return a dict mapping each network to the host's IP addresses on that network.

        Networks not registered in MREG produce a placeholder via
        :meth:`~mreg_api.models.Network.dummy_network_from_ip`.

        Args:
            ref (int | str | Host): Host instance or numeric ID.
        """
        host = resolve_host(ref, self._client)
        net_manager = NetworkManager(self._client)
        result: dict[Network, list[IPAddress]] = {}
        for ip in host.ipaddresses:
            network = net_manager.get_by_ip(ip.ipaddress)
            if network is None:
                network = Network.dummy_network_from_ip(ip)
            if network not in result:
                result[network] = []
            result[network].append(ip)
        return result

    def vlans(self, ref: int | str | Host) -> dict[int, list[IPAddress]]:
        """Return a dict mapping VLAN ID to host IPs on that VLAN. IPs with no VLAN map to 0.

        Args:
            ref (int | str | Host): Host instance or numeric ID.
        """
        result: dict[int, list[IPAddress]] = {}
        for network, ips in self.networks(ref).items():
            vlan = network.vlan or 0
            if vlan not in result:
                result[vlan] = []
            result[vlan].extend(ips)
        return result

    def all_ips_on_same_vlan(self, ref: int | str | Host) -> bool:
        """Return True if all host IPs share a single VLAN (or there are no IPs).

        Args:
            ref (int | str | Host): Host instance or numeric ID.
        """
        return len(self.vlans(ref)) <= 1


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
        """Create a host group.

        Args:
            name (str): Name of the host group.
            description (str | Unset): Description of the group. Omit to leave unset.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
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
        """Update a host group's mutable fields.

        Args:
            ref (int | HostGroup): HostGroup instance or numeric ID.
            description (str | Unset): New description. Omit to leave unchanged.
        """
        group = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(group, data)

    def set_description(self, group: int | HostGroup, description: str) -> HostGroup:
        """Set the description for the host group.

        Args:
            group (int | HostGroup): HostGroup instance or numeric ID.
            description (str): New description to set.
        """
        group = self._resolve(group)
        return self.update(group, description=description)

    def _resolve_hostgroup_name(self, hostgroup: str | HostGroup) -> str:
        """Resolve a host group reference (instance or name) to a string name."""
        if isinstance(hostgroup, HostGroup):
            return hostgroup.name
        return str(hostgroup)

    def add_group(self, group: int | HostGroup, subgroup: str | HostGroup) -> HostGroup:
        """Add a group to a host group.

        Args:
            group (int | HostGroup): The parent HostGroup instance or numeric ID.
            subgroup (str | HostGroup): HostGroup instance or name string to add as a subgroup.
        """
        group = self._resolve(group)
        subgroup_name = self._resolve_hostgroup_name(subgroup)

        self._client.post(
            Endpoint.HostGroupsAddHostGroups.with_params(group.name),
            json={"name": subgroup_name},
        )
        return self._refetch(group)

    def remove_group(self, group: int | HostGroup, subgroup: str | HostGroup) -> HostGroup:
        """Remove a group from a host group.

        Args:
            group (int | HostGroup): The parent HostGroup instance or numeric ID.
            subgroup (str | HostGroup): HostGroup instance or name string to remove.
        """
        group = self._resolve(group)
        subgroup_name = self._resolve_hostgroup_name(subgroup)

        self._client.delete(
            Endpoint.HostGroupsRemoveHostGroups.with_params(group.name, subgroup_name),
        )
        return self._refetch(group)

    def add_host(self, group: int | HostGroup, host: str | Host) -> HostGroup:
        """Add a host to a host group.

        Args:
            group (int | HostGroup): HostGroup instance or numeric ID.
            host (str | Host): Host reference (name string or Host instance).
        """
        group = self._resolve(group)
        hostname = resolve_host_name(host, self._client)

        self._client.post(
            Endpoint.HostGroupsAddHosts.with_params(group.name),
            json={"name": hostname},
        )
        return self._refetch(group)

    def remove_host(self, group: int | HostGroup, host: str | Host) -> HostGroup:
        """Remove a host from a host group.

        Args:
            group (int | HostGroup): HostGroup instance or numeric ID.
            host (str | Host): Host reference (name string or Host instance).
        """
        group = self._resolve(group)
        hostname = resolve_host_name(host, self._client)

        self._client.delete(
            Endpoint.HostGroupsRemoveHosts.with_params(group.name, hostname),
        )
        return self._refetch(group)

    def add_owner(self, group: int | HostGroup, name: str) -> HostGroup:
        """Add an owner to a host group.

        Args:
            group (int | HostGroup): HostGroup instance or numeric ID.
            name (str): Name of the owner to add.
        """
        group = self._resolve(group)

        self._client.post(
            Endpoint.HostGroupsAddOwner.with_params(group.name),
            json={"name": name},
        )
        return self._refetch(group)

    def remove_owner(self, group: int | HostGroup, name: str) -> HostGroup:
        """Remove an owner from a host group.

        Args:
            group (int | HostGroup): HostGroup instance or numeric ID.
            name (str): Name of the owner to remove.
        """
        group = self._resolve(group)

        self._client.delete(
            Endpoint.HostGroupsRemoveOwner.with_params(group.name, name),
        )
        return self._refetch(group)

    # RENAMED: get_all_parents -> list_parents
    def list_parents(self, group: int | HostGroup) -> list[HostGroup]:
        """Get all parent groups of a host group.

        Renamed from `get_all_parents` to `list_parents`

        Args:
            group (int | HostGroup): HostGroup instance or numeric ID.
        """
        group = self._resolve(group)
        parents: list[HostGroup] = []
        for parent in group.parent:  # why singular name?
            pobj = self._fetch_by_field("name", parent)
            if pobj:
                parents.append(pobj)
                parents.extend(self.list_parents(pobj))
        return parents

    def list_by_host(self, host: int | str | Host, *, traverse: bool = False) -> list[HostGroup]:
        """List all hostgroups that include the given host.

        Args:
            host: Host instance or numeric host ID.
            traverse: If True, also include all parent groups recursively.
        """
        host_id = resolve_host_id(host, self._client)
        direct = self._fetch_list_by_field("hosts", host_id)
        if not traverse:
            return sorted(direct, key=lambda g: g.name)
        groups: list[HostGroup] = list(direct)
        for group in direct:
            groups.extend(self.list_parents(group))
        return sorted(groups, key=lambda g: g.name)


class LabelManager(NamedResourceManager[Label]):
    """Operations on :class:`~mreg_api.models.Label` resources."""

    # NOTE: the regular labels endpoint uses IDs for lookups, but it is possible
    # to fetch by name when using the /labels/name endpoint.
    # This makes no sense, of course, but that's how it is.
    _url_identifier: ClassVar[str] = "id"

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
        fetch_after_create: bool = False,  # pyright: ignore[reportUnusedParameter]  # noqa: ARG002
    ) -> Label | None:
        """Create a label.

        NOTE: The API does not return a Location header for label creation, fetching after creation
        is not supported.

        Args:
            name (str): Name of the label.
            description (str): Description of the label.
            fetch_after_create (bool): Ignored; the API does not support fetching after creation.
        """
        return self._create({"name": name, "description": description}, fetch_after_create=False)

    def update(
        self,
        ref: int | Label,
        *,
        description: str | Unset = UNSET,
    ) -> Label:
        """Update a label's mutable fields.

        Args:
            ref (int | Label): Label instance or numeric ID.
            description (str | Unset): New description. Omit to leave unchanged.
        """
        label = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(label, data)

    def set_description(self, label: int | Label, description: str) -> Label:
        """Set the description for the label.

        Args:
            label (int | Label): Label instance or numeric ID.
            description (str): New description to set.
        """
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

    def _resolve_atom_name(self, atom: str | Atom) -> str:
        """Resolve an atom reference (instance or name) to a string name."""
        if isinstance(atom, Atom):
            return atom.name
        return str(atom)

    def create(
        self,
        *,
        name: str,
        description: str = "",
        fetch_after_create: bool = True,
    ) -> Role | None:
        """Create a role.

        Args:
            name (str): Name of the role.
            description (str): Description of the role. Defaults to "".
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        return self._create({"name": name, "description": description}, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | Role,
        *,
        description: str | Unset = UNSET,
    ) -> Role:
        """Update a role's mutable fields.

        Args:
            ref (int | Role): Role instance or numeric ID.
            description (str | Unset): New description. Omit to leave unchanged.
        """
        role = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(role, data)

    def set_description(self, ref: int | Role, description: str) -> Role:
        """Set the description for the role.

        Args:
            ref (int | Role): Role instance or numeric ID.
            description (str): New description to set.
        """
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

    def list_with_atom(self, atom: str | Atom) -> list[Role]:
        """List all roles that contain a given atom.

        Renamed from ``Role.get_roles_with_atom``.

        Args:
            atom (str | Atom): Atom instance or name string.
        """
        atom_name = self._resolve_atom_name(atom)
        return self._fetch_list_by_field("atoms__name__exact", atom_name)

    def add_atom(self, ref: int | Role, atom: str | Atom) -> bool:
        """Add an atom to the role.

        Args:
            ref (int | Role): Role instance or numeric ID.
            atom (str | Atom): Atom instance or name string.

        Returns:
            bool: True if the atom was successfully removed.
                DEPRECATED: Maintains parity with older library versions.
                Will never return False on failure; an exception is raised instead.

        Raises:
            EntityNotFound: If the atom does not exist.
            EntityAlreadyExists: If the atom is already a member of the role.
        """
        role = self._resolve(ref)
        atom_name = self._resolve_atom_name(atom)
        _ = AtomManager(self._client).get_by_name(atom_name, required=True)
        if atom_name in role.atoms:
            raise EntityAlreadyExists(f"Atom {atom_name!r} already a member of role {role.name!r}")
        # TODO: need a better abstraction for endpoints that
        # return the new version of the resource after modification
        self._client.post(Endpoint.HostPolicyRolesAddAtom.with_params(role.name), json={"name": atom_name})
        return True

    def remove_atom(self, ref: int | Role, atom: str | Atom) -> bool:
        """Remove an atom from the role.

        Args:
            ref (int | Role): Role instance or numeric ID.
            atom (str | Atom): Atom instance or name string.

        Returns:
            bool: True if the atom was successfully removed.
                DEPRECATED: Maintains parity with older library versions.
                Will never return False on failure; an exception is raised instead.

        Raises:
            EntityOwnershipMismatch: If the atom is not a member of the role.
        """
        role = self._resolve(ref)
        atom_name = self._resolve_atom_name(atom)
        if atom_name not in role.atoms:
            raise EntityOwnershipMismatch(f"Atom {atom_name!r} not a member of {role.name!r}")
        self._client.delete(Endpoint.HostPolicyRolesRemoveAtom.with_params(role.name, atom_name))
        return True

    def add_host(self, ref: int | Role, host: str | Host) -> bool:
        """Add a host to the role by name.

        Args:
            ref (int | Role): Role instance or numeric ID.
            host (str | Host): Host reference (name string or Host instance).

        Returns:
            bool: True if the host was successfully added.
                DEPRECATED: Maintains parity with older library versions.
                Will never return False on failure; an exception is raised instead.
        """
        role = self._resolve(ref)
        hostname = resolve_host_name(host, self._client)
        self._client.post(Endpoint.HostPolicyRolesAddHost.with_params(role.name), json={"name": hostname})
        return True

    def remove_host(self, ref: int | Role, host: str | Host) -> bool:
        """Remove a host from the role by name.

        Args:
            ref (int | Role): Role instance or numeric ID.
            host (str | Host): Host reference (name string or Host instance).

        Returns:
            bool: True if the host was successfully removed.
                DEPRECATED: Maintains parity with older library versions.
                Will never return False on failure; an exception is raised instead.
        """
        role = self._resolve(ref)
        hostname = resolve_host_name(host, self._client)
        self._client.delete(Endpoint.HostPolicyRolesRemoveHost.with_params(role.name, hostname))
        return True

    def get_labels(self, ref: int | Role) -> list[Label]:
        """Get the labels associated with the role.

        Args:
            ref (int | Role): Role instance or numeric ID.
        """
        role = self._resolve(ref)
        labels = LabelManager(self._client)
        return [labels.get(lid, required=True) for lid in role.labels]

    def add_label(self, ref: int | Role, label: int | str | Label) -> Role:
        """Add a label to the role.

        Args:
            ref (int | Role): Role instance or numeric ID.
            label (int | str | Label): Label instance, name, or numeric ID.

        Raises:
            EntityNotFound: If the label does not exist.
            EntityAlreadyExists: If the role already has the label.
        """
        role = self._resolve(ref)
        label_id = self._resolve_label_id(label)
        if label_id in role.labels:
            raise EntityAlreadyExists(f"Role {role.name!r} already has label {label!r}")
        return self._patch(role, {"labels": [*role.labels, label_id]})

    def remove_label(self, ref: int | Role, label: int | str | Label) -> Role:
        """Remove a label from the role.

        Args:
            ref (int | Role): Role instance or numeric ID.
            label (int | str | Label): Label instance, name, or numeric ID.

        Raises:
            EntityNotFound: If the label does not exist.
            EntityOwnershipMismatch: If the role does not have the label.
        """
        role = self._resolve(ref)
        label_id = self._resolve_label_id(label)
        if label_id not in role.labels:
            raise EntityOwnershipMismatch(f"Role {role.name!r} does not have label {label!r}")
        return self._patch(role, {"labels": [lid for lid in role.labels if lid != label_id]})

    def list_by_host(self, host: int | str | Host) -> list[Role]:
        """List all roles that include the given host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
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
        """Create an atom.

        Args:
            name (str): Name of the atom.
            description (str): Description of the atom. Defaults to "".
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        return self._create({"name": name, "description": description}, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | Atom,
        *,
        description: str | Unset = UNSET,
    ) -> Atom:
        """Update an atom's mutable fields.

        Args:
            ref (int | Atom): Atom instance or numeric ID.
            description (str | Unset): New description. Omit to leave unchanged.
        """
        atom = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(atom, data)

    def set_description(self, ref: int | Atom, description: str) -> Atom:
        """Set the description for the atom.

        Args:
            ref (int | Atom): Atom instance or numeric ID.
            description (str): New description to set.
        """
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
        """Update a permission's mutable fields.

        Args:
            ref (int | Permission): Permission instance or numeric ID.
            group (str | Unset): New netgroup name. Omit to leave unchanged.
            range (str | Unset): New network range (CIDR). Omit to leave unchanged.
            regex (str | Unset): New host regex pattern. Omit to leave unchanged.
            labels (list[int] | Unset): New list of label IDs. Omit to leave unchanged.
        """
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

    def add_label(self, ref: int | Permission, label: int | str | Label) -> Permission:
        """Add a label to the permission.

        Args:
            ref: The permission (instance or numeric id).
            label: The label to add (instance, name, or numeric id).

        Raises:
            EntityNotFound: If the label does not exist.
            EntityAlreadyExists: If the permission already has this label.
        """
        perm = self._resolve(ref)
        label_id = self._resolve_label_id(label)
        if label_id in perm.labels:
            raise EntityAlreadyExists(f"Permission already has label {label!r}.")
        return self.update(perm, labels=[*perm.labels, label_id])

    def remove_label(self, ref: int | Permission, label: int | str | Label) -> Permission:
        """Remove a label from the permission.

        Args:
            ref: The permission (instance or numeric id).
            label: The label to remove (instance, name, or numeric id).

        Raises:
            EntityNotFound: If the label does not exist or the permission lacks it.
        """
        perm = self._resolve(ref)
        label_id = self._resolve_label_id(label)
        if label_id not in perm.labels:
            raise EntityNotFound(f"Permission does not have label {label!r}.")
        return self.update(perm, labels=[lid for lid in perm.labels if lid != label_id])

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

        Args:
            group (str): The netgroup name.
            range (str): The network range (CIDR).
            regex (str): The host regex pattern.
            required (bool): When True, raise EntityNotFound if not found.

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
        """Create a network policy attribute.

        Args:
            name (str): Name of the attribute (lowercased).
            description (str): Description of the attribute.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        return self._create({"name": name, "description": description}, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | NetworkPolicyAttribute,
        *,
        description: str | Unset = UNSET,
    ) -> NetworkPolicyAttribute:
        """Update a network policy attribute's mutable fields.

        Args:
            ref (int | NetworkPolicyAttribute): NetworkPolicyAttribute instance or numeric ID.
            description (str | Unset): New description. Omit to leave unchanged.
        """
        attr = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        return self._patch(attr, data)

    def set_description(self, ref: int | NetworkPolicyAttribute, description: str) -> NetworkPolicyAttribute:
        """Set the description for the attribute.

        Args:
            ref (int | NetworkPolicyAttribute): NetworkPolicyAttribute instance or numeric ID.
            description (str): New description to set.
        """
        attr = self._resolve(ref)
        return self.update(attr, description=description)

    def get_policies(self, ref: int | NetworkPolicyAttribute) -> list[NetworkPolicy]:
        """Get all network policies that use this attribute.

        Args:
            ref (int | NetworkPolicyAttribute): NetworkPolicyAttribute instance or numeric ID.
        """
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

        Args:
            ref (int | NetworkPolicy): NetworkPolicy instance or numeric ID.
            description (str | Unset): New description. Omit to leave unchanged.
            community_template_pattern (str | None | Unset): New community name template pattern.
                Pass None to unset, omit to leave unchanged.
        """
        pol = self._resolve(ref)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        if community_template_pattern is not UNSET:
            data["community_template_pattern"] = community_template_pattern
        return self._patch(pol, data)

    def set_description(self, ref: int | NetworkPolicy, description: str) -> NetworkPolicy:
        """Set the description for the policy.

        Args:
            ref (int | NetworkPolicy): NetworkPolicy instance or numeric ID.
            description (str): New description to set.
        """
        pol = self._resolve(ref)
        return self.update(pol, description=description)

    def add_attribute(
        self,
        ref: int | NetworkPolicy,
        attr: NetworkPolicyAttribute,
        value: bool = True,
    ) -> NetworkPolicy:
        """Add an attribute to a policy.

        Args:
            ref (int | NetworkPolicy): NetworkPolicy instance or numeric ID.
            attr (NetworkPolicyAttribute): The attribute to add.
            value (bool): The boolean value to set for the attribute. Defaults to True.

        Raises:
            EntityAlreadyExists: If the policy already has this attribute.
        """
        pol = self._resolve(ref)
        if pol.get_attribute(attr.name):
            raise EntityAlreadyExists(f"Policy {pol.name!r} already has attribute {attr.name!r}.")
        attrs = [*pol.attributes, NetworkPolicyAttributeValue(name=attr.name, value=value)]
        return self._patch(pol, {"attributes": [{"name": a.name, "value": a.value} for a in attrs]})

    def _resolve_attribute_name(self, attribute: str | NetworkPolicyAttribute) -> str:
        """Resolve an attribute reference (instance or name) to a string name."""
        if isinstance(attribute, NetworkPolicyAttribute):
            return attribute.name
        return str(attribute)

    def remove_attribute(
        self, ref: int | NetworkPolicy, attribute: str | NetworkPolicyAttribute
    ) -> NetworkPolicy:
        """Remove an attribute from a policy.

        Args:
            ref (int | NetworkPolicy): NetworkPolicy instance or numeric ID.
            attribute (str | NetworkPolicyAttribute): NetworkPolicyAttribute instance or name string.

        Raises:
            EntityNotFound: If the policy does not have this attribute.
        """
        pol = self._resolve(ref)
        attribute_name = self._resolve_attribute_name(attribute)
        if not pol.get_attribute(attribute_name):
            raise EntityNotFound(f"Policy {pol.name!r} does not have attribute {attribute_name!r}.")
        attrs = [a for a in pol.attributes if a.name != attribute_name]
        return self._patch(pol, {"attributes": [{"name": a.name, "value": a.value} for a in attrs]})

    def networks(self, ref: int | NetworkPolicy) -> list[Network]:
        """Get all networks that use this policy.

        Args:
            ref (int | NetworkPolicy): NetworkPolicy instance or numeric ID.
        """
        pol = self._resolve(ref)
        return self._client.get_typed(Endpoint.Networks, list[Network], params={"policy": pol.id})

    @functools.cached_property
    def attribute(self) -> NetworkPolicyAttributeManager:
        """Manager for network policy attributes (``client.network.policy.attribute``)."""
        return NetworkPolicyAttributeManager(self._client)


# NOTE: WHY does this not inherit from the regular ResourceManager?
class CommunityManager:
    """Operations on network communities (``client.network.communities``).

    Communities are always scoped to a network — every method takes a network
    reference (address string or :class:`~mreg_api.models.Network` instance).
    """

    def __init__(self, client: MregClient, network_manager: NetworkManager) -> None:
        """Bind the manager to the client."""
        self._client = client
        self._network_manager = network_manager

    def _resolve_network_address(self, network: str | int | Network) -> str:
        if isinstance(network, str):
            return network
        elif isinstance(network, Network):
            return network.network
        else:
            return self._network_manager.get(network, required=True).network

    def _resolve_community_id(self, community: str | int | Community, network: str | int | Network) -> int:
        # Already an int or Community object - return ID
        if isinstance(community, int):
            return community
        elif isinstance(community, Community):
            return community.id

        # Community is a name: fetch it by name within the given network
        network_addr = self._resolve_network_address(network)
        com = self.get_by_name(community, network_addr, required=True)
        return com.id

    def list(self, network: str | int | Network) -> list[Community]:
        """List all communities for a network.

        Args:
            network (str | int | Network): Network reference (address string, ID, or Network instance).
        """
        addr = self._resolve_network_address(network)
        return self._client.get_typed(Endpoint.NetworkCommunities.with_params(addr), list[Community])

    def update(
        self,
        community: int | str | Community,
        network: str | int | Network,
        *,
        name: str | Unset = UNSET,
        description: str | Unset = UNSET,
    ) -> Community:
        """Update a community's mutable fields.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address string, ID, or Network instance).
            name (str | Unset): New name. Omit to leave unchanged.
            description (str | Unset): New description. Omit to leave unchanged.
        """
        addr = self._resolve_network_address(network)
        community_id = self._resolve_community_id(community, network)
        data: dict[str, Any] = {}
        if name is not UNSET:
            data["name"] = name
        if description is not UNSET:
            data["description"] = description
        resp = self._client.patch(Endpoint.NetworkCommunity.with_params(addr, community_id), json=data)
        return Community.model_validate_json(resp.text)

    @overload
    def get_by_name(
        self, name: str, network: str | int | Network, *, required: Literal[True]
    ) -> Community: ...
    @overload
    def get_by_name(
        self, name: str, network: str | int | Network, *, required: Literal[False] = ...
    ) -> Community | None: ...
    def get_by_name(
        self, name: str, network: str | int | Network, *, required: bool = False
    ) -> Community | None:
        """Get a community by name within a network.

        Args:
            name (str): The community name to look up.
            network (str | int | Network): Network reference (address string, ID, or Network instance).
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and the community is not found.
        """
        community = next((c for c in self.list(network) if c.name == name), None)
        if required and community is None:
            raise EntityNotFound(f"Community {name!r} not found.")
        return community

    def create(self, network: str | int | Network, *, name: str, description: str) -> bool:
        """Create a community in a network.

        Args:
            network (str | int | Network): Network reference (address string, ID, or Network instance).
            name (str): Name of the community.
            description (str): Description of the community.
        """
        addr = self._resolve_network_address(network)
        resp = self._client.post(
            Endpoint.NetworkCommunities.with_params(addr),
            json={"name": name, "description": description},
        )
        return resp.is_success if resp else False

    def delete(self, community: int | str | Community, network: str | int | Network) -> None:
        """Delete a community from a network.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address string, ID, or Network instance).
        """
        addr = self._resolve_network_address(network)
        community_id = self._resolve_community_id(community, network)
        self._client.delete(Endpoint.NetworkCommunity.with_params(addr, community_id))

    def get_hosts(self, community: int | str | Community, network: str | int | Network) -> list[Host]:
        """List all hosts in a community.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address string, ID, or Network instance).
        """
        addr = self._resolve_network_address(network)
        community_id = self._resolve_community_id(community, network)
        return self._client.get_typed(
            Endpoint.NetworkCommunityHosts.with_params(addr, community_id), list[Host]
        )

    # NOTE: Why add host by ID here?
    def add_host(
        self,
        community: int | str | Community,
        network: str | int | Network,
        host: int | str | Host,
        *,
        ipaddress: IP_AddressT | str | None = None,
    ) -> bool:
        """Add a host to a community.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address string, ID, or Network instance).
            host (int | str | Host): Host reference (ID or Host instance).
            ipaddress (IP_AddressT | str | None): Optional IP address to associate with the host
                in this community. Pass None to omit.
        """
        addr = self._resolve_network_address(network)
        community_id = self._resolve_community_id(community, network)
        host_id = resolve_host_id(host, self._client)
        data: dict[str, Any] = {"id": host_id}
        if ipaddress is not None:
            data["ipaddress"] = str(ipaddress)
        resp = self._client.post(Endpoint.NetworkCommunityHosts.with_params(addr, community_id), json=data)
        return resp.is_success if resp else False

    # NOTE: Why add host by ID here?
    def remove_host(
        self,
        community: int | str | Community,
        network: str | int | Network,
        host: int | str | Host,
    ) -> None:
        """Remove a host from a community.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address string, ID, or Network instance).
            host (int | str | Host): Host reference (ID or Host instance).
        """
        addr = self._resolve_network_address(network)
        community_id = self._resolve_community_id(community, network)
        host_id = resolve_host_id(host, self._client)
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

    # NOTE: expose as properties so we don't need to override __init__ and call `super()`
    @functools.cached_property
    def policy(self) -> NetworkPolicyManager:
        """Manager for network policies (``client.network.policy``)."""
        return NetworkPolicyManager(self._client)

    @functools.cached_property
    def communities(self) -> CommunityManager:
        """Manager for network communities (``client.network.communities``)."""
        return CommunityManager(self._client, self)

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
        """Get the network containing an IP address.

        Args:
            ip (str | IP_AddressT): The IP address to look up.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and no network is found.
        """
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
        """Create a network.

        Args:
            network (str): The network address in CIDR notation.
            description (str): Description of the network. Defaults to "".
            vlan (int | None | Unset): VLAN ID. Pass None to unset, omit to leave unchanged.
            dns_delegated (bool | Unset): Whether DNS is delegated. Omit to leave unchanged.
            category (str | Unset): Network category. Omit to leave unchanged.
            location (str | Unset): Network location. Omit to leave unchanged.
            frozen (bool | Unset): Whether the network is frozen. Omit to leave unchanged.
            reserved (int | Unset): Number of reserved addresses. Omit to leave unchanged.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
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

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
            description (str | Unset): New description. Omit to leave unchanged.
            vlan (int | None | Unset): New VLAN ID. Pass None to unset, omit to leave unchanged.
            dns_delegated (bool | Unset): Whether DNS is delegated. Omit to leave unchanged.
            category (str | Unset): New category. Omit to leave unchanged.
            location (str | Unset): New location. Omit to leave unchanged.
            frozen (bool | Unset): Whether the network is frozen. Omit to leave unchanged.
            reserved (int | Unset): Number of reserved addresses. Omit to leave unchanged.
            policy (int | None | Unset): Network policy ID. Pass None to unset, omit to leave unchanged.
            max_communities (int | None | Unset): Max communities. Pass None to unset, omit to leave unchanged.
        """  # noqa: E501
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
        """Return the first available IP address in the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return ipaddress.ip_address(
            self._client.get_typed(Endpoint.NetworksFirstUnused.with_params(net.network), str)
        )

    def get_random_available_ip(self, ref: str | int | Network) -> IP_AddressT:
        """Return a random available IP address in the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return ipaddress.ip_address(
            self._client.get_typed(Endpoint.NetworksRandomUnused.with_params(net.network), str)
        )

    def get_used_count(self, ref: str | int | Network) -> int:
        """Return the number of used IP addresses in the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return self._client.get_typed(Endpoint.NetworksUsedCount.with_params(net.network), int)

    def get_unused_count(self, ref: str | int | Network) -> int:
        """Return the number of unused IP addresses in the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return self._client.get_typed(Endpoint.NetworksUnusedCount.with_params(net.network), int)

    def get_used_list(self, ref: str | int | Network) -> list[IP_AddressT]:
        """Return the used IP addresses in the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return self._client.get_typed(Endpoint.NetworksUsedList.with_params(net.network), list[IP_AddressT])

    def get_unused_list(self, ref: str | int | Network) -> list[IP_AddressT]:
        """Return the unused IP addresses in the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return self._client.get_typed(Endpoint.NetworksUnusedList.with_params(net.network), list[IP_AddressT])

    def get_reserved_ips(self, ref: str | int | Network) -> list[IP_AddressT]:
        """Return the reserved IP addresses of the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksReservedList.with_params(net.network), list[IP_AddressT]
        )

    def get_used_host_list(self, ref: str | int | Network) -> dict[str, list[str]]:
        """Return a dict of used IP addresses to their associated hostnames.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksUsedHostList.with_params(net.network), dict[str, list[str]]
        )

    def get_ptroverride_host_list(self, ref: str | int | Network) -> dict[str, str]:
        """Return a dict of PTR override IPs to their associated hostnames.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksPTROverrideHostList.with_params(net.network), dict[str, str]
        )

    def get_ptr_overrides(self, ref: str | int | Network) -> list[IP_AddressT]:
        """Return IP addresses that have PTR overrides in the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
        """
        net = self._resolve_net(ref)
        return self._client.get_typed(
            Endpoint.NetworksPTROverrideList.with_params(net.network), list[IP_AddressT]
        )

    def add_excluded_range(self, ref: str | int | Network, start: str, end: str) -> None:
        """Add an excluded IP range to the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
            start (str): The start IP address of the excluded range.
            end (str): The end IP address of the excluded range.
        """
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
        """Remove an excluded IP range from the network.

        Args:
            ref (str | int | Network): Network reference (address string, numeric ID, or Network instance).
            start (str): The start IP address of the excluded range.
            end (str): The end IP address of the excluded range.
        """
        net = self._resolve_net(ref)
        exrange = next(
            (r for r in net.excluded_ranges if str(r.start_ip) == start and str(r.end_ip) == end),
            None,
        )
        if exrange is None:
            raise EntityNotFound(f"Excluded range {start} - {end} not found in {net.network!r}.")
        self._client.delete(Endpoint.NetworksRemoveExcludedRanges.with_params(net.network, exrange.id))

    def list_by_policy(self, policy: int | NetworkPolicy) -> list[Network]:
        """List networks that share the same policy as the given network.

        Alias for `policy.networks()`

        Args:
            policy (int | NetworkPolicy): NetworkPolicy instance or numeric ID.
        """
        # re-use NetworkPolicy.networks for this
        return self.policy.networks(policy)


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
        ipaddress: IP_AddressT | str,
        host: int | str | Host | None = None,
        macaddress: str | MacAddress | None = None,
        fetch_after_create: bool = True,
    ) -> IPAddress | None:
        """Create an IP address record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            ipaddress (IP_AddressT | str): The IP address to assign.
            macaddress (str | MacAddress | None): Optional MAC address to associate. Pass None to omit.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        data: dict[str, Any] = {"ipaddress": str(ipaddress)}
        if macaddress is not None:
            data["macaddress"] = str(macaddress)
        if host is not None:
            data["host"] = resolve_host_id(host, self._client)
        return self._create(data, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | IPAddress,
        *,
        ipaddress: IP_AddressT | str | Unset = UNSET,
        macaddress: str | MacAddress | None | Unset = UNSET,
        host: int | str | Host | Unset | None = UNSET,
    ) -> IPAddress:
        """Update an IP address record's mutable fields.

        Args:
            ref (int | IPAddress): IPAddress instance or numeric ID.
            ipaddress (IP_AddressT | str | Unset): New IP address. Omit to leave unchanged.
            macaddress (str | MacAddress | None | Unset): New MAC address. Pass None to unset,
                omit to leave unchanged.
            host (int | str | Host | Unset): Host to (dis)associate with IP. Omit to leave unchanged.
        """
        ip = self._resolve(ref)
        data: dict[str, Any] = {}
        if ipaddress is not UNSET:
            data["ipaddress"] = str(ipaddress)
        if macaddress is not UNSET:
            data["macaddress"] = str(macaddress) if macaddress is not None else ""
        if host is not UNSET:
            # TODO: decide if we can modify resolve_host_id to return int | None
            # on a general basis, or if it's only applicable here
            if host is not None:
                host_id = resolve_host_id(host, self._client)  # pyright: ignore[reportArgumentType]
            else:
                host_id = host
            data["host"] = host_id
        return self._patch(ip, data)

    def associate_mac(self, ref: int | IPAddress, mac: str | MacAddress, *, force: bool = False) -> IPAddress:
        """Associate a MAC address with an IP address.

        Args:
            ref (int | IPAddress): IPAddress instance or numeric ID.
            mac (str | MacAddress): The MAC address to associate.
            force (bool): When True, skip safety checks and overwrite an existing MAC.

        Raises:
            EntityAlreadyExists: If the IP already has a MAC and ``force`` is False.
        """
        ip = self._resolve(ref)
        if ip.macaddress and not force:
            raise EntityAlreadyExists(f"IP address {ip.ipaddress} already has MAC address {ip.macaddress}.")
        return self.update(ip, macaddress=mac)

    def disassociate_mac(self, ref: int | IPAddress) -> IPAddress:
        """Remove the MAC address from an IP address.

        Args:
            ref (int | IPAddress): IPAddress instance or numeric ID.
        """
        ip = self._resolve(ref)
        return self.update(ip, macaddress=None)

    def list_by_host(self, host: int | str | Host) -> list[IPAddress]:
        """List all IP address records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)

    def list_by_ip(self, ip: IP_AddressT | str) -> list[IPAddress]:
        """List all IP address records with a given IP address.

        Args:
            ip (IP_AddressT | str): The IP address to filter by.
        """
        return self._fetch_list_by_field("ipaddress", str(ip))

    def list_by_mac(self, mac: str | MacAddress) -> list[IPAddress]:
        """List all IP address records with a given MAC address.

        Args:
            mac (str | MacAddress): The MAC address to filter by.
        """
        addr = MacAddress.parse_or_raise(mac)
        return self._fetch_list_by_field("macaddress", str(addr))


class CNAMEManager(NamedResourceManager[CNAME]):
    """Operations on :class:`~mreg_api.models.CNAME` resources."""

    _url_identifier: ClassVar[str] = "name"

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
        host: int | str | Host,
        name: str | HostName,
        fetch_after_create: bool = True,
    ) -> CNAME | None:
        """Create a CNAME record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            name (str | HostName): The alias name for the CNAME.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create(
            {"host": str(host_id), "name": self._client.fqdn(name)},
            fetch_after_create=fetch_after_create,
        )

    def update(
        self,
        ref: int | CNAME,
        *,
        host: int | str | Host | Unset = UNSET,
        name: str | HostName | Unset = UNSET,
        ttl: int | None | Unset = UNSET,
    ) -> CNAME:
        """Update a CNAME record's mutable fields. Pass ``ttl=None`` to reset to default.

        Args:
            ref (int | CNAME): CNAME instance or numeric ID.
            host (int | str | Host | Unset): New host reference. Omit to leave unchanged.
            name (str | HostName | Unset): New alias name. Omit to leave unchanged.
            ttl (int | None | Unset): New TTL. Pass None to reset to default, omit to leave unchanged.
        """
        cname = self._resolve(ref)
        data: dict[str, Any] = {}
        if not isinstance(host, Unset):
            data["host"] = resolve_host_id(host, self._client)
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
        """Get a CNAME record by alias name.

        Args:
            name (str): The alias name to look up.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and the CNAME is not found.
        """
        obj = self._fetch_by_field("name", self._client.fqdn(name))
        if required and obj is None:
            raise EntityNotFound(f"CNAME {name!r} not found.")
        return obj

    def get_by_host_and_name(
        self, host: int | str | Host, name: str, *, required: bool = False
    ) -> CNAME | None:
        """Get a CNAME record matching both the host and alias name.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            name (str): The alias name to look up.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and the CNAME is not found.
        """
        host_id = resolve_host_id(host, self._client)
        fqdn = self._client.fqdn(name)

        cnames = self.list(name=fqdn, host=host_id)
        obj = next((c for c in cnames), None)
        if required and obj is None:
            raise EntityNotFound(f"CNAME {name!r} for host {host_id} not found.")
        return obj

    def list_by_host(self, host: int | str | Host) -> list[CNAME]:
        """List all CNAME records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
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
        host: int | str | Host,
        cpu: str,
        os: str,
        fetch_after_create: bool = True,
    ) -> HInfo | None:
        """Create an HInfo record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            cpu (str): CPU hardware type string.
            os (str): Operating system string.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create({"host": host_id, "cpu": cpu, "os": os}, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | HInfo,
        *,
        cpu: str | Unset = UNSET,
        os: str | Unset = UNSET,
    ) -> HInfo:
        """Update an HInfo record's mutable fields.

        Args:
            ref (int | HInfo): HInfo instance or numeric ID.
            cpu (str | Unset): New CPU hardware type string. Omit to leave unchanged.
            os (str | Unset): New operating system string. Omit to leave unchanged.
        """
        hinfo = self._resolve(ref)
        data: dict[str, Any] = {}
        if cpu is not UNSET:
            data["cpu"] = cpu
        if os is not UNSET:
            data["os"] = os
        return self._patch(hinfo, data)

    def get_by_host(self, host: int | str | Host, *, required: bool = False) -> HInfo | None:
        """Get the HInfo record for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and no HInfo record is found.
        """
        host_id = resolve_host_id(host, self._client)
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
        host: int | str | Host,
        txt: str,
        fetch_after_create: bool = True,
    ) -> TXT | None:
        """Create a TXT record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            txt (str): The TXT record value.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create({"host": host_id, "txt": txt}, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | TXT,
        *,
        txt: str | Unset = UNSET,
    ) -> TXT:
        """Update a TXT record's mutable fields.

        Args:
            ref (int | TXT): TXT instance or numeric ID.
            txt (str | Unset): New TXT record value. Omit to leave unchanged.
        """
        txt_obj = self._resolve(ref)
        data: dict[str, Any] = {}
        if txt is not UNSET:
            data["txt"] = txt
        return self._patch(txt_obj, data)

    def list_by_host(self, host: int | str | Host) -> list[TXT]:
        """List all TXT records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
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
        host: int | str | Host,
        mx: str,
        priority: int,
        fetch_after_create: bool = True,
    ) -> MX | None:
        """Create an MX record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            mx (str): The mail exchange hostname.
            priority (int): The MX priority value.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
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
        """Update an MX record's mutable fields.

        Args:
            ref (int | MX): MX instance or numeric ID.
            mx (str | Unset): New mail exchange hostname. Omit to leave unchanged.
            priority (int | Unset): New priority value. Omit to leave unchanged.
        """
        mx_obj = self._resolve(ref)
        data: dict[str, Any] = {}
        if mx is not UNSET:
            data["mx"] = mx
        if priority is not UNSET:
            data["priority"] = priority
        return self._patch(mx_obj, data)

    def list_by_host(self, host: int | str | Host) -> list[MX]:
        """List all MX records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)

    def get_by_all(self, host: int | str | Host, mx: str, priority: int) -> MX:
        """Get an MX record matching host, mx value, and priority.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            mx (str): The mail exchange hostname.
            priority (int): The MX priority value.

        Raises:
            EntityNotFound: If no matching MX record exists.
        """
        host_id = resolve_host_id(host, self._client)
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
        host: int | str | Host,
        preference: int,
        order: int,
        flag: str = "",
        service: str = "",
        regex: str = "",
        replacement: str,
        fetch_after_create: bool = True,
    ) -> NAPTR | None:
        """Create a NAPTR record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            preference (int): The NAPTR preference value.
            order (int): The NAPTR order value.
            flag (str): The NAPTR flag. Defaults to "".
            service (str): The NAPTR service. Defaults to "".
            regex (str): The NAPTR regular expression. Defaults to "".
            replacement (str): The NAPTR replacement string.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
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
        """Update a NAPTR record's mutable fields.

        Args:
            ref (int | NAPTR): NAPTR instance or numeric ID.
            preference (int | Unset): New preference value. Omit to leave unchanged.
            order (int | Unset): New order value. Omit to leave unchanged.
            flag (str | Unset): New flag. Omit to leave unchanged.
            service (str | Unset): New service. Omit to leave unchanged.
            regex (str | Unset): New regular expression. Omit to leave unchanged.
            replacement (str | Unset): New replacement string. Omit to leave unchanged.
        """
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

    def list_by_host(self, host: int | str | Host) -> list[NAPTR]:
        """List all NAPTR records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
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
        host: int | str | Host,
        name: str,
        priority: int,
        weight: int,
        port: int,
        ttl: int | None | Unset = UNSET,
        fetch_after_create: bool = True,
    ) -> Srv | None:
        """Create a SRV record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            name (str): The SRV service name.
            priority (int): The SRV priority value.
            weight (int): The SRV weight value.
            port (int): The SRV port number.
            ttl (int | None | Unset): TTL. Pass None to use default, omit to leave unchanged.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
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
        """Update a SRV record's mutable fields. Pass ``ttl=None`` to reset to default.

        Args:
            ref (int | Srv): Srv instance or numeric ID.
            name (str | Unset): New service name. Omit to leave unchanged.
            priority (int | Unset): New priority value. Omit to leave unchanged.
            weight (int | Unset): New weight value. Omit to leave unchanged.
            port (int | Unset): New port number. Omit to leave unchanged.
            ttl (int | None | Unset): New TTL. Pass None to reset to default, omit to leave unchanged.
        """
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

    def list_by_host(self, host: int | str | Host) -> list[Srv]:
        """List all SRV records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
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
        host: int | str | Host,
        ipaddress: IP_AddressT | str,
        fetch_after_create: bool = True,
    ) -> PTR_override | None:
        """Create a PTR override record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            ipaddress (IP_AddressT | str): The IP address for the PTR override.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create(
            {"host": host_id, "ipaddress": str(ipaddress)},
            fetch_after_create=fetch_after_create,
        )

    def update(
        self,
        ref: int | PTR_override,
        *,
        host: int | str | Host | Unset = UNSET,
        ipaddress: IP_AddressT | str | Unset = UNSET,
    ) -> PTR_override:
        """Update a PTR override record's mutable fields.

        Args:
            ref (int | PTR_override): PTR_override instance or numeric ID.
            host (int | str | Host | Unset): New host reference. Omit to leave unchanged.
            ipaddress (IP_AddressT | str | Unset): New IP address. Omit to leave unchanged.
        """
        ptr = self._resolve(ref)
        data: dict[str, Any] = {}
        if not isinstance(host, Unset):
            data["host"] = resolve_host_id(host, self._client)
        if ipaddress is not UNSET:
            data["ipaddress"] = str(ipaddress)
        return self._patch(ptr, data)

    def list_by_host(self, host: int | str | Host) -> list[PTR_override]:
        """List all PTR override records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
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
        host: int | str | Host,
        algorithm: int,
        hash_type: int,
        fingerprint: str,
        ttl: int | None | Unset = UNSET,
        fetch_after_create: bool = True,
    ) -> SSHFP | None:
        """Create an SSHFP record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            algorithm (int): The SSHFP algorithm number.
            hash_type (int): The SSHFP hash type number.
            fingerprint (str): The SSH key fingerprint.
            ttl (int | None | Unset): TTL. Pass None to use default, omit to leave unchanged.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
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
        """Update an SSHFP record's mutable fields. Pass ``ttl=None`` to reset to default.

        Args:
            ref (int | SSHFP): SSHFP instance or numeric ID.
            algorithm (int | Unset): New algorithm number. Omit to leave unchanged.
            hash_type (int | Unset): New hash type number. Omit to leave unchanged.
            fingerprint (str | Unset): New fingerprint. Omit to leave unchanged.
            ttl (int | None | Unset): New TTL. Pass None to reset to default, omit to leave unchanged.
        """
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

    def list_by_host(self, host: int | str | Host) -> list[SSHFP]:
        """List all SSHFP records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
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
        host: str | HostName | Host,
        id: int,  # noqa: A002
        fetch_after_create: bool = True,
    ) -> BacnetID | None:
        """Create a BacnetID record.

        Args:
            host: The host to create a BacnetID for.
            id: The BACnet id.
            fetch_after_create: Whether to fetch and return the created object.
        """
        return self._create(
            {"hostname": resolve_host_name(host, self._client), "id": id},
            fetch_after_create=fetch_after_create,
        )

    # NOTE: PATCH and PUT not allowed for this endpoint!

    def list_in_range(self, start: int, end: int) -> list[BacnetID]:
        """List BacnetID records within a numeric id range (inclusive).

        Args:
            start (int): The start of the BACnet ID range (inclusive).
            end (int): The end of the BACnet ID range (inclusive).
        """
        return self._client.get_typed(self.endpoint, list[BacnetID], params={"id__range": f"{start},{end}"})

    def get_by_host(self, host: str | HostName | Host, *, required: bool = False) -> BacnetID | None:
        """Get the BacnetID record for a host by its FQDN.

        Args:
            host (str | HostName | Host): Host reference (name string or Host instance).
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and no BacnetID record is found.
        """
        name = resolve_host_name(host, self._client)
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
        host: int | str | Host,
        loc: str,
        fetch_after_create: bool = True,
    ) -> Location | None:
        """Create a LOC record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            loc (str): The LOC record value.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create({"host": host_id, "loc": loc}, fetch_after_create=fetch_after_create)

    def update(
        self,
        ref: int | Location,
        *,
        loc: str | Unset = UNSET,
    ) -> Location:
        """Update a LOC record's mutable fields.

        Args:
            ref (int | Location): Location instance or numeric ID.
            loc (str | Unset): New LOC record value. Omit to leave unchanged.
        """
        loc_obj = self._resolve(ref)
        data: dict[str, Any] = {}
        if loc is not UNSET:
            data["loc"] = loc
        return self._patch(loc_obj, data)

    def get_by_host(self, host: int | str | Host, *, required: bool = False) -> Location | None:
        """Get the LOC record for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and no LOC record is found.
        """
        host_id = resolve_host_id(host, self._client)
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


def _verify_nameservers(client: MregClient, nameservers: list[str], force: bool = False) -> list[str]:
    """Verify nameservers exist in mreg and have an A-record / glue.

    Utility function shared by various managers that interact with nameservers.

    Raises:
        InputFailure: If no nameservers are given.
        ForceMissing: If a nameserver is missing (or lacks glue) and ``force`` is False.
    """
    if not nameservers:
        raise InputFailure("At least one nameserver is required")

    errors: list[str] = []
    verified: list[str] = []  # NOTE: should be list[HostName], but invariant, etc.. cba.
    for nameserver in nameservers:
        # HACK: bypass checks if force is enabled, return name as-is, expanded and validated
        if force:
            verified.append(client.fqdn(nameserver))  # HACK
            continue

        host = client.host.get_by_name(nameserver, required=False)
        if host is None:
            cname = client.cname.get_by_name(nameserver, required=False)
            if cname is not None:
                host = client.host.get_by_id(cname.host, required=False)
                if host is not None:
                    client.events.record(
                        Event(
                            kind=EventKind.RESOLUTION,
                            message=f"{nameserver} is a CNAME for {host.name}",
                            subject=ObjectRef.new(host),
                            related=(ObjectRef.new(cname),),
                            correlation_id=client.get_correlation_id(),
                        )
                    )

        if host is None:
            if not force:
                errors.append(f"{nameserver} is not in mreg, must force")
        else:
            if host.zone is None and not host.ipaddresses and not force:
                errors.append(f"{nameserver} has no A-record/glue, must force")

    if errors:
        raise ForceMissing("\n".join(errors))

    return verified


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
        """Create a zone of this manager's type. Caller verifies nameservers/absence.

        Args:
            name (str): The zone name.
            email (str): The zone admin email address.
            primary_ns (list[str]): List of primary nameserver names.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        # FIXME: if called directly, name servers do not get passed through client.fqdn
        # This could lead to bugs in the future.
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
        """Update the zone's SOA fields. At least one field must be provided.

        Args:
            zone (_ZoneT): The zone to update.
            primary_ns (str | Unset): New primary nameserver. Omit to leave unchanged.
            email (str | Unset): New zone admin email. Omit to leave unchanged.
            serialno (int | Unset): New serial number. Omit to leave unchanged.
            refresh (int | Unset): New refresh interval. Omit to leave unchanged.
            retry (int | Unset): New retry interval. Omit to leave unchanged.
            expire (int | Unset): New expire interval. Omit to leave unchanged.
            soa_ttl (int | Unset): New SOA TTL. Omit to leave unchanged.
        """
        data: dict[str, Any] = {}
        if primary_ns is not UNSET:
            # FIXME: if called directly, name servers do not get passed through client.fqdn
            # This could lead to bugs in the future. If called through ZoneManager,
            # name servers are verified and validated/formatted.
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
        """Set the zone's default TTL.

        Args:
            zone (_ZoneT): The zone to update.
            ttl (int): The new default TTL value (300–68400).
        """
        return self._patch(zone, {"default_ttl": _valid_zone_ttl(ttl)})

    def set_nameservers(self, zone: _ZoneT, nameservers: list[str], *, force: bool = False) -> None:
        """Replace the zone's nameservers (hits the per-type nameservers endpoint).

        Args:
            zone (_ZoneT): The zone to update.
            nameservers (list[str]): The new list of nameserver names.
            force (bool): When True, skip safety checks on nameserver existence.
        """
        nameservers = _verify_nameservers(self._client, nameservers, force=force)
        self._client.patch(self.nameservers_endpoint.with_params(zone.name), json={"primary_ns": nameservers})

    def list_subzones(self, zone: _ZoneT) -> list[_ZoneT]:
        """List subzones of the zone (excluding the zone itself).

        Args:
            zone (_ZoneT): The parent zone to list subzones for.
        """
        zones = self._fetch_list_by_field("name__endswith", f".{zone.name}")
        return [z for z in zones if z.name != zone.name]

    def _ensure_deletable(self, zone: _ZoneT) -> None:
        """Raise if the zone has registered entries or subzones."""
        # XXX: Not foolproof (e.g. SRVs are not hosts), parity with old Zone.ensure_deletable.
        hosts = self._client.host.list(zone=zone.id)
        if hosts:
            raise DeleteError(f"Zone has {len(hosts)} registered entries. Can not delete.")
        subzones = self.list_subzones(zone)
        if subzones:
            names = ", ".join(z.name for z in subzones)
            raise DeleteError(f"Zone has registered subzones: '{names}'. Can not delete")

    @override
    def delete(self, obj: _ZoneT, *, force: bool = False) -> None:
        """Delete the zone, guarding against non-empty zones unless ``force``.

        Args:
            obj (_ZoneT): The zone to delete.
            force (bool): When True, skip safety checks and delete even non-empty zones.
        """
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

    def get_from_host(self, host: str | HostName | Host) -> ForwardZoneDelegation | ForwardZone | None:
        """Get the forward zone (or delegation) responsible for a host or hostname.

        May return a :class:`ForwardZoneDelegation` when the hostname falls under a
        delegated subzone.

        Args:
            host (str | HostName | Host): Host reference (name string or Host instance).
        """
        name = resolve_host_name(host, self._client)
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
    """Public facade over the forward/reverse zone managers (``client.zone``).

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

    def verify_nameservers(self, nameservers: list[str], force: bool = False) -> list[str]:
        """Verify nameservers exist in mreg and have glue (raises otherwise).

        Args:
            nameservers (list[str]): List of nameserver names to verify.
            force (bool): When True, skip safety checks on nameserver existence.
        """
        return _verify_nameservers(self._client, nameservers, force=force)

    @overload
    def get_by_name(self, name: str, *, required: Literal[True]) -> ForwardZone | ReverseZone: ...
    @overload
    def get_by_name(
        self, name: str, *, required: Literal[False] = ...
    ) -> ForwardZone | ReverseZone | None: ...
    def get_by_name(self, name: str, *, required: bool = False) -> ForwardZone | ReverseZone | None:
        """Get a zone by name; forward/reverse chosen by name shape.

        Args:
            name (str): The zone name to look up.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and the zone is not found.
        """
        return self._sub_for_name(name).get_by_name(name, required=required)

    def ensure_absent(self, name: str) -> None:
        """Raise EntityAlreadyExists if a zone with ``name`` exists.

        Args:
            name (str): The zone name to check.
        """
        self._sub_for_name(name).ensure_absent(name)

    def list_forward(self) -> list[ForwardZone]:
        """List forward zones."""
        return self._forward.list()

    def list_reverse(self) -> list[ReverseZone]:
        """List reverse zones."""
        return self._reverse.list()

    def get_from_host(self, host: str | HostName | Host) -> ForwardZoneDelegation | ForwardZone | None:
        """Get the forward zone (or delegation) responsible for a host or hostname.

        Args:
            host (str | HostName | Host): Host reference (name string or Host instance).
        """
        return self._forward.get_from_host(host)

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

        Args:
            name (str): The zone name (determines forward vs. reverse by shape).
            email (str): The zone admin email address.
            primary_ns (list[str]): List of primary nameserver names.
            force (bool): When True, skip safety checks on nameservers.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        verified_ns = self.verify_nameservers(primary_ns, force=force)
        sub = self._sub_for_name(name)
        sub.ensure_absent(name)
        return sub.create(
            name=name, email=email, primary_ns=verified_ns, fetch_after_create=fetch_after_create
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
        """Update the zone's SOA fields.

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
            primary_ns (str | Unset): New primary nameserver. Omit to leave unchanged.
            email (str | Unset): New zone admin email. Omit to leave unchanged.
            serialno (int | Unset): New serial number. Omit to leave unchanged.
            refresh (int | Unset): New refresh interval. Omit to leave unchanged.
            retry (int | Unset): New retry interval. Omit to leave unchanged.
            expire (int | Unset): New expire interval. Omit to leave unchanged.
            soa_ttl (int | Unset): New SOA TTL. Omit to leave unchanged.
        """
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

    def set_default_ttl(self, zone: str | ForwardZone | ReverseZone, ttl: int) -> ForwardZone | ReverseZone:
        """Set the zone's default TTL.

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
            ttl (int): The new default TTL value (300–68400).
        """
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            return self._reverse.set_default_ttl(z, ttl)
        return self._forward.set_default_ttl(z, ttl)

    def set_nameservers(
        self, zone: str | ForwardZone | ReverseZone, nameservers: list[str], *, force: bool = False
    ) -> None:
        """Replace the zone's nameservers.

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
            nameservers (list[str]): The new list of nameserver names.
            force (bool): When True, skip safety checks on nameserver existence.
        """
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            self._reverse.set_nameservers(z, nameservers, force=force)
        else:
            self._forward.set_nameservers(z, nameservers, force=force)

    def list_subzones(self, zone: str | ForwardZone | ReverseZone) -> list[ForwardZone] | list[ReverseZone]:
        """List subzones of the zone (excluding the zone itself).

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
        """
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            return self._reverse.list_subzones(z)
        return self._forward.list_subzones(z)

    def delete(self, zone: str | ForwardZone | ReverseZone, *, force: bool = False) -> None:
        """Delete the zone, guarding against non-empty zones unless ``force``.

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
            force (bool): When True, skip safety checks and delete even non-empty zones.
        """
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            self._reverse.delete(z, force=force)
        else:
            self._forward.delete(z, force=force)

    def zone_file(self, zone_name: str, *, exclude_private: bool = False) -> str | None:
        """Return the zone file content for the named zone, or None if not found.

        Args:
            zone_name (str): The name of the zone to fetch the file for.
            exclude_private (bool): When True, exclude private records from the zone file.
        """
        params: QueryParams = {}
        if exclude_private:
            params["excludePrivate"] = "1"
        resp = self._client.get(Endpoint.Zonefiles.with_id(zone_name), params=params, ok404=True)
        if not resp:
            return None
        return ZoneFile.model_validate(resp.text).root

    @functools.cached_property
    def delegations(self) -> DelegationManager:
        """Manager for zone delegations (``client.zone.delegations``)."""
        return DelegationManager(self._client)


class DelegationManager:
    """Operations on zone delegations (``client.delegation``).

    Delegations have no standalone endpoint; their type (forward/reverse) is derived
    from the parent zone, so every method takes the parent zone as its first argument.
    Kept separate from :class:`ZoneManager` to stay composition-ready (a future
    ``client.zone.delegations``). See ADR-0007.
    """

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client."""
        self._client: MregClient = client

    def _model_for(self, zone: Zone) -> type[ForwardZoneDelegation | ReverseZoneDelegation]:
        return ReverseZoneDelegation if zone.is_reverse() else ForwardZoneDelegation

    def _endpoint_for(self, zone: Zone) -> Endpoint:
        """Return the delegations endpoint for the given zone's type."""
        return Endpoint.ReverseZonesDelegations if zone.is_reverse() else Endpoint.ForwardZonesDelegations

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
        """Get a delegation in ``zone`` by name.

        Args:
            zone (Zone): The parent zone to search in.
            name (str): The delegation name to look up.
            required (bool): When True, raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If ``required`` is True and the delegation is not found.
        """
        self._ensure_in_zone(zone, name)
        cls = self._model_for(zone)
        resp = self._client.get(cls.endpoint_with_name(zone, name), ok404=True)
        if not resp:
            if required:
                raise EntityNotFound(f"Could not find delegation {name!r} in zone {zone.name!r}")
            return None
        return cls.model_validate_json(resp.text)

    def list_by_zone(self, zone: Zone) -> list[ForwardZoneDelegation | ReverseZoneDelegation]:
        """List all delegations for a zone.

        Args:
            zone (Zone): The parent zone to list delegations for.
        """
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

        Args:
            zone (Zone): The parent zone to create the delegation in.
            name (str): The delegation name (must be within the parent zone).
            nameservers (list[str]): List of nameserver names for the delegation.
            comment (str): Optional comment for the delegation. Defaults to "".
            force (bool): When True, skip safety checks.
            fetch_after_create (bool): Whether to fetch and return the created object.
        """
        self._ensure_in_zone(zone, name)
        nameservers = _verify_nameservers(self._client, nameservers, force=force)

        if not force:
            delegated = self._client.zone.get_by_name(name)
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
        """Delete a delegation from ``zone``.

        Args:
            zone (Zone): The parent zone to delete the delegation from.
            name (str): The delegation name to delete.
        """
        self._ensure_in_zone(zone, name)
        _ = self.get(zone, name, required=True)
        cls = self._model_for(zone)
        self._client.delete(cls.endpoint_with_name(zone, name))

    def set_comment(
        self, zone: Zone, name: str, comment: str
    ) -> ForwardZoneDelegation | ReverseZoneDelegation:
        """Set (or clear, with ``""``) the comment for a delegation.

        Args:
            zone (Zone): The parent zone containing the delegation.
            name (str): The delegation name.
            comment (str): The new comment. Pass "" to clear.
        """
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
        """List DHCP hosts within the given IP range.

        Args:
            ip (str | IP_AddressT): The start IP address of the range.
            range (str): The range specifier.
        """
        return self._client.get_typed(
            Endpoint.DhcpHostsByRange.with_params(str(ip), range),
            list[self.model],
        )


class DhcpHostIPv4Manager(DhcpHostManager[DhcpHostIPv4]):
    """Read-only manager for IPv4 DHCP host records (``client.dhcphostipv4``)."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv4]:
        return DhcpHostIPv4

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv4


class DhcpHostIPv6Manager(DhcpHostManager[DhcpHostIPv6]):
    """Read-only manager for IPv6 DHCP host records (``client.dhcphostipv6``)."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv6]:
        return DhcpHostIPv6

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv6


class DhcpHostIPv6ByIPv4Manager(DhcpHostManager[DhcpHostIPv6ByIPv4]):
    """Read-only manager for IPv6-via-IPv4 DHCP host records (``client.dhcphostipv6byipv4``)."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv6ByIPv4]:
        return DhcpHostIPv6ByIPv4

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv6ByIpv4


class NameServerManager(NamedResourceManager[NameServer]):
    """Access to :class:`~mreg_api.models.NameServer` resources (``client.nameserver``).

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


class MetaManagerNamespace:
    """Class that composes managers for the various /api/meta endpoints."""

    def __init__(self, client: MregClient) -> None:
        """Initialize the namespace with a client instance."""
        self._client = client
        self.version = ServerVersionManager(client)
        self.libraries = ServerLibrariesManager(client)
        self.userinfo = UserInfoManager(client)
        self.ldap = LDAPHealthManager(client)
        self.heartbeat = HeartbeatHealthManager(client)
        self.health = HealthManager(
            client,
            heartbeat_manager=self.heartbeat,
            ldap_manager=self.ldap,
        )


class ServerVersionManager:
    """Access to server version metadata (``client.serverversion``)."""

    def __init__(self, client: MregClient) -> None:
        """Initialize the manager with a client instance."""
        self._client = client

    def get(self, *, required: bool = False) -> ServerVersion:
        """Fetch the server version from the meta endpoint.

        Args:
            required: When ``True``, raise on error. When ``False``, return
                ``ServerVersion(version="Unknown")`` on failure.
        """
        try:
            response = self._client.get(Endpoint.MetaVersion)
            return ServerVersion.model_validate(response.json())
        except Exception:
            if required:
                raise
            return ServerVersion(version="Unknown")


class ServerLibrariesManager:
    """Access to server library metadata (``client.serverlibraries``)."""

    def __init__(self, client: MregClient) -> None:
        """Initialize the manager with a client instance."""
        self._client = client

    def get(self, *, required: bool = False) -> ServerLibraries:
        """Fetch the server library list from the meta endpoint.

        Args:
            required: When ``True``, raise on error. When ``False``, return
                ``ServerLibraries(libraries=[])`` on failure.
        """
        try:
            response = self._client.get_typed(Endpoint.MetaLibraries, dict[str, str])
            libraries = [Library(name=name, version=version) for name, version in response.items()]
            return ServerLibraries(libraries=libraries)
        except Exception:
            if required:
                raise
            return ServerLibraries(libraries=[])


class UserInfoManager:
    """Access to user information (``client.userinfo``)."""

    def __init__(self, client: MregClient) -> None:
        """Initialize the manager with a client instance."""
        self._client = client

    def get(self, *, required: bool = False, user: str | None = None) -> UserInfo:
        """Fetch user information from the meta endpoint.

        Args:
            required: When ``True``, raise on error. When ``False``, return a
                zeroed-out ``UserInfo`` on failure.
            user: The username to fetch. If ``None``, fetches the current user.
        """
        try:
            endpoint: str = Endpoint.MetaUser
            if user:
                endpoint = f"{endpoint}?username={user}"
            response = self._client.get(endpoint)
            return UserInfo.model_validate(response.json())
        except Exception:
            if required:
                raise
            return UserInfo(
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


class LDAPHealthManager:
    """Access to LDAP health status (``client.ldaphealth``)."""

    def __init__(self, client: MregClient) -> None:
        """Initialize the manager with a client instance."""
        self._client = client

    def get(self, *, required: bool = False) -> LDAPHealth:
        """Fetch LDAP health from the health endpoint.

        A 503 response means LDAP is down and is not treated as a hard error.

        Args:
            required: When ``True``, raise on non-503 errors.
        """
        try:
            self._client.get(Endpoint.HealthLDAP)
            return LDAPHealth(status="OK")
        except GetError as e:
            if required:
                raise
            logger.error("Failed to fetch LDAP health: %s", e)
            if e.response and e.response.status_code == 503:
                return LDAPHealth(status="Down")
            return LDAPHealth(status="Unknown")


class HeartbeatHealthManager:
    """Access to heartbeat health status (``client.heartbeathealth``)."""

    def __init__(self, client: MregClient) -> None:
        """Initialize the manager with a client instance."""
        self._client = client

    def get(self, *, required: bool = False) -> HeartbeatHealth:
        """Fetch heartbeat health from the health endpoint.

        Args:
            required: When ``True``, raise on error. When ``False``, return
                ``HeartbeatHealth(uptime=-1, start_time=0)`` on failure.
        """
        try:
            result = self._client.get(Endpoint.HealthHeartbeat)
            return HeartbeatHealth.model_validate_json(result.text)
        except Exception as e:
            if required:
                raise
            logger.error("Failed to fetch heartbeat: %s", e)
            return HeartbeatHealth(uptime=-1, start_time=0)


class HealthManager:
    """Access to combined health information (``client.health``)."""

    def __init__(
        self,
        client: MregClient,
        heartbeat_manager: HeartbeatHealthManager | None = None,
        ldap_manager: LDAPHealthManager | None = None,
    ) -> None:
        """Initialize the manager with a client instance.

        Optionally takes in existing heartbeat and LDAP managers to avoid creating new ones.
        """
        self._client = client
        if heartbeat_manager is None:
            heartbeat_manager = HeartbeatHealthManager(client)
        self._heartbeat_manager = heartbeat_manager

        if ldap_manager is None:
            ldap_manager = LDAPHealthManager(client)
        self._ldap_manager = ldap_manager

    def get(self, *, required: bool = False) -> HealthInfo:
        """Fetch combined health from all health endpoints.

        Args:
            required: Forwarded to both sub-managers. When ``True``, raises
                on any failure instead of returning a default object.
        """
        heartbeat = self._heartbeat_manager.get(required=required)
        ldap = self._ldap_manager.get(required=required)
        return HealthInfo(heartbeat=heartbeat, ldap=ldap)
