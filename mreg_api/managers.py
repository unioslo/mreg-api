"""Resource managers: encapsulated access to API resources.

Each manager operates on a specific model type and exposes a set of operations.
Managers are bound to an `MregClient` instance.

As a baseline, each manager exposes:
- `get`
- `list`
- `assert_absent`  (NOTE: may be removed in a future version)

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

from httpx import Response
from pydantic import BaseModel
from pydantic import ValidationError
from typing_extensions import Sentinel
from typing_extensions import deprecated
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
from mreg_api.exceptions import PostError
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
from mreg_api.models import UserInfo
from mreg_api.models import Zone
from mreg_api.models import ZoneFile
from mreg_api.models.fields import HostName
from mreg_api.models.fields import MacAddress
from mreg_api.models.fields import VerifiedNS
from mreg_api.models.history import HistoryItem
from mreg_api.models.history import HistoryResource
from mreg_api.models.models import is_reverse_zone_name
from mreg_api.types import IP_AddressT
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter
from mreg_api.utilities.shared import convert_wildcard_to_regex

if TYPE_CHECKING:
    from mreg_api.client import MregClient
    from mreg_api.types import IP_NetworkT

logger = logging.getLogger(__name__)

# NOTE: Using the capitalized `Sentinel` name here (deprecated) instead of
# `sentinel`, because not all type checkers recognize the lowercase version.
UNSET = Sentinel("UNSET")


T = TypeVar("T", bound=BaseModel)


# ---- Various functions shared between managers that do not necessarily share the same base ---


def model_name(model: type[object] | object) -> str:
    """Get the name of the model for a manager."""
    if isinstance(model, type):
        return model.__name__
    return model.__class__.__name__


def validate_response(model: type[T], response: Response) -> T:
    """Attempt to construct the manager's model type from an HTTP response.

    Args:
        model (type[T]): The model class to construct.
        response (Response): The HTTP response object containing JSON data.

    Returns:
        T: An instance of the manager's model type.
    """
    return validate_json(model, response.text)


def validate_json(model: type[T], data: str) -> T:
    """Attempt to construct the manager's model type from JSON data.

    Args:
        model (type[T]): The model class to construct.
        data (str): JSON data to construct model with.

    Returns:
        T: An instance of the manager's model type.
    """
    return get_type_adapter(model).validate_json(data)


def validate(model: type[T], data: Any) -> T:
    """Attempt to construct the manager's model type from data.

    Args:
        model (type[T]): The model class to construct.
        data (Any): Data to construct model with. Usually a dict.

    Returns:
        T: An instance of the manager's model type.
    """
    return get_type_adapter(model).validate_python(data)


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

    _path_param_field: ClassVar[str] = "id"
    """Name of the model field whose value is the resource's URL path parameter.

    The path parameter is the variable trailing segment of a detail URL
    (`/hosts/{name}`, `/networks/{network}`, `/sshfps/{id}`) that identifies a
    single resource.

    For most resources this is the numeric `id` field:
        GET /api/sshfps/123 # 200

    But for other resources it is a different field (e.g. name, network, host):
        GET /api/hosts/example.com # 200
        GET /api/hosts/123 # 404

        GET /api/v1/networks/192.168.0.0/24 # 200
        GET /api/v1/networks/123 # 404
    """

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client that owns it."""
        self._client: MregClient = client

    def _normalize_path_param(self, ident: str | int) -> str | int:
        """Normalise a path parameter value before use in lookups.

        Can be overriden by subclasses to define custom normalization behavior
        for path parameter values passed to `get()`.
        """
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

    @property
    def model_name(self) -> str:
        """The canonical name of the manager's model."""
        return model_name(self.model)

    def _fetch(self, ref: str | int) -> T | None:
        """Fetch by a resource reference (ID or string identifier).

        Depending on the endpoint, the reference may be a numeric ID, or a string
        such as name, network, and more.

        A resource is only directly fetchable by an integer value if its path
        parameter field is named `id`, otherwise it must be searched for.

        Strings are always treated as the path parameter, and use the manager's
        endpoint to construct the direct lookup, i.e.:

        `_fetch("myhost")` -> `/hosts/{name}` -> `/hosts/myhost`
        """
        if isinstance(ref, int) and self._path_param_field != "id":
            return self._search_one("id", ref)
        return self._fetch_by_path(ref)

    def _resolve(self, ref: int | str | T) -> T:
        """Resolve an object reference (instance, path parameter, or numeric id) to a model.

        Fetches the object from the server if only an identifier is provided.

        Raises:
            EntityNotFound: If the object cannot be found.
        """
        if isinstance(ref, self.model):
            return ref
        elif isinstance(ref, BaseModel):  # MregModel but not the manager's bound type
            # Technically, type checker should prevent this class of errors, but
            # but in cases where the wrong model type is passed to this method
            # we have this runtime check to raise a more helpful error message than
            # some arcane error stemming from passing a model instance to `_fetch()`
            raise TypeError(f"{self.model_name} cannot resolve {model_name(ref)} instances.")

        # ref narrowed to int | str
        obj = self._fetch(ref)
        if obj is None:
            raise EntityNotFound(f"{model_name(self.model)} with {self._path_param_field} {ref!r} not found.")
        return obj

    def _resolve_hostname(self, host: str | HostName | Host) -> str:
        """Resolve a host reference to its fully qualified name.

        Args:
            host: A host reference, which can be a string, HostName, or Host instance.

        Returns:
            The fully qualified name of the host as a string.
        """
        return resolve_host_name(host, self._client)

    # NOTE: Refactor to function that takes in client obj?
    def _resolve_label_id(self, label: int | str | Label) -> int:
        """Resolve a label reference to its numeric ID."""
        if isinstance(label, int):
            return label
        if isinstance(label, Label):
            return label.id
        return LabelManager(self._client).get_by_name(str(label), required=True).id

    def _validate_response(self, response: Response) -> T:
        """Attempt to construct the manager's model type from an HTTP response.

        Args:
            response (Response): The HTTP response object containing JSON data.

        Returns:
            T: An instance of the manager's model type.
        """
        return validate_response(self.model, response)

    def _validate_json(self, data: str) -> T:
        """Attempt to construct the manager's model type from JSON data.

        Args:
            data (str): JSON data to construct model with.

        Returns:
            T: An instance of the manager's model type.
        """
        return validate_json(self.model, data)

    def _validate(self, data: JsonMapping) -> T:
        """Attempt to construct the manager's model type from a JSON mapping.

        Args:
            data (JsonMapping): Data to construct model with. Usually a dict.

        Returns:
            T: An instance of the manager's model type.
        """
        return validate(self.model, data)

    # NOTE: does this need to be a method that takes an input?
    # Yes, it allows us to more easily test it, but its type is `T`
    # which means in practice we only operate on the manager's declared type.
    def _path_param_value(self, obj: T) -> str | int:
        """Return the value of `obj`'s path parameter field (its URL path segment).

        Raises:
            InternalError: If `obj` has no attribute named by `_path_param_field`.
                This indicates a misconfigured manager rather than bad user input.
        """
        try:
            return getattr(obj, self._path_param_field)
        except AttributeError as e:
            raise InternalError(
                f"{model_name(obj)} has no path parameter field {self._path_param_field!r}."
            ) from e

    def _endpoint_with_path_param(self, obj: T) -> str:
        """Build the detail URL for `obj` from its path parameter value."""
        return self.endpoint.with_id(self._path_param_value(obj))

    def _fetch_by_path(self, value: str | int) -> T | None:
        """Fetch one object via its detail URL: `GET /endpoint/{value}`.

        `value` must be the path parameter (e.g. hostname, network, id, etc.).
        """
        value = self._normalize_path_param(value)
        resp = self._client.get(self.endpoint.with_id(value), ok404=True)
        if not resp:
            return None
        return self._validate_json(resp.text)

    def _search_one(self, field: str, value: str | int) -> T | None:
        """Fetch one object via a unique search: `GET /endpoint?field=value`.

        Used for any field that is not the path parameter (e.g. `/hosts?ipaddress=10.0.0.1`).
        """
        data = self._client.get_item_by_key_value(self.endpoint, field, value, ok404=True)
        if not data:
            return None
        return self._validate(data)

    def _fetch_by_field(self, field: str, value: str | int) -> T | None:
        """Fetch one object by any field, path-fetching when it is the path parameter."""
        if field.casefold() == self._path_param_field.casefold():
            return self._fetch_by_path(value)
        return self._search_one(field, value)

    def _fetch_list_by_field(self, field: str, value: str | int) -> list[T]:
        """Fetch all objects matching a field, mirroring old APIMixin.get_list_by_field."""
        params: QueryParams = {field: value}
        return self._client.get_typed(self.endpoint, list[self.model], params=params)

    def refresh(self, obj: T) -> T:
        """Fetch a fresh copy of the resource from the server.

        Args:
            obj: The resource object to refresh.

        Returns:
            A fresh copy of the resource object.

        Prefers the immutable `id` when the model has one, so refreshing a renamed
        resource does not break the lookup. Models without an `id` field
        (HInfo, Location — keyed by `host`) fall back to the path parameter.
        """
        if (obj_id := getattr(obj, "id", None)) is not None:
            fresh = self._fetch_by_field("id", obj_id)
        else:
            fresh = self._fetch_by_path(self._path_param_value(obj))
        if fresh is None:
            raise GetError(f"Could not refetch {self.model_name}.")
        return fresh

    @overload
    def get(self, ident: str | int | T, *, required: Literal[False]) -> T | None: ...
    @overload
    def get(self, ident: str | int | T, *, required: Literal[True] = ...) -> T: ...
    def get(self, ident: str | int | T, *, required: bool = True) -> T | None:
        """Get a resource by its endpoint identifier (str) or its ID (int).

        Args:
            ident: The path parameter (id / name / network, per resource).
                String arguments are only supported for resources that are
                addressed by a non-numeric path parameter (e.g. name, network, hostname).
            required: When `True` (default), raise `EntityNotFound` if missing.
                Pass `False` to return `T | None` instead.

        Raises:
            EntityNotFound: If `required` is True and the resource is not found.

        Returns:
            The resource, or `None` when `required` is False.
        """
        obj: T | None = None
        try:
            obj = self._resolve(ident)
        except EntityNotFound:
            if required:
                raise
        return obj

    def assert_absent(self, ident: str | int) -> None:
        """Assert that no resource with `ident` exists.

        Args:
            ident (str | int): The path parameter to check (id / name / network, per resource).

        Raises:
            EntityAlreadyExists: If a resource with `ident` exists.
        """
        ident = self._normalize_path_param(ident)
        if self._fetch_by_field(self._path_param_field, ident) is not None:
            raise EntityAlreadyExists(f"{self.model_name} {ident!r} already exists.")

    def list(
        self,
        *,
        limit: int | None = None,
        **query: str | int | float | bool | None,
    ) -> list[T]:
        """List resources, optionally filtered by query parameters.

        Args:
            limit (int | None): Maximum number of results to return, None means no limit. Defaults to None.
            **query: Optional filter parameters forwarded to the list endpoint.
        """
        params: QueryParams = dict(query)
        return self._client.get_typed(self.endpoint, list[self.model], params=params, limit=limit)

    # TODO: add warning or similar when used on non-paginated endpoints somehow.
    # Manually? Use contextvar? Who knows.
    @overload
    def first(self, *, required: Literal[False], **query: str | int | float | bool | None) -> T | None: ...
    @overload
    def first(self, *, required: Literal[True] = ..., **query: str | int | float | bool | None) -> T: ...
    def first(self, *, required: bool = True, **query: str | int | float | bool | None) -> T | None:
        """Return the first resource, raising if not found by default.

        Passes `page_size=1` to avoid over-fetching.

        Over-fetches on certain non-standard endpoints that do not implement pagination
        such as `networks/{network}/unused_list`, `/dhcphosts`, and others.

        Args:
            required: When `True` (default), raise `EntityNotFound` if no resource
                exists. Pass `False` to return `None` instead.
            **query: Optional filter parameters forwarded to the list endpoint.

        Raises:
            EntityNotFound: If `required` is True and no resource is found.

        Returns:
            The first resource found, or `None` when `required` is False.
        """
        params: QueryParams = dict(query)

        # XXX: No check for whether endpoint contains placeholders here.
        res = self._client.get_first(self.endpoint, params)
        if res is None:
            if required:
                # TODO: add query to error message if defined
                raise EntityNotFound(f"No {self.model_name} found.")
            else:
                return None
        return get_type_adapter(self.model).validate_python(res)

    def count(self, *, strict: bool = False) -> int:
        """Return the server-reported total count of resources at this endpoint.

        Falls back on fetching all results and counting them client-side
        if endpoint does not support counting via pagination and `strict` is `False`.

        Args:
            strict (bool): Raise exception instead of falling back on counting client-side.
        """
        return self._client.get_count(self.endpoint, strict=strict)


class WriteResourceManager(ResourceManager[T], ABC):
    """Manager for performing CRUD operations on an API resource type."""

    def _create(self, data: dict[str, Any]) -> T:
        """POST `data` to the resource endpoint, returning the created object.

        Args:
            data (dict[str, Any]): The data to POST to the resource endpoint.

        Returns:
            T: The created resource object.
        """
        response = self._client.post(self.endpoint, json=data)
        # Prefer bundled object
        if response.content:
            try:
                return self._validate_response(response)
            except ValidationError:  # NOTE: can it raise other types of errors?
                pass
        if "Location" in response.headers:
            return self._client.get_typed(response.headers["Location"], self.model)
        raise PostError(f"Failed to fetch {self.model_name} after creation.")

    def _patch(self, obj: T, data: dict[str, Any], *, params: QueryParams | None = None) -> None:
        """PATCH `obj` with `data`. Does not update the local object passed to this method.

        Call `refresh()` to refresh an object after patching it.

        Args:
            obj (T): The object to patch.
            data (dict[str, Any]): The data to PATCH to the resource endpoint.
            params (QueryParams | None): Optional query parameters to include in the PATCH request.

        Raises `PatchError` (from the client) if the server rejects the patch.
        """
        _ = self._client.patch(self._endpoint_with_path_param(obj), json=data, params=params)

    # TODO: rename to _delete so subclass implementations of `delete` can use
    # appropriately named parameters + add new parameters if required. This public
    # method is too inflexible.
    def delete(self, obj: int | T) -> None:
        """Delete a resource.

        Args:
            obj (T): The ID or object reference of resource to delete.
        """
        obj = self._resolve(obj)
        _ = self._client.delete(self._endpoint_with_path_param(obj))


class NamedResourceManager(WriteResourceManager[T], ABC):
    """Extended WriteResourceManager for resources that support name-based lookups."""

    # NOTE: write methods are inherited even for managers where the underlying
    # API endpoint may not support writes (e.g. NameServerManager).

    # TODO: should override _path_param_field instead?!
    name_field: ClassVar[str] = "name"
    name_lowercase: ClassVar[bool] = False

    def _case_name(self, name: str) -> str:
        return name.lower() if self.name_lowercase else name

    def _normalize_name(self, name: str) -> str:
        """Normalise a name before use in lookups. Override to add hostname expansion."""
        return self._case_name(name)

    @override
    def _resolve(self, ref: int | str | T) -> T:
        # Resolve by name search if we have a str arg and the API resource path
        # does not identify the resource by name, e.g.:
        # `/labels/{id}` -> must search by name -> `/labels?name={name}`
        # TODO: only labels endpoints defines a separate `<resource>/name` endpoint
        #       all others can be looked up via name. It makes more sense to override
        #       this method in `LabelManager` and just use the name endpoint there.
        if isinstance(ref, str) and self._path_param_field != self.name_field:
            return self.get_by_name(ref)
        return super()._resolve(ref)

    @override
    def assert_absent(self, ident: str | int) -> None:
        """Assert that a given resource does not exist.

        Args:
            ident (str | int): The path parameter or name to check.

        Raises:
            EntityAlreadyExists: If a resource with the given identifier exists.
        """
        if isinstance(ident, int):
            # If the identifier is an integer, we assume it's an ID and not a name.
            # In this case, we can delegate to the parent class's assert_absent method.
            super().assert_absent(ident)
            return
        # If we have a string, delegate to internal get_by_name method
        if self.get_by_name(ident, required=False) is not None:
            raise EntityAlreadyExists(f"{self.model_name} {ident!r} already exists.")

    # XXX:  Do we actually need an explicit name based lookup?
    #       Can we just override `get()` and use `fetch_by_field()` there, falling
    #       back on superclass `get()` if arg is not a string?
    #       There is some common abstraction for resolving name-based models that
    #       that is waiting to be found. See: `_NamedResourceManager._resolve`, which
    #       uses some basic heuristics to determine if it should fetch by name (using this method ironically),
    #       or if it should fall back on the path-identifier based approach.
    #
    #       get() should IDEALLY handle the "primary" path i.e. the path parameter.
    #       So if we could do `client.host.get("ExAMple.com")`, with the knowledge
    #       that the class is aware that string arguments are names, which then normalizes
    #       them like we do here, then we can remove this method entirely.
    #
    #       NOTE TO AGENTS: we now have `_normalize_path_param` and `_normalize_name`.
    #           `_normalize_path_param` takes `str | int` while `_normalize_name` takes `str`.
    #           Can we (and should we) make it stricter (`_normalize_path_param` is `int`,
    #           `_normalize_name` is `str`)?
    #           Would this break any assumptions or encode too strong assumptions into
    #           the code about the type of a path parameter field? Instinct says yes.
    #
    #       However, maybe it's fine to keep it to have an explicit name-only lookup?
    @overload
    def get_by_name(self, name: str, *, required: Literal[False]) -> T | None: ...
    @overload
    def get_by_name(self, name: str, *, required: Literal[True] = ...) -> T: ...
    def get_by_name(self, name: str, *, required: bool = True) -> T | None:
        """Get a resource by its name.

        Args:
            name (str): Name field for the resource to look up.
            required (bool): Raise if not found. Defaults to True.

        Raises:
            EntityNotFound: If `required` is True and the resource is not found.

        Returns:
            T | None: The resource object if found, else None.
        """
        name = self._normalize_name(name)
        obj = self._fetch_by_field(self.name_field, name)
        if required and obj is None:
            raise EntityNotFound(f"{self.model_name} {name!r} not found.")
        return obj

    @override
    def delete(self, obj: int | str | T) -> None:
        """Delete a resource.

        Args:
            obj (int | str | T): The resource, ID, or name to delete.
        """
        obj = self._resolve(obj)
        _ = self._client.delete(self._endpoint_with_path_param(obj))

    # TODO: add str | int support for obj?
    def rename(self, obj: T, new_name: str) -> None:
        """Rename the resource.

        Does not refresh the object locally.
        Call `refresh()` with the object to fetch the updated resource.

        Args:
            obj: The resource to rename.
            new_name: The new name to set.
        """
        self._patch(obj, {self.name_field: self._normalize_name(new_name)})

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

        Args:
            name (str): The name of the resource to fetch history for.

        Returns:
            A list of `HistoryItem` objects.
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
    """Operations on Host resources."""

    _path_param_field: ClassVar[str] = "name"

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

    def _normalize_path_param(self, ident: str | int) -> str | int:
        return self._client.fqdn(ident) if isinstance(ident, str) else ident

    # NOTE: why do we have a separate _normalize_history_name?
    #       They do the exact same thing...
    def _normalize_name(self, name: str) -> str:
        return self._client.fqdn(name)

    def _normalize_history_name(self, name: str) -> str:
        return self._client.fqdn(name)

    def _record_ptr_event(self, host: Host, ip: str) -> None:
        """Record that `ip` resolved to `host` via a PTR override."""
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
    def get_by_id(self, host_id: int, *, required: Literal[False]) -> Host | None: ...
    @overload
    def get_by_id(self, host_id: int, *, required: Literal[True] = ...) -> Host: ...
    def get_by_id(self, host_id: int, *, required: bool = True) -> Host | None:
        """Get a host by its numeric id.

        Distinct from :meth:`get`: the Host endpoint id-field is the hostname, so
        :meth:`get` resolves by name while this resolves by the numeric `id`.

        Args:
            host_id (int): The numeric id of the host.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the host is not found.
        """
        obj = self._fetch_by_field("id", host_id)
        if required and obj is None:
            raise EntityNotFound(f"Host with id {host_id!r} not found.")
        return obj

    @overload
    def get_by_ip(
        self, ip: str | IP_AddressT, *, required: Literal[False], ptr_fallback: bool = ...
    ) -> Host | None: ...
    @overload
    def get_by_ip(
        self, ip: str | IP_AddressT, *, required: Literal[True] = ..., ptr_fallback: bool = ...
    ) -> Host: ...
    def get_by_ip(
        self, ip: str | IP_AddressT, *, required: bool = True, ptr_fallback: bool = True
    ) -> Host | None:
        """Get a host by IP address (A/AAAA, optionally falling back to PTR override).

        Args:
            ip (str | IP_AddressT): The IP address to look up.
            required (bool): When True (default), raise EntityNotFound if not found.
            ptr_fallback (bool): When True (default), fall back to PTR override if no host is found.

        Raises:
            MultipleEntitiesFound: If more than one host matches the IP address.
            EntityNotFound: If `required` is True and no host is found.
        """
        addr = str(NetworkOrIP.parse_or_raise(str(ip), mode="ip"))
        try:
            host = self._fetch_by_field("ipaddresses__ipaddress", addr)
            if host is None and ptr_fallback:
                host = self._fetch_by_field("ptr_overrides__ipaddress", addr)
                if host is not None:
                    self._record_ptr_event(host, addr)
        except MultipleEntitiesFound as e:
            raise MultipleEntitiesFound(f"Multiple hosts found with IP address {addr}.") from e
        if required and host is None:
            raise EntityNotFound(f"Host with IP address {addr} not found.")
        return host

    @overload
    def get_by_mac(self, mac: str | MacAddress, *, required: Literal[False]) -> Host | None: ...
    @overload
    def get_by_mac(self, mac: str | MacAddress, *, required: Literal[True] = ...) -> Host: ...
    def get_by_mac(self, mac: str | MacAddress, *, required: bool = True) -> Host | None:
        """Get a host by MAC address.

        Args:
            mac (str | MacAddress): The MAC address to look up.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and no host is found.
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
    ) -> Host:
        """Create a host.

        Args:
            name (str | HostName): Name of the host to create.
            comment (str, optional): Comment for the host. Defaults to "".
            contacts (list[str] | None, optional): List of contacts for the host.
            ipaddress (IP_AddressT | str | None, optional): IP address of the host.
            network (IP_NetworkT | str | None, optional): Network of the host.

        Returns:
            Host: The created host.
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
        return self._create(data)

    def update(
        self,
        host: int | str | Host,
        *,
        name: str | HostName | UNSET = UNSET,
        comment: str | None | UNSET = UNSET,
        contacts: list[str] | UNSET = UNSET,
        ttl: int | None | UNSET = UNSET,
    ) -> None:
        """Update a host's mutable fields.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            name (str | HostName | UNSET): New name for the host. Omit to leave unchanged.
            comment (str | None | UNSET): New comment. Pass None to unset, omit to leave unchanged.
            contacts (list[str] | UNSET): New contacts list. Omit to leave unchanged.
            ttl (int | None | UNSET): New TTL. Pass None to reset to default, omit to leave unchanged.
        """
        host = resolve_host(host, self._client)
        data: dict[str, Any] = {}
        if name is not UNSET:
            data["name"] = self._client.fqdn(str(name))
        if comment is not UNSET:
            data["comment"] = comment
        if contacts is not UNSET:
            data["contacts"] = contacts
        if ttl is not UNSET:
            data["ttl"] = ttl
        self._patch(host, data)

    def add_contacts(self, host: int | str | Host, contacts: list[str]) -> HostContactModification:
        """Add contacts to a host (atomic; POST to /hosts/{name}/contacts/).

        Args:
            host (int | str | Host): Host instance or numeric ID.
            contacts (list[str]): Email addresses to add as contacts.
        """
        host = resolve_host(host, self._client)
        resp = self._client.post(Endpoint.HostsContacts.with_params(host.name), json={"emails": contacts})
        return get_type_adapter(HostContactModification).validate_json(resp.text)

    def clear_contacts(self, host: int | str | Host) -> HostContactModification:
        """Remove all contacts from a host (atomic; DELETE /hosts/{name}/contacts/).

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host = resolve_host(host, self._client)
        resp = self._client.delete(Endpoint.HostsContacts.with_params(host.name))
        return get_type_adapter(HostContactModification).validate_json(resp.text)

    def remove_contacts(self, host: int | str | Host, contacts: list[str]) -> HostContactModification:
        """Remove specific contacts from a host (atomic; DELETE /hosts/{name}/contacts/).

        Args:
            host (int | str | Host): Host instance or numeric ID.
            contacts (list[str]): Email addresses to remove from contacts.
        """
        host = resolve_host(host, self._client)
        resp = self._client.delete(Endpoint.HostsContacts.with_params(host.name), json={"emails": contacts})
        return get_type_adapter(HostContactModification).validate_json(resp.text)

    def networks(self, host: int | str | Host) -> dict[Network, list[IPAddress]]:
        """Return a dict mapping each network to the host's IP addresses on that network.

        Networks not registered in MREG produce a placeholder via
        :meth:`~mreg_api.models.Network.dummy_network_from_ip`.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host = resolve_host(host, self._client)
        net_manager = NetworkManager(self._client)
        result: dict[Network, list[IPAddress]] = {}
        for ip in host.ipaddresses:
            network = net_manager.get_by_ip(ip.ipaddress, required=False)
            if network is None:
                network = Network.dummy_network_from_ip(ip)
            if network not in result:
                result[network] = []
            result[network].append(ip)
        return result

    def vlans(self, host: int | str | Host) -> dict[int, list[IPAddress]]:
        """Return a dict mapping VLAN ID to host IPs on that VLAN. IPs with no VLAN map to 0.

        Performs API lookups to fetch all networks the host is associated with.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        result: dict[int, list[IPAddress]] = {}
        for network, ips in self.networks(host).items():
            vlan = network.vlan or 0
            if vlan not in result:
                result[vlan] = []
            result[vlan].extend(ips)
        return result

    def all_ips_on_same_vlan(self, host: int | str | Host) -> bool:
        """Return True if all host IPs share a single VLAN (or there are no IPs).

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        return len(self.vlans(host)) <= 1


class HostGroupManager(NamedResourceManager[HostGroup], HistoryManager[HostGroup]):
    """Operations on HostGroup resources."""

    _path_param_field: ClassVar[str] = "name"

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
        description: str | UNSET = UNSET,
    ) -> HostGroup:
        """Create a host group.

        Args:
            name (str): Name of the host group.
            description (str | UNSET): Description of the group. Omit to leave unset.

        Returns:
            HostGroup: The created host group.
        """
        data: dict[str, Any] = {"name": name}
        if description is not UNSET:
            data["description"] = description
        return self._create(data)

    # Not much to update here, but we implement update for future expansion + consistent interface
    def update(
        self,
        hostgroup: int | str | HostGroup,
        *,
        description: str | UNSET = UNSET,
    ) -> None:
        """Update a host group's mutable fields.

        Args:
            hostgroup (int | str | HostGroup): HostGroup instance, name string, or numeric ID.
            description (str | UNSET): New description. Omit to leave unchanged.
        """
        group = self._resolve(hostgroup)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        self._patch(group, data)

    def set_description(self, hostgroup: int | str | HostGroup, description: str) -> None:
        """Set the description for the host group.

        Args:
            hostgroup (int | str | HostGroup): HostGroup instance, name string, or numeric ID.
            description (str): New description to set.
        """
        hostgroup = self._resolve(hostgroup)
        self.update(hostgroup, description=description)

    def _resolve_hostgroup_name(self, hostgroup: str | HostGroup) -> str:
        """Resolve a host group reference (instance or name) to a string name."""
        if isinstance(hostgroup, HostGroup):
            return hostgroup.name
        return str(hostgroup)

    # TODO: Add int ID support for `subgroup` and `host` arguments in methods.
    #       requires refactoring `_resolve_hostgroup_name` and `resolve_host_name`
    #       to use `resolve_hostgroup` and `resolve_host` respectively, or
    #       some other clever solution. Very low priority to fix that right now.

    def add_group(self, hostgroup: int | str | HostGroup, subgroup: str | HostGroup) -> None:
        """Add a group to a host group.

        Args:
            hostgroup (int | str | HostGroup): The parent HostGroup instance, name string, or numeric ID.
            subgroup (str | HostGroup): HostGroup instance or name string to add as a subgroup.
        """
        hostgroup = self._resolve(hostgroup)
        subgroup_name = self._resolve_hostgroup_name(subgroup)

        self._client.post(
            Endpoint.HostGroupsAddHostGroups.with_params(hostgroup.name),
            json={"name": subgroup_name},
        )

    def remove_group(self, hostgroup: int | str | HostGroup, subgroup: str | HostGroup) -> None:
        """Remove a group from a host group.

        Args:
            hostgroup (int | str | HostGroup): The parent HostGroup instance, name string, or numeric ID.
            subgroup (str | HostGroup): HostGroup instance or name string to remove.
        """
        hostgroup = self._resolve(hostgroup)
        subgroup_name = self._resolve_hostgroup_name(subgroup)

        self._client.delete(
            Endpoint.HostGroupsRemoveHostGroups.with_params(hostgroup.name, subgroup_name),
        )

    def add_host(self, hostgroup: int | str | HostGroup, host: str | Host) -> None:
        """Add a host to a host group.

        Args:
            hostgroup (int | str | HostGroup): HostGroup instance, name string, or numeric ID.
            host (str | Host): Host reference (name string or Host instance).
        """
        hostgroup = self._resolve(hostgroup)
        hostname = resolve_host_name(host, self._client)

        self._client.post(
            Endpoint.HostGroupsAddHosts.with_params(hostgroup.name),
            json={"name": hostname},
        )

    def remove_host(self, hostgroup: int | str | HostGroup, host: str | Host) -> None:
        """Remove a host from a host group.

        Args:
            hostgroup (int | str | HostGroup): HostGroup instance, name string, or numeric ID.
            host (str | Host): Host reference (name string or Host instance).
        """
        hostgroup = self._resolve(hostgroup)
        hostname = resolve_host_name(host, self._client)

        self._client.delete(
            Endpoint.HostGroupsRemoveHosts.with_params(hostgroup.name, hostname),
        )

    def add_owner(self, hostgroup: int | str | HostGroup, name: str) -> None:
        """Add an owner to a host group.

        Args:
            hostgroup (int | str | HostGroup): HostGroup instance, name string, or numeric ID.
            name (str): Name of the owner to add.
        """
        hostgroup = self._resolve(hostgroup)

        self._client.post(
            Endpoint.HostGroupsAddOwner.with_params(hostgroup.name),
            json={"name": name},
        )

    def remove_owner(self, hostgroup: int | str | HostGroup, name: str) -> None:
        """Remove an owner from a host group.

        Args:
            hostgroup (int | str | HostGroup): HostGroup instance, name string, or numeric ID.
            name (str): Name of the owner to remove.
        """
        hostgroup = self._resolve(hostgroup)

        self._client.delete(
            Endpoint.HostGroupsRemoveOwner.with_params(hostgroup.name, name),
        )

    # RENAMED: get_all_parents -> list_parents
    def list_parents(self, hostgroup: int | str | HostGroup) -> list[HostGroup]:
        """Get all parent groups of a host group.

        Renamed from `get_all_parents` to `list_parents`

        Args:
            hostgroup (int | str | HostGroup): HostGroup instance, name string, or numeric ID.
        """
        hostgroup = self._resolve(hostgroup)
        parents: list[HostGroup] = []
        for parent in hostgroup.parent:  # why singular name?
            pobj = self._fetch_by_field("name", parent)
            if pobj:
                parents.append(pobj)
                parents.extend(self.list_parents(pobj))
        return parents

    def list_by_host(self, host: int | str | Host, *, traverse: bool = False) -> list[HostGroup]:
        """List all hostgroups that include the given host.

        Args:
            host (int | str | Host): Host instance, numeric ID, or name string.
            traverse (bool): If True, recursively include parent groups of the direct groups.
        """
        host_id = resolve_host_id(host, self._client)
        direct = self._fetch_list_by_field("hosts", host_id)
        if not traverse:
            return sorted(direct, key=lambda g: g.name)
        groups: list[HostGroup] = list(direct)
        for hostgroup in direct:
            groups.extend(self.list_parents(hostgroup))
        return sorted(groups, key=lambda g: g.name)


class LabelManager(NamedResourceManager[Label]):
    """Operations on Label resources."""

    # NOTE: the regular labels endpoint uses IDs for lookups, but it is possible
    # to fetch by name when using the /labels/name endpoint.
    # This makes no sense, of course, but that's how it is.
    _path_param_field: ClassVar[str] = "id"

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
    ) -> Label:
        """Create a label.

        Args:
            name (str): Name of the label.
            description (str): Description of the label.

        Returns:
            Label: The created label.
        """
        return self._create({"name": name, "description": description})

    def update(
        self,
        label: int | str | Label,
        *,
        description: str | UNSET = UNSET,
    ) -> None:
        """Update a label's mutable fields.

        Args:
            label (int | str | Label): Label instance, numeric ID, or name string.
            description (str | UNSET): New description. Omit to leave unchanged.
        """
        label = self._resolve(label)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        self._patch(label, data)

    def set_description(self, label: int | str | Label, description: str) -> None:
        """Set the description for the label.

        Args:
            label (int | str | Label): Label instance, numeric ID, or name string.
            description (str): New description to set.
        """
        label = self._resolve(label)
        self.update(label, description=description)


class HostPolicyManagerNamespace:
    """Namespace for grouping related host policy managers."""

    def __init__(self, client: MregClient) -> None:  # noqa: D107
        self._client = client
        self.role: RoleManager = client.role
        self.label: LabelManager = client.label
        self.atom: AtomManager = client.atom


class RoleManager(NamedResourceManager[Role], HistoryManager[Role]):
    """Operations on Role resources."""

    _path_param_field: ClassVar[str] = "name"

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

    def _get_atom(self, atom: str | int) -> Atom:
        """Helper method to provide common interface to fetch an atom by name (str) or ID (int)."""
        if isinstance(atom, int):
            return AtomManager(self._client).get(atom)
        return AtomManager(self._client).get_by_name(str(atom))

    def _resolve_atom_name(self, atom: str | int | Atom) -> str:
        """Resolve an atom reference (instance, numeric ID, or name) to a string name.

        Fetches the atom from the API if an integer ID is provided, to ensure the name is valid.

        Raises EntityNotFound if the atom does not exist when an integer ID is provided.
        """
        if isinstance(atom, Atom):
            return atom.name
        elif isinstance(atom, int):
            return self._get_atom(atom).name
        return str(atom)

    def create(
        self,
        *,
        name: str,
        description: str = "",
    ) -> Role:
        """Create a role.

        Args:
            name (str): Name of the role.
            description (str): Description of the role. Defaults to "".

        Returns:
            Role: The created role.
        """
        return self._create({"name": name, "description": description})

    def update(
        self,
        role: int | str | Role,
        *,
        description: str | UNSET = UNSET,
    ) -> None:
        """Update a role's mutable fields.

        Args:
            role (int | str | Role): Role instance, numeric ID, or name string.
            description (str | UNSET): New description. Omit to leave unchanged.
        """
        role = self._resolve(role)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        self._patch(role, data)

    def set_description(self, role: int | str | Role, description: str) -> None:
        """Set the description for the role.

        Args:
            role (int | str | Role): Role instance, numeric ID, or name string.
            description (str): New description to set.
        """
        role = self._resolve(role)
        self.update(role, description=description)

    @override
    def delete(self, obj: int | str | Role) -> None:
        """Delete a role.

        Args:
            obj (int | str | Role): Role instance, numeric ID, or name string.

        Raises:
            DeleteError: If the role is still in use on any hosts.
        """
        obj = self._resolve(obj)
        if obj.hosts:
            hosts = ", ".join(obj.hosts)
            raise DeleteError(f"Role {obj.name!r} used on hosts: {hosts}")
        super().delete(obj)

    def list_with_atom(self, atom: int | str | Atom) -> list[Role]:
        """List all roles that contain a given atom.

        Renamed from `Role.get_roles_with_atom`.

        Args:
            atom (int | str | Atom): Atom instance, numeric ID, or name string.
        """
        atom_name = self._resolve_atom_name(atom)
        return self._fetch_list_by_field("atoms__name__exact", atom_name)

    def add_atom(self, role: int | str | Role, atom: int | str | Atom) -> bool:
        """Add an atom to the role.

        Args:
            role (int | str | Role): Role instance, numeric ID, or name string.
            atom (int | str | Atom): Atom instance, numeric ID, or name string.

        Returns:
            bool: True if the atom was successfully added.
                DEPRECATED: Maintains parity with older library versions.
                Will never return False on failure; an exception is raised instead.

        Raises:
            EntityNotFound: If the atom does not exist.
            EntityAlreadyExists: If the atom is already a member of the role.
        """
        role = self._resolve(role)
        atom_name = self._resolve_atom_name(atom)
        _ = self._get_atom(atom_name)  # ensure atom exists
        if atom_name in role.atoms:
            raise EntityAlreadyExists(f"Atom {atom_name!r} already a member of role {role.name!r}")
        # TODO: need a better abstraction for endpoints that
        # return the new version of the resource after modification
        self._client.post(Endpoint.HostPolicyRolesAddAtom.with_params(role.name), json={"name": atom_name})
        return True

    def remove_atom(self, role: int | str | Role, atom: int | str | Atom) -> bool:
        """Remove an atom from the role.

        Args:
            role (int | str | Role): Role instance, numeric ID, or name string.
            atom (int | str | Atom): Atom instance, numeric ID, or name string.

        Returns:
            bool: True if the atom was successfully removed.
                DEPRECATED: Maintains parity with older library versions.
                Will never return False on failure; an exception is raised instead.

        Raises:
            EntityOwnershipMismatch: If the atom is not a member of the role.
        """
        role = self._resolve(role)
        atom_name = self._resolve_atom_name(atom)
        if atom_name not in role.atoms:  # NOTE: use a method for this? Casing, etc.
            raise EntityOwnershipMismatch(f"Atom {atom_name!r} not a member of {role.name!r}")
        self._client.delete(Endpoint.HostPolicyRolesRemoveAtom.with_params(role.name, atom_name))
        return True

    def add_host(self, role: int | Role, host: str | Host) -> bool:
        """Add a host to the role by name.

        Args:
            role (int | Role): Role instance or numeric ID.
            host (str | Host): Host reference (name string or Host instance).

        Returns:
            bool: True if the host was successfully added.
                DEPRECATED: Maintains parity with older library versions.
                Will never return False on failure; an exception is raised instead.
        """
        role = self._resolve(role)
        hostname = resolve_host_name(host, self._client)
        self._client.post(Endpoint.HostPolicyRolesAddHost.with_params(role.name), json={"name": hostname})
        return True

    def remove_host(self, role: int | Role, host: str | Host) -> bool:
        """Remove a host from the role by name.

        Args:
            role (int | Role): Role instance or numeric ID.
            host (str | Host): Host reference (name string or Host instance).

        Returns:
            bool: True if the host was successfully removed.
                DEPRECATED: Maintains parity with older library versions.
                Will never return False on failure; an exception is raised instead.
        """
        role = self._resolve(role)
        hostname = resolve_host_name(host, self._client)
        self._client.delete(Endpoint.HostPolicyRolesRemoveHost.with_params(role.name, hostname))
        return True

    @deprecated('use "list_labels()" instead')
    def get_labels(self, role: int | str | Role) -> list[Label]:
        """List the labels associated with the role.

        Args:
            role (int | str | Role): Role instance, numeric ID, or name string.

        Returns:
            list[Label]: List of labels associated with the role.
        """
        return self.list_labels(role)

    def list_labels(self, role: int | str | Role) -> list[Label]:
        """List the labels associated with the role.

        Args:
            role (int | str | Role): Role instance, numeric ID, or name string.

        Returns:
            list[Label]: List of labels associated with the role.
        """
        role = self._resolve(role)
        labels = LabelManager(self._client)
        return [labels.get(lid, required=True) for lid in role.labels]

    def add_label(self, role: int | str | Role, label: int | str | Label) -> None:
        """Add a label to the role.

        Args:
            role (int | str | Role): Role instance, numeric ID, or name string.
            label (int | str | Label): Label instance, name, or numeric ID.

        Raises:
            EntityNotFound: If the label does not exist.
            EntityAlreadyExists: If the role already has the label.
        """
        role = self._resolve(role)
        label_id = self._resolve_label_id(label)
        if label_id in role.labels:
            raise EntityAlreadyExists(f"Role {role.name!r} already has label {label!r}")
        self._patch(role, {"labels": [*role.labels, label_id]})

    def remove_label(self, role: int | str | Role, label: int | str | Label) -> None:
        """Remove a label from the role.

        Args:
            role (int | str | Role): Role instance, numeric ID, or name string.
            label (int | str | Label): Label instance, name, or numeric ID.

        Raises:
            EntityNotFound: If the label does not exist.
            EntityOwnershipMismatch: If the role does not have the label.
        """
        role = self._resolve(role)
        label_id = self._resolve_label_id(label)
        if label_id not in role.labels:
            raise EntityOwnershipMismatch(f"Role {role.name!r} does not have label {label!r}")
        self._patch(role, {"labels": [lid for lid in role.labels if lid != label_id]})

    def list_by_host(self, host: int | str | Host) -> list[Role]:
        """List all roles that include the given host.

        Args:
            host (int | str | Host): Host instance, name string, or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("hosts", host_id)


class AtomManager(NamedResourceManager[Atom], HistoryManager[Atom]):
    """Operations on Atom resources."""

    _path_param_field: ClassVar[str] = "name"

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
    ) -> Atom:
        """Create an atom.

        Args:
            name (str): Name of the atom.
            description (str): Description of the atom. Defaults to "".

        Returns:
            Atom: The created atom.
        """
        return self._create({"name": name, "description": description})

    def update(
        self,
        atom: int | str | Atom,
        *,
        description: str | UNSET = UNSET,
    ) -> None:
        """Update an atom's mutable fields.

        Args:
            atom (int | str | Atom): Atom instance, name string, or numeric ID.
            description (str | UNSET): New description. Omit to leave unchanged.
        """
        atom = self._resolve(atom)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        self._patch(atom, data)

    def set_description(self, atom: int | str | Atom, description: str) -> None:
        """Set the description for the atom.

        Args:
            atom (int | str | Atom): Atom instance, name string, or numeric ID.
            description (str): New description to set.
        """
        atom = self._resolve(atom)
        self.update(atom, description=description)

    @override
    def delete(self, obj: int | str | Atom) -> None:
        """Delete an atom.

        Args:
            obj (int | str | Atom): Atom instance, name string, or numeric ID.

        Raises:
            DeleteError: If the atom is still used in any roles.
        """
        obj = self._resolve(obj)
        if obj.roles:
            roles = ", ".join(obj.roles)
            raise DeleteError(f"Atom {obj.name!r} used in roles: {roles}")
        super().delete(obj)


class PermissionManager(WriteResourceManager[Permission]):
    """Operations on Permission resources."""

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
    ) -> Permission:
        """Create a permission.

        Args:
            group: The netgroup the permission applies to.
            range: The network range (CIDR) the permission covers.
            regex: The host regex pattern for the permission.
            labels: Optional list of label IDs to attach.

        Returns:
            Permission: The created permission.
        """
        data: dict[str, Any] = {"group": group, "range": range, "regex": regex}

        # TODO: check if we can actually pass labels during creation
        # TODO: also accept Label objects?
        if labels is not None:
            data["labels"] = labels
        return self._create(data)

    def update(
        self,
        permission: int | Permission,
        *,
        group: str | UNSET = UNSET,
        range: str | UNSET = UNSET,  # noqa: A002
        regex: str | UNSET = UNSET,
        labels: list[int] | UNSET = UNSET,
    ) -> None:
        """Update a permission's mutable fields.

        Args:
            permission (int | Permission): Permission instance or numeric ID.
            group (str | UNSET): New netgroup name. Omit to leave unchanged.
            range (str | UNSET): New network range (CIDR). Omit to leave unchanged.
            regex (str | UNSET): New host regex pattern. Omit to leave unchanged.
            labels (list[int] | UNSET): New list of label IDs. Omit to leave unchanged.
        """
        permission = self._resolve(permission)
        data: dict[str, Any] = {}
        if group is not UNSET:
            data["group"] = group
        if range is not UNSET:
            data["range"] = range
        if regex is not UNSET:
            data["regex"] = regex
        if labels is not UNSET:
            data["labels"] = labels
        self._patch(permission, data)

    def add_label(self, permission: int | Permission, label: int | str | Label) -> None:
        """Add a label to the permission.

        Args:
            permission: The permission (instance or numeric id).
            label: The label to add (instance, name, or numeric id).

        Raises:
            EntityNotFound: If the label does not exist.
            EntityAlreadyExists: If the permission already has this label.
        """
        permission = self._resolve(permission)
        label_id = self._resolve_label_id(label)
        if label_id in permission.labels:
            raise EntityAlreadyExists(f"Permission already has label {label!r}.")
        self.update(permission, labels=[*permission.labels, label_id])

    def remove_label(self, permission: int | Permission, label: int | str | Label) -> None:
        """Remove a label from the permission.

        Args:
            permission: The permission (instance or numeric id).
            label: The label to remove (instance, name, or numeric id).

        Raises:
            EntityNotFound: If the label does not exist or the permission lacks it.
        """
        permission = self._resolve(permission)
        label_id = self._resolve_label_id(label)
        if label_id not in permission.labels:
            raise EntityNotFound(f"Permission does not have label {label!r}.")
        self.update(permission, labels=[lid for lid in permission.labels if lid != label_id])

    @overload
    def get_by_triplet(
        self,
        group: str,
        range: str,  # noqa: A002
        regex: str,
        *,
        required: Literal[False],
    ) -> Permission | None: ...
    @overload
    def get_by_triplet(
        self,
        group: str,
        range: str,  # noqa: A002
        regex: str,
        *,
        required: Literal[True] = ...,
    ) -> Permission: ...
    def get_by_triplet(
        self,
        group: str,
        range: str,  # noqa: A002
        regex: str,
        *,
        required: bool = True,
    ) -> Permission | None:
        """Get a permission by the (group, range, regex) triplet.

        Replaces `Permission.get_by_query_unique_or_raise` from the old model API.

        Args:
            group (str): The netgroup name.
            range (str): The network range (CIDR).
            regex (str): The host regex pattern.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            MultipleEntitiesFound: If more than one permission matches the triplet.
            EntityNotFound: If `required` is True and no match is found.
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
    """Operations on NetworkPolicyAttribute resources."""

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
    ) -> NetworkPolicyAttribute:
        """Create a network policy attribute.

        Args:
            name (str): Name of the attribute (lowercased).
            description (str): Description of the attribute.

        Returns:
            NetworkPolicyAttribute: The created network policy attribute.
        """
        return self._create({"name": name, "description": description})

    def update(
        self,
        attr: int | str | NetworkPolicyAttribute,
        *,
        description: str | UNSET = UNSET,
    ) -> None:
        """Update a network policy attribute's mutable fields.

        Args:
            attr (int | str | NetworkPolicyAttribute): NetworkPolicyAttribute instance, numeric ID, or name.
            description (str | UNSET): New description. Omit to leave unchanged.
        """
        attr = self._resolve(attr)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        self._patch(attr, data)

    def set_description(self, attr: int | str | NetworkPolicyAttribute, description: str) -> None:
        """Set the description for the attribute.

        Args:
            attr (int | str | NetworkPolicyAttribute): NetworkPolicyAttribute instance, numeric ID, or name.
            description (str): New description to set.
        """
        attr = self._resolve(attr)
        self.update(attr, description=description)

    @deprecated('Use "list_policies()" instead.')
    def get_policies(self, attr: int | str | NetworkPolicyAttribute) -> list[NetworkPolicy]:
        """List all network policies that use this attribute.

        Args:
            attr (int | str | NetworkPolicyAttribute): NetworkPolicyAttribute instance, numeric ID, or name.

        Returns:
            list[NetworkPolicy]: List of network policies that use this attribute.
        """
        return self.list_policies(attr)

    def list_policies(self, attr: int | str | NetworkPolicyAttribute) -> list[NetworkPolicy]:
        """List all network policies that use this attribute.

        Args:
            attr (int | str | NetworkPolicyAttribute): NetworkPolicyAttribute instance, numeric ID, or name.

        Returns:
            list[NetworkPolicy]: List of network policies that use this attribute.
        """
        attr = self._resolve(attr)
        return self._client.get_typed(
            Endpoint.NetworkPolicies, list[NetworkPolicy], params={"attributes": attr.id}
        )


class NetworkPolicyManager(NamedResourceManager[NetworkPolicy]):
    """Operations on NetworkPolicy resources."""

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
        community_template_pattern: str | None | UNSET = UNSET,
    ) -> NetworkPolicy:
        """Create a network policy.

        Args:
            name: The policy name (lowercased).
            description: Optional description.
            attributes: Optional list of attribute name/value pairs to attach at creation.
            community_template_pattern: Optional community name template pattern.

        Returns:
            NetworkPolicy: The created network policy.
        """
        data: dict[str, Any] = {"name": name, "description": description}
        if attributes is not None:
            data["attributes"] = [{"name": a.name, "value": a.value} for a in attributes]
        if community_template_pattern is not UNSET:
            data["community_template_pattern"] = community_template_pattern
        return self._create(data)

    def update(
        self,
        policy: int | str | NetworkPolicy,
        *,
        description: str | UNSET = UNSET,
        community_template_pattern: str | None | UNSET = UNSET,
    ) -> None:
        """Update a network policy's mutable fields.

        Pass `community_template_pattern=None` to unset it.

        Args:
            policy (int | str | NetworkPolicy): NetworkPolicy instance, numeric ID, or name string.
            description (str | UNSET): New description. Omit to leave unchanged.
            community_template_pattern (str | None | UNSET): New community name template pattern.
                Pass None to unset, omit to leave unchanged.
        """
        policy = self._resolve(policy)
        data: dict[str, Any] = {}
        if description is not UNSET:
            data["description"] = description
        if community_template_pattern is not UNSET:
            data["community_template_pattern"] = community_template_pattern
        self._patch(policy, data)

    def set_description(self, policy: int | str | NetworkPolicy, description: str) -> None:
        """Set the description for the policy.

        Args:
            policy (int | str | NetworkPolicy): NetworkPolicy instance, numeric ID, or name string.
            description (str): New description to set.
        """
        policy = self._resolve(policy)
        self.update(policy, description=description)

    def add_attribute(
        self,
        policy: int | str | NetworkPolicy,
        attr: str | NetworkPolicyAttribute,
        value: bool = True,
    ) -> None:
        """Add an attribute to a policy.

        Args:
            policy (int | str | NetworkPolicy): NetworkPolicy instance, numeric ID, or name string.
            attr (str | NetworkPolicyAttribute): The attribute to add.
            value (bool): The boolean value to set for the attribute. Defaults to True.

        Raises:
            EntityAlreadyExists: If the policy already has this attribute.
        """
        policy = self._resolve(policy)

        # NOTE: potential for mistakes to happen here! Can we rely on name matching via model method?
        attr = self._resolve_attribute_name(attr)
        if policy.get_attribute(attr):
            raise EntityAlreadyExists(f"Policy {policy.name!r} already has attribute {attr!r}.")
        attrs = [*policy.attributes, NetworkPolicyAttributeValue(name=attr, value=value)]
        self._patch(policy, {"attributes": [{"name": a.name, "value": a.value} for a in attrs]})

    def _resolve_attribute_name(self, attribute: str | NetworkPolicyAttribute) -> str:
        """Resolve an attribute reference (instance or name) to a string name."""
        if isinstance(attribute, NetworkPolicyAttribute):
            return attribute.name
        return str(attribute)

    def remove_attribute(
        self, policy: int | str | NetworkPolicy, attribute: str | NetworkPolicyAttribute
    ) -> None:
        """Remove an attribute from a policy.

        Args:
            policy (int | str | NetworkPolicy): NetworkPolicy instance, numeric ID, or name string.
            attribute (str | NetworkPolicyAttribute): NetworkPolicyAttribute instance or name string.

        Raises:
            EntityNotFound: If the policy does not have this attribute.
        """
        policy = self._resolve(policy)
        attribute_name = self._resolve_attribute_name(attribute)
        if not policy.get_attribute(attribute_name):
            raise EntityNotFound(f"Policy {policy.name!r} does not have attribute {attribute_name!r}.")
        attrs = [a for a in policy.attributes if a.name != attribute_name]
        self._patch(policy, {"attributes": [{"name": a.name, "value": a.value} for a in attrs]})

    def networks(self, policy: int | str | NetworkPolicy) -> list[Network]:
        """Get all networks that use this policy.

        Args:
            policy (int | str | NetworkPolicy): NetworkPolicy instance, numeric ID, or name string.
        """
        policy = self._resolve(policy)
        return self._client.get_typed(Endpoint.Networks, list[Network], params={"policy": policy.id})

    @functools.cached_property
    def attribute(self) -> NetworkPolicyAttributeManager:
        """Manager for network policy attributes."""
        return NetworkPolicyAttributeManager(self._client)


# NOTE: WHY does this not inherit from the regular ResourceManager?
class CommunityManager:
    """Operations on network communities.

    Communities are always scoped to a network — every method takes a network
    reference (address string or `Network` instance).
    """

    def __init__(self, client: MregClient) -> None:
        """Bind the manager to the client."""
        self._client = client
        self._network_manager: NetworkManager = client.network

    @property
    def model_name(self) -> str:
        """The name of the manager's model type."""
        return model_name(Community)

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
            network (str | int | Network): Network reference (address, ID, or Network instance).
        """
        addr = self._resolve_network_address(network)
        return self._client.get_typed(Endpoint.NetworkCommunities.with_params(addr), list[Community])

    def update(
        self,
        community: int | str | Community,
        network: str | int | Network,
        *,
        name: str | UNSET = UNSET,
        description: str | UNSET = UNSET,
    ) -> Community:
        """Update a community's mutable fields.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address, ID, or Network instance).
            name (str | UNSET): New name. Omit to leave unchanged.
            description (str | UNSET): New description. Omit to leave unchanged.
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
        self, name: str, network: str | int | Network, *, required: Literal[False]
    ) -> Community | None: ...
    @overload
    def get_by_name(
        self, name: str, network: str | int | Network, *, required: Literal[True] = ...
    ) -> Community: ...
    def get_by_name(
        self, name: str, network: str | int | Network, *, required: bool = True
    ) -> Community | None:
        """Get a community by name within a network.

        Args:
            name (str): The community name to look up.
            network (str | int | Network): Network reference (address, ID, or Network instance).
                Attempts to perform a direct lookup if a string is provided.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the community is not found.
        """
        # If network is a string, try to perform lookup directly
        if isinstance(network, str):
            resp = self._client.get_list_unique(
                Endpoint.NetworkCommunities.with_params(network), params={"name": name}
            )
            if resp is not None:
                return Community.model_validate(resp)
            if required:
                raise EntityNotFound(f"Community {name!r} not found in network {network!r}.")
            return None

        community = next((c for c in self.list(network) if c.name == name), None)
        if required and community is None:
            raise EntityNotFound(f"Community {name!r} not found.")
        return community

    @overload
    def get_by_id(
        self, community_id: int, network: str | int | Network, *, required: Literal[False]
    ) -> Community | None: ...
    @overload
    def get_by_id(
        self, community_id: int, network: str | int | Network, *, required: Literal[True] = ...
    ) -> Community: ...
    def get_by_id(
        self, community_id: int, network: str | int | Network, *, required: bool = True
    ) -> Community | None:
        """Get a community by ID within a network.

        Args:
            community_id (int): The community ID to look up.
            network (str | int | Network): Network reference (address, ID, or Network instance).
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the community is not found.
        """
        community: Community | None = None
        nw_addr = self._resolve_network_address(network)
        try:
            # Attempt to directly fetch the community
            community = self._client.get_typed(
                Endpoint.NetworkCommunity.with_params(nw_addr, community_id), Community
            )
        except EntityNotFound:
            if required:
                raise EntityNotFound(f"Community {community_id!r} not found.") from None
        return community

    @overload
    def get(
        self, community: int | str, network: str | int | Network, *, required: Literal[False]
    ) -> Community | None: ...
    @overload
    def get(
        self, community: int | str, network: str | int | Network, *, required: Literal[True] = ...
    ) -> Community: ...
    def get(
        self, community: int | str, network: str | int | Network, *, required: bool = True
    ) -> Community | None:
        """Get a community by ID or name within a network.

        Args:
            community (int | str): Community ID or name.
            network (str | int | Network): Network reference (address, ID, or Network instance).
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the community is not found.
        """
        if isinstance(community, int):
            com = next((c for c in self.list(network) if c.id == community), None)
        else:
            com = self.get_by_name(community, network, required=False)
        if required and com is None:
            raise EntityNotFound(f"Community {community!r} not found in network {network!r}.")
        return com

    # NOTE: this API should change! `network` is the last param in other methods,
    # but here it comes first.
    def create(self, network: str | int | Network, *, name: str, description: str) -> Community:
        """Create a community in a network.

        Args:
            network (str | int | Network): Network reference (address, ID, or Network instance).
            name (str): Name of the community.
            description (str): Description of the community.

        Returns:
            Community: The created community.
        """
        addr = self._resolve_network_address(network)
        resp = self._client.post(
            Endpoint.NetworkCommunities.with_params(addr),
            json={"name": name, "description": description},
        )
        return validate_response(Community, resp)

    def delete(self, community: int | str | Community, network: str | int | Network) -> None:
        """Delete a community from a network.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address, ID, or Network instance).
        """
        addr = self._resolve_network_address(network)
        community_id = self._resolve_community_id(community, network)
        self._client.delete(Endpoint.NetworkCommunity.with_params(addr, community_id))

    def get_hosts(self, community: int | str | Community, network: str | int | Network) -> list[Host]:
        """List all hosts in a community.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address, ID, or Network instance).
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
            network (str | int | Network): Network reference (address, ID, or Network instance).
            host (int | str | Host): Host reference (ID or Host instance).
            ipaddress (IP_AddressT | str | None): Optional IP address to associate with the host
                in this community. Required if host has multiple IP addresses in the network.
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
        *,
        ipaddress: IP_AddressT | str | None = None,
    ) -> None:
        """Remove a host from a community.

        Args:
            community (int | str | Community): Community ID, name or object.
                Using a name performs an extra lookup to resolve the ID.
            network (str | int | Network): Network reference (address, ID, or Network instance).
            host (int | str | Host): Host reference (ID or Host instance).
            ipaddress (IP_AddressT | str | None): Optional IP address to disassociate from the host
                in this community. Required if the host has multiple IP addresses in the community.
        """
        addr = self._resolve_network_address(network)
        community_id = self._resolve_community_id(community, network)
        host_id = resolve_host_id(host, self._client)
        data: dict[str, Any] = {}
        if ipaddress is not None:
            data["ipaddress"] = str(ipaddress)
        self._client.delete(Endpoint.NetworkCommunityHost.with_params(addr, community_id, host_id), json=data)


class NetworkManager(WriteResourceManager[Network]):
    """Operations on Network resources."""

    _path_param_field: ClassVar[str] = "network"

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
        """Manager for network policies."""
        return NetworkPolicyManager(self._client)

    @functools.cached_property
    def community(self) -> CommunityManager:
        """Manager for network community."""
        return CommunityManager(self._client)

    @overload
    def get_by_ip(self, ip: str | IP_AddressT, *, required: Literal[False]) -> Network | None: ...
    @overload
    def get_by_ip(self, ip: str | IP_AddressT, *, required: Literal[True] = ...) -> Network: ...
    def get_by_ip(self, ip: str | IP_AddressT, *, required: bool = True) -> Network | None:
        """Get the network containing an IP address.

        Args:
            ip (str | IP_AddressT): The IP address to look up.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and no network is found.
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
        vlan: int | None | UNSET = UNSET,
        dns_delegated: bool | UNSET = UNSET,
        category: str | UNSET = UNSET,
        location: str | UNSET = UNSET,
        frozen: bool | UNSET = UNSET,
        reserved: int | UNSET = UNSET,
    ) -> Network:
        """Create a network.

        Args:
            network (str): The network address in CIDR notation.
            description (str): Description of the network. Defaults to "".
            vlan (int | None | UNSET): VLAN ID. Pass None to unset, omit to leave unchanged.
            dns_delegated (bool | UNSET): Whether DNS is delegated. Omit to leave unchanged.
            category (str | UNSET): Network category. Omit to leave unchanged.
            location (str | UNSET): Network location. Omit to leave unchanged.
            frozen (bool | UNSET): Whether the network is frozen. Omit to leave unchanged.
            reserved (int | UNSET): Number of reserved addresses. Omit to leave unchanged.

        Returns:
            Network: The created network.
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
        return self._create(data)

    def update(
        self,
        network: str | int | Network,
        *,
        description: str | UNSET = UNSET,
        vlan: int | None | UNSET = UNSET,
        dns_delegated: bool | UNSET = UNSET,
        category: str | UNSET = UNSET,
        location: str | UNSET = UNSET,
        frozen: bool | UNSET = UNSET,
        reserved: int | UNSET = UNSET,
        policy: int | None | UNSET = UNSET,
        max_communities: int | None | UNSET = UNSET,
    ) -> None:
        """Update a network's mutable fields.

        Pass `policy=None` or `max_communities=None` to unset; omit to leave unchanged.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
            description (str | UNSET): New description. Omit to leave unchanged.
            vlan (int | None | UNSET): New VLAN ID. Pass None to unset, omit to leave unchanged.
            dns_delegated (bool | UNSET): Whether DNS is delegated. Omit to leave unchanged.
            category (str | UNSET): New category. Omit to leave unchanged.
            location (str | UNSET): New location. Omit to leave unchanged.
            frozen (bool | UNSET): Whether the network is frozen. Omit to leave unchanged.
            reserved (int | UNSET): Number of reserved addresses. Omit to leave unchanged.
            policy (int | None | UNSET): Network policy ID. Pass None to unset, omit to leave unchanged.
            max_communities (int | None | UNSET): Max communities. Pass None to unset, omit to leave unchanged.
        """  # noqa: E501
        network = self._resolve(network)
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
        self._patch(network, data)

    def get_first_available_ip(self, network: str | int | Network) -> IP_AddressT:
        """Return the first available IP address in the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return ipaddress.ip_address(
            self._client.get_typed(Endpoint.NetworksFirstUnused.with_params(network.network), str)
        )

    def get_random_available_ip(self, network: str | int | Network) -> IP_AddressT:
        """Return a random available IP address in the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return ipaddress.ip_address(
            self._client.get_typed(Endpoint.NetworksRandomUnused.with_params(network.network), str)
        )

    def get_used_count(self, network: str | int | Network) -> int:
        """Return the number of used IP addresses in the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return self._client.get_typed(Endpoint.NetworksUsedCount.with_params(network.network), int)

    def get_unused_count(self, network: str | int | Network) -> int:
        """Return the number of unused IP addresses in the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return self._client.get_typed(Endpoint.NetworksUnusedCount.with_params(network.network), int)

    def get_used_list(self, network: str | int | Network) -> list[IP_AddressT]:
        """Return the used IP addresses in the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return self._client.get_typed(
            Endpoint.NetworksUsedList.with_params(network.network), list[IP_AddressT]
        )

    def get_unused_list(self, network: str | int | Network) -> list[IP_AddressT]:
        """Return the unused IP addresses in the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return self._client.get_typed(
            Endpoint.NetworksUnusedList.with_params(network.network), list[IP_AddressT]
        )

    def get_reserved_ips(self, network: str | int | Network) -> list[IP_AddressT]:
        """Return the reserved IP addresses of the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return self._client.get_typed(
            Endpoint.NetworksReservedList.with_params(network.network), list[IP_AddressT]
        )

    def get_used_host_list(self, network: str | int | Network) -> dict[str, list[str]]:
        """Return a dict of used IP addresses to their associated hostnames.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return self._client.get_typed(
            Endpoint.NetworksUsedHostList.with_params(network.network), dict[str, list[str]]
        )

    def get_ptroverride_host_list(self, network: str | int | Network) -> dict[str, str]:
        """Return a dict of PTR override IPs to their associated hostnames.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return self._client.get_typed(
            Endpoint.NetworksPTROverrideHostList.with_params(network.network), dict[str, str]
        )

    def get_ptr_overrides(self, network: str | int | Network) -> list[IP_AddressT]:
        """Return IP addresses that have PTR overrides in the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
        """
        network = self._resolve(network)
        return self._client.get_typed(
            Endpoint.NetworksPTROverrideList.with_params(network.network), list[IP_AddressT]
        )

    def add_excluded_range(self, network: str | int | Network, start: str, end: str) -> None:
        """Add an excluded IP range to the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
            start (str): The start IP address of the excluded range.
            end (str): The end IP address of the excluded range.
        """
        network = self._resolve(network)
        start_ip = NetworkOrIP.parse_or_raise(start, mode="ip")
        end_ip = NetworkOrIP.parse_or_raise(end, mode="ip")
        if start_ip.version != end_ip.version:
            raise InputFailure("Start and end IP addresses must be of the same version.")
        self._client.post(
            Endpoint.NetworksAddExcludedRanges.with_params(network.network),
            json={"network": network.id, "start_ip": str(start_ip), "end_ip": str(end_ip)},
        )

    def remove_excluded_range(self, network: str | int | Network, start: str, end: str) -> None:
        """Remove an excluded IP range from the network.

        Args:
            network (str | int | Network): Network reference (address, numeric ID, or Network instance).
            start (str): The start IP address of the excluded range.
            end (str): The end IP address of the excluded range.
        """
        network = self._resolve(network)
        # NOTE: should we actually perform this check client-side? Network object could be stale.
        exrange = next(
            (r for r in network.excluded_ranges if str(r.start_ip) == start and str(r.end_ip) == end),
            None,
        )
        if exrange is None:
            raise EntityNotFound(f"Excluded range {start} - {end} not found in {network.network!r}.")
        self._client.delete(Endpoint.NetworksRemoveExcludedRanges.with_params(network.network, exrange.id))

    def list_by_policy(self, policy: int | NetworkPolicy) -> list[Network]:
        """List networks that share the same policy as the given network.

        Alias for `policy.networks()`

        Args:
            policy (int | NetworkPolicy): NetworkPolicy instance or numeric ID.
        """
        # re-use NetworkPolicy.networks for this
        return self.policy.networks(policy)


class IPAddressManager(WriteResourceManager[IPAddress]):
    """Operations on IPAddress resources."""

    @property
    @override
    def model(self) -> type[IPAddress]:
        return IPAddress

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Ipaddresses

    # TODO: add IP_AddressT support for `get()` (eventually...!)
    #       Requires redefining overloads. Lots of boilerplate.

    def _list_by_ip(self, ip: str) -> list[IPAddress]:
        """Common interface for fetching all IPs matching a given IP address string.

        Used by multiple methods. Avoids repeating `_fetch_list_by_field` calls everywhere.
        """
        return self._fetch_list_by_field("ipaddress", ip)

    def _resolve(self, ref: int | str | IP_AddressT | IPAddress) -> IPAddress:
        """Resolve an IP address argument by ID, name, stdlib IP address object or IPAddress object.

        All argument types but IPAddress objects perform a GET call.
        """
        if isinstance(ref, str | IP_AddressT):
            ipaddr = str(ref)
            # NOTE: this is a bit of a hack, but we don't have a way to fetch by IP address directly
            # so we fetch the list of IP addresses with this IP and return the first one.
            ip_list = self._list_by_ip(ipaddr)
            if not ip_list:
                raise EntityNotFound(f"IP address {ipaddr!r} not found.")
            return ip_list[0]
        return super()._resolve(ref)

    def create(
        self,
        *,
        ipaddress: str | IP_AddressT,
        host: int | str | Host | None = None,
        macaddress: str | MacAddress | None = None,
    ) -> IPAddress:
        """Create an IP address record.

        Args:
            ipaddress (str | IP_AddressT): The IP address to assign.
            host (int | str | Host): Host instance, name, or numeric ID.
            macaddress (str | MacAddress | None): Optional MAC address to associate. Pass None to omit.

        Returns:
            IPAddress: The created IP address record.
        """
        data: dict[str, Any] = {"ipaddress": str(ipaddress)}
        if macaddress is not None:
            data["macaddress"] = str(macaddress)
        if host is not None:
            data["host"] = resolve_host_id(host, self._client)
        return self._create(data)

    def update(
        self,
        ip: int | str | IP_AddressT | IPAddress,
        *,
        ipaddress: IP_AddressT | str | UNSET = UNSET,
        macaddress: str | MacAddress | None | UNSET = UNSET,
        host: int | str | Host | UNSET | None = UNSET,
    ) -> None:
        """Update an IP address record's mutable fields.

        Args:
            ip (int | str | IP_AddressT | IPAddress): IPAddress instance, numeric ID, or IP address string.
            ipaddress (IP_AddressT | str | UNSET): New IP address. Omit to leave unchanged.
            macaddress (str | MacAddress | None | UNSET): New MAC address. Pass None to unset,
                omit to leave unchanged.
            host (int | str | Host | UNSET): Host to (dis)associate with IP. Omit to leave unchanged.
        """
        ip = self._resolve(ip)
        data: dict[str, Any] = {}
        if ipaddress is not UNSET:
            data["ipaddress"] = str(ipaddress)
        if macaddress is not UNSET:
            data["macaddress"] = str(macaddress) if macaddress is not None else ""
        if host is not UNSET:
            # TODO: decide if we can modify resolve_host_id to return int | None
            # on a general basis, or if it's only applicable here
            if host is not None:
                host_id = resolve_host_id(host, self._client)
            else:
                host_id = host
            data["host"] = host_id
        self._patch(ip, data)

    def associate_mac(
        self, ip: int | str | IP_AddressT | IPAddress, mac: str | MacAddress, *, force: bool = False
    ) -> None:
        """Associate a MAC address with an IP address.

        Args:
            ip (int | str | IP_AddressT | IPAddress): IPAddress instance, numeric ID, or IP address string.
            mac (str | MacAddress): The MAC address to associate.
            force (bool): When True, skip safety checks and overwrite an existing MAC.

        Raises:
            EntityAlreadyExists: If the IP already has a MAC and `force` is False.
        """
        ip = self._resolve(ip)
        if ip.macaddress and not force:
            raise EntityAlreadyExists(f"IP address {ip.ipaddress} already has MAC address {ip.macaddress}.")
        self.update(ip, macaddress=mac)

    def disassociate_mac(self, ip: int | str | IP_AddressT | IPAddress) -> None:
        """Remove the MAC address from an IP address.

        Args:
            ip (int | str | IP_AddressT | IPAddress): IPAddress instance, numeric ID, or IP address string.
        """
        ip = self._resolve(ip)
        self.update(ip, macaddress=None)

    def list_by_host(self, host: int | str | Host) -> list[IPAddress]:
        """List all IP address records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)

    def list_by_ip(self, ip: int | str | IP_AddressT | IPAddress) -> list[IPAddress]:
        """List all IP address records with a given IP address.

        Args:
            ip (IP_AddressT | str): The IP address to filter by.
        """
        if isinstance(ip, (str, IP_AddressT)):
            ipaddr = str(ip)
        else:
            ipaddr = str(self._resolve(ip).ipaddress)
        return self._list_by_ip(ipaddr)

    def list_by_mac(self, mac: str | MacAddress) -> list[IPAddress]:
        """List all IP address records with a given MAC address.

        Args:
            mac (str | MacAddress): The MAC address to filter by.
        """
        addr = MacAddress.parse_or_raise(mac)
        return self._fetch_list_by_field("macaddress", str(addr))


class CNAMEManager(NamedResourceManager[CNAME]):
    """Operations on CNAME resources."""

    _path_param_field: ClassVar[str] = "name"

    @property
    @override
    def model(self) -> type[CNAME]:
        return CNAME

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Cnames

    @override
    def _normalize_name(self, name: str) -> str:
        # NOTE: this does not respect name casing defined by `name_lowercase`!
        return self._client.fqdn(name)

    @override
    def _normalize_path_param(self, ident: str | int) -> str | int:
        return self._normalize_name(ident) if isinstance(ident, str) else ident

    def create(
        self,
        *,
        host: int | str | Host,
        name: str | HostName,
    ) -> CNAME:
        """Create a CNAME record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            name (str | HostName): The alias name for the CNAME.

        Returns:
            CNAME: The created CNAME record.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create({"host": str(host_id), "name": self._client.fqdn(name)})

    def update(
        self,
        cname: int | str | CNAME,
        *,
        host: int | str | Host | UNSET = UNSET,
        name: str | HostName | UNSET = UNSET,
        ttl: int | None | UNSET = UNSET,
    ) -> None:
        """Update a CNAME record's mutable fields. Pass `ttl=None` to reset to default.

        Args:
            cname (int | CNAME): CNAME instance or numeric ID.
            host (int | str | Host | UNSET): New host reference. Omit to leave unchanged.
            name (str | HostName | UNSET): New alias name. Omit to leave unchanged.
            ttl (int | None | UNSET): New TTL. Pass None to reset to default, omit to leave unchanged.
        """
        cname = self._resolve(cname)
        data: dict[str, Any] = {}
        if host is not UNSET:
            data["host"] = resolve_host_id(host, self._client)
        if name is not UNSET:
            data["name"] = self._client.fqdn(str(name))
        if ttl is not UNSET:
            data["ttl"] = ttl
        self._patch(cname, data)

    @overload
    def get_by_name(self, name: str, *, required: Literal[False]) -> CNAME | None: ...
    @overload
    def get_by_name(self, name: str, *, required: Literal[True] = ...) -> CNAME: ...
    def get_by_name(self, name: str, *, required: bool = True) -> CNAME | None:
        """Get a CNAME record by alias name.

        Args:
            name (str): The alias name to look up.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the CNAME is not found.
        """
        obj = self._fetch_by_field("name", self._client.fqdn(name))
        if required and obj is None:
            raise EntityNotFound(f"CNAME {name!r} not found.")
        return obj

    def get_by_host_and_name(
        self, host: int | str | Host, name: str | HostName, *, required: bool = True
    ) -> CNAME | None:
        """Get a CNAME record matching both the host and alias name.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            name (str | HostName): The alias name to look up.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the CNAME is not found.
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
    """Operations on HInfo resources.

    HInfo is a 1-per-host record; the path parameter is the host ID (not a numeric row id).
    """

    _path_param_field: ClassVar[str] = "host"

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
    ) -> HInfo:
        """Create an HInfo record.

        Args:
            host (int | str | Host): Host ID, name or instance.
            cpu (str): CPU hardware type string.
            os (str): Operating system string.

        Returns:
            HInfo: The created HInfo record.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create({"host": host_id, "cpu": cpu, "os": os})

    def update(
        self,
        hinfo: int | HInfo,
        *,
        cpu: str | UNSET = UNSET,
        os: str | UNSET = UNSET,
    ) -> None:
        """Update an HInfo record's mutable fields.

        Args:
            hinfo (int | HInfo): HInfo instance or numeric ID.
            cpu (str | UNSET): New CPU hardware type string. Omit to leave unchanged.
            os (str | UNSET): New operating system string. Omit to leave unchanged.
        """
        hinfo = self._resolve(hinfo)
        data: dict[str, Any] = {}
        if cpu is not UNSET:
            data["cpu"] = cpu
        if os is not UNSET:
            data["os"] = os
        self._patch(hinfo, data)

    def get_by_host(self, host: int | str | Host, *, required: bool = True) -> HInfo | None:
        """Get the HInfo record for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and no HInfo record is found.
        """
        host_id = resolve_host_id(host, self._client)
        obj = self._fetch_by_field("host", host_id)
        if required and obj is None:
            raise EntityNotFound(f"HInfo for host id {host_id!r} not found.")
        return obj


class TXTManager(WriteResourceManager[TXT]):
    """Operations on TXT resources."""

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
    ) -> TXT:
        """Create a TXT record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            txt (str): The TXT record value.

        Returns:
            TXT: The created TXT record.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create({"host": host_id, "txt": txt})

    # NOTE: This method does not follow the naming convention of other manager methods
    # due to name collision with own field. The first positional arg is usually
    # named after the resource type, but in this case one of the fields has
    # the same name as the resource itself (txt) and can't be shortened further.
    # This applies to many resource record types, since they often have
    # fields named after themselves.
    def update(
        self,
        ref: int | TXT,
        *,
        txt: str | UNSET = UNSET,
    ) -> None:
        """Update a TXT record's mutable fields.

        Args:
            ref (int | TXT): TXT instance or numeric ID.
            txt (str | UNSET): New TXT record value. Omit to leave unchanged.
        """
        txt_obj = self._resolve(ref)
        data: dict[str, Any] = {}
        if txt is not UNSET:
            data["txt"] = txt
        self._patch(txt_obj, data)

    def list_by_host(self, host: int | str | Host) -> list[TXT]:
        """List all TXT records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)


class MXManager(WriteResourceManager[MX]):
    """Operations on MX resources."""

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
    ) -> MX:
        """Create an MX record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            mx (str): The mail exchange hostname.
            priority (int): The MX priority value.

        Returns:
            MX: The created MX record.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create(
            {"host": host_id, "mx": mx, "priority": priority},
        )

    def update(
        self,
        ref: int | MX,
        *,
        mx: str | UNSET = UNSET,
        priority: int | UNSET = UNSET,
    ) -> None:
        """Update an MX record's mutable fields.

        Args:
            ref (int | MX): MX instance or numeric ID.
            mx (str | UNSET): New mail exchange hostname. Omit to leave unchanged.
            priority (int | UNSET): New priority value. Omit to leave unchanged.
        """
        mx_obj = self._resolve(ref)
        data: dict[str, Any] = {}
        if mx is not UNSET:
            data["mx"] = mx
        if priority is not UNSET:
            data["priority"] = priority
        self._patch(mx_obj, data)

    def list_by_host(self, host: int | str | Host) -> list[MX]:
        """List all MX records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)

    @overload
    def get_unique(
        self, host: int | str | Host, mx: str, priority: int, *, required: Literal[False]
    ) -> MX | None: ...
    @overload
    def get_unique(
        self, host: int | str | Host, mx: str, priority: int, *, required: Literal[True] = ...
    ) -> MX: ...
    def get_unique(
        self, host: int | str | Host, mx: str, priority: int, *, required: bool = True
    ) -> MX | None:
        """Get an MX record matching host, mx value, and priority.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            mx (str): The mail exchange hostname.
            priority (int): The MX priority value.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If no matching MX record exists and `required` is True.

        Returns:
            MX | None: The matching MX record, or None if not found and `required` is False.
        """
        host_id = resolve_host_id(host, self._client)
        obj = self._client.get_list_unique(
            self.endpoint,
            params={"host": str(host_id), "mx": mx, "priority": str(priority)},
            ok404=True,
        )
        if obj:
            return MX.model_validate(obj)

        if required:
            raise EntityNotFound(f"MX {mx!r} with priority {priority} not found for host {host_id}.")
        return None

    # NOTE: This is a somewhat clumsy interface. It would be better if we had
    # a get method that takes in `host`, `mx`, `priority` perhaps?
    # Although that would remove the ability to fetch by ID (how common is that?)
    # A similar method is named `get_by_triplet` in `PermissionManager`.
    # Maybe a better general-purpose name would be something like `get_unique`? or something
    #
    # Alternatively, `get()` needs to be decoupled from `ResourceManager` and redesigned to
    # have a bespoke implementation per-manager, with different parameters for each,
    # denoting the "canonical" way to fetch the resource (name, triplet, etc.),
    # and then move fetching by ID to a separate `get_by_id` method.
    @deprecated('Use "get_unique" instead.')
    def get_by_all(self, host: int | str | Host, mx: str, priority: int) -> MX:
        """Get an MX record matching host, mx value, and priority.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            mx (str): The mail exchange hostname.
            priority (int): The MX priority value.

        Raises:
            EntityNotFound: If no matching MX record exists.
        """
        return self.get_unique(host, mx, priority)


class NAPTRManager(WriteResourceManager[NAPTR]):
    """Operations on NAPTR resources."""

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
    ) -> NAPTR:
        """Create a NAPTR record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            preference (int): The NAPTR preference value.
            order (int): The NAPTR order value.
            flag (str): The NAPTR flag. Defaults to "".
            service (str): The NAPTR service. Defaults to "".
            regex (str): The NAPTR regular expression. Defaults to "".
            replacement (str): The NAPTR replacement string.

        Returns:
            NAPTR: The created NAPTR record.
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
        )

    def update(
        self,
        naptr: int | NAPTR,
        *,
        preference: int | UNSET = UNSET,
        order: int | UNSET = UNSET,
        flag: str | UNSET = UNSET,
        service: str | UNSET = UNSET,
        regex: str | UNSET = UNSET,
        replacement: str | UNSET = UNSET,
    ) -> None:
        """Update a NAPTR record's mutable fields.

        Args:
            naptr (int | NAPTR): NAPTR instance or numeric ID.
            preference (int | UNSET): New preference value. Omit to leave unchanged.
            order (int | UNSET): New order value. Omit to leave unchanged.
            flag (str | UNSET): New flag. Omit to leave unchanged.
            service (str | UNSET): New service. Omit to leave unchanged.
            regex (str | UNSET): New regular expression. Omit to leave unchanged.
            replacement (str | UNSET): New replacement string. Omit to leave unchanged.
        """
        naptr = self._resolve(naptr)
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
        self._patch(naptr, data)

    # TODO: ensure overload is sound; does `= ...` capture both defaults and explicit str args?
    @overload
    def get_unique(
        self,
        host: int | str | Host,
        preference: int,
        order: int,
        flag: str = ...,
        service: str = ...,
        regex: str = ...,
        replacement: str = ...,
        *,
        required: Literal[False],
    ) -> NAPTR | None: ...
    @overload
    def get_unique(
        self,
        host: int | str | Host,
        preference: int,
        order: int,
        flag: str = ...,
        service: str = ...,
        regex: str = ...,
        replacement: str = ...,
        *,
        required: Literal[True] = ...,
    ) -> NAPTR: ...
    def get_unique(
        self,
        host: int | str | Host,
        preference: int,
        order: int,
        flag: str = "",
        service: str = "",
        regex: str = "",
        replacement: str = "",
        *,
        required: bool = True,
    ) -> NAPTR | None:
        """Get a NAPTR record matching host, preference, order, flag, service, regex, and replacement.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            preference (int): The NAPTR preference value.
            order (int): The NAPTR order value.
            flag (str): The NAPTR flag. Defaults to "".
            service (str): The NAPTR service. Defaults to "".
            regex (str): The NAPTR regular expression. Defaults to "".
            replacement (str): The NAPTR replacement string.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If no matching NAPTR record exists and `required` is True.

        Returns:
            NAPTR | None: The matching NAPTR record, or None if not found and `required` is False.
        """
        host_id = resolve_host_id(host, self._client)
        obj = self._client.get_list_unique(
            self.endpoint,
            params={
                "host": str(host_id),
                "preference": preference,
                "order": order,
                "flag": flag,
                "service": service,
                "regex": regex,
                "replacement": replacement,
            },
            ok404=True,
        )
        if obj:
            return NAPTR.model_validate(obj)

        if required:
            raise EntityNotFound(
                f"NAPTR with preference {preference}, order {order}, flag {flag}, "  # pyright: ignore[reportImplicitStringConcatenation]
                f"service {service}, regex {regex}, replacement {replacement} not found "
                f"for host {host_id}."
            )
        return None

    def list_by_host(self, host: int | str | Host) -> list[NAPTR]:
        """List all NAPTR records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)


class SrvManager(WriteResourceManager[Srv]):
    """Operations on Srv resources."""

    @property
    @override
    def model(self) -> type[Srv]:
        return Srv

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Srvs

    @overload
    def get_unique(
        self,
        *,
        name: str,
        priority: int,
        weight: int,
        port: int,
        host: int | str | Host,
        required: Literal[False],
    ) -> Srv | None: ...
    @overload
    def get_unique(
        self,
        *,
        name: str,
        priority: int,
        weight: int,
        port: int,
        host: int | str | Host,
        required: Literal[True] = ...,
    ) -> Srv: ...
    def get_unique(
        self,
        *,
        name: str,
        priority: int,
        weight: int,
        port: int,
        host: int | str | Host,
        required: bool = True,
    ) -> Srv | None:
        """Get a SRV record matching name, priority, weight, port, and host."""
        host_id = resolve_host_id(host, self._client)
        obj = self._client.get_list_unique(
            self.endpoint,
            params={
                "name": name,
                "priority": priority,
                "weight": weight,
                "port": port,
                "host": host_id,
            },
        )
        if obj:
            return Srv.model_validate(obj)

        if required:
            raise EntityNotFound(
                f"SRV {name!r} with priority {priority}, weight {weight}, port {port} "  # pyright: ignore[reportImplicitStringConcatenation]
                f"not found for host {host_id}."
            )
        return None

    def create(
        self,
        *,
        host: int | str | Host,
        name: str,
        priority: int,
        weight: int,
        port: int,
        ttl: int | None | UNSET = UNSET,
    ) -> Srv:
        """Create a SRV record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            name (str): The SRV service name.
            priority (int): The SRV priority value.
            weight (int): The SRV weight value.
            port (int): The SRV port number.
            ttl (int | None | UNSET): TTL. Pass None to use default, omit to leave unchanged.

        Returns:
            Srv: The created SRV record.
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
        return self._create(data)

    def update(
        self,
        srv: int | Srv,
        *,
        name: str | UNSET = UNSET,
        priority: int | UNSET = UNSET,
        weight: int | UNSET = UNSET,
        port: int | UNSET = UNSET,
        ttl: int | None | UNSET = UNSET,
    ) -> None:
        """Update a SRV record's mutable fields. Pass `ttl=None` to reset to default.

        Args:
            srv (int | Srv): Srv instance or numeric ID.
            name (str | UNSET): New service name. Omit to leave unchanged.
            priority (int | UNSET): New priority value. Omit to leave unchanged.
            weight (int | UNSET): New weight value. Omit to leave unchanged.
            port (int | UNSET): New port number. Omit to leave unchanged.
            ttl (int | None | UNSET): New TTL. Pass None to reset to default, omit to leave unchanged.
        """
        srv = self._resolve(srv)
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
        self._patch(srv, data)

    def list_by_host(self, host: int | str | Host) -> list[Srv]:
        """List all SRV records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)


class PTROverrideManager(WriteResourceManager[PTR_override]):
    """Operations on PTR_override resources."""

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
    ) -> PTR_override:
        """Create a PTR override record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            ipaddress (IP_AddressT | str): The IP address for the PTR override.

        Returns:
            PTR_override: The created PTR override record.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create(
            {"host": host_id, "ipaddress": str(ipaddress)},
        )

    def update(
        self,
        ptr: int | PTR_override,
        *,
        host: int | str | Host | UNSET = UNSET,
        ipaddress: IP_AddressT | str | UNSET = UNSET,
    ) -> None:
        """Update a PTR override record's mutable fields.

        Args:
            ptr (int | PTR_override): PTR_override instance or numeric ID.
            host (int | str | Host | UNSET): New host reference. Omit to leave unchanged.
            ipaddress (IP_AddressT | str | UNSET): New IP address. Omit to leave unchanged.
        """
        ptr = self._resolve(ptr)
        data: dict[str, Any] = {}
        if host is not UNSET:
            data["host"] = resolve_host_id(host, self._client)
        if ipaddress is not UNSET:
            data["ipaddress"] = str(ipaddress)
        self._patch(ptr, data)

    # NOTE: potential for `get_by_ip()` here, since ipaddress field is unique per record

    def list_by_host(self, host: int | str | Host) -> list[PTR_override]:
        """List all PTR override records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)


class SSHFPManager(WriteResourceManager[SSHFP]):
    """Operations on SSHFP resources."""

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
        ttl: int | None | UNSET = UNSET,
    ) -> SSHFP:
        """Create an SSHFP record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            algorithm (int): The SSHFP algorithm number.
            hash_type (int): The SSHFP hash type number.
            fingerprint (str): The SSH key fingerprint.
            ttl (int | None | UNSET): TTL. Pass None to use default, omit to leave unchanged.

        Returns:
            SSHFP: The created SSHFP record.
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
        return self._create(data)

    def update(
        self,
        sshfp: int | SSHFP,
        *,
        algorithm: int | UNSET = UNSET,
        hash_type: int | UNSET = UNSET,
        fingerprint: str | UNSET = UNSET,
        ttl: int | None | UNSET = UNSET,
    ) -> None:
        """Update an SSHFP record's mutable fields. Pass `ttl=None` to reset to default.

        Args:
            sshfp (int | SSHFP): SSHFP instance or numeric ID.
            algorithm (int | UNSET): New algorithm number. Omit to leave unchanged.
            hash_type (int | UNSET): New hash type number. Omit to leave unchanged.
            fingerprint (str | UNSET): New fingerprint. Omit to leave unchanged.
            ttl (int | None | UNSET): New TTL. Pass None to reset to default, omit to leave unchanged.
        """
        sshfp = self._resolve(sshfp)
        data: dict[str, Any] = {}
        if algorithm is not UNSET:
            data["algorithm"] = algorithm
        if hash_type is not UNSET:
            data["hash_type"] = hash_type
        if fingerprint is not UNSET:
            data["fingerprint"] = fingerprint
        if ttl is not UNSET:
            data["ttl"] = ttl
        self._patch(sshfp, data)

    def list_by_host(self, host: int | str | Host) -> list[SSHFP]:
        """List all SSHFP records for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
        """
        host_id = resolve_host_id(host, self._client)
        return self._fetch_list_by_field("host", host_id)


class BacnetIDManager(WriteResourceManager[BacnetID]):
    """Operations on BacnetID resources."""

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
    ) -> BacnetID:
        """Create a BacnetID record.

        Args:
            host: The host to create a BacnetID for.
            id: The BACnet id.

        Returns:
            BacnetID: The created BacnetID record.
        """
        return self._create(
            {"hostname": resolve_host_name(host, self._client), "id": id},
        )

    # NOTE: PATCH and PUT not allowed for this endpoint!

    def list_in_range(self, start: int, end: int) -> list[BacnetID]:
        """List BacnetID records within a numeric id range (inclusive).

        Args:
            start (int): The start of the BACnet ID range (inclusive).
            end (int): The end of the BACnet ID range (inclusive).
        """
        return self._client.get_typed(self.endpoint, list[BacnetID], params={"id__range": f"{start},{end}"})

    def get_by_host(self, host: str | HostName | Host, *, required: bool = True) -> BacnetID | None:
        """Get the BacnetID record for a host by its FQDN.

        Args:
            host (str | HostName | Host): Host reference (name string or Host instance).
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and no BacnetID record is found.
        """
        name = resolve_host_name(host, self._client)
        obj = self._fetch_by_field("hostname", name)
        if required and obj is None:
            raise EntityNotFound(f"BacnetID record for host {name!r} not found.")
        return obj


class LocationManager(WriteResourceManager[Location]):
    """Operations on Location resources."""

    _path_param_field: ClassVar[str] = "host"

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
    ) -> Location:
        """Create a LOC record.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            loc (str): The LOC record value.

        Returns:
            Location: The created LOC record.
        """
        host_id = resolve_host_id(host, self._client)
        return self._create({"host": host_id, "loc": loc})

    def update(
        self,
        location: int | Location,
        *,
        loc: str | UNSET = UNSET,
    ) -> None:
        """Update a LOC record's mutable fields.

        Args:
            location (int | Location): Location instance or numeric ID.
            loc (str | UNSET): New LOC record value. Omit to leave unchanged.
        """
        location = self._resolve(location)
        data: dict[str, Any] = {}
        if loc is not UNSET:
            data["loc"] = loc
        self._patch(location, data)

    def get_by_host(self, host: int | str | Host, *, required: bool = True) -> Location | None:
        """Get the LOC record for a host.

        Args:
            host (int | str | Host): Host instance or numeric ID.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and no LOC record is found.
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


def _verify_nameservers(client: MregClient, nameservers: list[str], force: bool = False) -> list[VerifiedNS]:
    """Verify nameservers exist in mreg and have an A-record / glue.

    Utility function shared by various managers that interact with nameservers.

    Raises:
        InputFailure: If no nameservers are given.
        ForceMissing: If a nameserver is missing (or lacks glue) and `force` is False.
    """
    if not nameservers:
        raise InputFailure("At least one nameserver is required")

    errors: list[str] = []
    verified: list[VerifiedNS] = []

    for nameserver in nameservers:
        # HACK: bypass checks if force is enabled, return name as-is, expanded and validated
        ns = client.fqdn(nameserver)
        if force:
            verified.append(VerifiedNS(ns))
            continue

        host = client.host.get_by_name(ns, required=False)
        if host is None:
            cname = client.cname.get_by_name(ns, required=False)
            if cname is not None:
                host = client.host.get_by_id(cname.host, required=False)
                if host is not None:
                    client.events.record(
                        Event(
                            kind=EventKind.RESOLUTION,
                            message=f"{ns} is a CNAME for {host.name}",
                            subject=ObjectRef.new(host),
                            related=(ObjectRef.new(cname),),
                            correlation_id=client.get_correlation_id(),
                        )
                    )

        if host is None:
            if not force:
                errors.append(f"{ns} is not in mreg, must force")
        else:
            if host.zone is None and not host.ipaddresses and not force:
                errors.append(f"{ns} has no A-record/glue, must force")

        # All checks passed, consider it verified
        verified.append(VerifiedNS(ns))

    if errors:
        raise ForceMissing("\n".join(errors))

    return verified


_ZoneT = TypeVar("_ZoneT", bound=Zone)


class _ZoneSubManager(NamedResourceManager[_ZoneT], ABC):
    """Base class for forward/reverse Zone managers.

    Each concrete subclass binds a zone subclass (and unique endpoint).
    Accessed via the `ZoneManager` by zone-name or concrete `Zone` subclass type.
    """

    _path_param_field: ClassVar[str] = "name"
    nameservers_endpoint: ClassVar[Endpoint]
    """The per-type nameservers endpoint (forward/reverse differ)."""

    def create(
        self,
        *,
        name: str,
        email: str,
        primary_ns: list[VerifiedNS],
    ) -> _ZoneT:
        """Create a zone of this manager's type. Caller verifies nameservers/absence.

        Args:
            name (str): The zone name.
            email (str): The zone admin email address.
            primary_ns (list[VerifiedNS]): List of primary nameserver names.

        Returns:
            _ZoneT: The created zone.
        """
        return self._create(
            {"name": name, "email": email, "primary_ns": primary_ns},
        )

    def update_soa(
        self,
        zone: _ZoneT,
        *,
        primary_ns: VerifiedNS | UNSET = UNSET,
        email: str | UNSET = UNSET,
        serialno: int | UNSET = UNSET,
        refresh: int | UNSET = UNSET,
        retry: int | UNSET = UNSET,
        expire: int | UNSET = UNSET,
        soa_ttl: int | UNSET = UNSET,
    ) -> None:
        """Update the zone's SOA fields. At least one field must be provided.

        Args:
            zone (_ZoneT): The zone to update.
            primary_ns (VerifiedNS | UNSET): New primary nameserver. Omit to leave unchanged.
            email (str | UNSET): New zone admin email. Omit to leave unchanged.
            serialno (int | UNSET): New serial number. Omit to leave unchanged.
            refresh (int | UNSET): New refresh interval. Omit to leave unchanged.
            retry (int | UNSET): New retry interval. Omit to leave unchanged.
            expire (int | UNSET): New expire interval. Omit to leave unchanged.
            soa_ttl (int | UNSET): New SOA TTL. Omit to leave unchanged.
        """
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
        self._patch(zone, data)

    def set_default_ttl(self, zone: _ZoneT, ttl: int) -> None:
        """Set the zone's default TTL.

        Args:
            zone (_ZoneT): The zone to update.
            ttl (int): The new default TTL value (300–68400).
        """
        self._patch(zone, {"default_ttl": _valid_zone_ttl(ttl)})

    def set_nameservers(self, zone: _ZoneT, nameservers: list[VerifiedNS]) -> None:
        """Replace the zone's nameservers (hits the per-type nameservers endpoint).

        Args:
            zone (_ZoneT): The zone to update.
            nameservers (list[VerifiedNS]): The new list of nameserver names.
        """
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
        # XXX: Not foolproof (e.g. SRVs are not hosts), but added for parity with old Zone.ensure_deletable.
        hosts = self._client.host.list(zone=zone.id)
        if hosts:
            raise DeleteError(f"Zone has {len(hosts)} registered entries. Can not delete.")
        subzones = self.list_subzones(zone)
        if subzones:
            names = ", ".join(z.name for z in subzones)
            raise DeleteError(f"Zone has registered subzones: '{names}'. Can not delete")

    # NOTE: force should not propagate to this method.
    # Ideally, we resolve all safety issues in the ZoneManager itself.
    @override
    def delete(self, obj: int | str | _ZoneT, *, force: bool = False) -> None:
        """Delete the zone, guarding against non-empty zones unless `force`.

        Args:
            obj (int | str | _ZoneT): The zone to delete, by numeric ID, name string, or instance.
            force (bool): When True, skip safety checks and delete even non-empty zones.
        """
        obj = self._resolve(obj)
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

        # TODO: use Pydantic JsonMapping validator for this
        # These checks are dangerous! `in` checks and direct key access may break.
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


# NOTE: If we need to support resolving zones by ID, we must expose the sub managers
# as public (standalone?) managers. Currently, we can't just make the existing
# sub managers public, because their methods expect verified nameserver arguments (`VeriifedNS`),
# which we only produce through a private function (_verify_nameservers).
class ZoneManager:
    """Public facade over the forward/reverse zone managers.

    Zones split into forward/reverse, but are otherwise very similar in their APIs.
    This manager delegates to the correct forward/reverse manager based on the name
    or object types passed in to methods.

    Similar to other managers, methods take names or instances, but crucially NOT IDs,
    since we cannot distinguish between forward/reverse zones by ID alone.
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

    def verify_nameservers(self, nameservers: list[str], force: bool = False) -> list[VerifiedNS]:
        """Verify nameservers exist in mreg and have glue (raises otherwise).

        Args:
            nameservers (list[str]): List of nameserver names to verify.
            force (bool): When True, skip safety checks on nameserver existence.
        """
        return _verify_nameservers(self._client, nameservers, force=force)

    @overload
    def get(self, name: str, *, required: Literal[False]) -> ForwardZone | ReverseZone | None: ...
    @overload
    def get(self, name: str, *, required: Literal[True] = ...) -> ForwardZone | ReverseZone: ...
    def get(self, name: str, *, required: bool = True) -> ForwardZone | ReverseZone | None:
        """Get a zone by name; forward/reverse chosen by name shape.

        Alias for `get_by_name`.

        Args:
            name (str): The zone name to look up.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the zone is not found.
        """
        return self._sub_for_name(name).get_by_name(name, required=required)

    @overload
    def get_by_name(self, name: str, *, required: Literal[False]) -> ForwardZone | ReverseZone | None: ...
    @overload
    def get_by_name(self, name: str, *, required: Literal[True] = ...) -> ForwardZone | ReverseZone: ...
    def get_by_name(self, name: str, *, required: bool = True) -> ForwardZone | ReverseZone | None:
        """Get a zone by name; forward/reverse chosen by name shape.

        Args:
            name (str): The zone name to look up.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the zone is not found.
        """
        return self._sub_for_name(name).get_by_name(name, required=required)

    def assert_absent(self, name: str) -> None:
        """Raise EntityAlreadyExists if a zone with `name` exists.

        Args:
            name (str): The zone name to check.
        """
        self._sub_for_name(name).assert_absent(name)

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
    ) -> ForwardZone | ReverseZone:
        """Create a forward or reverse zone (type chosen by name shape).

        Verifies the nameservers and that no zone with this name exists first.

        Args:
            name (str): The zone name (determines forward vs. reverse by shape).
            email (str): The zone admin email address.
            primary_ns (list[str]): List of primary nameserver names.
            force (bool): When True, skip safety checks on nameservers.

        Returns:
            ForwardZone | ReverseZone: The created zone.
        """
        verified_ns = self.verify_nameservers(primary_ns, force=force)
        sub = self._sub_for_name(name)
        sub.assert_absent(name)
        return sub.create(name=name, email=email, primary_ns=verified_ns)

    def update_soa(
        self,
        zone: str | ForwardZone | ReverseZone,
        *,
        primary_ns: str | UNSET = UNSET,
        email: str | UNSET = UNSET,
        serialno: int | UNSET = UNSET,
        refresh: int | UNSET = UNSET,
        retry: int | UNSET = UNSET,
        expire: int | UNSET = UNSET,
        soa_ttl: int | UNSET = UNSET,
    ) -> None:
        """Update the zone's SOA fields.

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
            primary_ns (str | UNSET): New primary nameserver. Omit to leave unchanged.
            email (str | UNSET): New zone admin email. Omit to leave unchanged.
            serialno (int | UNSET): New serial number. Omit to leave unchanged.
            refresh (int | UNSET): New refresh interval. Omit to leave unchanged.
            retry (int | UNSET): New retry interval. Omit to leave unchanged.
            expire (int | UNSET): New expire interval. Omit to leave unchanged.
            soa_ttl (int | UNSET): New SOA TTL. Omit to leave unchanged.
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
        # NOTE: no verification here...?
        if isinstance(z, ReverseZone):
            self._reverse.update_soa(z, **kwargs)
        else:
            self._forward.update_soa(z, **kwargs)

    def set_default_ttl(self, zone: str | ForwardZone | ReverseZone, ttl: int) -> None:
        """Set the zone's default TTL.

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
            ttl (int): The new default TTL value (300–68400).
        """
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            self._reverse.set_default_ttl(z, ttl)
        else:
            self._forward.set_default_ttl(z, ttl)

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

        verified_ns = _verify_nameservers(self._client, nameservers, force=force)

        if isinstance(z, ReverseZone):
            self._reverse.set_nameservers(z, verified_ns)
        else:
            self._forward.set_nameservers(z, verified_ns)

    def list_subzones(self, zone: str | ForwardZone | ReverseZone) -> list[ForwardZone] | list[ReverseZone]:
        """List subzones of the zone (excluding the zone itself).

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
        """
        z = self._resolve_zone(zone)
        if isinstance(z, ReverseZone):
            return self._reverse.list_subzones(z)
        else:
            return self._forward.list_subzones(z)

    def delete(self, zone: str | ForwardZone | ReverseZone, *, force: bool = False) -> None:
        """Delete the zone, guarding against non-empty zones unless `force`.

        Args:
            zone (str | ForwardZone | ReverseZone): Zone reference (name string or instance).
            force (bool): When True, skip safety checks and delete even non-empty zones.
        """
        # NOTE: cannot delete by ID with the current architecture, since we delegate
        # the deletion to a reverse or forward sub-manager based on the identified
        # zone type.
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
        """Manager for zone delegations."""
        return DelegationManager(self._client)


class DelegationManager:
    """Operations on zone delegations.

    Delegations have no standalone endpoint; their type (forward/reverse) is derived
    from the parent zone, so every method takes the parent zone as its first argument.
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
        self, zone: Zone, name: str, *, required: Literal[False]
    ) -> ForwardZoneDelegation | ReverseZoneDelegation | None: ...
    @overload
    def get(
        self, zone: Zone, name: str, *, required: Literal[True] = ...
    ) -> ForwardZoneDelegation | ReverseZoneDelegation: ...
    def get(
        self, zone: Zone, name: str, *, required: bool = True
    ) -> ForwardZoneDelegation | ReverseZoneDelegation | None:
        """Get a delegation in `zone` by name.

        Args:
            zone (Zone): The parent zone to search in.
            name (str): The delegation name to look up.
            required (bool): When True (default), raise EntityNotFound if not found.

        Raises:
            EntityNotFound: If `required` is True and the delegation is not found.
        """
        self._ensure_in_zone(zone, name)
        cls = self._model_for(zone)
        try:
            return self._get(cls.endpoint_with_name(zone, name), cls)
        except Exception as e:
            if required:
                raise EntityNotFound(f"Could not find delegation {name!r} in zone {zone.name!r}") from e
        return None

    def _get(
        self, endpoint: str, model: type[ForwardZoneDelegation | ReverseZoneDelegation]
    ) -> ForwardZoneDelegation | ReverseZoneDelegation:
        return self._client.get_typed(endpoint, model)

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
    ) -> ForwardZoneDelegation | ReverseZoneDelegation:
        """Create a delegation in `zone`.

        Verifies the delegation name is within the zone and the nameservers exist.
        Unless `force`, also checks the delegated zone exists and matches the parent
        zone type, and that the delegation does not already exist.

        Args:
            zone (Zone): The parent zone to create the delegation in.
            name (str): The delegation name (must be within the parent zone).
            nameservers (list[str]): List of nameserver names for the delegation.
            comment (str): Optional comment for the delegation. Defaults to "".
            force (bool): When True, skip safety checks.

        Returns:
            ForwardZoneDelegation | ReverseZoneDelegation: The created zone delegation.
        """
        self._ensure_in_zone(zone, name)
        verified_ns = _verify_nameservers(self._client, nameservers, force=force)

        if not force:
            delegated = self._client.zone.get_by_name(name, required=False)
            if not delegated:
                raise InputFailure(f"Zone {name!r} does not exist. Must force.")
            if delegated.is_reverse() != zone.is_reverse():
                # TODO: fix formatting here
                raise InputFailure(f"Delegation {name!r} is not a {model_name(zone)} zone")

        if self.get(zone, name, required=False) is not None:
            raise EntityAlreadyExists(f"Zone {zone.name!r} already has a delegation named {name!r}")

        return self._create(
            zone,
            name=name,
            nameservers=verified_ns,
            comment=comment,
        )

    # TODO: Inherit from WriteResourceManager and override _patch, _create, etc.
    # to optionally override with endpoint and model, so we can call them in DelegationManager
    # using _model_for and _endpoint_for
    def _create(
        self,
        zone: Zone,
        *,
        name: str,
        nameservers: list[VerifiedNS],  # ensure we have list of verified nameservers
        comment: str = "",
    ) -> ForwardZoneDelegation | ReverseZoneDelegation:
        """Create a zone delegation (with FQDN nameservers)."""
        response = self._client.post(
            self._endpoint_for(zone).with_params(zone.name),
            json={"name": name, "nameservers": nameservers, "comment": comment},
        )
        if response.content:
            try:
                return validate_response(self._model_for(zone), response)
            except Exception:
                pass
        if loc := response.headers.get("Location"):
            return self._get(loc, self._model_for(zone))
        raise PostError(f"Failed to retrieve zone {zone.name!r} after creation")

    def delete(self, zone: Zone, name: str) -> None:
        """Delete a delegation from `zone`.

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
        """Set (or clear, with `""`) the comment for a delegation.

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
    """Read-only manager for DHCP host records.

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
    """Read-only manager for IPv4 DHCP host records."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv4]:
        return DhcpHostIPv4

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv4


class DhcpHostIPv6Manager(DhcpHostManager[DhcpHostIPv6]):
    """Read-only manager for IPv6 DHCP host records."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv6]:
        return DhcpHostIPv6

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv6


class DhcpHostIPv6ByIPv4Manager(DhcpHostManager[DhcpHostIPv6ByIPv4]):
    """Read-only manager for IPv6-via-IPv4 DHCP host records."""

    @property
    @override
    def model(self) -> type[DhcpHostIPv6ByIPv4]:
        return DhcpHostIPv6ByIPv4

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.DhcpHostsIpv6ByIpv4


class NameServerManager(NamedResourceManager[NameServer]):
    """Access to :class:`~mreg_api.models.NameServer` resources."""

    _path_param_field: ClassVar[str] = "name"

    @property
    @override
    def model(self) -> type[NameServer]:
        return NameServer

    @property
    @override
    def endpoint(self) -> Endpoint:
        return Endpoint.Nameservers

    def create(
        self,
        *,
        name: str,
        ttl: int | None | UNSET = UNSET,
    ) -> NameServer:
        """Create a nameserver.

        NOTE: this endpoint does _not_ return the resource after creation.

        Args:
            name (str): The nameserver name to create.
            ttl (int | None | UNSET): Optional TTL for the nameserver. If None, uses default TTL.

        Returns:
            NameServer: The created nameserver.
        """
        data: dict[str, Any] = {"name": self._client.fqdn(name)}
        if ttl is not UNSET:
            data["ttl"] = ttl
        return self._create(data)


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


# TODO: make GetManager a type that ResourceManager can inherit, so that we
# have a common interface for these `get`-only managers and the rest of
# the managers, along with shared convenience methods for validation and formatting.
#
# Might need a rename, since I'm not sure if you can override like this:
# Parent: `get(self, *, required: bool = False) -> T`
# Child: `get(self, ident: str | int | T, *, required: bool = False) -> T | None`
#
# So, instead, maybe something along the lines of `ManagerBase`, which we can then
# define `MetaManager` from.
#
# This would give us the following hierarchies:
# - BaseManager[T] -> ResourceManager -> WriteResourceManager -> NamedResourceManager -> HostManager, ...
# - BaseManager[T] -> MetaManager -> ServerVersionManager, ...
#
# Is this complexity worth it just to share a few convencience methods and
# avoid writing interfaces? Who knows.
#
# NOTE: The `get` methods of these meta endpoints never returning None
# is a CLI-specific thing where we want to show some data even when
# an endpoint doesn't return anything useful. We should consider if these CLI
# commands should just propagate errors instead of showing dummy data...


class GetManager(Generic[T], ABC):
    """A manager implementing only a `get()` method that fetches a meta-endpoint resource.

    Each of these manager's get method contains some idiosyncratic logic
    for fetching and validating the resource, and has some built-in
    error handling for when the resource is not available.
    """

    def __init__(self, client: MregClient) -> None:
        """Initialize the manager with a client instance."""
        self._client = client

    @property
    @abstractmethod
    def model(self) -> type[T]:
        """The model type this manager operates on."""
        raise NotImplementedError

    @abstractmethod
    def get(self, *, required: bool = False) -> T:
        """Get the resource from the endpoint."""
        raise NotImplementedError

    def validate_response(self, response: Response) -> T:
        """Validate a JSON response and construct the manager's type."""
        return self.model.model_validate_json(response.text)


class ServerVersionManager(GetManager[ServerVersion]):
    """Access to server version metadata."""

    @property
    @override
    def model(self) -> type[ServerVersion]:
        return ServerVersion

    def get(self, *, required: bool = False) -> ServerVersion:
        """Fetch the server version from the meta endpoint.

        Args:
            required: When `True`, raise on error. When `False`, return
                `ServerVersion(version="Unknown")` on failure.
        """
        try:
            response = self._client.get(Endpoint.MetaVersion)
            return ServerVersion.model_validate(response.json())
        except Exception:
            if required:
                raise
            return ServerVersion(version="Unknown")


class ServerLibrariesManager:
    """Access to server library metadata."""

    def __init__(self, client: MregClient) -> None:
        """Initialize the manager with a client instance."""
        self._client = client

    def get(self, *, required: bool = False) -> ServerLibraries:
        """Fetch the server library list from the meta endpoint.

        Args:
            required: When `True`, raise on error. When `False`, return
                `ServerLibraries(libraries=[])` on failure.
        """
        try:
            response = self._client.get_typed(Endpoint.MetaLibraries, dict[str, str])
            libraries = [Library(name=name, version=version) for name, version in response.items()]
            return ServerLibraries(libraries=libraries)
        except Exception:
            if required:
                raise
            return ServerLibraries(libraries=[])


class UserInfoManager(GetManager[UserInfo]):
    """Access to user information."""

    @property
    @override
    def model(self) -> type[UserInfo]:
        return UserInfo

    @override
    def get(self, *, required: bool = False, user: str | None = None) -> UserInfo:
        """Fetch user information from the meta endpoint.

        Args:
            required: When `True`, raise on error. When `False`, return a
                zeroed-out `UserInfo` on failure.
            user: The username to fetch. If `None`, fetches the current user.
        """
        try:
            endpoint: str = Endpoint.MetaUser
            if user:
                endpoint = f"{endpoint}?username={user}"
            response = self._client.get(endpoint)
            return self.model.model_validate(response.json())
        except Exception:
            if required:
                raise
            return self.model(username="Unknown")


class LDAPHealthManager(GetManager[LDAPHealth]):
    """Access to LDAP health status."""

    @property
    @override
    def model(self) -> type[LDAPHealth]:
        return LDAPHealth

    def get(self, *, required: bool = False) -> LDAPHealth:
        """Fetch LDAP health from the health endpoint.

        A 503 response means LDAP is down and is not treated as a hard error.

        Args:
            required: When `True`, raise on non-503 errors.
        """
        # NOTE: this manager just interprets HTTP status codes to build its model
        try:
            self._client.get(Endpoint.HealthLDAP)
            return LDAPHealth(status="OK")
        except GetError as e:
            # LDAP being down causes a 503 error, which we should
            # interpret as a valid response, not an error.
            if e.response and e.response.status_code == 503:
                return LDAPHealth(status="Down")
            if required:
                raise
            logger.error("Failed to fetch LDAP health: %s", e)
            return LDAPHealth(status="Unknown")


class HeartbeatHealthManager(GetManager[HeartbeatHealth]):
    """Access to heartbeat health status."""

    @property
    @override
    def model(self) -> type[HeartbeatHealth]:
        return HeartbeatHealth

    def get(self, *, required: bool = False) -> HeartbeatHealth:
        """Fetch heartbeat health from the health endpoint.

        Args:
            required: When `True`, raise on error. When `False`, return
                `HeartbeatHealth(uptime=-1, start_time=0)` on failure.
        """
        try:
            resp = self._client.get(Endpoint.HealthHeartbeat)
            return self.validate_response(resp)
        except Exception as e:
            if required:
                raise
            logger.error("Failed to fetch heartbeat: %s", e)
            return HeartbeatHealth(uptime=-1, start_time=0)


class HealthManager(GetManager[HealthInfo]):
    """Access to combined health information."""

    @property
    @override
    def model(self) -> type[HealthInfo]:
        return HealthInfo

    def __init__(
        self,
        client: MregClient,
        heartbeat_manager: HeartbeatHealthManager | None = None,
        ldap_manager: LDAPHealthManager | None = None,
    ) -> None:
        """Initialize the manager with a client instance.

        Optionally takes in existing heartbeat and LDAP managers to avoid creating new ones.
        """
        super().__init__(client)
        if heartbeat_manager is None:
            heartbeat_manager = HeartbeatHealthManager(client)
        self._heartbeat_manager = heartbeat_manager

        if ldap_manager is None:
            ldap_manager = LDAPHealthManager(client)
        self._ldap_manager = ldap_manager

    def get(self, *, required: bool = False) -> HealthInfo:
        """Fetch combined health from all health endpoints.

        Args:
            required: Forwarded to both sub-managers. When `True`, raises
                on any failure instead of returning a default object.
        """
        heartbeat = self._heartbeat_manager.get(required=required)
        ldap = self._ldap_manager.get(required=required)
        return HealthInfo(heartbeat=heartbeat, ldap=ldap)
