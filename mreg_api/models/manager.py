"""Client-bound model managers for MREG API resources."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType
from typing import Any
from typing import Generic
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.types import ClientProtocol
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams
from typing_extensions import Unpack

if TYPE_CHECKING:
    from mreg_api.models.models import Atom
    from mreg_api.models.models import Community
    from mreg_api.models.models import ForwardZone
    from mreg_api.models.models import Host
    from mreg_api.models.models import HostGroup
    from mreg_api.models.models import HostPolicy
    from mreg_api.models.models import IPAddress
    from mreg_api.models.models import Label
    from mreg_api.models.models import Network
    from mreg_api.models.models import NetworkPolicy
    from mreg_api.models.models import Permission
    from mreg_api.models.models import ReverseZone
    from mreg_api.models.models import Role
    from mreg_api.models.models import Zone
    from mreg_api.models.patch_types import CommunityPatch
    from mreg_api.models.patch_types import HostGroupPatch
    from mreg_api.models.patch_types import HostPatch
    from mreg_api.models.patch_types import HostPolicyPatch
    from mreg_api.models.patch_types import IPAddressPatch
    from mreg_api.models.patch_types import LabelPatch
    from mreg_api.models.patch_types import NetworkPatch
    from mreg_api.models.patch_types import NetworkPolicyPatch
    from mreg_api.models.patch_types import PermissionPatch
    from mreg_api.models.patch_types import ZonePatch

T = TypeVar("T")
LT = TypeVar("LT", bound="ModelList[Any]")

_SPECIAL_SNAKE_CASE: dict[str, str] = {
    "BacnetID": "bacnet_id",
    "CNAME": "cname",
    "HInfo": "hinfo",
    "IPAddress": "ip_address",
    "LDAPHealth": "ldap_health",
    "MX": "mx",
    "NAPTR": "naptr",
    "PTR_override": "ptr_override",
    "SSHFP": "sshfp",
    "TXT": "txt",
}


class ModelList(list[T], Generic[T]):
    """A list of models with convenience bulk operations."""

    def _bulk_call(self, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call the same method on each item and fail fast on unsupported/failed operations."""
        results: list[Any] = []
        for item in self:
            method = getattr(item, method_name, None)
            if not callable(method):
                raise TypeError(f"Cannot bulk-{method_name} item of type {type(item).__name__}.")
            result = method(*args, **kwargs)
            if result is False or result is None:
                raise RuntimeError(f"Failed to {method_name} item of type {type(item).__name__}.")
            results.append(result)
        return results

    def delete(self) -> None:
        """Delete all objects in this list.

        :raises TypeError: If an item does not define a callable ``delete`` method.
        :raises RuntimeError: If any ``delete()`` call reports failure.
        """
        _ = self._bulk_call("delete")

    def patch(
        self,
        fields: dict[str, Any] | None = None,
        validate: bool = True,
        **field_kwargs: Any,
    ) -> "ModelList[T]":
        """Patch all objects in this list and return updated objects.

        :raises TypeError: If an item does not define a callable ``patch`` method.
        :raises RuntimeError: If any ``patch()`` call reports failure.
        :returns: A ``ModelList`` containing the patched objects.
        """
        return ModelList(
            cast(list[T], self._bulk_call("patch", fields, validate=validate, **field_kwargs))
        )

    def patch_raw(self, fields: dict[str, Any], validate: bool = True) -> "ModelList[T]":
        """Patch all objects in this list using an explicit dictionary payload."""
        return ModelList(cast(list[T], self._bulk_call("patch_raw", fields, validate=validate)))

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Any) -> "ModelList[T]":
        """Patch all objects in this list using each item's typed patch helper."""
        return ModelList(
            cast(list[T], self._bulk_call("patch_typed", validate=validate, **field_kwargs))
        )


class HostModelList(ModelList["Host"]):
    """Host-specialized model list with typed bulk patch support."""

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack["HostPatch"]) -> "HostModelList":
        """Patch all host objects in this list with typed keyword arguments."""
        return cast(HostModelList, super().patch_typed(validate=validate, **field_kwargs))


class PermissionModelList(ModelList["Permission"]):
    """Permission-specialized model list with typed bulk patch support."""

    def patch_typed(
        self, *, validate: bool = True, **field_kwargs: Unpack["PermissionPatch"]
    ) -> "PermissionModelList":
        return cast(PermissionModelList, super().patch_typed(validate=validate, **field_kwargs))


class ZoneModelList(ModelList["Zone"]):
    """Zone-specialized model list with typed bulk patch support."""

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack["ZonePatch"]) -> "ZoneModelList":
        return cast(ZoneModelList, super().patch_typed(validate=validate, **field_kwargs))


class ForwardZoneModelList(ModelList["ForwardZone"]):
    """ForwardZone-specialized model list with typed bulk patch support."""

    def patch_typed(
        self, *, validate: bool = True, **field_kwargs: Unpack["ZonePatch"]
    ) -> "ForwardZoneModelList":
        return cast(ForwardZoneModelList, super().patch_typed(validate=validate, **field_kwargs))


class ReverseZoneModelList(ModelList["ReverseZone"]):
    """ReverseZone-specialized model list with typed bulk patch support."""

    def patch_typed(
        self, *, validate: bool = True, **field_kwargs: Unpack["ZonePatch"]
    ) -> "ReverseZoneModelList":
        return cast(ReverseZoneModelList, super().patch_typed(validate=validate, **field_kwargs))


class HostPolicyModelList(ModelList["HostPolicy"]):
    """HostPolicy-specialized model list with typed bulk patch support."""

    def patch_typed(
        self, *, validate: bool = True, **field_kwargs: Unpack["HostPolicyPatch"]
    ) -> "HostPolicyModelList":
        return cast(HostPolicyModelList, super().patch_typed(validate=validate, **field_kwargs))


class RoleModelList(ModelList["Role"]):
    """Role-specialized model list with typed bulk patch support."""

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack["HostPolicyPatch"]) -> "RoleModelList":
        return cast(RoleModelList, super().patch_typed(validate=validate, **field_kwargs))


class AtomModelList(ModelList["Atom"]):
    """Atom-specialized model list with typed bulk patch support."""

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack["HostPolicyPatch"]) -> "AtomModelList":
        return cast(AtomModelList, super().patch_typed(validate=validate, **field_kwargs))


class LabelModelList(ModelList["Label"]):
    """Label-specialized model list with typed bulk patch support."""

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack["LabelPatch"]) -> "LabelModelList":
        return cast(LabelModelList, super().patch_typed(validate=validate, **field_kwargs))


class NetworkModelList(ModelList["Network"]):
    """Network-specialized model list with typed bulk patch support."""

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack["NetworkPatch"]) -> "NetworkModelList":
        return cast(NetworkModelList, super().patch_typed(validate=validate, **field_kwargs))


class CommunityModelList(ModelList["Community"]):
    """Community-specialized model list with typed bulk patch support."""

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack["CommunityPatch"]) -> "CommunityModelList":
        return cast(CommunityModelList, super().patch_typed(validate=validate, **field_kwargs))


class NetworkPolicyModelList(ModelList["NetworkPolicy"]):
    """NetworkPolicy-specialized model list with typed bulk patch support."""

    def patch_typed(
        self, *, validate: bool = True, **field_kwargs: Unpack["NetworkPolicyPatch"]
    ) -> "NetworkPolicyModelList":
        return cast(NetworkPolicyModelList, super().patch_typed(validate=validate, **field_kwargs))


class IPAddressModelList(ModelList["IPAddress"]):
    """IPAddress-specialized model list with typed bulk patch support."""

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack["IPAddressPatch"]) -> "IPAddressModelList":
        return cast(IPAddressModelList, super().patch_typed(validate=validate, **field_kwargs))


class HostGroupModelList(ModelList["HostGroup"]):
    """HostGroup-specialized model list with typed bulk patch support."""

    def patch_typed(
        self, *, validate: bool = True, **field_kwargs: Unpack["HostGroupPatch"]
    ) -> "HostGroupModelList":
        return cast(HostGroupModelList, super().patch_typed(validate=validate, **field_kwargs))


def to_snake_case(name: str) -> str:
    """Convert a model class name to a client attribute name."""
    if name in _SPECIAL_SNAKE_CASE:
        return _SPECIAL_SNAKE_CASE[name]
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


def client_model_map(models_module: ModuleType) -> dict[str, type]:
    """Build a mapping of client attribute name -> model class."""
    mapping: dict[str, type] = {}
    for model_name in getattr(models_module, "__all__", []):
        model_obj = getattr(models_module, model_name)
        if any(getattr(model_obj, attr, None) is not None for attr in ["get", "fetch"]):
            mapping[to_snake_case(model_obj.__name__)] = model_obj
    return mapping


class ModelManager(Generic[T]):
    """Bind a model's class API to a specific client instance."""

    def __init__(self, client: ClientProtocol, model: type[T]) -> None:
        """Create a manager bound to a client and model."""
        self._client = client
        self._model: Any = model

    def __call__(self) -> "ModelManager[T]":
        """Return self to support fluent call chains."""
        return self

    @property
    def model(self) -> type[T]:
        """Return the model class this manager wraps."""
        return self._model

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access, binding client for compatible callables."""
        attr = getattr(self._model, name)
        if not callable(attr):
            return attr
        if inspect.ismethod(attr):
            try:
                sig = inspect.signature(attr)
            except (TypeError, ValueError):
                return attr
            params = list(sig.parameters.values())
            if params and params[0].name == "client":
                return functools.partial(attr, self._client, _manager=True)
        return attr

    def get(self, _id: int) -> T | None:
        """Get an object."""
        return cast(T | None, self._model.get(self._client, _id, _manager=True))

    def _wrap_model_list(self, values: list[T]) -> ModelList[T]:
        """Wrap a plain list in a model-aware list container."""
        return ModelList(values)

    def get_list_by_id(self, _id: int) -> ModelList[T]:
        """Get a list of objects by their ID."""
        return self._wrap_model_list(
            cast(list[T], self._model.get_list_by_id(self._client, _id, _manager=True))
        )

    def get_by_id(self, _id: int) -> T | None:
        """Get an object by its ID."""
        return cast(T | None, self._model.get_by_id(self._client, _id, _manager=True))

    def get_by_id_or_raise(
        self,
        _id: int,
        exc_type: type[Exception] = EntityNotFound,
        exc_message: str | None = None,
    ) -> T:
        """Get an object by ID and raise if not found."""
        obj = self.get_by_id(_id)
        if obj is not None:
            return obj
        model_name = getattr(self._model, "__name__", "Object")
        raise exc_type(exc_message or f"{model_name} with ID {_id} not found.")

    def get_by_field(self, field: str, value: str | int) -> T | None:
        """Get an object by a field."""
        return cast(
            T | None, self._model.get_by_field(self._client, field, value, _manager=True)
        )

    def get_by_field_or_raise(
        self,
        field: str,
        value: str,
        exc_type: type[Exception] = EntityNotFound,
        exc_message: str | None = None,
    ) -> T:
        """Get an object by a field and raise if not found."""
        return cast(
            T,
            self._model.get_by_field_or_raise(
                self._client,
                field,
                value,
                exc_type=exc_type,
                exc_message=exc_message,
                _manager=True,
            ),
        )

    def get_by_field_and_raise(
        self,
        field: str,
        value: str,
        exc_type: type[Exception] = EntityAlreadyExists,
        exc_message: str | None = None,
    ) -> None:
        """Get an object by a field and raise if found."""
        _ = self._model.get_by_field_and_raise(
            self._client,
            field,
            value,
            exc_type=exc_type,
            exc_message=exc_message,
            _manager=True,
        )

    def get_list(self, params: QueryParams | None = None, limit: int | None = None) -> ModelList[T]:
        """Get a list of all objects."""
        return self._wrap_model_list(
            cast(list[T], self._model.get_list(self._client, params=params, limit=limit, _manager=True))
        )

    def get_by_query(
        self,
        query: QueryParams,
        ordering: str | None = None,
        limit: int | None = 500,
    ) -> ModelList[T]:
        """Get a list of objects by a query."""
        return self._wrap_model_list(
            cast(
                list[T],
                self._model.get_by_query(
                    self._client, query=query, ordering=ordering, limit=limit, _manager=True
                ),
            )
        )

    def get_list_by_field(
        self,
        field: str,
        value: str | int,
        ordering: str | None = None,
        limit: int = 500,
    ) -> ModelList[T]:
        """Get a list of objects by a field."""
        return self._wrap_model_list(
            cast(
                list[T],
                self._model.get_list_by_field(
                    self._client,
                    field,
                    value,
                    ordering=ordering,
                    limit=limit,
                    _manager=True,
                ),
            )
        )

    def get_by_query_unique_or_raise(
        self,
        query: QueryParams,
        exc_type: type[Exception] = EntityNotFound,
        exc_message: str | None = None,
    ) -> T:
        """Get an object by a query and raise if not found."""
        return cast(
            T,
            self._model.get_by_query_unique_or_raise(
                self._client,
                query=query,
                exc_type=exc_type,
                exc_message=exc_message,
                _manager=True,
            ),
        )

    def get_by_query_unique_and_raise(
        self,
        query: QueryParams,
        exc_type: type[Exception] = EntityAlreadyExists,
        exc_message: str | None = None,
    ) -> None:
        """Get an object by a query and raise if found."""
        _ = self._model.get_by_query_unique_and_raise(
            self._client,
            query=query,
            exc_type=exc_type,
            exc_message=exc_message,
            _manager=True,
        )

    def get_by_query_unique(self, data: QueryParams) -> T | None:
        """Get an object with the given data."""
        return cast(
            T | None, self._model.get_by_query_unique(self._client, data=data, _manager=True)
        )

    def get_first(self) -> T | None:
        """Get the first object from the list."""
        return cast(T | None, self._model.get_first(self._client, _manager=True))

    def get_first_or_raise(self) -> T:
        """Get the first object from the list, raising if empty."""
        return cast(T, self._model.get_first_or_raise(self._client, _manager=True))

    def get_count(self) -> int:
        """Get the count of items from the list."""
        return self._model.get_count(self._client, _manager=True)

    def create(self, params: JsonMapping, fetch_after_create: bool = True) -> T | None:
        """Create the object."""
        return cast(
            T | None,
            self._model.create(
                self._client, params=params, fetch_after_create=fetch_after_create, _manager=True
            ),
        )


class TypedListManager(ModelManager[T], Generic[T, LT]):
    """Manager that returns a specialized model list type."""

    _list_type: type[LT]

    def _wrap_model_list(self, values: list[T]) -> ModelList[T]:
        return cast(ModelList[T], self._list_type(values))

    def get_list_by_id(self, _id: int) -> LT:
        return cast(LT, super().get_list_by_id(_id))

    def get_list(self, params: QueryParams | None = None, limit: int | None = None) -> LT:
        return cast(LT, super().get_list(params=params, limit=limit))

    def get_by_query(
        self,
        query: QueryParams,
        ordering: str | None = None,
        limit: int | None = 500,
    ) -> LT:
        return cast(LT, super().get_by_query(query=query, ordering=ordering, limit=limit))

    def get_list_by_field(
        self,
        field: str,
        value: str | int,
        ordering: str | None = None,
        limit: int = 500,
    ) -> LT:
        return cast(LT, super().get_list_by_field(field=field, value=value, ordering=ordering, limit=limit))


class HostManager(TypedListManager["Host", HostModelList]):
    """Typed manager specialization for Host."""

    _list_type = HostModelList


class PermissionManager(TypedListManager["Permission", PermissionModelList]):
    """Typed manager specialization for Permission."""

    _list_type = PermissionModelList


class ZoneManager(TypedListManager["Zone", ZoneModelList]):
    """Typed manager specialization for Zone."""

    _list_type = ZoneModelList


class ForwardZoneManager(TypedListManager["ForwardZone", ForwardZoneModelList]):
    """Typed manager specialization for ForwardZone."""

    _list_type = ForwardZoneModelList


class ReverseZoneManager(TypedListManager["ReverseZone", ReverseZoneModelList]):
    """Typed manager specialization for ReverseZone."""

    _list_type = ReverseZoneModelList


class HostPolicyManager(TypedListManager["HostPolicy", HostPolicyModelList]):
    """Typed manager specialization for HostPolicy."""

    _list_type = HostPolicyModelList


class RoleManager(TypedListManager["Role", RoleModelList]):
    """Typed manager specialization for Role."""

    _list_type = RoleModelList


class AtomManager(TypedListManager["Atom", AtomModelList]):
    """Typed manager specialization for Atom."""

    _list_type = AtomModelList


class LabelManager(TypedListManager["Label", LabelModelList]):
    """Typed manager specialization for Label."""

    _list_type = LabelModelList


class NetworkManager(TypedListManager["Network", NetworkModelList]):
    """Typed manager specialization for Network."""

    _list_type = NetworkModelList


class CommunityManager(TypedListManager["Community", CommunityModelList]):
    """Typed manager specialization for Community."""

    _list_type = CommunityModelList


class NetworkPolicyManager(TypedListManager["NetworkPolicy", NetworkPolicyModelList]):
    """Typed manager specialization for NetworkPolicy."""

    _list_type = NetworkPolicyModelList


class IPAddressManager(TypedListManager["IPAddress", IPAddressModelList]):
    """Typed manager specialization for IPAddress."""

    _list_type = IPAddressModelList


class HostGroupManager(TypedListManager["HostGroup", HostGroupModelList]):
    """Typed manager specialization for HostGroup."""

    _list_type = HostGroupModelList
