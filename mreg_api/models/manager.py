"""Client-bound model managers for MREG API resources."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import cast

from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.types import ClientProtocol
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams

T = TypeVar("T")

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

    def patch(self, fields: dict[str, Any], validate: bool = True) -> "ModelList[T]":
        """Patch all objects in this list and return updated objects.

        :raises TypeError: If an item does not define a callable ``patch`` method.
        :raises RuntimeError: If any ``patch()`` call reports failure.
        :returns: A ``ModelList`` containing the patched objects.
        """
        return ModelList(cast(list[T], self._bulk_call("patch", fields, validate=validate)))


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

    def get_list_by_id(self, _id: int) -> ModelList[T]:
        """Get a list of objects by their ID."""
        return ModelList(cast(list[T], self._model.get_list_by_id(self._client, _id, _manager=True)))

    def get_by_id(self, _id: int) -> T | None:
        """Get an object by its ID."""
        return cast(T | None, self._model.get_by_id(self._client, _id, _manager=True))

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
        return ModelList(
            cast(list[T], self._model.get_list(self._client, params=params, limit=limit, _manager=True))
        )

    def get_by_query(
        self,
        query: QueryParams,
        ordering: str | None = None,
        limit: int | None = 500,
    ) -> ModelList[T]:
        """Get a list of objects by a query."""
        return ModelList(
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
        return ModelList(
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
