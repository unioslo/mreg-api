"""Typing definitions for mreg_cli."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Literal
from typing import Protocol
from typing import TypeAlias
from typing import TypeVar
from typing import overload

from httpx import Response
from pydantic import TypeAdapter

HTTPMethod: TypeAlias = Literal[
    "GET",
    "POST",
    "PATCH",
    "DELETE",
    # NYI: not currently used methods
    # "PUT",
    # "HEAD",
    # "OPTIONS",
]


IP_AddressT = ipaddress.IPv4Address | ipaddress.IPv6Address
IP_NetworkT = ipaddress.IPv4Network | ipaddress.IPv6Network


if TYPE_CHECKING:
    # NOTE: Json ported from Pydantic here, but with Sequence and Mapping
    #       instead of list and dict
    Json: TypeAlias = Sequence["Json"] | Mapping[str, "Json"] | str | bool | int | float | None
else:
    from pydantic import JsonValue

    Json = JsonValue

JsonMapping = Mapping[str, Json]
QueryParams = MutableMapping[str, str | int | float | bool | None]


T = TypeVar("T")


class ClientProtocol(Protocol):
    """Structural protocol for client methods used by models."""

    _manager_call_depth: int

    @overload
    def get(
        self, path: str, params: QueryParams | None, ok404: Literal[True]
    ) -> Response | None: ...

    @overload
    def get(self, path: str, params: QueryParams | None, ok404: Literal[False]) -> Response: ...

    @overload
    def get(
        self, path: str, params: QueryParams | None = ..., *, ok404: bool
    ) -> Response | None: ...

    @overload
    def get(self, path: str, params: QueryParams | None = ...) -> Response: ...

    @overload
    def post(
        self, path: str, params: QueryParams | None, ok404: Literal[True], **data: Json
    ) -> Response | None: ...

    @overload
    def post(
        self, path: str, params: QueryParams | None, ok404: Literal[False], **data: Json
    ) -> Response: ...

    @overload
    def post(
        self, path: str, params: QueryParams | None = ..., *, ok404: bool, **data: Json
    ) -> Response: ...

    @overload
    def post(self, path: str, params: QueryParams | None = ..., **data: Json) -> Response: ...

    @overload
    def patch(
        self, path: str, params: QueryParams | None, ok404: Literal[True], **data: Json
    ) -> Response | None: ...

    @overload
    def patch(
        self, path: str, params: QueryParams | None, ok404: Literal[False], **data: Json
    ) -> Response: ...

    @overload
    def patch(
        self, path: str, params: QueryParams | None = ..., *, ok404: bool, **data: Json
    ) -> Response: ...

    @overload
    def patch(self, path: str, params: QueryParams | None = ..., **data: Json) -> Response: ...

    @overload
    def delete(
        self, path: str, params: QueryParams | None, ok404: Literal[True], **data: Json
    ) -> Response | None: ...

    @overload
    def delete(
        self, path: str, params: QueryParams | None, ok404: Literal[False], **data: Json
    ) -> Response: ...

    @overload
    def delete(
        self, path: str, params: QueryParams | None = ..., *, ok404: bool, **data: Json
    ) -> Response: ...

    @overload
    def delete(self, path: str, params: QueryParams | None = ..., **data: Json) -> Response: ...

    def get_item_by_key_value(
        self, path: str, search_field: str, search_value: str | int, ok404: bool = False
    ) -> JsonMapping | None:
        """Fetch a single item by field value."""
        ...

    def get_list_unique(
        self, path: str, params: QueryParams | None = None, ok404: bool = False
    ) -> JsonMapping | None:
        """Fetch a single unique result from a list endpoint."""
        ...

    def get_typed(
        self, path: str, type_: type[T], params: QueryParams | None = None, limit: int | None = None
    ) -> T:
        """Fetch and deserialize JSON into the requested type."""
        ...

    def get_first(self, path: str) -> JsonMapping | None:
        """Fetch the first item from a list endpoint."""
        ...

    def get_count(self, path: str) -> int:
        """Fetch a count from a list endpoint."""
        ...

    def _bind_client(self, value: T) -> T:
        """Bind a client to any model instances in the value."""
        ...


@lru_cache(maxsize=100)
def get_type_adapter(t: type[T]) -> TypeAdapter[T]:
    """Get the type adapter for a given type.

    :param t: The type to get the adapter for.
    :returns: The type adapter.

    """
    return TypeAdapter(t)


def parse_json_string(json_string: str) -> Json:
    """Parse a JSON string into a Python object.

    :param json_string: The JSON string to parse.
    :returns: The parsed Python object.

    """
    adapter = get_type_adapter(Json)
    return adapter.validate_json(json_string)


def parse_json_mapping_string(json_string: str) -> JsonMapping:
    """Parse a JSON string into a Python mapping.

    :param json_string: The JSON string to parse.
    :returns: The parsed Python mapping.

    """
    adapter = get_type_adapter(JsonMapping)
    return adapter.validate_json(json_string)
