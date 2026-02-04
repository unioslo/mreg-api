"""Typing definitions for mreg_cli."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeAlias
from typing import TypeVar

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
    from typing import Union

    # NOTE: Json ported from Pydantic here, but with Sequence and Mapping
    #       instead of list and dict
    Json: TypeAlias = Union[
        Sequence["Json"],
        Mapping[str, "Json"],
        str,
        bool,
        int,
        float,
        None,
    ]
else:
    from pydantic import JsonValue

    Json = JsonValue

JsonMapping = Mapping[str, Json]
QueryParams = MutableMapping[str, str | int | float | bool | None]


T = TypeVar("T")


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
