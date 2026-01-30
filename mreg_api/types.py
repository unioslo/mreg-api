"""Typing definitions for mreg_cli."""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import MutableMapping
from functools import lru_cache
from typing import Any
from typing import Literal
from typing import TypeAlias
from typing import TypedDict
from typing import TypeVar

from pydantic import TypeAdapter
from pydantic.types import JsonValue

CommandFunc = Callable[[argparse.Namespace], None]


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


class TimeInfo(TypedDict):
    """Type definition for time-related information in the recording entry."""

    timestamp: str
    timestamp_as_epoch: int
    runtime_in_ms: int


# TODO: remove if we don't need it in this package
class RecordingEntry(TypedDict):
    """Type definition for a recording entry."""

    command: str
    command_filter: str | None
    command_filter_negate: bool
    command_issued: str
    ok: list[str]
    warning: list[str]
    error: list[str]
    output: list[str]
    api_requests: list[dict[str, Any]]
    time: TimeInfo | None


IP_Version: TypeAlias = Literal[4, 6]
IP_AddressT = ipaddress.IPv4Address | ipaddress.IPv6Address
IP_NetworkT = ipaddress.IPv4Network | ipaddress.IPv6Network

IP_networkTV = TypeVar("IP_networkTV", ipaddress.IPv4Network, ipaddress.IPv6Network)

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
