"""Typing definitions for mreg_cli."""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Callable
from enum import Enum
from functools import lru_cache
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Mapping
from typing import MutableMapping
from typing import Sequence
from typing import TypeAlias
from typing import TypedDict
from typing import TypeVar
from typing import Union

from pydantic import TypeAdapter
from pydantic import ValidationError
from pydantic import ValidationInfo
from pydantic import ValidatorFunctionWrapHandler
from pydantic import WrapValidator
from pydantic_core import PydanticCustomError
from typing_extensions import TypeAliasType

CommandFunc = Callable[[argparse.Namespace], None]


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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


# Source: https://docs.pydantic.dev/2.7/concepts/types/#named-recursive-types
def json_custom_error_validator(
    value: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo
) -> Any:
    """Simplify the error message to avoid a gross error stemming from
    exhaustive checking of all union options.
    """  # noqa: D205
    try:
        return handler(value)
    except ValidationError:
        raise PydanticCustomError(
            "invalid_json",
            "Input is not valid json",
        ) from None


Json = TypeAliasType(
    "Json",
    Annotated[
        Union[Mapping[str, "Json"], Sequence["Json"], str, int, float, bool, None],
        WrapValidator(json_custom_error_validator),
    ],
)
JsonMapping = Mapping[str, Json]
QueryParams = MutableMapping[str, str | int | None]


T = TypeVar("T")


@lru_cache(maxsize=100)
def get_type_adapter(t: type[T]) -> TypeAdapter[T]:
    """Get the type adapter for a given type.

    :param t: The type to get the adapter for.
    :returns: The type adapter.

    """
    return TypeAdapter(t)
