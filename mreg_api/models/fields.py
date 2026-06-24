"""Custom field types for Pydantic models.

The types resolve to basic types like str, int, etc., but with additional
validation added to them. The types are used in Pydantic models for consistent
validation of common fields such as hostnames, MAC addresses, etc.

Warning:
Values constructed from these types should NOT be checked at runtime with isinstance()!
Pydantic will always coerce these types to their schema types (str, int, etc.).

"""

from __future__ import annotations

import logging
import re
from typing import Annotated
from typing import Any
from typing import NewType

from pydantic import AfterValidator
from pydantic import BeforeValidator
from pydantic_extra_types.mac_address import MacAddress as PydanticMacAddress

from mreg_api.exceptions import InputFailure
from mreg_api.types import get_type_adapter

logger = logging.getLogger(__name__)

HostName = NewType("HostName", str)
"""Validated hostname string type."""

_HOSTNAME_RE = re.compile(r"^(\*\.)?([a-z0-9_][a-z0-9\-]*\.?)+$")


def parse_hostname(value: str, domain: str = "") -> HostName:
    """Normalise and optionally expand a hostname.

    Normalisation: lowercase, strip trailing dot, validate shape.
    Expansion: if no dot remains after normalisation and ``domain`` is non-empty,
    append the domain.

    Args:
        value: Raw hostname string.
        domain: Domain to append when the name has no dot after normalisation.
            Pass ``""`` (default) to skip expansion.

    Returns:
        Normalised (and optionally expanded) :data:`HostName`.

    Raises:
        InputFailure: If the value is not a valid hostname shape.
    """
    value = value.lower()
    if value.endswith("."):  # strip trailing dot - may be unnecessary
        value = value[:-1]
    if _HOSTNAME_RE.search(value) is None:
        raise InputFailure(f"Invalid input for hostname: {value}")
    if "." not in value and domain:
        value = f"{value}.{domain}"
    return HostName(value)


HostNameField = Annotated[HostName, AfterValidator(parse_hostname)]
"""Pydantic-compatible hostname field type.

Use as a model field type where the server returns hostnames that should be
normalised (lowercase, trailing dot stripped) on deserialisation.

No expansion occurs, since the server should always return fully qualified hostnames.
Furthermore, we have no way to know what domain to expand to, so we don't attempt it.

TODO: Consider if we should remove validation of incoming hostnames from the server.
It is not really the client's responsibility to validate server data to _this_ level
of granularity.
"""


class MacAddress(PydanticMacAddress):
    """MAC address string type used in Pydantic models."""

    @classmethod
    def parse(cls, obj: Any) -> MacAddress | None:
        """Parse a MAC address from a string. Returns None if the MAC address is invalid.

        Args:
            obj: The object to parse.

        Returns:
            The MAC address as a string or None if it is invalid.
        """
        try:
            return cls.parse_or_raise(obj)
        except InputFailure:
            return None

    @classmethod
    def parse_or_raise(cls, obj: Any) -> MacAddress:
        """Parse a MAC address from a string. Returns the MAC address as a string.

        Args:
            obj: The object to parse.

        Returns:
            The MAC address as a string.

        Raises:
            ValueError: If the object is not a valid MAC address.
        """
        try:
            adapter = get_type_adapter(cls)
            return cls(adapter.validate_python(obj))
        except ValueError as e:
            raise InputFailure(f"Invalid MAC address '{obj}'") from e


def _extract_name(value: Any) -> str:
    """Extract the "name" value from a dictionary.

    Args:
        value: Dictionary containing the name.

    Returns:
        Extracted name as a string.
    """
    if isinstance(value, dict):
        try:
            return str(value["name"])  # pyright: ignore[reportUnknownArgumentType]
        except KeyError:
            logger.error("No 'name' key in %s", value)  # pyright: ignore[reportUnknownArgumentType]
            return ""
    return value


def _remove_falsy_list_items(value: Any) -> Any:
    """Remove falsy items from a list.

    For use in validators only.
    """
    if isinstance(value, list):
        return [i for i in value if i]  # pyright: ignore[reportUnknownVariableType]
    return value


NameList = Annotated[
    list[Annotated[str, BeforeValidator(_extract_name)]],
    AfterValidator(_remove_falsy_list_items),
]
"""List of names extracted from a list of dicts."""
