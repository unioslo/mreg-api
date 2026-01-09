"""Pydantic models for the mreg_cli package."""

from __future__ import annotations

import ipaddress
import logging
from datetime import datetime
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Literal
from typing import Self
from typing import TypeVar
from typing import cast
from typing import overload

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import ValidationError as PydanticValidationError
from pydantic import field_validator
from typing_extensions import Unpack

from mreg_api.api.abstracts import APIMixin
from mreg_api.api.abstracts import FrozenModelWithTimestamps
from mreg_api.api.endpoints import Endpoint
from mreg_api.api.history import HistoryItem
from mreg_api.api.history import HistoryResource
from mreg_api.exceptions import DeleteError
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import ForceMissing
from mreg_api.exceptions import InputFailure
from mreg_api.exceptions import InvalidIPAddress
from mreg_api.exceptions import InvalidIPv4Address
from mreg_api.exceptions import InvalidIPv6Address
from mreg_api.exceptions import InvalidNetwork
from mreg_api.types import IP_AddressT
from mreg_api.types import IP_NetworkT
from mreg_api.utilities.api import get_item_by_key_value
from mreg_api.utilities.api import get_typed
from mreg_api.utilities.shared import convert_wildcard_to_regex

logger = logging.getLogger(__name__)

T = TypeVar("T")

IPNetMode = Literal["ipv4", "ipv6", "ip", "network", "networkv4", "networkv6"]


class NetworkOrIP(BaseModel):
    """A model for either a network or an IP address."""

    ip_or_network: IP_AddressT | IP_NetworkT

    @classmethod
    def validate(cls, value: str | IP_AddressT | IP_NetworkT | Self) -> Self:
        """Create a NetworkOrIP model instance from a value.

        This constructor validates and wraps the IP/network in the model.

        :param value:The value to convert (string or IP object)
        :returns: A NetworkOrIP model instance
        :raises InputFailure: If validation fails
        """
        if isinstance(value, NetworkOrIP):
            return cls.validate(value.ip_or_network)
        try:
            return cls(ip_or_network=value)  # pyright: ignore[reportArgumentType] # validator handles this
        except PydanticValidationError as e:
            raise InputFailure(f"Invalid IP address or network: {value}") from e

    @overload
    @classmethod
    def parse_or_raise(cls, value: Any, mode: None = None) -> IP_AddressT | IP_NetworkT: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: Any, mode: Literal["ip"]) -> IP_AddressT: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: Any, mode: Literal["ipv4"]) -> ipaddress.IPv4Address: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: Any, mode: Literal["ipv6"]) -> ipaddress.IPv6Address: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: Any, mode: Literal["network"]) -> IP_NetworkT: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: Any, mode: Literal["networkv4"]) -> ipaddress.IPv4Network: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: Any, mode: Literal["networkv6"]) -> ipaddress.IPv6Network: ...

    @classmethod
    def parse_or_raise(
        cls, value: Any, mode: IPNetMode | None = None
    ) -> IP_AddressT | IP_NetworkT:
        """Parse a value as an IP address or network.

        Optionally specify the mode to validate the input as.

        :param value:The value to parse.
        :param mode: The mode to validate the input as.
        :returns: The parsed value as an IP address or network.
        :raises IPNetworkWarning: If the value is not an IP address or network.
        """
        ipnet = cls.validate(value)
        funcmap: dict[IPNetMode, Callable[..., IP_AddressT | IP_NetworkT]] = {
            "ip": cls.as_ip,
            "ipv4": cls.as_ipv4,
            "ipv6": cls.as_ipv6,
            "network": cls.as_network,
            "networkv4": cls.as_ipv4_network,
            "networkv6": cls.as_ipv6_network,
        }
        if mode and (func := funcmap.get(mode)):
            return func(ipnet)
        return ipnet.ip_or_network

    @overload
    @classmethod
    def parse(cls, value: Any, mode: None = None) -> IP_AddressT | IP_NetworkT | None: ...

    @overload
    @classmethod
    def parse(cls, value: Any, mode: Literal["ip"]) -> IP_AddressT | None: ...

    @overload
    @classmethod
    def parse(cls, value: Any, mode: Literal["ipv4"]) -> ipaddress.IPv4Address | None: ...

    @overload
    @classmethod
    def parse(cls, value: Any, mode: Literal["ipv6"]) -> ipaddress.IPv6Address | None: ...

    @overload
    @classmethod
    def parse(cls, value: Any, mode: Literal["network"]) -> IP_NetworkT | None: ...

    @overload
    @classmethod
    def parse(cls, value: Any, mode: Literal["networkv4"]) -> ipaddress.IPv4Network | None: ...

    @overload
    @classmethod
    def parse(cls, value: Any, mode: Literal["networkv6"]) -> ipaddress.IPv6Network | None: ...

    @classmethod
    def parse(cls, value: Any, mode: IPNetMode | None = None) -> IP_AddressT | IP_NetworkT | None:
        """Parse a value as an IP address or network, or None if parsing fails.

        Optionally specify the mode to validate the input as.

        :param value:The value to parse.
        :param mode: The mode to validate the input as.
        :returns: The parsed value as an IP address or network, or None.
        """
        try:
            return cls.parse_or_raise(value, mode)
        except ValueError:
            return None

    @field_validator("ip_or_network", mode="before")
    @classmethod
    def validate_ip_or_network(cls, value: Any) -> IP_AddressT | IP_NetworkT:
        """Validate and convert the input to an IP address or network."""
        if not isinstance(value, str):
            return value

        value = value.removesuffix("/")

        try:
            return ipaddress.ip_address(value)
        except ValueError:
            pass

        try:
            return ipaddress.ip_network(value)
        except ValueError:
            pass

        raise InputFailure(f"Invalid input for IP address or network: {value}")

    def __str__(self) -> str:
        """Return the value as a string."""
        return str(self.ip_or_network)

    def is_ipv4(self) -> bool:
        """Return True if the value is an IPv4 address."""
        return isinstance(self.ip_or_network, ipaddress.IPv4Address)

    def as_ipv4(self) -> ipaddress.IPv4Address:
        """Return the value as an IPv4 address."""
        if not self.is_ipv4():
            raise InvalidIPv4Address("Value is not an IPv4 address.")
        return cast(ipaddress.IPv4Address, self.ip_or_network)

    def as_ipv6(self) -> ipaddress.IPv6Address:
        """Return the value as an IPv6 address."""
        if not self.is_ipv6():
            raise InvalidIPv6Address("Value is not an IPv6 address.")
        return cast(ipaddress.IPv6Address, self.ip_or_network)

    def as_ip(self) -> IP_AddressT:
        """Return the value as an IP address."""
        if not self.is_ip():
            raise InvalidIPAddress(f"{self.ip_or_network} is not an IP address.")
        return cast(IP_AddressT, self.ip_or_network)

    def as_network(self) -> IP_NetworkT:
        """Return the value as a network."""
        if not self.is_network():
            raise InvalidNetwork(f"{self.ip_or_network} is not a network.")
        return cast(IP_NetworkT, self.ip_or_network)

    def as_ipv4_network(self) -> ipaddress.IPv4Network:
        """Return the value as a network."""
        if not self.is_ipv4_network():
            raise InvalidNetwork(f"{self.ip_or_network} is not an IPv4 network.")
        return cast(ipaddress.IPv4Network, self.ip_or_network)

    def as_ipv6_network(self) -> IP_NetworkT:
        """Return the value as a network."""
        if not self.is_ipv6_network():
            raise InvalidNetwork(f"{self.ip_or_network} is not an IPv6 network.")
        return cast(ipaddress.IPv6Network, self.ip_or_network)

    def is_ipv6(self) -> bool:
        """Return True if the value is an IPv6 address."""
        return isinstance(self.ip_or_network, ipaddress.IPv6Address)

    def is_ipv4_network(self) -> bool:
        """Return True if the value is an IPv4 network."""
        return isinstance(self.ip_or_network, ipaddress.IPv4Network)

    def is_ipv6_network(self) -> bool:
        """Return True if the value is an IPv6 network."""
        return isinstance(self.ip_or_network, ipaddress.IPv6Network)

    def is_ip(self) -> bool:
        """Return True if the value is an IP address."""
        return self.is_ipv4() or self.is_ipv6()

    def is_network(self) -> bool:
        """Return True if the value is a network."""
        return self.is_ipv4_network() or self.is_ipv6_network()


class WithHost(BaseModel):
    """Model for an object that has a host element."""

    host: int

    def resolve_host(self) -> Host | None:
        """Resolve the host ID to a Host object.

        Notes
        -----
            - This method will call the API to resolve the host ID to a Host object.
            - This assumes that there is a host attribute in the object.

        """
        data = get_item_by_key_value(Endpoint.Hosts, "id", str(self.host))

        if not data:
            return None

        return Host.model_validate(data)


class WithZone(BaseModel, APIMixin):
    """Model for an object that has a zone element."""

    zone: int

    def resolve_zone(self) -> ForwardZone | None:
        """Resolve the zone ID to a (Forward)Zone object.

        Notes
        -----
            - This method will call the API to resolve the zone ID to a Zone object.
            - This assumes that there is a zone attribute in the object.

        """
        data = get_item_by_key_value(Endpoint.ForwardZones, "id", str(self.zone))

        if not data:
            return None

        return ForwardZone.model_validate(data)


class WithTTL(BaseModel, APIMixin):
    """Model for an object that needs to work with TTL values."""

    _ttl_nullable: ClassVar[bool] = True
    """TTL field(s) of model are nullable."""

    @property
    def MAX_TTL(self) -> int:
        """Return the maximum TTL value."""
        return 68400

    @property
    def MIN_TTL(self) -> int:
        """Return the minimum TTL value."""
        return 300

    def set_ttl(self, ttl: str | int | None, field: str | None = None) -> Self:
        """Set a new TTL for the object and returns the updated object.

        Updates the `ttl` field of the object unless a different field name
        is specified.

        :param ttl: The TTL value to set. Can be an integer, "default", or None.
        :param field: The field to set the TTL value in.
        :raises InputFailure: If the TTL value is outside the bounds.
        :returns: The updated object.
        """
        # NOTE: could add some sort of validation that model has `field`
        ttl_field = field or "ttl"

        # str args can either be numeric or "default"
        # Turn it into an int or None
        if isinstance(ttl, str):
            if self._ttl_nullable and ttl == "default":
                ttl = None
            else:
                try:
                    ttl = int(ttl)
                except ValueError as e:
                    raise InputFailure(f"Invalid TTL value: {ttl}") from e

        if isinstance(ttl, int):
            ttl = self.valid_numeric_ttl(ttl)

        return self.patch({ttl_field: ttl})

    def valid_numeric_ttl(self, ttl: int) -> int:
        """Return a valid TTL value.

        Valid TTL values are: 300 - 68400.

        :param ttl: The TTL target to set.
        :raises InputFailure: If the TTL value is outside the bounds.
        :returns: A valid TTL vale
        """
        if ttl < self.MIN_TTL or ttl > self.MAX_TTL:
            raise InputFailure(f"Invalid TTL value: {ttl} ({self.MIN_TTL}->{self.MAX_TTL})")

        return ttl


class WithName(BaseModel, APIMixin):
    """Mixin type for an object that has a name element."""

    __name_field__: str = "name"
    """Name of the API field that holds the object's name."""

    __name_lowercase__: bool = False
    """Lower case name in API requests."""

    @classmethod
    def _case_name(cls, name: str) -> str:
        """Set the name case based on the class attribute."""
        return name.lower() if cls.__name_lowercase__ else name

    @classmethod
    def get_by_name(cls, name: str) -> Self | None:
        """Get a resource by name.

        :param name: The resource name to search for.
        :returns: The resource if found.
        """
        return cls.get_by_field(cls.__name_field__, cls._case_name(name))

    @classmethod
    def get_by_name_and_raise(cls, name: str) -> None:
        """Get a resource by name, raising EntityAlreadyExists if found.

        :param name: The resource name to search for.
        :raises EntityAlreadyExists: If the resource is found.
        """
        return cls.get_by_field_and_raise(cls.__name_field__, cls._case_name(name))

    @classmethod
    def get_by_name_or_raise(cls, name: str) -> Self:
        """Get a resource by name, raising EntityNotFound if not found.

        :param name: The resource name to search for.
        :returns: The resource.
        :raises EntityNotFound: If the resource is not found.
        """
        return cls.get_by_field_or_raise(cls.__name_field__, cls._case_name(name))

    @classmethod
    def get_list_by_name_regex(cls, name: str) -> list[Self]:
        """Get multiple resources by a name regex.

        :param name: The regex pattern for names to search for.
        :returns: A list of resource objects.
        """
        param, value = convert_wildcard_to_regex(cls.__name_field__, cls._case_name(name), True)
        return get_typed(cls.endpoint(), list[cls], params={param: value})

    def rename(self, new_name: str) -> Self:
        """Rename the resource.

        :param new_name: The new name to set.
        :returns: The patched resource.
        """
        return self.patch({self.__name_field__: self._case_name(new_name)})


ClassVarNotSet = object()


def AbstractClassVar() -> Any:
    """Hack to implement an abstract class variable on a Pydantic model."""
    return ClassVarNotSet


class WithHistory(BaseModel, APIMixin):
    """Resource that supports history lookups.

    Subclasses must implement the `history_resource` class variable.
    """

    history_resource: ClassVar[HistoryResource] = AbstractClassVar()

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
        """Ensure that subclasses implement the history_resource class var."""
        # NOTE: Only works for Pydantic model subclasses!
        for attr in cls.__class_vars__:
            if getattr(cls, attr) == ClassVarNotSet:
                raise NotImplementedError(
                    f"Subclass {cls.__name__} must implement abstract class var `{attr}`."
                )
        return super().__init_subclass__(**kwargs)

    @classmethod
    def get_history(cls, name: str) -> list[HistoryItem]:
        """Get the history for the object."""
        return HistoryItem.get(name, cls.history_resource)


class NameServer(FrozenModelWithTimestamps, WithTTL):
    """Model for representing a nameserver within a DNS zone."""

    id: int  # noqa: A003
    name: str

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Nameservers


class Permission(FrozenModelWithTimestamps, APIMixin):
    """Model for a permission object."""

    id: int  # noqa: A003
    group: str
    range: IP_NetworkT  # noqa: A003
    regex: str
    labels: list[int]

    @field_validator("range", mode="before")
    @classmethod
    def validate_ip_or_network(cls, value: Any) -> IP_NetworkT:
        """Validate and convert the input to a network."""
        try:
            return ipaddress.ip_network(value)
        except ValueError as e:
            raise InputFailure(f"Invalid input for network: {value}") from e

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.PermissionNetgroupRegex

    @classmethod
    def add_label(self, label_name: str) -> Self:
        """Add a label to the permission.

        :param label_name: The name of the label to add.
        :returns: The updated Permission object.
        """
        label = Label.get_by_name_or_raise(label_name)
        if label.id in self.labels:
            raise EntityAlreadyExists(f"The permission already has the label {label_name!r}")

        label_ids = self.labels.copy()
        label_ids.append(label.id)
        return self.patch({"labels": label_ids})

    def remove_label(self, label_name: str) -> Self:
        """Remove a label from the permission.

        :param label_name: The name of the label to remove.
        :returns: The updated Permission object.
        """
        label = Label.get_by_name_or_raise(label_name)
        if label.id not in self.labels:
            raise EntityNotFound(f"The permission does not have the label {label_name!r}")

        label_ids = self.labels.copy()
        label_ids.remove(label.id)
        return self.patch({"labels": label_ids})


def is_reverse_zone_name(name: str) -> bool:
    """Determine if a zone is a reverse zone by its name.

    :param name: The name of the zone.
    :returns: True if the zone is a reverse zone.
    """
    return name.endswith(".arpa")


class Zone(FrozenModelWithTimestamps, WithTTL, APIMixin):
    """Model representing a DNS zone with various attributes and related nameservers."""

    id: int  # noqa: A003
    nameservers: list[NameServer]
    updated: bool
    primary_ns: str
    email: str
    serialno: int
    serialno_updated_at: datetime
    refresh: int
    retry: int
    expire: int
    soa_ttl: int
    default_ttl: int
    name: str

    # Specify that TTL fields are NOT nullable for Zone objects
    _ttl_nullable: ClassVar[bool] = False

    def is_delegated(self) -> bool:
        """Return True if the zone is delegated."""
        return False

    def is_reverse(self) -> bool:
        """Return True if the zone is a reverse zone."""
        return is_reverse_zone_name(self.name)

    # Default to forward zone endpoints for the base class
    # This can be overridden in the subclasses
    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZones

    @classmethod
    def endpoint_nameservers(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZonesNameservers

    @classmethod
    @classmethod
    def ensure_delegation_in_zone(self, name: str) -> None:
        """Ensure a delegation is in the zone.

        :param name: The name of the delegation to check.
        :returns: True if the delegation is in the zone.
        """
        if not name.endswith(f".{self.name}"):
            raise InputFailure(f"Delegation '{name}' is not in '{self.name}'")

    @classmethod
    def type_by_name(cls, name: str) -> type[ForwardZone | ReverseZone]:
        """Determine the zone type based on the name.

        :param name: The name of the zone.
        :returns: The zone type.
        """
        if is_reverse_zone_name(name):
            return ReverseZone
        return ForwardZone

    @classmethod
    def verify_nameservers(cls, nameservers: list[str], force: bool = False) -> None:
        """Verify that nameservers are in mreg and have A-records."""
        if not nameservers:
            raise InputFailure("At least one nameserver is required")

        errors: list[str] = []
        for nameserver in nameservers:
            try:
                host = Host.get_by_any_means_or_raise(nameserver)
            except EntityNotFound:
                if not force:
                    errors.append(f"{nameserver} is not in mreg, must force")
            else:
                if host.zone is None:
                    if not host.ipaddresses and not force:
                        errors.append(f"{nameserver} has no A-record/glue, must force")
        if errors:
            raise ForceMissing("\n".join(errors))

    @classmethod
    def create_zone(
        cls,
        name: str,
        email: str,
        primary_ns: list[str],
        force: bool,
    ) -> ForwardZone | ReverseZone | None:
        """Create a forward or reverse zone based on zone name.

        :param name: The name of the zone to create.
        :param email: The email address for the zone.
        :param primary_ns: The primary nameserver for the zone.
        :returns: The created zone object.
        """
        cls.verify_nameservers(primary_ns, force=force)
        zone_t = cls.type_by_name(name)
        zone_t.get_zone_and_raise(name)
        return zone_t.create({"name": name, "email": email, "primary_ns": primary_ns})

    @classmethod
    def get_zone(cls, name: str) -> ForwardZone | ReverseZone | None:
        """Get a zone by name.

        :param name: The name of the zone to get.
        :returns: The zone object.
        """
        zone_t = cls.type_by_name(name)
        return zone_t.get_by_name(name)

    @classmethod
    def get_zone_or_raise(cls, name: str) -> ForwardZone | ReverseZone:
        """Get a zone by name, and raise if not found.

        :param name: The name of the zone to get.
        :returns: The zone object.
        """
        zone_t = cls.type_by_name(name)
        return zone_t.get_by_name_or_raise(name)

    @classmethod
    def get_zone_and_raise(cls, name: str) -> None:
        """Get a zone by name, and raise if found.

        :param name: The name of the zone to get.
        """
        zone_t = cls.type_by_name(name)
        return zone_t.get_by_name_and_raise(name)

    def get_subzones(self) -> list[Self]:
        """Get subzones of the zone, excluding self.

        :returns: A list of subzones.
        """
        zones = self.get_list_by_field("name__endswith", f".{self.name}")
        return [zone for zone in zones if zone.name != self.name]

    def ensure_deletable(self) -> None:
        """Ensure the zone can be deleted. Raises exception if not.

        :raises DeleteError: If zone has entries or subzones.
        """
        # XXX: Not a fool proof check, as e.g. SRVs are not hosts. (yet.. ?)
        hosts = Host.get_list_by_field("zone", self.id)
        if hosts:
            raise DeleteError(f"Zone has {len(hosts)} registered entries. Can not delete.")

        zones = self.get_subzones()
        if zones:
            names = ", ".join(zone.name for zone in zones)
            raise DeleteError(f"Zone has registered subzones: '{names}'. Can not delete")

    def delete_zone(self, force: bool) -> bool:
        """Delete the zone.

        :param force: Whether to force the deletion.
        :returns: True if the deletion was successful.
        """
        if not force:
            self.ensure_deletable()
        return self.delete()
