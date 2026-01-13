"""Pydantic models for API resources."""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Iterable
from datetime import date
from datetime import datetime
from datetime import timedelta
from functools import cached_property
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Literal
from typing import Protocol
from typing import Self
from typing import TypeVar
from typing import cast
from typing import overload

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError as PydanticValidationError
from pydantic import computed_field
from pydantic import field_validator
from typing_extensions import Unpack

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import APIError
from mreg_api.exceptions import DeleteError
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import EntityOwnershipMismatch
from mreg_api.exceptions import ForceMissing
from mreg_api.exceptions import InputFailure
from mreg_api.exceptions import InvalidIPAddress
from mreg_api.exceptions import InvalidIPv4Address
from mreg_api.exceptions import InvalidIPv6Address
from mreg_api.exceptions import InvalidNetwork
from mreg_api.exceptions import IPNetworkWarning
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.exceptions import PatchError
from mreg_api.exceptions import PostError
from mreg_api.exceptions import UnexpectedDataError
from mreg_api.models.abstracts import APIMixin
from mreg_api.models.abstracts import FrozenModel
from mreg_api.models.abstracts import FrozenModelWithTimestamps
from mreg_api.models.fields import HostName
from mreg_api.models.fields import MacAddress
from mreg_api.models.fields import NameList
from mreg_api.models.history import HistoryItem
from mreg_api.models.history import HistoryResource
from mreg_api.types import IP_AddressT
from mreg_api.types import IP_NetworkT
from mreg_api.types import QueryParams
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
    def parse_or_raise(cls, value: Any, mode: IPNetMode | None = None) -> IP_AddressT | IP_NetworkT:
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

        Notes:
        -----
            - This method will call the API to resolve the host ID to a Host object.
            - This assumes that there is a host attribute in the object.

        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        data = MregClient().get_item_by_key_value(Endpoint.Hosts, "id", str(self.host))

        if not data:
            return None

        return Host.model_validate(data)


class WithZone(BaseModel, APIMixin):
    """Model for an object that has a zone element."""

    zone: int

    def resolve_zone(self) -> ForwardZone | None:
        """Resolve the zone ID to a (Forward)Zone object.

        Notes:
        -----
            - This method will call the API to resolve the zone ID to a Zone object.
            - This assumes that there is a zone attribute in the object.

        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        data = MregClient().get_item_by_key_value(Endpoint.ForwardZones, "id", str(self.zone))

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
        from mreg_api.client import MregClient  # noqa: PLC0415

        param, value = convert_wildcard_to_regex(cls.__name_field__, cls._case_name(name), True)
        return MregClient().get_typed(cls.endpoint(), list[cls], params={param: value})

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

    def update_soa(
        self,
        primary_ns: str | None = None,
        email: str | None = None,
        serialno: int | None = None,
        refresh: int | None = None,
        retry: int | None = None,
        expire: int | None = None,
        soa_ttl: int | None = None,
    ) -> Self:
        """Update SOA (Start of Authority) record for the zone.

        :param primary_ns: The primary nameserver for the zone.
        :param email: The email address for the zone.
        :param serialno: The serial number for the zone.
        :param refresh: The refresh interval for the zone.
        :param retry: The retry interval for the zone.
        :param expire: The expire interval for the zone.
        :param soa_ttl: The TTL for the zone.
        """
        params: QueryParams = {
            "primary_ns": primary_ns,
            "email": email,
            "serialno": serialno,
            "refresh": refresh,
            "retry": retry,
            "expire": expire,
            "soa_ttl": self.valid_numeric_ttl(soa_ttl) if soa_ttl is not None else None,
        }
        params = {k: v for k, v in params.items() if v is not None}
        if not params:
            raise InputFailure("No parameters to update")
        return self.patch(params)

    def create_delegation(
        self,
        delegation: str,
        nameservers: list[str],
        comment: str,
        force: bool = False,
        fetch_after_create: bool = True,
    ) -> Delegation | None:
        """Create a delegation for the zone.

        :param delegation: The name of the delegation.
        :param nameservers: The nameservers for the delegation.
        :param comment: A comment for the delegation.
        :param force: Force creation if ns/zone doesn't exist.
        :returns: The created delegation object.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        self.ensure_delegation_in_zone(delegation)
        self.verify_nameservers(nameservers, force=force)

        if not force:
            # Ensure delegated zone exists and is same type as parent zone
            delegated_zone = Zone.get_zone(delegation)
            if not delegated_zone:
                raise InputFailure(f"Zone {delegation!r} does not exist. Must force.")
            if delegated_zone.is_reverse() != self.is_reverse():
                raise InputFailure(f"Delegation '{delegation}' is not a {self.__class__.__name__} zone")

        self.get_delegation_and_raise(delegation)

        cls = Delegation.type_by_zone(self)
        resp = MregClient().post(
            cls.endpoint().with_params(self.name),
            name=delegation,
            nameservers=nameservers,
            comment=comment,
        )
        if not resp or not resp.ok:
            raise PostError(f"Failed to create delegation {delegation!r} in zone {self.name!r}")

        if fetch_after_create:
            return self.get_delegation_or_raise(delegation)
        return None

    def get_delegation(self, name: str) -> ForwardZoneDelegation | ReverseZoneDelegation | None:
        """Get a delegation for the zone by name.

        :param name: The name of the delegation to get.
        :returns: The delegation object if found.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        self.ensure_delegation_in_zone(name)
        cls = Delegation.type_by_zone(self)
        resp = MregClient().get(cls.endpoint_with_id(self, name), ok404=True)
        if not resp:
            return None
        return cls.model_validate_json(resp.text)

    def get_delegation_or_raise(self, name: str) -> ForwardZoneDelegation | ReverseZoneDelegation:
        """Get a delegation for the zone by name, raising EntityNotFound if not found.

        :param zone: The zone to search in.
        :param name: The name of the delegation to get.
        :returns: The delegation object.
        :raises EntityNotFound: If the delegation is not found.
        """
        delegation = self.get_delegation(name)
        if not delegation:
            raise EntityNotFound(f"Could not find delegation {name!r} in zone {name!r}")
        return delegation

    def get_delegation_and_raise(self, name: str) -> None:
        """Get a delegation for the zone by name, raising EntityAlreadyExists if found.

        :param zone: The zone to search in.
        :param name: The name of the delegation to get.
        :raises EntityAlreadyExists: If the delegation is found.
        """
        delegation = self.get_delegation(name)
        if delegation:
            raise EntityAlreadyExists(f"Zone {self.name!r} already has a delegation named {name!r}")

    def get_delegations(self) -> list[ForwardZoneDelegation | ReverseZoneDelegation]:
        """Get all delegations for a zone.

        :param zone: The zone to search in.
        :param name: The name of the delegation to get.
        :returns: The delegation object.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        cls = Delegation.type_by_zone(self)
        return MregClient().get_typed(cls.endpoint().with_params(self.name), list[cls])

    def delete_delegation(self, name: str) -> bool:
        """Delete a delegation from the zone.

        :param delegation: The name of the delegation.
        :returns: True if the deletion was successful.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        # Check if delegation exists
        self.ensure_delegation_in_zone(name)  # check name
        delegation = self.get_delegation_or_raise(name)
        resp = MregClient().delete(delegation.endpoint_with_id(self, name))
        return resp.ok if resp else False

    def set_delegation_comment(self, name: str, comment: str) -> None:
        """Set the comment for a delegation.

        :param name: The name of the delegation.
        :param comment: The comment to set.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        delegation = self.get_delegation_or_raise(name)
        resp = MregClient().patch(delegation.endpoint_with_id(self, delegation.name), comment=comment)
        if not resp or not resp.ok:
            raise PatchError(f"Failed to update comment for delegation {delegation.name!r}")

    def set_default_ttl(self, ttl: int) -> Self:
        """Set the default TTL for the zone.

        :param ttl: The TTL to set.
        """
        return self.set_ttl(ttl, "default_ttl")

    def update_nameservers(self, nameservers: list[str], force: bool = False) -> None:
        """Update the nameservers of the zone.

        :param nameservers: The new nameservers for the zone.
        :param force: Whether to force the update.
        :returns: True if the update was successful.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        self.verify_nameservers(nameservers, force=force)
        path = self.endpoint_nameservers().with_params(self.name)
        resp = MregClient().patch(path, primary_ns=nameservers)
        if not resp or not resp.ok:
            raise PatchError(f"Failed to update nameservers for {self.__class__.__name__} {self.name!r}")


class ForwardZone(Zone, WithName, APIMixin):
    """A forward zone."""

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZones

    @classmethod
    def endpoint_nameservers(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZonesNameservers

    @classmethod
    def get_from_hostname(cls, hostname: HostName) -> ForwardZoneDelegation | ForwardZone | None:
        """Get the zone from a hostname.

        Note: This method may return either a ForwardZoneDelegation or a ForwardZone object.

        :param hostname: The hostname to search for.
        :returns: The zone if found, None otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().get(Endpoint.ForwardZoneForHost.with_id(hostname), ok404=True)
        if not resp:
            return None

        zoneblob = resp.json()

        if "delegate" in zoneblob:
            return ForwardZoneDelegation.model_validate(zoneblob)

        if "zone" in zoneblob:
            return ForwardZone.model_validate(zoneblob["zone"])

        if "delegation" in zoneblob:
            return ForwardZoneDelegation.model_validate(zoneblob["delegation"])

        raise UnexpectedDataError(f"Unexpected response from server: {zoneblob}", resp)


class ReverseZone(Zone, WithName, APIMixin):
    """A reverse zone."""

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ReverseZones

    @classmethod
    def endpoint_nameservers(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ReverseZonesNameservers


class Delegation(FrozenModelWithTimestamps, WithZone):
    """A delegated zone."""

    id: int  # noqa: A003
    nameservers: list[NameServer]
    name: str
    comment: str | None = None

    # NOTE: Delegations are created through zone objects!
    # Call Zone.create_delegation() on an existing zone to create one.
    # We do not implement APIMixin here, since we cannot determine
    # the path and type of a delegation to create without information
    # about the zone in which to create it.

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZonesDelegations

    @classmethod
    def endpoint_with_id(cls, zone: Zone, name: str) -> str:
        """Return the path to a delegation in a specific zone."""
        if cls.is_reverse():
            endpoint = Endpoint.ReverseZonesDelegationsZone
        else:
            endpoint = Endpoint.ForwardZonesDelegationsZone
        return endpoint.with_params(zone.name, name)

    def is_delegated(self) -> bool:
        """Return True if the zone is delegated."""
        return True

    @classmethod
    def is_reverse(cls) -> bool:
        """Return True if the delegation is for a reverse zone."""
        return False

    @classmethod
    def type_by_zone(cls, zone: Zone) -> type[ForwardZoneDelegation | ReverseZoneDelegation]:
        """Get the delegation type for a zone."""
        if zone.is_reverse():
            return ReverseZoneDelegation
        return ForwardZoneDelegation


class ForwardZoneDelegation(Delegation, APIMixin):
    """A forward zone delegation."""

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZonesDelegations


class ReverseZoneDelegation(Delegation, APIMixin):
    """A reverse zone delegation."""

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ReverseZonesDelegations

    @classmethod
    def is_reverse(cls) -> bool:
        """Return True if the delegation is for a reverse zone."""
        return True


class HostPolicy(FrozenModel, WithName):
    """Base model for Host Policy objects.

    Note:
    ----
    Host policy models in MREG have a different `created_at` field than
    other models. It is called `create_date` and is a date - not a datetime.

    This model has a custom validator to validate and convert the `create_date`
    field to a datetime object with the expected `created_at` name.

    """

    created_at_tz_naive: datetime = Field(
        ...,
        validation_alias=AliasChoices("create_date", "created_at", "created_at_tz_naive"),
        exclude=True,
        repr=False,
    )
    """Constructed datetime field from `create_date` in the API.

    WARNING
    ----
    DO NOT USE THIS FIELD FOR TIMEZONE-AWARE COMPARISONS!
    Always use `created_at` instead when comparing with timezone-aware
    fields such as `update_time`."""
    updated_at: datetime
    name: str
    description: str

    @field_validator("created_at_tz_naive", mode="before")
    @classmethod
    def validate_created_at(cls, value: Any) -> datetime:
        """Convert a datetime string to a datetime object.

        :param value: The input value - should be a datetime string.
        :returns: The input value converted to a datetime object.
        """
        # Fast path for str (most likely input type)
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        # Re-validating model (field is already a datetime object)
        elif isinstance(value, datetime):
            return value
        # Date object is also valid input (parity with API)
        elif isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        return value  # let pydantic throw the ValidationError

    @computed_field  # noqa: A003
    def created_at(self) -> datetime:
        """Creation time."""
        return self.created_at_tz_naive.replace(tzinfo=self.updated_at.tzinfo)

    # Fetching Host Policy objects is a special case where we cannot
    # re-use the methods defined in WithName, because we don't have an endpoint
    # defined on the class that can fetch both Roles and Atoms.
    # Thus, we need to define our own implementations of these methods.
    @classmethod
    def get_role_or_atom(cls, name: str) -> Atom | Role | None:
        """Get an Atom or Role by name.

        :param name: The name to search for.
        :returns: The Atom or Role if found, else None.
        """
        funcs: list[Callable[[str], Atom | Role | None]] = [
            Atom.get_by_name,
            Role.get_by_name,
        ]
        for func in funcs:
            role_or_atom = func(name)
            if role_or_atom:
                return role_or_atom
        return None

    @classmethod
    def get_role_or_atom_or_raise(cls, name: str) -> Atom | Role:
        """Get an Atom or Role by name and raise if not found.

        :param name: The name to search for.
        :returns: The Atom or Role if found.
        :raises EntityNotFound: If the Atom or Role is not found.
        """
        role_or_atom = cls.get_role_or_atom(name)
        if role_or_atom:
            return role_or_atom
        raise EntityNotFound(f"Could not find an atom or a role with name {name}")

    @classmethod
    def get_role_or_atom_and_raise(cls, name: str) -> None:
        """Get an Atom or Role by name and raise if found.

        :param name: The name to search for.
        :returns: The Atom or Role if found.
        :raises EntityAlreadyExists: If the Atom or Role is found.
        """
        role_or_atom = cls.get_role_or_atom(name)
        if role_or_atom:
            raise EntityAlreadyExists(f"An atom or a role with name {name} already exists.")

    def set_description(self, description: str) -> Self:
        """Set a new description."""
        return self.patch({"description": description})


class Role(HostPolicy, WithHistory):
    """Model for a role."""

    id: int  # noqa: A003
    hosts: NameList
    atoms: NameList
    labels: list[int]

    history_resource: ClassVar[HistoryResource] = HistoryResource.HostPolicy_Role

    def __hash__(self) -> int:
        """Hash the role by ID and name."""
        return hash(str(self.id) + self.name)

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HostPolicyRoles

    @classmethod
    def get_roles_with_atom(cls, name: str) -> list[Self]:
        """Get all roles with a specific atom.

        :param atom: Name of the atom to search for.
        :returns: A list of Role objects.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(cls.endpoint(), list[cls], params={"atoms__name__exact": name})

    def add_atom(self, atom_name: str) -> bool:
        """Add an atom to the role.

        :param atom_name: The name of the atom to add.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        # Ensure the atom exists
        Atom.get_by_name_or_raise(atom_name)
        for atom in self.atoms:
            if atom_name == atom:
                raise EntityAlreadyExists(f"Atom {atom!r} already a member of role {self.name!r}")

        resp = MregClient().post(Endpoint.HostPolicyRolesAddAtom.with_params(self.name), name=atom_name)
        return resp.ok if resp else False

    def remove_atom(self, atom_name: str) -> bool:
        """Remove an atom from the role.

        :param atom_name: The name of the atom to remove.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        for atom in self.atoms:
            if atom_name == atom:
                break
        else:
            raise EntityOwnershipMismatch(f"Atom {atom_name!r} not a member of {self.name!r}")

        resp = MregClient().delete(Endpoint.HostPolicyRolesRemoveAtom.with_params(self.name, atom))
        return resp.ok if resp else False

    def get_labels(self) -> list[Label]:
        """Get the labels associated with the role.

        :returns: A list of Label objects.
        """
        return [Label.get_by_id_or_raise(id_) for id_ in self.labels]

    def add_label(self, label_name: str) -> Self:
        """Add a label to the role.

        :param label_name: The name of the label to add.

        :returns: The updated Role object.
        """
        label = Label.get_by_name_or_raise(label_name)
        if label.id in self.labels:
            raise EntityAlreadyExists(f"The role {self.name!r} already has the label {label_name!r}")

        label_ids = self.labels.copy()
        label_ids.append(label.id)
        return self.patch({"labels": label_ids})

    def remove_label(self, label_name: str) -> Self:
        """Add a label to the role.

        :param label_name: The name of the label to add.

        :returns: The updated Role object.
        """
        label = Label.get_by_name_or_raise(label_name)
        if label.id not in self.labels:
            raise EntityOwnershipMismatch(
                f"The role {self.name!r} doesn't have the label {label_name!r}"
            )

        label_ids = self.labels.copy()
        label_ids.remove(label.id)
        return self.patch({"labels": label_ids})

    def add_host(self, name: str) -> bool:
        """Add a host to the role by name.

        :param name: The name of the host to add.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().post(Endpoint.HostPolicyRolesAddHost.with_params(self.name), name=name)
        return resp.ok if resp else False

    def remove_host(self, name: str) -> bool:
        """Remove a host from the role by name.

        :param name: The name of the host to remove.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().delete(Endpoint.HostPolicyRolesRemoveHost.with_params(self.name, name))
        return resp.ok if resp else False

    def delete(self) -> bool:
        """Delete the role."""
        if self.hosts:
            hosts = ", ".join(self.hosts)
            raise DeleteError(f"Role {self.name!r} used on hosts: {hosts}")
        return super().delete()


class Atom(HostPolicy, WithHistory):
    """Model for an atom."""

    id: int  # noqa: A003
    roles: NameList

    history_resource: ClassVar[HistoryResource] = HistoryResource.HostPolicy_Atom

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HostPolicyAtoms

    def delete(self) -> bool:
        """Delete the atom."""
        roles = Role.get_roles_with_atom(self.name)
        if self.roles:
            roles = ", ".join(self.roles)
            raise DeleteError(f"Atom {self.name!r} used in roles: {roles}")
        return super().delete()


class Label(FrozenModelWithTimestamps, WithName):
    """Model for a label."""

    id: int  # noqa: A003
    name: str
    description: str

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Labels

    @classmethod
    def get_all(cls) -> list[Self]:
        """Get all labels.

        :returns: A list of Label objects.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(cls.endpoint(), list[cls], params={"ordering": "name"})

    @classmethod
    def get_by_id_or_raise(cls, _id: int) -> Self:
        """Get a Label by ID.

        :param _id: The Label ID to search for.
        :returns: The Label if found.
        :raises EntityNotFound: If the Label is not found.
        """
        label = cls.get_by_id(_id)
        if not label:
            raise EntityNotFound(f"Label with ID {_id} not found.")
        return label

    def set_description(self, description: str) -> Self:
        """Set a new description."""
        return self.patch({"description": description})


class ExcludedRange(FrozenModelWithTimestamps):
    """Model for an excluded IP range for a network."""

    id: int  # noqa: A003
    network: int
    start_ip: IP_AddressT
    end_ip: IP_AddressT

    def excluded_ips(self) -> int:
        """Return the number of IP addresses in the excluded range."""
        return int(self.end_ip) - int(self.start_ip) + 1


class Network(FrozenModelWithTimestamps, APIMixin):
    """Model for a network."""

    id: int  # noqa: A003
    excluded_ranges: list[ExcludedRange]
    network: str  # for now
    description: str
    vlan: int | None = None
    dns_delegated: bool
    category: str
    location: str
    frozen: bool
    reserved: int
    policy: NetworkPolicy | None = None
    communities: list[Community] = []
    max_communities: int | None = None

    def __hash__(self):
        """Return a hash of the network."""
        return hash((self.id, self.network))

    @cached_property
    def ip_network(self) -> IP_NetworkT:
        """IP network object for the network."""
        try:
            return NetworkOrIP.parse_or_raise(self.network, mode="network")
        except IPNetworkWarning as e:
            logger.error("Invalid network address %s for network with ID %s", self.network, self.id)
            raise e

    @property
    def network_address(self) -> IP_AddressT:
        """The network address of the network."""
        return self.ip_network.network_address

    @property
    def broadcast_address(self) -> IP_AddressT:
        """The broadcast address of the network."""
        return self.ip_network.broadcast_address

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Networks

    @classmethod
    def dummy_network_from_ip(cls, ip: IPAddress) -> Self:
        """Create a Network object for an unknown network given an IP.

        NOTE: Does not perform any API calls. This is purely to work around
        the fact that MREG supports creating IP addresses for networks that
        are not registered in MREG.
        """
        if ip.is_ipv4():
            network_addr = "0.0.0.0/24"
            description = "Unknown IPv4 network"
        else:
            network_addr = "::/64"
            description = "Unknown IPv6 network"

        return cls(
            id=0,
            excluded_ranges=[],
            network=network_addr,
            description=description,
            vlan=None,
            dns_delegated=False,
            category="",
            location="",
            frozen=False,
            reserved=0,
            policy=None,
            communities=[],
            # epoch time
            created_at=datetime.fromtimestamp(0),
            updated_at=datetime.fromtimestamp(0),
        )

    @classmethod
    def get_by_any_means(cls, identifier: str) -> Self | None:
        """Get a network by the given identifier.

        - If the identifier is numeric, it is treated as an ID.
        - If the identifier is a valid IP address, it is treated as an IP.
        - If the identifier is a valid network, it is treated as a network.

        :param identifier: The identifier to search for.
        :returns: The network if found.
        :raises EntityNotFound: If the network is not found.
        """
        # Check if identifier is IP or network
        try:
            net_or_ip = NetworkOrIP.validate(identifier)
        except InputFailure:
            pass
        else:
            # We (should) have a valid ip or network
            if net_or_ip.is_network():
                return cls.get_by_network(str(net_or_ip))
            elif net_or_ip.is_ip():
                return cls.get_by_ip(net_or_ip.as_ip())
        # Check if identifier is an ID
        if identifier.isdigit():
            try:
                return cls.get_by_id(int(identifier))
            except ValueError:
                pass
        return None

    @classmethod
    def get_by_any_means_or_raise(cls, identifier: str) -> Self:
        """Get a network by the given identifier, and raise if not found.

        See `get_by_any_means` for details.
        """
        net = cls.get_by_any_means(identifier)
        if not net:
            raise EntityNotFound(f"Network {identifier!r} not found.")
        return net

    @classmethod
    def get_by_ip(cls, ip: IP_AddressT) -> Self | None:
        """Get a network by IP address.

        :param ip: The IP address to search for.
        :returns: The network if found, None otherwise.
        :raises EntityNotFound: If the network is not found.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().get(Endpoint.NetworksByIP.with_id(str(ip)), ok404=True)
        if not resp:
            return None
        return cls.model_validate_json(resp.text)

    @classmethod
    def get_by_ip_or_raise(cls, ip: IP_AddressT) -> Network:
        """Get a network by IP address, and raise if not found.

        :param ip: The IP address to search for.
        :returns: The network if found, None otherwise.
        :raises EntityNotFound: If the network is not found.
        """
        network = cls.get_by_ip(ip)
        if not network:
            raise EntityNotFound(f"Network with IP address {ip} not found.")
        return network

    @classmethod
    def get_by_network(cls, network: str) -> Self | None:
        """Get a network by network address.

        :param network: The network string to search for.
        :returns: The network if found.
        """
        return cls.get_by_field("network", network)

    @classmethod
    def get_by_network_or_raise(cls, network: str) -> Self:
        """Get a network by its network address, and raise if not found.

        :param network: The network string to search for.
        :returns: The network if found.
        :raises EntityNotFound: If the network is not found.
        """
        net = cls.get_by_network(network)
        if not net:
            raise EntityNotFound(f"Network {network} not found.")
        return net

    def get_community_or_raise(self, name: str) -> Community:
        """Get a community by name, and raise if not found.

        :param name: The name of the community to search for.
        :returns: The community if found.
        :raises EntityNotFound: If the community is not found.
        """
        community = self.get_community(name)
        if not community:
            raise EntityNotFound(f"Community {name!r} not found.")
        return community

    def get_community(self, name: str) -> Community | None:
        """Get a community by name.

        :param name: The name of the community to search for.
        :returns: The community if found, None otherwise.
        """
        for community in self.communities:
            if community.name == name:
                return community
        return None

    def create_community(self, name: str, description: str) -> bool:
        """Create a community for the network."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().post(
            Endpoint.NetworkCommunities.with_params(self.network),
            name=name,
            description=description,
        )
        return resp.ok if resp else False

    def overlaps(self, other: Network | str | IP_NetworkT) -> bool:
        """Check if the network overlaps with another network."""
        # Network -> str -> ipaddress.IPv{4,6}Network
        if isinstance(other, Network):
            other = other.network
        if isinstance(other, str):
            other = NetworkOrIP.parse_or_raise(other, mode="network")

        self_net = NetworkOrIP.parse_or_raise(self.network, mode="network")
        return self_net.overlaps(other)

    def get_first_available_ip(self) -> IP_AddressT:
        """Return the first available IPv4 address of the network."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return ipaddress.ip_address(
            MregClient().get_typed(Endpoint.NetworksFirstUnused.with_params(self.network), str)
        )

    def get_reserved_ips(self) -> list[IP_AddressT]:
        """Return the reserved IP addresses of the network."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(
            Endpoint.NetworksReservedList.with_params(self.network), list[IP_AddressT]
        )

    def get_used_count(self) -> int:
        """Return the number of used IP addresses in the network."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(Endpoint.NetworksUsedCount.with_params(self.network), int)

    def get_used_list(self) -> list[IP_AddressT]:
        """Return the list of used IP addresses in the network."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(
            Endpoint.NetworksUsedList.with_params(self.network), list[IP_AddressT]
        )

    def get_unused_count(self) -> int:
        """Return the number of unused IP addresses in the network."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(Endpoint.NetworksUnusedCount.with_params(self.network), int)

    def get_unused_list(self) -> list[IP_AddressT]:
        """Return the list of unused IP addresses in the network."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(
            Endpoint.NetworksUnusedList.with_params(self.network), list[IP_AddressT]
        )

    def get_used_host_list(self) -> dict[str, list[str]]:
        """Return a dict of used IP addresses and their associated hosts."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(
            Endpoint.NetworksUsedHostList.with_params(self.network), dict[str, list[str]]
        )

    def get_ptroverride_host_list(self) -> dict[str, str]:
        """Return a dict of PTR override IP addresses and their associated hosts."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(
            Endpoint.NetworksPTROverrideHostList.with_params(self.network), dict[str, str]
        )

    def is_reserved_ip(self, ip: IP_AddressT) -> bool:
        """Return True if the IP address is in the reserved list.

        :param ip: The IP address to check.
        :returns: True if the IP address is in the reserved list.
        """
        return ip in self.get_reserved_ips()

    def add_excluded_range(self, start: str, end: str) -> None:
        """Add an excluded range to the network.

        :param start: The start of the excluded range.
        :param end: The end of the excluded range.

        :returns: The new ExcludedRange object.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        start_ip = NetworkOrIP.parse_or_raise(start, mode="ip")
        end_ip = NetworkOrIP.parse_or_raise(end, mode="ip")
        if start_ip.version != end_ip.version:
            raise InputFailure("Start and end IP addresses must be of the same version")

        resp = MregClient().post(
            Endpoint.NetworksAddExcludedRanges.with_params(self.network),
            network=self.id,
            start_ip=str(start_ip),
            end_ip=str(end_ip),
        )
        if not resp or not resp.ok:
            raise PostError(f"Failed to create excluded range for network {self.network}")

    def remove_excluded_range(self, start: str, end: str) -> None:
        """Remove an excluded range from the network.

        :param start: The start of the excluded range.
        :param end: The end of the excluded range.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        # No need to validate IPs - if we find a match it's valid
        exrange: ExcludedRange | None = None
        for excluded_range in self.excluded_ranges:
            if str(excluded_range.start_ip) == start and str(excluded_range.end_ip) == end:
                exrange = excluded_range
                break
        else:
            raise EntityNotFound(f"Excluded range {start} - {end} not found")
        resp = MregClient().delete(
            Endpoint.NetworksRemoveExcludedRanges.with_params(self.network, exrange.id)
        )
        if not resp or not resp.ok:
            raise DeleteError(f"Failed to delete excluded range {start} - {end}")

    def set_category(self, category: str) -> Self:
        """Set the category tag of the network.

        :param category: The new category tag.
        :returns: The updated Network object.
        """
        return self.patch({"category": category})

    def set_location(self, location: str) -> Self:
        """Set the location tag of the network.

        :param category: The new category.
        :returns: The updated Network object.
        """
        return self.patch({"location": location})

    def set_description(self, description: str) -> Self:
        """Set the description of the network.

        :param description: The new description.
        :returns: The updated Network object.
        """
        return self.patch({"description": description})

    def set_dns_delegation(self, delegated: bool) -> Self:
        """Set the DNS delegation status of the network.

        :param dns_delegated: The new DNS delegation status.
        :returns: The updated Network object.
        """
        return self.patch({"dns_delegated": delegated})

    def set_frozen(self, frozen: bool) -> Self:
        """Set the frozen status of the network.

        :param frozen: The new frozen status.
        :returns: The updated Network object.
        """
        return self.patch({"frozen": frozen})

    def set_reserved(self, reserved: int) -> Self:
        """Set the number of reserved IP addresses.

        :param reserved: The new number of reserved IP addresses.
        :returns: The updated Network object.
        """
        return self.patch({"reserved": reserved})

    def set_vlan(self, vlan: int) -> Self:
        """Set the VLAN of the network.

        :param vlan: The new VLAN.
        :returns: The updated Network object.
        """
        return self.patch({"vlan": vlan})

    def set_policy(self, policy: NetworkPolicy) -> Self:
        """Set the network policy of the network.

        :param policy: The new network policy.
        :returns: The updated Network object.
        """
        return self.patch({"policy": policy.id}, validate=False)

    def set_max_communities(self, max_communities: int) -> Self:
        """Set the maximum number of communities for the network.

        :param max_communities: The new maximum number of communities.
        :returns: The updated Network object.
        """
        return self.patch({"max_communities": max_communities}, validate=False)

    def unset_policy(self) -> Self:
        """Unset the network policy of the network.

        :returns: The updated Network object.
        """
        return self.patch({"policy": None}, validate=False)

    def unset_max_communities(self) -> Self:
        """Unset the maximum number of communities for the network.

        :returns: The updated Network object.
        """
        return self.patch({"max_communities": None}, validate=False)


class NetworkPolicyAttribute(FrozenModelWithTimestamps, WithName):
    """The definition of a network policy attribute.

    See NetworkPolicyAttr for the representation of attributes in Policies.
    """

    __name_lowercase__ = True  # name is always lower case

    id: int
    name: str
    description: str

    def get_policies(self) -> list[NetworkPolicy]:
        """Get all policies using this attribute."""
        return NetworkPolicy.get_list_by_field("attributes", self.id)

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.NetworkPolicyAttributes


class Community(FrozenModelWithTimestamps, APIMixin):
    """Network community."""

    id: int
    name: str
    description: str
    network: int
    hosts: list[str] = []
    global_name: str | None = None

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.NetworkCommunity

    def endpoint_with_id(self) -> str:
        """Return the endpoint with the community ID."""
        return self.endpoint().with_params(self.network_address, self.id)

    @property
    def network_address(self) -> str:
        """Return the network object for the community."""
        return Network.get_by_field_or_raise("id", str(self.network)).network

    @property
    def hosts_endpoint(self) -> str:
        """Return the endpoint with policy and community IDs."""
        return Endpoint.NetworkCommunityHosts.with_params(self.network_address, self.id)

    def refetch(self) -> Self:
        """Refetch the community object."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(self.endpoint_with_id(), self.__class__)

    def patch(self, fields: dict[str, Any], validate: bool = True) -> Self:  # noqa: ARG002 # validate not implemented
        """Patch the community.

        :param fields: The fields to patch.
        :param validate: Whether to validate the response. (Not implemented)
        :returns: The updated Community object.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().patch(self.endpoint_with_id(), **fields)
        if not resp or not resp.ok:
            raise PatchError(f"Failed to patch community {self.name!r}")
        new_object = self.refetch()
        return new_object

    def delete(self) -> bool:
        """Delete the community."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().delete(self.endpoint_with_id())
        return resp.ok if resp else False

    def get_hosts(self) -> list[Host]:
        """Get a list of hosts in the community.

        :returns: A list of Host objects.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_typed(self.hosts_endpoint, list[Host])

    def add_host(self, host: Host, ipaddress: IP_AddressT | None = None) -> bool:
        """Add a host to the community.

        :param host: The host to add.
        :returns: True if the host was added, False otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        kwargs: QueryParams = {"id": host.id}
        if ipaddress:
            kwargs["ipaddress"] = str(ipaddress)
        resp = MregClient().post(self.hosts_endpoint, params=None, **kwargs)
        return resp.ok if resp else False

    def remove_host(self, host: Host, ipaddress: IP_AddressT | None) -> bool:
        """Remove a host from the community.

        :param host: The host to remove.
        :returns: True if the host was removed, False otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        params: QueryParams = {}
        if ipaddress:
            params["ipaddress"] = str(ipaddress)
        resp = MregClient().delete(
            Endpoint.NetworkCommunityHost.with_params(
                self.network_address,
                self.id,
                host.id,
            )
        )
        return resp.ok if resp else False


class NetworkPolicyAttributeValue(BaseModel):
    """Name and value of a network policy's attribute."""

    name: str
    value: bool


class NetworkPolicy(FrozenModelWithTimestamps, WithName):
    """Network policy used in a community."""

    __name_lowercase__ = True  # name is always lower case

    id: int
    name: str
    description: str | None = None
    attributes: list[NetworkPolicyAttributeValue] = []
    community_template_pattern: str | None = Field(
        default=None,
        validation_alias=AliasChoices("community_template_pattern", "community_mapping_prefix"),
    )

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.NetworkPolicies

    def get_attribute_or_raise(self, name: str) -> NetworkPolicyAttributeValue:
        """Get a network attribute value by name, and raise if not found.

        :param name: The name of the attribute to search for.
        :returns: The attribute if found.
        :raises EntityNotFound: If the attribute is not found.
        """
        attribute = self.get_attribute(name)
        if not attribute:
            raise EntityNotFound(f"Attribute {name!r} not found in policy.")
        return attribute

    def get_attribute(self, name: str) -> NetworkPolicyAttributeValue | None:
        """Get a attribute by name.

        :param name: The name of the attribute to search for.
        :returns: The attribute if found, None otherwise.
        """
        for attribute in self.attributes:
            if attribute.name == name:
                return attribute
        return None

    def add_attribute(self, attribute: NetworkPolicyAttribute, value: bool = True) -> None:
        """Add an attribute to the policy.

        :param attribute: The attribute to add.
        :param value: The value of the attribute.
        """
        if self.get_attribute(attribute.name):
            raise EntityAlreadyExists(f"Attribute {attribute.name!r} already exists in policy.")
        attrs = self.attributes.copy()
        attrs.append(NetworkPolicyAttributeValue(name=attribute.name, value=value))
        self._patch_attrs(attrs)

    def remove_attribute(self, attribute: str) -> None:
        """Add an attribute to the policy.

        :param attribute: The attribute to add.
        :param value: The value of the attribute.
        """
        attr = self.get_attribute_or_raise(attribute)
        attrs = self.attributes.copy()
        attrs.remove(attr)
        self._patch_attrs(attrs)

    def set_attribute_value(self, attribute: str, value: bool) -> None:
        """Add an attribute to the policy.

        :param attribute: The attribute to add.
        :param value: The value of the attribute.
        """
        # Check if attribute exists
        # NOTE: yes, we iterate over it twice here, but it's a small list
        self.get_attribute_or_raise(attribute)
        attrs = self.attributes.copy()
        for a in attrs:
            if a.name == attribute:
                a.value = value
                break
        self._patch_attrs(attrs)

    def _patch_attrs(self, attrs: list[NetworkPolicyAttributeValue]) -> None:
        """Patch the attributes of the policy.

        Sets the attributes on the model itself after a successful patch.

        :param attrs: The new attributes.
        """
        self.patch(
            {"attributes": [{"name": a.name, "value": a.value} for a in attrs]},
            validate=False,
        )
        # NOTE: can return self.refetch() here if we need to refresh the object

    def networks(self) -> list[Network]:
        """Get all networks using this policy."""
        return Network.get_list_by_field("policy", self.id)

    def create_community(self, name: str, description: str) -> Community | None:
        """Create a new community.

        :param name: The name of the community.
        :param description: The description of the community.
        :returns: The new Community object.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        client = MregClient()

        resp = client.post(
            Endpoint.NetworkCommunities.with_params(self.id),
            name=name,
            description=description,
        )
        if resp and (location := resp.headers.get("Location")):
            return client.get_typed(location, Community)
        return None


class IPAddress(FrozenModelWithTimestamps, WithHost, APIMixin):
    """Represents an IP address with associated details."""

    id: int  # noqa: A003
    macaddress: MacAddress | None = None
    ipaddress: IP_AddressT

    @field_validator("macaddress", mode="before")
    @classmethod
    def create_valid_macadress_or_none(cls, v: Any) -> MacAddress | None:
        """Create macaddress or convert empty strings to None.

        The API can return an empty string for this field, which fails to validate
        as a MAC address. Therefore, treat empty strings as None.
        """
        if v:
            return v
        return None

    @classmethod
    def get_by_ip(cls, ip: IP_AddressT) -> list[Self]:
        """Get a list of IP address objects by IP address.

        Note that the IP addresses can be duplicated across hosts,
        so this method may return multiple IP addresses.

        :param ip: The IP address to search for.
        :returns: The IP address if found, None otherwise.
        """
        return cls.get_list_by_field("ipaddress", str(ip))

    @classmethod
    def ensure_associable(cls, mac: MacAddress, force: bool) -> None:
        """Check if a MAC address is available to be associated with an IP address.

        Raise an exception if the MAC address is already associated with an IP address,
        and force is not set.

        :param mac: The MAC address to check.
        :param force: Force is active. If True, the check is skipped.
        :raises EntityAlreadyExists: If the MAC address is already associated with one IP.
        :raises MultipleEntitiesFound: If the MAC address is already associated multiple IPs.
        """
        if force:
            return

        ips = cls.get_ips_by_mac(mac)
        if not ips:
            return

        if len(ips) == 1:
            raise EntityAlreadyExists(
                f"MAC address {mac} is already associated with IP address {ips[0].ipaddress}, must force."  # noqa: E501
            )
        else:
            ips_str = ", ".join([str(ip.ipaddress) for ip in ips])
            raise MultipleEntitiesFound(
                f"MAC address {mac} is already associated with multiple IP addresses: {ips_str}, must force."  # noqa: E501
            )

    @classmethod
    def get_by_mac(cls, mac: MacAddress) -> Self | None:
        """Get the IP address objects by MAC address.

        :param mac: The MAC address to search for.
        :returns: The IP address if found, None otherwise.
        """
        try:
            return cls.get_by_field("macaddress", mac)
        except MultipleEntitiesFound as e:
            raise MultipleEntitiesFound(f"Multiple IPs found with MAC address {mac}.") from e

    @classmethod
    def get_ips_by_mac(cls, mac: MacAddress) -> list[Self]:
        """Get a list of IP addresses by MAC address.

        :param mac: The MAC address to search for.
        :returns: A list of IP addresses.
        """
        return cls.get_list_by_field("macaddress", mac)

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Ipaddresses

    def __str__(self):
        """Return the IP address as a string."""
        return self.ipaddress.__str__()

    def is_ipv4(self) -> bool:
        """Return True if the IP address is IPv4."""
        return self.ipaddress.version == 4

    def is_ipv6(self) -> bool:
        """Return True if the IP address is IPv6."""
        return self.ipaddress.version == 6

    def network(self) -> Network | None:
        """Return the network of the IP address."""
        from mreg_api.client import MregClient  # noqa: PLC0415

        try:
            return MregClient().get_typed(Endpoint.NetworksByIP.with_id(str(self.ipaddress)), Network)
        except APIError:
            return None

    def vlan(self) -> int | None:
        """Return the VLAN of the IP address."""
        if net := self.network():
            return net.vlan
        return None

    def associate_mac(self, mac: MacAddress, force: bool = False) -> IPAddress:
        """Associate a MAC address with the IP address.

        :param mac: The MAC address to associate.
        :param force: If True, force the association even if the IP address already has
                      a MAC address.

        :returns: A new IPAddress object fetched from the API with the updated MAC address.
        """
        if self.macaddress and not force:
            raise EntityAlreadyExists(
                f"IP address {self.ipaddress} already has MAC address {self.macaddress}."
            )
        return self.patch(fields={"macaddress": mac})

    def disassociate_mac(self) -> IPAddress:
        """Disassociate the MAC address from the IP address.

        Note that this does NOT validate the MAC address.

        :returns: A new IPAddress object fetched from the API with the MAC address removed.
        """
        # Model converts empty string to None so we must validate this ourselves.
        patched = self.patch(fields={"macaddress": ""}, validate=False)
        if patched.macaddress:
            raise PatchError(f"Failed to disassociate MAC address from {self.ipaddress}")
        return patched

    def __hash__(self):
        """Return a hash of the IP address."""
        return hash((self.id, self.ipaddress, self.macaddress))


class HInfo(FrozenModelWithTimestamps, WithHost, APIMixin):
    """Represents a HINFO record."""

    cpu: str
    os: str

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Hinfos


class CNAME(FrozenModelWithTimestamps, WithHost, WithZone, WithTTL, APIMixin):
    """Represents a CNAME record."""

    id: int  # noqa: A003
    name: HostName
    ttl: int | None = None

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Cnames

    @classmethod
    def get_by_name(cls, name: HostName) -> CNAME:
        """Get a CNAME record by name.

        :param name: The name to search for.
        :returns: The CNAME record if found, None otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        data = MregClient().get_item_by_key_value(Endpoint.Cnames, "name", name)
        if not data:
            raise EntityNotFound(f"CNAME record for {name} not found.")
        return CNAME.model_validate(data)

    @classmethod
    def get_by_host_and_name(cls, host: HostName | int, name: HostName) -> CNAME:
        """Get a CNAME record by host and name.

        :param host: The host to search for, either a hostname or an ID.
        :param name: The name to search for.
        :returns: The CNAME record if found, None otherwise.
        """
        target_hostname = None
        if isinstance(host, str):
            hostobj = Host.get_by_any_means(host, inform_as_cname=False)
            if not hostobj:
                raise EntityNotFound(f"Host with name {host} not found.")

            host = hostobj.id
            target_hostname = hostobj.name
        else:
            hostobj = Host.get_by_id(host)
            if not hostobj:
                raise EntityNotFound(f"Host with ID {host} not found.")
            target_hostname = hostobj.name

        results = cls.get_by_query({"host": str(host), "name": name})

        if not results or len(results) == 0:
            raise EntityNotFound(f"CNAME record for {name} not found for {target_hostname}.")

        if len(results) > 1:
            raise MultipleEntitiesFound(f"Multiple CNAME records found for {host} with {name}!")

        return results[0]


class TXT(FrozenModelWithTimestamps, WithHost, APIMixin):
    """Represents a TXT record."""

    id: int  # noqa: A003
    txt: str

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Txts


class MX(FrozenModelWithTimestamps, WithHost, APIMixin):
    """Represents a MX record."""

    id: int  # noqa: A003
    mx: str
    priority: int

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Mxs

    @classmethod
    def get_by_all(cls, host: int, mx: str, priority: int) -> MX:
        """Get an MX record by all fields.

        :param host: The host ID.
        :param mx: The MX record.
        :param priority: The priority.
        :returns: The MX record if found, None otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        data = MregClient().get_list_unique(
            Endpoint.Mxs, params={"host": str(host), "mx": mx, "priority": str(priority)}
        )
        if not data:
            raise EntityNotFound(f"MX record for {mx} not found.")
        return MX.model_validate(data)

    def has_mx_with_priority(self, mx: str, priority: int) -> bool:
        """Return True if the MX record has the given MX and priority.

        :param mx: The MX record to check.
        :param priority: The priority to check.
        :returns: True if the MX record has the given MX and priority.
        """
        return self.mx == mx and self.priority == priority


class NAPTR(FrozenModelWithTimestamps, WithHost, APIMixin):
    """Represents a NAPTR record."""

    id: int  # noqa: A003
    preference: int
    order: int
    flag: str | None = None
    service: str | None = None
    regex: str | None = None
    replacement: str

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Naptrs

    @classmethod
    def headers(cls) -> list[str]:
        """Return the headers for the NAPTR record."""
        return [
            "NAPTRs:",
            "Preference",
            "Order",
            "Flag",
            "Service",
            "Regex",
            "Replacement",
        ]


class Srv(FrozenModelWithTimestamps, WithHost, WithZone, WithTTL, APIMixin):
    """Represents a SRV record."""

    id: int  # noqa: A003
    name: str
    priority: int
    weight: int
    port: int
    ttl: int | None = None

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Srvs

    def __str__(self) -> str:
        """Return a string representation of the SRV record."""
        return self.name


class PTR_override(FrozenModelWithTimestamps, WithHost, APIMixin):
    """Represents a PTR override record."""

    id: int  # noqa: A003
    ipaddress: IP_AddressT

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.PTR_overrides


class SSHFP(FrozenModelWithTimestamps, WithHost, WithTTL, APIMixin):
    """Represents a SSHFP record."""

    id: int  # noqa: A003
    algorithm: int
    hash_type: int
    fingerprint: str
    ttl: int | None = None

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Sshfps


class BacnetID(FrozenModel, WithHost, APIMixin):
    """Represents a Bacnet ID record."""

    id: int  # noqa: A003
    hostname: str

    @classmethod
    def MAX_ID(cls) -> int:
        """Return the maximum ID for a Bacnet ID."""
        return 4194302

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.BacnetID

    @classmethod
    def get_in_range(cls, start: int, end: int) -> list[Self]:
        """Get Bacnet IDs in a range.

        :param start: The start of the range.
        :param end: The end of the range.
        :returns: List of BacnetID objects in the range.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        params: QueryParams = {"id__range": f"{start},{end}"}
        return MregClient().get_typed(Endpoint.BacnetID, list[cls], params=params)


class Location(FrozenModelWithTimestamps, WithHost, APIMixin):
    """Represents a LOC record."""

    loc: str

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Locs


class HostCommunity(FrozenModel):
    """Model for a host's community.

    Communities are associated with hosts via IP addresses.
    """

    ipaddress: int
    """ID of the IP address associated with the community"""

    community: Community


class Host(FrozenModelWithTimestamps, WithTTL, WithHistory, APIMixin):
    """Model for an individual host."""

    id: int  # noqa: A003
    name: HostName
    ipaddresses: list[IPAddress]
    cnames: list[CNAME] = []
    mxs: list[MX] = []
    txts: list[TXT] = []
    ptr_overrides: list[PTR_override] = []
    hinfo: HInfo | None = None
    loc: Location | None = None
    bacnetid: int | None = None
    contact: str
    ttl: int | None = None
    srvs: list[Srv] = []
    naptrs: list[NAPTR] = []
    sshfps: list[SSHFP] = []
    roles: list[str] = []
    hostgroups: list[str] = []
    comment: str

    communities: list[HostCommunity] = []

    # Note, we do not use WithZone here as this is optional and we resolve it differently.
    zone: int | None = None

    history_resource: ClassVar[HistoryResource] = HistoryResource.Host

    @field_validator("communities", mode="before")
    @classmethod
    def _validate_none_communities_as_empty_list(cls, v: Any) -> Any:
        """Convert None value to empty list for communities."""
        return v or []

    @field_validator("bacnetid", mode="before")
    @classmethod
    def convert_bacnetid(cls, v: Any) -> Any:
        """Use nested ID value in bacnetid value."""
        if isinstance(v, dict):
            return v.get("id")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        return None

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Hosts

    @classmethod
    def get_list_by_ip(cls, ip: IP_AddressT, inform_as_ptr: bool = True) -> list[Self]:
        """Get a list of hosts by IP address.

        :param ip: The IP address to search for.
        :param check_ptr: If True, check for PTR overrides as well.
        :returns: A list of Host objects.
        """
        hosts = cls.get_list_by_field("ipaddresses__ipaddress", str(ip))
        if not hosts:
            hosts = cls.get_list_by_field("ptr_overrides__ipaddress", str(ip))
            if hosts and inform_as_ptr:
                for host in hosts:
                    host.add_note(f"{ip} is a PTR override for {host.name}")
        return hosts

    @classmethod
    def get_list_by_ip_or_raise(cls, ip: IP_AddressT, inform_as_ptr: bool = True) -> list[Self]:
        """Get a list of hosts by IP address or raise EntityNotFound.

        :param ip: The IP address to search for.
        :returns: A list of Host objects.
        :param check_ptr: If True, check for PTR overrides as well.
        """
        hosts = cls.get_list_by_ip(ip, inform_as_ptr=inform_as_ptr)
        if not hosts:
            raise EntityNotFound(f"Host with IP address {ip} not found.")
        return hosts

    @classmethod
    def get_by_ip(cls, ip: IP_AddressT, inform_as_ptr: bool = True) -> Host | None:
        """Get a host by IP address.

        :param ip: The IP address to search for.
        :param check_ptr: If True, check for PTR overrides as well.
        :returns: The Host object if found, None otherwise.
        """
        try:
            host = cls.get_by_field("ipaddresses__ipaddress", str(ip))
            if not host:
                host = cls.get_by_field("ptr_overrides__ipaddress", str(ip))
                if host and inform_as_ptr:
                    host.add_note(f"{ip} is a PTR override for {host.name}")
            return host
        except MultipleEntitiesFound as e:
            raise MultipleEntitiesFound(f"Multiple hosts found with IP address {ip}.") from e

    @classmethod
    def get_by_ip_or_raise(cls, ip: IP_AddressT, inform_as_ptr: bool = True) -> Host:
        """Get a host by IP address or raise EntityNotFound.

        :param ip: The IP address to search for.
        :returns: The Host object if found.
        :param check_ptr: If True, check for PTR overrides as well.
        """
        host = cls.get_by_ip(ip, inform_as_ptr=inform_as_ptr)
        if not host:
            raise EntityNotFound(f"Host with IP address {ip} not found.")
        return host

    @classmethod
    def get_by_mac(cls, mac: MacAddress) -> Host | None:
        """Get a host by MAC address.

        :param ip: The MAC address to search for.
        :returns: The Host object if found, None otherwise.
        """
        return cls.get_by_field("ipaddresses__macaddress", str(mac))

    @classmethod
    def get_by_mac_or_raise(cls, mac: MacAddress) -> Host:
        """Get a host by MAC address or raise EntityNotFound.

        :param ip: The MAC address to search for.
        :returns: The Host object if found.
        """
        host = cls.get_by_mac(mac)
        if not host:
            raise EntityNotFound(f"Host with MAC address {mac} not found.")
        return host

    @classmethod
    def get_list_by_mac(cls, mac: MacAddress) -> list[Self]:
        """Get a list of host by MAC address.

        :param ip: The MAC address to search for.
        :returns: The Host object if found, None otherwise.
        """
        return cls.get_list_by_field("ipaddresses__macaddress", str(mac))

    @classmethod
    def get_list_by_mac_or_raise(cls, mac: MacAddress) -> list[Self]:
        """Get a list of hosts by MAC address or raise EntityNotFound.

        :param ip: The MAC address to search for.
        :returns: The Host object if found.
        """
        hosts = cls.get_list_by_mac(mac)
        if not hosts:
            raise EntityNotFound(f"Host with MAC address {mac} not found.")
        return hosts

    @classmethod
    def get_by_any_means_or_raise(
        cls, identifier: str | HostName, inform_as_cname: bool = True, inform_as_ptr: bool = True
    ) -> Host:
        """Get a host by the given identifier or raise EntityNotFound.

        See also `get_by_any_means`.

        :param identifier: The identifier to search for.
        :param inform_as_cname: If True, inform the user if the host is a CNAME.
        :param inform_as_ptr: If True, inform the user if the host is a PTR override.

        :raises EntityNotFound: If the host is not found.

        :returns: A Host object if the host was found.
        """
        host = cls.get_by_any_means(
            identifier, inform_as_cname=inform_as_cname, inform_as_ptr=inform_as_ptr
        )
        if not host:
            raise EntityNotFound(f"Host {identifier} not found.")
        return host

    @classmethod
    def get_by_any_means(
        cls, identifier: str, inform_as_cname: bool = True, inform_as_ptr: bool = True
    ) -> Host | None:
        """Get a host by the given identifier.

        - If the identifier is numeric, it will be treated as an ID.
        - If the identifier is an IP address, it will be treated as an IP address (v4 or v6).
        - If the identifier is a MAC address, it will be treated as a MAC address.
        - Otherwise, it will be treated as a hostname. If the hostname is a CNAME,
        the host it points to will be returned.

        To check if a returned host is a cname, one can do the following:

        ```python
        hostname = "host.example.com"
        host = get_host(hostname, ok404=True)
        if host is None:
            print("Host not found.")
        elif host.name != hostname:
            print(f"{hostname} is a CNAME pointing to {host.name}")
        else:
            print(f"{host.name} is a host.")
        ```

        Note that get_host will perform a case-insensitive search for a fully qualified version
        of the hostname, so the comparison above may fail.

        :param identifier: The identifier to search for.
        :param ok404: If True, don't raise a EntityNotFound if the host is not found.
        :param inform_as_cname: If True, inform the user if the host is a CNAME.
        :param inform_as_ptr: If True, inform the user if the host is a PTR override.

        :raises EntityNotFound: If we don't find the host and `ok404` is False.

        :returns: A Host object if the host was found, otherwise None.
        """
        if identifier.isdigit():
            return Host.get_by_id(int(identifier))

        if ip := NetworkOrIP.parse(identifier, mode="ip"):
            host = cls.get_by_ip_or_raise(ip, inform_as_ptr=inform_as_ptr)
            return host

        if mac := MacAddress.parse(identifier):
            return cls.get_by_mac_or_raise(mac)

        # Let us try to find the host by name...
        identifier = HostName.parse_or_raise(identifier)

        if host := cls.get_by_field("name", identifier):
            return host

        cname = CNAME.get_by_field("name", identifier)
        # If we found a CNAME, get the host it points to. We're not interested in the
        # CNAME itself.
        if cname is not None:
            host = Host.get_by_id(cname.host)

            if host and inform_as_cname:
                host.add_note(f"{identifier} is a CNAME for {host.name}")

        return host

    @classmethod
    def get_list_by_any_means_or_raise(
        cls, identifier: str | HostName, inform_as_cname: bool = True, inform_as_ptr: bool = True
    ) -> list[Self]:
        """Get a host by the given identifier or raise EntityNotFound.

        See also `get_by_any_means`.

        :param identifier: The identifier to search for.
        :param inform_as_cname: If True, inform the user if the host is a CNAME.
        :param inform_as_ptr: If True, inform the user if the host is a PTR override.

        :raises EntityNotFound: If the host is not found.

        :returns: A Host object if the host was found.
        """
        hosts = cls.get_list_by_any_means(
            identifier, inform_as_cname=inform_as_cname, inform_as_ptr=inform_as_ptr
        )
        if not hosts:
            raise EntityNotFound(f"Host {identifier} not found.")
        return hosts

    @classmethod
    def get_list_by_any_means(
        cls, identifier: str, inform_as_cname: bool = True, inform_as_ptr: bool = True
    ) -> list[Self]:
        """Get a host by the given identifier.

        - If the identifier is numeric, it will be treated as an ID.
        - If the identifier is an IP address, it will be treated as an IP address (v4 or v6).
        - If the identifier is a MAC address, it will be treated as a MAC address.
        - Otherwise, it will be treated as a hostname. If the hostname is a CNAME,
        the host it points to will be returned.

        To check if a returned host is a cname, one can do the following:

        ```python
        hostname = "host.example.com"
        host = get_host(hostname, ok404=True)
        if host is None:
            print("Host not found.")
        elif host.name != hostname:
            print(f"{hostname} is a CNAME pointing to {host.name}")
        else:
            print(f"{host.name} is a host.")
        ```

        Note that get_host will perform a case-insensitive search for a fully qualified version
        of the hostname, so the comparison above may fail.

        :param identifier: The identifier to search for.
        :param ok404: If True, don't raise a EntityNotFound if the host is not found.
        :param inform_as_cname: If True, inform the user if the host is a CNAME.
        :param inform_as_ptr: If True, inform the user if the host is a PTR override.

        :raises EntityNotFound: If we don't find the host and `ok404` is False.

        :returns: A Host object if the host was found, otherwise None.
        """
        if identifier.isdigit() and (host := cls.get_by_id(int(identifier))):
            return [host]

        if ip := NetworkOrIP.parse(identifier, mode="ip"):
            return cls.get_list_by_ip_or_raise(ip, inform_as_ptr=inform_as_ptr)

        if mac := MacAddress.parse(identifier):
            return cls.get_list_by_mac_or_raise(mac)

        # Let us try to find the host by name...
        identifier = HostName.parse_or_raise(identifier)

        if host := cls.get_by_field("name", identifier):
            return [host]

        # If we found a CNAME, get the host it points to. We're not interested in the
        # CNAME itself.
        if cname := CNAME.get_by_field("name", identifier):
            host = cls.get_by_id(cname.host)
            if host and inform_as_cname:
                host.add_note(f"{identifier} is a CNAME for {host.name}")
                return [host]

        return []

    def delete(self) -> bool:
        """Delete the host.

        :raises DeleteError: If the operation to delete the host fails.

        :returns: True if the host was deleted successfully, False otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        # Note, we can't use .id as the identifier here, as the host name is used
        # in the endpoint URL...
        op = MregClient().delete(Endpoint.Hosts.with_id(str(self.name)))
        if not op:
            raise DeleteError(f"Failed to delete host {self.name}, operation failed.")

        return op.status_code >= 200 and op.status_code < 300

    def rename(self, new_name: HostName) -> Host:
        """Rename the host.

        :param new_name: The new name for the host.

        :returns: A new Host object fetched from the API with the updated name.
        """
        return self.patch(fields={"name": new_name})

    def set_comment(self, comment: str) -> Host:
        """Set the comment for the host.

        :param comment: The comment to set.

        :returns: A new Host object fetched from the API with the updated comment.
        """
        return self.patch(fields={"comment": comment})

    def set_contact(self, contact: str) -> Host:
        """Set the contact for the host.

        :param contact: The contact to set. Should be a valid email, but we leave it to the
                        server to validate the data.

        :returns: A new Host object fetched from the API with the updated contact.
        """
        return self.patch(fields={"contact": contact})

    def add_ip(self, ip: IP_AddressT, mac: MacAddress | None = None) -> Host:
        """Add an IP address to the host.

        :param ip: The IP address to add. IPv4 or IPv6.

        :returns: A new Host object fetched from the API with the updated IP address.
        """
        params: QueryParams = {"ipaddress": str(ip), "host": str(self.id)}
        if mac:
            params["macaddress"] = mac

        IPAddress.create(params=params)
        return self.refetch()

    def has_ip(self, arg_ip: IP_AddressT) -> bool:
        """Check if the host has the given IP address.

        :param ip: The IP address to check for.

        :returns: True if the host has the IP address, False otherwise.
        """
        return any([ip.ipaddress == arg_ip for ip in self.ipaddresses])

    def has_ip_with_mac(self, arg_mac: MacAddress) -> IPAddress | None:
        """Check if the host has the given MAC address.

        :param mac: The MAC address to check for.

        :returns: The IP address object if found, None otherwise.
        """
        return next((ip for ip in self.ipaddresses if ip.macaddress == arg_mac), None)

    def ips_with_macaddresses(self) -> list[IPAddress]:
        """Return a list of IP addresses with a MAC address."""
        return [ip for ip in self.ipaddresses if ip.macaddress]

    def get_associatable_ip(self) -> IPAddress:
        """Get the only IP address of the host that can have a MAC associated to it.

        Constraints:
          - If the host only has one IP, return that IP.
          - If the host has two IPs of different versions and they are on the same VLAN,
            return the IPv4 address.

        :raises EntityNotFound: If the host has no IP addresses.
        :raises EntityOwnershipMismatch: If the host multiple IPs and the constriants aren't met.

        :returns: An IP address that can be associated with the host.
        """
        if len(self.ipaddresses) == 0:
            raise EntityNotFound(f"Host {self} has no IP addresses.")

        if len(self.ipaddresses) == 1:
            return self.ipaddresses[0]

        ipv4s = self.ipv4_addresses()
        ipv6s = self.ipv6_addresses()

        if len(ipv4s) == 1 and len(ipv6s) == 1:
            ipv4_address = ipv4s[0]
            ipv6_address = ipv6s[0]
            ipv4_network = ipv4_address.network()
            ipv6_network = ipv6_address.network()

            if ipv4_network and ipv6_network:
                if ipv4_network.vlan == ipv6_network.vlan:
                    return ipv4_address
            elif ipv4_network:  # only IPv4 is in mreg
                logger.warning(
                    "Host '%s' has IPv6 address not in MREG: %s",
                    self.name,
                    str(ipv6_address.ipaddress),
                )
                return ipv4_address
            elif ipv6_network:  # only IPv6 is in mreg
                logger.warning(
                    "Host '%s' has IPv4 address not in MREG: %s",
                    self.name,
                    str(ipv4_address.ipaddress),
                )
                return ipv6_address

        ips = ", ".join(str(ip.ipaddress) for ip in self.ipaddresses)
        raise EntityOwnershipMismatch(
            f"Host {self} has multiple IPs, cannot determine which one to use: {ips}."
        )

    def has_ptr_override(self, arg_ip: IP_AddressT) -> bool:
        """Check if the host has a PTR override for the given IP address.

        :param ip: The IP address to check for.

        :returns: True if the host has a PTR override for the IP address, False otherwise.
        """
        return any([ptr.ipaddress == arg_ip for ptr in self.ptr_overrides])

    def has_txt(self, arg_txt: str) -> bool:
        """Check if the host has the given TXT record.

        :param txt: The TXT record to check for.

        :returns: True if the host has the TXT record, False otherwise.
        """
        return any([txt.txt == arg_txt for txt in self.txts])

    def get_ip(self, arg_ip: IP_AddressT) -> IPAddress | None:
        """Get the IP address object for the given IP address.

        :param ip: The IP address to search for.

        :returns: The IP address object if found, None otherwise.
        """
        return next((ip for ip in self.ipaddresses if ip.ipaddress == arg_ip), None)

    def get_ip_by_id(self, ip_id: int) -> IPAddress | None:
        """Get the IP address object for the given ID.

        :param ip_id: The ID to search for.

        :returns: The IP address object if found, None otherwise.
        """
        return next((ip for ip in self.ipaddresses if ip.id == ip_id), None)

    def get_ptr_override(self, ip: IP_AddressT) -> PTR_override | None:
        """Get the PTR override for the given IP address.

        :param ip: The IP address to search for.

        :returns: The PTR override object if found, None otherwise.
        """
        return next((ptr for ptr in self.ptr_overrides if ptr.ipaddress == ip), None)

    def ipv4_addresses(self) -> list[IPAddress]:
        """Return a list of IPv4 addresses."""
        return [ip for ip in self.ipaddresses if ip.is_ipv4()]

    def ipv6_addresses(self) -> list[IPAddress]:
        """Return a list of IPv6 addresses."""
        return [ip for ip in self.ipaddresses if ip.is_ipv6()]

    def associate_mac_to_ip(self, mac: MacAddress, ip: IP_AddressT | str, force: bool = False) -> Host:
        """Associate a MAC address to an IP address.

        :param mac: The MAC address to associate.
        :param ip: The IP address to associate.

        :returns: A new Host object fetched from the API after updating the IP address.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        if isinstance(ip, str):
            ip = NetworkOrIP.parse_or_raise(ip, mode="ip")

        params: QueryParams = {
            "macaddress": mac,
            "ordering": "ipaddress",
        }

        ipadresses = MregClient().get_typed(Endpoint.Ipaddresses, list[IPAddress], params=params)

        if ip in [ip.ipaddress for ip in ipadresses]:
            raise EntityAlreadyExists(f"IP address {ip} already has MAC address {mac} associated.")

        if len(ipadresses) and not force:
            raise EntityOwnershipMismatch(
                f"mac {mac} already in use by: {', '.join(str(ip.ipaddress) for ip in ipadresses)}. Use force to add {ip} -> {mac} as well."  # noqa: E501
            )

        ip_found_in_host = False
        for myip in self.ipaddresses:
            if myip.ipaddress == ip:
                myip.associate_mac(mac, force=force)
                ip_found_in_host = True

        if not ip_found_in_host:
            raise EntityNotFound(f"IP address {ip} not found in host {self.name}.")

        return self.refetch()

    def disassociate_mac_from_ip(self, ip: IP_AddressT | str) -> Host:
        """Disassociate a MAC address from an IP address.

        Note: This method blindly disassociates the current MAC address
              from associated the given IP address.

        :param ip: The IP address to disassociate.

        :returns: A new Host object fetched from the API after updating the IP address.
        """
        if isinstance(ip, str):
            ip = NetworkOrIP.parse_or_raise(ip, mode="ip")

        ip_found_in_host = False
        for myip in self.ipaddresses:
            if myip.ipaddress == ip:
                myip.disassociate_mac()
                ip_found_in_host = True

        if not ip_found_in_host:
            raise EntityNotFound(f"IP address {ip} not found in host {self.name}.")

        return self.refetch()

    def get_community_or_raise(self, name: str, ip: IPAddress | None) -> Community:
        """Get a community by name, and raise if not found.

        :param name: The name of the community to search for.
        :param ip: The IP address associated with the community.
        :returns: The community if found.
        :raises EntityNotFound: If the community is not found.
        """
        community = self.get_community(name, ip)
        if not community:
            msg = f"Community {name!r}"
            if ip:
                msg += f" for IP address {ip}"
            raise EntityNotFound(f"{msg} not found.")
        return community

    def get_community(self, name: str, ip: IPAddress | None = None) -> Community | None:
        """Get a community by name.

        :param name: The name of the community to search for.
        :param ip: The IP address associated with the community.
        :returns: The community if found, None otherwise.
        """
        for community in self.communities:
            if community.community.name != name:
                continue
            if ip and community.ipaddress != ip.id:
                continue
            return community.community
        return None

    def networks(self) -> dict[Network, list[IPAddress]]:
        """Return a dict of unique networks and a list of associated IP addresses for the host.

        Does not return networks that are not registered in MREG.

        :returns: A dictionary of networks and the associated IP addresses.
        """
        ret_dict: dict[Network, list[IPAddress]] = {}

        for ip in self.ipaddresses:
            network = ip.network()
            if not network:
                # If network is not in MREG, we create a placeholder network
                network = Network.dummy_network_from_ip(ip)

            if network not in ret_dict:
                ret_dict[network] = []

            ret_dict[network].append(ip)

        return ret_dict

    def vlans(self) -> dict[int, list[IPAddress]]:
        """Return a dict of unique VLANs ID and a list of associated IP addresses for the host.

        IP addresses without a VLAN are assigned to VLAN 0.

        Note that this method will call self.networks() to determine the networks associated with
        the IP addresses. If you wish to report more details about what networks the IP addresses
        (ie, beyond simply the VLAN ID), use self.networks() and parse the VLAN from the network
        manually.

        :returns: A dictionary of VLAN ID and the associated IP addresses.
        """
        ret_dict: dict[int, list[IPAddress]] = {}

        for network, ips in self.networks().items():
            vlan = network.vlan or 0
            if vlan not in ret_dict:
                ret_dict[vlan] = []

            ret_dict[vlan].extend(ips)

        return ret_dict

    def resolve_zone(
        self, accept_delegation: bool = False, validate_zone_resolution: bool = False
    ) -> Zone | Delegation | None:
        """Return the zone for the host.

        :param accept_delegation: If True, accept delegation and return a Delegation object if the
                zone of the host is delegated. Otherwise raise EntityOwnershipMismatch.
        :param validate_zone_resolution: If True, validate that the resolved zone matches the
                expected zone ID. Fail with ValidationFailure if it does not.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        if not self.zone:
            return None

        data = MregClient().get(Endpoint.ForwardZoneForHost.with_id(str(self.name)))
        data_as_dict = data.json()

        if data_as_dict["zone"]:
            zone = ForwardZone.model_validate(data_as_dict["zone"])
            if validate_zone_resolution and zone.id != self.zone:
                raise MregValidationError(f"Expected zone ID {self.zone} but resolved as {zone.id}.")
            return zone

        if data_as_dict["delegation"]:
            if not accept_delegation:
                raise EntityOwnershipMismatch(
                    f"Host {self.name} is delegated to zone {data_as_dict['delegation']['name']}."
                )
            return ForwardZoneDelegation.model_validate(data_as_dict["delegation"])

        raise EntityNotFound(f"Failed to resolve zone for host {self.name}.")

    # This would be greatly improved by having a proper error returned to avoid the need for
    # manually calling networks() or vlans() to determine the issue. One option is to use
    # a custom exception, or to return a tuple of (bool, str) where the str is the error message.
    def all_ips_on_same_vlan(self) -> bool:
        """Return True if all IP addresses are on the same VLAN.

        - If there are no IP addresses, return True.
        - If there is only one IP address, return True.
        - If there are multiple IP addresses and they are all on the same VLAN, return True.

        Note that this method will call self.vlans() to determine if all IP addresses are on the
        same VLAN, which in turn calls self.networks() to determine the networks associated with
        the IP addresses.

        If you wish to report more details about what VLANs the IP addresses are on, use
        self.vlans() or self.networks().

        :returns: True if all IP addresses are on the same VLAN, False otherwise.
        """
        vlans = self.vlans()
        if not vlans:
            return True

        if len(vlans) == 1:
            return True

        return False

    def get_roles(self) -> list[Role]:
        """List all roles for the host."""
        return Role.get_list_by_field("hosts", self.id)

    def get_hostgroups(self, traverse: bool = False) -> list[HostGroup]:
        """Return all hostgroups for the host.

        :param traverse: If True, traverse the parent groups and include them in the list.

        :returns: A list of HostGroup objects sorted by name.
        """
        groups: list[HostGroup] = []
        direct = HostGroup.get_list_by_field("hosts", self.id)
        groups.extend(direct)

        if traverse:
            for group in direct:
                groups.extend(group.get_all_parents())

        return sorted(groups, key=lambda group: group.name)

    def bacnet(self) -> BacnetID | None:
        """Return the BacnetID for the host."""
        if not self.bacnetid:
            return None

        return BacnetID.get_by_id(self.bacnetid)

    def has_mx_with_priority(self, mx_arg: str, priority: int) -> MX | None:
        """Check if the host has an MX record.

        :param mx: The MX record to check for.
        :param priority: The priority of the MX record.

        :returns: True if the host has the MX record, False otherwise.
        """
        return next((mx for mx in self.mxs if mx.has_mx_with_priority(mx_arg, priority)), None)

    def __str__(self) -> str:
        """Return the host name as a string."""
        return self.name

    def __hash__(self):
        """Return a hash of the host."""
        return hash((self.id, self.name))


class HostList2(list[T]):
    """List of hosts that may be CNAMEs or PTRs."""

    def __init__(self, iterable: Iterable[T], is_cname: bool = False, is_ptr: bool = False) -> None:
        """Initialize a list of hosts.

        Args:
            iterable (Iterable[T]): The iterable to initialize with.
            is_cname (bool, optional): Hosts are cnames. Defaults to False.
            is_ptr (bool, optional): Hosts are ptrs. Defaults to False.
        """
        super().__init__(iterable)
        self.is_ptr = is_ptr
        self.is_cname = is_cname


class HostList(FrozenModel):
    """Model for a list of hosts.

    This is the endpoint at /api/v1/hosts/.
    """

    results: list[Host]
    is_ptr: bool = False
    is_cname: bool = False

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Hosts

    @classmethod
    def get(cls, params: QueryParams | None = None) -> HostList:
        """Get a list of hosts.

        :param params: Optional parameters to pass to the API.

        :returns: A HostList object.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        if params is None:
            params = {}

        if "ordering" not in params:
            params["ordering"] = "name"

        hosts = MregClient().get_typed(cls.endpoint(), list[Host], params=params)
        return cls(results=hosts)

    @classmethod
    def get_by_ip(cls, ip: IP_AddressT) -> HostList:
        """Get a list of hosts by IP address.

        :param ip: The IP address to search for.

        :returns: A HostList object.
        """
        return cls.get(params={"ipaddresses__ipaddress": str(ip), "ordering": "name"})

    def __len__(self):
        """Return the number of results."""
        return len(self.results)

    def __getitem__(self, key: int) -> Host:
        """Get a result by index."""
        return self.results[key]

    def __str__(self):
        """Return a string representation of the results."""
        return str(self.results)

    def __repr__(self):
        """Return a string representation of the results."""
        return repr(self.results)

    def hostnames(self) -> list[str]:
        """Return a list of hostnames."""
        return [host.name for host in self.results]

    def count(self):
        """Return the number of results."""
        return len(self.results)


class HostGroup(FrozenModelWithTimestamps, WithName, WithHistory, APIMixin):
    """Model for a hostgroup."""

    id: int  # noqa: A003
    name: str
    description: str | None = None
    parent: NameList
    groups: NameList
    hosts: NameList
    owners: NameList

    history_resource: ClassVar[HistoryResource] = HistoryResource.Group

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HostGroups

    def set_description(self, description: str) -> Self:
        """Set the description for the hostgroup.

        :param description: The description to set.

        :returns: A new HostGroup object fetched from the API with the updated description.
        """
        return self.patch(fields={"description": description})

    def has_group(self, groupname: str) -> bool:
        """Check if the hostgroup has the given group.

        :param groupname: The group to check for.

        :returns: True if the hostgroup has the group, False otherwise.
        """
        return groupname in self.groups

    def add_group(self, groupname: str) -> Self:
        """Add a group to the hostgroup.

        :param group: The group to add.

        :returns: A new HostGroup object fetched from the API with the updated groups.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().post(Endpoint.HostGroupsAddHostGroups.with_params(self.name), name=groupname)
        if resp and resp.ok:
            return self.refetch()
        else:
            raise PostError(f"Failed to add group {groupname} to hostgroup {self.name}.")

    def remove_group(self, groupname: str) -> Self:
        """Remove a group from the hostgroup.

        :param group: The group to remove.

        :returns: A new HostGroup object fetched from the API with the updated groups.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().delete(Endpoint.HostGroupsRemoveHostGroups.with_params(self.name, groupname))
        if resp and resp.ok:
            return self.refetch()
        else:
            raise DeleteError(f"Failed to remove group {groupname} from hostgroup {self.name}.")

    def has_host(self, hostname: str) -> bool:
        """Check if the hostgroup has the given host.

        :param hostname: The host to check for.

        :returns: True if the hostgroup has the host, False otherwise.
        """
        return hostname in self.hosts

    def add_host(self, hostname: str) -> Self:
        """Add a host to the hostgroup.

        :param hostname: The host to add.

        :returns: A new HostGroup object fetched from the API with the updated hosts.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().post(Endpoint.HostGroupsAddHosts.with_params(self.name), name=hostname)
        if resp and resp.ok:
            return self.refetch()
        else:
            raise PostError(f"Failed to add host {hostname} to hostgroup {self.name}.")

    def remove_host(self, hostname: str) -> Self:
        """Remove a host from the hostgroup.

        :param hostname: The host to remove.

        :returns: A new HostGroup object fetched from the API with the updated hosts.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().delete(Endpoint.HostGroupsRemoveHosts.with_params(self.name, hostname))
        if resp and resp.ok:
            return self.refetch()
        else:
            raise DeleteError(f"Failed to remove host {hostname} from hostgroup {self.name}.")

    def has_owner(self, ownername: str) -> bool:
        """Check if the hostgroup has the given owner.

        :param ownername: The owner to check for.

        :returns: True if the hostgroup has the owner, False otherwise.
        """
        return ownername in self.owners

    def add_owner(self, ownername: str) -> Self:
        """Add an owner to the hostgroup.

        :param ownername: The owner to add.

        :returns: A new HostGroup object fetched from the API with the updated owners.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().post(Endpoint.HostGroupsAddOwner.with_params(self.name), name=ownername)
        if resp and resp.ok:
            return self.refetch()
        else:
            raise PostError(f"Failed to add owner {ownername} to hostgroup {self.name}.")

    def remove_owner(self, ownername: str) -> Self:
        """Remove an owner from the hostgroup.

        :param ownername: The owner to remove.

        :returns: A new HostGroup object fetched from the API with the updated owners.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        resp = MregClient().delete(Endpoint.HostGroupsRemoveOwner.with_params(self.name, ownername))
        if resp and resp.ok:
            return self.refetch()
        else:
            raise DeleteError(f"Failed to remove owner {ownername} from hostgroup {self.name}.")

    def get_all_parents(self) -> list[HostGroup]:
        """Return a list of all parent groups."""
        parents: list[HostGroup] = []
        for parent in self.parent:
            pobj = HostGroup.get_by_field("name", parent)
            if pobj:
                parents.append(pobj)
                parents.extend(pobj.get_all_parents())

        return parents


### Meta models


class UserDjangoStatus(BaseModel):
    """Model for Django status in the user response."""

    superuser: bool
    staff: bool
    active: bool


class UserMregStatus(BaseModel):
    """Model for Mreg status in the user response."""

    superuser: bool
    admin: bool
    group_admin: bool
    network_admin: bool
    hostpolicy_admin: bool
    dns_wildcard_admin: bool
    underscore_admin: bool


class UserPermission(BaseModel):
    """Model for permissions in the user response."""

    group: str
    range: str
    regex: str
    labels: list[str]

    # NOTE: _needs_ to be a computed field in order to use it in
    # OutputManager.add_formatted_table, since we dump the model to a dict
    # inside that method.
    @computed_field
    @property
    def labels_str(self) -> str:
        """Return the labels as a string."""
        return ", ".join(self.labels)


FetchT_co = TypeVar("FetchT_co", covariant=True)


class Fetchable(Protocol[FetchT_co]):
    """Interface for classes that implement a `fetch()` method.

    This protocol is designed for endpoints that don't operate on resource IDs,
    unlike standard APIMixin classes which implement `get()` with an ID parameter.

    The `fetch()` method sends a GET request to the class's endpoint without
    requiring an ID, typically for singleton resources, status endpoints, or
    metadata endpoints that return a single object or status response rather
    than a collection.

    Since the response structure and object construction vary by endpoint type,
    there is no general `fetch()` implementation provided by APIMixin. Each
    implementing class must define its own `fetch()` method to handle the
    specific response format and construct the appropriate object.
    """

    def fetch(self, *, ignore_errors: bool = True) -> FetchT_co: ...  # noqa: D102


class ServerVersion(BaseModel):
    """Model for server version metadata."""

    version: str

    @classmethod
    def endpoint(cls) -> str:
        """Return the endpoint for the class."""
        return Endpoint.MetaVersion

    @classmethod
    def fetch(cls, *, ignore_errors: bool = True) -> ServerVersion:
        """Fetch the server version from the endpoint.

        :param ignore_errors: Whether to ignore errors.
        :raises ValidationError: If the response data is invalid and ignore_errors is False.
        :raises requests.RequestException: If the HTTP request fails and ignore_errors is False.
        :returns: An instance of ServerVersion with the fetched data.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        try:
            response = MregClient().get(cls.endpoint())
            return cls.model_validate(response.json())
        except Exception as e:
            if ignore_errors:
                return cls(version="Unknown")
            raise e


class Library(BaseModel):
    """Model for library metadata."""

    name: str
    version: str


class ServerLibraries(BaseModel):
    """Model for server libraries metadata."""

    libraries: list[Library]

    @classmethod
    def endpoint(cls) -> str:
        """Return the endpoint for the class."""
        return Endpoint.MetaLibraries

    @classmethod
    def fetch(cls, *, ignore_errors: bool = True) -> ServerLibraries:
        """Fetch the server libraries from the endpoint.

        :param ignore_errors: Whether to ignore errors.
        :raises ValidationError: If the response data is invalid and ignore_errors is False.
        :raises requests.RequestException: If the HTTP request fails and ignore_errors is False.
        :returns: An instance of ServerLibraries with the fetched data.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        try:
            response = MregClient().get(cls.endpoint())
            libraries: list[Library] = []

            for name, version in response.json().items():
                libraries.append(Library(name=name, version=version))
            return cls(libraries=libraries)
        except Exception as e:
            if ignore_errors:
                return cls(libraries=[])
            raise e


class TokenInfo(BaseModel):
    """Model for token information."""

    is_valid: bool
    created: str
    expire: str
    last_used: str | None = None
    lifespan: str


class UserInfo(BaseModel):
    """Model for the user information."""

    username: str
    last_login: str | None = None
    token: TokenInfo | None = None
    django_status: UserDjangoStatus
    mreg_status: UserMregStatus
    groups: list[str]
    permissions: list[UserPermission]

    @classmethod
    def endpoint(cls) -> str:
        """Return the endpoint for the class."""
        return Endpoint.MetaUser

    @classmethod
    def fetch(cls, *, ignore_errors: bool = True, user: str | None = None) -> UserInfo:
        """Fetch the user information from the endpoint.

        :param ignore_errors: Whether to ignore errors.
        :param user: The username to fetch information for. If None, fetch information for the
                              current user.

        :raises ValidationError: If the response data is invalid and ignore_errors is False.
        :raises requests.RequestException: If the HTTP request fails and ignore_errors is False.
        :returns: An instance of UserInfo with the fetched data.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        try:
            endpoint = cls.endpoint()
            if user:
                endpoint = f"{endpoint}?username={user}"

            response = MregClient().get(endpoint)
            return cls.model_validate(response.json())
        except Exception as e:
            if ignore_errors:
                return cls(
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
            raise e


class LDAPHealth(BaseModel, APIMixin):
    """Model for LDAP health endpoint."""

    status: str

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HealthLDAP

    @classmethod
    def fetch(cls, *, ignore_errors: bool = True) -> Self:
        """Fetch the LDAP status from the endpoint.

        :param ignore_errors: Ignore non-503 errors. 503 means LDAP is down,
            and should not be treated as an error in the traditional sense.
        :raises requests.APIError: If the response code is not 200 or 503.
        :returns: An instance of LDAPStatus.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        try:
            MregClient().get(cls.endpoint())
            return cls(status="OK")
        except APIError as e:
            if ignore_errors:
                logger.error("Failed to fetch LDAP health: %s", e)
                if e.response.status_code == 503:
                    return cls(status="Down")
                else:
                    return cls(status="Unknown")
            raise e


class HeartbeatHealth(BaseModel, APIMixin):
    """Model for heartbeat health endpoint."""

    uptime: int
    start_time: int

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HealthHeartbeat

    def as_str(self) -> str:
        """Return the uptime as a string."""
        # If we got a negative datetime, we weren't able to fetch the heartbeat
        return str(timedelta(seconds=self.uptime)) if self.uptime > 0 else "Unknown"

    @classmethod
    def fetch(cls, *, ignore_errors: bool = True) -> Self:
        """Fetch the heartbeat status from the endpoint.

        :param ignore_errors: Ignore HTTP errors and return dummy object with negative uptime.
        :returns: An instance of HeartbeatHealth with the fetched data.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        try:
            result = MregClient().get(cls.endpoint())
            return cls.model_validate_json(result.text)
        except Exception as e:
            if ignore_errors:
                logger.error("Failed to fetch heartbeat: %s", e)
                return cls(uptime=-1, start_time=0)
            raise e


class HealthInfo(BaseModel):
    """Combined information from all health endpoints."""

    heartbeat: HeartbeatHealth
    ldap: LDAPHealth

    @classmethod
    def fetch(cls) -> HealthInfo:
        """Fetch the health information from the endpoint.

        :returns: An instance of HealthInfo with the fetched data.
        """
        return cls(
            heartbeat=HeartbeatHealth.fetch(),
            ldap=LDAPHealth.fetch(),
        )
