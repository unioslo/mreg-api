"""Pydantic models for API resources."""

from __future__ import annotations

import ipaddress
import logging
import warnings
from abc import ABC
from collections.abc import Callable
from datetime import date
from datetime import datetime
from datetime import timedelta
from functools import cached_property
from typing import Any
from typing import Literal
from typing import Self
from typing import cast
from typing import overload

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import Field
from pydantic import RootModel
from pydantic import ValidationError as PydanticValidationError
from pydantic import computed_field
from pydantic import field_validator
from pydantic import model_validator

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import InputFailure
from mreg_api.exceptions import InvalidIPAddress
from mreg_api.exceptions import InvalidIPv4Address
from mreg_api.exceptions import InvalidIPv6Address
from mreg_api.exceptions import InvalidNetwork
from mreg_api.exceptions import IPNetworkError
from mreg_api.models.abstracts import MregModel
from mreg_api.models.abstracts import MregModelWithTimestamps
from mreg_api.models.fields import HostNameField
from mreg_api.models.fields import MacAddress
from mreg_api.models.fields import NameList
from mreg_api.types import IP_AddressT
from mreg_api.types import IP_NetworkT

logger = logging.getLogger(__name__)

IPNetMode = Literal["ipv4", "ipv6", "ip", "network", "networkv4", "networkv6"]


class NetworkOrIP(BaseModel):
    """A model for either a network or an IP address."""

    ip_or_network: IP_AddressT | IP_NetworkT

    @classmethod
    def validate(cls, value: str | IP_AddressT | IP_NetworkT | Self) -> Self:
        """Create a NetworkOrIP model instance from a value.

        This constructor validates and wraps the IP/network in the model.

        Args:
            value: The value to convert (string or IP object)

        Returns:
            A NetworkOrIP model instance

        Raises:
            InputFailure: If validation fails
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

        Args:
            value: The value to parse.
            mode: The mode to validate the input as.

        Returns:
            The parsed value as an IP address or network.

        Raises:
            IPNetworkError: If the value is not an IP address or network.
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

        Args:
            value: The value to parse.
            mode: The mode to validate the input as.

        Returns:
            The parsed value as an IP address or network, or None.
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


class NameServer(MregModelWithTimestamps):
    """Model for representing a nameserver within a DNS zone."""

    id: int  # noqa: A003
    name: str
    ttl: int | None


class Permission(MregModelWithTimestamps):
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


def is_reverse_zone_name(name: str) -> bool:
    """Determine if a zone is a reverse zone by its name.

    Args:
        name: The name of the zone.

    Returns:
        True if the zone is a reverse zone.
    """
    return name.endswith(".arpa")


class Zone(MregModelWithTimestamps):
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

    def is_delegated(self) -> bool:
        """Return True if the zone is delegated."""
        return False

    def is_reverse(self) -> bool:
        """Return True if the zone is a reverse zone."""
        return is_reverse_zone_name(self.name)

    def ensure_delegation_in_zone(self, name: str) -> None:
        """Ensure a delegation is in the zone.

        Args:
            name: The name of the delegation to check.

        Returns:
            True if the delegation is in the zone.
        """
        if not name.endswith(f".{self.name}"):
            raise InputFailure(f"Delegation '{name}' is not in '{self.name}'")

    @classmethod
    def type_by_name(cls, name: str) -> type[ForwardZone | ReverseZone]:
        """Determine the zone type based on the name.

        Args:
            name: The name of the zone.

        Returns:
            The zone type.
        """
        if is_reverse_zone_name(name):
            return ReverseZone
        return ForwardZone


class ForwardZone(Zone):
    """A forward zone."""


class ReverseZone(Zone):
    """A reverse zone."""


class Delegation(MregModelWithTimestamps):
    """A delegated zone."""

    id: int  # noqa: A003
    nameservers: list[NameServer]
    name: str
    comment: str | None = None
    zone: int | None = None

    # NOTE: Delegations are created through zone objects!
    # Call Zone.create_delegation() on an existing zone to create one.
    # We do not implement APIMixin here, since we cannot determine
    # the path and type of a delegation to create without information
    # about the zone in which to create it.

    @classmethod
    def endpoint_with_name(cls, zone: Zone, name: str) -> str:
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


class ForwardZoneDelegation(Delegation):
    """A forward zone delegation."""


class ReverseZoneDelegation(Delegation):
    """A reverse zone delegation."""

    @classmethod
    def is_reverse(cls) -> bool:
        """Return True if the delegation is for a reverse zone."""
        return True


class ZoneFile(RootModel[str]):
    """Zone file model.

    Uses a RootModel-based approach that does not inherit from
    MregModel and APIMixin, because the endpoint itself just
    returns text blobs.
    """

    root: str

    def __str__(self) -> str:
        """Return the zone file contents."""
        return self.root


class HostPolicy(MregModel, ABC):
    """Base model for Host Policy objects.

    Note:
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

        Args:
            value: The input value - should be a datetime string.

        Returns:
            The input value converted to a datetime object.
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

    @computed_field
    @property
    def created_at(self) -> datetime:
        """Creation time."""
        return self.created_at_tz_naive.replace(tzinfo=self.updated_at.tzinfo)


class Role(HostPolicy):
    """Model for a role."""

    id: int  # noqa: A003
    hosts: NameList
    atoms: NameList
    labels: list[int]

    def __hash__(self) -> int:
        """Hash the role by ID and name."""
        return hash(str(self.id) + self.name)


class Atom(HostPolicy):
    """Model for an atom."""

    id: int  # noqa: A003
    roles: NameList


class Label(MregModelWithTimestamps):
    """Model for a label."""

    id: int  # noqa: A003
    name: str
    description: str


class ExcludedRange(MregModelWithTimestamps):
    """Model for an excluded IP range for a network."""

    id: int  # noqa: A003
    network: int
    start_ip: IP_AddressT
    end_ip: IP_AddressT

    def excluded_ips(self) -> int:
        """Return the number of IP addresses in the excluded range."""
        return int(self.end_ip) - int(self.start_ip) + 1


class Network(MregModelWithTimestamps):
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
        except IPNetworkError as e:
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

    def overlaps(self, other: Network | str | IP_NetworkT) -> bool:
        """Check if the network overlaps with another network."""
        # Network -> str -> ipaddress.IPv{4,6}Network
        if isinstance(other, Network):
            other = other.network
        if isinstance(other, str):
            other = NetworkOrIP.parse_or_raise(other, mode="network")
        return self.ip_network.overlaps(other)


class NetworkPolicyAttribute(MregModelWithTimestamps):
    """The definition of a network policy attribute.

    See NetworkPolicyAttr for the representation of attributes in Policies.
    """

    __name_lowercase__ = True  # name is always lower case

    id: int
    name: str
    description: str


class Community(MregModelWithTimestamps):
    """Network community."""

    id: int
    name: str
    description: str
    network: int
    hosts: list[str] = []
    global_name: str | None = None


class NetworkPolicyAttributeValue(BaseModel):
    """Name and value of a network policy's attribute."""

    name: str
    value: bool


class NetworkPolicy(MregModelWithTimestamps):
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

    def get_attribute_or_raise(self, name: str) -> NetworkPolicyAttributeValue:
        """Get a network attribute value by name, and raise if not found.

        Args:
            name: The name of the attribute to search for.

        Returns:
            The attribute if found.

        Raises:
            EntityNotFound: If the attribute is not found.
        """
        attribute = self.get_attribute(name)
        if not attribute:
            raise EntityNotFound(f"Attribute {name!r} not found in policy.")
        return attribute

    def get_attribute(self, name: str) -> NetworkPolicyAttributeValue | None:
        """Get a attribute by name.

        Args:
            name: The name of the attribute to search for.

        Returns:
            The attribute if found, None otherwise.
        """
        for attribute in self.attributes:
            if attribute.name == name:
                return attribute
        return None


class IPAddress(MregModelWithTimestamps):
    """Represents an IP address with associated details."""

    id: int  # noqa: A003
    macaddress: MacAddress | None = None
    ipaddress: IP_AddressT
    host: int

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

    def __str__(self) -> str:
        """Return the IP address as a string."""
        return self.ipaddress.__str__()

    def is_ipv4(self) -> bool:
        """Return True if the IP address is IPv4."""
        return self.ipaddress.version == 4

    def is_ipv6(self) -> bool:
        """Return True if the IP address is IPv6."""
        return self.ipaddress.version == 6

    def __hash__(self):
        """Return a hash of the IP address."""
        return hash((self.id, self.ipaddress, self.macaddress))


class HInfo(MregModelWithTimestamps):
    """Represents a HINFO record."""

    cpu: str
    os: str
    host: int


class CNAME(MregModelWithTimestamps):
    """Represents a CNAME record."""

    id: int  # noqa: A003
    name: HostNameField
    host: int
    ttl: int | None = None
    zone: int | None = None


class TXT(MregModelWithTimestamps):
    """Represents a TXT record."""

    id: int  # noqa: A003
    txt: str
    host: int


class MX(MregModelWithTimestamps):
    """Represents a MX record."""

    id: int  # noqa: A003
    mx: str
    priority: int
    host: int

    def has_mx_with_priority(self, mx: str, priority: int) -> bool:
        """Return True if the MX record has the given MX and priority.

        Args:
            mx: The MX record to check.
            priority: The priority to check.

        Returns:
            True if the MX record has the given MX and priority.
        """
        return self.mx == mx and self.priority == priority


class NAPTR(MregModelWithTimestamps):
    """Represents a NAPTR record."""

    id: int  # noqa: A003
    preference: int
    order: int
    flag: str = ""
    service: str = ""
    regex: str = ""
    replacement: str
    host: int

    @classmethod
    def headers(cls) -> list[str]:  # TODO: remove. Move to mreg-cli
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


class Srv(MregModelWithTimestamps):
    """Represents a SRV record."""

    id: int  # noqa: A003
    name: str
    priority: int
    weight: int
    port: int
    host: int
    ttl: int | None = None
    zone: int | None = None

    def __str__(self) -> str:
        """Return a string representation of the SRV record."""
        return self.name


class PTR_override(MregModelWithTimestamps):
    """Represents a PTR override record."""

    id: int  # noqa: A003
    ipaddress: IP_AddressT
    host: int


class SSHFP(MregModelWithTimestamps):
    """Represents a SSHFP record."""

    id: int  # noqa: A003
    algorithm: int
    hash_type: int
    fingerprint: str
    host: int
    ttl: int | None = None


class BacnetID(MregModel):
    """Represents a Bacnet ID record."""

    id: int  # noqa: A003
    hostname: str
    host: int

    @classmethod
    def MAX_ID(cls) -> int:
        """Return the maximum ID for a Bacnet ID."""
        return 4194302


class Location(MregModelWithTimestamps):
    """Represents a LOC record."""

    loc: str
    host: int


class HostCommunity(MregModel):
    """Model for a host's community.

    Communities are associated with hosts via IP addresses.
    """

    ipaddress: int
    """ID of the IP address associated with the community"""

    community: Community


class HostContactModification(MregModel):
    """Model for host contact email modifications."""

    added: list[str] = Field(default_factory=list)
    already_exists: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    not_found: list[str] = Field(default_factory=list)


class ContactEmail(MregModelWithTimestamps):
    """Model for a host's contact email."""

    id: int
    email: str


class Host(MregModelWithTimestamps):
    """Model for an individual host."""

    id: int  # noqa: A003
    name: HostNameField
    ipaddresses: list[IPAddress]
    cnames: list[CNAME] = []
    mxs: list[MX] = []
    txts: list[TXT] = []
    ptr_overrides: list[PTR_override] = []
    hinfo: HInfo | None = None
    loc: Location | None = None
    bacnetid: int | None = None
    ttl: int | None = None
    srvs: list[Srv] = []
    naptrs: list[NAPTR] = []
    sshfps: list[SSHFP] = []
    roles: list[str] = []
    hostgroups: list[str] = []
    comment: str
    contacts: list[ContactEmail] = []
    contact: str | None = Field(default=None, deprecated=True)
    zone: int | None = None

    communities: list[HostCommunity] = []

    @model_validator(mode="after")
    def _set_deprecated_contact_field(self) -> Self:
        """Set the contacts field from the deprecated contact field, if needed.

        Ensures backwards compatibility with older server versions that do not
        implement the `contacts` field.
        """
        if not self.contacts:
            # Only enter context manager if we have no contacts field
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                if self.contact:
                    # HACK: The field itself is immutable, but it always contains
                    #       a list object that we can append to.
                    dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=self.created_at.tzinfo)  # tz-aware epoch time
                    self.contacts.append(ContactEmail(id=0, email=self.contact, created_at=dt, updated_at=dt))
        return self

    @property
    def contact_emails(self) -> list[str]:
        """A list of contact email addresses for the host."""
        return [contact.email for contact in self.contacts]

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


class HostList(MregModel):
    """Model for a list of hosts."""

    results: list[Host]


class DhcpHost(MregModel, ABC):
    """Base model for all DHCP hosts."""

    # NOTE: no timestamps from this endpoint
    name: str = Field(validation_alias=AliasChoices("name", "host__name"))
    ipaddress: IP_AddressT
    macaddress: MacAddress
    zone: str | None = Field(
        default=None, validation_alias=AliasChoices("zone", "host__zone__name")
    )  # Name of the zone


class DhcpHostIPv4(DhcpHost):
    """Model for a DHCP host with an IPv4 address."""


class DhcpHostIPv6(DhcpHost):
    """Model for a DHCP host with an IPv6 address."""


class DhcpHostIPv6ByIPv4(DhcpHost):
    """Model for a DHCP hosts with an IPv6 address via IPv4 address."""


class HostGroup(MregModelWithTimestamps):
    """Model for a hostgroup."""

    id: int  # noqa: A003
    name: str
    description: str | None = None
    parent: NameList
    groups: NameList
    hosts: NameList
    owners: NameList

    def has_group(self, groupname: str) -> bool:
        """Check if the hostgroup has the given group.

        Args:
            groupname: The group to check for.

        Returns:
            True if the hostgroup has the group, False otherwise.
        """
        return groupname in self.groups

    def has_host(self, hostname: str) -> bool:
        """Check if the hostgroup has the given host.

        Args:
            hostname: The host to check for.

        Returns:
            True if the hostgroup has the host, False otherwise.
        """
        return hostname in self.hosts

    def has_owner(self, ownername: str) -> bool:
        """Check if the hostgroup has the given owner.

        Args:
            ownername: The owner to check for.

        Returns:
            True if the hostgroup has the owner, False otherwise.
        """
        return ownername in self.owners


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


class ServerVersion(BaseModel):
    """Model for server version metadata."""

    version: str


class Library(BaseModel):
    """Model for library metadata."""

    name: str
    version: str


class ServerLibraries(BaseModel):
    """Model for server libraries metadata."""

    libraries: list[Library]


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


class LDAPHealth(BaseModel):
    """Model for LDAP health endpoint."""

    status: str


class HeartbeatHealth(BaseModel):
    """Model for heartbeat health endpoint."""

    uptime: int
    start_time: int

    def as_str(self) -> str:
        """Return the uptime as a string."""
        # If we got a negative datetime, we weren't able to fetch the heartbeat
        return str(timedelta(seconds=self.uptime)) if self.uptime > 0 else "Unknown"


class HealthInfo(BaseModel):
    """Combined information from all health endpoints."""

    heartbeat: HeartbeatHealth
    ldap: LDAPHealth


class TokenAuth(MregModel):
    """Model for token authentication response."""

    token: str
