"""Pydantic models for API resources."""

from __future__ import annotations

import importlib
import ipaddress
import logging
from abc import ABC
from collections.abc import Mapping
from datetime import date
from datetime import datetime
from functools import cached_property
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
from typing_extensions import override

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
from mreg_api.exceptions import IPNetworkError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.exceptions import PatchError
from mreg_api.exceptions import PostError
from mreg_api.exceptions import UnexpectedDataError
from mreg_api.models.abstracts import APIMixin
from mreg_api.models.abstracts import FrozenModel
from mreg_api.models.abstracts import FrozenModelWithTimestamps
from mreg_api.models.abstracts import MregBaseModel
from mreg_api.models.abstracts import manager_only
from mreg_api.models.abstracts import normalize_patch_fields
from mreg_api.models.fields import HostName
from mreg_api.models.fields import MacAddress
from mreg_api.models.fields import NameList
from mreg_api.models.history import HistoryItem
from mreg_api.models.history import HistoryResource
from mreg_api.models.patch_types import CommunityPatch
from mreg_api.models.patch_types import HostPolicyPatch
from mreg_api.models.patch_types import IPAddressPatch
from mreg_api.models.patch_types import LabelPatch
from mreg_api.models.patch_types import NetworkPatch
from mreg_api.models.patch_types import NetworkPolicyPatch
from mreg_api.models.patch_types import PermissionPatch
from mreg_api.models.patch_types import ZonePatch
from mreg_api.types import ClientProtocol
from mreg_api.types import IP_AddressT
from mreg_api.types import IP_NetworkT
from mreg_api.types import Json
from mreg_api.types import QueryParams
from mreg_api.utilities.shared import convert_wildcard_to_regex

logger = logging.getLogger(__name__)

T = TypeVar("T")

IPNetMode = Literal["ipv4", "ipv6", "ip", "network", "networkv4", "networkv6"]

class HostLike(Protocol):
    """Structural type for host objects used in infra models."""

    id: int
    name: HostName
    zone: int | None
    ipaddresses: list[object]


class HostModelLike(Protocol):
    """Structural type for host model class access via lazy import."""

    @classmethod
    def model_validate(cls, obj: object) -> HostLike: ...  # noqa: D102

    @classmethod
    def get_by_any_means_or_raise(  # noqa: D102
        cls,
        client: "ClientProtocol",
        identifier: str | int,
        *,
        inform_as_ptr: bool = True,
        inform_as_cname: bool = True,
    ) -> HostLike: ...

    @classmethod
    def get_by_any_means(  # noqa: D102
        cls,
        client: "ClientProtocol",
        identifier: str | int,
        *,
        inform_as_ptr: bool = True,
        inform_as_cname: bool = True,
    ) -> HostLike | None: ...

    @classmethod
    def get_by_id(cls, client: "ClientProtocol", _id: int) -> HostLike | None: ...  # noqa: D102

    @classmethod
    def get_list_by_field(  # noqa: D102
        cls, client: "ClientProtocol", field: str, value: object, _manager: bool = False
    ) -> list[HostLike]: ...


def _host_model() -> HostModelLike:
    """Lazily resolve Host to avoid module import cycles."""
    module = importlib.import_module("mreg_api.models.host")
    return cast(HostModelLike, module.Host)


class NetworkOrIP(BaseModel):
    """A model for either a network or an IP address."""

    ip_or_network: IP_AddressT | IP_NetworkT

    @classmethod
    @override
    def validate(cls, value: str | IP_AddressT | IP_NetworkT | NetworkOrIP) -> Self:
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
    def parse_or_raise(cls, value: object, mode: None = None) -> IP_AddressT | IP_NetworkT: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: object, mode: Literal["ip"]) -> IP_AddressT: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: object, mode: Literal["ipv4"]) -> ipaddress.IPv4Address: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: object, mode: Literal["ipv6"]) -> ipaddress.IPv6Address: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: object, mode: Literal["network"]) -> IP_NetworkT: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: object, mode: Literal["networkv4"]) -> ipaddress.IPv4Network: ...

    @overload
    @classmethod
    def parse_or_raise(cls, value: object, mode: Literal["networkv6"]) -> ipaddress.IPv6Network: ...

    @classmethod
    def parse_or_raise(cls, value: object, mode: IPNetMode | None = None) -> IP_AddressT | IP_NetworkT:
        """Parse a value as an IP address or network.

        Optionally specify the mode to validate the input as.

        :param value:The value to parse.
        :param mode: The mode to validate the input as.
        :returns: The parsed value as an IP address or network.
        :raises IPNetworkError: If the value is not an IP address or network.
        """
        if not isinstance(
            value,
            (
                str,
                ipaddress.IPv4Address,
                ipaddress.IPv6Address,
                ipaddress.IPv4Network,
                ipaddress.IPv6Network,
                NetworkOrIP,
            ),
        ):
            raise InputFailure(f"Invalid IP address or network value: {value!r}")
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
    def parse(cls, value: object, mode: None = None) -> IP_AddressT | IP_NetworkT | None: ...

    @overload
    @classmethod
    def parse(cls, value: object, mode: Literal["ip"]) -> IP_AddressT | None: ...

    @overload
    @classmethod
    def parse(cls, value: object, mode: Literal["ipv4"]) -> ipaddress.IPv4Address | None: ...

    @overload
    @classmethod
    def parse(cls, value: object, mode: Literal["ipv6"]) -> ipaddress.IPv6Address | None: ...

    @overload
    @classmethod
    def parse(cls, value: object, mode: Literal["network"]) -> IP_NetworkT | None: ...

    @overload
    @classmethod
    def parse(cls, value: object, mode: Literal["networkv4"]) -> ipaddress.IPv4Network | None: ...

    @overload
    @classmethod
    def parse(cls, value: object, mode: Literal["networkv6"]) -> ipaddress.IPv6Network | None: ...

    @classmethod
    def parse(cls, value: object, mode: IPNetMode | None = None) -> IP_AddressT | IP_NetworkT | None:
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
    def validate_ip_or_network(cls, value: object) -> IP_AddressT | IP_NetworkT:
        """Validate and convert the input to an IP address or network."""
        if isinstance(value, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            return value
        if isinstance(value, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
            return value
        if not isinstance(value, str):
            raise InputFailure(f"Invalid input for IP address or network: {value}")

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

    @override
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


class APIModelMixin(MregBaseModel, APIMixin, ABC):
    """Shared base for API-backed model mixins."""


class WithHost(APIModelMixin, ABC):
    """Model for an object that has a host element."""

    host: int

    def resolve_host(self) -> HostLike | None:
        """Resolve the host ID to a Host object.

        Notes:
        -----
            - This method will call the API to resolve the host ID to a Host object.
            - This assumes that there is a host attribute in the object.

        """
        client = self._require_client()

        data = client.get_item_by_key_value(Endpoint.Hosts, "id", str(self.host))

        if not data:
            return None

        host_model = _host_model()
        return client._bind_client(host_model.model_validate(data))


class WithZone(APIModelMixin, ABC):
    """Model for an object that has a zone element."""

    zone: int | None = None

    def resolve_zone(self) -> ForwardZone | None:
        """Resolve the zone ID to a (Forward)Zone object.

        Notes:
        -----
            - This method will call the API to resolve the zone ID to a Zone object.
            - This assumes that there is a zone attribute in the object.
        """
        if self.zone is None:
            return None

        client = self._require_client()

        data = client.get_item_by_key_value(Endpoint.ForwardZones, "id", str(self.zone))

        if not data:
            return None

        return ForwardZone.model_validate(data)


class WithTTL(APIModelMixin, ABC):
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


class WithName(APIModelMixin, ABC):
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
    @manager_only
    def get_by_name(cls, client: "ClientProtocol", name: str) -> Self | None:
        """Get a resource by name.

        :param name: The resource name to search for.
        :returns: The resource if found.
        """
        return cls.get_by_field(client, cls.__name_field__, cls._case_name(name))

    @classmethod
    @manager_only
    def get_by_name_and_raise(cls, client: "ClientProtocol", name: str) -> None:
        """Get a resource by name, raising EntityAlreadyExists if found.

        :param name: The resource name to search for.
        :raises EntityAlreadyExists: If the resource is found.
        """
        return cls.get_by_field_and_raise(client, cls.__name_field__, cls._case_name(name))

    @classmethod
    @manager_only
    def get_by_name_or_raise(cls, client: "ClientProtocol", name: str) -> Self:
        """Get a resource by name, raising EntityNotFound if not found.

        :param name: The resource name to search for.
        :returns: The resource.
        :raises EntityNotFound: If the resource is not found.
        """
        return cls.get_by_field_or_raise(client, cls.__name_field__, cls._case_name(name))

    @classmethod
    @manager_only
    def get_list_by_name_regex(cls, client: "ClientProtocol", name: str) -> list[Self]:
        """Get multiple resources by a name regex.

        :param name: The regex pattern for names to search for.
        :returns: A list of resource objects.
        """
        param, value = convert_wildcard_to_regex(cls.__name_field__, cls._case_name(name), True)
        return client.get_typed(cls.endpoint(), list[cls], params={param: value})

    def rename(self, new_name: str) -> Self:
        """Rename the resource.

        :param new_name: The new name to set.
        :returns: The patched resource.
        """
        return self.patch({self.__name_field__: self._case_name(new_name)})


ClassVarNotSet = object()


def AbstractClassVar() -> HistoryResource:
    """Hack to implement an abstract class variable on a Pydantic model."""
    return cast(HistoryResource, ClassVarNotSet)


class WithHistory(APIModelMixin, ABC):
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
    @manager_only
    def get_history(cls, client: "ClientProtocol", name: str) -> list[HistoryItem]:
        """Get the history for the object."""
        return HistoryItem.get(client, name, cls.history_resource)


class NameServer(FrozenModelWithTimestamps, WithTTL):
    """Model for representing a nameserver within a DNS zone."""

    id: int  # noqa: A003
    name: str

    @classmethod
    @override
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
    def validate_ip_or_network(cls, value: object) -> IP_NetworkT:
        """Validate and convert the input to a network."""
        if not isinstance(
            value,
            (
                str,
                int,
                bytes,
                ipaddress.IPv4Address,
                ipaddress.IPv6Address,
                ipaddress.IPv4Network,
                ipaddress.IPv6Network,
                tuple,
            ),
        ):
            raise InputFailure(f"Invalid input for network: {value}")
        try:
            return ipaddress.ip_network(value)
        except (TypeError, ValueError) as e:
            raise InputFailure(f"Invalid input for network: {value}") from e

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.PermissionNetgroupRegex

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[PermissionPatch]) -> Self:
        """Patch permission with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)

    def add_label(self, label_name: str) -> Self:
        """Add a label to the permission.

        :param label_name: The name of the label to add.
        :returns: The updated Permission object.
        """
        client = self._require_client()
        label = cast(Label, Label.get_by_name_or_raise(client, label_name, _manager=True))
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
        client = self._require_client()
        label = cast(Label, Label.get_by_name_or_raise(client, label_name, _manager=True))
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


class Zone(FrozenModelWithTimestamps, WithTTL):
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
    @override
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
    @manager_only
    def verify_nameservers(
        cls, client: "ClientProtocol", nameservers: list[str], force: bool = False
    ) -> None:
        """Verify that nameservers are in mreg and have A-records."""
        if not nameservers:
            raise InputFailure("At least one nameserver is required")

        errors: list[str] = []
        for nameserver in nameservers:
            try:
                host = _host_model().get_by_any_means_or_raise(client, nameserver)
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
    @manager_only
    def create_zone(
        cls,
        client: "ClientProtocol",
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
        cls.verify_nameservers(client, primary_ns, force=force)
        zone_t = cls.type_by_name(name)
        zone_t.get_zone_and_raise(client, name)
        return zone_t.create(client, {"name": name, "email": email, "primary_ns": primary_ns})

    @classmethod
    @manager_only
    def get_zone(cls, client: "ClientProtocol", name: str) -> ForwardZone | ReverseZone | None:
        """Get a zone by name.

        :param name: The name of the zone to get.
        :returns: The zone object.
        """
        zone_t = cls.type_by_name(name)
        return zone_t.get_by_name(client, name)

    @classmethod
    @manager_only
    def get_zone_or_raise(cls, client: "ClientProtocol", name: str) -> ForwardZone | ReverseZone:
        """Get a zone by name, and raise if not found.

        :param name: The name of the zone to get.
        :returns: The zone object.
        """
        zone_t = cls.type_by_name(name)
        return zone_t.get_by_name_or_raise(client, name)

    @classmethod
    @manager_only
    def get_zone_and_raise(cls, client: "ClientProtocol", name: str) -> None:
        """Get a zone by name, and raise if found.

        :param name: The name of the zone to get.
        """
        zone_t = cls.type_by_name(name)
        return zone_t.get_by_name_and_raise(client, name)

    def get_subzones(self) -> list[Self]:
        """Get subzones of the zone, excluding self.

        :returns: A list of subzones.
        """
        client = self._require_client()
        zones = cast(list[Self], self.get_list_by_field(client, "name__endswith", f".{self.name}", _manager=True))
        return [zone for zone in zones if zone.name != self.name]

    def ensure_deletable(self) -> None:
        """Ensure the zone can be deleted. Raises exception if not.

        :raises DeleteError: If zone has entries or subzones.
        """
        client = self._require_client()
        # XXX: Not a fool proof check, as e.g. SRVs are not hosts. (yet.. ?)
        hosts = _host_model().get_list_by_field(client, "zone", self.id, _manager=True)
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

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[ZonePatch]) -> Self:
        """Patch zone with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)

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
        client = self._require_client()
        self.ensure_delegation_in_zone(delegation)
        self.verify_nameservers(client, nameservers, force=force, _manager=True)

        if not force:
            # Ensure delegated zone exists and is same type as parent zone
            delegated_zone = cast(Zone | None, Zone.get_zone(client, delegation, _manager=True))
            if not delegated_zone:
                raise InputFailure(f"Zone {delegation!r} does not exist. Must force.")
            if delegated_zone.is_reverse() != self.is_reverse():
                raise InputFailure(f"Delegation '{delegation}' is not a {self.__class__.__name__} zone")

        self.get_delegation_and_raise(delegation)

        cls = Delegation.type_by_zone(self)
        try:
            _ = client.post(
                cls.endpoint().with_params(self.name),
                name=delegation,
                nameservers=nameservers,
                comment=comment,
            )
        except PostError as e:
            # TODO: implement after mreg-cli parity
            # raise PostError(
            #     f"Failed to create delegation {delegation!r} in zone {self.name!r}", e.response
            # ) from e
            raise e

        if fetch_after_create:
            return self.get_delegation_or_raise(delegation)
        return None

    def get_delegation(self, name: str) -> ForwardZoneDelegation | ReverseZoneDelegation | None:
        """Get a delegation for the zone by name.

        :param name: The name of the delegation to get.
        :returns: The delegation object if found.
        """
        client = self._require_client()
        self.ensure_delegation_in_zone(name)
        cls = Delegation.type_by_zone(self)
        resp = client.get(cls.endpoint_with_id(self, name), ok404=True)
        if not resp:
            return None
        delegation = cls.model_validate_json(resp.text)
        return client._bind_client(delegation)

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

        :returns: The delegations for this zone.
        """
        client = self._require_client()
        cls = Delegation.type_by_zone(self)
        return client.get_typed(cls.endpoint().with_params(self.name), list[cls])

    def delete_delegation(self, name: str) -> bool:
        """Delete a delegation from the zone.

        :param delegation: The name of the delegation.
        :returns: True if the deletion was successful.
        """
        client = self._require_client()
        # Check if delegation exists
        self.ensure_delegation_in_zone(name)  # check name
        delegation = self.get_delegation_or_raise(name)
        resp = client.delete(delegation.endpoint_with_id(self, name))
        return resp.is_success if resp else False

    def set_delegation_comment(self, name: str, comment: str) -> None:
        """Set the comment for a delegation.

        :param name: The name of the delegation.
        :param comment: The comment to set.
        """
        client = self._require_client()
        delegation = self.get_delegation_or_raise(name)
        try:
            _ = client.patch(delegation.endpoint_with_id(self, delegation.name), comment=comment)
        except PatchError as e:
            # TODO: implement after mreg-cli parity
            # raise PatchError(
            #     f"Failed to update comment for delegation {delegation.name!r}", e.response
            # ) from e
            raise e

    def set_default_ttl(self, ttl: int) -> Self:
        """Set the default TTL for the zone.

        :param ttl: The TTL to set.
        """
        return self.set_ttl(ttl, "default_ttl")

    def update_nameservers(
        self,
        nameservers: list[str],
        force: bool = False,
    ) -> None:
        """Update the nameservers of the zone.

        :param nameservers: The new nameservers for the zone.
        :param force: Whether to force the update.
        :returns: True if the update was successful.
        """
        client = self._require_client()
        self.verify_nameservers(client, nameservers, force=force, _manager=True)
        path = self.endpoint_nameservers().with_params(self.name)
        try:
            _ = client.patch(path, primary_ns=nameservers)
        except PatchError as e:
            # TODO: implement after mreg-cli parity
            # raise PatchError(
            #     f"Failed to update nameservers for {self.__class__.__name__} {self.name!r}",
            #     e.response,
            # ) from e
            raise e


class ForwardZone(Zone, WithName):
    """A forward zone."""

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZones

    @classmethod
    @override
    def endpoint_nameservers(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZonesNameservers

    @classmethod
    @manager_only
    def get_from_hostname(
        cls, client: "ClientProtocol", hostname: HostName
    ) -> ForwardZoneDelegation | ForwardZone | None:
        """Get the zone from a hostname.

        Note: This method may return either a ForwardZoneDelegation or a ForwardZone object.

        :param hostname: The hostname to search for.
        :returns: The zone if found, None otherwise.
        """
        resp = client.get(Endpoint.ForwardZoneForHost.with_id(hostname), ok404=True)
        if not resp:
            return None

        zoneblob_raw = cast(object, resp.json())
        if not isinstance(zoneblob_raw, Mapping):
            raise UnexpectedDataError(f"Unexpected response from server: {zoneblob_raw}", resp)
        zoneblob = cast(Mapping[str, object], zoneblob_raw)

        if "delegate" in zoneblob:
            return client._bind_client(ForwardZoneDelegation.model_validate(zoneblob))

        zone_data = zoneblob.get("zone")
        if zone_data is not None:
            return client._bind_client(ForwardZone.model_validate(zone_data))

        delegation_data = zoneblob.get("delegation")
        if delegation_data is not None:
            return client._bind_client(ForwardZoneDelegation.model_validate(delegation_data))

        raise UnexpectedDataError(f"Unexpected response from server: {zoneblob}", resp)


class ReverseZone(Zone, WithName):
    """A reverse zone."""

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ReverseZones

    @classmethod
    @override
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
    @override
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


class ForwardZoneDelegation(Delegation):
    """A forward zone delegation."""

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ForwardZonesDelegations


class ReverseZoneDelegation(Delegation):
    """A reverse zone delegation."""

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.ReverseZonesDelegations

    @classmethod
    @override
    def is_reverse(cls) -> bool:
        """Return True if the delegation is for a reverse zone."""
        return True


class HostPolicy(FrozenModel, WithName, ABC):
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
    def validate_created_at(cls, value: object) -> datetime:
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
        raise ValueError(f"Invalid datetime value {value!r}")

    @computed_field
    @property
    def created_at(self) -> datetime:
        """Creation time."""
        return self.created_at_tz_naive.replace(tzinfo=self.updated_at.tzinfo)

    # Fetching Host Policy objects is a special case where we cannot
    # re-use the methods defined in WithName, because we don't have an endpoint
    # defined on the class that can fetch both Roles and Atoms.
    # Thus, we need to define our own implementations of these methods.
    @classmethod
    @manager_only
    def get_role_or_atom(cls, client: "ClientProtocol", name: str) -> Atom | Role | None:
        """Get an Atom or Role by name.

        :param name: The name to search for.
        :returns: The Atom or Role if found, else None.
        """
        funcs: list[Callable[[str], Atom | Role | None]] = [
            lambda n: Atom.get_by_name(client, n),
            lambda n: Role.get_by_name(client, n),
        ]
        for func in funcs:
            role_or_atom = func(name)
            if role_or_atom:
                return role_or_atom
        return None

    @classmethod
    @manager_only
    def get_role_or_atom_or_raise(cls, client: "ClientProtocol", name: str) -> Atom | Role:
        """Get an Atom or Role by name and raise if not found.

        :param name: The name to search for.
        :returns: The Atom or Role if found.
        :raises EntityNotFound: If the Atom or Role is not found.
        """
        role_or_atom = cls.get_role_or_atom(client, name)
        if role_or_atom:
            return role_or_atom
        raise EntityNotFound(f"Could not find an atom or a role with name {name}")

    @classmethod
    @manager_only
    def get_role_or_atom_and_raise(cls, client: "ClientProtocol", name: str) -> None:
        """Get an Atom or Role by name and raise if found.

        :param name: The name to search for.
        :returns: The Atom or Role if found.
        :raises EntityAlreadyExists: If the Atom or Role is found.
        """
        role_or_atom = cls.get_role_or_atom(client, name)
        if role_or_atom:
            raise EntityAlreadyExists(f"An atom or a role with name {name} already exists.")

    def set_description(self, description: str) -> Self:
        """Set a new description."""
        return self.patch({"description": description})

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[HostPolicyPatch]) -> Self:
        """Patch host policy with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)


class Role(HostPolicy, WithHistory):
    """Model for a role."""

    id: int  # noqa: A003
    hosts: NameList
    atoms: NameList
    labels: list[int]

    history_resource: ClassVar[HistoryResource] = HistoryResource.HostPolicy_Role

    @override
    def __hash__(self) -> int:
        """Hash the role by ID and name."""
        return hash(str(self.id) + self.name)

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HostPolicyRoles

    @classmethod
    @manager_only
    def get_roles_with_atom(cls, client: "ClientProtocol", name: str) -> list[Self]:
        """Get all roles with a specific atom.

        :param atom: Name of the atom to search for.
        :returns: A list of Role objects.
        """
        return client.get_typed(cls.endpoint(), list[cls], params={"atoms__name__exact": name})

    def add_atom(self, atom_name: str) -> bool:
        """Add an atom to the role.

        :param atom_name: The name of the atom to add.
        """
        client = self._require_client()
        # Ensure the atom exists
        _ = Atom.get_by_name_or_raise(client, atom_name, _manager=True)
        for atom in self.atoms:
            if atom_name == atom:
                raise EntityAlreadyExists(f"Atom {atom!r} already a member of role {self.name!r}")

        resp = client.post(Endpoint.HostPolicyRolesAddAtom.with_params(self.name), name=atom_name)
        return resp.is_success if resp else False

    def remove_atom(self, atom_name: str) -> bool:
        """Remove an atom from the role.

        :param atom_name: The name of the atom to remove.
        """
        client = self._require_client()
        for atom in self.atoms:
            if atom_name == atom:
                break
        else:
            raise EntityOwnershipMismatch(f"Atom {atom_name!r} not a member of {self.name!r}")

        resp = client.delete(Endpoint.HostPolicyRolesRemoveAtom.with_params(self.name, atom))
        return resp.is_success if resp else False

    def get_labels(self) -> list[Label]:
        """Get the labels associated with the role.

        :returns: A list of Label objects.
        """
        client = self._require_client()
        return [Label.get_by_id_or_raise(client, id_, _manager=True) for id_ in self.labels]

    def add_label(self, label_name: str) -> Self:
        """Add a label to the role.

        :param label_name: The name of the label to add.

        :returns: The updated Role object.
        """
        client = self._require_client()
        label = cast(Label, Label.get_by_name_or_raise(client, label_name, _manager=True))
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
        client = self._require_client()
        label = cast(Label, Label.get_by_name_or_raise(client, label_name, _manager=True))
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
        client = self._require_client()
        resp = client.post(Endpoint.HostPolicyRolesAddHost.with_params(self.name), name=name)
        return resp.is_success if resp else False

    def remove_host(self, name: str) -> bool:
        """Remove a host from the role by name.

        :param name: The name of the host to remove.
        """
        client = self._require_client()
        resp = client.delete(Endpoint.HostPolicyRolesRemoveHost.with_params(self.name, name))
        return resp.is_success if resp else False

    @override
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
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HostPolicyAtoms

    @override
    def delete(self) -> bool:
        """Delete the atom."""
        client = self._require_client()
        roles = Role.get_roles_with_atom(client, self.name, _manager=True)
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
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Labels

    @classmethod
    @manager_only
    def get_all(cls, client: "ClientProtocol") -> list[Self]:
        """Get all labels.

        :returns: A list of Label objects.
        """
        return client.get_typed(cls.endpoint(), list[cls], params={"ordering": "name"})

    @classmethod
    @manager_only
    def get_by_id_or_raise(cls, client: "ClientProtocol", _id: int) -> Self:
        """Get a Label by ID.

        :param _id: The Label ID to search for.
        :returns: The Label if found.
        :raises EntityNotFound: If the Label is not found.
        """
        label = cls.get_by_id(client, _id)
        if not label:
            raise EntityNotFound(f"Label with ID {_id} not found.")
        return label

    def set_description(self, description: str) -> Self:
        """Set a new description."""
        return self.patch({"description": description})

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[LabelPatch]) -> Self:
        """Patch label with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)


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

    @override
    def __hash__(self) -> int:
        """Return a hash of the network."""
        return hash((self.id, self.network))

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[NetworkPatch]) -> Self:
        """Patch network with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)

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
    @override
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
    @manager_only
    def get_by_any_means(cls, client: "ClientProtocol", identifier: str) -> Self | None:
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
                return cls.get_by_network(client, str(net_or_ip))
            elif net_or_ip.is_ip():
                return cls.get_by_ip(client, net_or_ip.as_ip())
        # Check if identifier is an ID
        if identifier.isdigit():
            try:
                return cls.get_by_id(client, int(identifier))
            except ValueError:
                pass
        return None

    @classmethod
    @manager_only
    def get_by_any_means_or_raise(cls, client: "ClientProtocol", identifier: str) -> Self:
        """Get a network by the given identifier, and raise if not found.

        See `get_by_any_means` for details.
        """
        net = cls.get_by_any_means(client, identifier)
        if not net:
            raise EntityNotFound(f"Network {identifier!r} not found.")
        return net

    @classmethod
    @manager_only
    def get_by_ip(cls, client: "ClientProtocol", ip: IP_AddressT) -> Self | None:
        """Get a network by IP address.

        :param ip: The IP address to search for.
        :returns: The network if found, None otherwise.
        :raises EntityNotFound: If the network is not found.
        """
        resp = client.get(Endpoint.NetworksByIP.with_id(str(ip)), ok404=True)
        if not resp:
            return None
        return client._bind_client(cls.model_validate_json(resp.text))

    @classmethod
    @manager_only
    def get_by_ip_or_raise(cls, client: "ClientProtocol", ip: IP_AddressT) -> Network:
        """Get a network by IP address, and raise if not found.

        :param ip: The IP address to search for.
        :returns: The network if found, None otherwise.
        :raises EntityNotFound: If the network is not found.
        """
        network = cls.get_by_ip(client, ip)
        if not network:
            raise EntityNotFound(f"Network with IP address {ip} not found.")
        return network

    @classmethod
    @manager_only
    def get_by_network(cls, client: "ClientProtocol", network: str) -> Self | None:
        """Get a network by network address.

        :param network: The network string to search for.
        :returns: The network if found.
        """
        return cls.get_by_field(client, "network", network)

    @classmethod
    @manager_only
    def get_by_network_or_raise(cls, client: "ClientProtocol", network: str) -> Self:
        """Get a network by its network address, and raise if not found.

        :param network: The network string to search for.
        :returns: The network if found.
        :raises EntityNotFound: If the network is not found.
        """
        net = cls.get_by_network(client, network)
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
        client = self._require_client()
        resp = client.post(
            Endpoint.NetworkCommunities.with_params(self.network),
            name=name,
            description=description,
        )
        return resp.is_success if resp else False

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
        client = self._require_client()
        return ipaddress.ip_address(
            client.get_typed(Endpoint.NetworksFirstUnused.with_params(self.network), str)
        )

    def get_reserved_ips(self) -> list[IP_AddressT]:
        """Return the reserved IP addresses of the network."""
        client = self._require_client()
        return client.get_typed(
            Endpoint.NetworksReservedList.with_params(self.network), list[IP_AddressT]
        )

    def get_used_count(self) -> int:
        """Return the number of used IP addresses in the network."""
        client = self._require_client()
        return client.get_typed(Endpoint.NetworksUsedCount.with_params(self.network), int)

    def get_used_list(self) -> list[IP_AddressT]:
        """Return the list of used IP addresses in the network."""
        client = self._require_client()
        return client.get_typed(
            Endpoint.NetworksUsedList.with_params(self.network), list[IP_AddressT]
        )

    def get_unused_count(self) -> int:
        """Return the number of unused IP addresses in the network."""
        client = self._require_client()
        return client.get_typed(Endpoint.NetworksUnusedCount.with_params(self.network), int)

    def get_unused_list(self) -> list[IP_AddressT]:
        """Return the list of unused IP addresses in the network."""
        client = self._require_client()
        return client.get_typed(
            Endpoint.NetworksUnusedList.with_params(self.network), list[IP_AddressT]
        )

    def get_used_host_list(self) -> dict[str, list[str]]:
        """Return a dict of used IP addresses and their associated hosts."""
        client = self._require_client()
        return client.get_typed(
            Endpoint.NetworksUsedHostList.with_params(self.network), dict[str, list[str]]
        )

    def get_ptroverride_host_list(self) -> dict[str, str]:
        """Return a dict of PTR override IP addresses and their associated hosts."""
        client = self._require_client()
        return client.get_typed(
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
        client = self._require_client()
        start_ip = NetworkOrIP.parse_or_raise(start, mode="ip")
        end_ip = NetworkOrIP.parse_or_raise(end, mode="ip")
        if start_ip.version != end_ip.version:
            raise InputFailure("Start and end IP addresses must be of the same version")

        try:
            _ = client.post(
                Endpoint.NetworksAddExcludedRanges.with_params(self.network),
                network=self.id,
                start_ip=str(start_ip),
                end_ip=str(end_ip),
            )
        except PostError as e:
            # TODO: implement after mreg-cli parity
            # raise PostError(
            #     f"Failed to create excluded range for network {self.network}", e.response
            # ) from e
            raise e

    def remove_excluded_range(self, start: str, end: str) -> None:
        """Remove an excluded range from the network.

        :param start: The start of the excluded range.
        :param end: The end of the excluded range.
        """
        client = self._require_client()
        # No need to validate IPs - if we find a match it's valid
        exrange: ExcludedRange | None = None
        for excluded_range in self.excluded_ranges:
            if str(excluded_range.start_ip) == start and str(excluded_range.end_ip) == end:
                exrange = excluded_range
                break
        else:
            raise EntityNotFound(f"Excluded range {start} - {end} not found")
        resp = client.delete(
            Endpoint.NetworksRemoveExcludedRanges.with_params(self.network, exrange.id)
        )
        if not resp or not resp.is_success:
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
        client = self._require_client()
        return NetworkPolicy.get_list_by_field(client, "attributes", self.id, _manager=True)

    @classmethod
    @override
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
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.NetworkCommunity

    @property
    def endpoint_with_id(self) -> str:
        """Return the endpoint with the community ID."""
        return self.endpoint().with_params(self.network_address, self.id)

    @property
    def hosts_endpoint(self) -> str:
        """Return the endpoint with policy and community IDs."""
        return Endpoint.NetworkCommunityHosts.with_params(self.network_address, self.id)

    @property
    def network_address(self) -> str:
        """Return the network object for the community."""
        client = self._require_client()
        network = cast(Network, Network.get_by_field_or_raise(client, "id", str(self.network), _manager=True))
        return network.network

    @override
    def refetch(self) -> Self:
        """Refetch the community object."""
        client = self._require_client()
        return client.get_typed(self.endpoint_with_id, self.__class__)

    @override
    def patch(
        self,
        fields: Mapping[str, object] | None = None,
        validate: bool = True,
        **field_kwargs: object,
    ) -> Self:
        """Patch the community.

        :param fields: The fields to patch.
        :param validate: Whether to validate the response. (Not implemented)
        :param field_kwargs: Keyword patch arguments.
        :returns: The updated Community object.
        """
        client = self._require_client()
        _ = validate
        patch_fields = normalize_patch_fields(fields, field_kwargs)
        patch_payload = cast(dict[str, Json], patch_fields)
        try:
            client.patch(self.endpoint_with_id, params=None, **patch_payload)
        except PatchError as e:
            # TODO: implement after mreg-cli parity
            # raise PatchError(f"Failed to patch community {self.name!r}", e.response) from e
            raise e
        new_object = self.refetch()
        return new_object

    @override
    def patch_raw(self, fields: dict[str, object], validate: bool = True) -> Self:
        """Patch the community using a raw dictionary payload."""
        return self.patch(fields=fields, validate=validate)

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[CommunityPatch]) -> Self:
        """Patch community with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)

    @override
    def delete(self) -> bool:
        """Delete the community."""
        client = self._require_client()
        resp = client.delete(self.endpoint_with_id)
        return resp.is_success if resp else False

    def get_hosts(self) -> list[HostLike]:
        """Get a list of hosts in the community.

        :returns: A list of Host objects.
        """
        client = self._require_client()
        response = client.get(self.hosts_endpoint)
        payload = cast(object, response.json())
        if not isinstance(payload, list):
            raise UnexpectedDataError(f"Unexpected response from server: {payload}", response)

        host_model = _host_model()
        payload_items = cast(list[object], payload)
        hosts: list[HostLike] = []
        for item in payload_items:
            if not isinstance(item, Mapping):
                raise UnexpectedDataError(f"Unexpected host payload from server: {item}", response)
            host_payload = cast(dict[str, object], item)
            hosts.append(client._bind_client(host_model.model_validate(host_payload)))
        return hosts

    def add_host(
        self,
        host: HostLike,
        ipaddress: IP_AddressT | None = None,
    ) -> bool:
        """Add a host to the community.

        :param host: The host to add.
        :returns: True if the host was added, False otherwise.
        """
        client = self._require_client()
        kwargs: QueryParams = {"id": host.id}
        if ipaddress:
            kwargs["ipaddress"] = str(ipaddress)
        resp = client.post(self.hosts_endpoint, params=None, ok404=False, **kwargs)
        return resp.is_success if resp else False

    def remove_host(self, host: HostLike, ipaddress: IP_AddressT | None = None) -> bool:
        """Remove a host from the community.

        :param host: The host to remove.
        :returns: True if the host was removed, False otherwise.
        """
        client = self._require_client()
        params: QueryParams = {}
        if ipaddress:
            params["ipaddress"] = str(ipaddress)
        resp = client.delete(
            Endpoint.NetworkCommunityHost.with_params(
                self.network_address,
                self.id,
                host.id,
            )
        )
        return resp.is_success if resp else False


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
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.NetworkPolicies

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[NetworkPolicyPatch]) -> Self:
        """Patch network policy with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)

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
        _ = self.get_attribute_or_raise(attribute)
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
        _ = self.patch(
            {"attributes": [{"name": a.name, "value": a.value} for a in attrs]},
            validate=False,
        )
        # NOTE: can return self.refetch() here if we need to refresh the object

    def networks(self) -> list[Network]:
        """Get all networks using this policy."""
        client = self._require_client()
        return Network.get_list_by_field(client, "policy", self.id, _manager=True)

    def create_community(self, name: str, description: str) -> Community | None:
        """Create a new community.

        :param name: The name of the community.
        :param description: The description of the community.
        :returns: The new Community object.
        """
        client = self._require_client()
        resp = client.post(
            Endpoint.NetworkCommunities.with_params(self.id),
            name=name,
            description=description,
        )
        if resp and "Location" in resp.headers:
            return client.get_typed(resp.headers["Location"], Community)
        return None


class IPAddress(FrozenModelWithTimestamps, WithHost):
    """Represents an IP address with associated details."""

    id: int  # noqa: A003
    macaddress: MacAddress | None = None
    ipaddress: IP_AddressT

    @field_validator("macaddress", mode="before")
    @classmethod
    def create_valid_macadress_or_none(cls, v: object) -> MacAddress | str | None:
        """Create macaddress or convert empty strings to None.

        The API can return an empty string for this field, which fails to validate
        as a MAC address. Therefore, treat empty strings as None.
        """
        if isinstance(v, str) and v:
            return v
        if isinstance(v, MacAddress):
            return v
        return None

    @classmethod
    @manager_only
    def get_by_ip(cls, client: "ClientProtocol", ip: IP_AddressT) -> list[Self]:
        """Get a list of IP address objects by IP address.

        Note that the IP addresses can be duplicated across hosts,
        so this method may return multiple IP addresses.

        :param ip: The IP address to search for.
        :returns: The IP address if found, None otherwise.
        """
        return cls.get_list_by_field(client, "ipaddress", str(ip))

    @classmethod
    @manager_only
    def ensure_associable(cls, client: "ClientProtocol", mac: MacAddress, force: bool) -> None:
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

        ips = cls.get_ips_by_mac(client, mac)
        ips = cast(list[Self], ips)
        if not ips:
            return

        if len(ips) == 1:
            raise EntityAlreadyExists(
                f"MAC address {mac} is already associated with IP address {ips[0].ipaddress}, must force."  # noqa: E501
            )
        else:
            ips_str = ", ".join(str(ip.ipaddress) for ip in ips)
            raise MultipleEntitiesFound(
                f"MAC address {mac} is already associated with multiple IP addresses: {ips_str}, must force."  # noqa: E501
            )

    @classmethod
    @manager_only
    def get_by_mac(cls, client: "ClientProtocol", mac: MacAddress) -> Self | None:
        """Get the IP address objects by MAC address.

        :param mac: The MAC address to search for.
        :returns: The IP address if found, None otherwise.
        """
        try:
            return cls.get_by_field(client, "macaddress", mac)
        except MultipleEntitiesFound as e:
            raise MultipleEntitiesFound(f"Multiple IPs found with MAC address {mac}.") from e

    @classmethod
    @manager_only
    def get_ips_by_mac(cls, client: "ClientProtocol", mac: MacAddress) -> list[Self]:
        """Get a list of IP addresses by MAC address.

        :param mac: The MAC address to search for.
        :returns: A list of IP addresses.
        """
        return cls.get_list_by_field(client, "macaddress", mac)

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Ipaddresses

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[IPAddressPatch]) -> IPAddress:
        """Patch IP address with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)

    @override
    def __str__(self) -> str:
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
        client = self._require_client()
        try:
            return client.get_typed(Endpoint.NetworksByIP.with_id(str(self.ipaddress)), Network)
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

    @override
    def __hash__(self) -> int:
        """Return a hash of the IP address."""
        return hash((self.id, self.ipaddress, self.macaddress))


class HInfo(FrozenModelWithTimestamps, WithHost):
    """Represents a HINFO record."""

    cpu: str
    os: str

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Hinfos


class CNAME(FrozenModelWithTimestamps, WithHost, WithZone, WithTTL):
    """Represents a CNAME record."""

    id: int  # noqa: A003
    name: HostName
    ttl: int | None = None

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Cnames

    @classmethod
    @manager_only
    def get_by_name(cls, client: "ClientProtocol", name: HostName) -> CNAME:
        """Get a CNAME record by name.

        :param name: The name to search for.
        :returns: The CNAME record if found, None otherwise.
        """
        data = client.get_item_by_key_value(Endpoint.Cnames, "name", name)
        if not data:
            raise EntityNotFound(f"CNAME record for {name} not found.")
        return client._bind_client(CNAME.model_validate(data))

    @classmethod
    @manager_only
    def get_by_host_and_name(
        cls, client: "ClientProtocol", host: HostName | int, name: HostName
    ) -> CNAME:
        """Get a CNAME record by host and name.

        :param host: The host to search for, either a hostname or an ID.
        :param name: The name to search for.
        :returns: The CNAME record if found, None otherwise.
        """
        target_hostname: HostName | None = None
        if isinstance(host, str):
            hostobj = _host_model().get_by_any_means(client, host, inform_as_cname=False)
            if not hostobj:
                raise EntityNotFound(f"Host with name {host} not found.")

            host = hostobj.id
            target_hostname = hostobj.name
        else:
            hostobj = _host_model().get_by_id(client, host)
            if not hostobj:
                raise EntityNotFound(f"Host with ID {host} not found.")
            target_hostname = hostobj.name

        results = cast(list[CNAME], cls.get_by_query(client, {"host": str(host), "name": name}))

        if not results or len(results) == 0:
            raise EntityNotFound(f"CNAME record for {name} not found for {target_hostname}.")

        if len(results) > 1:
            raise MultipleEntitiesFound(f"Multiple CNAME records found for {host} with {name}!")

        return results[0]


class TXT(FrozenModelWithTimestamps, WithHost):
    """Represents a TXT record."""

    id: int  # noqa: A003
    txt: str

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Txts


class MX(FrozenModelWithTimestamps, WithHost):
    """Represents a MX record."""

    id: int  # noqa: A003
    mx: str
    priority: int

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Mxs

    @classmethod
    @manager_only
    def get_by_all(cls, client: "ClientProtocol", host: int, mx: str, priority: int) -> MX:
        """Get an MX record by all fields.

        :param host: The host ID.
        :param mx: The MX record.
        :param priority: The priority.
        :returns: The MX record if found, None otherwise.
        """
        data = client.get_list_unique(
            Endpoint.Mxs, params={"host": str(host), "mx": mx, "priority": str(priority)}
        )
        if not data:
            raise EntityNotFound(f"MX record for {mx} not found.")
        return client._bind_client(MX.model_validate(data))

    def has_mx_with_priority(self, mx: str, priority: int) -> bool:
        """Return True if the MX record has the given MX and priority.

        :param mx: The MX record to check.
        :param priority: The priority to check.
        :returns: True if the MX record has the given MX and priority.
        """
        return self.mx == mx and self.priority == priority


class NAPTR(FrozenModelWithTimestamps, WithHost):
    """Represents a NAPTR record."""

    id: int  # noqa: A003
    preference: int
    order: int
    flag: str | None = None
    service: str | None = None
    regex: str | None = None
    replacement: str

    @classmethod
    @override
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


class Srv(FrozenModelWithTimestamps, WithHost, WithZone, WithTTL):
    """Represents a SRV record."""

    id: int  # noqa: A003
    name: str
    priority: int
    weight: int
    port: int
    ttl: int | None = None

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Srvs

    @override
    def __str__(self) -> str:
        """Return a string representation of the SRV record."""
        return self.name


class PTR_override(FrozenModelWithTimestamps, WithHost):
    """Represents a PTR override record."""

    id: int  # noqa: A003
    ipaddress: IP_AddressT

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.PTR_overrides


class SSHFP(FrozenModelWithTimestamps, WithHost, WithTTL):
    """Represents a SSHFP record."""

    id: int  # noqa: A003
    algorithm: int
    hash_type: int
    fingerprint: str
    ttl: int | None = None

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Sshfps


class BacnetID(FrozenModel, WithHost):
    """Represents a Bacnet ID record."""

    id: int  # noqa: A003
    hostname: str

    @classmethod
    def MAX_ID(cls) -> int:
        """Return the maximum ID for a Bacnet ID."""
        return 4194302

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.BacnetID

    @classmethod
    @manager_only
    def get_in_range(cls, client: "ClientProtocol", start: int, end: int) -> list[Self]:
        """Get Bacnet IDs in a range.

        :param start: The start of the range.
        :param end: The end of the range.
        :returns: List of BacnetID objects in the range.
        """
        params: QueryParams = {"id__range": f"{start},{end}"}
        return client.get_typed(Endpoint.BacnetID, list[cls], params=params)


class Location(FrozenModelWithTimestamps, WithHost):
    """Represents a LOC record."""

    loc: str

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Locs
