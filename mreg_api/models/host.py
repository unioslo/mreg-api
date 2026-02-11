"""Pydantic models for API resources."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from datetime import datetime
from typing import ClassVar
from typing import Self
from typing import cast

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Unpack
from typing_extensions import override

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import DeleteError
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import EntityOwnershipMismatch
from mreg_api.exceptions import MregValidationError
from mreg_api.exceptions import MultipleEntitiesFound
from mreg_api.exceptions import PostError
from mreg_api.models.abstracts import FrozenModel
from mreg_api.models.abstracts import FrozenModelWithTimestamps
from mreg_api.models.abstracts import manager_only
from mreg_api.models.abstracts import normalize_patch_fields
from mreg_api.models.fields import HostName
from mreg_api.models.fields import MacAddress
from mreg_api.models.fields import NameList
from mreg_api.models.history import HistoryResource
from mreg_api.models.infra import CNAME
from mreg_api.models.infra import MX
from mreg_api.models.infra import NAPTR
from mreg_api.models.infra import SSHFP
from mreg_api.models.infra import TXT
from mreg_api.models.infra import BacnetID
from mreg_api.models.infra import Community
from mreg_api.models.infra import Delegation
from mreg_api.models.infra import ForwardZone
from mreg_api.models.infra import ForwardZoneDelegation
from mreg_api.models.infra import HInfo
from mreg_api.models.infra import IPAddress
from mreg_api.models.infra import Location
from mreg_api.models.infra import Network
from mreg_api.models.infra import NetworkOrIP
from mreg_api.models.infra import PTR_override
from mreg_api.models.infra import Role
from mreg_api.models.infra import Srv
from mreg_api.models.infra import WithHistory
from mreg_api.models.infra import WithName
from mreg_api.models.infra import WithTTL
from mreg_api.models.infra import Zone
from mreg_api.models.patch_types import HostGroupPatch
from mreg_api.models.patch_types import HostPatch
from mreg_api.types import ClientProtocol
from mreg_api.types import IP_AddressT
from mreg_api.types import QueryParams
from mreg_api.types import get_type_adapter

logger = logging.getLogger(__name__)

class HostCommunity(FrozenModel):
    """Model for a host's community.

    Communities are associated with hosts via IP addresses.
    """

    ipaddress: int
    """ID of the IP address associated with the community"""

    community: Community


class HostContactModification(FrozenModel):
    """Model for host contact email modifications."""

    added: list[str] = Field(default_factory=list)
    already_exists: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    not_found: list[str] = Field(default_factory=list)


class ContactEmail(FrozenModelWithTimestamps):
    """Model for a host's contact email."""

    id: int
    email: str


class Host(FrozenModelWithTimestamps, WithTTL, WithHistory):
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
    ttl: int | None = None
    srvs: list[Srv] = []
    naptrs: list[NAPTR] = []
    sshfps: list[SSHFP] = []
    roles: list[str] = []
    hostgroups: list[str] = []
    comment: str
    contacts: list[ContactEmail] = []
    contact: str | None = Field(default=None, deprecated=True)

    communities: list[HostCommunity] = []

    # Note, we do not use WithZone here as this is optional and we resolve it differently.
    zone: int | None = None

    history_resource: ClassVar[HistoryResource] = HistoryResource.Host

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
                    dt = datetime(
                        1970, 1, 1, 0, 0, 0, tzinfo=self.created_at.tzinfo
                    )  # tz-aware epoch time
                    self.contacts.append(
                        ContactEmail(id=0, email=self.contact, created_at=dt, updated_at=dt)
                    )
        return self

    @property
    def contact_emails(self) -> list[str]:
        """A list of contact email addresses for the host."""
        return [contact.email for contact in self.contacts]

    @field_validator("communities", mode="before")
    @classmethod
    def _validate_none_communities_as_empty_list(cls, v: object) -> object:
        """Convert None value to empty list for communities."""
        if v is None:
            return []
        return v

    @field_validator("bacnetid", mode="before")
    @classmethod
    def convert_bacnetid(cls, v: object) -> object:
        """Use nested ID value in bacnetid value."""
        if isinstance(v, dict):
            return cast(dict[str, object], v).get("id")
        return None

    @classmethod
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Hosts

    @override
    def patch(
        self,
        fields: Mapping[str, object] | None = None,
        validate: bool = True,
        **field_kwargs: object,
    ) -> Host:
        """Patch the host using typed keyword arguments by default."""
        patch_fields = normalize_patch_fields(fields, field_kwargs)
        return super().patch(fields=patch_fields, validate=validate)

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[HostPatch]) -> Host:
        """Patch the host using a statically typed keyword interface."""
        return self.patch(validate=validate, **field_kwargs)

    @override
    def patch_raw(self, fields: dict[str, object], validate: bool = True) -> Host:
        """Patch the host using a raw dictionary payload."""
        return super().patch(fields=fields, validate=validate)

    @classmethod
    @manager_only
    def get_list_by_ip(
        cls, client: "ClientProtocol", ip: IP_AddressT, inform_as_ptr: bool = True
    ) -> list[Self]:
        """Get a list of hosts by IP address.

        :param ip: The IP address to search for.
        :param check_ptr: If True, check for PTR overrides as well.
        :returns: A list of Host objects.
        """
        hosts = cast(list[Self], cls.get_list_by_field(client, "ipaddresses__ipaddress", str(ip)))
        if not hosts:
            hosts = cast(list[Self], cls.get_list_by_field(client, "ptr_overrides__ipaddress", str(ip)))
            if hosts and inform_as_ptr:
                for host in hosts:
                    host.add_note(f"{ip} is a PTR override for {host.name}")
        return hosts

    @classmethod
    @manager_only
    def get_list_by_ip_or_raise(
        cls, client: "ClientProtocol", ip: IP_AddressT, inform_as_ptr: bool = True
    ) -> list[Self]:
        """Get a list of hosts by IP address or raise EntityNotFound.

        :param ip: The IP address to search for.
        :returns: A list of Host objects.
        :param check_ptr: If True, check for PTR overrides as well.
        """
        hosts = cls.get_list_by_ip(client, ip, inform_as_ptr=inform_as_ptr)
        if not hosts:
            raise EntityNotFound(f"Host with IP address {ip} not found.")
        return hosts

    @classmethod
    @manager_only
    def get_by_ip(
        cls, client: "ClientProtocol", ip: IP_AddressT, inform_as_ptr: bool = True
    ) -> Host | None:
        """Get a host by IP address.

        :param ip: The IP address to search for.
        :param check_ptr: If True, check for PTR overrides as well.
        :returns: The Host object if found, None otherwise.
        """
        try:
            host = cast(Host | None, cls.get_by_field(client, "ipaddresses__ipaddress", str(ip)))
            if not host:
                host = cast(Host | None, cls.get_by_field(client, "ptr_overrides__ipaddress", str(ip)))
                if host and inform_as_ptr:
                    host.add_note(f"{ip} is a PTR override for {host.name}")
            return host
        except MultipleEntitiesFound as e:
            raise MultipleEntitiesFound(f"Multiple hosts found with IP address {ip}.") from e

    @classmethod
    @manager_only
    def get_by_ip_or_raise(
        cls, client: "ClientProtocol", ip: IP_AddressT, inform_as_ptr: bool = True
    ) -> Host:
        """Get a host by IP address or raise EntityNotFound.

        :param ip: The IP address to search for.
        :returns: The Host object if found.
        :param check_ptr: If True, check for PTR overrides as well.
        """
        host = cls.get_by_ip(client, ip, inform_as_ptr=inform_as_ptr)
        if not host:
            raise EntityNotFound(f"Host with IP address {ip} not found.")
        return host

    @classmethod
    @manager_only
    def get_by_mac(cls, client: "ClientProtocol", mac: MacAddress) -> Host | None:
        """Get a host by MAC address.

        :param ip: The MAC address to search for.
        :returns: The Host object if found, None otherwise.
        """
        return cls.get_by_field(client, "ipaddresses__macaddress", str(mac))

    @classmethod
    @manager_only
    def get_by_mac_or_raise(cls, client: "ClientProtocol", mac: MacAddress) -> Host:
        """Get a host by MAC address or raise EntityNotFound.

        :param ip: The MAC address to search for.
        :returns: The Host object if found.
        """
        host = cls.get_by_mac(client, mac)
        if not host:
            raise EntityNotFound(f"Host with MAC address {mac} not found.")
        return host

    @classmethod
    @manager_only
    def get_list_by_mac(cls, client: "ClientProtocol", mac: MacAddress) -> list[Self]:
        """Get a list of host by MAC address.

        :param ip: The MAC address to search for.
        :returns: The Host object if found, None otherwise.
        """
        return cls.get_list_by_field(client, "ipaddresses__macaddress", str(mac))

    @classmethod
    @manager_only
    def get_list_by_mac_or_raise(cls, client: "ClientProtocol", mac: MacAddress) -> list[Self]:
        """Get a list of hosts by MAC address or raise EntityNotFound.

        :param ip: The MAC address to search for.
        :returns: The Host object if found.
        """
        hosts = cls.get_list_by_mac(client, mac)
        if not hosts:
            raise EntityNotFound(f"Host with MAC address {mac} not found.")
        return hosts

    @classmethod
    @manager_only
    def get_by_any_means_or_raise(
        cls,
        client: "ClientProtocol",
        identifier: str | HostName,
        inform_as_cname: bool = True,
        inform_as_ptr: bool = True,
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
            client, identifier, inform_as_cname=inform_as_cname, inform_as_ptr=inform_as_ptr
        )
        if not host:
            raise EntityNotFound(f"Host {identifier} not found.")
        return host

    @classmethod
    @manager_only
    def get_by_any_means(
        cls,
        client: "ClientProtocol",
        identifier: str,
        inform_as_cname: bool = True,
        inform_as_ptr: bool = True,
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
            return Host.get_by_id(client, int(identifier))

        if ip := NetworkOrIP.parse(identifier, mode="ip"):
            host = cls.get_by_ip_or_raise(client, ip, inform_as_ptr=inform_as_ptr)
            return host

        if mac := MacAddress.parse(identifier):
            return cls.get_by_mac_or_raise(client, mac)

        # Let us try to find the host by name...
        identifier = HostName.parse_or_raise(identifier)

        host = cast(Host | None, cls.get_by_field(client, "name", identifier))
        if host:
            return host

        cname = cast(CNAME | None, CNAME.get_by_field(client, "name", identifier))
        # If we found a CNAME, get the host it points to. We're not interested in the
        # CNAME itself.
        if cname is not None:
            host = cast(Host | None, Host.get_by_id(client, cname.host))

            if host and inform_as_cname:
                host.add_note(f"{identifier} is a CNAME for {host.name}")

        return host

    @classmethod
    @manager_only
    def get_list_by_any_means_or_raise(
        cls,
        client: "ClientProtocol",
        identifier: str | HostName,
        inform_as_cname: bool = True,
        inform_as_ptr: bool = True,
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
            client, identifier, inform_as_cname=inform_as_cname, inform_as_ptr=inform_as_ptr
        )
        if not hosts:
            raise EntityNotFound(f"Host {identifier} not found.")
        return hosts

    @classmethod
    @manager_only
    def get_list_by_any_means(
        cls,
        client: "ClientProtocol",
        identifier: str,
        inform_as_cname: bool = True,
        inform_as_ptr: bool = True,
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
        if identifier.isdigit() and (host := cls.get_by_id(client, int(identifier))):
            return [host]

        if ip := NetworkOrIP.parse(identifier, mode="ip"):
            return cls.get_list_by_ip_or_raise(client, ip, inform_as_ptr=inform_as_ptr)

        if mac := MacAddress.parse(identifier):
            return cls.get_list_by_mac_or_raise(client, mac)

        # Let us try to find the host by name...
        identifier = HostName.parse_or_raise(identifier)

        host = cast(Self | None, cls.get_by_field(client, "name", identifier))
        if host:
            return [host]

        # If we found a CNAME, get the host it points to. We're not interested in the
        # CNAME itself.
        cname = cast(CNAME | None, CNAME.get_by_field(client, "name", identifier))
        if cname:
            host = cast(Self | None, cls.get_by_id(client, cname.host))
            if host and inform_as_cname:
                host.add_note(f"{identifier} is a CNAME for {host.name}")
                return [host]

        return []

    @override
    def delete(self) -> bool:
        """Delete the host.

        :raises DeleteError: If the operation to delete the host fails.

        :returns: True if the host was deleted successfully, False otherwise.
        """
        client = self._require_client()
        # Note, we can't use .id as the identifier here, as the host name is used
        # in the endpoint URL...
        try:
            resp = client.delete(Endpoint.Hosts.with_id(str(self.name)), ok404=False)
        except DeleteError as e:
            # TODO: implement after mreg-cli parity
            # raise DeleteError(f"Failed to delete host {self.name}.", e.response) from e
            raise e

        return resp.is_success

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
        """DEPRECATED: Set the contact for the host.

        Uses the old `contact` field to set a single contact.
        Used for backwards compatibility.

        :param contact: The contact to set. Should be a valid email, but we leave it to the
                        server to validate the data.

        :returns: A new Host object fetched from the API with the updated contact.
        """
        return self.patch(fields={"contact": contact})

    def set_contacts(self, contacts: list[str]) -> Host:
        """Set a new list of contacts for the host. Overwrites existing contacts.

        Args:
            contacts (list[str]): New contacts for the host.

        Returns:
            Host: Updated Host object.
        """
        # Uses non-atomic host update via PATCH to set the contacts list.
        return self.patch(fields={"contacts": contacts}, validate=False)

    def add_contacts(self, contacts: list[str]) -> HostContactModification:
        """Add contacts to the host.

        Args:
            contacts (list[str]): Contacts to add.

        Raises:
            CreateError: Failed to add contacts to host.

        Returns:
            HostContactModification: Summary of host contact modifications.
        """
        client = self._require_client()
        # Uses atomic endpoint for contact updates
        endpoint = Endpoint.HostsContacts.with_params(self.name)
        try:
            resp = client.post(endpoint, emails=contacts)
        except PostError:
            # TODO: implement after mreg-cli parity
            # raise PostError(f"Failed to add contacts to host {self.name}.", e.response) from e
            raise
        adapter = get_type_adapter(HostContactModification)
        return adapter.validate_json(resp.text)

    def clear_contacts(self) -> HostContactModification:
        """Remove all contacts from the host.

        Raises:
            DeleteError: _description_

        Returns:
            HostContactModification: _description_
        """
        client = self._require_client()
        endpoint = Endpoint.HostsContacts.with_params(self.name)
        try:
            resp = client.delete(endpoint)
        except DeleteError as e:
            # TODO: implement after mreg-cli parity
            # raise DeleteError(f"Failed to remove contacts from host {self.name}.", e.response) from e
            raise e
        adapter = get_type_adapter(HostContactModification)
        return adapter.validate_json(resp.text)

    def remove_contacts(self, contacts: list[str]) -> HostContactModification:
        """Remove the given contacts from the host.

        Args:
            contacts (list[str]): Contacts to remove.

        Raises:
            DeleteError: Failed to remove contacts.

        Returns:
            HostContactModification: Summary of host contact modifications.
        """
        client = self._require_client()
        endpoint = Endpoint.HostsContacts.with_params(self.name)
        try:
            resp = client.delete(endpoint, emails=contacts)
        except DeleteError as e:
            # TODO: implement after mreg-cli parity
            # raise DeleteError(f"Failed to remove contacts from host {self.name}.", e.response) from e
            raise e
        adapter = get_type_adapter(HostContactModification)
        return adapter.validate_json(resp.text)

    def unset_contacts(self) -> Host:
        """Remove all contacts from the host.

        Returns:
            Host: Updated host after contact removal
        """
        # TODO: add try/except with error message
        return self.patch(fields={"contacts": []})

    def add_ip(self, ip: IP_AddressT, mac: MacAddress | None = None) -> Host:
        """Add an IP address to the host.

        :param ip: The IP address to add. IPv4 or IPv6.

        :returns: A new Host object fetched from the API with the updated IP address.
        """
        client = self._require_client()
        params: QueryParams = {"ipaddress": str(ip), "host": str(self.id)}
        if mac:
            params["macaddress"] = mac

        _ = IPAddress.create(client, params=params, _manager=True)
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

    def associate_mac_to_ip(
        self,
        mac: MacAddress,
        ip: IP_AddressT | str,
        force: bool = False,
    ) -> Host:
        """Associate a MAC address to an IP address.

        :param mac: The MAC address to associate.
        :param ip: The IP address to associate.

        :returns: A new Host object fetched from the API after updating the IP address.
        """
        client = self._require_client()
        if isinstance(ip, str):
            ip = NetworkOrIP.parse_or_raise(ip, mode="ip")

        params: QueryParams = {
            "macaddress": mac,
            "ordering": "ipaddress",
        }

        ipadresses = client.get_typed(Endpoint.Ipaddresses, list[IPAddress], params=params)

        if ip in [ip.ipaddress for ip in ipadresses]:
            raise EntityAlreadyExists(f"IP address {ip} already has MAC address {mac} associated.")

        if len(ipadresses) and not force:
            raise EntityOwnershipMismatch(
                f"mac {mac} already in use by: {', '.join(str(ip.ipaddress) for ip in ipadresses)}. Use force to add {ip} -> {mac} as well."  # noqa: E501
            )

        ip_found_in_host = False
        for myip in self.ipaddresses:
            if myip.ipaddress == ip:
                _ = myip.associate_mac(mac, force=force)
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
                _ = myip.disassociate_mac()
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
        _ = self._require_client()
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
        _ = self._require_client()
        ret_dict: dict[int, list[IPAddress]] = {}

        for network, ips in self.networks().items():
            vlan = network.vlan or 0
            if vlan not in ret_dict:
                ret_dict[vlan] = []

            ret_dict[vlan].extend(ips)

        return ret_dict

    def resolve_zone(
        self,
        accept_delegation: bool = False,
        validate_zone_resolution: bool = False,
    ) -> Zone | Delegation | None:
        """Return the zone for the host.

        :param accept_delegation: If True, accept delegation and return a Delegation object if the
                zone of the host is delegated. Otherwise raise EntityOwnershipMismatch.
        :param validate_zone_resolution: If True, validate that the resolved zone matches the
                expected zone ID. Fail with ValidationFailure if it does not.
        """
        client = self._require_client()
        if not self.zone:
            return None

        data = client.get(Endpoint.ForwardZoneForHost.with_id(str(self.name)))
        raw_data = cast(object, data.json())
        if not isinstance(raw_data, Mapping):
            raise EntityNotFound(f"Failed to resolve zone for host {self.name}.")
        data_as_dict = cast(Mapping[str, object], raw_data)

        zone_data = data_as_dict.get("zone")
        if isinstance(zone_data, Mapping) and zone_data:
            zone = ForwardZone.model_validate(zone_data)
            zone = client._bind_client(zone)
            if validate_zone_resolution and zone.id != self.zone:
                raise MregValidationError(f"Expected zone ID {self.zone} but resolved as {zone.id}.")
            return zone

        delegation_data = cast(Mapping[str, object] | None, data_as_dict.get("delegation"))
        if isinstance(delegation_data, Mapping) and delegation_data:
            if not accept_delegation:
                delegation_name = delegation_data.get("name")
                raise EntityOwnershipMismatch(
                    f"Host {self.name} is delegated to zone {delegation_name}."
                )
            delegation = ForwardZoneDelegation.model_validate(delegation_data)
            return client._bind_client(delegation)

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
        client = self._require_client()
        return Role.get_list_by_field(client, "hosts", self.id, _manager=True)

    def get_hostgroups(self, traverse: bool = False) -> list[HostGroup]:
        """Return all hostgroups for the host.

        :param traverse: If True, traverse the parent groups and include them in the list.

        :returns: A list of HostGroup objects sorted by name.
        """
        client = self._require_client()
        groups: list[HostGroup] = []
        direct = cast(list[HostGroup], HostGroup.get_list_by_field(client, "hosts", self.id, _manager=True))
        groups.extend(direct)

        if traverse:
            for group in direct:
                groups.extend(group.get_all_parents())

        return sorted(groups, key=lambda group: group.name)

    def bacnet(self) -> BacnetID | None:
        """Return the BacnetID for the host."""
        if not self.bacnetid:
            return None

        client = self._require_client()
        return BacnetID.get_by_id(client, self.bacnetid, _manager=True)

    def has_mx_with_priority(self, mx_arg: str, priority: int) -> MX | None:
        """Check if the host has an MX record.

        :param mx: The MX record to check for.
        :param priority: The priority of the MX record.

        :returns: True if the host has the MX record, False otherwise.
        """
        return next((mx for mx in self.mxs if mx.has_mx_with_priority(mx_arg, priority)), None)

    @override
    def __str__(self) -> str:
        """Return the host name as a string."""
        return self.name

    @override
    def __hash__(self) -> int:
        """Return a hash of the host."""
        return hash((self.id, self.name))


class HostList(FrozenModel):
    """Model for a list of hosts.

    This is the endpoint at /api/v1/hosts/.
    """

    results: list[Host]
    is_ptr: bool = False  # FIXME: unused
    is_cname: bool = False  # FIXME: unused

    @classmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.Hosts

    @classmethod
    @manager_only
    def get(cls, client: "ClientProtocol", params: QueryParams | None = None) -> HostList:
        """Get a list of hosts.

        :param params: Optional parameters to pass to the API.

        :returns: A HostList object.
        """
        if params is None:
            params = {}

        if "ordering" not in params:
            params["ordering"] = "name"

        hosts = client.get_typed(cls.endpoint(), list[Host], params=params)
        return cls(results=hosts)

    @classmethod
    @manager_only
    def get_by_ip(cls, client: "ClientProtocol", ip: IP_AddressT) -> HostList:
        """Get a list of hosts by IP address.

        :param ip: The IP address to search for.

        :returns: A HostList object.
        """
        return cls.get(client, params={"ipaddresses__ipaddress": str(ip), "ordering": "name"})

    def __len__(self):
        """Return the number of results."""
        return len(self.results)

    def __getitem__(self, key: int) -> Host:
        """Get a result by index."""
        return self.results[key]

    @override
    def __str__(self) -> str:
        """Return a string representation of the results."""
        return str(self.results)

    @override
    def __repr__(self) -> str:
        """Return a string representation of the results."""
        return repr(self.results)

    def hostnames(self) -> list[str]:
        """Return a list of hostnames."""
        return [host.name for host in self.results]

    def count(self):
        """Return the number of results."""
        return len(self.results)


class HostGroup(FrozenModelWithTimestamps, WithName, WithHistory):
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
    @override
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the class."""
        return Endpoint.HostGroups

    def patch_typed(self, *, validate: bool = True, **field_kwargs: Unpack[HostGroupPatch]) -> Self:
        """Patch hostgroup with typed keyword arguments."""
        return self.patch(validate=validate, **field_kwargs)

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
        client = self._require_client()
        try:
            _ = client.post(Endpoint.HostGroupsAddHostGroups.with_params(self.name), name=groupname)
        except PostError as e:
            # TODO: implement after mreg-cli parity
            # raise PostError(f"Failed to add group {groupname} to hostgroup {self.name}.", e.response)
            raise e
        return self.refetch()

    def remove_group(self, groupname: str) -> Self:
        """Remove a group from the hostgroup.

        :param group: The group to remove.

        :returns: A new HostGroup object fetched from the API with the updated groups.
        """
        client = self._require_client()
        try:
            _ = client.delete(Endpoint.HostGroupsRemoveHostGroups.with_params(self.name, groupname))
        except DeleteError as e:
            # TODO: implement after mreg-cli parity
            # raise DeleteError(
            #     f"Failed to remove group {groupname} from hostgroup {self.name}.", e.response
            # ) from e
            raise e
        return self.refetch()

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
        client = self._require_client()
        try:
            _ = client.post(Endpoint.HostGroupsAddHosts.with_params(self.name), name=hostname)
        except PostError as e:
            # TODO: implement after mreg-cli parity
            # raise PostError(f"Failed to add host {hostname} to hostgroup {self.name}.", e.response)
            raise e
        return self.refetch()

    def remove_host(self, hostname: str) -> Self:
        """Remove a host from the hostgroup.

        :param hostname: The host to remove.

        :returns: A new HostGroup object fetched from the API with the updated hosts.
        """
        client = self._require_client()
        try:
            _ = client.delete(Endpoint.HostGroupsRemoveHosts.with_params(self.name, hostname))
        except DeleteError as e:
            # TODO: implement after mreg-cli parity
            # raise DeleteError(
            #     f"Failed to remove host {hostname} from hostgroup {self.name}.", e.response
            # ) from e
            raise e
        return self.refetch()

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
        client = self._require_client()
        try:
            _ = client.post(Endpoint.HostGroupsAddOwner.with_params(self.name), name=ownername)
        except PostError as e:
            # TODO: implement after mreg-cli parity
            # raise PostError(
            #     f"Failed to add owner {ownername} to hostgroup {self.name}.", e.response
            # ) from e
            raise e
        return self.refetch()

    def remove_owner(self, ownername: str) -> Self:
        """Remove an owner from the hostgroup.

        :param ownername: The owner to remove.

        :returns: A new HostGroup object fetched from the API with the updated owners.
        """
        client = self._require_client()
        try:
            _ = client.delete(Endpoint.HostGroupsRemoveOwner.with_params(self.name, ownername))
        except DeleteError as e:
            # TODO: implement after mreg-cli parity
            # raise DeleteError(
            #     f"Failed to remove owner {ownername} from hostgroup {self.name}.", e.response
            # ) from e
            raise e
        return self.refetch()

    def get_all_parents(self) -> list[HostGroup]:
        """Return a list of all parent groups."""
        client = self._require_client()
        parents: list[HostGroup] = []
        for parent in self.parent:
            pobj = cast(HostGroup | None, HostGroup.get_by_field(client, "name", parent, _manager=True))
            if pobj:
                parents.append(pobj)
                parents.extend(pobj.get_all_parents())

        return parents


### Meta models
