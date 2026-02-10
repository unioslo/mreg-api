"""Typed patch payloads for model helpers."""

from __future__ import annotations

from typing_extensions import TypedDict

from mreg_api.models.fields import HostName
from mreg_api.models.fields import MacAddress


class PermissionPatch(TypedDict, total=False):
    """Typed patch payload for Permission."""

    labels: list[int]


class ZonePatch(TypedDict, total=False):
    """Typed patch payload for Zone."""

    primary_ns: str
    email: str
    serialno: int
    refresh: int
    retry: int
    expire: int
    soa_ttl: int
    default_ttl: int


class HostPolicyPatch(TypedDict, total=False):
    """Typed patch payload for HostPolicy-derived models."""

    description: str
    labels: list[int]


class LabelPatch(TypedDict, total=False):
    """Typed patch payload for Label."""

    description: str


class NetworkPatch(TypedDict, total=False):
    """Typed patch payload for Network."""

    category: str
    location: str
    description: str
    dns_delegated: bool
    frozen: bool
    reserved: int
    vlan: int | None
    policy: int | None
    max_communities: int | None


class CommunityPatch(TypedDict, total=False):
    """Typed patch payload for Community."""

    name: str
    description: str
    hosts: list[str]
    global_name: str | None


class NetworkPolicyAttributePatch(TypedDict):
    """Typed value for NetworkPolicy attribute updates."""

    name: str
    value: bool


class NetworkPolicyPatch(TypedDict, total=False):
    """Typed patch payload for NetworkPolicy."""

    description: str | None
    community_template_pattern: str | None
    attributes: list[NetworkPolicyAttributePatch]


class IPAddressPatch(TypedDict, total=False):
    """Typed patch payload for IPAddress."""

    macaddress: MacAddress | str | None


class HostPatch(TypedDict, total=False):
    """Typed patch payload for Host."""

    name: HostName | str
    comment: str
    contact: str | None
    contacts: list[str]
    ttl: int | None


class HostGroupPatch(TypedDict, total=False):
    """Typed patch payload for HostGroup."""

    description: str | None
