"""Compatibility re-export module for API resource models."""

from __future__ import annotations

from mreg_api.models.host import ContactEmail
from mreg_api.models.host import Host
from mreg_api.models.host import HostCommunity
from mreg_api.models.host import HostContactModification
from mreg_api.models.host import HostGroup
from mreg_api.models.host import HostList
from mreg_api.models.infra import CNAME
from mreg_api.models.infra import MX
from mreg_api.models.infra import NAPTR
from mreg_api.models.infra import SSHFP
from mreg_api.models.infra import TXT
from mreg_api.models.infra import APIModelMixin
from mreg_api.models.infra import Atom
from mreg_api.models.infra import BacnetID
from mreg_api.models.infra import Community
from mreg_api.models.infra import Delegation
from mreg_api.models.infra import ExcludedRange
from mreg_api.models.infra import ForwardZone
from mreg_api.models.infra import ForwardZoneDelegation
from mreg_api.models.infra import HInfo
from mreg_api.models.infra import HostPolicy
from mreg_api.models.infra import IPAddress
from mreg_api.models.infra import IPNetMode
from mreg_api.models.infra import Label
from mreg_api.models.infra import Location
from mreg_api.models.infra import NameServer
from mreg_api.models.infra import Network
from mreg_api.models.infra import NetworkOrIP
from mreg_api.models.infra import NetworkPolicy
from mreg_api.models.infra import NetworkPolicyAttribute
from mreg_api.models.infra import NetworkPolicyAttributeValue
from mreg_api.models.infra import Permission
from mreg_api.models.infra import PTR_override
from mreg_api.models.infra import ReverseZone
from mreg_api.models.infra import ReverseZoneDelegation
from mreg_api.models.infra import Role
from mreg_api.models.infra import Srv
from mreg_api.models.infra import Zone
from mreg_api.models.system import Fetchable
from mreg_api.models.system import HealthInfo
from mreg_api.models.system import HeartbeatHealth
from mreg_api.models.system import LDAPHealth
from mreg_api.models.system import Library
from mreg_api.models.system import ServerLibraries
from mreg_api.models.system import ServerVersion
from mreg_api.models.system import TokenAuth
from mreg_api.models.system import TokenInfo
from mreg_api.models.system import UserDjangoStatus
from mreg_api.models.system import UserInfo
from mreg_api.models.system import UserMregStatus
from mreg_api.models.system import UserPermission

__all__ = [
    "APIModelMixin",
    "Atom",
    "BacnetID",
    "CNAME",
    "Community",
    "ContactEmail",
    "Delegation",
    "ExcludedRange",
    "Fetchable",
    "ForwardZone",
    "ForwardZoneDelegation",
    "HInfo",
    "HealthInfo",
    "HeartbeatHealth",
    "Host",
    "HostCommunity",
    "HostContactModification",
    "HostGroup",
    "HostList",
    "HostPolicy",
    "IPAddress",
    "IPNetMode",
    "LDAPHealth",
    "Label",
    "Library",
    "Location",
    "MX",
    "NAPTR",
    "NameServer",
    "Network",
    "NetworkOrIP",
    "NetworkPolicy",
    "NetworkPolicyAttribute",
    "NetworkPolicyAttributeValue",
    "PTR_override",
    "Permission",
    "ReverseZone",
    "ReverseZoneDelegation",
    "Role",
    "SSHFP",
    "ServerLibraries",
    "ServerVersion",
    "Srv",
    "TXT",
    "TokenAuth",
    "TokenInfo",
    "UserDjangoStatus",
    "UserInfo",
    "UserMregStatus",
    "UserPermission",
    "Zone",
]
