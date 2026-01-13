"""Model definitions for resources in the MREG API."""

from __future__ import annotations

from mreg_api.models.models import CNAME
from mreg_api.models.models import MX
from mreg_api.models.models import NAPTR
from mreg_api.models.models import SSHFP
from mreg_api.models.models import TXT
from mreg_api.models.models import Atom
from mreg_api.models.models import BacnetID
from mreg_api.models.models import Community
from mreg_api.models.models import Delegation
from mreg_api.models.models import ExcludedRange
from mreg_api.models.models import ForwardZone
from mreg_api.models.models import ForwardZoneDelegation
from mreg_api.models.models import HealthInfo
from mreg_api.models.models import HeartbeatHealth
from mreg_api.models.models import HInfo
from mreg_api.models.models import Host
from mreg_api.models.models import HostCommunity
from mreg_api.models.models import HostList
from mreg_api.models.models import HostList2
from mreg_api.models.models import HostPolicy
from mreg_api.models.models import IPAddress
from mreg_api.models.models import Label
from mreg_api.models.models import LDAPHealth
from mreg_api.models.models import Library
from mreg_api.models.models import Location
from mreg_api.models.models import NameServer
from mreg_api.models.models import Network
from mreg_api.models.models import NetworkOrIP
from mreg_api.models.models import NetworkPolicy
from mreg_api.models.models import NetworkPolicyAttribute
from mreg_api.models.models import NetworkPolicyAttributeValue
from mreg_api.models.models import Permission
from mreg_api.models.models import PTR_override
from mreg_api.models.models import ReverseZone
from mreg_api.models.models import ReverseZoneDelegation
from mreg_api.models.models import Role
from mreg_api.models.models import ServerLibraries
from mreg_api.models.models import ServerVersion
from mreg_api.models.models import Srv
from mreg_api.models.models import TokenInfo
from mreg_api.models.models import UserDjangoStatus
from mreg_api.models.models import UserPermission
from mreg_api.models.models import Zone

__all__ = [
    "Atom",
    "BacnetID",
    "CNAME",
    "Community",
    "Delegation",
    "ExcludedRange",
    "ForwardZone",
    "ForwardZoneDelegation",
    "HealthInfo",
    "HeartbeatHealth",
    "HInfo",
    "Host",
    "HostCommunity",
    "HostList",
    "HostList2",
    "HostPolicy",
    "IPAddress",
    "Label",
    "LDAPHealth",
    "Library",
    "Location",
    "MX",
    "NameServer",
    "NAPTR",
    "Network",
    "NetworkOrIP",
    "NetworkPolicy",
    "NetworkPolicyAttribute",
    "NetworkPolicyAttributeValue",
    "Permission",
    "PTR_override",
    "ReverseZone",
    "ReverseZoneDelegation",
    "Role",
    "ServerLibraries",
    "ServerVersion",
    "Srv",
    "SSHFP",
    "TokenInfo",
    "TXT",
    "UserDjangoStatus",
    "UserPermission",
    "Zone",
]
