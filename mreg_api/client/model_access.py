"""Model manager accessors for the MREG client."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

from mreg_api.models.manager import AtomManager
from mreg_api.models.manager import CommunityManager
from mreg_api.models.manager import ForwardZoneManager
from mreg_api.models.manager import HostGroupManager
from mreg_api.models.manager import HostManager
from mreg_api.models.manager import HostPolicyManager
from mreg_api.models.manager import IPAddressManager
from mreg_api.models.manager import LabelManager
from mreg_api.models.manager import ModelManager
from mreg_api.models.manager import NetworkManager
from mreg_api.models.manager import NetworkPolicyManager
from mreg_api.models.manager import PermissionManager
from mreg_api.models.manager import ReverseZoneManager
from mreg_api.models.manager import RoleManager
from mreg_api.models.manager import ZoneManager
from mreg_api.models.manager import client_model_map
from mreg_api.types import ClientProtocol

if TYPE_CHECKING:
    from mreg_api.models import CNAME
    from mreg_api.models import MX
    from mreg_api.models import NAPTR
    from mreg_api.models import SSHFP
    from mreg_api.models import TXT
    from mreg_api.models import BacnetID
    from mreg_api.models import Delegation
    from mreg_api.models import ForwardZoneDelegation
    from mreg_api.models import HealthInfo
    from mreg_api.models import HeartbeatHealth
    from mreg_api.models import HInfo
    from mreg_api.models import HostList
    from mreg_api.models import LDAPHealth
    from mreg_api.models import Location
    from mreg_api.models import NameServer
    from mreg_api.models import NetworkPolicyAttribute
    from mreg_api.models import PTR_override
    from mreg_api.models import ReverseZoneDelegation
    from mreg_api.models import ServerLibraries
    from mreg_api.models import ServerVersion
    from mreg_api.models import Srv
    from mreg_api.models import UserInfo


_client_model_map_cache: dict[str, type] | None = None


T = TypeVar("T")


def _get_client_model_map() -> dict[str, type]:
    global _client_model_map_cache
    if _client_model_map_cache is None:
        models_module = importlib.import_module("mreg_api.models")
        _client_model_map_cache = client_model_map(models_module)
    return _client_model_map_cache


class ModelAccessMixin:
    """Typed model manager access methods for MregClient."""

    def _as_client(self) -> ClientProtocol:
        """Return this instance viewed as the client protocol."""
        return cast(ClientProtocol, cast(object, self))

    def __getattr__(self, name: str) -> ModelManager[object]:
        """Return a client-bound model manager for the named resource."""
        model = _get_client_model_map().get(name)
        if model is None:
            raise AttributeError(f"{self.__class__.__name__!s} object has no attribute {name!r}")
        return ModelManager(self._as_client(), model)

    def manager(self, model: type[T]) -> ModelManager[T]:
        """Get a model manager for a specific model class."""
        return ModelManager(self._as_client(), model)

    def atom(self) -> AtomManager:
        """Return a client-bound manager for Atom."""
        return AtomManager(self._as_client(), _get_client_model_map()["atom"])

    def bacnet_id(self) -> ModelManager["BacnetID"]:
        """Return a client-bound manager for BacnetID."""
        return ModelManager(self._as_client(), _get_client_model_map()["bacnet_id"])

    def cname(self) -> ModelManager["CNAME"]:
        """Return a client-bound manager for CNAME."""
        return ModelManager(self._as_client(), _get_client_model_map()["cname"])

    def community(self) -> CommunityManager:
        """Return a client-bound manager for Community."""
        return CommunityManager(self._as_client(), _get_client_model_map()["community"])

    def delegation(self) -> ModelManager["Delegation"]:
        """Return a client-bound manager for Delegation."""
        return ModelManager(self._as_client(), _get_client_model_map()["delegation"])

    def forward_zone(self) -> ForwardZoneManager:
        """Return a client-bound manager for ForwardZone."""
        return ForwardZoneManager(self._as_client(), _get_client_model_map()["forward_zone"])

    def forward_zone_delegation(self) -> ModelManager["ForwardZoneDelegation"]:
        """Return a client-bound manager for ForwardZoneDelegation."""
        return ModelManager(self._as_client(), _get_client_model_map()["forward_zone_delegation"])

    def health_info(self) -> ModelManager["HealthInfo"]:
        """Return a client-bound manager for HealthInfo."""
        return ModelManager(self._as_client(), _get_client_model_map()["health_info"])

    def heartbeat_health(self) -> ModelManager["HeartbeatHealth"]:
        """Return a client-bound manager for HeartbeatHealth."""
        return ModelManager(self._as_client(), _get_client_model_map()["heartbeat_health"])

    def hinfo(self) -> ModelManager["HInfo"]:
        """Return a client-bound manager for HInfo."""
        return ModelManager(self._as_client(), _get_client_model_map()["hinfo"])

    def host(self) -> HostManager:
        """Return a client-bound manager for Host."""
        return HostManager(self._as_client(), _get_client_model_map()["host"])

    def host_group(self) -> HostGroupManager:
        """Return a client-bound manager for HostGroup."""
        return HostGroupManager(self._as_client(), _get_client_model_map()["host_group"])

    def host_list(self) -> ModelManager["HostList"]:
        """Return a client-bound manager for HostList."""
        return ModelManager(self._as_client(), _get_client_model_map()["host_list"])

    def host_policy(self) -> HostPolicyManager:
        """Return a client-bound manager for HostPolicy."""
        return HostPolicyManager(self._as_client(), _get_client_model_map()["host_policy"])

    def ip_address(self) -> IPAddressManager:
        """Return a client-bound manager for IPAddress."""
        return IPAddressManager(self._as_client(), _get_client_model_map()["ip_address"])

    def label(self) -> LabelManager:
        """Return a client-bound manager for Label."""
        return LabelManager(self._as_client(), _get_client_model_map()["label"])

    def ldap_health(self) -> ModelManager["LDAPHealth"]:
        """Return a client-bound manager for LDAPHealth."""
        return ModelManager(self._as_client(), _get_client_model_map()["ldap_health"])

    def location(self) -> ModelManager["Location"]:
        """Return a client-bound manager for Location."""
        return ModelManager(self._as_client(), _get_client_model_map()["location"])

    def mx(self) -> ModelManager["MX"]:
        """Return a client-bound manager for MX."""
        return ModelManager(self._as_client(), _get_client_model_map()["mx"])

    def name_server(self) -> ModelManager["NameServer"]:
        """Return a client-bound manager for NameServer."""
        return ModelManager(self._as_client(), _get_client_model_map()["name_server"])

    def naptr(self) -> ModelManager["NAPTR"]:
        """Return a client-bound manager for NAPTR."""
        return ModelManager(self._as_client(), _get_client_model_map()["naptr"])

    def network(self) -> NetworkManager:
        """Return a client-bound manager for Network."""
        return NetworkManager(self._as_client(), _get_client_model_map()["network"])

    def network_policy(self) -> NetworkPolicyManager:
        """Return a client-bound manager for NetworkPolicy."""
        return NetworkPolicyManager(self._as_client(), _get_client_model_map()["network_policy"])

    def network_policy_attribute(self) -> ModelManager["NetworkPolicyAttribute"]:
        """Return a client-bound manager for NetworkPolicyAttribute."""
        return ModelManager(self._as_client(), _get_client_model_map()["network_policy_attribute"])

    def permission(self) -> PermissionManager:
        """Return a client-bound manager for Permission."""
        return PermissionManager(self._as_client(), _get_client_model_map()["permission"])

    def ptr_override(self) -> ModelManager["PTR_override"]:
        """Return a client-bound manager for PTR_override."""
        return ModelManager(self._as_client(), _get_client_model_map()["ptr_override"])

    def reverse_zone(self) -> ReverseZoneManager:
        """Return a client-bound manager for ReverseZone."""
        return ReverseZoneManager(self._as_client(), _get_client_model_map()["reverse_zone"])

    def reverse_zone_delegation(self) -> ModelManager["ReverseZoneDelegation"]:
        """Return a client-bound manager for ReverseZoneDelegation."""
        return ModelManager(self._as_client(), _get_client_model_map()["reverse_zone_delegation"])

    def role(self) -> RoleManager:
        """Return a client-bound manager for Role."""
        return RoleManager(self._as_client(), _get_client_model_map()["role"])

    def server_libraries(self) -> ModelManager["ServerLibraries"]:
        """Return a client-bound manager for ServerLibraries."""
        return ModelManager(self._as_client(), _get_client_model_map()["server_libraries"])

    def server_version(self) -> ModelManager["ServerVersion"]:
        """Return a client-bound manager for ServerVersion."""
        return ModelManager(self._as_client(), _get_client_model_map()["server_version"])

    def srv(self) -> ModelManager["Srv"]:
        """Return a client-bound manager for Srv."""
        return ModelManager(self._as_client(), _get_client_model_map()["srv"])

    def sshfp(self) -> ModelManager["SSHFP"]:
        """Return a client-bound manager for SSHFP."""
        return ModelManager(self._as_client(), _get_client_model_map()["sshfp"])

    def txt(self) -> ModelManager["TXT"]:
        """Return a client-bound manager for TXT."""
        return ModelManager(self._as_client(), _get_client_model_map()["txt"])

    def user_info(self) -> ModelManager["UserInfo"]:
        """Return a client-bound manager for UserInfo."""
        return ModelManager(self._as_client(), _get_client_model_map()["user_info"])

    def zone(self) -> ZoneManager:
        """Return a client-bound manager for Zone."""
        return ZoneManager(self._as_client(), _get_client_model_map()["zone"])
