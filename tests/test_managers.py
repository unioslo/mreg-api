from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import pytest
from inline_snapshot import snapshot

from mreg_api import MregClient
from mreg_api.exceptions import InternalError
from mreg_api.managers import ResourceManager
from mreg_api.models.abstracts import MregModel


def get_resource_managers() -> Sequence[tuple[str, type[ResourceManager[Any]]]]:
    from mreg_api import managers

    managers_list: list[tuple[str, type[ResourceManager[Any]]]] = []
    for name in dir(managers):
        if name.endswith("Manager"):
            manager_class = getattr(managers, name)
            if (
                inspect.isabstract(manager_class)
                # Must inherit from ResourceManager.
                # As of 2026-07-03, some managers are implemented as
                # standalone classes that do not inherit from ResourceManager,
                # e.g. CommunityManager, ZoneManager, DelegationManager,
                # due to their use of scoped resources that do not play well
                # with the abstractions in ResourceManager.
                or ResourceManager not in manager_class.__mro__
                # Specialized /dhcphosts endpoints (ipv4, ipv6, ipv6byipv4)
                # do not have a path field, so skip them for this test.
                or "Dhcp" in name
            ):
                continue
            if isinstance(manager_class, type):
                managers_list.append((manager_class.__name__, manager_class))
    return managers_list


@pytest.mark.parametrize(
    "manager_name, manager_t", get_resource_managers(), ids=[name for name, _ in get_resource_managers()]
)
def test_manager_path_param_field(
    manager_name: str,  # pyright: ignore[reportUnusedParameter]  # noqa: ARG001
    manager_t: type[ResourceManager[MregModel]],
    client: Any,
) -> None:
    """Test that the manager's _path_param_field exists in the model's fields."""
    manager = manager_t(client)

    model_class = manager.model
    path_param_field = manager._path_param_field  # pyright: ignore[reportPrivateUsage]

    # Check that model has a field with the same name as the manager's path parameter
    assert path_param_field in model_class.model_fields


@pytest.mark.parametrize(
    "manager_name, manager_t", get_resource_managers(), ids=[name for name, _ in get_resource_managers()]
)
def test_path_param_value_returns_field_value(
    manager_name: str,  # pyright: ignore[reportUnusedParameter]  # noqa: ARG001
    manager_t: type[ResourceManager[MregModel]],
    client: Any,
) -> None:
    """_path_param_value returns the value of the model's path parameter field."""
    manager = manager_t(client)
    path_param_field = manager._path_param_field  # pyright: ignore[reportPrivateUsage]

    # Build a bare instance with only the path parameter field set. model_construct
    # skips validation, so a stand-in value is fine regardless of the field's type.
    sentinel = "path-param-stub"
    fields: dict[str, Any] = {path_param_field: sentinel}
    obj = manager.model.model_construct(**fields)

    assert manager._path_param_value(obj) == sentinel  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "manager_name, manager_t", get_resource_managers(), ids=[name for name, _ in get_resource_managers()]
)
def test_path_param_value_missing_field_raises_internal_error(
    manager_name: str,  # pyright: ignore[reportUnusedParameter]  # noqa: ARG001
    manager_t: type[ResourceManager[MregModel]],
    client: Any,
) -> None:
    """_path_param_value raises InternalError (not AttributeError) if the field is absent."""
    manager = manager_t(client)

    # An object lacking the path parameter attribute must yield a domain error,
    # never a raw AttributeError (see issue #42).
    with pytest.raises(InternalError):
        _ = manager._path_param_value(object())  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage,reportArgumentType]


def test_get_resource_managers_snapshot() -> None:
    """Snapshot test for get_resource_managers() output."""
    managers_list = get_resource_managers()
    assert isinstance(managers_list, list)  # func has Sequence type for variance
    # Sort by manager name for consistent ordering
    managers_list.sort(key=lambda x: x[0])
    # Extract only the manager names for snapshot comparison
    manager_names = [name for name, _ in managers_list]
    assert manager_names == snapshot(
        [
            "AtomManager",
            "BacnetIDManager",
            "CNAMEManager",
            "HInfoManager",
            "HostGroupManager",
            "HostManager",
            "IPAddressManager",
            "LabelManager",
            "LocationManager",
            "MXManager",
            "NAPTRManager",
            "NameServerManager",
            "NetworkManager",
            "NetworkPolicyAttributeManager",
            "NetworkPolicyManager",
            "PTROverrideManager",
            "PermissionManager",
            "RoleManager",
            "SSHFPManager",
            "SrvManager",
            "TXTManager",
            "_ForwardZoneManager",
            "_ReverseZoneManager",
        ]
    )


def test_manager_model_name_snapshot(client: MregClient) -> None:
    """Snapshot test for ResourceManager.model_name output."""
    managers_list = get_resource_managers()
    assert isinstance(managers_list, list)  # func has Sequence type for variance
    # Sort by manager name for consistent ordering
    managers_list.sort(key=lambda x: x[0])
    # Match manager name with model name
    managers = [manager_t(client) for _, manager_t in managers_list]
    man_names = [(m.__class__.__name__, m.model_name) for m in managers]
    assert man_names == snapshot(
        [
            ("AtomManager", "Atom"),
            ("BacnetIDManager", "BacnetID"),
            ("CNAMEManager", "CNAME"),
            ("HInfoManager", "HInfo"),
            ("HostGroupManager", "HostGroup"),
            ("HostManager", "Host"),
            ("IPAddressManager", "IPAddress"),
            ("LabelManager", "Label"),
            ("LocationManager", "Location"),
            ("MXManager", "MX"),
            ("NAPTRManager", "NAPTR"),
            ("NameServerManager", "NameServer"),
            ("NetworkManager", "Network"),
            ("NetworkPolicyAttributeManager", "NetworkPolicyAttribute"),
            ("NetworkPolicyManager", "NetworkPolicy"),
            ("PTROverrideManager", "PTR_override"),
            ("PermissionManager", "Permission"),
            ("RoleManager", "Role"),
            ("SSHFPManager", "SSHFP"),
            ("SrvManager", "Srv"),
            ("TXTManager", "TXT"),
            ("_ForwardZoneManager", "ForwardZone"),
            ("_ReverseZoneManager", "ReverseZone"),
        ]
    )
